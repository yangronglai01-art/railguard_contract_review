"""合同审核工作流服务。

ReviewService封装LangGraph的启动、查询和人工恢复操作，
FastAPI路由不直接操作checkpoint或Command。
"""

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import cast

from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, StateSnapshot

from railguard.agents.llm import (
    LlmAnalyzerError,
    LlmInputTooLargeError,
    LlmInvocationError,
    LlmProtocolError,
    LlmResponseError,
)
from railguard.models.schemas import (
    ContractDocument,
    Evidence,
    RiskFinding,
    new_id,
)
from railguard.observability import (
    new_trace_id,
    now_iso,
    percentile,
)
from railguard.rag.client import (
    RagProtocolError,
    RagRequestError,
    RagTimeoutError,
    RagUnavailableError,
)
from railguard.workflow.nodes import WorkflowProtocolError
from railguard.workflow.results import (
    MonitoringSummary,
    NodeLatencyStats,
    ReviewSnapshot,
)
from railguard.workflow.state import (
    HumanReviewDecision,
    ReviewState,
    WorkflowError,
)

logger = logging.getLogger(__name__)

# 各节点对应的审核进度权重（百分比）。
_NODE_PROGRESS: dict[str, float] = {
    "initialize_review": 10.0,
    "retrieve_evidence": 25.0,
    "commercial_risk_agent": 50.0,
    "legal_risk_agent": 50.0,
    "security_risk_agent": 50.0,
    "aggregate_findings": 65.0,
    "verify_citations": 80.0,
    "human_review": 90.0,
    "finalize_review": 100.0,
}


class ReviewServiceError(RuntimeError):
    """审核服务异常的公共基类。"""


class ReviewNotFoundError(ReviewServiceError):
    """指定的审核任务不存在。"""


class ReviewConflictError(ReviewServiceError):
    """审核任务当前状态不允许执行请求的操作。"""


class ReviewDecisionError(ReviewServiceError):
    """人工审核决定引用了无效的风险记录。"""


class ReviewExecutionError(ReviewServiceError):
    """审核图执行失败，并保留可查询的review_id。"""

    def __init__(
        self,
        *,
        review_id: str,
        code: str,
        message: str,
        retryable: bool,
    ) -> None:
        """保存接口需要返回的稳定错误信息。"""
        super().__init__(message)
        self.review_id = review_id
        self.code = code
        self.retryable = retryable


class ReviewService:
    """通过持久化LangGraph管理合同审核任务。"""

    def __init__(
        self,
        graph: CompiledStateGraph,
    ) -> None:
        """保存已绑定checkpointer的审核图。

        每个review_id使用独立异步锁，避免同一进程内
        两个请求同时恢复同一个人工审批任务。
        """
        self._graph = graph
        self._review_locks: dict[str, asyncio.Lock] = {}

    async def start_review(
        self,
        contract: ContractDocument,
        *,
        review_id: str | None = None,
    ) -> ReviewSnapshot:
        """创建审核ID并运行至完成或人工暂停。"""
        review_id = review_id or new_id()
        config = self._config(review_id)

        try:
            await self._graph.ainvoke(
                {
                    "review_id": review_id,
                    "contract": contract,
                    "trace_id": new_trace_id(),
                    "started_at_iso": now_iso(),
                },
                config=config,
            )
        except (
            RagTimeoutError,
            RagUnavailableError,
            RagRequestError,
            RagProtocolError,
            LlmAnalyzerError,
            WorkflowProtocolError,
        ) as exc:
            raise self._map_execution_error(
                review_id=review_id,
                error=exc,
            ) from exc

        snapshot = await self._load_state_snapshot(review_id)
        return self._to_review_snapshot(snapshot)

    async def get_review(
        self,
        review_id: str,
    ) -> ReviewSnapshot:
        """从checkpoint读取审核任务当前状态。"""
        snapshot = await self._load_state_snapshot(review_id)
        return self._to_review_snapshot(snapshot)

    async def submit_decision(
        self,
        *,
        review_id: str,
        decision: HumanReviewDecision,
        authorize: Callable[[], None] | None = None,
    ) -> ReviewSnapshot:
        """校验人工决定并恢复暂停的审核任务。

        所有校验都在调用Command(resume=...)之前完成。
        无效决定不会写入checkpoint，用户可以随后重新提交。
        """
        lock = self._review_locks.setdefault(
            review_id,
            asyncio.Lock(),
        )

        async with lock:
            snapshot = await self._load_state_snapshot(review_id)
            state = cast(ReviewState, snapshot.values)

            if (
                state.get("status") != "awaiting_human"
                or "human_review" not in snapshot.next
                or not snapshot.interrupts
            ):
                raise ReviewConflictError(
                    "Review is not waiting for a human decision"
                )

            valid_finding_ids = {
                RiskFinding.model_validate(item).finding_id
                for item in state.get("findings", [])
            }
            unknown_finding_ids = (
                set(decision.finding_decisions)
                - valid_finding_ids
            )

            if unknown_finding_ids:
                raise ReviewDecisionError(
                    "Decision referenced an unknown finding_id"
                )

            try:
                if authorize is not None:
                    # Revalidate after the per-review lock and checkpoint reads.
                    authorize()
                await self._graph.ainvoke(
                    Command(
                        resume=decision.model_dump(mode="json")
                    ),
                    config=self._config(review_id),
                )
            except (
                RagTimeoutError,
                RagUnavailableError,
                RagRequestError,
                RagProtocolError,
                LlmAnalyzerError,
                WorkflowProtocolError,
            ) as exc:
                raise self._map_execution_error(
                    review_id=review_id,
                    error=exc,
                ) from exc

            resumed_snapshot = (
                await self._load_state_snapshot(review_id)
            )
            return self._to_review_snapshot(
                resumed_snapshot
            )

    async def _load_state_snapshot(
        self,
        review_id: str,
    ) -> StateSnapshot:
        """读取checkpoint，不存在时抛出明确异常。"""
        if not review_id.strip():
            raise ReviewNotFoundError(
                "Review task was not found"
            )

        snapshot = await self._graph.aget_state(
            self._config(review_id)
        )

        if not snapshot.values:
            raise ReviewNotFoundError(
                "Review task was not found"
            )

        return snapshot

    @staticmethod
    def _config(
        review_id: str,
    ) -> dict[str, dict[str, str]]:
        """创建当前审核任务的LangGraph配置。"""
        return {
            "configurable": {
                "thread_id": review_id,
            }
        }

    @classmethod
    def _to_review_snapshot(
        cls,
        snapshot: StateSnapshot,
    ) -> ReviewSnapshot:
        """把LangGraph内部状态转换为稳定的业务结果。"""
        state = cast(ReviewState, snapshot.values)
        contract = ContractDocument.model_validate(
            state["contract"]
        )
        findings = [
            RiskFinding.model_validate(item)
            for item in state.get("findings", [])
        ]
        completed_nodes = list(
            state.get("completed_nodes", [])
        )

        # 只有finalize_review完成后，final_findings才是最终结果。
        if "finalize_review" in completed_nodes:
            final_findings = [
                RiskFinding.model_validate(item)
                for item in state.get(
                    "final_findings",
                    [],
                )
            ]
        else:
            final_findings = None

        human_decision_value = state.get("human_decision")
        human_decision = (
            HumanReviewDecision.model_validate(
                human_decision_value
            )
            if human_decision_value is not None
            else None
        )

        task_errors = [
            cls._task_error(
                node=task.name,
                error=task.error,
            )
            for task in snapshot.tasks
            if task.error is not None
        ]
        state_errors = [
            WorkflowError.model_validate(item)
            for item in state.get("errors", [])
        ]
        errors = [
            *state_errors,
            *task_errors,
        ]

        status = (
            "failed"
            if task_errors
            else state.get("status", "created")
        )

        started_at = state.get("started_at_iso")
        trace_id = state.get("trace_id")
        node_durations = cls._node_durations(state)
        retry_stats = cls._retry_stats(state)
        progress_percent = cls._progress_percent(completed_nodes)
        elapsed_seconds = cls._elapsed_seconds(started_at)

        return ReviewSnapshot(
            review_id=state["review_id"],
            contract_id=contract.contract_id,
            filename=contract.filename,
            status=status,
            findings=findings,
            final_findings=final_findings,
            evidence=cls._collect_evidence(state),
            human_decision=human_decision,
            completed_nodes=completed_nodes,
            errors=errors,
            final_summary=state.get("final_summary", ""),
            started_at=started_at,
            trace_id=trace_id,
            progress_percent=progress_percent,
            elapsed_seconds=elapsed_seconds,
            node_durations=node_durations,
            retry_stats=retry_stats,
        )

    @staticmethod
    def _progress_percent(
        completed_nodes: list[str],
    ) -> float:
        """根据已完成节点估算审核进度（0-100）。"""
        if not completed_nodes:
            return 0.0

        return max(
            (
                _NODE_PROGRESS.get(name, 0.0)
                for name in completed_nodes
            ),
            default=0.0,
        )

    @staticmethod
    def _node_durations(
        state: ReviewState,
    ) -> dict[str, float]:
        """把node_timings记录转换为节点到秒的映射。"""
        durations: dict[str, float] = {}

        for item in state.get("node_timings", []):
            durations[str(item[0])] = float(item[1])

        return durations

    @staticmethod
    def _retry_stats(
        state: ReviewState,
    ) -> dict[str, int]:
        """把retry_counts记录转换为节点到次数的映射。"""
        stats: dict[str, int] = {}

        for item in state.get("retry_counts", []):
            stats[str(item[0])] = int(item[1])

        return stats

    @staticmethod
    def _elapsed_seconds(
        started_at: str | None,
    ) -> float | None:
        """计算从启动到当前的耗时（秒），无法解析时返回None。"""
        if not started_at:
            return None

        try:
            started = datetime.fromisoformat(started_at)
        except ValueError:
            return None

        return (
            datetime.now(UTC) - started
        ).total_seconds()

    async def list_review_ids(self) -> list[str]:
        """枚举checkpoint中全部审核任务的thread_id。"""
        checkpointer = getattr(
            self._graph,
            "checkpointer",
            None,
        )
        connection = getattr(checkpointer, "conn", None)

        if connection is None:
            return []

        try:
            cursor = await connection.execute(
                "SELECT DISTINCT thread_id FROM checkpoints"
            )
            try:
                rows = await cursor.fetchall()
            finally:
                await cursor.close()
        except Exception:
            logger.warning(
                "failed to enumerate review checkpoints",
                exc_info=True,
            )
            return []

        return [str(row[0]) for row in rows]

    async def list_reviews(self) -> list[ReviewSnapshot]:
        """枚举并返回全部审核任务的当前快照。"""
        review_ids = await self.list_review_ids()

        snapshots: list[ReviewSnapshot] = []
        for review_id in review_ids:
            try:
                snapshots.append(
                    await self.get_review(review_id)
                )
            except ReviewNotFoundError:
                continue

        return snapshots

    async def monitoring_summary(self) -> MonitoringSummary:
        """汇总全部审核任务的进度、耗时和重试统计。"""
        return self._aggregate_monitoring(
            await self.list_reviews()
        )

    @staticmethod
    def _aggregate_monitoring(
        snapshots: list[ReviewSnapshot],
    ) -> MonitoringSummary:
        """从快照列表计算跨任务的监控汇总。"""
        status_counts: dict[str, int] = {}
        latency_samples: dict[str, list[float]] = {}
        retry_stats: dict[str, int] = {}
        failed_reviews = 0

        for snapshot in snapshots:
            status_counts[snapshot.status] = (
                status_counts.get(snapshot.status, 0) + 1
            )

            if snapshot.status == "failed":
                failed_reviews += 1

            for node_name, seconds in (
                snapshot.node_durations.items()
            ):
                latency_samples.setdefault(
                    node_name,
                    [],
                ).append(seconds * 1000)

            for node_name, retries in (
                snapshot.retry_stats.items()
            ):
                retry_stats[node_name] = (
                    retry_stats.get(node_name, 0) + retries
                )

        node_latency: dict[str, NodeLatencyStats] = {}
        for node_name, samples in latency_samples.items():
            p50 = percentile(samples, 0.50)
            p95 = percentile(samples, 0.95)
            p99 = percentile(samples, 0.99)
            node_latency[node_name] = NodeLatencyStats(
                count=len(samples),
                avg_ms=round(sum(samples) / len(samples), 3),
                p50_ms=round(p50, 3) if p50 is not None else None,
                p95_ms=round(p95, 3) if p95 is not None else None,
                p99_ms=round(p99, 3) if p99 is not None else None,
                min_ms=round(min(samples), 3),
                max_ms=round(max(samples), 3),
            )

        return MonitoringSummary(
            total_reviews=len(snapshots),
            failed_reviews=failed_reviews,
            status_counts=status_counts,
            node_latency=node_latency,
            retry_stats=retry_stats,
        )

    @staticmethod
    def _collect_evidence(
        state: ReviewState,
    ) -> list[Evidence]:
        """汇总并去重一次审核实际使用的全部RAG证据。"""
        evidence_by_id: dict[str, Evidence] = {}

        for evidence_items in state.get(
            "evidence_by_clause",
            {},
        ).values():
            for evidence_value in evidence_items:
                evidence = Evidence.model_validate(
                    evidence_value
                )
                evidence_by_id.setdefault(
                    evidence.evidence_id,
                    evidence,
                )

        for evidence_value in state.get(
            "contract_evidence",
            [],
        ):
            evidence = Evidence.model_validate(
                evidence_value
            )
            evidence_by_id.setdefault(
                evidence.evidence_id,
                evidence,
            )

        return list(evidence_by_id.values())

    @staticmethod
    def _task_error(
        *,
        node: str,
        error: object,
    ) -> WorkflowError:
        """把checkpoint中的内部异常转换为安全错误信息。"""
        code = "workflow_execution_failed"
        retryable = False

        if isinstance(error, RagTimeoutError):
            code = "rag_timeout"
            retryable = True
        elif isinstance(error, RagUnavailableError):
            code = "rag_unavailable"
            retryable = True
        elif isinstance(error, RagRequestError):
            code = "rag_request_rejected"
        elif isinstance(error, RagProtocolError):
            code = "rag_protocol_error"
        elif isinstance(error, LlmInputTooLargeError):
            code = "model_input_too_large"
        elif isinstance(error, LlmInvocationError):
            code = "model_unavailable"
            retryable = True
        elif isinstance(error, LlmResponseError):
            code = "model_response_invalid"
            retryable = True
        elif isinstance(error, LlmProtocolError):
            code = "model_protocol_error"
        elif isinstance(error, LlmAnalyzerError):
            code = "model_execution_failed"
        elif isinstance(error, WorkflowProtocolError):
            code = "workflow_protocol_error"

        return WorkflowError(
            node=node,
            code=code,
            message="审核节点执行失败，请检查服务日志。",
            retryable=retryable,
            details={
                "error_type": type(error).__name__,
            },
        )

    @staticmethod
    def _map_execution_error(
        *,
        review_id: str,
        error: Exception,
    ) -> ReviewExecutionError:
        """把已知执行异常转换为稳定的服务异常。"""
        if isinstance(error, RagTimeoutError):
            return ReviewExecutionError(
                review_id=review_id,
                code="rag_timeout",
                message="RAG服务请求超时。",
                retryable=True,
            )

        if isinstance(error, RagUnavailableError):
            return ReviewExecutionError(
                review_id=review_id,
                code="rag_unavailable",
                message="RAG服务暂时不可用。",
                retryable=True,
            )

        if isinstance(error, RagRequestError):
            return ReviewExecutionError(
                review_id=review_id,
                code="rag_request_rejected",
                message="RAG服务拒绝了检索请求。",
                retryable=False,
            )

        if isinstance(error, RagProtocolError):
            return ReviewExecutionError(
                review_id=review_id,
                code="rag_protocol_error",
                message="RAG服务返回了无效数据。",
                retryable=False,
            )

        if isinstance(error, LlmInputTooLargeError):
            return ReviewExecutionError(
                review_id=review_id,
                code="model_input_too_large",
                message="合同和证据超过模型输入限制。",
                retryable=False,
            )

        if isinstance(error, LlmInvocationError):
            return ReviewExecutionError(
                review_id=review_id,
                code="model_unavailable",
                message="大模型服务暂时不可用。",
                retryable=True,
            )

        if isinstance(error, LlmResponseError):
            return ReviewExecutionError(
                review_id=review_id,
                code="model_response_invalid",
                message="大模型返回了无效的结构化数据。",
                retryable=True,
            )

        if isinstance(error, LlmProtocolError):
            return ReviewExecutionError(
                review_id=review_id,
                code="model_protocol_error",
                message="大模型结果违反了审核协议。",
                retryable=False,
            )

        if isinstance(error, LlmAnalyzerError):
            return ReviewExecutionError(
                review_id=review_id,
                code="model_execution_failed",
                message="大模型审核执行失败。",
                retryable=False,
            )

        return ReviewExecutionError(
            review_id=review_id,
            code="workflow_protocol_error",
            message="审核流程返回了无效数据。",
            retryable=False,
        )
