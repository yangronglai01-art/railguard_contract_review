"""合同审核工作流服务。

ReviewService封装LangGraph的启动、查询和人工恢复操作，
FastAPI路由不直接操作checkpoint或Command。
"""

import asyncio
from typing import cast

from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, StateSnapshot

from railguard.models.schemas import (
    ContractDocument,
    Evidence,
    RiskFinding,
    new_id,
)
from railguard.rag.client import (
    RagProtocolError,
    RagRequestError,
    RagTimeoutError,
    RagUnavailableError,
)
from railguard.workflow.nodes import WorkflowProtocolError
from railguard.workflow.results import ReviewSnapshot
from railguard.workflow.state import (
    HumanReviewDecision,
    ReviewState,
    WorkflowError,
)


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
    ) -> ReviewSnapshot:
        """创建审核ID并运行至完成或人工暂停。"""
        review_id = new_id()
        config = self._config(review_id)

        try:
            await self._graph.ainvoke(
                {
                    "review_id": review_id,
                    "contract": contract,
                },
                config=config,
            )
        except (
            RagTimeoutError,
            RagUnavailableError,
            RagRequestError,
            RagProtocolError,
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

        return ReviewExecutionError(
            review_id=review_id,
            code="workflow_protocol_error",
            message="审核流程返回了无效数据。",
            retryable=False,
        )