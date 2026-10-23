"""supervisor_v2 主从编排图的节点与图构建。

document 25 目标架构的执行实现：

1. 主 Agent 受约束规划（``plan_review``）：调用 :class:`ConstrainedPlanner`
   把一次审核分解为有限任务语法内的原子任务，产出 :class:`ReviewPlan`。
   规划器未配置大模型时自动使用确定性规划器。
2. 确定性校验（``validate_plan``）：用纯函数校验计划自洽，
   不依赖大模型，校验失败直接终止。
3. 专业子 Agent 执行：复用 ``ReviewNodes`` 中的检索与三个风险
   Agent，以及引用核验、人工审批、收尾节点。
4. 任务级共享记忆（``write_memory`` → ``aggregate_from_memory``）：
   三个 Agent 执行后把完整结论写入共享记忆，汇总节点**从共享记忆
   读取**并重建最终发现，形成记忆写读闭环。
5. PG 事务 Outbox 事件驱动（``outbox.py``）：关键节点产出领域
   事件，与业务数据同事务提交，由后台投递器异步发布。
6. 有限协同复核（``detect_conflicts``）：仅做一轮冲突检测并应用
   固定解决策略，绝不引入无限争论循环。

该实现刻意保持与 ``fixed_v1`` 的差异最小化：复用既有的检索、
分析、引用核验、人工审批与收尾节点，仅在它们之前插入规划、
校验，在分析之后插入共享记忆、记忆汇总与冲突复核。
"""

from __future__ import annotations

import logging

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from railguard.models.schemas import RiskFinding
from railguard.observability import now_iso
from railguard.rag.client import RagRetriever
from railguard.workflow.analyzers import (
    DemoCommercialRiskAnalyzer,
    DemoLegalRiskAnalyzer,
    DemoSecurityRiskAnalyzer,
    RiskAnalyzer,
)
from railguard.workflow.graph import _timed_node
from railguard.workflow.nodes import (
    AnalyzerProtocolError,
    ReviewNodes,
    route_after_verification,
)
from railguard.workflow.orchestration import (
    ConflictRecord,
    MemoryRecord,
    ReviewEvent,
    ReviewPlan,
)
from railguard.workflow.planner import (
    ConstrainedPlanner,
    validate_plan_against_contract,
)
from railguard.workflow.state import ReviewState

logger = logging.getLogger(__name__)


class SupervisorPlanError(RuntimeError):
    """主 Agent 产出的计划无法通过确定性校验。"""


# 共享记忆中，各专业 Agent 结论的写入键。
_MEMORY_FINDINGS_KEY = "agent:{agent}:findings"
_MEMORY_COUNT_KEY = "agent:{agent}:finding_count"

# 三个专业 Agent 及其在状态中的发现通道，顺序固定以保证稳定汇总。
_AGENT_SOURCES = (
    ("commercial_analyzer", "commercial_findings"),
    ("legal_analyzer", "legal_findings"),
    ("security_analyzer", "security_findings"),
)


def _memory_index(
    records: list[MemoryRecord],
) -> dict[str, object]:
    """把共享记忆记录转成按key索引的查找表（同key取最新版本）。"""
    index: dict[str, object] = {}
    for record in records:
        index[record.key] = record.value
    return index


class SupervisorNodes:
    """主从编排新增节点：规划、校验、共享记忆、记忆汇总、冲突复核。"""

    def __init__(
        self,
        *,
        planner: ConstrainedPlanner | None = None,
        require_human_review: bool = False,
    ) -> None:
        self._planner = (
            planner
            if planner is not None
            else ConstrainedPlanner(invoker=None)
        )
        self._require_human_review = require_human_review

    # ------------------------------------------------------------------
    # 1. 主 Agent 受约束规划
    # ------------------------------------------------------------------
    async def plan_review(
        self,
        state: ReviewState,
    ) -> dict[str, object]:
        """把一次审核分解为受约束的任务计划。

        调用受约束规划器；规划器未配置大模型时自动回退到
        确定性规划器，保证始终产出可执行的计划。
        """
        contract = state["contract"]
        review_id = state["review_id"]
        trace_id = str(state.get("trace_id", ""))

        plan = await self._planner.plan(
            contract=contract,
            review_id=review_id,
            trace_id=trace_id,
            business_profile=str(
                state.get("business_profile", "")
            )
            or "baus",
            started_at_iso=str(
                state.get("started_at_iso", now_iso())
            ),
        )

        event = ReviewEvent(
            event_type="plan_created",
            aggregate_id=review_id,
            trace_id=trace_id,
            occurred_at_iso=now_iso(),
            payload={
                "plan_id": plan.plan_id,
                "task_count": len(plan.tasks),
            },
        )

        return {
            "plan": plan,
            "events": [event],
        }

    # ------------------------------------------------------------------
    # 2. 确定性校验
    # ------------------------------------------------------------------
    async def validate_plan(
        self,
        state: ReviewState,
    ) -> dict[str, object]:
        """用纯函数校验主 Agent 计划，不依赖大模型。

        校验规则由 :func:`validate_plan_against_contract` 统一定义：
        任务类型受约束、依赖自洽、条款ID真实存在、必要任务齐全。
        """
        plan_value = state.get("plan")
        if plan_value is None:
            raise SupervisorPlanError("plan is missing")

        plan = ReviewPlan.model_validate(plan_value)

        try:
            validate_plan_against_contract(
                plan,
                contract=state["contract"],
            )
        except Exception as exc:
            raise SupervisorPlanError(str(exc)) from exc

        validated_plan = plan.model_copy(
            update={"status": "validated"}
        )

        event = ReviewEvent(
            event_type="plan_validated",
            aggregate_id=state["review_id"],
            trace_id=str(state.get("trace_id", "")),
            occurred_at_iso=now_iso(),
            payload={"plan_id": validated_plan.plan_id},
        )

        return {
            "plan": validated_plan,
            "events": [event],
        }

    # ------------------------------------------------------------------
    # 4. 任务级共享记忆：写入
    # ------------------------------------------------------------------
    async def write_memory(
        self,
        state: ReviewState,
    ) -> dict[str, object]:
        """三个专业 Agent 完成后，把完整结论写入共享记忆。

        写入的不仅是指标，而是每个 Agent 的完整发现（JSON可序列化），
        使后续的汇总与冲突复核节点可以仅凭共享记忆重建结果，
        不再依赖三个并行通道的原始状态。
        """
        review_id = state["review_id"]
        written_at = now_iso()

        records: list[MemoryRecord] = []
        for agent_key, findings_key in _AGENT_SOURCES:
            findings = state.get(findings_key, [])
            records.append(
                MemoryRecord(
                    review_id=review_id,
                    scope="review",
                    key=_MEMORY_FINDINGS_KEY.format(agent=agent_key),
                    value=[
                        finding.model_dump(mode="json")
                        for finding in findings
                    ],
                    written_by="supervisor",
                    written_at_iso=written_at,
                )
            )
            records.append(
                MemoryRecord(
                    review_id=review_id,
                    scope="review",
                    key=_MEMORY_COUNT_KEY.format(agent=agent_key),
                    value=len(findings),
                    written_by="supervisor",
                    written_at_iso=written_at,
                )
            )

        event = ReviewEvent(
            event_type="memory_written",
            aggregate_id=review_id,
            trace_id=str(state.get("trace_id", "")),
            occurred_at_iso=written_at,
            payload={"memory_record_count": len(records)},
        )

        return {
            "memory_records": records,
            "events": [event],
        }

    # ------------------------------------------------------------------
    # 4. 任务级共享记忆：读取并汇总
    # ------------------------------------------------------------------
    async def aggregate_from_memory(
        self,
        state: ReviewState,
    ) -> dict[str, object]:
        """从共享记忆读取三路结论，重建稳定的最终发现列表。

        这是共享记忆的读侧：汇总不再读取三个并行通道的原始状态，
        而是从共享记忆重建，证明记忆是子 Agent 与汇总之间的
        协调通道。重建后仍执行 fixed_v1 同等的确定性校验
        （发现ID唯一、条款ID真实存在）。
        """
        index = _memory_index(state.get("memory_records", []))

        findings: list[RiskFinding] = []
        for agent_key, _ in _AGENT_SOURCES:
            raw = index.get(
                _MEMORY_FINDINGS_KEY.format(agent=agent_key),
                [],
            )
            if not isinstance(raw, list):
                raise AnalyzerProtocolError(
                    "shared memory contains a non-list findings value"
                )
            for item in raw:
                findings.append(RiskFinding.model_validate(item))

        valid_clause_ids = {
            clause.clause_id
            for clause in state["contract"].clauses
        }
        seen_finding_ids: set[str] = set()

        for finding in findings:
            if finding.finding_id in seen_finding_ids:
                raise AnalyzerProtocolError(
                    "Analyzer returned a duplicate finding_id"
                )
            seen_finding_ids.add(finding.finding_id)

            if (
                finding.finding_kind == "clause_risk"
                and finding.clause_id not in valid_clause_ids
            ):
                raise AnalyzerProtocolError(
                    "Analyzer referenced an unknown clause_id"
                )

        return {
            "status": "verifying",
            "findings": findings,
            "completed_nodes": ["aggregate_findings"],
        }

    # ------------------------------------------------------------------
    # 6. 有限协同复核（冲突检测）
    # ------------------------------------------------------------------
    async def detect_conflicts(
        self,
        state: ReviewState,
    ) -> dict[str, object]:
        """检测三路 Agent 对同一目标产生的矛盾结论，做一轮有限复核。

        从共享记忆读取各 Agent 的结论，仅当两个 Agent 对同一
        clause_id 给出不同 level 的风险时视为冲突；应用固定解决
        策略（保留更高级别），绝不再次调用 Agent，从结构上保证
        不会陷入争论循环。
        """
        review_id = state["review_id"]
        now = now_iso()

        index = _memory_index(state.get("memory_records", []))
        agent_sources: dict[str, str] = {}
        findings_by_clause: dict[str, list[tuple[str, str]]] = {}

        for agent_key, _ in _AGENT_SOURCES:
            raw = index.get(
                _MEMORY_FINDINGS_KEY.format(agent=agent_key),
                [],
            )
            if not isinstance(raw, list):
                continue
            for item in raw:
                finding = RiskFinding.model_validate(item)
                if finding.finding_kind != "clause_risk":
                    continue
                clause_id = finding.clause_id or ""
                findings_by_clause.setdefault(
                    clause_id,
                    [],
                ).append((finding.finding_id, finding.level))
                agent_sources[finding.finding_id] = agent_key

        level_rank = {"high": 3, "medium": 2, "low": 1}
        conflicts: list[ConflictRecord] = []

        for clause_id, entries in findings_by_clause.items():
            if len(entries) < 2:
                continue
            levels = {level for _, level in entries}
            if len(levels) < 2:
                continue

            # 冲突解决：保留更高级别，作为主结论。
            primary_id, _ = max(
                entries,
                key=lambda item: level_rank[item[1]],
            )
            for other_id, _ in entries:
                if other_id == primary_id:
                    continue
                conflicts.append(
                    ConflictRecord(
                        review_id=review_id,
                        task_a=agent_sources[primary_id],
                        task_b=agent_sources[other_id],
                        subject=f"clause:{clause_id}",
                        description=(
                            "Two agents disagree on the risk level "
                            "for the same clause."
                        ),
                        resolution="keep_primary",
                        resolved_by="supervisor",
                        created_at_iso=now,
                    )
                )

        event = ReviewEvent(
            event_type="conflict_detected",
            aggregate_id=review_id,
            trace_id=str(state.get("trace_id", "")),
            occurred_at_iso=now,
            payload={"conflict_count": len(conflicts)},
        )

        return {
            "conflict_records": conflicts,
            "events": [event],
        }


def build_supervisor_graph(
    *,
    retriever: RagRetriever,
    commercial_analyzer: RiskAnalyzer | None = None,
    legal_analyzer: RiskAnalyzer | None = None,
    security_analyzer: RiskAnalyzer | None = None,
    planner: ConstrainedPlanner | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    require_human_review: bool = False,
) -> CompiledStateGraph:
    """构建并编译 supervisor_v2 主从编排审核图。

    在 fixed_v1 的节点基础上插入：受约束规划、确定性校验、
    任务级共享记忆（写读闭环）、有限协同复核。检索与三路分析、
    引用核验、人工审批、收尾节点全部复用 ReviewNodes。
    """
    resolved_commercial = (
        commercial_analyzer
        if commercial_analyzer is not None
        else DemoCommercialRiskAnalyzer()
    )
    resolved_legal = (
        legal_analyzer
        if legal_analyzer is not None
        else DemoLegalRiskAnalyzer()
    )
    resolved_security = (
        security_analyzer
        if security_analyzer is not None
        else DemoSecurityRiskAnalyzer()
    )

    review_nodes = ReviewNodes(
        retriever=retriever,
        commercial_analyzer=resolved_commercial,
        legal_analyzer=resolved_legal,
        security_analyzer=resolved_security,
        require_human_review=require_human_review,
    )
    supervisor_nodes = SupervisorNodes(
        planner=planner,
        require_human_review=require_human_review,
    )

    builder = StateGraph(ReviewState)

    # 编排新增节点（统一套用时耗统计与结构化 span 日志）。
    builder.add_node(
        "plan_review",
        _timed_node("plan_review", supervisor_nodes.plan_review),
    )
    builder.add_node(
        "validate_plan",
        _timed_node("validate_plan", supervisor_nodes.validate_plan),
    )
    builder.add_node(
        "write_memory",
        _timed_node("write_memory", supervisor_nodes.write_memory),
    )
    builder.add_node(
        "aggregate_from_memory",
        _timed_node(
            "aggregate_from_memory",
            supervisor_nodes.aggregate_from_memory,
        ),
    )
    builder.add_node(
        "detect_conflicts",
        _timed_node(
            "detect_conflicts",
            supervisor_nodes.detect_conflicts,
        ),
    )

    # 复用 fixed_v1 的执行节点（统一套用时耗统计）。
    builder.add_node(
        "initialize_review",
        _timed_node("initialize_review", review_nodes.initialize_review),
    )
    builder.add_node(
        "retrieve_evidence",
        _timed_node("retrieve_evidence", review_nodes.retrieve_evidence),
    )
    builder.add_node(
        "commercial_risk_agent",
        _timed_node(
            "commercial_risk_agent",
            review_nodes.commercial_risk_agent,
        ),
    )
    builder.add_node(
        "legal_risk_agent",
        _timed_node("legal_risk_agent", review_nodes.legal_risk_agent),
    )
    builder.add_node(
        "security_risk_agent",
        _timed_node(
            "security_risk_agent",
            review_nodes.security_risk_agent,
        ),
    )
    builder.add_node(
        "verify_citations",
        _timed_node("verify_citations", review_nodes.verify_citations),
    )
    builder.add_node(
        "human_review",
        _timed_node("human_review", review_nodes.human_review),
    )
    builder.add_node(
        "finalize_review",
        _timed_node("finalize_review", review_nodes.finalize_review),
    )

    # 主 Agent 规划 → 确定性校验 → 初始化 → 检索。
    builder.add_edge(START, "plan_review")
    builder.add_edge("plan_review", "validate_plan")
    builder.add_edge("validate_plan", "initialize_review")
    builder.add_edge("initialize_review", "retrieve_evidence")

    # 检索后并行三路专业分析。
    builder.add_edge("retrieve_evidence", "commercial_risk_agent")
    builder.add_edge("retrieve_evidence", "legal_risk_agent")
    builder.add_edge("retrieve_evidence", "security_risk_agent")

    # 三路分析 → 写共享记忆 → 从记忆汇总 → 有限冲突复核 → 引用核验。
    builder.add_edge(
        [
            "commercial_risk_agent",
            "legal_risk_agent",
            "security_risk_agent",
        ],
        "write_memory",
    )
    builder.add_edge("write_memory", "aggregate_from_memory")
    builder.add_edge("aggregate_from_memory", "detect_conflicts")
    builder.add_edge("detect_conflicts", "verify_citations")

    builder.add_conditional_edges(
        "verify_citations",
        (
            (lambda state: "human_review")
            if require_human_review
            else route_after_verification
        ),
        {
            "human_review": "human_review",
            "finalize_review": "finalize_review",
        },
    )

    builder.add_edge("human_review", "finalize_review")
    builder.add_edge("finalize_review", END)

    return builder.compile(
        checkpointer=checkpointer,
        name="railguard_supervisor_review",
    )
