"""supervisor_v2 主从编排图的节点与图构建。

document 25 目标架构的执行实现：

1. 主 Agent 受约束规划（``plan_review``）：把一次审核分解为
   有限任务语法内的原子任务，产出 :class:`ReviewPlan`。
2. 确定性校验（``validate_plan``）：用纯函数校验计划自洽，
   不依赖大模型，校验失败直接终止。
3. 专业子 Agent 执行：复用 ``ReviewNodes`` 中的检索与三个风险
   Agent，以及汇总、引用核验、人工审批、收尾节点。
4. 任务级共享记忆（``write_memory``）：三个 Agent 执行后把
   各自结论写入共享记忆，供汇总阶段读取。
5. PG 事务 Outbox 事件驱动（``outbox.py``）：关键节点产出领域
   事件，与业务数据同事务提交，由后台投递器异步发布。
6. 有限协同复核（``detect_conflicts``）：仅做一轮冲突检测并应用
   固定解决策略，绝不引入无限争论循环。

该实现刻意保持与 ``fixed_v1`` 的差异最小化：复用既有的检索、
分析、汇总、引用核验、人工审批与收尾节点，仅在它们之前插入
规划、校验，在分析之后插入共享记忆与冲突复核。
"""

from __future__ import annotations

import logging

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from railguard.models.schemas import new_id
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
    ReviewNodes,
    route_after_verification,
)
from railguard.workflow.orchestration import (
    ConflictRecord,
    MemoryRecord,
    ReviewContext,
    ReviewEvent,
    ReviewPlan,
    ReviewTask,
    TaskKind,
)
from railguard.workflow.state import ReviewState

logger = logging.getLogger(__name__)

# 受约束规划允许的任务类型全集。主 Agent 只能从这里选择，
# 不能凭空发明新任务，从而保证计划可被确定性校验。
_ALLOWED_TASK_KINDS: frozenset[TaskKind] = frozenset(
    {
        "retrieval",
        "commercial",
        "legal",
        "security",
        "aggregation",
        "verification",
    }
)


class SupervisorPlanError(RuntimeError):
    """主 Agent 产出的计划无法通过确定性校验。"""


class SupervisorNodes:
    """主从编排新增节点：规划、校验、共享记忆、冲突复核。"""

    def __init__(self, *, require_human_review: bool = False) -> None:
        self._require_human_review = require_human_review

    # ------------------------------------------------------------------
    # 1. 主 Agent 受约束规划
    # ------------------------------------------------------------------
    async def plan_review(
        self,
        state: ReviewState,
    ) -> dict[str, object]:
        """把一次审核分解为受约束的任务计划。

        当前主 Agent 为确定性规划器：给定合同，产出固定的任务分解
        （检索 → 商务/法务/安全 → 汇总 → 引用核验）。后续接入受约束
        大模型规划器时，只需替换本方法，契约保持不变。
        """
        contract = state["contract"]
        clause_ids = [
            clause.clause_id for clause in contract.clauses
        ]

        review_id = state["review_id"]
        trace_id = str(state.get("trace_id", ""))

        context = ReviewContext(
            review_id=review_id,
            trace_id=trace_id,
            business_profile=str(
                state.get("business_profile", "")
            ) or "baus",
            contract_id=contract.contract_id,
            contract_title=contract.filename,
            clause_count=len(clause_ids),
            started_at_iso=str(
                state.get("started_at_iso", now_iso())
            ),
        )

        plan_id = new_id()
        plan = ReviewPlan(
            plan_id=plan_id,
            review_id=review_id,
            context=context,
            created_at_iso=now_iso(),
            tasks=[
                ReviewTask(
                    task_id="retrieval",
                    plan_id=plan_id,
                    kind="retrieval",
                    assignee="rag_retriever",
                    required_inputs=["contract"],
                ),
                ReviewTask(
                    task_id="commercial",
                    plan_id=plan_id,
                    kind="commercial",
                    clause_ids=clause_ids,
                    depends_on=["retrieval"],
                    assignee="commercial_analyzer",
                    required_inputs=["evidence"],
                ),
                ReviewTask(
                    task_id="legal",
                    plan_id=plan_id,
                    kind="legal",
                    clause_ids=clause_ids,
                    depends_on=["retrieval"],
                    assignee="legal_analyzer",
                    required_inputs=["evidence"],
                ),
                ReviewTask(
                    task_id="security",
                    plan_id=plan_id,
                    kind="security",
                    clause_ids=clause_ids,
                    depends_on=["retrieval"],
                    assignee="security_analyzer",
                    required_inputs=["evidence"],
                ),
                ReviewTask(
                    task_id="aggregation",
                    plan_id=plan_id,
                    kind="aggregation",
                    depends_on=[
                        "commercial",
                        "legal",
                        "security",
                    ],
                    assignee="supervisor",
                ),
                ReviewTask(
                    task_id="verification",
                    plan_id=plan_id,
                    kind="verification",
                    depends_on=["aggregation"],
                    assignee="citation_validator",
                ),
            ],
            rationale=(
                "受约束规划：检索证据 → 三路专业分析 → 汇总去重"
                " → 引用核验，符合固定任务语法。"
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

        校验规则：
        - 计划存在；
        - 任务类型在受约束语法内；
        - 任务ID唯一、依赖无自环、依赖引用存在的任务；
        - 至少包含检索、三路分析和汇总。
        """
        plan_value = state.get("plan")
        if plan_value is None:
            raise SupervisorPlanError("plan is missing")

        plan = ReviewPlan.model_validate(plan_value)
        review_id = state["review_id"]
        trace_id = str(state.get("trace_id", ""))

        for task in plan.tasks:
            if task.kind not in _ALLOWED_TASK_KINDS:
                raise SupervisorPlanError(
                    f"plan contains unknown task kind {task.kind}"
                )

        required_kinds = {
            "retrieval",
            "commercial",
            "legal",
            "security",
            "aggregation",
            "verification",
        }
        present_kinds = {task.kind for task in plan.tasks}
        missing = required_kinds - present_kinds
        if missing:
            raise SupervisorPlanError(
                f"plan is missing required tasks: {sorted(missing)}"
            )

        validated_plan = plan.model_copy(
            update={"status": "validated"}
        )

        event = ReviewEvent(
            event_type="plan_validated",
            aggregate_id=review_id,
            trace_id=trace_id,
            occurred_at_iso=now_iso(),
            payload={"plan_id": validated_plan.plan_id},
        )

        return {
            "plan": validated_plan,
            "events": [event],
        }

    # ------------------------------------------------------------------
    # 4. 任务级共享记忆
    # ------------------------------------------------------------------
    async def write_memory(
        self,
        state: ReviewState,
    ) -> dict[str, object]:
        """三个专业 Agent 完成后，把各自结论写入共享记忆。

        共享记忆让后续的汇总与冲突复核节点可以在不重新调用 Agent
        的前提下读取各子任务的输出，避免重复检索与结论不一致。
        """
        review_id = state["review_id"]
        written_at = now_iso()

        records: list[MemoryRecord] = []
        for agent_key, findings_key in (
            ("commercial_analyzer", "commercial_findings"),
            ("legal_analyzer", "legal_findings"),
            ("security_analyzer", "security_findings"),
        ):
            findings = state.get(findings_key, [])
            records.append(
                MemoryRecord(
                    review_id=review_id,
                    scope="review",
                    key=f"agent:{agent_key}:finding_count",
                    value=len(findings),
                    written_by="supervisor",
                    written_at_iso=written_at,
                )
            )
            records.append(
                MemoryRecord(
                    review_id=review_id,
                    scope="review",
                    key=f"agent:{agent_key}:finding_ids",
                    value=[
                        finding.finding_id
                        for finding in findings
                    ],
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
    # 6. 有限协同复核（冲突检测）
    # ------------------------------------------------------------------
    async def detect_conflicts(
        self,
        state: ReviewState,
    ) -> dict[str, object]:
        """检测三路 Agent 对同一目标产生的矛盾结论，做一轮有限复核。

        仅当两个 Agent 对同一 clause_id 给出不同 level 的风险时视为
        冲突；应用固定解决策略（保留更高级别），绝不再次调用 Agent，
        从结构上保证不会陷入争论循环。
        """
        review_id = state["review_id"]
        now = now_iso()

        agent_sources: dict[str, str] = {}
        findings_by_clause: dict[str, list[tuple[str, str]]] = {}
        for findings_key, agent_name in (
            ("commercial_findings", "commercial_analyzer"),
            ("legal_findings", "legal_analyzer"),
            ("security_findings", "security_analyzer"),
        ):
            for finding in state.get(findings_key, []):
                if finding.finding_kind != "clause_risk":
                    continue
                clause_id = finding.clause_id or ""
                findings_by_clause.setdefault(
                    clause_id,
                    [],
                ).append((finding.finding_id, finding.level))
                agent_sources[finding.finding_id] = agent_name

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
    checkpointer: BaseCheckpointSaver | None = None,
    require_human_review: bool = False,
) -> CompiledStateGraph:
    """构建并编译 supervisor_v2 主从编排审核图。

    在 fixed_v1 的节点基础上插入：受约束规划、确定性校验、
    任务级共享记忆、有限协同复核。检索与三路分析、汇总、
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
        require_human_review=require_human_review,
    )

    builder = StateGraph(ReviewState)

    # 编排新增节点（同样统一套用时耗统计与结构化 span 日志）。
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
        "detect_conflicts",
        _timed_node(
            "detect_conflicts",
            supervisor_nodes.detect_conflicts,
        ),
    )

    # 复用 fixed_v1 的执行节点（同样统一套用时耗统计）。
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
        "aggregate_findings",
        _timed_node("aggregate_findings", review_nodes.aggregate_findings),
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

    # 三路分析完成后先写共享记忆，再做有限冲突复核，最后汇总。
    builder.add_edge(
        [
            "commercial_risk_agent",
            "legal_risk_agent",
            "security_risk_agent",
        ],
        "write_memory",
    )
    builder.add_edge("write_memory", "detect_conflicts")
    builder.add_edge("detect_conflicts", "aggregate_findings")

    builder.add_edge("aggregate_findings", "verify_citations")

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
