"""supervisor_v2 主从编排契约与图的回归测试。

覆盖 document 25 目标架构的核心不变量：
- 编排契约可被确定性校验（计划自洽、任务语法受约束）；
- supervisor_v2 图端到端可运行，产出计划、共享记忆、冲突记录与领域事件；
- fixed_v1 与 supervisor_v2 通过配置开关解耦。
"""

import asyncio

import pytest

from railguard.models.schemas import Clause, ContractDocument, new_id
from railguard.observability import new_trace_id, now_iso
from railguard.rag.mock import MockRagRetriever
from railguard.workflow.orchestration import (
    ReviewContext,
    ReviewEvent,
    ReviewPlan,
    ReviewTask,
)
from railguard.workflow.supervisor import build_supervisor_graph


def _make_contract() -> ContractDocument:
    c1 = "甲方应于验收通过后三十日内支付合同款项。"
    c2 = "乙方应采取必要措施保护甲方数据安全。"
    text = c1 + c2
    return ContractDocument(
        contract_id="c1",
        filename="合同.pdf",
        full_text=text,
        clauses=[
            Clause(
                clause_id="cl1",
                title="付款条款",
                text=c1,
                start_offset=0,
                end_offset=len(c1),
            ),
            Clause(
                clause_id="cl2",
                title="数据条款",
                text=c2,
                start_offset=len(c1),
                end_offset=len(text),
            ),
        ],
    )


def _valid_plan() -> ReviewPlan:
    context = ReviewContext(
        review_id="r1",
        trace_id=new_trace_id(),
        business_profile="baus",
        contract_id="c1",
        contract_title="合同.pdf",
        clause_count=2,
        started_at_iso=now_iso(),
    )
    plan_id = new_id()
    return ReviewPlan(
        plan_id=plan_id,
        review_id="r1",
        context=context,
        created_at_iso=now_iso(),
        tasks=[
            ReviewTask(task_id="retrieval", plan_id=plan_id, kind="retrieval"),
            ReviewTask(
                task_id="commercial",
                plan_id=plan_id,
                kind="commercial",
                clause_ids=["cl1"],
                depends_on=["retrieval"],
            ),
            ReviewTask(task_id="legal", plan_id=plan_id, kind="legal", depends_on=["retrieval"]),
            ReviewTask(task_id="security", plan_id=plan_id, kind="security", depends_on=["retrieval"]),
            ReviewTask(
                task_id="aggregation",
                plan_id=plan_id,
                kind="aggregation",
                depends_on=["commercial", "legal", "security"],
            ),
            ReviewTask(
                task_id="verification",
                plan_id=plan_id,
                kind="verification",
                depends_on=["aggregation"],
            ),
        ],
    )


def test_plan_integrity_accepts_valid_plan() -> None:
    plan = _valid_plan()
    assert plan.status == "draft"
    assert len(plan.tasks) == 6


def test_plan_rejects_unknown_task_kind() -> None:
    plan = _valid_plan()
    with pytest.raises(ValueError):
        ReviewTask(
            task_id="weird",
            plan_id=plan.plan_id,
            kind="not-a-real-kind",  # type: ignore[arg-type]
        )


def test_plan_rejects_self_dependency() -> None:
    plan = _valid_plan()
    context = plan.context
    with pytest.raises(ValueError):
        ReviewPlan(
            plan_id=plan.plan_id,
            review_id=plan.review_id,
            context=context,
            created_at_iso=now_iso(),
            tasks=[
                ReviewTask(
                    task_id="retrieval",
                    plan_id=plan.plan_id,
                    kind="retrieval",
                    depends_on=["retrieval"],
                )
            ],
        )


def test_plan_rejects_unknown_dependency() -> None:
    plan = _valid_plan()
    with pytest.raises(ValueError):
        ReviewPlan(
            plan_id=plan.plan_id,
            review_id=plan.review_id,
            context=plan.context,
            created_at_iso=now_iso(),
            tasks=[
                ReviewTask(
                    task_id="commercial",
                    plan_id=plan.plan_id,
                    kind="commercial",
                    depends_on=["missing-task"],
                )
            ],
        )


def test_event_envelope_defaults_to_pending() -> None:
    event = ReviewEvent(
        event_type="plan_created",
        aggregate_id="r1",
        trace_id=new_trace_id(),
        occurred_at_iso=now_iso(),
    )
    assert event.delivery_status == "pending"


def test_supervisor_graph_runs_end_to_end() -> None:
    async def _run() -> dict:
        graph = build_supervisor_graph(
            retriever=MockRagRetriever([]),
            require_human_review=False,
        )
        initial = {
            "review_id": "r1",
            "contract": _make_contract(),
            "trace_id": new_trace_id(),
            "started_at_iso": now_iso(),
            "business_profile": "baus",
        }
        return await graph.ainvoke(initial)

    result = asyncio.run(_run())

    plan = result["plan"]
    assert plan.status == "validated"
    assert [t.kind for t in plan.tasks] == [
        "retrieval",
        "commercial",
        "legal",
        "security",
        "aggregation",
        "verification",
    ]

    # 共享记忆：三个 Agent 各写入 2 条记录。
    assert len(result["memory_records"]) == 6

    # 领域事件：规划→校验→记忆→冲突检测。
    event_types = [e.event_type for e in result["events"]]
    assert "plan_created" in event_types
    assert "plan_validated" in event_types
    assert "memory_written" in event_types
    assert "conflict_detected" in event_types

    # 冲突记录为列表，可被 add reducer 合并。
    assert isinstance(result["conflict_records"], list)
