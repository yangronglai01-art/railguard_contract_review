"""合同审核工作流服务测试。"""

from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from railguard.models.schemas import (
    ContractDocument,
    Evidence,
)
from railguard.parsers.clauses import split_clauses
from railguard.rag.client import (
    RagRetriever,
    RagUnavailableError,
)
from railguard.rag.mock import MockRagRetriever
from railguard.rag.models import RagSearchRequest
from railguard.workflow.graph import build_review_graph
from railguard.workflow.service import (
    ReviewConflictError,
    ReviewDecisionError,
    ReviewExecutionError,
    ReviewNotFoundError,
    ReviewService,
)
from railguard.workflow.state import HumanReviewDecision


def create_contract(
    *,
    filename: str,
    full_text: str,
) -> ContractDocument:
    """创建带有可追溯条款的测试合同。"""
    return ContractDocument(
        filename=filename,
        full_text=full_text,
        clauses=split_clauses(full_text),
    )


def create_demo_contract() -> ContractDocument:
    """创建会触发六项风险的演示合同。"""
    full_text = (
        "设备监测平台软件采购合同\n\n"
        "第一条 项目内容\n\n"
        "乙方为甲方建设设备监测平台。\n\n"
        "第二条 付款方式\n\n"
        "合同签订后五日内，甲方支付全部合同款。\n\n"
        "第三条 验收\n\n"
        "系统上线三日后，甲方未提出异议视为验收合格。"
    )

    return create_contract(
        filename="software-purchase-demo.docx",
        full_text=full_text,
    )


def create_complete_contract() -> ContractDocument:
    """创建不会触发演示风险的完整合同。"""
    full_text = (
        "设备监测平台软件采购合同\n\n"
        "第一条 付款方式\n\n"
        "最终验收并经双方书面确认后支付尾款。\n\n"
        "第二条 验收\n\n"
        "双方按照测试标准完成书面验收确认。\n\n"
        "第三条 知识产权\n\n"
        "双方明确源代码和开发成果的知识产权归属。\n\n"
        "第四条 违约责任\n\n"
        "延期交付时乙方承担违约责任和赔偿责任。\n\n"
        "第五条 运维服务\n\n"
        "乙方提供一年质保期和运维服务。\n\n"
        "第六条 数据安全\n\n"
        "乙方承担数据安全和保密义务。"
    )

    return create_contract(
        filename="complete-contract.docx",
        full_text=full_text,
    )


def create_mock_retriever() -> MockRagRetriever:
    """创建本地演示RAG检索器。"""
    return MockRagRetriever.from_json_file(
        Path("data/demo/rag-corpus.json")
    )


def create_service(
    retriever: RagRetriever,
) -> ReviewService:
    """创建使用内存checkpoint的测试审核服务。"""
    graph = build_review_graph(
        retriever=retriever,
        checkpointer=InMemorySaver(),
    )

    return ReviewService(graph)


class FailingRetriever:
    """始终模拟外部RAG连接失败的测试检索器。"""

    async def retrieve(
        self,
        request: RagSearchRequest,
    ) -> list[Evidence]:
        """抛出服务不可用异常。"""
        del request

        raise RagUnavailableError(
            "mock RAG service is unavailable"
        )


async def test_complete_review_returns_final_empty_result() -> None:
    """验证无风险合同自动完成且最终风险为空。"""
    service = create_service(create_mock_retriever())

    result = await service.start_review(
        create_complete_contract()
    )

    assert result.status == "approved"
    assert result.findings == []
    assert result.final_findings == []
    assert result.human_decision is None
    assert "自动审核完成" in result.final_summary
    assert result.errors == []


async def test_invalid_decision_does_not_consume_interrupt() -> None:
    """验证无效决定后仍可提交有效决定并阻止重复审批。"""
    service = create_service(create_mock_retriever())
    waiting_result = await service.start_review(
        create_demo_contract()
    )

    assert waiting_result.status == "awaiting_human"
    assert len(waiting_result.findings) == 6
    assert waiting_result.final_findings is None
    assert waiting_result.evidence

    invalid_decision = HumanReviewDecision(
        action="approve",
        reviewer="法务审核员",
        finding_decisions={
            "unknown-finding": "dismiss",
        },
        comment="测试无效风险ID。",
    )

    with pytest.raises(ReviewDecisionError):
        await service.submit_decision(
            review_id=waiting_result.review_id,
            decision=invalid_decision,
        )

    # 无效决定没有恢复LangGraph，任务仍处于人工等待状态。
    still_waiting = await service.get_review(
        waiting_result.review_id
    )
    assert still_waiting.status == "awaiting_human"
    assert still_waiting.human_decision is None

    dismissed_finding_id = (
        waiting_result.findings[0].finding_id
    )
    valid_decision = HumanReviewDecision(
        action="approve",
        reviewer="法务审核员",
        finding_decisions={
            dismissed_finding_id: "dismiss",
        },
        comment="驳回一项，其余风险保留。",
    )

    completed = await service.submit_decision(
        review_id=waiting_result.review_id,
        decision=valid_decision,
    )

    assert completed.status == "approved"
    assert completed.final_findings is not None
    assert len(completed.final_findings) == 5
    assert len(completed.findings) == 6
    assert completed.human_decision == valid_decision

    # 同一任务完成后不能再次提交决定。
    with pytest.raises(ReviewConflictError):
        await service.submit_decision(
            review_id=waiting_result.review_id,
            decision=valid_decision,
        )


async def test_execution_failure_preserves_queryable_review() -> None:
    """验证执行失败时返回review_id并可查询失败checkpoint。"""
    service = create_service(FailingRetriever())

    with pytest.raises(ReviewExecutionError) as exc_info:
        await service.start_review(
            create_demo_contract()
        )

    execution_error = exc_info.value

    assert execution_error.code == "rag_unavailable"
    assert execution_error.retryable is True

    failed_review = await service.get_review(
        execution_error.review_id
    )

    assert failed_review.status == "failed"
    assert failed_review.final_findings is None
    assert failed_review.errors
    # checkpoint中的异常采用安全的通用错误展示。
    # 精确的RAG错误代码已经由ReviewExecutionError返回。
    assert (
        failed_review.errors[0].code
        == "workflow_execution_failed"
    )
    assert failed_review.errors[0].retryable is False


async def test_missing_review_raises_not_found() -> None:
    """验证查询不存在的审核任务会返回明确异常。"""
    service = create_service(create_mock_retriever())

    with pytest.raises(ReviewNotFoundError):
        await service.get_review("not-found")