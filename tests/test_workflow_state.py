"""LangGraph共享状态中的运行时模型测试。"""

import pytest
from pydantic import ValidationError

from railguard.workflow.state import (
    HumanReviewDecision,
    WorkflowError,
)


def test_human_reviewer_can_approve_findings() -> None:
    """验证人工审核可以分别接受或驳回风险项。"""
    decision = HumanReviewDecision(
        action="approve",
        reviewer="法务审核员",
        finding_decisions={
            "finding-payment": "accept",
            "finding-acceptance": "dismiss",
        },
        comment="付款风险保留，验收风险经核对后驳回。",
    )

    assert decision.action == "approve"
    assert decision.reviewer == "法务审核员"
    assert (
        decision.finding_decisions["finding-payment"]
        == "accept"
    )
    assert (
        decision.finding_decisions["finding-acceptance"]
        == "dismiss"
    )


@pytest.mark.parametrize(
    "action",
    [
        "reject",
        "request_changes",
    ],
)
def test_negative_decision_requires_comment(
    action: str,
) -> None:
    """验证驳回或要求修改时必须留下审核意见。"""
    with pytest.raises(ValidationError):
        HumanReviewDecision(
            action=action,
            reviewer="法务审核员",
            comment="   ",
        )


def test_human_decision_rejects_blank_identity_and_finding_id() -> None:
    """验证审核人与风险ID不能是空白字符串。"""
    with pytest.raises(ValidationError):
        HumanReviewDecision(
            action="approve",
            reviewer="   ",
        )

    with pytest.raises(ValidationError):
        HumanReviewDecision(
            action="approve",
            reviewer="法务审核员",
            finding_decisions={"   ": "accept"},
        )


def test_workflow_error_preserves_retry_information() -> None:
    """验证结构化错误可以记录节点和重试属性。"""
    error = WorkflowError(
        node="retrieve_evidence",
        code="rag_timeout",
        message="RAG服务请求超时",
        retryable=True,
        details={"provider": "external_rag"},
    )

    assert error.node == "retrieve_evidence"
    assert error.code == "rag_timeout"
    assert error.retryable is True
    assert error.details["provider"] == "external_rag"