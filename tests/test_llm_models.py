"""大模型风险Agent严格结构化输出模型测试。"""

import json

import pytest
from pydantic import ValidationError

from railguard.agents.models import (
    LlmRiskAnalysis,
    LlmRiskCandidate,
)


def test_accepts_clause_risk_candidate() -> None:
    """验证已有条款风险可以引用真实条款和证据ID。"""
    candidate = LlmRiskCandidate(
        finding_kind="clause_risk",
        clause_id="clause-001",
        expected_clause=None,
        category="payment",
        level="high",
        reason="合同要求在交付前支付全部价款。",
        suggested_revision="建议设置分期付款并保留验收尾款。",
        evidence_ids=[
            "evidence-001",
            "evidence-002",
        ],
    )

    assert candidate.clause_id == "clause-001"
    assert candidate.expected_clause is None
    assert candidate.evidence_ids == [
        "evidence-001",
        "evidence-002",
    ]


def test_accepts_missing_clause_candidate() -> None:
    """验证缺失条款风险必须说明应该补充的条款。"""
    candidate = LlmRiskCandidate(
        finding_kind="missing_clause",
        clause_id=None,
        expected_clause="数据安全和保密义务条款",
        category="data_security",
        level="high",
        reason="合同没有约定供应商的数据安全义务。",
        suggested_revision="建议补充数据访问和删除要求。",
        evidence_ids=["evidence-001"],
    )

    assert candidate.clause_id is None
    assert candidate.expected_clause == (
        "数据安全和保密义务条款"
    )


def test_clause_risk_requires_clause_id() -> None:
    """验证已有条款风险不能缺少条款ID。"""
    with pytest.raises(
        ValidationError,
        match="clause_risk requires clause_id",
    ):
        LlmRiskCandidate(
            finding_kind="clause_risk",
            clause_id=None,
            expected_clause=None,
            category="payment",
            level="high",
            reason="存在付款风险。",
            suggested_revision="建议调整付款节点。",
            evidence_ids=[],
        )


def test_missing_clause_rejects_clause_id() -> None:
    """验证缺失条款风险不能伪造原文条款ID。"""
    with pytest.raises(
        ValidationError,
        match="must not define clause_id",
    ):
        LlmRiskCandidate(
            finding_kind="missing_clause",
            clause_id="clause-001",
            expected_clause="数据安全条款",
            category="data_security",
            level="high",
            reason="合同缺少数据安全要求。",
            suggested_revision="建议补充数据安全义务。",
            evidence_ids=[],
        )


def test_rejects_blank_business_text() -> None:
    """验证风险理由和修改建议不能只有空白字符。"""
    with pytest.raises(
        ValidationError,
        match="must not be blank",
    ):
        LlmRiskCandidate(
            finding_kind="missing_clause",
            clause_id=None,
            expected_clause="知识产权条款",
            category="intellectual_property",
            level="high",
            reason="   ",
            suggested_revision="建议明确成果归属。",
            evidence_ids=[],
        )


def test_rejects_duplicate_evidence_ids() -> None:
    """验证模型不能重复引用同一项证据。"""
    with pytest.raises(
        ValidationError,
        match="must not contain duplicates",
    ):
        LlmRiskCandidate(
            finding_kind="missing_clause",
            clause_id=None,
            expected_clause="运维服务条款",
            category="support",
            level="medium",
            reason="合同没有约定运维服务。",
            suggested_revision="建议明确质保和响应时间。",
            evidence_ids=[
                "evidence-001",
                "evidence-001",
            ],
        )


def test_analysis_rejects_unknown_fields() -> None:
    """验证模型输出不能包含协议以外的控制字段。"""
    with pytest.raises(ValidationError):
        LlmRiskAnalysis.model_validate(
            {
                "findings": [],
                "status": "approved",
            }
        )


def test_analysis_requires_findings_field() -> None:
    """验证没有风险时也必须显式返回空findings列表。"""
    with pytest.raises(ValidationError):
        LlmRiskAnalysis.model_validate({})

    analysis = LlmRiskAnalysis(
        findings=[]
    )

    assert analysis.findings == []


def test_json_schema_supports_strict_structured_output() -> None:
    """验证结构化输出Schema的全部字段都是必填字段。"""
    schema = LlmRiskAnalysis.model_json_schema()
    candidate_schema = schema["$defs"][
        "LlmRiskCandidate"
    ]

    assert set(schema["required"]) == {
        "findings"
    }
    assert set(candidate_schema["required"]) == set(
        candidate_schema["properties"]
    )
    assert (
        candidate_schema["additionalProperties"]
        is False
    )
    assert schema["additionalProperties"] is False

    category_schema = candidate_schema[
        "properties"
    ]["category"]

    assert set(category_schema["enum"]) == {
        "payment",
        "acceptance",
        "intellectual_property",
        "liability",
        "support",
        "data_security",
    }

    serialized_schema = json.dumps(schema)

    assert '"default"' not in serialized_schema
    assert '"minLength"' not in serialized_schema
    assert '"maxLength"' not in serialized_schema