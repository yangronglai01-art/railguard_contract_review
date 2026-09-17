"""监督微调标注协议测试。"""

import pytest
from pydantic import ValidationError

from railguard.agents.models import LlmRiskAnalysis, LlmRiskCandidate
from railguard.models.schemas import Clause, ContractDocument, Evidence
from railguard.training.models import AnnotationQuality, AnnotationSource, SftAnnotation


def create_annotation() -> SftAnnotation:
    """创建通过全部训练数据约束的商务风险标注。"""
    clause_text = "合同签订后支付全部价款。"
    clause = Clause(
        clause_id="payment-clause",
        title="付款方式",
        text=clause_text,
        start_offset=0,
        end_offset=len(clause_text),
    )
    evidence = Evidence(
        evidence_id="payment-evidence",
        document_id="rule-payment",
        title="付款审核规则",
        content="付款应与交付验收挂钩。",
        source="付款规则第1条",
        metadata={"category": "payment"},
    )
    return SftAnnotation(
        example_id="sft-payment-001",
        split="sft_train",
        agent_name="commercial_risk_agent",
        source=AnnotationSource(
            origin="synthetic",
            created_by="annotator-01",
            license_or_authorization="project-owned",
            deidentified=True,
        ),
        contract=ContractDocument(
            contract_id="sft-contract-001",
            filename="sft-contract-001.txt",
            full_text=clause.text,
            clauses=[clause],
        ),
        evidence_by_clause={clause.clause_id: [evidence]},
        target=LlmRiskAnalysis(
            findings=[
                LlmRiskCandidate(
                    finding_kind="clause_risk",
                    clause_id=clause.clause_id,
                    expected_clause=None,
                    category="payment",
                    level="high",
                    reason="交付验收前已支付全部价款。",
                    suggested_revision="改为分期付款并保留尾款。",
                    evidence_ids=[evidence.evidence_id],
                )
            ]
        ),
        quality=AnnotationQuality(
            annotator="annotator-01",
            reviewer="reviewer-01",
            review_status="approved",
        ),
    )


def test_approved_annotation_is_training_ready() -> None:
    """验证合规标注可以进入训练数据导出阶段。"""
    create_annotation().require_training_ready()


def test_annotation_rejects_category_outside_agent_scope() -> None:
    """验证商务Agent标注不能混入数据安全目标。"""
    payload = create_annotation().model_dump()
    payload["target"]["findings"][0]["category"] = "data_security"
    with pytest.raises(ValidationError, match="outside agent scope"):
        SftAnnotation.model_validate(payload)


def test_annotation_rejects_forbidden_evidence() -> None:
    """验证目标不能引用当前输入白名单之外的证据。"""
    payload = create_annotation().model_dump()
    payload["target"]["findings"][0]["evidence_ids"] = ["unknown"]
    with pytest.raises(ValidationError, match="forbidden evidence_id"):
        SftAnnotation.model_validate(payload)


def test_draft_annotation_is_not_training_ready() -> None:
    """验证未完成复核的草稿不能进入训练导出。"""
    annotation = create_annotation()
    annotation.quality.review_status = "draft"
    with pytest.raises(ValueError, match="not approved"):
        annotation.require_training_ready()
