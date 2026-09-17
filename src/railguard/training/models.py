"""监督微调标注记录和业务一致性校验。"""

from typing import Literal, Self

from pydantic import Field, model_validator

from railguard.agents.models import LlmRiskAnalysis
from railguard.models.schemas import ContractDocument, Evidence, SchemaModel

AgentName = Literal[
    "commercial_risk_agent",
    "legal_risk_agent",
    "security_risk_agent",
]
ALLOWED_CATEGORIES = {
    "commercial_risk_agent": {"payment", "acceptance"},
    "legal_risk_agent": {"intellectual_property", "liability", "support"},
    "security_risk_agent": {"data_security"},
}


class AnnotationSource(SchemaModel):
    """训练样本的来源、授权和去标识信息。"""

    origin: Literal["synthetic", "authorized_internal", "licensed"]
    created_by: str = Field(min_length=1)
    license_or_authorization: str = Field(min_length=1)
    deidentified: bool


class AnnotationQuality(SchemaModel):
    """训练样本的标注与复核状态。"""

    annotator: str = Field(min_length=1)
    reviewer: str = Field(min_length=1)
    review_status: Literal["draft", "approved", "rejected"]
    notes: str = ""


class SftAnnotation(SchemaModel):
    """一条可转换为专业Agent训练消息的严格标注记录。"""

    example_id: str = Field(min_length=1)
    split: Literal["sft_train", "sft_validation"]
    agent_name: AgentName
    source: AnnotationSource
    contract: ContractDocument
    evidence_by_clause: dict[str, list[Evidence]] = Field(default_factory=dict)
    contract_evidence: list[Evidence] = Field(default_factory=list)
    target: LlmRiskAnalysis
    quality: AnnotationQuality

    @model_validator(mode="after")
    def validate_business_protocol(self) -> Self:
        """验证职责范围、合同定位和证据白名单。"""
        clause_ids = {clause.clause_id for clause in self.contract.clauses}
        if set(self.evidence_by_clause) - clause_ids:
            raise ValueError("evidence_by_clause contains unknown clause_id")

        contract_evidence = {item.evidence_id: item for item in self.contract_evidence}
        for finding in self.target.findings:
            if finding.category not in ALLOWED_CATEGORIES[self.agent_name]:
                raise ValueError("target category is outside agent scope")
            if finding.finding_kind == "clause_risk" and finding.clause_id not in clause_ids:
                raise ValueError("target references unknown clause_id")

            allowed = dict(contract_evidence)
            if finding.clause_id is not None:
                allowed.update(
                    {
                        item.evidence_id: item
                        for item in self.evidence_by_clause.get(finding.clause_id, [])
                    }
                )
            for evidence_id in finding.evidence_ids:
                evidence = allowed.get(evidence_id)
                if evidence is None:
                    raise ValueError("target references forbidden evidence_id")
                if evidence.metadata.get("category") != finding.category:
                    raise ValueError("target evidence category does not match")
        return self

    def require_training_ready(self) -> None:
        """拒绝未完成复核或未去标识的训练输入。"""
        if self.quality.review_status != "approved":
            raise ValueError("annotation is not approved")
        if not self.source.deidentified:
            raise ValueError("annotation is not deidentified")
