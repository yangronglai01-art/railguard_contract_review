"""大模型风险Agent使用的严格结构化输出模型。"""

from typing import Literal, Self

from pydantic import (
    field_validator,
    model_validator,
)

from railguard.models.schemas import SchemaModel


class LlmRiskCandidate(SchemaModel):
    """大模型提出的一项待验证合同风险。"""

    # 风险针对已有条款或合同缺失的必要条款。
    finding_kind: Literal[
        "clause_risk",
        "missing_clause",
    ]

    # 字段必须返回；缺失条款风险显式返回null。
    clause_id: str | None

    # 字段必须返回；已有条款风险显式返回null。
    expected_clause: str | None

    # 模型只能输出首版六种风险分类。
    # 具体Agent还会进一步校验自己的允许分类。
    category: Literal[
        "payment",
        "acceptance",
        "intellectual_property",
        "liability",
        "support",
        "data_security",
    ]

    # 风险等级与现有业务模型保持一致。
    level: Literal[
        "low",
        "medium",
        "high",
    ]

    # 面向人工审核人的风险判断理由。
    reason: str

    # 可以交给合同经办人参考的修改建议。
    suggested_revision: str

    # 字段必须返回；没有支持证据时显式返回空列表。
    evidence_ids: list[str]

    @field_validator(
        "category",
        "reason",
        "suggested_revision",
    )
    @classmethod
    def validate_non_blank_text(
        cls,
        value: str,
    ) -> str:
        """在本地拒绝只有空白字符的业务文本。"""
        if not value.strip():
            raise ValueError(
                "risk candidate text must not be blank"
            )

        return value

    @field_validator("evidence_ids")
    @classmethod
    def validate_evidence_ids(
        cls,
        value: list[str],
    ) -> list[str]:
        """在本地拒绝空白或重复的证据ID。"""
        if any(
            not evidence_id.strip()
            for evidence_id in value
        ):
            raise ValueError(
                "evidence_id must not be blank"
            )

        if len(value) != len(set(value)):
            raise ValueError(
                "evidence_ids must not contain duplicates"
            )

        return value

    @model_validator(mode="after")
    def validate_location(self) -> Self:
        """验证风险类型与合同定位字段一致。"""
        if self.finding_kind == "clause_risk":
            if (
                self.clause_id is None
                or not self.clause_id.strip()
            ):
                raise ValueError(
                    "clause_risk requires clause_id"
                )

            if self.expected_clause is not None:
                raise ValueError(
                    "clause_risk must not define "
                    "expected_clause"
                )
        else:
            if self.clause_id is not None:
                raise ValueError(
                    "missing_clause must not define "
                    "clause_id"
                )

            if (
                self.expected_clause is None
                or not self.expected_clause.strip()
            ):
                raise ValueError(
                    "missing_clause requires "
                    "expected_clause"
                )

        return self


class LlmRiskAnalysis(SchemaModel):
    """一次专业风险Agent的完整严格结构化输出。"""

    # 字段必须返回；没有发现风险时显式返回空列表。
    findings: list[LlmRiskCandidate]