"""合同审核离线评测使用的数据模型。"""

from datetime import datetime
from typing import Literal, Self

from pydantic import Field, model_validator

from railguard.models.schemas import (
    RiskFinding,
    SchemaModel,
)

# 评测标签使用与业务风险模型相同的风险类型。
ExpectedFindingKind = Literal[
    "clause_risk",
    "missing_clause",
]

# 评测标签使用与业务风险模型相同的风险等级。
ExpectedRiskLevel = Literal[
    "low",
    "medium",
    "high",
]


class ExpectedFinding(SchemaModel):
    """一项由人工标注的预期合同风险。"""

    # 风险类别，例如payment或data_security。
    category: str = Field(min_length=1)

    # 风险针对已有条款或缺失条款。
    finding_kind: ExpectedFindingKind

    # 人工标注的风险等级。
    level: ExpectedRiskLevel

    # 已有条款风险必须在对应条款原文中包含该文本。
    source_text_contains: str | None = None

    # 缺失条款风险必须在Agent说明中包含该文本。
    expected_clause_contains: str | None = None

    # 标注说明用于记录判定依据，不参与自动计分。
    annotation_note: str = ""

    @model_validator(mode="after")
    def validate_location_expectation(self) -> Self:
        """验证风险类型包含相应的原文或缺失条款标记。"""
        if self.finding_kind == "clause_risk":
            if (
                self.source_text_contains is None
                or not self.source_text_contains.strip()
            ):
                raise ValueError(
                    "clause_risk requires source_text_contains"
                )

            if self.expected_clause_contains is not None:
                raise ValueError(
                    "clause_risk must not define "
                    "expected_clause_contains"
                )
        else:
            if (
                self.expected_clause_contains is None
                or not self.expected_clause_contains.strip()
            ):
                raise ValueError(
                    "missing_clause requires "
                    "expected_clause_contains"
                )

            if self.source_text_contains is not None:
                raise ValueError(
                    "missing_clause must not define "
                    "source_text_contains"
                )

        return self

    def match_key(self) -> str:
        """返回当前首版评测使用的稳定风险匹配键。"""
        return f"{self.finding_kind}:{self.category}"


class EvaluationCase(SchemaModel):
    """一份合同及其人工预期风险标签。"""

    # 评测案例的稳定唯一标识。
    case_id: str = Field(min_length=1)

    # 便于阅读的案例名称。
    title: str = Field(min_length=1)

    # 构造ContractDocument时使用的文件名。
    filename: str = Field(min_length=1)

    # 已规范化的合同全文。
    contract_text: str = Field(min_length=1)

    # 人工标注的预期风险。
    expected_findings: list[ExpectedFinding] = Field(
        default_factory=list
    )

    # 用于筛选和统计案例类型的标签。
    tags: list[str] = Field(default_factory=list)

    # 案例设计说明，不参与自动计分。
    notes: str = ""

    @model_validator(mode="after")
    def validate_unique_labels(self) -> Self:
        """确保同一案例没有重复的首版匹配键。"""
        match_keys = [
            finding.match_key()
            for finding in self.expected_findings
        ]

        if len(match_keys) != len(set(match_keys)):
            raise ValueError(
                "expected finding match keys must be unique "
                "within one case"
            )

        if any(not tag.strip() for tag in self.tags):
            raise ValueError(
                "evaluation tags must not be blank"
            )

        return self


class EvaluationDataset(SchemaModel):
    """一组使用相同版本和标注规则的合同评测案例。"""

    # 评测集名称。
    dataset_name: str = Field(min_length=1)

    # 数据版本，用于比较不同实验结果。
    version: str = Field(min_length=1)

    # 评测集范围和标注口径说明。
    description: str = Field(min_length=1)

    # 参与本次评测的全部案例。
    cases: list[EvaluationCase] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_case_ids(self) -> Self:
        """确保评测集中的案例ID不重复。"""
        case_ids = [
            evaluation_case.case_id
            for evaluation_case in self.cases
        ]

        if len(case_ids) != len(set(case_ids)):
            raise ValueError(
                "evaluation case_id values must be unique"
            )

        return self


class FindingMatch(SchemaModel):
    """一项预期风险与Agent预测风险的匹配明细。"""

    # finding_kind和category组成的匹配键。
    match_key: str = Field(min_length=1)

    # Agent返回的风险ID。
    finding_id: str = Field(min_length=1)

    # 人工标注风险等级。
    expected_level: ExpectedRiskLevel

    # Agent预测风险等级。
    predicted_level: ExpectedRiskLevel

    # 风险等级是否与人工标注一致。
    level_matched: bool

    # 条款原文或缺失条款名称是否匹配。
    location_matched: bool

    # 风险是否至少引用了一项证据。
    has_citation: bool

    # 引用验证状态是否为source_matched。
    citation_source_matched: bool


class EvaluationMetrics(SchemaModel):
    """单案例或整个评测集的风险识别指标。"""

    # 人工标注风险总数。
    expected_count: int = Field(ge=0)

    # Agent预测风险总数。
    predicted_count: int = Field(ge=0)

    # 正确匹配的风险数量。
    true_positive: int = Field(ge=0)

    # Agent额外预测的风险数量。
    false_positive: int = Field(ge=0)

    # Agent遗漏的人工标注风险数量。
    false_negative: int = Field(ge=0)

    # 风险识别精确率。
    precision: float = Field(ge=0.0, le=1.0)

    # 风险识别召回率。
    recall: float = Field(ge=0.0, le=1.0)

    # 精确率和召回率的调和平均值。
    f1: float = Field(ge=0.0, le=1.0)

    # 已匹配风险中的等级准确率。
    level_accuracy: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    # 已匹配风险中的原文定位准确率。
    location_accuracy: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    # 已匹配风险中至少包含一项证据的比例。
    citation_coverage: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    # 已匹配风险中通过来源验证的比例。
    source_matched_rate: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )


class EvaluationCaseResult(SchemaModel):
    """一份合同的预测结果和评测明细。"""

    # 对应评测数据中的案例ID。
    case_id: str = Field(min_length=1)

    # 案例名称。
    title: str = Field(min_length=1)

    # Agent产生的原始风险，供人工复核。
    predicted_findings: list[RiskFinding] = Field(
        default_factory=list
    )

    # 成功匹配的预期风险和预测风险。
    matches: list[FindingMatch] = Field(
        default_factory=list
    )

    # 未被Agent识别的预期风险匹配键。
    missing_expected: list[str] = Field(
        default_factory=list
    )

    # 没有对应人工标签的预测风险匹配键。
    unexpected_predictions: list[str] = Field(
        default_factory=list
    )

    # 当前案例的量化指标。
    metrics: EvaluationMetrics


class EvaluationReport(SchemaModel):
    """一次完整离线评测的可序列化报告。"""

    # 被评测系统名称，例如deterministic_rules或base_llm。
    system_name: str = Field(min_length=1)

    # 记录模型供应商、模型名、提示词版本等实验条件。
    # 字段只保存可公开的追溯信息，不能写入API密钥。
    system_metadata: dict[str, str] = Field(
        default_factory=dict
    )

    # 评测集名称和版本。
    dataset_name: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)

    # 报告生成时间，统一使用UTC时间。
    generated_at: datetime

    # 每个案例的详细结果。
    case_results: list[EvaluationCaseResult] = Field(
        default_factory=list
    )

    # 使用全部案例累计计数计算的总体指标。
    aggregate_metrics: EvaluationMetrics