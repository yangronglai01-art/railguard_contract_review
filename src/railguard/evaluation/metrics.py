"""合同审核离线评测的匹配和指标计算。"""

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

from railguard.evaluation.models import (
    EvaluationCase,
    EvaluationCaseResult,
    EvaluationDataset,
    EvaluationMetrics,
    EvaluationReport,
    ExpectedFinding,
    FindingMatch,
)
from railguard.models.schemas import (
    ContractDocument,
    RiskFinding,
)

# 指标版本用于区分缺失条款定位口径调整前后的报告。
EVALUATION_METRICS_VERSION = "evaluation-metrics-v2"


def normalize_text(text: str) -> str:
    """统一大小写并移除空白，降低格式差异对匹配的影响。"""
    return "".join(text.casefold().split())


def finding_match_key(finding: RiskFinding) -> str:
    """返回Agent预测风险使用的首版分类匹配键。"""
    return (
        f"{finding.finding_kind}:"
        f"{finding.category}"
    )


def location_matches(
    *,
    expected: ExpectedFinding,
    predicted: RiskFinding,
    clause_by_id: dict[str, str],
) -> bool | None:
    """检查已有条款风险是否定位到人工标注的原文位置。

    缺失条款在合同原文中没有可定位位置，因此返回None，
    避免把建议条款的同义措辞误算成原文定位错误。
    """
    if expected.finding_kind == "clause_risk":
        clause_id = predicted.clause_id

        if clause_id is None:
            return False

        clause_text = clause_by_id.get(clause_id)

        if clause_text is None:
            return False

        expected_text = expected.source_text_contains or ""

        return (
            normalize_text(expected_text)
            in normalize_text(clause_text)
        )

    return None


def divide_or_zero(
    numerator: int,
    denominator: int,
) -> float:
    """执行比例计算，并在分母为零时返回零。"""
    if denominator == 0:
        return 0.0

    return round(numerator / denominator, 4)


def optional_rate(
    successful_count: int,
    total_count: int,
) -> float | None:
    """计算可选比例，没有可评估样本时返回None。"""
    if total_count == 0:
        return None

    return round(
        successful_count / total_count,
        4,
    )


def calculate_f1(
    *,
    precision: float,
    recall: float,
) -> float:
    """计算精确率和召回率的调和平均值。"""
    if precision + recall == 0:
        return 0.0

    return round(
        2 * precision * recall
        / (precision + recall),
        4,
    )


def build_metrics(
    *,
    expected_count: int,
    predicted_count: int,
    matches: Sequence[FindingMatch],
) -> EvaluationMetrics:
    """根据一对一匹配结果计算风险识别和质量指标。"""
    true_positive = len(matches)
    false_positive = predicted_count - true_positive
    false_negative = expected_count - true_positive

    precision = divide_or_zero(
        true_positive,
        predicted_count,
    )
    recall = divide_or_zero(
        true_positive,
        expected_count,
    )

    return EvaluationMetrics(
        expected_count=expected_count,
        predicted_count=predicted_count,
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        precision=precision,
        recall=recall,
        f1=calculate_f1(
            precision=precision,
            recall=recall,
        ),
        level_accuracy=optional_rate(
            sum(
                match.level_matched
                for match in matches
            ),
            true_positive,
        ),
        location_accuracy=optional_rate(
            sum(
                match.location_matched is True
                for match in matches
            ),
            sum(
                match.location_matched is not None
                for match in matches
            ),
        ),
        citation_coverage=optional_rate(
            sum(
                match.has_citation
                for match in matches
            ),
            true_positive,
        ),
        source_matched_rate=optional_rate(
            sum(
                match.citation_source_matched
                for match in matches
            ),
            true_positive,
        ),
    )


def select_prediction(
    *,
    expected: ExpectedFinding,
    predictions: Sequence[RiskFinding],
    available_indices: set[int],
    clause_by_id: dict[str, str],
) -> int | None:
    """为一项人工标签选择最合适且尚未使用的预测。"""
    expected_key = expected.match_key()
    candidates = [
        index
        for index in available_indices
        if finding_match_key(predictions[index])
        == expected_key
    ]

    if not candidates:
        return None

    # 同一类别出现多个预测时，优先选择定位和等级都正确的结果。
    return max(
        candidates,
        key=lambda index: (
            location_matches(
                expected=expected,
                predicted=predictions[index],
                clause_by_id=clause_by_id,
            )
            is True,
            predictions[index].level
            == expected.level,
            -index,
        ),
    )


def evaluate_case(
    *,
    evaluation_case: EvaluationCase,
    contract: ContractDocument,
    predicted_findings: Sequence[RiskFinding],
) -> EvaluationCaseResult:
    """对一份合同执行一对一风险匹配并生成案例结果。"""
    if contract.full_text != evaluation_case.contract_text:
        raise ValueError(
            "evaluation contract text does not match the case"
        )

    predictions = list(predicted_findings)
    clause_by_id = {
        clause.clause_id: clause.text
        for clause in contract.clauses
    }
    available_indices = set(range(len(predictions)))
    matches: list[FindingMatch] = []
    missing_expected: list[str] = []

    for expected in evaluation_case.expected_findings:
        selected_index = select_prediction(
            expected=expected,
            predictions=predictions,
            available_indices=available_indices,
            clause_by_id=clause_by_id,
        )

        if selected_index is None:
            missing_expected.append(
                expected.match_key()
            )
            continue

        available_indices.remove(selected_index)
        predicted = predictions[selected_index]

        matches.append(
            FindingMatch(
                match_key=expected.match_key(),
                finding_id=predicted.finding_id,
                expected_level=expected.level,
                predicted_level=predicted.level,
                level_matched=(
                    expected.level
                    == predicted.level
                ),
                location_matched=location_matches(
                    expected=expected,
                    predicted=predicted,
                    clause_by_id=clause_by_id,
                ),
                has_citation=bool(
                    predicted.evidence_ids
                ),
                citation_source_matched=(
                    predicted.citation_status
                    == "source_matched"
                ),
            )
        )

    unexpected_predictions = [
        (
            f"{finding_match_key(predictions[index])}"
            f"#{predictions[index].finding_id}"
        )
        for index in sorted(available_indices)
    ]

    return EvaluationCaseResult(
        case_id=evaluation_case.case_id,
        title=evaluation_case.title,
        predicted_findings=predictions,
        matches=matches,
        missing_expected=missing_expected,
        unexpected_predictions=unexpected_predictions,
        metrics=build_metrics(
            expected_count=len(
                evaluation_case.expected_findings
            ),
            predicted_count=len(predictions),
            matches=matches,
        ),
    )


def aggregate_case_results(
    case_results: Sequence[EvaluationCaseResult],
) -> EvaluationMetrics:
    """使用全部案例累计计数计算总体微平均指标。"""
    matches = [
        match
        for case_result in case_results
        for match in case_result.matches
    ]

    return build_metrics(
        expected_count=sum(
            case_result.metrics.expected_count
            for case_result in case_results
        ),
        predicted_count=sum(
            case_result.metrics.predicted_count
            for case_result in case_results
        ),
        matches=matches,
    )


def build_evaluation_report(
    *,
    system_name: str,
    dataset: EvaluationDataset,
    case_results: Sequence[EvaluationCaseResult],
    system_metadata: Mapping[str, str] | None = None,
) -> EvaluationReport:
    """创建包含实验条件、逐案例结果和总体指标的评测报告。"""
    resolved_results = list(case_results)

    return EvaluationReport(
        system_name=system_name,
        system_metadata=dict(system_metadata or {}),
        dataset_name=dataset.dataset_name,
        dataset_version=dataset.version,
        generated_at=datetime.now(UTC),
        case_results=resolved_results,
        aggregate_metrics=aggregate_case_results(
            resolved_results
        ),
    )
