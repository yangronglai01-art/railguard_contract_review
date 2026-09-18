"""基础模型历史报告的精确率护栏离线重放工具。"""

from railguard.agents.guardrails import (
    PRECISION_GUARDRAIL_VERSION,
    calibrated_risk_level,
    risk_candidate_is_eligible,
)
from railguard.evaluation.metrics import (
    build_evaluation_report,
    evaluate_case,
)
from railguard.evaluation.models import (
    EvaluationDataset,
    EvaluationReport,
)
from railguard.evaluation.runner import build_evaluation_contract


def replay_precision_guardrail(
    *,
    dataset: EvaluationDataset,
    source_report: EvaluationReport,
) -> EvaluationReport:
    """在历史原始预测上重放本地护栏并重新计算评测指标。"""
    result_by_case_id = {
        result.case_id: result
        for result in source_report.case_results
    }
    case_results = []

    for evaluation_case in dataset.cases:
        source_result = result_by_case_id.get(
            evaluation_case.case_id
        )
        if source_result is None:
            continue

        contract = build_evaluation_contract(evaluation_case)
        retained_findings = []

        for finding in source_result.predicted_findings:
            if not risk_candidate_is_eligible(
                contract=contract,
                finding_kind=finding.finding_kind,
                clause_id=finding.clause_id,
                category=finding.category,
            ):
                continue

            retained_findings.append(
                finding.model_copy(
                    update={
                        "level": calibrated_risk_level(
                            contract=contract,
                            finding_kind=finding.finding_kind,
                            clause_id=finding.clause_id,
                            category=finding.category,
                        )
                    }
                )
            )

        case_results.append(
            evaluate_case(
                evaluation_case=evaluation_case,
                contract=contract,
                predicted_findings=retained_findings,
            )
        )

    return build_evaluation_report(
        system_name=(
            f"{source_report.system_name}_guardrail_replay"
        ),
        dataset=dataset,
        case_results=case_results,
        system_metadata={
            **source_report.system_metadata,
            "source_report_system": source_report.system_name,
            "precision_guardrail_version": (
                PRECISION_GUARDRAIL_VERSION
            ),
            "evaluation_mode": "offline_historical_replay",
        },
    )
