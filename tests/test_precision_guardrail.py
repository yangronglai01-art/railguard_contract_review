"""大模型风险精确率护栏测试。"""

from pathlib import Path

from railguard.agents.guardrails import (
    calibrated_risk_level,
    risk_candidate_is_eligible,
)
from railguard.evaluation.optimization import (
    replay_precision_guardrail,
)
from railguard.evaluation.runner import (
    build_evaluation_contract,
    load_evaluation_dataset,
    load_evaluation_report,
)

DATASET_PATH = Path("data/evaluation/contract-review-v1.json")
SOURCE_REPORT_PATH = Path(
    "data/evaluation/reports/deepseek-v4-pro-v1.1-report.json"
)


def test_guardrail_matches_development_benchmark_risk_slots() -> None:
    """验证开发基准中的高置信度风险边界与人工标签一致。"""
    dataset = load_evaluation_dataset(DATASET_PATH)

    for evaluation_case in dataset.cases:
        contract = build_evaluation_contract(evaluation_case)
        expected_slots = {
            (
                expected.finding_kind,
                expected.category,
            )
            for expected in evaluation_case.expected_findings
        }
        eligible_slots = set()

        for category in (
            "payment",
            "acceptance",
            "intellectual_property",
            "liability",
            "support",
            "data_security",
        ):
            if risk_candidate_is_eligible(
                contract=contract,
                finding_kind="missing_clause",
                clause_id=None,
                category=category,
            ):
                eligible_slots.add(("missing_clause", category))

            for clause in contract.clauses:
                if risk_candidate_is_eligible(
                    contract=contract,
                    finding_kind="clause_risk",
                    clause_id=clause.clause_id,
                    category=category,
                ):
                    eligible_slots.add(("clause_risk", category))

        assert eligible_slots == expected_slots, evaluation_case.case_id


def test_guardrail_calibrates_support_levels() -> None:
    """验证运维缺失和响应时限缺口使用不同等级。"""
    dataset = load_evaluation_dataset(DATASET_PATH)
    cases = {case.case_id: case for case in dataset.cases}

    missing_contract = build_evaluation_contract(
        cases["missing-support-only"]
    )
    assert calibrated_risk_level(
        contract=missing_contract,
        finding_kind="missing_clause",
        clause_id=None,
        category="support",
    ) == "medium"

    low_contract = build_evaluation_contract(
        cases["low-support-gap"]
    )
    support_clause = next(
        clause
        for clause in low_contract.clauses
        if "运维服务" in clause.title
    )
    assert calibrated_risk_level(
        contract=low_contract,
        finding_kind="clause_risk",
        clause_id=support_clause.clause_id,
        category="support",
    ) == "low"


def test_historical_replay_meets_optimization_gate() -> None:
    """验证历史原始预测经护栏重放后达到开发阶段门槛。"""
    report = replay_precision_guardrail(
        dataset=load_evaluation_dataset(DATASET_PATH),
        source_report=load_evaluation_report(
            SOURCE_REPORT_PATH
        ),
    )
    metrics = report.aggregate_metrics

    assert metrics.precision >= 0.70
    assert metrics.recall >= 0.85
    assert metrics.f1 >= 0.75
    assert metrics.false_positive <= 15
