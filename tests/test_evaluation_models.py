"""合同审核离线评测数据模型测试。"""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from railguard.evaluation.models import (
    EvaluationCase,
    EvaluationCaseResult,
    EvaluationDataset,
    EvaluationMetrics,
    EvaluationReport,
    ExpectedFinding,
    FindingMatch,
)
from railguard.models.schemas import RiskFinding


def create_zero_metrics() -> EvaluationMetrics:
    """创建没有预期风险和预测风险时使用的指标。"""
    return EvaluationMetrics(
        expected_count=0,
        predicted_count=0,
        true_positive=0,
        false_positive=0,
        false_negative=0,
        precision=0.0,
        recall=0.0,
        f1=0.0,
        level_accuracy=None,
        location_accuracy=None,
        citation_coverage=None,
        source_matched_rate=None,
    )


def create_safe_case(case_id: str = "complete-contract") -> EvaluationCase:
    """创建没有预期风险的完整合同案例。"""
    return EvaluationCase(
        case_id=case_id,
        title="完整条款合同",
        filename="complete-contract.txt",
        contract_text=(
            "软件采购合同\n\n"
            "第一条 付款方式\n\n"
            "最终验收合格后支付尾款。\n\n"
            "第二条 数据安全\n\n"
            "乙方承担数据安全和保密义务。"
        ),
        expected_findings=[],
        tags=["negative", "complete"],
        notes="用于检查误报。",
    )


def test_dataset_accepts_clause_and_missing_findings() -> None:
    """验证评测集可以同时标注已有条款和缺失条款风险。"""
    evaluation_case = EvaluationCase(
        case_id="payment-and-security",
        title="付款与数据安全风险",
        filename="payment-risk.txt",
        contract_text=(
            "软件采购合同\n\n"
            "第一条 付款方式\n\n"
            "合同签订后支付全部合同款。"
        ),
        expected_findings=[
            ExpectedFinding(
                category="payment",
                finding_kind="clause_risk",
                level="high",
                source_text_contains="支付全部合同款",
                annotation_note="全额预付风险。",
            ),
            ExpectedFinding(
                category="data_security",
                finding_kind="missing_clause",
                level="high",
                expected_clause_contains="数据安全",
                annotation_note="合同没有数据安全约定。",
            ),
        ],
        tags=["positive", "mixed"],
    )

    dataset = EvaluationDataset(
        dataset_name="railguard-contract-review",
        version="1.0.0",
        description="采购方合同风险评测集。",
        cases=[evaluation_case],
    )

    assert len(dataset.cases) == 1
    assert {
        finding.match_key()
        for finding in dataset.cases[0].expected_findings
    } == {
        "clause_risk:payment",
        "missing_clause:data_security",
    }


def test_clause_risk_requires_source_text() -> None:
    """验证已有条款风险必须提供稳定原文标记。"""
    with pytest.raises(
        ValidationError,
        match="source_text_contains",
    ):
        ExpectedFinding(
            category="payment",
            finding_kind="clause_risk",
            level="high",
        )


def test_missing_clause_requires_expected_clause_text() -> None:
    """验证缺失条款风险必须说明期望条款。"""
    with pytest.raises(
        ValidationError,
        match="expected_clause_contains",
    ):
        ExpectedFinding(
            category="data_security",
            finding_kind="missing_clause",
            level="high",
        )


def test_location_fields_must_match_finding_kind() -> None:
    """验证两类风险不能使用彼此的定位字段。"""
    with pytest.raises(
        ValidationError,
        match="must not define source_text_contains",
    ):
        ExpectedFinding(
            category="data_security",
            finding_kind="missing_clause",
            level="high",
            source_text_contains="数据",
            expected_clause_contains="数据安全",
        )


def test_case_rejects_duplicate_matching_keys() -> None:
    """验证同一案例不能包含重复的首版风险匹配键。"""
    duplicated_findings = [
        ExpectedFinding(
            category="payment",
            finding_kind="clause_risk",
            level="high",
            source_text_contains="支付全部合同款",
        ),
        ExpectedFinding(
            category="payment",
            finding_kind="clause_risk",
            level="medium",
            source_text_contains="预付款",
        ),
    ]

    with pytest.raises(
        ValidationError,
        match="match keys must be unique",
    ):
        EvaluationCase(
            case_id="duplicate-payment",
            title="重复付款标注",
            filename="duplicate.txt",
            contract_text="支付全部合同款并支付预付款。",
            expected_findings=duplicated_findings,
        )


def test_dataset_rejects_duplicate_case_ids() -> None:
    """验证同一评测集不能重复使用案例ID。"""
    with pytest.raises(
        ValidationError,
        match="case_id values must be unique",
    ):
        EvaluationDataset(
            dataset_name="duplicate-dataset",
            version="1.0.0",
            description="包含重复案例ID的无效数据集。",
            cases=[
                create_safe_case("same-id"),
                create_safe_case("same-id"),
            ],
        )


def test_report_serializes_case_details_and_utc_time() -> None:
    """验证评测报告可以保存匹配明细、预测结果和时间。"""
    predicted = RiskFinding(
        finding_kind="missing_clause",
        expected_clause="数据安全和保密义务条款",
        category="data_security",
        level="high",
        reason="合同没有约定供应商的数据安全义务。",
        suggested_revision="建议补充数据安全条款。",
        evidence_ids=["evidence-001"],
        citation_status="source_matched",
    )
    metrics = EvaluationMetrics(
        expected_count=1,
        predicted_count=1,
        true_positive=1,
        false_positive=0,
        false_negative=0,
        precision=1.0,
        recall=1.0,
        f1=1.0,
        level_accuracy=1.0,
        location_accuracy=1.0,
        citation_coverage=1.0,
        source_matched_rate=1.0,
    )
    case_result = EvaluationCaseResult(
        case_id="missing-security",
        title="缺失数据安全条款",
        predicted_findings=[predicted],
        matches=[
            FindingMatch(
                match_key="missing_clause:data_security",
                finding_id=predicted.finding_id,
                expected_level="high",
                predicted_level="high",
                level_matched=True,
                location_matched=True,
                has_citation=True,
                citation_source_matched=True,
            )
        ],
        metrics=metrics,
    )
    report = EvaluationReport(
        system_name="deterministic_rules",
        dataset_name="railguard-contract-review",
        dataset_version="1.0.0",
        generated_at=datetime.now(UTC),
        case_results=[case_result],
        aggregate_metrics=metrics,
    )

    payload = report.model_dump(mode="json")

    assert payload["system_name"] == "deterministic_rules"
    assert payload["generated_at"].endswith("Z")
    assert payload["case_results"][0]["metrics"]["f1"] == 1.0


def test_metric_rates_must_remain_between_zero_and_one() -> None:
    """验证比例指标不能超出零到一的范围。"""
    with pytest.raises(ValidationError):
        EvaluationMetrics(
            expected_count=1,
            predicted_count=1,
            true_positive=1,
            false_positive=0,
            false_negative=0,
            precision=1.1,
            recall=1.0,
            f1=1.0,
        )

    assert create_zero_metrics().level_accuracy is None