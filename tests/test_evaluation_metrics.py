"""合同审核离线评测指标测试。"""

import pytest

from railguard.evaluation.metrics import (
    aggregate_case_results,
    build_evaluation_report,
    evaluate_case,
)
from railguard.evaluation.models import (
    EvaluationCase,
    EvaluationDataset,
    ExpectedFinding,
)
from railguard.models.schemas import (
    ContractDocument,
    RiskFinding,
)
from railguard.parsers.clauses import split_clauses


def create_contract() -> ContractDocument:
    """创建带有付款风险且缺少数据安全条款的合同。"""
    full_text = (
        "设备监测平台软件采购合同\n\n"
        "第一条 项目内容\n\n"
        "乙方为甲方建设设备监测平台。\n\n"
        "第二条 付款方式\n\n"
        "合同签订后五日内，甲方支付全部合同款。"
    )

    return ContractDocument(
        contract_id="evaluation-contract",
        filename="evaluation-contract.txt",
        full_text=full_text,
        clauses=split_clauses(full_text),
    )


def create_case() -> EvaluationCase:
    """创建包含付款和数据安全预期风险的案例。"""
    contract = create_contract()

    return EvaluationCase(
        case_id="payment-and-security",
        title="付款与数据安全风险",
        filename=contract.filename,
        contract_text=contract.full_text,
        expected_findings=[
            ExpectedFinding(
                category="payment",
                finding_kind="clause_risk",
                level="high",
                source_text_contains="支付全部合同款",
            ),
            ExpectedFinding(
                category="data_security",
                finding_kind="missing_clause",
                level="high",
                expected_clause_contains="数据安全",
            ),
        ],
        tags=["mixed", "positive"],
    )


def payment_clause_id(contract: ContractDocument) -> str:
    """返回演示合同中付款条款的ID。"""
    return next(
        clause.clause_id
        for clause in contract.clauses
        if "支付全部合同款" in clause.text
    )


def project_clause_id(contract: ContractDocument) -> str:
    """返回演示合同中项目内容条款的ID。"""
    return next(
        clause.clause_id
        for clause in contract.clauses
        if "建设设备监测平台" in clause.text
    )


def create_payment_finding(
    contract: ContractDocument,
    *,
    clause_id: str | None = None,
    level: str = "high",
) -> RiskFinding:
    """创建付款风险预测。"""
    resolved_clause_id = (
        clause_id
        if clause_id is not None
        else payment_clause_id(contract)
    )

    return RiskFinding(
        finding_id="payment-finding",
        finding_kind="clause_risk",
        clause_id=resolved_clause_id,
        category="payment",
        level=level,
        reason="合同要求在交付验收前支付全部价款。",
        suggested_revision="建议设置分期付款并保留尾款。",
        evidence_ids=["payment-evidence"],
        citation_status="source_matched",
    )


def create_security_finding() -> RiskFinding:
    """创建缺失数据安全条款的风险预测。"""
    return RiskFinding(
        finding_id="security-finding",
        finding_kind="missing_clause",
        expected_clause="数据安全和保密义务条款",
        category="data_security",
        level="high",
        reason="合同没有约定供应商的数据安全义务。",
        suggested_revision="建议补充数据安全条款。",
        evidence_ids=["security-evidence"],
        citation_status="source_matched",
    )


def test_exact_predictions_receive_full_scores() -> None:
    """验证类别、等级、位置和引用正确时获得满分。"""
    contract = create_contract()
    evaluation_case = create_case()

    result = evaluate_case(
        evaluation_case=evaluation_case,
        contract=contract,
        predicted_findings=[
            create_payment_finding(contract),
            create_security_finding(),
        ],
    )

    assert result.metrics.expected_count == 2
    assert result.metrics.predicted_count == 2
    assert result.metrics.true_positive == 2
    assert result.metrics.false_positive == 0
    assert result.metrics.false_negative == 0
    assert result.metrics.precision == 1.0
    assert result.metrics.recall == 1.0
    assert result.metrics.f1 == 1.0
    assert result.metrics.level_accuracy == 1.0
    assert result.metrics.location_accuracy == 1.0
    assert result.metrics.citation_coverage == 1.0
    assert result.metrics.source_matched_rate == 1.0


def test_level_and_location_are_scored_separately() -> None:
    """验证分类命中后仍会单独检查风险等级和原文位置。"""
    contract = create_contract()
    wrong_payment = create_payment_finding(
        contract,
        clause_id=project_clause_id(contract),
        level="medium",
    )
    wrong_payment.evidence_ids = []
    wrong_payment.citation_status = "unsupported"

    result = evaluate_case(
        evaluation_case=create_case(),
        contract=contract,
        predicted_findings=[wrong_payment],
    )

    assert result.metrics.true_positive == 1
    assert result.metrics.false_positive == 0
    assert result.metrics.false_negative == 1
    assert result.metrics.precision == 1.0
    assert result.metrics.recall == 0.5
    assert result.metrics.f1 == 0.6667
    assert result.metrics.level_accuracy == 0.0
    assert result.metrics.location_accuracy == 0.0
    assert result.metrics.citation_coverage == 0.0
    assert result.metrics.source_matched_rate == 0.0
    assert result.missing_expected == [
        "missing_clause:data_security"
    ]


def test_duplicate_prediction_is_counted_as_false_positive() -> None:
    """验证两个同类预测只能与一项人工标签匹配一次。"""
    contract = create_contract()
    payment_only_case = EvaluationCase(
        case_id="payment-only",
        title="单一付款风险",
        filename=contract.filename,
        contract_text=contract.full_text,
        expected_findings=[
            ExpectedFinding(
                category="payment",
                finding_kind="clause_risk",
                level="high",
                source_text_contains="支付全部合同款",
            )
        ],
    )
    duplicate = create_payment_finding(contract)
    duplicate.finding_id = "duplicate-payment-finding"

    result = evaluate_case(
        evaluation_case=payment_only_case,
        contract=contract,
        predicted_findings=[
            create_payment_finding(contract),
            duplicate,
        ],
    )

    assert result.metrics.true_positive == 1
    assert result.metrics.false_positive == 1
    assert result.metrics.false_negative == 0
    assert result.metrics.precision == 0.5
    assert result.metrics.recall == 1.0
    assert result.metrics.f1 == 0.6667
    assert len(result.unexpected_predictions) == 1


def test_unexpected_risk_is_counted_on_safe_case() -> None:
    """验证无风险合同中的预测会被统计为误报。"""
    contract = create_contract()
    safe_case = EvaluationCase(
        case_id="safe-case",
        title="人工标注无风险",
        filename=contract.filename,
        contract_text=contract.full_text,
        expected_findings=[],
    )

    result = evaluate_case(
        evaluation_case=safe_case,
        contract=contract,
        predicted_findings=[create_security_finding()],
    )

    assert result.metrics.expected_count == 0
    assert result.metrics.predicted_count == 1
    assert result.metrics.true_positive == 0
    assert result.metrics.false_positive == 1
    assert result.metrics.false_negative == 0
    assert result.metrics.precision == 0.0
    assert result.metrics.recall == 0.0
    assert result.metrics.f1 == 0.0


def test_aggregate_metrics_use_micro_average_counts() -> None:
    """验证总体指标使用全部案例累计计数计算。"""
    contract = create_contract()
    exact_result = evaluate_case(
        evaluation_case=create_case(),
        contract=contract,
        predicted_findings=[
            create_payment_finding(contract),
            create_security_finding(),
        ],
    )
    safe_case = EvaluationCase(
        case_id="safe-case",
        title="人工标注无风险",
        filename=contract.filename,
        contract_text=contract.full_text,
        expected_findings=[],
    )
    false_positive_result = evaluate_case(
        evaluation_case=safe_case,
        contract=contract,
        predicted_findings=[create_security_finding()],
    )

    metrics = aggregate_case_results(
        [
            exact_result,
            false_positive_result,
        ]
    )

    assert metrics.expected_count == 2
    assert metrics.predicted_count == 3
    assert metrics.true_positive == 2
    assert metrics.false_positive == 1
    assert metrics.false_negative == 0
    assert metrics.precision == 0.6667
    assert metrics.recall == 1.0
    assert metrics.f1 == 0.8


def test_report_uses_dataset_identity_and_case_results() -> None:
    """验证评测报告保存系统、数据集和总体结果。"""
    contract = create_contract()
    evaluation_case = create_case()
    dataset = EvaluationDataset(
        dataset_name="railguard-evaluation",
        version="1.0.0",
        description="合同审核离线评测集。",
        cases=[evaluation_case],
    )
    case_result = evaluate_case(
        evaluation_case=evaluation_case,
        contract=contract,
        predicted_findings=[
            create_payment_finding(contract),
            create_security_finding(),
        ],
    )

    report = build_evaluation_report(
        system_name="deterministic_rules",
        dataset=dataset,
        case_results=[case_result],
    )

    assert report.dataset_name == "railguard-evaluation"
    assert report.dataset_version == "1.0.0"
    assert report.aggregate_metrics.f1 == 1.0
    assert len(report.case_results) == 1


def test_case_rejects_different_contract_text() -> None:
    """验证评测时不能把预测结果关联到其他合同。"""
    contract = create_contract()
    evaluation_case = create_case()
    changed_contract = contract.model_copy(
        update={"full_text": "另一份合同文本"}
    )

    with pytest.raises(
        ValueError,
        match="does not match",
    ):
        evaluate_case(
            evaluation_case=evaluation_case,
            contract=changed_contract,
            predicted_findings=[],
        )