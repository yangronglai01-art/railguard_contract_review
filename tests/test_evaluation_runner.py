"""合同审核离线评测执行器测试。"""

from pathlib import Path

from railguard.evaluation.models import (
    EvaluationReport,
)
from railguard.evaluation.runner import (
    build_evaluation_contract,
    load_evaluation_dataset,
    run_rules_evaluation,
    save_evaluation_report,
)


def evaluation_dataset_path() -> Path:
    """返回仓库内第一版评测集路径。"""
    return Path(
        "data/evaluation/"
        "contract-review-v1.json"
    )


def mock_rag_corpus_path() -> Path:
    """返回仓库内演示RAG知识库路径。"""
    return Path("data/demo/rag-corpus.json")


def test_loads_versioned_evaluation_dataset() -> None:
    """验证第一版评测集包含预期案例和标签数量。"""
    dataset = load_evaluation_dataset(
        evaluation_dataset_path()
    )

    assert dataset.dataset_name == (
        "railguard-contract-review"
    )
    assert dataset.version == "1.1.0"
    assert len(dataset.cases) == 21
    assert (
        sum(
            len(evaluation_case.expected_findings)
            for evaluation_case in dataset.cases
        )
        == 26
    )


def test_builds_contract_with_stable_clause_ids() -> None:
    """验证同一案例每次构造出相同的合同和条款ID。"""
    dataset = load_evaluation_dataset(
        evaluation_dataset_path()
    )
    evaluation_case = dataset.cases[0]

    first_contract = build_evaluation_contract(
        evaluation_case
    )
    second_contract = build_evaluation_contract(
        evaluation_case
    )

    assert first_contract.contract_id == (
        "evaluation-demo-six-risks"
    )
    assert [
        clause.clause_id
        for clause in first_contract.clauses
    ] == [
        clause.clause_id
        for clause in second_contract.clauses
    ]
    assert all(
        clause.clause_id.startswith(
            "demo-six-risks-clause-"
        )
        for clause in first_contract.clauses
    )
    assert all(
        first_contract.full_text[
            clause.start_offset:clause.end_offset
        ]
        == clause.text
        for clause in first_contract.clauses
    )


async def test_rules_baseline_produces_expected_metrics(
    tmp_path: Path,
) -> None:
    """运行完整规则基线并验证已知指标和报告写入。"""
    dataset = load_evaluation_dataset(
        evaluation_dataset_path()
    )

    report = await run_rules_evaluation(
        dataset=dataset,
        rag_corpus_path=mock_rag_corpus_path(),
    )

    metrics = report.aggregate_metrics

    assert len(report.case_results) == 21
    assert metrics.expected_count == 26
    assert metrics.predicted_count == 15
    assert metrics.true_positive == 13
    assert metrics.false_positive == 2
    assert metrics.false_negative == 13
    assert metrics.precision == 0.8667
    assert metrics.recall == 0.5
    assert metrics.f1 == 0.6342
    assert metrics.level_accuracy == 1.0
    assert metrics.location_accuracy == 1.0
    assert metrics.citation_coverage == 1.0
    assert metrics.source_matched_rate == 1.0

    results_by_id = {
        case_result.case_id: case_result
        for case_result in report.case_results
    }

    assert (
        results_by_id[
            "demo-six-risks"
        ].metrics.f1
        == 1.0
    )
    assert (
        results_by_id[
            "negated-payment-safe"
        ].metrics.false_positive
        == 1
    )
    assert (
        results_by_id[
            "paraphrased-payment-risk"
        ].metrics.false_negative
        == 1
    )
    assert (
        results_by_id[
            "weak-data-security"
        ].metrics.false_negative
        == 1
    )

    report_path = tmp_path / "rules-report.json"

    save_evaluation_report(
        report=report,
        path=report_path,
    )

    restored_report = (
        EvaluationReport.model_validate_json(
            report_path.read_text(
                encoding="utf-8"
            )
        )
    )

    assert restored_report.aggregate_metrics == metrics
    assert not (
        tmp_path / "rules-report.json.tmp"
    ).exists()