"""离线评测案例筛选、增量保存和断点恢复测试。"""

from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from railguard.evaluation.models import EvaluationReport
from railguard.evaluation.runner import (
    load_evaluation_dataset,
    run_llm_evaluation,
    select_evaluation_cases,
)
from railguard.models.schemas import ContractDocument, Evidence, RiskFinding


class CountingAnalyzer:
    """记录评测调用次数并返回空风险的测试Agent。"""

    def __init__(self, name: str) -> None:
        """保存节点名称并初始化调用记录。"""
        self.name = name
        self.case_ids: list[str] = []

    async def analyze(
        self,
        *,
        contract: ContractDocument,
        evidence_by_clause: Mapping[str, Sequence[Evidence]],
        contract_evidence: Sequence[Evidence],
    ) -> list[RiskFinding]:
        """记录合同ID，验证依赖已经准备完成。"""
        del evidence_by_clause
        del contract_evidence
        self.case_ids.append(contract.contract_id)
        return []


def dataset():
    """读取固定21例评测集。"""
    return load_evaluation_dataset(
        Path("data/evaluation/contract-review-v1.json")
    )


def analyzers() -> tuple[CountingAnalyzer, CountingAnalyzer, CountingAnalyzer]:
    """创建三个可计数的专业Agent替身。"""
    return (
        CountingAnalyzer("commercial_risk_agent"),
        CountingAnalyzer("legal_risk_agent"),
        CountingAnalyzer("security_risk_agent"),
    )


async def run_cases(
    case_ids: list[str],
    *,
    existing_report: EvaluationReport | None = None,
    reports: list[EvaluationReport] | None = None,
) -> tuple[EvaluationReport, tuple[CountingAnalyzer, ...]]:
    """使用固定实验条件运行指定案例。"""
    current = analyzers()
    report = await run_llm_evaluation(
        dataset=dataset(),
        rag_corpus_path=Path("data/demo/rag-corpus.json"),
        commercial_analyzer=current[0],
        legal_analyzer=current[1],
        security_analyzer=current[2],
        system_name="resume-test",
        model_provider="deepseek",
        model_name="deepseek-test",
        prompt_version="contract-risk-v2",
        case_ids=case_ids,
        existing_report=existing_report,
        report_callback=(reports.append if reports is not None else None),
    )
    return report, current


def test_case_filter_rejects_unknown_and_duplicate_ids() -> None:
    """验证案例筛选不会静默忽略拼写错误或重复ID。"""
    loaded = dataset()
    with pytest.raises(ValueError, match="unknown evaluation case_id"):
        select_evaluation_cases(loaded, ["not-a-case"])
    with pytest.raises(ValueError, match="must not contain duplicates"):
        select_evaluation_cases(
            loaded,
            ["demo-six-risks", "demo-six-risks"],
        )


async def test_evaluation_saves_after_each_case_and_resumes() -> None:
    """验证恢复时只调用尚未完成的案例。"""
    saved_reports: list[EvaluationReport] = []
    first, first_analyzers = await run_cases(
        ["demo-six-risks", "complete-safe-contract"],
        reports=saved_reports,
    )
    assert [len(report.case_results) for report in saved_reports] == [1, 2]
    assert all(len(agent.case_ids) == 2 for agent in first_analyzers)

    resumed, resumed_analyzers = await run_cases(
        [
            "demo-six-risks",
            "complete-safe-contract",
            "literal-payment-risk",
        ],
        existing_report=first,
    )
    assert len(resumed.case_results) == 3
    assert all(len(agent.case_ids) == 1 for agent in resumed_analyzers)
    assert all(
        agent.case_ids == ["evaluation-literal-payment-risk"]
        for agent in resumed_analyzers
    )


async def test_resume_rejects_different_experiment_metadata() -> None:
    """验证不能把其它模型或提示词的报告混入当前实验。"""
    report, _ = await run_cases(["demo-six-risks"])
    report.system_metadata["model_name"] = "another-model"
    with pytest.raises(ValueError, match="metadata does not match"):
        await run_cases(
            ["demo-six-risks", "complete-safe-contract"],
            existing_report=report,
        )
