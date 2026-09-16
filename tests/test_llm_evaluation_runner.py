"""基础模型和微调模型共用离线评测流程的测试。"""

from collections.abc import (
    Mapping,
    Sequence,
)
from pathlib import Path

import pytest

from railguard.evaluation.models import EvaluationReport
from railguard.evaluation.runner import (
    load_evaluation_dataset,
    run_llm_evaluation,
    run_rules_evaluation,
)
from railguard.models.schemas import (
    ContractDocument,
    Evidence,
    RiskFinding,
)


class RecordingAnalyzer:
    """记录收到的合同并返回空风险列表的测试Agent。"""

    def __init__(
        self,
        name: str,
    ) -> None:
        """保存Agent名称并初始化合同调用记录。"""
        self.name = name
        self.received_contract_ids: list[str] = []

    async def analyze(
        self,
        *,
        contract: ContractDocument,
        evidence_by_clause: Mapping[
            str,
            Sequence[Evidence],
        ],
        contract_evidence: Sequence[Evidence],
    ) -> list[RiskFinding]:
        """记录合同ID并验证评测流程已经完成RAG检索。"""
        self.received_contract_ids.append(
            contract.contract_id
        )

        # 每个真实条款都应该具有独立的证据范围，
        # 即使某个范围最终没有检索到证据。
        assert set(evidence_by_clause) == {
            clause.clause_id
            for clause in contract.clauses
        }
        assert isinstance(contract_evidence, Sequence)

        return []


def load_dataset():
    """读取项目固定的11例评测数据集。"""
    return load_evaluation_dataset(
        Path(
            "data/evaluation/"
            "contract-review-v1.json"
        )
    )


def create_recording_analyzers() -> tuple[
    RecordingAnalyzer,
    RecordingAnalyzer,
    RecordingAnalyzer,
]:
    """创建三个名称与LangGraph节点一致的记录Agent。"""
    return (
        RecordingAnalyzer(
            "commercial_risk_agent"
        ),
        RecordingAnalyzer(
            "legal_risk_agent"
        ),
        RecordingAnalyzer(
            "security_risk_agent"
        ),
    )


async def test_llm_evaluation_reuses_all_dataset_cases() -> None:
    """验证三个模型Agent都处理同一批完整评测案例。"""
    dataset = load_dataset()
    commercial, legal, security = (
        create_recording_analyzers()
    )

    report = await run_llm_evaluation(
        dataset=dataset,
        rag_corpus_path=Path(
            "data/demo/rag-corpus.json"
        ),
        commercial_analyzer=commercial,
        legal_analyzer=legal,
        security_analyzer=security,
        system_name="base_llm",
        model_provider="openai",
        model_name="contract-base-model",
        prompt_version="contract-risk-v1",
    )

    expected_contract_ids = [
        f"evaluation-{evaluation_case.case_id}"
        for evaluation_case in dataset.cases
    ]

    assert len(dataset.cases) == 21
    assert commercial.received_contract_ids == (
        expected_contract_ids
    )
    assert legal.received_contract_ids == (
        expected_contract_ids
    )
    assert security.received_contract_ids == (
        expected_contract_ids
    )

    assert report.system_name == "base_llm"
    assert report.system_metadata == {
        "model_provider": "openai",
        "model_name": "contract-base-model",
        "prompt_version": "contract-risk-v1",
        "validator_version": "citation-validator-v2",
        "metrics_version": "evaluation-metrics-v2",
    }
    assert len(report.case_results) == 21
    assert report.aggregate_metrics.expected_count == 26
    assert report.aggregate_metrics.predicted_count == 0
    assert report.aggregate_metrics.false_negative == 26


async def test_llm_report_metadata_is_safe_and_round_trippable() -> None:
    """验证实验元数据可恢复且不会包含连接密钥字段。"""
    dataset = load_dataset()
    commercial, legal, security = (
        create_recording_analyzers()
    )
    report = await run_llm_evaluation(
        dataset=dataset,
        rag_corpus_path=Path(
            "data/demo/rag-corpus.json"
        ),
        commercial_analyzer=commercial,
        legal_analyzer=legal,
        security_analyzer=security,
        system_name="sft_llm",
        model_provider="openai",
        model_name="ft:contract-review-model",
        prompt_version="contract-risk-v1",
    )

    serialized = report.model_dump_json()
    restored = EvaluationReport.model_validate_json(
        serialized
    )

    assert restored.system_metadata == (
        report.system_metadata
    )
    assert "api_key" not in serialized
    assert "openai_api_key" not in serialized
    assert "base_url" not in serialized


@pytest.mark.parametrize(
    (
        "model_provider",
        "model_name",
        "prompt_version",
    ),
    [
        (
            "   ",
            "contract-base-model",
            "contract-risk-v1",
        ),
        (
            "openai",
            "   ",
            "contract-risk-v1",
        ),
        (
            "openai",
            "contract-base-model",
            "   ",
        ),
    ],
)
async def test_llm_evaluation_rejects_blank_metadata(
    model_provider: str,
    model_name: str,
    prompt_version: str,
) -> None:
    """验证缺失实验条件时不会启动任何模型评测。"""
    dataset = load_dataset()
    commercial, legal, security = (
        create_recording_analyzers()
    )

    with pytest.raises(
        ValueError,
        match="metadata must not be blank",
    ):
        await run_llm_evaluation(
            dataset=dataset,
            rag_corpus_path=Path(
                "data/demo/rag-corpus.json"
            ),
            commercial_analyzer=commercial,
            legal_analyzer=legal,
            security_analyzer=security,
            system_name="base_llm",
            model_provider=model_provider,
            model_name=model_name,
            prompt_version=prompt_version,
        )

    assert commercial.received_contract_ids == []
    assert legal.received_contract_ids == []
    assert security.received_contract_ids == []


async def test_rules_baseline_records_reproducible_metadata() -> None:
    """验证阶段A基线指标和实验条件在重构后保持不变。"""
    dataset = load_dataset()
    report = await run_rules_evaluation(
        dataset=dataset,
        rag_corpus_path=Path(
            "data/demo/rag-corpus.json"
        ),
    )
    metrics = report.aggregate_metrics

    assert report.system_name == "deterministic_rules"
    assert report.system_metadata == {
        "model_provider": "mock",
        "model_name": "deterministic_rules",
        "prompt_version": "not_applicable",
        "validator_version": "citation-validator-v2",
        "metrics_version": "evaluation-metrics-v2",
    }
    assert metrics.true_positive == 13
    assert metrics.false_positive == 2
    assert metrics.false_negative == 13
    assert metrics.precision == 0.8667
    assert metrics.recall == 0.5
