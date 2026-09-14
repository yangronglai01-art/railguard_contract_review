"""合同审核离线评测执行器。"""

from collections.abc import Mapping
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver

from railguard.evaluation.metrics import (
    build_evaluation_report,
    evaluate_case,
)
from railguard.evaluation.models import (
    EvaluationCase,
    EvaluationCaseResult,
    EvaluationDataset,
    EvaluationReport,
)
from railguard.models.schemas import (
    Clause,
    ContractDocument,
)
from railguard.parsers.clauses import split_clauses
from railguard.rag.mock import MockRagRetriever
from railguard.workflow.analyzers import (
    DemoCommercialRiskAnalyzer,
    DemoLegalRiskAnalyzer,
    DemoSecurityRiskAnalyzer,
    RiskAnalyzer,
)
from railguard.workflow.checkpoint import (
    create_checkpoint_serializer,
)
from railguard.workflow.graph import build_review_graph
from railguard.workflow.service import ReviewService


def load_evaluation_dataset(
    path: Path,
) -> EvaluationDataset:
    """从UTF-8 JSON文件读取并校验评测数据集。"""
    content = path.read_text(encoding="utf-8-sig")

    return EvaluationDataset.model_validate_json(
        content
    )


def build_evaluation_contract(
    evaluation_case: EvaluationCase,
) -> ContractDocument:
    """为评测案例创建使用稳定合同和条款ID的业务模型。"""
    parsed_clauses = split_clauses(
        evaluation_case.contract_text
    )
    clauses = [
        Clause(
            clause_id=(
                f"{evaluation_case.case_id}-"
                f"clause-{index:03d}"
            ),
            title=clause.title,
            text=clause.text,
            start_offset=clause.start_offset,
            end_offset=clause.end_offset,
        )
        for index, clause in enumerate(
            parsed_clauses,
            start=1,
        )
    ]

    return ContractDocument(
        contract_id=(
            f"evaluation-{evaluation_case.case_id}"
        ),
        filename=evaluation_case.filename,
        full_text=evaluation_case.contract_text,
        clauses=clauses,
    )


async def run_analyzer_evaluation(
    *,
    dataset: EvaluationDataset,
    rag_corpus_path: Path,
    commercial_analyzer: RiskAnalyzer,
    legal_analyzer: RiskAnalyzer,
    security_analyzer: RiskAnalyzer,
    system_name: str,
    system_metadata: Mapping[str, str] | None = None,
) -> EvaluationReport:
    """使用指定的三个Agent运行统一离线评测流程。

    规则模型、基础模型和微调模型都通过此函数执行，
    保证它们使用相同数据集、RAG、LangGraph节点和评分方法。
    """
    retriever = MockRagRetriever.from_json_file(
        rag_corpus_path
    )
    graph = build_review_graph(
        retriever=retriever,
        commercial_analyzer=commercial_analyzer,
        legal_analyzer=legal_analyzer,
        security_analyzer=security_analyzer,
        checkpointer=InMemorySaver(
            serde=create_checkpoint_serializer()
        ),
    )
    service = ReviewService(graph)
    case_results: list[EvaluationCaseResult] = []

    # 案例保持数据集中的固定顺序，便于比较不同实验报告。
    for evaluation_case in dataset.cases:
        contract = build_evaluation_contract(
            evaluation_case
        )
        snapshot = await service.start_review(
            contract
        )

        if snapshot.status not in {
            "awaiting_human",
            "approved",
        }:
            raise RuntimeError(
                "evaluation workflow did not reach "
                "a scorable state"
            )

        case_results.append(
            evaluate_case(
                evaluation_case=evaluation_case,
                contract=contract,
                predicted_findings=(
                    snapshot.findings
                ),
            )
        )

    return build_evaluation_report(
        system_name=system_name,
        system_metadata=system_metadata,
        dataset=dataset,
        case_results=case_results,
    )


async def run_rules_evaluation(
    *,
    dataset: EvaluationDataset,
    rag_corpus_path: Path,
) -> EvaluationReport:
    """运行确定性规则Agent并生成阶段A基线报告。"""
    return await run_analyzer_evaluation(
        dataset=dataset,
        rag_corpus_path=rag_corpus_path,
        commercial_analyzer=(
            DemoCommercialRiskAnalyzer()
        ),
        legal_analyzer=DemoLegalRiskAnalyzer(),
        security_analyzer=DemoSecurityRiskAnalyzer(),
        system_name="deterministic_rules",
        system_metadata={
            "model_provider": "mock",
            "model_name": "deterministic_rules",
            "prompt_version": "not_applicable",
        },
    )


async def run_llm_evaluation(
    *,
    dataset: EvaluationDataset,
    rag_corpus_path: Path,
    commercial_analyzer: RiskAnalyzer,
    legal_analyzer: RiskAnalyzer,
    security_analyzer: RiskAnalyzer,
    system_name: str,
    model_provider: str,
    model_name: str,
    prompt_version: str,
) -> EvaluationReport:
    """运行基础或微调大模型并生成可追溯评测报告。"""
    metadata = {
        "model_provider": model_provider.strip(),
        "model_name": model_name.strip(),
        "prompt_version": prompt_version.strip(),
    }

    if any(
        not value
        for value in metadata.values()
    ):
        raise ValueError(
            "LLM evaluation metadata must not be blank"
        )

    # 只写入公开实验条件，不保存API密钥和内部接口地址。
    return await run_analyzer_evaluation(
        dataset=dataset,
        rag_corpus_path=rag_corpus_path,
        commercial_analyzer=commercial_analyzer,
        legal_analyzer=legal_analyzer,
        security_analyzer=security_analyzer,
        system_name=system_name,
        system_metadata=metadata,
    )


def save_evaluation_report(
    *,
    report: EvaluationReport,
    path: Path,
) -> None:
    """使用临时文件原子写入UTF-8 JSON评测报告。"""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary_path = path.with_suffix(
        f"{path.suffix}.tmp"
    )
    temporary_path.write_text(
        report.model_dump_json(
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary_path.replace(path)