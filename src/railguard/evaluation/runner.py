"""合同审核离线评测执行器。"""

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver

from railguard.evaluation.metrics import (
    build_evaluation_report,
    evaluate_case,
)
from railguard.evaluation.models import (
    EvaluationCase,
    EvaluationDataset,
    EvaluationReport,
)
from railguard.models.schemas import Clause, ContractDocument
from railguard.parsers.clauses import split_clauses
from railguard.rag.mock import MockRagRetriever
from railguard.workflow.analyzers import (
    DemoCommercialRiskAnalyzer,
    DemoLegalRiskAnalyzer,
    DemoSecurityRiskAnalyzer,
    RiskAnalyzer,
)
from railguard.workflow.checkpoint import create_checkpoint_serializer
from railguard.workflow.graph import build_review_graph
from railguard.workflow.nodes import CITATION_VALIDATOR_VERSION
from railguard.workflow.service import ReviewService

ReportCallback = Callable[[EvaluationReport], None]
CaseProgressCallback = Callable[[int, int, str], None]


def load_evaluation_dataset(path: Path) -> EvaluationDataset:
    """从UTF-8 JSON文件读取并校验评测数据集。"""
    content = path.read_text(encoding="utf-8-sig")
    return EvaluationDataset.model_validate_json(content)


def build_evaluation_contract(
    evaluation_case: EvaluationCase,
) -> ContractDocument:
    """为评测案例创建使用稳定合同和条款ID的业务模型。"""
    parsed_clauses = split_clauses(evaluation_case.contract_text)
    clauses = [
        Clause(
            clause_id=(
                f"{evaluation_case.case_id}-clause-{index:03d}"
            ),
            title=clause.title,
            text=clause.text,
            start_offset=clause.start_offset,
            end_offset=clause.end_offset,
        )
        for index, clause in enumerate(parsed_clauses, start=1)
    ]
    return ContractDocument(
        contract_id=f"evaluation-{evaluation_case.case_id}",
        filename=evaluation_case.filename,
        full_text=evaluation_case.contract_text,
        clauses=clauses,
    )


def select_evaluation_cases(
    dataset: EvaluationDataset,
    case_ids: Sequence[str] | None,
) -> list[EvaluationCase]:
    """按数据集顺序选择案例，并拒绝未知或重复的案例ID。"""
    if case_ids is None:
        return list(dataset.cases)

    requested = list(case_ids)
    if len(requested) != len(set(requested)):
        raise ValueError("case_id filters must not contain duplicates")

    known_ids = {case.case_id for case in dataset.cases}
    unknown_ids = set(requested) - known_ids
    if unknown_ids:
        unknown = ", ".join(sorted(unknown_ids))
        raise ValueError(f"unknown evaluation case_id: {unknown}")

    requested_ids = set(requested)
    return [
        case
        for case in dataset.cases
        if case.case_id in requested_ids
    ]


def validate_resume_report(
    *,
    report: EvaluationReport,
    dataset: EvaluationDataset,
    system_name: str,
    system_metadata: Mapping[str, str],
) -> None:
    """防止把不同实验条件的旧结果混入本次恢复任务。"""
    if (
        report.dataset_name != dataset.dataset_name
        or report.dataset_version != dataset.version
    ):
        raise ValueError("resume report dataset does not match")
    if report.system_name != system_name:
        raise ValueError("resume report system_name does not match")
    if report.system_metadata != dict(system_metadata):
        raise ValueError("resume report metadata does not match")

    known_ids = {case.case_id for case in dataset.cases}
    report_ids = [result.case_id for result in report.case_results]
    if len(report_ids) != len(set(report_ids)):
        raise ValueError("resume report contains duplicate case results")
    if set(report_ids) - known_ids:
        raise ValueError("resume report contains unknown case results")


async def run_analyzer_evaluation(
    *,
    dataset: EvaluationDataset,
    rag_corpus_path: Path,
    commercial_analyzer: RiskAnalyzer,
    legal_analyzer: RiskAnalyzer,
    security_analyzer: RiskAnalyzer,
    system_name: str,
    system_metadata: Mapping[str, str] | None = None,
    case_ids: Sequence[str] | None = None,
    existing_report: EvaluationReport | None = None,
    report_callback: ReportCallback | None = None,
    progress_callback: CaseProgressCallback | None = None,
) -> EvaluationReport:
    """运行统一评测，并支持案例筛选、增量保存和断点恢复。"""
    resolved_metadata = dict(system_metadata or {})
    selected_cases = select_evaluation_cases(dataset, case_ids)

    if existing_report is not None:
        validate_resume_report(
            report=existing_report,
            dataset=dataset,
            system_name=system_name,
            system_metadata=resolved_metadata,
        )

    result_by_id = {
        result.case_id: result
        for result in (
            existing_report.case_results
            if existing_report is not None
            else []
        )
    }
    pending_cases = [
        case
        for case in selected_cases
        if case.case_id not in result_by_id
    ]

    retriever = MockRagRetriever.from_json_file(rag_corpus_path)
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

    for position, evaluation_case in enumerate(pending_cases, start=1):
        if progress_callback is not None:
            progress_callback(
                position,
                len(pending_cases),
                evaluation_case.case_id,
            )

        contract = build_evaluation_contract(evaluation_case)
        snapshot = await service.start_review(contract)
        if snapshot.status not in {"awaiting_human", "approved"}:
            raise RuntimeError(
                "evaluation workflow did not reach a scorable state"
            )

        result_by_id[evaluation_case.case_id] = evaluate_case(
            evaluation_case=evaluation_case,
            contract=contract,
            predicted_findings=snapshot.findings,
        )
        ordered_results = [
            result_by_id[case.case_id]
            for case in dataset.cases
            if case.case_id in result_by_id
        ]
        partial_report = build_evaluation_report(
            system_name=system_name,
            system_metadata=resolved_metadata,
            dataset=dataset,
            case_results=ordered_results,
        )
        if report_callback is not None:
            report_callback(partial_report)

    ordered_results = [
        result_by_id[case.case_id]
        for case in dataset.cases
        if case.case_id in result_by_id
    ]
    return build_evaluation_report(
        system_name=system_name,
        system_metadata=resolved_metadata,
        dataset=dataset,
        case_results=ordered_results,
    )


async def run_rules_evaluation(
    *,
    dataset: EvaluationDataset,
    rag_corpus_path: Path,
    case_ids: Sequence[str] | None = None,
    experiment_metadata: Mapping[str, str] | None = None,
) -> EvaluationReport:
    """运行确定性规则Agent并生成阶段A基线报告。"""
    return await run_analyzer_evaluation(
        dataset=dataset,
        rag_corpus_path=rag_corpus_path,
        commercial_analyzer=DemoCommercialRiskAnalyzer(),
        legal_analyzer=DemoLegalRiskAnalyzer(),
        security_analyzer=DemoSecurityRiskAnalyzer(),
        system_name="deterministic_rules",
        system_metadata={
            "model_provider": "mock",
            "model_name": "deterministic_rules",
            "prompt_version": "not_applicable",
            "validator_version": CITATION_VALIDATOR_VERSION,
            **dict(experiment_metadata or {}),
        },
        case_ids=case_ids,
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
    experiment_metadata: Mapping[str, str] | None = None,
    case_ids: Sequence[str] | None = None,
    existing_report: EvaluationReport | None = None,
    report_callback: ReportCallback | None = None,
    progress_callback: CaseProgressCallback | None = None,
) -> EvaluationReport:
    """运行基础或微调大模型并生成可恢复的追溯报告。"""
    metadata = {
        "model_provider": model_provider.strip(),
        "model_name": model_name.strip(),
        "prompt_version": prompt_version.strip(),
        "validator_version": CITATION_VALIDATOR_VERSION,
        **dict(experiment_metadata or {}),
    }
    if any(not value.strip() for value in metadata.values()):
        raise ValueError("LLM evaluation metadata must not be blank")

    return await run_analyzer_evaluation(
        dataset=dataset,
        rag_corpus_path=rag_corpus_path,
        commercial_analyzer=commercial_analyzer,
        legal_analyzer=legal_analyzer,
        security_analyzer=security_analyzer,
        system_name=system_name,
        system_metadata=metadata,
        case_ids=case_ids,
        existing_report=existing_report,
        report_callback=report_callback,
        progress_callback=progress_callback,
    )


def load_evaluation_report(path: Path) -> EvaluationReport:
    """读取并校验用于断点恢复的已有评测报告。"""
    return EvaluationReport.model_validate_json(
        path.read_text(encoding="utf-8-sig")
    )


def save_evaluation_report(
    *,
    report: EvaluationReport,
    path: Path,
) -> None:
    """使用临时文件原子写入UTF-8 JSON评测报告。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        report.model_dump_json(indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)
