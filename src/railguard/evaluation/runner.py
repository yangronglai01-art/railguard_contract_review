"""合同审核离线评测执行器。"""

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
from railguard.models.schemas import (
    Clause,
    ContractDocument,
)
from railguard.parsers.clauses import split_clauses
from railguard.rag.mock import MockRagRetriever
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


async def run_rules_evaluation(
    *,
    dataset: EvaluationDataset,
    rag_corpus_path: Path,
) -> EvaluationReport:
    """运行当前确定性规则Agent并生成完整评测报告。"""
    retriever = MockRagRetriever.from_json_file(
        rag_corpus_path
    )
    graph = build_review_graph(
        retriever=retriever,
        checkpointer=InMemorySaver(),
    )
    service = ReviewService(graph)
    case_results = []

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
        system_name="deterministic_rules",
        dataset=dataset,
        case_results=case_results,
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