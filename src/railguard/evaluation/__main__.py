"""RailGuard离线评测命令行入口。"""

import argparse
import asyncio
from pathlib import Path

from railguard.agents.guardrails import (
    PRECISION_GUARDRAIL_VERSION,
)
from railguard.agents.prompts import PROMPT_VERSION
from railguard.agents.provider import (
    create_deepseek_risk_analyzers,
    create_openai_risk_analyzers,
)
from railguard.config import Settings
from railguard.evaluation.models import EvaluationMetrics, EvaluationReport
from railguard.evaluation.provenance import git_metadata, sha256_file
from railguard.evaluation.runner import (
    load_evaluation_dataset,
    load_evaluation_report,
    run_llm_evaluation,
    run_rules_evaluation,
    save_evaluation_report,
)


def build_argument_parser() -> argparse.ArgumentParser:
    """创建离线评测命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="运行RailGuard合同审核离线评测。"
    )
    parser.add_argument(
        "--target",
        choices=("rules", "llm"),
        default="rules",
        help="rules运行阶段A；llm运行阶段B或阶段C。",
    )
    parser.add_argument(
        "--system-name",
        default="base_llm",
        help="写入报告的实验名称，例如deepseek_base_llm。",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/evaluation/contract-review-v1.json"),
        help="评测数据集JSON文件路径。",
    )
    parser.add_argument(
        "--rag-corpus",
        type=Path,
        default=Path("data/demo/rag-corpus.json"),
        help="本地Mock RAG知识库文件路径。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="报告输出路径；不填写时根据target选择默认路径。",
    )
    parser.add_argument(
        "--case-id",
        dest="case_ids",
        action="append",
        default=None,
        help="只运行指定案例；可重复传入多个case_id。",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="读取现有输出报告并跳过已经完成的案例。",
    )
    return parser


def resolve_output_path(arguments: argparse.Namespace) -> Path:
    """根据命令行目标选择默认报告输出路径。"""
    if arguments.output is not None:
        return arguments.output
    filename = (
        "rules-v1.1-report.json"
        if arguments.target == "rules"
        else "base-llm-v1.1-report.json"
    )
    return Path("data/runtime/evaluations") / filename


def format_optional_rate(value: float | None) -> str:
    """把可选比例转换成适合终端阅读的百分比。"""
    if value is None:
        return "无可评估样本"
    return f"{value:.2%}"


def print_metrics(metrics: EvaluationMetrics) -> None:
    """在终端输出最重要的总体评测指标。"""
    print()
    print("RailGuard离线评测完成")
    print(
        "风险计数："
        f"TP={metrics.true_positive}，"
        f"FP={metrics.false_positive}，"
        f"FN={metrics.false_negative}"
    )
    print(f"精确率：{metrics.precision:.2%}")
    print(f"召回率：{metrics.recall:.2%}")
    print(f"F1：{metrics.f1:.2%}")
    print(f"风险等级准确率：{format_optional_rate(metrics.level_accuracy)}")
    print(f"原文定位准确率：{format_optional_rate(metrics.location_accuracy)}")
    print(f"引用覆盖率：{format_optional_rate(metrics.citation_coverage)}")
    print(f"来源匹配率：{format_optional_rate(metrics.source_matched_rate)}")


async def run_rules_target(arguments: argparse.Namespace) -> EvaluationReport:
    """执行阶段A确定性规则基线。"""
    dataset = load_evaluation_dataset(arguments.dataset)
    return await run_rules_evaluation(
        dataset=dataset,
        rag_corpus_path=arguments.rag_corpus,
        case_ids=getattr(arguments, "case_ids", None),
        experiment_metadata={
            "dataset_sha256": sha256_file(arguments.dataset),
            "rag_corpus_sha256": sha256_file(
                arguments.rag_corpus
            ),
            **git_metadata(Path.cwd()),
        },
    )


def _model_connection(settings: Settings):
    """返回当前真实模型供应商的连接配置和Agent工厂。"""
    if settings.model_provider == "openai":
        return (
            settings.openai_api_key,
            settings.openai_base_url,
            create_openai_risk_analyzers,
            "OPENAI_API_KEY",
        )
    if settings.model_provider == "deepseek":
        return (
            settings.deepseek_api_key,
            settings.deepseek_base_url,
            create_deepseek_risk_analyzers,
            "DEEPSEEK_API_KEY",
        )
    raise ValueError(
        "LLM evaluation requires MODEL_PROVIDER=openai or deepseek"
    )


async def run_llm_target(
    arguments: argparse.Namespace,
    *,
    settings: Settings | None = None,
    output_path: Path | None = None,
) -> EvaluationReport:
    """执行支持增量保存和恢复的真实模型评测。"""
    active_settings = settings or Settings()
    api_key, base_url, factory, key_name = _model_connection(
        active_settings
    )
    if api_key is None or not api_key.strip():
        raise ValueError(f"LLM evaluation requires {key_name}")
    if not arguments.system_name.strip():
        raise ValueError("system_name must not be blank")

    dataset = load_evaluation_dataset(arguments.dataset)
    analyzers = factory(
        model_name=active_settings.model_name,
        api_key=api_key,
        base_url=base_url,
        timeout_seconds=active_settings.model_timeout_seconds,
        max_retries=active_settings.model_max_retries,
        max_input_chars=active_settings.model_max_input_chars,
    )
    selected_ids = getattr(arguments, "case_ids", None)
    selected_count = (
        len(selected_ids)
        if selected_ids is not None
        else len(dataset.cases)
    )
    print(
        "即将运行真实模型评测："
        f"{selected_count}个案例，"
        f"预计{selected_count * 3}次模型调用。"
    )

    resolved_output = output_path or resolve_output_path(arguments)
    resume = bool(getattr(arguments, "resume", False))
    existing_report = (
        load_evaluation_report(resolved_output)
        if resume and resolved_output.exists()
        else None
    )
    experiment_metadata = {
        "dataset_sha256": sha256_file(arguments.dataset),
        "rag_corpus_sha256": sha256_file(arguments.rag_corpus),
        "model_timeout_seconds": str(
            active_settings.model_timeout_seconds
        ),
        "model_max_retries": str(active_settings.model_max_retries),
        "model_max_input_chars": str(
            active_settings.model_max_input_chars
        ),
        "precision_guardrail_version": (
            PRECISION_GUARDRAIL_VERSION
        ),
        **git_metadata(Path.cwd()),
    }

    def save_progress(report: EvaluationReport) -> None:
        """每完成一个案例就原子保存当前报告。"""
        save_evaluation_report(report=report, path=resolved_output)

    def print_progress(position: int, total: int, case_id: str) -> None:
        """在终端显示即将执行的案例进度。"""
        print(f"[{position}/{total}] 正在评测：{case_id}")

    return await run_llm_evaluation(
        dataset=dataset,
        rag_corpus_path=arguments.rag_corpus,
        commercial_analyzer=analyzers.commercial,
        legal_analyzer=analyzers.legal,
        security_analyzer=analyzers.security,
        system_name=arguments.system_name.strip(),
        model_provider=active_settings.model_provider,
        model_name=active_settings.model_name,
        prompt_version=PROMPT_VERSION,
        experiment_metadata=experiment_metadata,
        case_ids=selected_ids,
        existing_report=existing_report,
        report_callback=save_progress,
        progress_callback=print_progress,
    )


async def run_from_arguments(
    arguments: argparse.Namespace,
    *,
    settings: Settings | None = None,
) -> None:
    """执行选定目标、保存报告并输出总体指标。"""
    output_path = resolve_output_path(arguments)
    if arguments.target == "rules":
        report = await run_rules_target(arguments)
    else:
        report = await run_llm_target(
            arguments,
            settings=settings,
            output_path=output_path,
        )
    save_evaluation_report(report=report, path=output_path)

    print(f"系统：{report.system_name}")
    print(f"数据集：{report.dataset_name} {report.dataset_version}")
    print(f"案例数量：{len(report.case_results)}")
    if report.system_metadata:
        print(f"实验条件：{report.system_metadata}")
    print_metrics(report.aggregate_metrics)
    print(f"报告路径：{output_path}")


def main() -> None:
    """解析命令行参数并运行异步评测。"""
    arguments = build_argument_parser().parse_args()
    asyncio.run(run_from_arguments(arguments))


if __name__ == "__main__":
    main()
