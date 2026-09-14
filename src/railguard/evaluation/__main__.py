"""RailGuard离线评测命令行入口。"""

import argparse
import asyncio
from pathlib import Path

from railguard.agents.prompts import PROMPT_VERSION
from railguard.agents.provider import (
    create_deepseek_risk_analyzers,
    create_openai_risk_analyzers,
)
from railguard.config import Settings
from railguard.evaluation.models import (
    EvaluationMetrics,
    EvaluationReport,
)
from railguard.evaluation.runner import (
    load_evaluation_dataset,
    run_llm_evaluation,
    run_rules_evaluation,
    save_evaluation_report,
)


def build_argument_parser() -> argparse.ArgumentParser:
    """创建离线评测命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description=(
            "运行RailGuard合同审核离线评测。"
        )
    )
    parser.add_argument(
        "--target",
        choices=(
            "rules",
            "llm",
        ),
        default="rules",
        help=(
            "rules运行阶段A规则基线；"
            "llm运行阶段B或阶段C模型评测。"
        ),
    )
    parser.add_argument(
        "--system-name",
        default="base_llm",
        help=(
            "写入报告的实验名称，例如base_llm或sft_llm。"
        ),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "data/evaluation/"
            "contract-review-v1.json"
        ),
        help="评测数据集JSON文件路径。",
    )
    parser.add_argument(
        "--rag-corpus",
        type=Path,
        default=Path(
            "data/demo/rag-corpus.json"
        ),
        help="本地Mock RAG知识库文件路径。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "评测报告输出路径；不填写时根据target选择默认路径。"
        ),
    )

    return parser


def resolve_output_path(
    arguments: argparse.Namespace,
) -> Path:
    """根据命令行目标选择默认报告输出路径。"""
    if arguments.output is not None:
        return arguments.output

    filename = (
        "rules-v1-report.json"
        if arguments.target == "rules"
        else "base-llm-v1-report.json"
    )

    return Path(
        "data/runtime/evaluations"
    ) / filename


def format_optional_rate(
    value: float | None,
) -> str:
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
    print(
        "风险等级准确率："
        f"{format_optional_rate(metrics.level_accuracy)}"
    )
    print(
        "原文定位准确率："
        f"{format_optional_rate(metrics.location_accuracy)}"
    )
    print(
        "引用覆盖率："
        f"{format_optional_rate(metrics.citation_coverage)}"
    )
    print(
        "来源匹配率："
        f"{format_optional_rate(metrics.source_matched_rate)}"
    )


async def run_rules_target(
    arguments: argparse.Namespace,
) -> EvaluationReport:
    """执行阶段A确定性规则基线。"""
    dataset = load_evaluation_dataset(
        arguments.dataset
    )

    return await run_rules_evaluation(
        dataset=dataset,
        rag_corpus_path=arguments.rag_corpus,
    )


async def run_llm_target(
    arguments: argparse.Namespace,
    *,
    settings: Settings | None = None,
) -> EvaluationReport:
    """根据供应商配置执行通用模型或合同领域微调模型评测。"""
    active_settings = settings or Settings()

    # 根据供应商选择对应密钥、接口地址和模型创建工厂。
    if active_settings.model_provider == "openai":
        api_key = active_settings.openai_api_key
        base_url = active_settings.openai_base_url
        factory = create_openai_risk_analyzers
        key_name = "OPENAI_API_KEY"

    elif active_settings.model_provider == "deepseek":
        api_key = active_settings.deepseek_api_key
        base_url = active_settings.deepseek_base_url
        factory = create_deepseek_risk_analyzers
        key_name = "DEEPSEEK_API_KEY"

    else:
        # 真实模型目标必须明确启用对应供应商。
        raise ValueError(
            "LLM evaluation requires "
            "MODEL_PROVIDER=openai or deepseek"
        )

    # 再次检查密钥，保证模型工厂收到有效的字符串。
    if api_key is None or not api_key.strip():
        raise ValueError(
            f"LLM evaluation requires {key_name}"
        )

    # 实验名称必须明确，便于区分不同阶段的评测报告。
    if not arguments.system_name.strip():
        raise ValueError(
            "system_name must not be blank"
        )

    dataset = load_evaluation_dataset(
        arguments.dataset
    )

    # 两家供应商使用相同的调用限制和专业Agent接口。
    analyzers = factory(
        model_name=active_settings.model_name,
        api_key=api_key,
        base_url=base_url,
        timeout_seconds=(
            active_settings.model_timeout_seconds
        ),
        max_retries=active_settings.model_max_retries,
        max_input_chars=(
            active_settings.model_max_input_chars
        ),
    )

    # 每份案例分别调用商务、法务和安全三个模型Agent。
    # 这里的次数不包含SDK发生临时故障时产生的重试请求。
    request_count = len(dataset.cases) * 3
    print(
        "即将运行真实模型评测："
        f"{len(dataset.cases)}个案例，"
        f"预计{request_count}次模型调用。"
    )

    # 复用同一数据集、RAG知识库和评测口径。
    # 元数据只记录供应商、模型和提示词版本。
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
    )


async def run_from_arguments(
    arguments: argparse.Namespace,
    *,
    settings: Settings | None = None,
) -> None:
    """执行选定评测目标、保存报告并输出总体指标。"""
    if arguments.target == "rules":
        report = await run_rules_target(arguments)
    else:
        report = await run_llm_target(
            arguments,
            settings=settings,
        )

    output_path = resolve_output_path(arguments)
    save_evaluation_report(
        report=report,
        path=output_path,
    )

    print(f"系统：{report.system_name}")
    print(
        "数据集："
        f"{report.dataset_name} "
        f"{report.dataset_version}"
    )
    print(
        f"案例数量：{len(report.case_results)}"
    )

    if report.system_metadata:
        print(
            "实验条件："
            f"{report.system_metadata}"
        )

    print_metrics(report.aggregate_metrics)
    print(f"报告路径：{output_path}")


def main() -> None:
    """解析命令行参数并运行异步评测。"""
    parser = build_argument_parser()
    arguments = parser.parse_args()
    asyncio.run(
        run_from_arguments(arguments)
    )


if __name__ == "__main__":
    main()