"""RailGuard离线评测命令行入口。"""

import argparse
import asyncio
from pathlib import Path

from railguard.evaluation.models import (
    EvaluationMetrics,
)
from railguard.evaluation.runner import (
    load_evaluation_dataset,
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
        default=Path(
            "data/runtime/evaluations/"
            "rules-v1-report.json"
        ),
        help="评测报告输出路径。",
    )

    return parser


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


async def run_from_arguments(
    arguments: argparse.Namespace,
) -> None:
    """读取参数、执行规则基线并保存评测报告。"""
    dataset = load_evaluation_dataset(
        arguments.dataset
    )
    report = await run_rules_evaluation(
        dataset=dataset,
        rag_corpus_path=arguments.rag_corpus,
    )
    save_evaluation_report(
        report=report,
        path=arguments.output,
    )

    print(
        f"系统：{report.system_name}"
    )
    print(
        "数据集："
        f"{report.dataset_name} "
        f"{report.dataset_version}"
    )
    print(
        f"案例数量：{len(report.case_results)}"
    )
    print_metrics(report.aggregate_metrics)
    print(f"报告路径：{arguments.output}")


def main() -> None:
    """解析命令行参数并运行异步评测。"""
    parser = build_argument_parser()
    arguments = parser.parse_args()
    asyncio.run(
        run_from_arguments(arguments)
    )


if __name__ == "__main__":
    main()