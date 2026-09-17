"""RailGuard监督微调数据校验和导出命令行入口。"""

import argparse
from pathlib import Path

from railguard.training.exporter import (
    build_sft_export_record,
    load_sft_annotations,
    save_sft_export_records,
)


def build_argument_parser() -> argparse.ArgumentParser:
    """创建训练数据工具的子命令参数解析器。"""
    parser = argparse.ArgumentParser(
        description="校验并导出RailGuard监督微调数据。"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate",
        help="校验JSONL结构、职责范围和证据白名单。",
    )
    validate_parser.add_argument("--input", type=Path, required=True)

    export_parser = subparsers.add_parser(
        "export",
        help="只导出已批准且已去标识的指定分区。",
    )
    export_parser.add_argument("--input", type=Path, required=True)
    export_parser.add_argument("--output", type=Path, required=True)
    export_parser.add_argument(
        "--split",
        choices=("sft_train", "sft_validation"),
        required=True,
    )
    return parser


def run_from_arguments(arguments: argparse.Namespace) -> None:
    """执行校验或受控导出，并输出不包含合同正文的摘要。"""
    annotations = load_sft_annotations(arguments.input)
    split_counts = {
        "sft_train": 0,
        "sft_validation": 0,
    }
    ready_count = 0
    for annotation in annotations:
        split_counts[annotation.split] += 1
        try:
            annotation.require_training_ready()
        except ValueError:
            continue
        ready_count += 1

    if arguments.command == "validate":
        print("RailGuard SFT标注校验完成")
        print(f"总记录：{len(annotations)}")
        print(f"训练分区：{split_counts['sft_train']}")
        print(f"验证分区：{split_counts['sft_validation']}")
        print(f"训练就绪：{ready_count}")
        return

    selected = [
        annotation
        for annotation in annotations
        if annotation.split == arguments.split
    ]
    if not selected:
        raise ValueError(f"no annotations found for split: {arguments.split}")
    records = [build_sft_export_record(item) for item in selected]
    save_sft_export_records(records, arguments.output)
    print("RailGuard SFT消息导出完成")
    print(f"分区：{arguments.split}")
    print(f"导出记录：{len(records)}")
    print(f"输出路径：{arguments.output}")


def main() -> None:
    """解析命令行并执行训练数据工具。"""
    arguments = build_argument_parser().parse_args()
    run_from_arguments(arguments)


if __name__ == "__main__":
    main()
