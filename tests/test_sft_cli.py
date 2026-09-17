"""监督微调数据命令行入口测试。"""

import argparse
import json
from pathlib import Path

import pytest

from railguard.training import __main__ as cli_module


def create_payload(*, status: str = "approved") -> dict[str, object]:
    """创建可用于命令行测试的最小安全负例。"""
    payload = json.loads(
        Path("data/training/sft-annotation-template-v1.json").read_text(
            encoding="utf-8"
        )
    )
    payload.update(
        {
            "example_id": "cli-safe-001",
            "agent_name": "security_risk_agent",
            "contract": {
                "contract_id": "cli-contract-001",
                "filename": "cli-safe.txt",
                "full_text": "本合同不涉及供应商处理生产数据。",
                "clauses": [],
            },
        }
    )
    payload["source"] = {
        "origin": "synthetic",
        "created_by": "annotator-01",
        "license_or_authorization": "project-owned",
        "deidentified": True,
    }
    payload["quality"] = {
        "annotator": "annotator-01",
        "reviewer": "reviewer-01",
        "review_status": status,
        "notes": "",
    }
    return payload


def write_jsonl(path: Path, payload: dict[str, object]) -> None:
    """把单条测试标注写成UTF-8 JSONL文件。"""
    path.write_text(
        f"{json.dumps(payload, ensure_ascii=False)}\n",
        encoding="utf-8",
    )


def test_parser_requires_subcommand() -> None:
    """验证命令行必须显式选择校验或导出。"""
    parser = cli_module.build_argument_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_validate_prints_safe_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """验证校验摘要不回显合同正文。"""
    input_path = tmp_path / "annotations.jsonl"
    write_jsonl(input_path, create_payload(status="draft"))
    arguments = argparse.Namespace(command="validate", input=input_path)

    cli_module.run_from_arguments(arguments)
    output = capsys.readouterr().out

    assert "总记录：1" in output
    assert "训练就绪：0" in output
    assert "本合同不涉及" not in output


def test_export_writes_selected_ready_split(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """验证命令行导出指定分区的已批准消息。"""
    input_path = tmp_path / "annotations.jsonl"
    output_path = tmp_path / "training.jsonl"
    write_jsonl(input_path, create_payload())
    arguments = argparse.Namespace(
        command="export",
        input=input_path,
        output=output_path,
        split="sft_train",
    )

    cli_module.run_from_arguments(arguments)
    exported = json.loads(output_path.read_text(encoding="utf-8"))

    assert exported["example_id"] == "cli-safe-001"
    assert len(exported["messages"]) == 3
    assert "导出记录：1" in capsys.readouterr().out


def test_export_rejects_draft_annotation(tmp_path: Path) -> None:
    """验证命令行不能导出尚未复核的草稿。"""
    input_path = tmp_path / "annotations.jsonl"
    write_jsonl(input_path, create_payload(status="draft"))
    arguments = argparse.Namespace(
        command="export",
        input=input_path,
        output=tmp_path / "training.jsonl",
        split="sft_train",
    )

    with pytest.raises(ValueError, match="not approved"):
        cli_module.run_from_arguments(arguments)
