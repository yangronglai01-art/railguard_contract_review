"""监督微调JSONL加载和消息导出测试。"""

import json
from pathlib import Path

import pytest

from railguard.training.exporter import (
    build_sft_export_record,
    load_sft_annotations,
    save_sft_export_records,
)
from railguard.training.models import SftAnnotation


def load_template() -> dict[str, object]:
    """读取项目内空白模板并补成最小安全负例。"""
    payload = json.loads(
        Path("data/training/sft-annotation-template-v1.json").read_text(
            encoding="utf-8"
        )
    )
    payload.update(
        {
            "example_id": "sft-safe-001",
            "agent_name": "security_risk_agent",
            "contract": {
                "contract_id": "sft-contract-safe-001",
                "filename": "safe.txt",
                "full_text": "本合同不含供应商数据处理事项。",
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
        "review_status": "approved",
        "notes": "",
    }
    return payload


def create_annotation() -> SftAnnotation:
    """创建可以直接导出的已批准安全负例。"""
    return SftAnnotation.model_validate(load_template())


def test_loader_reads_jsonl_and_rejects_duplicate_ids(
    tmp_path: Path,
) -> None:
    """验证加载器接受空行但拒绝重复样本ID。"""
    annotation = create_annotation()
    path = tmp_path / "annotations.jsonl"
    line = annotation.model_dump_json()
    path.write_text(f"{line}\n\n{line}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate SFT example_id"):
        load_sft_annotations(path)


def test_loader_reports_invalid_line_number(tmp_path: Path) -> None:
    """验证无效JSON能够定位到原始行号。"""
    path = tmp_path / "annotations.jsonl"
    path.write_text("\n{invalid-json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line 2"):
        load_sft_annotations(path)


def test_export_reuses_production_prompts_and_target() -> None:
    """验证训练消息复用线上提示词并保留结构化目标。"""
    annotation = create_annotation()
    record = build_sft_export_record(annotation)

    assert record.example_id == annotation.example_id
    assert record.prompt_version == "contract-risk-v2"
    assert [message.role for message in record.messages] == [
        "system",
        "user",
        "assistant",
    ]
    assert "数据安全风险审核Agent" in record.messages[0].content
    assert annotation.contract.contract_id in record.messages[1].content
    assert json.loads(record.messages[2].content) == {"findings": []}


def test_export_requires_approved_annotation() -> None:
    """验证草稿标注无法绕过训练就绪检查。"""
    annotation = create_annotation()
    annotation.quality.review_status = "draft"

    with pytest.raises(ValueError, match="not approved"):
        build_sft_export_record(annotation)


def test_writer_is_deterministic_and_rejects_mixed_splits(
    tmp_path: Path,
) -> None:
    """验证导出内容稳定，并阻止训练集与验证集混写。"""
    training = build_sft_export_record(create_annotation())
    path = tmp_path / "training.jsonl"
    save_sft_export_records([training], path)
    first_content = path.read_text(encoding="utf-8")
    save_sft_export_records([training], path)

    assert path.read_text(encoding="utf-8") == first_content
    validation = training.model_copy(
        update={
            "example_id": "sft-safe-validation-001",
            "split": "sft_validation",
        }
    )
    with pytest.raises(ValueError, match="exactly one split"):
        save_sft_export_records([training, validation], path)
