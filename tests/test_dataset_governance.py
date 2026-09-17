"""微调数据分区治理测试。"""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from railguard.evaluation.governance import (
    DatasetGovernanceManifest,
    DatasetPartition,
    load_governance_manifest,
    verify_frozen_partition,
)
from railguard.evaluation.provenance import sha256_file


def test_current_development_benchmark_is_frozen() -> None:
    """验证当前21例的哈希、顺序和开发集身份已经冻结。"""
    manifest = load_governance_manifest(
        Path("data/evaluation/dataset-governance-v1.json")
    )
    partition = manifest.partitions[0]

    assert partition.role == "development_benchmark"
    assert partition.used_for_prompt_tuning is True
    assert partition.eligible_for_final_metrics is False
    assert len(partition.case_ids) == 21
    verify_frozen_partition(partition, repository_root=Path.cwd())


def test_manifest_rejects_case_leakage_between_partitions() -> None:
    """验证同一案例不能同时进入训练集和最终测试集。"""
    common = {
        "status": "planned",
        "dataset_path": "future.json",
        "case_ids": ["shared-case"],
    }

    with pytest.raises(ValidationError, match="multiple partitions"):
        DatasetGovernanceManifest(
            schema_version="1.0.0",
            partitions=[
                DatasetPartition(
                    partition_id="train",
                    role="sft_train",
                    **common,
                ),
                DatasetPartition(
                    partition_id="test",
                    role="held_out_test",
                    **common,
                ),
            ],
        )


def test_prompt_tuned_partition_cannot_report_final_metrics() -> None:
    """验证参与提示词调优的数据不得作为最终无偏测试集。"""
    with pytest.raises(ValidationError, match="final metrics"):
        DatasetPartition(
            partition_id="leaked-test",
            role="held_out_test",
            status="planned",
            dataset_path="future.json",
            case_ids=[],
            used_for_prompt_tuning=True,
            eligible_for_final_metrics=True,
        )


def create_ready_sft_payload() -> dict[str, object]:
    """创建可冻结的最小SFT安全负例记录。"""
    payload = json.loads(
        Path("data/training/sft-annotation-template-v1.json").read_text(
            encoding="utf-8"
        )
    )
    payload.update(
        {
            "example_id": "governance-sft-001",
            "agent_name": "security_risk_agent",
            "contract": {
                "contract_id": "governance-contract-001",
                "filename": "governance.txt",
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
        "review_status": "approved",
        "notes": "",
    }
    return payload


def test_frozen_sft_partition_verifies_jsonl_records(
    tmp_path: Path,
) -> None:
    """验证冻结SFT分区会检查哈希、审批状态和样本ID。"""
    dataset_path = tmp_path / "sft-train.jsonl"
    dataset_path.write_text(
        f"{json.dumps(create_ready_sft_payload(), ensure_ascii=False)}\n",
        encoding="utf-8",
    )
    partition = DatasetPartition(
        partition_id="sft-train-test",
        role="sft_train",
        status="frozen",
        data_format="sft_jsonl",
        dataset_path=dataset_path.name,
        dataset_sha256=sha256_file(dataset_path),
        case_ids=["governance-sft-001"],
    )

    verify_frozen_partition(partition, repository_root=tmp_path)


def test_frozen_sft_partition_rejects_wrong_split(
    tmp_path: Path,
) -> None:
    """验证标注内部split必须与治理清单的分区角色一致。"""
    payload = create_ready_sft_payload()
    payload["split"] = "sft_validation"
    dataset_path = tmp_path / "sft-train.jsonl"
    dataset_path.write_text(
        f"{json.dumps(payload, ensure_ascii=False)}\n",
        encoding="utf-8",
    )
    partition = DatasetPartition(
        partition_id="sft-train-test",
        role="sft_train",
        status="frozen",
        data_format="sft_jsonl",
        dataset_path=dataset_path.name,
        dataset_sha256=sha256_file(dataset_path),
        case_ids=["governance-sft-001"],
    )

    with pytest.raises(ValueError, match="split does not match"):
        verify_frozen_partition(partition, repository_root=tmp_path)
