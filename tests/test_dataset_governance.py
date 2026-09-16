"""微调数据分区治理测试。"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from railguard.evaluation.governance import (
    DatasetGovernanceManifest,
    DatasetPartition,
    load_governance_manifest,
    verify_frozen_partition,
)


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
