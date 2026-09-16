"""微调前的数据集分区治理和防泄漏校验。"""

from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from railguard.evaluation.provenance import sha256_file
from railguard.evaluation.runner import load_evaluation_dataset
from railguard.models.schemas import SchemaModel

PartitionRole = Literal[
    "development_benchmark",
    "sft_train",
    "sft_validation",
    "held_out_test",
]
PartitionStatus = Literal["frozen", "planned"]


class DatasetPartition(SchemaModel):
    """一个用途明确且可追溯的数据集分区。"""

    partition_id: str = Field(min_length=1)
    role: PartitionRole
    status: PartitionStatus
    dataset_path: str = Field(min_length=1)
    dataset_sha256: str | None = None
    case_ids: list[str] = Field(default_factory=list)
    used_for_prompt_tuning: bool = False
    eligible_for_final_metrics: bool = False
    notes: str = ""

    @model_validator(mode="after")
    def validate_partition_policy(self) -> Self:
        """验证冻结信息、案例唯一性和最终测试资格。"""
        if len(self.case_ids) != len(set(self.case_ids)):
            raise ValueError("partition case_ids must be unique")
        if self.status == "frozen" and (
            not self.case_ids or self.dataset_sha256 is None
        ):
            raise ValueError(
                "frozen partition requires case_ids and dataset_sha256"
            )
        if self.used_for_prompt_tuning and self.eligible_for_final_metrics:
            raise ValueError(
                "prompt-tuned partition cannot provide final metrics"
            )
        return self


class DatasetGovernanceManifest(SchemaModel):
    """记录训练、验证和测试分区的版本化治理清单。"""

    schema_version: str = Field(min_length=1)
    partitions: list[DatasetPartition] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_disjoint_partitions(self) -> Self:
        """确保分区名称唯一且案例不会跨分区泄漏。"""
        partition_ids = [item.partition_id for item in self.partitions]
        if len(partition_ids) != len(set(partition_ids)):
            raise ValueError("partition_id values must be unique")

        owner_by_case: dict[str, str] = {}
        for partition in self.partitions:
            for case_id in partition.case_ids:
                if case_id in owner_by_case:
                    raise ValueError(
                        "case_id appears in multiple partitions: "
                        f"{case_id}"
                    )
                owner_by_case[case_id] = partition.partition_id
        return self


def load_governance_manifest(path: Path) -> DatasetGovernanceManifest:
    """从UTF-8 JSON读取并校验数据治理清单。"""
    return DatasetGovernanceManifest.model_validate_json(
        path.read_text(encoding="utf-8-sig")
    )


def verify_frozen_partition(
    partition: DatasetPartition,
    *,
    repository_root: Path,
) -> None:
    """校验冻结分区的文件哈希和案例顺序没有漂移。"""
    if partition.status != "frozen":
        raise ValueError("only frozen partitions can be verified")

    dataset_path = repository_root / partition.dataset_path
    if sha256_file(dataset_path) != partition.dataset_sha256:
        raise ValueError("frozen dataset sha256 does not match")

    dataset = load_evaluation_dataset(dataset_path)
    case_ids = [case.case_id for case in dataset.cases]
    if case_ids != partition.case_ids:
        raise ValueError("frozen dataset case_ids do not match")
