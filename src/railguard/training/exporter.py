"""监督微调标注文件的读取和可追溯消息导出。"""

from hashlib import sha256
from pathlib import Path
from typing import Literal

from pydantic import Field

from railguard.agents.prompts import (
    COMMERCIAL_PROFILE,
    LEGAL_PROFILE,
    PROMPT_VERSION,
    SECURITY_PROFILE,
    AgentProfile,
    build_system_prompt,
    build_user_prompt,
)
from railguard.models.schemas import SchemaModel
from railguard.training.models import AgentName, SftAnnotation

PROFILES: dict[AgentName, AgentProfile] = {
    "commercial_risk_agent": COMMERCIAL_PROFILE,
    "legal_risk_agent": LEGAL_PROFILE,
    "security_risk_agent": SECURITY_PROFILE,
}


class TrainingMessage(SchemaModel):
    """一条兼容聊天模型监督微调格式的消息。"""

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class SftExportRecord(SchemaModel):
    """带追溯元数据的供应商无关训练消息记录。"""

    example_id: str = Field(min_length=1)
    split: Literal["sft_train", "sft_validation"]
    agent_name: AgentName
    prompt_version: str = Field(min_length=1)
    messages: list[TrainingMessage] = Field(min_length=3, max_length=3)


def load_sft_annotations(path: Path) -> list[SftAnnotation]:
    """逐行读取JSONL标注，并拒绝重复ID和跨分区合同泄漏。"""
    annotations: list[SftAnnotation] = []
    seen_ids: set[str] = set()
    fingerprint_by_contract_id: dict[str, str] = {}
    split_by_contract_id: dict[str, str] = {}
    split_by_fingerprint: dict[str, str] = {}

    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue
        try:
            annotation = SftAnnotation.model_validate_json(raw_line)
        except ValueError as exc:
            raise ValueError(
                f"invalid SFT annotation at line {line_number}"
            ) from exc
        if annotation.example_id in seen_ids:
            raise ValueError(
                f"duplicate SFT example_id: {annotation.example_id}"
            )

        contract_id = annotation.contract.contract_id
        fingerprint = sha256(
            annotation.contract.full_text.encode("utf-8")
        ).hexdigest()
        known_fingerprint = fingerprint_by_contract_id.get(contract_id)
        if known_fingerprint is not None and known_fingerprint != fingerprint:
            raise ValueError(
                f"contract_id maps to different content: {contract_id}"
            )

        known_contract_split = split_by_contract_id.get(contract_id)
        if (
            known_contract_split is not None
            and known_contract_split != annotation.split
        ):
            raise ValueError(
                f"contract_id appears in multiple splits: {contract_id}"
            )

        known_content_split = split_by_fingerprint.get(fingerprint)
        if (
            known_content_split is not None
            and known_content_split != annotation.split
        ):
            raise ValueError("contract content appears in multiple splits")

        seen_ids.add(annotation.example_id)
        fingerprint_by_contract_id[contract_id] = fingerprint
        split_by_contract_id[contract_id] = annotation.split
        split_by_fingerprint[fingerprint] = annotation.split
        annotations.append(annotation)

    if not annotations:
        raise ValueError("SFT annotation file must not be empty")
    return annotations


def build_sft_export_record(annotation: SftAnnotation) -> SftExportRecord:
    """把一条已批准标注转换成与线上推理一致的消息记录。"""
    annotation.require_training_ready()
    profile = PROFILES[annotation.agent_name]
    return SftExportRecord(
        example_id=annotation.example_id,
        split=annotation.split,
        agent_name=annotation.agent_name,
        prompt_version=PROMPT_VERSION,
        messages=[
            TrainingMessage(
                role="system",
                content=build_system_prompt(profile),
            ),
            TrainingMessage(
                role="user",
                content=build_user_prompt(
                    contract=annotation.contract,
                    evidence_by_clause=annotation.evidence_by_clause,
                    contract_evidence=annotation.contract_evidence,
                ),
            ),
            TrainingMessage(
                role="assistant",
                content=annotation.target.model_dump_json(),
            ),
        ],
    )


def save_sft_export_records(
    records: list[SftExportRecord],
    path: Path,
) -> None:
    """按单一分区原子写入确定性的UTF-8 JSONL导出文件。"""
    if not records:
        raise ValueError("SFT export records must not be empty")

    example_ids = [record.example_id for record in records]
    if len(example_ids) != len(set(example_ids)):
        raise ValueError("SFT export example_ids must be unique")

    splits = {record.split for record in records}
    if len(splits) != 1:
        raise ValueError("SFT export must contain exactly one split")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    serialized = "\n".join(
        record.model_dump_json() for record in records
    )
    temporary_path.write_text(f"{serialized}\n", encoding="utf-8")
    temporary_path.replace(path)
