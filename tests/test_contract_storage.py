"""合同持久化测试。

测试使用pytest提供的临时目录，
不会修改项目的真实数据库。
"""

import sqlite3
from pathlib import Path

import pytest

from railguard.models.schemas import ContractDocument
from railguard.storage.contracts import ContractRepository


def test_contract_survives_repository_restart(tmp_path: Path) -> None:
    """验证合同可以被新创建的存储实例读取。

    这证明数据保存在磁盘中，而非只存在于Python内存中。
    """
    db_path = tmp_path / "runtime" / "contracts.db"
    repository = ContractRepository(db_path)
    repository.initialize()

    contract = ContractDocument(
        filename="demo.docx",
        full_text="甲方验收合格后支付尾款。",
    )
    repository.save(contract)

    # 模拟重新启动后创建一个新的存储实例。
    restarted_repository = ContractRepository(db_path)
    restored = restarted_repository.get(contract.contract_id)

    assert restored == contract


def test_missing_contract_returns_none(tmp_path: Path) -> None:
    """验证查询不存在的合同不会返回伪造数据。"""
    repository = ContractRepository(tmp_path / "contracts.db")
    repository.initialize()

    assert repository.get("not-found") is None


def test_duplicate_id_does_not_replace_original(tmp_path: Path) -> None:
    """验证重复ID不能覆盖原始合同，并且异常后数据库仍可读取。"""
    repository = ContractRepository(tmp_path / "contracts.db")
    repository.initialize()

    original = ContractDocument(
        filename="original.docx",
        full_text="甲方验收合格后付款。",
    )
    repository.save(original)

    # 使用相同合同ID，尝试保存不同内容。
    duplicate = ContractDocument(
        contract_id=original.contract_id,
        filename="changed.docx",
        full_text="甲方验收前支付全部款项。",
    )

    with pytest.raises(sqlite3.IntegrityError):
        repository.save(duplicate)

    # 重复写入失败后，原合同必须保持完整。
    assert repository.get(original.contract_id) == original