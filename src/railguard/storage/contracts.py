"""合同存储模块。

使用Python标准库sqlite3，无需额外安装数据库依赖。
演示阶段将完整合同模型保存为JSON，保留条款和原文位置。
"""

import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path

from railguard.models.schemas import ContractDocument


class ContractRepository:
    """封装合同数据库操作。

    API负责请求和响应，本类负责保存和读取合同，
    避免在接口函数中直接编写SQL。
    """

    def __init__(self, db_path: Path) -> None:
        """接收数据库文件路径。

        此方法只保存路径，不创建数据库。
        应在应用启动时调用initialize完成初始化。
        """
        self.db_path = Path(db_path)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """提供自动提交、回滚和关闭的数据库连接。

        每次操作创建独立连接，避免多个请求共享同一个连接。

        正常结束：提交事务。
        执行异常：回滚事务。
        无论是否异常：关闭连接。

        timeout表示数据库被其他写操作占用时，最多等待10秒。
        """
        with closing(
            sqlite3.connect(self.db_path, timeout=10.0)
        ) as connection:
            # 查询结果支持通过字段名称访问。
            connection.row_factory = sqlite3.Row

            # sqlite3连接的上下文管理器负责提交或回滚。
            # 外层closing负责真正关闭连接。
            with connection:
                yield connection

    def initialize(self) -> None:
        """创建数据库目录和合同表。

        IF NOT EXISTS使初始化可以重复执行，
        应用重新启动时不会删除已有合同。
        """
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS contracts (
                    contract_id TEXT PRIMARY KEY NOT NULL,
                    document_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def save(self, contract: ContractDocument) -> None:
        """新增一份合同。

        将Pydantic模型序列化为JSON，保留完整字段。

        使用参数占位符传递数据，避免将合同内容拼接进SQL。
        合同ID重复时抛出IntegrityError，不覆盖原有合同。
        """
        # 使用UTC记录保存时间，避免服务器时区不同造成混乱。
        created_at = datetime.now(UTC).isoformat()

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO contracts (
                    contract_id,
                    document_json,
                    created_at
                )
                VALUES (?, ?, ?)
                """,
                (
                    contract.contract_id,
                    contract.model_dump_json(),
                    created_at,
                ),
            )

    def get(self, contract_id: str) -> ContractDocument | None:
        """根据合同ID读取合同。

        找不到记录时返回None，由API决定是否返回HTTP 404。
        找到记录时重新解析JSON，并执行Pydantic模型校验。
        """
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT document_json
                FROM contracts
                WHERE contract_id = ?
                """,
                (contract_id,),
            ).fetchone()

        if row is None:
            return None

        return ContractDocument.model_validate_json(
            row["document_json"]
        )