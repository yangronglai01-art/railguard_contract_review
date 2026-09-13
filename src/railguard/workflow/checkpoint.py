"""LangGraph SQLite checkpoint生命周期工具。"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


def create_review_config(
    review_id: str,
) -> dict[str, dict[str, str]]:
    """根据审核ID创建LangGraph运行配置。

    review_id同时作为thread_id，使启动、查询和恢复操作
    始终访问同一条checkpoint记录。
    """
    if not review_id.strip():
        raise ValueError("review_id must not be blank")

    return {
        "configurable": {
            "thread_id": review_id,
        }
    }


@asynccontextmanager
async def open_sqlite_checkpointer(
    path: Path,
) -> AsyncIterator[AsyncSqliteSaver]:
    """创建、初始化并在退出时关闭SQLite checkpointer。

    checkpoint数据库与合同业务数据库使用不同文件，
    避免审核运行状态和业务数据相互耦合。
    """
    resolved_path = path.expanduser().resolve()

    # 首次运行时自动创建数据库所在目录。
    resolved_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # from_conn_string负责创建和关闭aiosqlite连接。
    async with AsyncSqliteSaver.from_conn_string(
        str(resolved_path)
    ) as checkpointer:
        # setup创建LangGraph所需的数据表，可重复调用。
        await checkpointer.setup()
        yield checkpointer