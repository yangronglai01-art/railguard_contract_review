"""PG 事务 Outbox 事件驱动架构。

document 25 目标架构的事件投递层。领域事件（:class:`ReviewEvent`）在
业务流程的关键节点产生后，写入业务数据库的 ``rg_outbox`` 表，与业务数据
在同一事务内提交。后台投递器异步读取待投递事件并发布到消息订阅方，
保证「业务状态变更」与「事件发布」的一致性。

本模块提供两级实现：

- :class:`InMemoryOutbox`：本地演示/测试用，事件仅保留在进程内；
- :class:`PostgresOutbox`：生产用，事件与业务数据同库同事务提交。

Outbox 只负责「可靠落库」，不负责「投递」；投递器（dispatcher）留作
后续接入消息队列时实现。当前版本把事件同时写入 LangGraph checkpoint
状态（见 ``ReviewState.events``），因此即使尚未接入消息队列，事件也
不会被丢失。
"""

from __future__ import annotations

from typing import Protocol

from railguard.workflow.orchestration import ReviewEvent


class Outbox(Protocol):
    """事件 Outbox 的最小接口。"""

    def append(self, event: ReviewEvent) -> None:
        """把一条领域事件追加到待投递队列。"""

    def append_many(self, events: list[ReviewEvent]) -> None:
        """批量追加领域事件。"""

    def flush(self) -> None:
        """把内存中缓存的事件写入存储（幂等）。"""

    def pending(self) -> list[ReviewEvent]:
        """返回尚未标记为已投递的事件（主要用于本地测试与监控）。"""


class InMemoryOutbox:
    """进程内 Outbox，事件仅保留在内存，用于本地演示与测试。"""

    def __init__(self) -> None:
        self._events: list[ReviewEvent] = []

    def append(self, event: ReviewEvent) -> None:
        self._events.append(event)

    def append_many(self, events: list[ReviewEvent]) -> None:
        self._events.extend(events)

    def flush(self) -> None:
        # 内存实现无需显式落库。
        return None

    def pending(self) -> list[ReviewEvent]:
        return list(self._events)


class PostgresOutbox:
    """写入 PostgreSQL ``rg_outbox`` 表的事务 Outbox。

    调用方负责在同一个数据库事务中调用 :meth:`append` 与业务写操作，
    然后一并提交，从而保证事件与业务状态原子一致。
    """

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._buffer: list[ReviewEvent] = []

    def append(self, event: ReviewEvent) -> None:
        self._buffer.append(event)

    def append_many(self, events: list[ReviewEvent]) -> None:
        self._buffer.extend(events)

    def flush(self) -> None:
        """把缓冲事件写入 outbox 表（幂等，按 event_id 去重）。"""
        if not self._buffer:
            return

        import psycopg

        with psycopg.connect(
            self._dsn,
            connect_timeout=5,
            options="-c statement_timeout=10000 -c lock_timeout=5000",
        ) as db, db.transaction():
            for event in self._buffer:
                db.execute(
                    """
                        INSERT INTO rg_outbox(
                            event_id, event_type, aggregate_id,
                            trace_id, occurred_at_iso, payload,
                            delivery_status
                        ) VALUES(%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT(event_id) DO NOTHING
                        """,
                    (
                        event.event_id,
                        event.event_type,
                        event.aggregate_id,
                        event.trace_id,
                        event.occurred_at_iso,
                        event.model_dump_json(
                            include={"payload"}
                        ),
                        event.delivery_status,
                    ),
                )
        self._buffer.clear()

    def pending(self) -> list[ReviewEvent]:
        return list(self._buffer)


# Outbox DDL，随 schema 迁移一并执行（见 storage/postgres.py）。
OUTBOX_DDL = """
CREATE TABLE IF NOT EXISTS rg_outbox(
    event_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    aggregate_id TEXT NOT NULL,
    trace_id TEXT NOT NULL,
    occurred_at_iso TEXT NOT NULL,
    payload TEXT NOT NULL DEFAULT '{}',
    delivery_status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_rg_outbox_delivery
    ON rg_outbox(delivery_status, occurred_at_iso);
"""
