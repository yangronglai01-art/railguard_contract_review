"""Durable PostgreSQL checkpoints and cross-process review serialization."""
import hashlib
from contextlib import asynccontextmanager

from railguard.workflow.checkpoint import create_checkpoint_serializer


@asynccontextmanager
async def open_postgres_checkpointer(dsn):
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    async with AsyncPostgresSaver.from_conn_string(dsn) as saver:
        saver.serde=create_checkpoint_serializer()
        # Schema is installed explicitly by the migration command, not the web process.
        yield saver


@asynccontextmanager
async def review_lock(dsn, review_id):
    import psycopg
    key=int.from_bytes(hashlib.sha256(('railguard:'+review_id).encode()).digest()[:8],'big',signed=True)
    async with await psycopg.AsyncConnection.connect(dsn,autocommit=True,connect_timeout=5) as db:
        await db.execute("SET lock_timeout='5s'")
        await db.execute('SELECT pg_advisory_lock(%s)',(key,))
        try:yield
        finally:await db.execute('SELECT pg_advisory_unlock(%s)',(key,))


async def migrate_checkpoints(dsn):
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    async with AsyncPostgresSaver.from_conn_string(dsn) as saver:
        await saver.setup()


if __name__=='__main__':
    import asyncio
    import os
    import sys
    if sys.platform=='win32':asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(migrate_checkpoints(os.environ['RAILGUARD_CHECKPOINT_URL']))
