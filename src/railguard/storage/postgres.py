"""PostgreSQL repository and narrow SQL adapter for existing fixed ACL queries."""
from contextlib import contextmanager
from datetime import UTC, datetime

from railguard.storage.contracts import ContractRepository
from railguard.workflow.outbox import OUTBOX_DDL

DDL='''
CREATE TABLE IF NOT EXISTS contracts(contract_id TEXT PRIMARY KEY,document_json TEXT NOT NULL,created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS rg_users(subject TEXT PRIMARY KEY,role TEXT NOT NULL,active INTEGER NOT NULL,revision INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS rg_tokens(fingerprint TEXT PRIMARY KEY,subject TEXT NOT NULL,revision INTEGER NOT NULL,expires DOUBLE PRECISION NOT NULL,revoked INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS rg_owners(contract_id TEXT PRIMARY KEY,subject TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS rg_grants(contract_id TEXT NOT NULL,subject TEXT NOT NULL,PRIMARY KEY(contract_id,subject));
CREATE TABLE IF NOT EXISTS rg_reviews(review_id TEXT PRIMARY KEY,contract_id TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS rg_audit(id BIGSERIAL PRIMARY KEY,timestamp DOUBLE PRECISION NOT NULL,subject TEXT NOT NULL,action TEXT NOT NULL,resource TEXT NOT NULL,outcome TEXT NOT NULL,target TEXT NOT NULL DEFAULT '');
CREATE TABLE IF NOT EXISTS rg_decision_commands(review_id TEXT NOT NULL,subject TEXT NOT NULL,command_id TEXT NOT NULL,fingerprint TEXT NOT NULL,status TEXT NOT NULL,result TEXT,PRIMARY KEY(review_id,subject,command_id));
CREATE TABLE IF NOT EXISTS railguard_schema(version INTEGER PRIMARY KEY);
INSERT INTO railguard_schema VALUES(1) ON CONFLICT DO NOTHING;
'''
DDL = DDL + OUTBOX_DDL


class ACLConnection:
    def __init__(self, connection):
        self.connection=connection

    def execute(self, query, args=()):
        # Only fixed repository/ACL SQL is passed here; values always remain parameters.
        if query=='INSERT OR IGNORE INTO rg_grants VALUES(?,?)':
            query='INSERT INTO rg_grants VALUES(?,?) ON CONFLICT DO NOTHING'
        return self.connection.execute(query.replace('?','%s'),args)


class PostgresContractRepository(ContractRepository):
    def __init__(self, dsn):
        self.dsn=dsn

    @contextmanager
    def _connect(self):
        import psycopg
        from psycopg.rows import dict_row
        with psycopg.connect(self.dsn, row_factory=dict_row,connect_timeout=5,
                             options='-c statement_timeout=10000 -c lock_timeout=5000') as db:
            yield ACLConnection(db)

    def initialize(self):
        with self._connect() as db:
            if not db.execute('SELECT 1 FROM railguard_schema WHERE version=1').fetchone():
                raise RuntimeError('Railguard PostgreSQL migration required')

    def save(self, contract):
        with self._connect() as db:
            db.execute('INSERT INTO contracts VALUES(?,?,?)',
                       (contract.contract_id,contract.model_dump_json(),datetime.now(UTC).isoformat()))


def migrate(dsn):
    import psycopg
    with psycopg.connect(dsn,connect_timeout=5) as db:
        db.execute('SELECT pg_advisory_xact_lock(79342011)')
        db.execute(DDL)


if __name__=='__main__':
    import os
    migrate(os.environ['RAILGUARD_DATABASE_URL'])
