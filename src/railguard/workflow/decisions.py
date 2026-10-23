"""Recoverable decision commands; the business DB and checkpoint DB are not atomic."""
import hashlib
import json

from railguard.storage.postgres import PostgresContractRepository
from railguard.workflow.service import ReviewConflictError


class Commands:
    def __init__(self,dsn):self.repository=PostgresContractRepository(dsn)

    def begin(self,review,subject,key,decision,allow_create=True):
        fingerprint=hashlib.sha256(json.dumps(decision.model_dump(mode='json'),sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        with self.repository._connect() as db:
            row=db.execute('SELECT * FROM rg_decision_commands WHERE review_id=? AND subject=? AND command_id=?',
                           (review,subject,key)).fetchone()
            if row:
                if row['fingerprint']!=fingerprint:raise ReviewConflictError('Decision idempotency key reused with different input')
                return row,False
            if not allow_create:raise ReviewConflictError('Review is not awaiting a new decision')
            db.execute("INSERT INTO rg_decision_commands VALUES(?,?,?,?,'pending',NULL)",(review,subject,key,fingerprint))
            return {'status':'pending','result':None},True

    def complete(self,review,subject,key,result):
        with self.repository._connect() as db:
            db.execute("UPDATE rg_decision_commands SET status='applied',result=? WHERE review_id=? AND subject=? AND command_id=?",
                       (result.model_dump_json(),review,subject,key))
