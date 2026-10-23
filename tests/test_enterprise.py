import asyncio
import os
import sys
import uuid
from types import SimpleNamespace

import pytest

from railguard.config import Settings
from railguard.enterprise_identity import Identity, IdentityError
from railguard.enterprise_security import EnterpriseSecurityStore
from railguard.models.schemas import ContractDocument
from railguard.security import AccessDenied
from railguard.storage.postgres import PostgresContractRepository, migrate
from railguard.workflow.graph import build_review_graph
from railguard.workflow.postgres import migrate_checkpoints, open_postgres_checkpointer
from railguard.workflow.service import ReviewConflictError, ReviewService
from railguard.workflow.state import HumanReviewDecision


@pytest.fixture
def pg():
    dsn=os.environ.get('PETKIT_TEST_CONTRACT_DSN')
    if not dsn:pytest.skip('Independent PostgreSQL17 QA database required')
    from urllib.parse import urlsplit
    u=urlsplit(dsn);assert u.hostname in {'localhost','127.0.0.1'} and u.path.endswith('_qa')
    migrate(dsn);repo=PostgresContractRepository(dsn);repo.initialize()
    with repo._connect() as db:db.execute('TRUNCATE contracts,rg_users,rg_tokens,rg_owners,rg_grants,rg_reviews,rg_audit')
    return repo


def test_enterprise_identity_grants_revocation_and_pg_restart(pg):
    class IdP:
        revoked=False
        def resolve(self,token):
            if self.revoked:raise IdentityError()
            roles={'owner':'submitter','lawyer':'reviewer','other':'submitter'}
            return Identity(token,'t',(roles[token],),{})
    idp=IdP();security=EnterpriseSecurityStore(pg,SimpleNamespace(enterprise_tenant_id='t'),oidc=idp)
    owner=security.authenticate('owner');lawyer=security.authenticate('lawyer');other=security.authenticate('other')
    contract=ContractDocument(filename='contract.docx',full_text='Enterprise contract')
    security.save_contract(owner,contract)
    with pytest.raises(AccessDenied):security.authorize(other,contract.contract_id)
    with pytest.raises(AccessDenied):security.authorize(lawyer,contract.contract_id)
    security.grant(owner,contract.contract_id,lawyer.subject)
    security.authorize(lawyer,contract.contract_id,'decide')
    assert PostgresContractRepository(pg.dsn).get(contract.contract_id)==contract
    security.grant(owner,contract.contract_id,lawyer.subject,remove=True)
    with pytest.raises(AccessDenied):security.authorize(lawyer,contract.contract_id,'decide')
    idp.revoked=True
    with pytest.raises(AccessDenied):security.authorize(owner,contract.contract_id)
    assert 'access_token=' not in repr(owner)


def test_postgres_checkpoint_restarts_and_serializes_decisions(pg,monkeypatch):
    dsn=os.environ.get('PETKIT_TEST_CHECKPOINT_DSN')
    if not dsn:pytest.skip('Separate checkpoint QA database required')
    class Retriever:
        async def retrieve(self,request):return []
    async def scenario():
        await migrate_checkpoints(dsn)
        rid='qa-'+uuid.uuid4().hex
        async with open_postgres_checkpointer(dsn) as saver:
            service=ReviewService(build_review_graph(retriever=Retriever(),checkpointer=saver,require_human_review=True),postgres_lock_dsn=pg.dsn)
            result=await service.start_review(ContractDocument(filename='agreement.docx',full_text='Software services agreement'),review_id=rid)
            assert result.status=='awaiting_human'
        async with open_postgres_checkpointer(dsn) as saver:
            graph=build_review_graph(retriever=Retriever(),checkpointer=saver,require_human_review=True)
            replicas=[ReviewService(graph,postgres_lock_dsn=pg.dsn),ReviewService(graph,postgres_lock_dsn=pg.dsn)]
            assert (await replicas[0].get_review(rid)).status=='awaiting_human'
            decision=HumanReviewDecision(action='approve',reviewer='lawyer')
            results=await asyncio.gather(*(s.submit_decision(review_id=rid,decision=decision) for s in replicas),return_exceptions=True)
            assert sum(isinstance(r,ReviewConflictError) for r in results)==1
            assert sum(getattr(r,'status',None)=='approved' for r in results)==1
            # Inject a crash after checkpoint commit but before business command acknowledgement.
            from railguard.workflow.decisions import Commands
            rid2='qa-'+uuid.uuid4().hex
            service=replicas[0]
            await service.start_review(ContractDocument(filename='agreement.docx',full_text='Software agreement'),review_id=rid2)
            complete=Commands.complete
            def crash(*args,**kwargs):raise RuntimeError('injected acknowledgement failure')
            monkeypatch.setattr(Commands,'complete',crash)
            with pytest.raises(RuntimeError):await service.submit_decision(review_id=rid2,decision=decision,decision_id='command')
            assert (await service.get_review(rid2)).status=='approved'
            monkeypatch.setattr(Commands,'complete',complete)
            recovered=await service.submit_decision(review_id=rid2,decision=decision,decision_id='command')
            assert recovered.status=='approved'
            assert (await service.submit_decision(review_id=rid2,decision=decision,decision_id='command')).status=='approved'
            with pytest.raises(ReviewConflictError):
                await service.submit_decision(review_id=rid2,decision=decision.model_copy(update={'comment':'different'}),decision_id='command')
            with pytest.raises(ReviewConflictError):
                await service.submit_decision(review_id=rid2,decision=decision,decision_id='unrelated-new-command')
    loop=asyncio.SelectorEventLoop() if sys.platform=='win32' else asyncio.new_event_loop()
    try:loop.run_until_complete(scenario())
    finally:loop.close()


def test_production_settings_reject_local_demo():
    with pytest.raises(ValueError):Settings(_env_file=None,app_env='production')
