# Production contract review path

APP_ENV=production requires Petkit, enterprise OIDC, PostgreSQL business/checkpoint
databases, local-chat through the private gateway and real HTTP evidence retrieval.
Mock providers, local-demo auth, SQLite and external contract generation are refused.
Dev fixtures remain available only through separate development paths.

Required configuration:

```
APP_ENV=production
BUSINESS_PROFILE=petkit
AUTH_MODE=oidc
RAILGUARD_DATABASE_URL=<PostgreSQL17 DSN with sslmode=verify-full>
RAILGUARD_CHECKPOINT_URL=<separate PostgreSQL17 checkpoint DSN with sslmode=verify-full>
ENTERPRISE_TENANT_ID=<approved tenant>
OIDC_INTROSPECTION_URL=<HTTPS introspection endpoint>
OIDC_ISSUER=<issuer>
OIDC_AUDIENCE=<contract API audience>
OIDC_CLIENT_ID=<resource server client>
OIDC_CLIENT_SECRET=<secret injected at runtime>
MODEL_PROVIDER=ai_platform
MODEL_NAME=local-chat
AI_PLATFORM_BASE_URL=<HTTPS private gateway /v1>
AI_PLATFORM_API_KEY=<contract service token without cloud privileges>
RAG_MODE=http
RAG_BASE_URL=<HTTPS isolated contract-domain RAG base ending />
RAG_API_KEY=<short-lived IdP service access token for contract_reader>
```

The evidence service identity also needs the configured tenant, departments, markets
and clearance claims. This batch accepts an injected service token; platform token
refresh/rotation must be arranged before unattended production use. Evidence hits
from another knowledge domain are rejected.

Apply explicit migrations using the migration role:

```
python -m railguard.storage.postgres
python -m railguard.workflow.postgres
```

Then start `uvicorn railguard.api.main:app --host 0.0.0.0 --port 8000 --no-access-log`
with DML-only runtime roles and the approved CA. AsyncPostgresSaver has exact business
type allowlisting; checkpoint pickle fallback is not enabled. PostgreSQL schema setup
is not silently run by application workers.

Stable IdP subject becomes the submitter/reviewer identity. One contract role per client
identity is required: submitter, reviewer or auditor. Reviewer first authenticates so
its identity is known, then the owner grants access to the contract. Object grants and
audit are held in the business PG database. Each authorization re-introspects the token;
the application never issues enterprise user passwords or local enterprise tokens.

Approval requests require `Idempotency-Key` (1..128 characters), scoped to review and
authenticated reviewer. Decision command fingerprint is persisted as pending before
checkpoint resume, then marked applied with the result. A crash between these steps
is recovered by comparing the stored human decision and finishing pending graph work.
Changed input under the same key returns conflict. A PostgreSQL advisory review lock
serializes competing resumes across workers; permissions are checked again under lock.

Remaining gates: immutable original files in Ceph, validated OCR/page provenance,
approved real legal rules/evidence, long-contract token-budget/segmentation evaluation,
React/OIDC portal, initial review durable task queue and retry operations, real provider
quality, token refresh, full failover/restore drills and data retention. This backend
change is not an assertion that all contract production acceptance work is complete.
