"""Local persistent identities, contract grants and metadata-only audit.

This is a single-instance application authorization boundary, not enterprise SSO.
Credentials are stored as hashes; plaintext issuance is a separate local CLI step.
"""

import hashlib
import secrets
import time
from dataclasses import dataclass
from datetime import UTC, datetime

ROLES = {"submitter", "reviewer", "auditor"}


class AccessDenied(Exception):
    def __init__(self, status: int, detail: str):
        self.status = status
        self.detail = detail


@dataclass(frozen=True)
class Principal:
    subject: str
    role: str
    fingerprint: str = ""
    revision: int = 0


class SecurityStore:
    def __init__(self, repository, enabled: bool):
        self.repository = repository
        self.enabled = enabled

    def initialize(self):
        if not self.enabled:
            with self.repository._connect() as conn:
                existing = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='rg_users'").fetchone()
                if existing and conn.execute("SELECT 1 FROM rg_users LIMIT 1").fetchone():
                    raise ValueError("Personal identities exist; use token mode or a separate demo directory")
            return
        with self.repository._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS rg_users (
                    subject TEXT PRIMARY KEY, role TEXT NOT NULL,
                    active INTEGER NOT NULL, revision INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS rg_tokens (
                    fingerprint TEXT PRIMARY KEY, subject TEXT NOT NULL,
                    revision INTEGER NOT NULL, expires REAL NOT NULL, revoked INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS rg_owners (
                    contract_id TEXT PRIMARY KEY, subject TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS rg_grants (
                    contract_id TEXT NOT NULL, subject TEXT NOT NULL,
                    PRIMARY KEY(contract_id,subject));
                CREATE TABLE IF NOT EXISTS rg_reviews (
                    review_id TEXT PRIMARY KEY, contract_id TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS rg_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp REAL NOT NULL,
                    subject TEXT NOT NULL, action TEXT NOT NULL,
                    resource TEXT NOT NULL, outcome TEXT NOT NULL,
                    target TEXT NOT NULL DEFAULT '');
            """)
            if "target" not in {row["name"] for row in conn.execute("PRAGMA table_info(rg_audit)")}:
                conn.execute("ALTER TABLE rg_audit ADD COLUMN target TEXT NOT NULL DEFAULT ''")

    @staticmethod
    def _event(conn, subject, action, resource, outcome, target=""):
        conn.execute("INSERT INTO rg_audit(timestamp,subject,action,resource,outcome,target) VALUES(?,?,?,?,?,?)",
                     (time.time(), subject, action, resource, outcome, target))

    def record(self, principal, action, resource, outcome):
        if self.enabled:
            with self.repository._connect() as conn:
                self._event(conn, principal.subject if principal else "anonymous", action, resource, outcome)

    @staticmethod
    def _valid(conn, principal):
        row = conn.execute("""SELECT u.role,u.revision FROM rg_users u JOIN rg_tokens t
            ON t.subject=u.subject WHERE t.fingerprint=? AND u.subject=? AND u.active=1
            AND t.revoked=0 AND t.expires>? AND t.revision=u.revision""",
            (principal.fingerprint, principal.subject, time.time())).fetchone()
        return row is not None and row["role"] == principal.role and row["revision"] == principal.revision

    def authenticate(self, token):
        if not self.enabled:
            return Principal("local-demo", "submitter")
        # Bound hashing work and do not log supplied token material.
        fingerprint = hashlib.sha256(token.encode()).hexdigest() if 1 <= len(token) <= 512 else ""
        with self.repository._connect() as conn:
            row = conn.execute("""SELECT u.subject,u.role,u.revision FROM rg_users u JOIN rg_tokens t
                ON t.subject=u.subject WHERE t.fingerprint=? AND u.active=1 AND t.revoked=0
                AND t.expires>? AND t.revision=u.revision""", (fingerprint, time.time())).fetchone()
            if row is None:
                self._event(conn, "anonymous", "authenticate", "", "denied")
        if row is None:
            raise AccessDenied(401, "Invalid or expired credentials")
        return Principal(row["subject"], row["role"], fingerprint, row["revision"])

    def verify(self, principal):
        if not self.enabled:
            return
        with self.repository._connect() as conn:
            valid = self._valid(conn, principal)
        if not valid:
            self.record(principal, "verify", "", "denied")
            raise AccessDenied(401, "Invalid or expired credentials")

    def require_role(self, principal, role):
        if not self.enabled:
            return
        self.verify(principal)
        if principal.role != role:
            self.record(principal, "role_check", role, "denied")
            raise AccessDenied(403, "Role does not permit this action")

    def upsert_user(self, subject, role, active=True):
        if role not in ROLES or not subject.strip() or len(subject) > 100:
            raise ValueError("Invalid identity")
        with self.repository._connect() as conn:
            conn.execute("""INSERT INTO rg_users VALUES(?,?,?,1) ON CONFLICT(subject) DO UPDATE SET
                role=excluded.role,active=excluded.active,revision=rg_users.revision+1""",
                (subject, role, int(active)))
            # A changed/reenabled identity needs explicit contract grants again.
            conn.execute("DELETE FROM rg_grants WHERE subject=?", (subject,))
            self._event(conn, "local-operator", "user_update", subject, "completed")

    def issue_token(self, subject, ttl_seconds=28800):
        if not 1 <= ttl_seconds <= 604800:
            raise ValueError("Token lifetime must be between 1 second and 7 days")
        token = secrets.token_urlsafe(32)
        with self.repository._connect() as conn:
            user = conn.execute("SELECT revision FROM rg_users WHERE subject=? AND active=1", (subject,)).fetchone()
            if user is None:
                raise ValueError("Active user not found")
            conn.execute("UPDATE rg_tokens SET revoked=1 WHERE subject=?", (subject,))
            conn.execute("INSERT INTO rg_tokens VALUES(?,?,?,?,0)", (
                hashlib.sha256(token.encode()).hexdigest(), subject, user["revision"], time.time() + ttl_seconds,
            ))
            self._event(conn, "local-operator", "token_rotate", subject, "completed")
        return token

    def revoke_tokens(self, subject):
        with self.repository._connect() as conn:
            conn.execute("UPDATE rg_tokens SET revoked=1 WHERE subject=?", (subject,))
            self._event(conn, "local-operator", "token_revoke", subject, "completed")

    @staticmethod
    def _allowed(conn, principal, contract_id, action):
        owner = conn.execute("SELECT subject FROM rg_owners WHERE contract_id=?", (contract_id,)).fetchone()
        if owner is None:
            return False
        if action in {"start", "grant"}:
            return principal.role == "submitter" and owner["subject"] == principal.subject
        if action == "decide":
            return principal.role == "reviewer" and owner["subject"] != principal.subject and conn.execute(
                "SELECT 1 FROM rg_grants WHERE contract_id=? AND subject=?", (contract_id, principal.subject),
            ).fetchone() is not None
        return ((principal.role == "submitter" and owner["subject"] == principal.subject)
                or (principal.role == "reviewer" and conn.execute(
                    "SELECT 1 FROM rg_grants WHERE contract_id=? AND subject=?", (contract_id, principal.subject),
                ).fetchone() is not None))

    def authorize(self, principal, contract_id, action="read"):
        if not self.enabled:
            return
        with self.repository._connect() as conn:
            valid = self._valid(conn, principal)
            allowed = valid and self._allowed(conn, principal, contract_id, action)
            self._event(conn, principal.subject, action, contract_id, "authorized" if allowed else "denied")
        if not valid:
            raise AccessDenied(401, "Invalid or expired credentials")
        if not allowed:
            raise AccessDenied(404, "Resource not found or unavailable")

    def save_contract(self, principal, contract):
        if not self.enabled:
            self.repository.save(contract)
            return
        self.require_role(principal, "submitter")
        with self.repository._connect() as conn:
            if not self._valid(conn, principal):
                raise AccessDenied(401, "Invalid or expired credentials")
            conn.execute("INSERT INTO contracts VALUES(?,?,?)", (
                contract.contract_id, contract.model_dump_json(), datetime.now(UTC).isoformat(),
            ))
            conn.execute("INSERT INTO rg_owners VALUES(?,?)", (contract.contract_id, principal.subject))
            self._event(conn, principal.subject, "upload", contract.contract_id, "completed")

    def grant(self, principal, contract_id, subject, remove=False):
        self.authorize(principal, contract_id, "grant")
        with self.repository._connect() as conn:
            if not self._valid(conn, principal) or not self._allowed(conn, principal, contract_id, "grant"):
                raise AccessDenied(404, "Resource not found or unavailable")
            target = conn.execute("SELECT 1 FROM rg_users WHERE subject=? AND role='reviewer' AND active=1", (subject,)).fetchone()
            if target is None and not remove:
                raise AccessDenied(404, "Reviewer unavailable")
            if remove:
                conn.execute("DELETE FROM rg_grants WHERE contract_id=? AND subject=?", (contract_id, subject))
            else:
                conn.execute("INSERT OR IGNORE INTO rg_grants VALUES(?,?)", (contract_id, subject))
            self._event(conn, principal.subject, "revoke_grant" if remove else "grant", contract_id, "completed", subject)

    def reserve_review(self, principal, review_id, contract_id):
        if not self.enabled:
            return
        self.authorize(principal, contract_id, "start")
        with self.repository._connect() as conn:
            if not self._valid(conn, principal) or not self._allowed(conn, principal, contract_id, "start"):
                raise AccessDenied(404, "Resource not found or unavailable")
            conn.execute("INSERT INTO rg_reviews VALUES(?,?)", (review_id, contract_id))
            self._event(conn, principal.subject, "start_review", review_id, "intent")

    def review_contract(self, principal, review_id, action="read"):
        if not self.enabled:
            return None
        self.verify(principal)
        with self.repository._connect() as conn:
            row = conn.execute("SELECT contract_id FROM rg_reviews WHERE review_id=?", (review_id,)).fetchone()
        if row is None:
            self.record(principal, action, review_id, "denied")
            raise AccessDenied(404, "Resource not found or unavailable")
        self.authorize(principal, row["contract_id"], action)
        return row["contract_id"]

    def visible_reviews(self, principal, limit=100, offset=0):
        self.verify(principal)
        with self.repository._connect() as conn:
            rows = conn.execute("""SELECT r.review_id FROM rg_reviews r JOIN rg_owners o
                ON o.contract_id=r.contract_id WHERE (?='submitter' AND o.subject=?)
                OR (?='reviewer' AND EXISTS(SELECT 1 FROM rg_grants g
                    WHERE g.contract_id=r.contract_id AND g.subject=?))
                ORDER BY r.review_id LIMIT ? OFFSET ?""",
                (principal.role, principal.subject, principal.role, principal.subject, limit, offset)).fetchall()
            return [row["review_id"] for row in rows]

    def audit_rows(self, principal, limit=100):
        self.require_role(principal, "auditor")
        with self.repository._connect() as conn:
            return [dict(row) for row in conn.execute("SELECT * FROM rg_audit ORDER BY id DESC LIMIT ?", (limit,))]
