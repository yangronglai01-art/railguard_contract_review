"""Local operator CLI for short-lived personal credentials; never print tokens."""

import argparse
import json
import os
import tempfile
from pathlib import Path

from railguard.petkit_demo import demo_settings
from railguard.security import SecurityStore
from railguard.storage.contracts import ContractRepository


def write_credentials(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix="credentials-", suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            os.chmod(temporary, 0o600)
            json.dump(values, output, ensure_ascii=False, indent=2)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["provision", "rotate", "revoke", "disable", "status"])
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--credential-file", type=Path)
    parser.add_argument("--subject")
    args = parser.parse_args()
    settings = demo_settings(args.data_dir)
    repository = ContractRepository(settings.database_path)
    repository.initialize()
    store = SecurityStore(repository, True)
    store.initialize()
    credentials = args.credential_file or settings.database_path.parent / "credentials.json"
    if args.command == "provision":
        if credentials.exists():
            parser.error("Credential file exists; use rotate for one identity")
        with repository._connect() as conn:
            if conn.execute("SELECT COUNT(*) FROM rg_users").fetchone()[0]:
                parser.error("Users already exist; use rotate/status instead")
        values = {}
        for subject, role in [("buyer-01", "submitter"), ("buyer-02", "submitter"),
                              ("legal-01", "reviewer"), ("auditor-01", "auditor")]:
            store.upsert_user(subject, role)
            values[subject] = {"role": role, "token": store.issue_token(subject)}
        write_credentials(credentials, values)
        print(f"Private credentials saved to {credentials}; tokens expire in 8 hours.")
    elif args.command == "status":
        with repository._connect() as conn:
            users = [dict(row) for row in conn.execute("SELECT subject,role,active,revision FROM rg_users")]
        print(json.dumps(users, ensure_ascii=False))
    else:
        if not args.subject:
            parser.error("--subject is required")
        with repository._connect() as conn:
            user = conn.execute("SELECT role,active FROM rg_users WHERE subject=?", (args.subject,)).fetchone()
        if user is None:
            parser.error("User not found")
        if args.command == "rotate":
            values = json.loads(credentials.read_text(encoding="utf-8")) if credentials.exists() else {}
            values[args.subject] = {"role": user["role"], "token": store.issue_token(args.subject)}
            write_credentials(credentials, values)
            print(f"Credential rotated; read private file {credentials}")
        elif args.command == "revoke":
            store.revoke_tokens(args.subject)
            print("Credentials revoked.")
        else:
            store.upsert_user(args.subject, user["role"], active=False)
            print("Identity disabled.")


if __name__ == "__main__":
    main()
