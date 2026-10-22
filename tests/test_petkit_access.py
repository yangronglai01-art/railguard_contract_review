"""Real API authorization checks with separate persistent personal identities."""

import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

from railguard.petkit_demo import case_docx, create_demo_app, load_cases


def headers(token):
    return {"Authorization": "Bearer " + token}


@pytest.fixture
def secured(tmp_path):
    app = create_demo_app(tmp_path, secure=True)
    with TestClient(app) as client:
        store = app.state.security
        tokens = {}
        for name, role in [("buyer-a", "submitter"), ("buyer-b", "submitter"),
                           ("legal-a", "reviewer"), ("legal-b", "reviewer"), ("audit", "auditor")]:
            store.upsert_user(name, role)
            tokens[name] = store.issue_token(name)
        yield client, store, tokens


def upload(secured, owner="buyer-a"):
    client, _, tokens = secured
    response = client.post("/contracts", headers=headers(tokens[owner]), files={"file": (
        "private-contract.docx", case_docx(load_cases()[0]),
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )})
    assert response.status_code == 201, response.text
    return response.json()["contract_id"]


def start(secured, contract_id):
    client, _, tokens = secured
    response = client.post(f"/contracts/{contract_id}/reviews", headers=headers(tokens["buyer-a"]))
    assert response.status_code == 201, response.text
    return response.json()["review_id"]


def grant(secured, contract_id):
    client, _, tokens = secured
    response = client.put(f"/contracts/{contract_id}/reviewers/legal-a", headers=headers(tokens["buyer-a"]))
    assert response.status_code == 200


def decision():
    return {"action": "request_changes", "reviewer": "forged-admin", "comment": "修订模拟合同", "finding_decisions": {}}


@pytest.mark.parametrize("path", ["/contracts/missing", "/reviews/missing", "/reviews", "/monitoring/summary", "/security/me", "/security/audit"])
def test_protected_reads_require_valid_credentials(secured, path):
    client, _, _ = secured
    assert client.get(path).status_code == 401
    assert client.get(path, headers=headers("not-a-real-token")).status_code == 401


def test_contract_and_review_isolation_including_list(secured):
    client, _, tokens = secured
    contract_id = upload(secured)
    review_id = start(secured, contract_id)
    for subject in ["buyer-b", "legal-a", "legal-b", "audit"]:
        auth = headers(tokens[subject])
        for path in [f"/contracts/{contract_id}", f"/reviews/{review_id}"]:
            response = client.get(path, headers=auth)
            assert response.status_code == 404 and "private-contract" not in response.text
        assert client.get("/reviews", headers=auth).json() == []
    response = client.post(f"/contracts/{contract_id}/reviews", headers=headers(tokens["buyer-b"]))
    assert response.status_code == 404


def test_only_assigned_reviewer_can_decide_and_identity_is_server_owned(secured):
    client, _, tokens = secured
    contract_id = upload(secured)
    review_id = start(secured, contract_id)
    assert client.post(f"/reviews/{review_id}/decision", headers=headers(tokens["buyer-a"]), json=decision()).status_code == 403
    assert client.post(f"/reviews/{review_id}/decision", headers=headers(tokens["legal-b"]), json=decision()).status_code == 404
    grant(secured, contract_id)
    assert client.get(f"/contracts/{contract_id}", headers=headers(tokens["legal-a"])).status_code == 200
    assert len(client.get("/reviews", headers=headers(tokens["legal-a"])).json()) == 1
    result = client.post(f"/reviews/{review_id}/decision", headers=headers(tokens["legal-a"]), json=decision())
    assert result.status_code == 200
    assert result.json()["human_decision"]["reviewer"] == "legal-a"
    assert result.json()["status"] == "changes_requested"


@pytest.mark.parametrize("subject", ["legal-a", "audit"])
def test_non_submitters_cannot_upload(secured, subject):
    client, _, tokens = secured
    result = client.post("/contracts", headers=headers(tokens[subject]), files={"file": (
        "sample.docx", case_docx(load_cases()[0]), "application/octet-stream",
    )})
    assert result.status_code == 403


def test_grants_are_owner_controlled_and_revocation_applies_to_old_tasks(secured):
    client, _, tokens = secured
    contract_id = upload(secured)
    review_id = start(secured, contract_id)
    assert client.put(f"/contracts/{contract_id}/reviewers/legal-a", headers=headers(tokens["buyer-b"])).status_code == 404
    assert client.put(f"/contracts/{contract_id}/reviewers/buyer-a", headers=headers(tokens["buyer-a"])).status_code == 404
    grant(secured, contract_id)
    assert client.delete(f"/contracts/{contract_id}/reviewers/legal-a", headers=headers(tokens["legal-a"])).status_code == 404
    assert client.delete(f"/contracts/{contract_id}/reviewers/legal-a", headers=headers(tokens["buyer-a"])).status_code == 200
    assert client.get(f"/reviews/{review_id}", headers=headers(tokens["legal-a"])).status_code == 404
    assert client.get("/reviews", headers=headers(tokens["legal-a"])).json() == []
    assert client.post(f"/reviews/{review_id}/decision", headers=headers(tokens["legal-a"]), json=decision()).status_code == 404
    assert client.get(f"/reviews/{review_id}", headers=headers(tokens["buyer-a"])).json()["status"] == "awaiting_human"


@pytest.mark.parametrize("operation", ["rotate", "revoke", "expire", "disable", "change_role"])
def test_changed_credentials_cannot_read_old_contracts(secured, operation):
    client, store, tokens = secured
    contract_id = upload(secured)
    if operation == "rotate":
        replacement = store.issue_token("buyer-a")
        assert client.get(f"/contracts/{contract_id}", headers=headers(replacement)).status_code == 200
    elif operation == "revoke":
        store.revoke_tokens("buyer-a")
    elif operation == "expire":
        with store.repository._connect() as conn:
            conn.execute("UPDATE rg_tokens SET expires=0 WHERE subject='buyer-a'")
    else:
        store.upsert_user("buyer-a", "reviewer" if operation == "change_role" else "submitter", active=operation != "disable")
    assert client.get(f"/contracts/{contract_id}", headers=headers(tokens["buyer-a"])).status_code == 401


def test_disabled_reviewer_does_not_regain_grants_when_reenabled(secured):
    client, store, _ = secured
    contract_id = upload(secured)
    grant(secured, contract_id)
    store.upsert_user("legal-a", "reviewer", active=False)
    store.upsert_user("legal-a", "reviewer")
    replacement = store.issue_token("legal-a")
    assert client.get(f"/contracts/{contract_id}", headers=headers(replacement)).status_code == 404


def test_audit_is_role_restricted_and_contains_no_credentials_or_contract_body(secured):
    client, _, tokens = secured
    contract_id = upload(secured)
    start(secured, contract_id)
    assert client.get("/security/audit", headers=headers(tokens["buyer-a"])).status_code == 403
    assert client.get("/monitoring/summary", headers=headers(tokens["buyer-a"])).status_code == 403
    assert client.get("/monitoring/summary", headers=headers(tokens["audit"])).status_code == 200
    response = client.get("/security/audit", headers=headers(tokens["audit"]))
    assert response.status_code == 200 and response.json()["items"]
    assert all(set(item) == {"id", "timestamp", "subject", "action", "resource", "outcome", "target"} for item in response.json()["items"])
    assert "private-contract" not in response.text and "宠物影像" not in response.text
    assert all(token not in response.text for token in tokens.values())


def test_revoke_during_checkpoint_load_blocks_decision_before_resume(secured, monkeypatch):
    client, store, tokens = secured
    contract_id = upload(secured)
    review_id = start(secured, contract_id)
    grant(secured, contract_id)
    service = client.app.state.review_service
    original = service._load_state_snapshot
    calls = 0
    async def revoke_before_authorization(identifier):
        nonlocal calls
        value = await original(identifier)
        calls += 1
        if calls == 2:
            store.revoke_tokens("legal-a")
        return value
    monkeypatch.setattr(service, "_load_state_snapshot", revoke_before_authorization)
    response = client.post(f"/reviews/{review_id}/decision", headers=headers(tokens["legal-a"]), json=decision())
    assert response.status_code == 401
    assert client.get(f"/reviews/{review_id}", headers=headers(tokens["buyer-a"])).json()["status"] == "awaiting_human"


def test_revoke_during_read_prevents_result_release(secured, monkeypatch):
    client, store, tokens = secured
    contract_id = upload(secured)
    review_id = start(secured, contract_id)
    grant(secured, contract_id)
    service = client.app.state.review_service
    original = service.get_review
    async def revoked_read(identifier):
        value = await original(identifier)
        store.grant(store.authenticate(tokens["buyer-a"]), contract_id, "legal-a", remove=True)
        return value
    monkeypatch.setattr(service, "get_review", revoked_read)
    response = client.get(f"/reviews/{review_id}", headers=headers(tokens["legal-a"]))
    assert response.status_code == 404 and "private-contract" not in response.text


def test_audit_failure_prevents_resume_without_exposing_internal_error(secured, monkeypatch):
    client, store, tokens = secured
    contract_id = upload(secured)
    review_id = start(secured, contract_id)
    grant(secured, contract_id)
    def fail(*args, **kwargs):
        raise sqlite3.OperationalError("PRIVATE-LOCATION-AND-SECRET")
    monkeypatch.setattr(store, "record", fail)
    response = client.post(f"/reviews/{review_id}/decision", headers=headers(tokens["legal-a"]), json=decision())
    assert response.status_code == 503 and "PRIVATE" not in response.text
    assert client.get(f"/reviews/{review_id}", headers=headers(tokens["buyer-a"])).json()["status"] == "awaiting_human"


def test_upload_and_ownership_roll_back_if_audit_write_fails(secured, monkeypatch):
    client, store, tokens = secured
    def fail(*args, **kwargs):
        raise sqlite3.OperationalError("PRIVATE-ERROR")
    monkeypatch.setattr(store, "_event", fail)
    result = client.post("/contracts", headers=headers(tokens["buyer-a"]), files={"file": (
        "private.docx", case_docx(load_cases()[0]), "application/octet-stream",
    )})
    assert result.status_code == 503 and "PRIVATE" not in result.text
    with store.repository._connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM contracts").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM rg_owners").fetchone()[0] == 0


def test_persistent_permissions_and_review_survive_restart(tmp_path):
    with TestClient(create_demo_app(tmp_path, secure=True)) as client:
        store = client.app.state.security
        store.upsert_user("buyer-a", "submitter")
        store.upsert_user("legal-a", "reviewer")
        tokens = {subject: store.issue_token(subject) for subject in ["buyer-a", "legal-a"]}
        context = client, store, tokens
        contract_id = upload(context)
        review_id = start(context, contract_id)
        grant(context, contract_id)
    with TestClient(create_demo_app(tmp_path, secure=True)) as restarted:
        response = restarted.post(f"/reviews/{review_id}/decision", headers=headers(tokens["legal-a"]), json=decision())
        assert response.status_code == 200 and response.json()["human_decision"]["reviewer"] == "legal-a"


def test_token_material_is_not_stored_in_database(secured):
    _, store, tokens = secured
    with store.repository._connect() as conn:
        raw = json.dumps([dict(row) for row in conn.execute("SELECT * FROM rg_tokens")])
    assert all(token not in raw for token in tokens.values())


def test_personal_database_cannot_silently_downgrade_to_demo(tmp_path):
    with TestClient(create_demo_app(tmp_path, secure=True)) as client:
        client.app.state.security.upsert_user("buyer", "submitter")
    with pytest.raises(ValueError, match="use token mode"), TestClient(create_demo_app(tmp_path)):
        pass


@pytest.mark.parametrize("method,path", [
    ("post", "/contracts/missing/reviews"), ("put", "/contracts/missing/reviewers/legal-a"),
    ("delete", "/contracts/missing/reviewers/legal-a"), ("post", "/reviews/missing/decision"),
])
def test_mutations_require_authentication(secured, method, path):
    client, _, _ = secured
    kwargs = {"json": decision()} if path.endswith("decision") else {}
    assert getattr(client, method)(path, **kwargs).status_code == 401


def test_personal_review_pagination_is_bounded(secured):
    client, _, tokens = secured
    contract_id = upload(secured)
    ids = {start(secured, contract_id) for _ in range(2)}
    auth = headers(tokens["buyer-a"])
    first = client.get("/reviews?limit=1&offset=0", headers=auth).json()
    second = client.get("/reviews?limit=1&offset=1", headers=auth).json()
    assert {first[0]["review_id"], second[0]["review_id"]} == ids
    assert client.get("/reviews?limit=201", headers=auth).status_code == 422
    assert client.get("/reviews?offset=-1", headers=auth).status_code == 422


def test_cli_credentials_are_private_and_repeat_provision_does_not_rotate(tmp_path, monkeypatch, capsys):
    import sys

    from railguard.identities import main

    monkeypatch.setattr(sys, "argv", ["identities", "provision", "--data-dir", str(tmp_path)])
    main()
    path = tmp_path / "credentials.json"
    before = path.read_bytes()
    values = json.loads(before)
    printed = capsys.readouterr().out
    assert all(identity["token"] not in printed for identity in values.values())
    with pytest.raises(SystemExit):
        main()
    assert path.read_bytes() == before
    monkeypatch.setattr(sys, "argv", ["identities", "rotate", "--data-dir", str(tmp_path), "--subject", "buyer-01"])
    main()
    after = json.loads(path.read_bytes())
    assert after["buyer-01"]["token"] != values["buyer-01"]["token"]
    assert after["buyer-02"] == values["buyer-02"]
