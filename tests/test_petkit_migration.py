"""Synthetic Petkit contracts: scope, citations, human gate and durable recovery."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from railguard.agents.prompts import build_system_prompt
from railguard.business import PETKIT_RULE_VERSION, profiles_for_business
from railguard.config import Settings
from railguard.petkit_demo import case_docx, create_demo_app, demo_settings, load_cases


def test_petkit_settings_isolate_host_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "host-key-must-not-be-used")
    monkeypatch.setenv("RAG_MOCK_CORPUS_PATH", "data/demo/rag-corpus-baus-v1.json")
    monkeypatch.setenv("DATABASE_PATH", "old-contracts.db")
    settings = demo_settings(tmp_path)
    assert settings.business_profile == "petkit"
    assert settings.model_provider == "mock" and settings.openai_api_key is None
    assert settings.database_path == tmp_path / "contracts.db"
    assert settings.checkpoint_path == tmp_path / "checkpoints.sqlite3"
    assert settings.rag_mock_corpus_path.name == "rag-corpus-petkit-v1.json"


def test_petkit_default_paths_do_not_reuse_legacy_storage():
    settings = Settings(_env_file=None, business_profile="petkit")
    assert settings.database_path == Path("data/runtime/petkit/contracts.db")
    assert settings.checkpoint_path == Path("data/runtime/petkit/checkpoints.sqlite3")
    with pytest.raises(ValueError, match="frozen legacy"):
        Settings(_env_file=None, business_profile="petkit", rag_mock_corpus_path="data/demo/rag-corpus.json")


def test_prompt_profiles_do_not_mutate_frozen_legacy():
    legacy = profiles_for_business("baus")
    petkit = profiles_for_business("petkit")
    assert "宠物影像" in build_system_prompt(petkit[2])
    assert "标准SaaS" in build_system_prompt(petkit[1])
    assert "小佩" not in build_system_prompt(legacy[2])
    assert profiles_for_business("baus") == legacy
    with pytest.raises(ValueError):
        profiles_for_business("unknown")


def upload_review(client, case):
    uploaded = client.post("/contracts", files={"file": (
        case["id"] + ".docx", case_docx(case),
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )})
    assert uploaded.status_code == 201, uploaded.text
    contract = uploaded.json()
    for clause in contract["clauses"]:
        assert contract["full_text"][clause["start_offset"]:clause["end_offset"]] == clause["text"]
    result = client.post(f"/contracts/{contract['contract_id']}/reviews")
    assert result.status_code == 201, result.text
    return result.json()


@pytest.mark.parametrize("case", load_cases(), ids=lambda case: case["id"])
def test_petkit_contracts_require_human_even_without_findings(tmp_path, case):
    with TestClient(create_demo_app(tmp_path)) as client:
        profile = client.get("/profile").json()
        assert profile["business_profile"] == "petkit" and profile["synthetic"]
        assert profile["rule_version"] == PETKIT_RULE_VERSION
        waiting = upload_review(client, case)
        assert waiting["status"] == "awaiting_human"
        assert {finding["category"] for finding in waiting["findings"]} == set(case["expected_categories"])
        assert len(waiting["findings"]) == len(case["expected_categories"])
        for finding in waiting["findings"]:
            assert finding["citation_status"] == "source_matched"
            assert finding["evidence_ids"]
        for evidence in waiting["evidence"]:
            assert evidence["metadata"]["business_profile"] == "petkit"
            assert evidence["metadata"]["version"] == PETKIT_RULE_VERSION
        response = client.post(f"/reviews/{waiting['review_id']}/decision", json={
            "action": "request_changes" if waiting["findings"] else "approve",
            "reviewer": "模拟法务复核人", "finding_decisions": {}, "comment": "合成场景验证",
        })
        assert response.status_code == 200, response.text
        assert response.json()["status"] == ("changes_requested" if waiting["findings"] else "approved")


def test_petkit_resume_after_application_restart(tmp_path):
    with TestClient(create_demo_app(tmp_path)) as client:
        waiting = upload_review(client, load_cases()[0])
    with TestClient(create_demo_app(tmp_path)) as restarted:
        current = restarted.get(f"/reviews/{waiting['review_id']}")
        assert current.status_code == 200 and current.json()["status"] == "awaiting_human"
        resumed = restarted.post(f"/reviews/{waiting['review_id']}/decision", json={
            "action": "request_changes", "reviewer": "模拟采购", "comment": "补齐保护后再审",
            "finding_decisions": {item["finding_id"]: "accept" for item in waiting["findings"]},
        })
        assert resumed.status_code == 200 and resumed.json()["status"] == "changes_requested"


def test_petkit_rejects_mixed_corpus(tmp_path):
    from railguard.api.main import create_app

    settings = demo_settings(tmp_path)
    body = json.loads(settings.rag_mock_corpus_path.read_text(encoding="utf-8"))
    body["hits"][0]["metadata"]["business_profile"] = "baus"
    bad = tmp_path / "renamed-corpus.json"
    bad.write_text(json.dumps(body), encoding="utf-8")
    settings.rag_mock_corpus_path = bad
    with pytest.raises(ValueError, match="versioned synthetic"), TestClient(create_app(settings=settings)):
        pass


@pytest.mark.parametrize("provider", ["openai", "deepseek"])
def test_api_passes_petkit_profile_to_model_factory(tmp_path, monkeypatch, provider):
    import railguard.api.main as main_module
    from railguard.petkit_rules import create_petkit_rule_analyzers

    captured = {}
    def fake_factory(**kwargs):
        captured.update(kwargs)
        return create_petkit_rule_analyzers()
    monkeypatch.setattr(main_module, f"create_{provider}_risk_analyzers", fake_factory)
    settings = demo_settings(tmp_path)
    settings.model_provider = provider
    settings.openai_api_key = "unit-test-placeholder"
    settings.deepseek_api_key = "unit-test-placeholder"
    with TestClient(main_module.create_app(settings=settings)) as client:
        assert client.get("/profile").json()["business_profile"] == "petkit"
    assert captured["business_profile"] == "petkit"


async def test_llm_analyzers_receive_versioned_petkit_context():
    from railguard.agents.provider import _create_analyzer_set
    from railguard.models.schemas import ContractDocument
    from railguard.parsers.clauses import split_clauses

    captured = []
    class Invoker:
        async def ainvoke(self, messages):
            captured.append(messages)
            return {"findings": []}
    analyzers = _create_analyzer_set(invoker=Invoker(), max_input_chars=120000, business_profile="petkit")
    text = "第一条 项目内容\n小佩模拟云服务采购。"
    contract = ContractDocument(filename="test.docx", full_text=text, clauses=split_clauses(text))
    for analyzer in [analyzers.commercial, analyzers.legal, analyzers.security]:
        await analyzer.analyze(contract=contract, evidence_by_clause={}, contract_evidence=[])
    assert len(captured) == 3
    assert all("小佩" in messages[0].content for messages in captured)
    assert "宠物影像" in captured[2][0].content
