"""FastAPI启动阶段的风险Agent选择测试。"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import railguard.api.main as main_module
from railguard.agents.provider import RiskAnalyzerSet
from railguard.config import Settings
from railguard.workflow.analyzers import (
    DemoCommercialRiskAnalyzer,
    DemoLegalRiskAnalyzer,
    DemoSecurityRiskAnalyzer,
)


def create_settings(
    *,
    tmp_path: Path,
    model_provider: str,
    model_name: str,
    openai_api_key: str | None,
) -> Settings:
    """创建使用临时数据库的应用测试配置。"""
    return Settings(
        _env_file=None,
        model_provider=model_provider,
        model_name=model_name,
        openai_api_key=openai_api_key,
        openai_base_url=None,
        database_path=tmp_path / "contracts.db",
        checkpoint_path=tmp_path / "checkpoints.sqlite3",
        rag_mode="mock",
        rag_mock_corpus_path=Path(
            "data/demo/rag-corpus.json"
        ),
    )


def test_mock_mode_keeps_deterministic_analyzers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证Mock模式不会创建任何外部模型客户端。"""

    def fail_if_model_factory_is_called(
        **kwargs: object,
    ) -> RiskAnalyzerSet:
        """模型工厂被意外调用时立即使测试失败。"""
        del kwargs

        raise AssertionError(
            "model factory must not run in mock mode"
        )

    monkeypatch.setattr(
        main_module,
        "create_openai_risk_analyzers",
        fail_if_model_factory_is_called,
    )
    settings = create_settings(
        tmp_path=tmp_path,
        model_provider="mock",
        model_name="mock-contract-reviewer",
        openai_api_key=None,
    )
    application = main_module.create_app(
        settings=settings,
    )

    with TestClient(application) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_provider"] == "mock"


def test_openai_mode_injects_structured_model_analyzers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证OpenAI模式使用全部配置创建三个模型Agent。"""
    captured: dict[str, object] = {}

    def fake_model_factory(
        *,
        model_name: str,
        api_key: str,
        base_url: str | None,
        timeout_seconds: float,
        max_retries: int,
        max_input_chars: int,
    ) -> RiskAnalyzerSet:
        """记录工厂参数并返回不会访问网络的三个Agent。"""
        captured.update(
            {
                "model_name": model_name,
                "api_key": api_key,
                "base_url": base_url,
                "timeout_seconds": timeout_seconds,
                "max_retries": max_retries,
                "max_input_chars": max_input_chars,
            }
        )

        # 确定性Agent满足相同的RiskAnalyzer协议，
        # 可以用于验证依赖注入而不访问外部模型服务。
        return RiskAnalyzerSet(
            commercial=DemoCommercialRiskAnalyzer(),
            legal=DemoLegalRiskAnalyzer(),
            security=DemoSecurityRiskAnalyzer(),
        )

    monkeypatch.setattr(
        main_module,
        "create_openai_risk_analyzers",
        fake_model_factory,
    )
    settings = Settings(
        _env_file=None,
        model_provider="openai",
        model_name="contract-base-model",
        openai_api_key="test-api-key",
        openai_base_url="https://model.example/v1",
        model_timeout_seconds=42.0,
        model_max_retries=4,
        model_max_input_chars=90_000,
        database_path=tmp_path / "contracts.db",
        checkpoint_path=tmp_path / "checkpoints.sqlite3",
        rag_mode="mock",
        rag_mock_corpus_path=Path(
            "data/demo/rag-corpus.json"
        ),
    )
    application = main_module.create_app(
        settings=settings,
    )

    with TestClient(application) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_provider"] == "openai"
    assert captured == {
        "model_name": "contract-base-model",
        "api_key": "test-api-key",
        "base_url": "https://model.example/v1",
        "timeout_seconds": 42.0,
        "max_retries": 4,
        "max_input_chars": 90_000,
    }