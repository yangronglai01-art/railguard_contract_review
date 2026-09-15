"""应用健康检查接口测试。"""

from fastapi.testclient import TestClient

from railguard.api.main import create_app
from railguard.config import Settings


def test_health_check() -> None:
    """验证健康接口返回显式注入的隔离测试配置。"""
    # 测试不读取项目.env，避免本机真实模型配置影响断言。
    settings = Settings(
        _env_file=None,
        app_name="RailGuard AI",
        app_env="development",
        app_version="0.1.0",
        model_provider="mock",
        model_name="mock-contract-reviewer",
        rag_mode="mock",
    )
    application = create_app(settings=settings)
    client = TestClient(application)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "RailGuard AI",
        "version": "0.1.0",
        "environment": "development",
        "model_provider": "mock",
        "rag_mode": "mock",
    }
