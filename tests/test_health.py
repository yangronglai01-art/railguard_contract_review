from fastapi.testclient import TestClient

from railguard.api.main import app

client = TestClient(app)


def test_health_check() -> None:
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
