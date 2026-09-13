from fastapi import FastAPI

from railguard.config import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Multi-agent contract review demo built with LangGraph.",
)


@app.get("/health", tags=["system"])
def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": settings.app_version,
        "environment": settings.app_env,
        "model_provider": settings.model_provider,
        "rag_mode": settings.rag_mode,
    }
