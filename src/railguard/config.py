from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "RailGuard AI"
    app_env: str = "development"
    app_version: str = "0.1.0"

    model_provider: str = "mock"
    model_name: str = "mock-contract-reviewer"
    openai_api_key: str | None = None

    rag_mode: str = "mock"
    rag_base_url: str = "http://localhost:8001"
    rag_api_key: str | None = None
    rag_timeout_seconds: float = 10.0

    database_url: str = "sqlite:///./data/runtime/railguard.db"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
