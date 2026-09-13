"""RailGuard应用配置。"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """从默认值和环境变量中读取应用配置。"""

    # 应用基本信息。
    app_name: str = "RailGuard AI"
    app_env: str = "development"
    app_version: str = "0.1.0"

    # 大模型配置。当前默认使用Mock模式。
    model_provider: str = "mock"
    model_name: str = "mock-contract-reviewer"
    openai_api_key: str | None = None

    # 外部RAG服务配置。
    rag_mode: str = "mock"
    rag_base_url: str = "http://localhost:8001"
    rag_api_key: str | None = None
    rag_timeout_seconds: float = 10.0

    # SQLite数据库文件路径。
    # 使用明确的Path，避免把sqlite:///误当成Windows文件路径。
    database_path: Path = Path("data/runtime/railguard.db")

    # 允许从项目根目录的.env文件加载配置。
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """创建并缓存应用配置。

    配置在进程内只读取一次，避免每个请求重复读取.env文件。
    """
    return Settings()