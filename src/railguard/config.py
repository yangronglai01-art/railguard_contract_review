"""RailGuard应用配置。"""

from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import (
    Field,
    model_validator,
)
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)


class Settings(BaseSettings):
    """从默认值和环境变量中读取应用配置。"""

    # 应用基本信息。
    app_name: str = "RailGuard AI"
    app_env: str = "development"
    app_version: str = "0.1.0"

    # 风险分析模式：
    # mock使用确定性规则，openai使用结构化大模型Agent。
    model_provider: Literal["mock", "openai"] = "mock"

    # 基础模型或后续微调模型的供应商模型名称。
    model_name: str = "mock-contract-reviewer"

    # OpenAI或OpenAI兼容服务的访问密钥。
    openai_api_key: str | None = None

    # 可选的OpenAI兼容接口地址。
    # 留空时使用SDK默认的OpenAI服务地址。
    openai_base_url: str | None = None

    # 单次模型请求的超时时间。
    model_timeout_seconds: float = Field(
        default=60.0,
        gt=0,
    )

    # 模型SDK遇到临时故障时的最大重试次数。
    model_max_retries: int = Field(
        default=2,
        ge=0,
    )

    # 系统提示词、合同和证据的最大总字符数。
    model_max_input_chars: int = Field(
        default=120_000,
        gt=0,
    )

    # 外部RAG服务配置。
    rag_mode: str = "mock"
    rag_base_url: str = "http://localhost:8001"
    rag_api_key: str | None = None
    rag_timeout_seconds: float = 10.0

    # 本地Mock模式使用的演示知识库文件。
    rag_mock_corpus_path: Path = Path(
        "data/demo/rag-corpus.json"
    )

    # SQLite数据库文件路径。
    # 使用明确的Path，避免把sqlite:///误当成Windows文件路径。
    database_path: Path = Path(
        "data/runtime/railguard.db"
    )

    # LangGraph审核流程checkpoint数据库路径。
    # 与合同业务数据库分开，便于独立迁移和故障排查。
    checkpoint_path: Path = Path(
        "data/runtime/review-checkpoints.sqlite3"
    )

    # 允许从项目根目录的.env文件加载配置。
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @model_validator(mode="after")
    def validate_model_configuration(self) -> Self:
        """验证模型名称和真实模型模式所需的访问密钥。"""
        if not self.model_name.strip():
            raise ValueError("MODEL_NAME must not be blank")

        if (
            self.model_provider == "openai"
            and (
                self.openai_api_key is None
                or not self.openai_api_key.strip()
            )
        ):
            raise ValueError(
                "OPENAI_API_KEY is required when "
                "MODEL_PROVIDER=openai"
            )

        return self


@lru_cache
def get_settings() -> Settings:
    """创建并缓存应用配置。

    配置在进程内只读取一次，避免每个请求重复读取.env文件。
    """
    return Settings()