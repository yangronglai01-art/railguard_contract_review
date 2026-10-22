"""RailGuard应用配置。"""

import os
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
    """从默认值、环境变量和.env文件中读取应用配置。"""

    # 应用基本信息。
    app_name: str = "RailGuard AI"
    app_env: str = "development"
    app_version: str = "0.1.0"

    # Legacy defaults remain reproducible; the Petkit launcher selects its profile explicitly.
    business_profile: Literal["baus", "petkit"] = "baus"
    auth_mode: Literal["local_demo", "token"] = "local_demo"

    # 风险分析模式：
    # mock使用确定性规则；
    # openai使用OpenAI严格结构化接口；
    # deepseek使用DeepSeek Responses接口；
    # ai_platform使用统一AI网关的Chat Completions+JSON模式。
    model_provider: Literal[
        "mock",
        "openai",
        "deepseek",
        "ai_platform",
    ] = "mock"

    # 当前模式使用的供应商模型名称。
    # 可以填写通用审核模型或合同领域微调模型的名称。
    model_name: str = "mock-contract-reviewer"

    # OpenAI或相应兼容服务的访问密钥。
    openai_api_key: str | None = None

    # OpenAI兼容接口地址。
    # 留空时使用SDK默认的OpenAI服务地址。
    openai_base_url: str | None = None

    # DeepSeek开放平台或企业DeepSeek网关的访问密钥。
    deepseek_api_key: str | None = None

    # DeepSeek默认使用官方接口，也可以配置企业代理网关。
    deepseek_base_url: str = "https://api.deepseek.com"

    # 统一AI网关（ai-platform）的访问密钥。
    ai_platform_api_key: str | None = None

    # 统一AI网关（ai-platform）的OpenAI兼容接口根地址。
    ai_platform_base_url: str = "http://127.0.0.1:18700/v1"

    # 单次模型请求的超时时间，单位为秒。
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
    # 超出限制时明确拒绝请求，避免静默截断审核资料。
    model_max_input_chars: int = Field(
        default=120_000,
        gt=0,
    )

    # 外部RAG服务的运行模式和连接配置。
    rag_mode: str = "mock"
    rag_base_url: str = "http://localhost:8001"
    rag_api_key: str | None = None
    rag_timeout_seconds: float = 10.0

    # 本地Mock模式使用的演示知识库文件。
    rag_mock_corpus_path: Path = Path(
        "data/demo/rag-corpus-baus-v1.json"
    )

    # 保存合同资料的SQLite业务数据库路径。
    # 使用明确的Path，避免误用数据库连接字符串。
    database_path: Path = Path(
        "data/runtime/railguard.db"
    )

    # 保存LangGraph审核状态的checkpoint数据库路径。
    # 与合同业务数据库分开，便于迁移和故障排查。
    checkpoint_path: Path = Path(
        "data/runtime/review-checkpoints.sqlite3"
    )

    # 从项目根目录的.env文件加载配置。
    # 环境变量可以覆盖.env文件中的同名配置。
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @model_validator(mode="after")
    def validate_model_configuration(self) -> Self:
        """验证模型名称和所选供应商需要的访问密钥。"""
        if self.auth_mode == "token" and self.business_profile != "petkit":
            raise ValueError("Token authorization is currently validated for Petkit only")
        if self.business_profile == "petkit":
            from railguard.business import PETKIT_CORPUS

            if "rag_mock_corpus_path" not in self.model_fields_set:
                self.rag_mock_corpus_path = Path(PETKIT_CORPUS)
            if "database_path" not in self.model_fields_set:
                self.database_path = Path("data/runtime/petkit/contracts.db")
            if "checkpoint_path" not in self.model_fields_set:
                self.checkpoint_path = Path("data/runtime/petkit/checkpoints.sqlite3")
            if self.rag_mode == "mock" and self.rag_mock_corpus_path.name in {
                "rag-corpus-baus-v1.json", "rag-corpus.json",
            }:
                raise ValueError("Petkit profile cannot use a frozen legacy corpus")

        if not self.model_name.strip():
            raise ValueError(
                "MODEL_NAME must not be blank"
            )

        # OpenAI模式必须具有对应的API密钥。
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

        # DeepSeek模式必须具有独立的DeepSeek API密钥。
        if (
            self.model_provider == "deepseek"
            and (
                self.deepseek_api_key is None
                or not self.deepseek_api_key.strip()
            )
        ):
            raise ValueError(
                "DEEPSEEK_API_KEY is required when "
                "MODEL_PROVIDER=deepseek"
            )

        # 统一AI网关模式必须具有对应的网关访问密钥。
        if (
            self.model_provider == "ai_platform"
            and (
                self.ai_platform_api_key is None
                or not self.ai_platform_api_key.strip()
            )
        ):
            raise ValueError(
                "AI_PLATFORM_API_KEY is required when "
                "MODEL_PROVIDER=ai_platform"
            )

        return self


@lru_cache
def get_settings() -> Settings:
    """创建并缓存应用配置。

    每个进程只读取一次配置，避免每个请求重复读取.env文件。
    修改配置后需要重新启动对应的服务进程。
    """
    if os.getenv("RAILGUARD_ISOLATED_DEMO") == "1":
        # Explicit defaults override inherited model/provider and secret variables.
        return Settings(_env_file=None, **{
            name: field.default for name, field in Settings.model_fields.items()
        })
    return Settings()
