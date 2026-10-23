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
    app_env: Literal["development", "integration", "staging", "production", "local-simulation"] = "development"
    app_version: str = "0.1.0"

    # 审核编排模式（document 25 目标架构的切换开关）：
    # fixed_v1：固定三路并行（商务/法务/安全）的既有工作流，保持兼容；
    # supervisor_v2：主 Agent 受约束规划 → 确定性校验 → 专业子 Agent 执行
    #   → 任务级共享记忆 → PG 事务 Outbox 事件驱动 → 有限协同复核。
    review_orchestration_mode: Literal[
        "fixed_v1",
        "supervisor_v2",
    ] = Field(
        default="fixed_v1",
        validation_alias="REVIEW_ORCHESTRATION_MODE",
    )

    # Legacy defaults remain reproducible; the Petkit launcher selects its profile explicitly.
    business_profile: Literal["baus", "petkit"] = "baus"
    auth_mode: Literal["local_demo", "token", "oidc"] = "local_demo"

    database_url: str | None = Field(default=None, repr=False,validation_alias="RAILGUARD_DATABASE_URL")
    checkpoint_url: str | None = Field(default=None, repr=False,validation_alias="RAILGUARD_CHECKPOINT_URL")
    enterprise_tenant_id: str = ""
    oidc_introspection_url: str = ""
    oidc_issuer: str = ""
    oidc_audience: str = ""
    oidc_client_id: str = ""
    oidc_client_secret: str = Field(default="", repr=False)

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
    openai_api_key: str | None = Field(default=None,repr=False)

    # OpenAI兼容接口地址。
    # 留空时使用SDK默认的OpenAI服务地址。
    openai_base_url: str | None = None

    # DeepSeek开放平台或企业DeepSeek网关的访问密钥。
    deepseek_api_key: str | None = Field(default=None,repr=False)

    # DeepSeek默认使用官方接口，也可以配置企业代理网关。
    deepseek_base_url: str = "https://api.deepseek.com"

    # 统一AI网关（ai-platform）的访问密钥。
    ai_platform_api_key: str | None = Field(default=None,repr=False)

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
    rag_api_key: str | None = Field(default=None,repr=False)
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
        extra="ignore", hide_input_in_errors=True,
        populate_by_name=True,
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

        if self.app_env in {"staging", "production"}:
            if self.business_profile != "petkit" or self.auth_mode != "oidc":
                raise ValueError("Enterprise deployment requires Petkit and OIDC")
            if not all(url and url.startswith(("postgresql://","postgres://")) for url in [self.database_url,self.checkpoint_url]):
                raise ValueError("Enterprise deployment requires PostgreSQL business and checkpoint DSNs")
            if self.model_provider != "ai_platform" or self.model_name != "local-chat" or self.rag_mode != "http":
                raise ValueError("Enterprise contract review requires local-chat gateway and real HTTP RAG")
            from urllib.parse import parse_qs, urlsplit
            if any(parse_qs(urlsplit(url).query).get('sslmode')!=['verify-full'] for url in [self.database_url,self.checkpoint_url]):
                raise ValueError('Enterprise PostgreSQL requires sslmode=verify-full')
            for url in [self.ai_platform_base_url,self.rag_base_url]:
                if urlsplit(url).scheme != "https":
                    raise ValueError("Enterprise model and evidence endpoints require HTTPS")
            if not self.rag_api_key or not self.enterprise_tenant_id:
                raise ValueError("Enterprise evidence credentials and tenant are required")
            from railguard.enterprise_identity import OIDC
            OIDC(self.oidc_introspection_url,self.oidc_issuer,self.oidc_audience,
                 self.oidc_client_id,self.oidc_client_secret)
        return self


@lru_cache
def get_settings() -> Settings:
    """创建并缓存应用配置。

    每个进程只读取一次配置，避免每个请求重复读取.env文件。
    修改配置后需要重新启动对应的服务进程。
    """
    if os.getenv("RAILGUARD_ISOLATED_DEMO") == "1":
        if os.getenv("APP_ENV") in {"staging","production"}:
            raise ValueError("Isolated demo cannot override enterprise environment")
        # Explicit defaults override inherited model/provider and secret variables.
        return Settings(_env_file=None, **{
            name: field.default for name, field in Settings.model_fields.items()
        })
    return Settings(_env_file=None) if os.getenv("APP_ENV") in {"staging","production"} else Settings()
