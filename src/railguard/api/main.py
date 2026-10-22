"""RailGuard FastAPI应用入口。"""

import sqlite3
from collections.abc import AsyncIterator
from contextlib import (
    AsyncExitStack,
    asynccontextmanager,
)

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from railguard.agents.provider import (
    RiskAnalyzerSet,
    create_ai_platform_risk_analyzers,
    create_deepseek_risk_analyzers,
    create_openai_risk_analyzers,
)
from railguard.api.contracts import router as contracts_router
from railguard.api.reviews import router as reviews_router
from railguard.config import Settings, get_settings
from railguard.rag.client import (
    HttpRagRetriever,
    RagRetriever,
)
from railguard.rag.mock import MockRagRetriever
from railguard.storage.contracts import ContractRepository
from railguard.workflow.checkpoint import (
    open_sqlite_checkpointer,
)
from railguard.workflow.graph import build_review_graph
from railguard.workflow.service import ReviewService


async def _create_rag_retriever(
    *,
    settings: Settings,
    stack: AsyncExitStack,
) -> RagRetriever:
    """根据配置创建Mock或HTTP RAG检索器。

    HTTP客户端交给应用AsyncExitStack管理，
    关闭FastAPI应用时会自动释放连接池。
    """
    if settings.rag_mode == "mock":
        if settings.business_profile == "petkit":
            from railguard.business import validate_petkit_corpus

            validate_petkit_corpus(settings.rag_mock_corpus_path)
        return MockRagRetriever.from_json_file(
            settings.rag_mock_corpus_path
        )

    if settings.rag_mode == "http":
        headers: dict[str, str] = {}

        if settings.rag_api_key:
            headers["Authorization"] = (
                f"Bearer {settings.rag_api_key}"
            )

        client = await stack.enter_async_context(
            httpx.AsyncClient(
                base_url=settings.rag_base_url,
                timeout=settings.rag_timeout_seconds,
                headers=headers,
            )
        )

        return HttpRagRetriever(client)

    raise ValueError(
        f"Unsupported RAG mode: {settings.rag_mode}"
    )


def _create_risk_analyzers(
    settings: Settings,
) -> RiskAnalyzerSet | None:
    """根据配置选择确定性规则或对应供应商的大模型Agent。

    返回None时，工作流使用默认的确定性规则Agent。
    返回RiskAnalyzerSet时，将三个专业模型Agent注入同一工作流。
    """
    # Mock模式不创建外部模型客户端。
    if settings.model_provider == "mock":
        if settings.business_profile == "petkit":
            from railguard.petkit_rules import create_petkit_rule_analyzers

            return create_petkit_rule_analyzers()
        return None

    # 根据供应商选择对应密钥、接口地址和创建工厂。
    if settings.model_provider == "openai":
        api_key = settings.openai_api_key
        base_url = settings.openai_base_url
        factory = create_openai_risk_analyzers
        key_name = "OPENAI_API_KEY"

    elif settings.model_provider == "deepseek":
        api_key = settings.deepseek_api_key
        base_url = settings.deepseek_base_url
        factory = create_deepseek_risk_analyzers
        key_name = "DEEPSEEK_API_KEY"

    elif settings.model_provider == "ai_platform":
        api_key = settings.ai_platform_api_key
        base_url = settings.ai_platform_base_url
        factory = create_ai_platform_risk_analyzers
        key_name = "AI_PLATFORM_API_KEY"

    else:
        raise ValueError(
            "Unsupported model provider: "
            f"{settings.model_provider}"
        )

    # Settings已经执行配置校验。
    # 此处再次检查并收窄类型，保证工厂收到有效字符串。
    if api_key is None or not api_key.strip():
        raise ValueError(
            f"{key_name} is required when "
            f"MODEL_PROVIDER={settings.model_provider}"
        )

    # 两家供应商共用超时、重试和输入限制。
    # 工厂内部负责处理各自的结构化输出协议差异。
    profile_options = (
        {"business_profile": settings.business_profile}
        if settings.business_profile != "baus" else {}
    )
    return factory(
        model_name=settings.model_name,
        api_key=api_key,
        base_url=base_url,
        timeout_seconds=settings.model_timeout_seconds,
        max_retries=settings.model_max_retries,
        max_input_chars=settings.model_max_input_chars,
        **profile_options,
    )


def create_app(
    repository: ContractRepository | None = None,
    *,
    review_service: ReviewService | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    """创建FastAPI应用。

    repository、review_service和settings参数用于测试注入。
    生产环境默认从配置创建SQLite存储、RAG和审核工作流。
    """
    active_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(
        application: FastAPI,
    ) -> AsyncIterator[None]:
        """管理数据库、RAG客户端和LangGraph生命周期。"""
        async with AsyncExitStack() as stack:
            active_repository = repository

            if active_repository is None:
                active_repository = ContractRepository(
                    active_settings.database_path
                )

            active_repository.initialize()
            from railguard.security import SecurityStore

            application.state.security = SecurityStore(active_repository, active_settings.auth_mode == "token")
            application.state.security.initialize()

            active_review_service = review_service

            if active_review_service is None:
                checkpointer = await stack.enter_async_context(
                    open_sqlite_checkpointer(
                        active_settings.checkpoint_path
                    )
                )
                retriever = await _create_rag_retriever(
                    settings=active_settings,
                    stack=stack,
                )
                analyzer_set = _create_risk_analyzers(
                    active_settings
                )

                graph = build_review_graph(
                    retriever=retriever,
                    commercial_analyzer=(
                        analyzer_set.commercial
                        if analyzer_set is not None
                        else None
                    ),
                    legal_analyzer=(
                        analyzer_set.legal
                        if analyzer_set is not None
                        else None
                    ),
                    security_analyzer=(
                        analyzer_set.security
                        if analyzer_set is not None
                        else None
                    ),
                    checkpointer=checkpointer,
                    require_human_review=active_settings.business_profile == "petkit",
                )
                active_review_service = ReviewService(graph)

            application.state.contract_repository = (
                active_repository
            )
            application.state.review_service = (
                active_review_service
            )

            yield

    application = FastAPI(
        title=active_settings.app_name,
        version=active_settings.app_version,
        description=(
            "Multi-agent contract review demo built with LangGraph."
        ),
        lifespan=lifespan,
    )

    # 注册合同上传、查询和审核路由。
    application.include_router(contracts_router)
    application.include_router(reviews_router)
    from railguard.api.security import router as security_router
    from railguard.security import AccessDenied

    application.include_router(security_router)

    @application.exception_handler(AccessDenied)
    async def access_denied_handler(request, exc):
        return JSONResponse(status_code=exc.status, content={"detail": exc.detail})

    @application.exception_handler(sqlite3.Error)
    async def storage_error_handler(request, exc):
        return JSONResponse(status_code=503, content={"detail": "Persistent storage unavailable"})

    @application.get("/profile", tags=["system"])
    def business_profile() -> dict:
        from railguard.business import profile_manifest

        return {
            **profile_manifest(active_settings.business_profile),
            "model_provider": active_settings.model_provider,
            "rag_mode": active_settings.rag_mode,
            "auth_mode": active_settings.auth_mode,
        }

    @application.get("/health", tags=["system"])
    def health_check() -> dict[str, str]:
        """返回应用状态和当前运行配置。"""
        return {
            "status": "ok",
            "service": active_settings.app_name,
            "version": active_settings.app_version,
            "environment": active_settings.app_env,
            "model_provider": (
                active_settings.model_provider
            ),
            "rag_mode": active_settings.rag_mode,
        }

    return application


# Uvicorn通过该变量启动默认应用。
app = create_app()
