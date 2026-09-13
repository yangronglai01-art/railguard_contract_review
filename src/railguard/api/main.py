"""RailGuard FastAPI应用入口。"""

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager

import httpx
from fastapi import FastAPI

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
                graph = build_review_graph(
                    retriever=retriever,
                    checkpointer=checkpointer,
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

    @application.get("/health", tags=["system"])
    def health_check() -> dict[str, str]:
        """返回应用状态和当前运行配置。"""
        return {
            "status": "ok",
            "service": active_settings.app_name,
            "version": active_settings.app_version,
            "environment": active_settings.app_env,
            "model_provider": active_settings.model_provider,
            "rag_mode": active_settings.rag_mode,
        }

    return application


# Uvicorn通过该变量启动默认应用。
app = create_app()