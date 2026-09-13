"""RailGuard FastAPI应用入口。"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from railguard.api.contracts import router as contracts_router
from railguard.config import get_settings
from railguard.storage.contracts import ContractRepository


def create_app(
    repository: ContractRepository | None = None,
) -> FastAPI:
    """创建FastAPI应用。

    repository参数主要用于测试：
    生产环境使用配置文件中的数据库，
    测试环境可以传入位于临时目录的数据库。
    """
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(
        application: FastAPI,
    ) -> AsyncIterator[None]:
        """管理应用启动与停止生命周期。

        启动时初始化数据库并放入app.state，
        接口通过依赖注入取得同一个存储实例。
        """
        active_repository = repository

        if active_repository is None:
            active_repository = ContractRepository(
                settings.database_path
            )

        active_repository.initialize()
        application.state.contract_repository = active_repository

        yield

    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Multi-agent contract review demo built with LangGraph."
        ),
        lifespan=lifespan,
    )

    # 注册合同上传与查询路由。
    application.include_router(contracts_router)

    @application.get("/health", tags=["system"])
    def health_check() -> dict[str, str]:
        """返回应用状态和当前运行配置。"""
        return {
            "status": "ok",
            "service": settings.app_name,
            "version": settings.app_version,
            "environment": settings.app_env,
            "model_provider": settings.model_provider,
            "rag_mode": settings.rag_mode,
        }

    return application


# Uvicorn通过该变量启动默认应用。
app = create_app()