"""FastAPI依赖项。"""

from fastapi import Header, Request

from railguard.security import Principal, SecurityStore
from railguard.storage.contracts import ContractRepository
from railguard.workflow.service import ReviewService


def get_security(request: Request) -> SecurityStore:
    return request.app.state.security


def get_principal(request: Request, authorization: str | None = Header(default=None)) -> Principal:
    security = get_security(request)
    token = authorization[7:] if authorization and authorization.startswith("Bearer ") else ""
    return security.authenticate(token)


def get_contract_repository(
    request: Request,
) -> ContractRepository:
    """从当前FastAPI应用中取得合同存储实例。

    存储实例在应用生命周期启动阶段创建。
    测试可以为create_app传入临时存储，避免污染真实数据库。
    """
    repository = getattr(
        request.app.state,
        "contract_repository",
        None,
    )

    if repository is None:
        raise RuntimeError(
            "Contract repository is not initialized."
        )

    return repository


def get_review_service(
    request: Request,
) -> ReviewService:
    """从当前FastAPI应用中取得审核工作流服务。

    ReviewService在应用生命周期中创建，并共享同一个
    LangGraph图、checkpoint连接和RAG检索器。
    """
    service = getattr(
        request.app.state,
        "review_service",
        None,
    )

    if service is None:
        raise RuntimeError(
            "Review service is not initialized."
        )

    return service
