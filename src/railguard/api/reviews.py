"""合同审核任务的启动、查询和人工决定接口。"""

from typing import Annotated, Never

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)

from railguard.api.dependencies import (
    get_contract_repository,
    get_review_service,
)
from railguard.storage.contracts import ContractRepository
from railguard.workflow.results import ReviewSnapshot
from railguard.workflow.service import (
    ReviewConflictError,
    ReviewDecisionError,
    ReviewExecutionError,
    ReviewNotFoundError,
    ReviewService,
)
from railguard.workflow.state import HumanReviewDecision

router = APIRouter(tags=["reviews"])


def _execution_status_code(
    error: ReviewExecutionError,
) -> int:
    """把审核执行错误映射为HTTP状态码。"""
    status_by_code = {
        "rag_timeout": status.HTTP_504_GATEWAY_TIMEOUT,
        "rag_unavailable": status.HTTP_503_SERVICE_UNAVAILABLE,
        "rag_request_rejected": status.HTTP_502_BAD_GATEWAY,
        "rag_protocol_error": status.HTTP_502_BAD_GATEWAY,
        "workflow_protocol_error": (
            status.HTTP_500_INTERNAL_SERVER_ERROR
        ),
    }

    return status_by_code.get(
        error.code,
        status.HTTP_500_INTERNAL_SERVER_ERROR,
    )


def _raise_execution_error(
    error: ReviewExecutionError,
) -> Never:
    """返回不暴露内部异常细节的结构化HTTP错误。"""
    raise HTTPException(
        status_code=_execution_status_code(error),
        detail={
            "review_id": error.review_id,
            "code": error.code,
            "message": str(error),
            "retryable": error.retryable,
        },
    ) from error


@router.post(
    "/contracts/{contract_id}/reviews",
    response_model=ReviewSnapshot,
    status_code=status.HTTP_201_CREATED,
)
async def start_contract_review(
    contract_id: str,
    repository: Annotated[
        ContractRepository,
        Depends(get_contract_repository),
    ],
    service: Annotated[
        ReviewService,
        Depends(get_review_service),
    ],
) -> ReviewSnapshot:
    """为已保存的合同启动一次新的多Agent审核。"""
    contract = repository.get(contract_id)

    if contract is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Contract not found.",
        )

    try:
        return await service.start_review(contract)
    except ReviewExecutionError as exc:
        _raise_execution_error(exc)


@router.get(
    "/reviews/{review_id}",
    response_model=ReviewSnapshot,
)
async def get_contract_review(
    review_id: str,
    service: Annotated[
        ReviewService,
        Depends(get_review_service),
    ],
) -> ReviewSnapshot:
    """从LangGraph checkpoint读取审核任务状态。"""
    try:
        return await service.get_review(review_id)
    except ReviewNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Review not found.",
        ) from exc


@router.post(
    "/reviews/{review_id}/decision",
    response_model=ReviewSnapshot,
)
async def submit_review_decision(
    review_id: str,
    decision: HumanReviewDecision,
    service: Annotated[
        ReviewService,
        Depends(get_review_service),
    ],
) -> ReviewSnapshot:
    """提交人工决定并恢复暂停的LangGraph审核任务。"""
    try:
        return await service.submit_decision(
            review_id=review_id,
            decision=decision,
        )
    except ReviewNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Review not found.",
        ) from exc
    except ReviewConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except ReviewDecisionError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc
    except ReviewExecutionError as exc:
        _raise_execution_error(exc)