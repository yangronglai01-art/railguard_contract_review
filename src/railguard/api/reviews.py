"""合同审核任务的启动、查询和人工决定接口。"""

from typing import Annotated, Never

from fastapi import (
    APIRouter,
    Depends,
    Header,
    HTTPException,
    Query,
    status,
)

from railguard.api.dependencies import (
    get_contract_repository,
    get_principal,
    get_review_service,
    get_security,
)
from railguard.models.schemas import new_id
from railguard.security import Principal, SecurityStore
from railguard.storage.contracts import ContractRepository
from railguard.workflow.results import (
    MonitoringSummary,
    ReviewSnapshot,
)
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
        "rag_unavailable": (
            status.HTTP_503_SERVICE_UNAVAILABLE
        ),
        "rag_request_rejected": status.HTTP_502_BAD_GATEWAY,
        "rag_protocol_error": status.HTTP_502_BAD_GATEWAY,
        "model_input_too_large": (
            status.HTTP_413_CONTENT_TOO_LARGE
        ),
        "model_unavailable": (
            status.HTTP_503_SERVICE_UNAVAILABLE
        ),
        "model_response_invalid": (
            status.HTTP_502_BAD_GATEWAY
        ),
        "model_protocol_error": (
            status.HTTP_502_BAD_GATEWAY
        ),
        "model_execution_failed": (
            status.HTTP_500_INTERNAL_SERVER_ERROR
        ),
        "workflow_protocol_error": (
            status.HTTP_500_INTERNAL_SERVER_ERROR
        ),
        "workflow_execution_failed": (
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
    principal: Annotated[Principal, Depends(get_principal)],
    security: Annotated[SecurityStore, Depends(get_security)],
) -> ReviewSnapshot:
    """为已保存的合同启动一次新的多Agent审核。"""
    security.authorize(principal, contract_id, "start")
    contract = repository.get(contract_id)

    if contract is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Contract not found.",
        )

    try:
        if security.enabled:
            review_id = new_id()
            security.reserve_review(principal, review_id, contract_id)
            result = await service.start_review(contract, review_id=review_id)
        else:
            result = await service.start_review(contract)
        security.authorize(principal, contract_id)
        security.record(principal, "start_review", result.review_id, "completed")
        return result
    except ReviewExecutionError as exc:
        security.record(principal, "start_review", exc.review_id, "failed")
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
    principal: Annotated[Principal, Depends(get_principal)],
    security: Annotated[SecurityStore, Depends(get_security)],
) -> ReviewSnapshot:
    """从LangGraph checkpoint读取审核任务状态。"""
    try:
        contract_id = security.review_contract(principal, review_id)
        result = await service.get_review(review_id)
        if contract_id is not None and result.contract_id != contract_id:
            raise HTTPException(503, "Persistent mapping unavailable")
        security.review_contract(principal, review_id)
        return result
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
    principal: Annotated[Principal, Depends(get_principal)],
    security: Annotated[SecurityStore, Depends(get_security)],
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> ReviewSnapshot:
    """提交人工决定并恢复暂停的LangGraph审核任务。"""
    try:
        security.require_role(principal, "reviewer")
        contract_id = security.review_contract(principal, review_id, "decide")
        if security.enabled:
            snapshot = await service.get_review(review_id)
            if snapshot.contract_id != contract_id:
                raise HTTPException(503, "Persistent mapping unavailable")
            decision = decision.model_copy(update={"reviewer": principal.subject})
        def authorize_decision():
            security.review_contract(principal, review_id, "decide")
            security.record(principal, "decision", review_id, "intent")

        options = {"authorize": authorize_decision} if security.enabled else {}
        if getattr(service,"_postgres_lock_dsn",None):
            if not idempotency_key or len(idempotency_key)>128:
                raise HTTPException(422,"Idempotency-Key required for enterprise approval")
            options['decision_id']=idempotency_key
        result = await service.submit_decision(
            review_id=review_id,
            decision=decision,
            **options,
        )
        security.review_contract(principal, review_id, "decide")
        security.record(principal, "decision", review_id, result.status)
        return result
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
        security.record(principal, "decision", review_id, "failed")
        _raise_execution_error(exc)


@router.get(
    "/reviews",
    response_model=list[ReviewSnapshot],
)
async def list_reviews(
    service: Annotated[
        ReviewService,
        Depends(get_review_service),
    ],
    principal: Annotated[Principal, Depends(get_principal)],
    security: Annotated[SecurityStore, Depends(get_security)],
    limit: int = Query(default=100, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> list[ReviewSnapshot]:
    """列出checkpoint中全部审核任务的当前快照。"""
    if not security.enabled:
        return await service.list_reviews()
    security.record(principal, "list_reviews", "", "intent")
    results = []
    for review_id in security.visible_reviews(principal, limit, offset):
        try:
            result = await service.get_review(review_id)
        except ReviewNotFoundError:
            # Reserved mapping may precede the first durable checkpoint.
            continue
        contract_id = security.review_contract(principal, review_id)
        if result.contract_id != contract_id:
            raise HTTPException(503, "Persistent mapping unavailable")
        results.append(result)
    security.verify(principal)
    for result in results:
        security.review_contract(principal, result.review_id)
    return results


@router.get(
    "/monitoring/summary",
    response_model=MonitoringSummary,
    tags=["monitoring"],
)
async def get_monitoring_summary(
    service: Annotated[
        ReviewService,
        Depends(get_review_service),
    ],
    principal: Annotated[Principal, Depends(get_principal)],
    security: Annotated[SecurityStore, Depends(get_security)],
) -> MonitoringSummary:
    """返回跨审核任务的进度、耗时和重试监控汇总。"""
    security.require_role(principal, "auditor")
    result = await service.monitoring_summary()
    security.verify(principal)
    security.record(principal, "monitoring", "", "completed")
    return result
