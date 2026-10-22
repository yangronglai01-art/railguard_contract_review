"""Authenticated identity and metadata audit endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from railguard.api.dependencies import get_principal, get_security
from railguard.security import AccessDenied, Principal, SecurityStore

router = APIRouter(prefix="/security", tags=["security"])


@router.get("/me")
def me(principal: Annotated[Principal, Depends(get_principal)]):
    return {"subject": principal.subject, "role": principal.role}


@router.get("/audit")
def audit(
    principal: Annotated[Principal, Depends(get_principal)],
    security: Annotated[SecurityStore, Depends(get_security)],
    limit: int = Query(default=100, ge=1, le=500),
):
    if not security.enabled:
        raise AccessDenied(403, "Requires token authorization mode")
    items = security.audit_rows(principal, limit)
    security.verify(principal)
    return {"items": items}
