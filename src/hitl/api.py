"""Optional FastAPI adapter for the HITL control plane.

FastAPI remains an edge concern. The service can also be called by an Azure
Container Apps worker, a Celery task, or a LangGraph Server endpoint without
changing the graph or approval semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from src.hitl.models import HITLResumeResult, HITLStatus
from src.hitl.service import (
    HITLAuthorizationError,
    HITLConflict,
    HITLResumeTimeout,
    HumanApprovalService,
)


@dataclass(frozen=True, slots=True)
class HITLPrincipal:
    tenant_id: str
    subject_id: str
    principal_ref: str
    roles: tuple[str, ...]


class PrincipalDependency(Protocol):
    async def __call__(self) -> HITLPrincipal:
        """Return claims verified by Entra ID, Keycloak, or the API gateway."""


def create_hitl_router(
    service: HumanApprovalService,
    *,
    principal_dependency: Any,
):
    """Build routes without making unverified HTTP headers an identity source."""

    try:
        from fastapi import APIRouter, Depends, HTTPException
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise RuntimeError("Install fastapi to expose the HITL HTTP adapter") from exc

    from pydantic import BaseModel, ConfigDict, Field

    class ApprovalSubmission(BaseModel):
        model_config = ConfigDict(extra="forbid", strict=True)

        approval_request_id: str = Field(min_length=1, max_length=128)
        decision: str
        reason: str = Field(min_length=1, max_length=2_000)

    router = APIRouter(prefix="/v1/hitl", tags=["human-approval"])

    @router.get("/threads/{thread_id}", response_model=HITLStatus)
    async def get_status(
        thread_id: str,
        principal: HITLPrincipal = Depends(principal_dependency),  # noqa: B008
    ):
        return await service.status(
            thread_id=thread_id,
            tenant_id=principal.tenant_id,
        )

    @router.post("/threads/{thread_id}/resume", response_model=HITLResumeResult)
    async def resume(
        thread_id: str,
        submission: ApprovalSubmission,
        principal: HITLPrincipal = Depends(principal_dependency),  # noqa: B008
    ):
        try:
            return await service.resume(
                thread_id=thread_id,
                tenant_id=principal.tenant_id,
                approver_ref=principal.principal_ref,
                approver_roles=principal.roles,
                approval_request_id=submission.approval_request_id,
                decision=submission.decision,
                reason=submission.reason,
            )
        except HITLAuthorizationError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except HITLConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except HITLResumeTimeout as exc:
            raise HTTPException(status_code=504, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return router
