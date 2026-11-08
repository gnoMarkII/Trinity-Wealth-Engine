"""REST Router for Institutional Morning PM Review & Daily Delta Briefings."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from api.auth import require_session
from core.security import AuthenticatedPrincipal
from api.dependencies import get_pm_review_service
from application.manager.pm_review_service import (
    PMReviewApplicationService,
    SignoffPersistenceUnavailable,
)

router = APIRouter(prefix="/api/manager/pm-review", tags=["pm-review"], dependencies=[Depends(require_session)])


class PMSignoffRequestDTO(BaseModel):
    review_date: str = Field(..., description="Review date YYYY-MM-DD")
    portfolio_id: str = Field(default="default", description="Portfolio identifier being signed off")
    signed_off_by: str = Field(default="HUMAN_PM", description="Identifier of the Portfolio Manager")
    notes: Optional[str] = Field(default=None, description="Optional rationale or review notes")
    packet_sha256: str = Field(..., min_length=1, description="Checksum of the packet being approved (mandatory)")
    verified_sample_ids: List[str] = Field(
        default_factory=list,
        description="List of sample IDs explicitly inspected and verified by the PM",
    )
    acknowledged_exception_ids: List[str] = Field(
        default_factory=list,
        description="List of pending exception IDs explicitly reviewed and acknowledged by the PM",
    )


class PMSignoffResponseDTO(BaseModel):
    signoff_id: str
    portfolio_id: str = Field(default="default")
    review_date: str
    status: str
    signed_off_by: str
    signed_off_at_utc: str
    acknowledged_exceptions_count: int
    notes: Optional[str] = None
    packet_sha256: str
    verified_sample_ids: Optional[List[str]] = None
    acknowledged_exception_ids: Optional[List[str]] = None


class MorningReviewPacketDTO(BaseModel):
    review_date: str
    status: str                          # "REVIEW_REQUIRED" or "SIGNED_OFF"
    portfolio_delta: Dict[str, Any]
    macro_delta: Dict[str, Any]
    thesis_exceptions: List[Dict[str, Any]]
    pending_decisions: List[Dict[str, Any]]
    packet_sha256: str
    signoff_info: Optional[Dict[str, Any]] = None
    thesis_coverage: Optional[Dict[str, Any]] = None
    quarantined_holdings: List[Dict[str, Any]] = Field(default_factory=list)
    review_sample: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="3-5 risk-based sample items for PM inspection",
    )
    run_quality: str = Field(default="ON_TIME")
    drift_minutes: int = Field(default=0)


@router.get("/today", response_model=MorningReviewPacketDTO)
def get_today_morning_packet(
    portfolio_id: str = Query(default="default"),
    service: PMReviewApplicationService = Depends(get_pm_review_service),
) -> MorningReviewPacketDTO:
    """Retrieves the 2-minute morning delta briefing packet for today."""
    try:
        packet = service.build_morning_packet(portfolio_id=portfolio_id)
        return MorningReviewPacketDTO(**packet.to_dict())
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to assemble morning review packet: {exc}",
        )


@router.post("/sign-off", response_model=PMSignoffResponseDTO)
def sign_off_morning_review(
    req: PMSignoffRequestDTO,
    user: AuthenticatedPrincipal = Depends(require_session),
    service: PMReviewApplicationService = Depends(get_pm_review_service),
) -> PMSignoffResponseDTO:
    """Records human PM sign-off and acknowledges exceptions with tamper-proof checksum."""
    principal_id = getattr(user, "subject_id", None)
    if not principal_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authenticated principal is required for PM sign-off.",
        )
    try:
        result = service.sign_off(
            review_date=req.review_date,
            signed_off_by=str(principal_id),
            notes=req.notes,
            packet_sha256=req.packet_sha256,
            verified_sample_ids=req.verified_sample_ids,
            acknowledged_exception_ids=req.acknowledged_exception_ids,
            portfolio_id=req.portfolio_id,
        )
        return PMSignoffResponseDTO(**result)
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(val_err),
        )
    except SignoffPersistenceUnavailable as persistence_err:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(persistence_err),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record PM review sign-off: {exc}",
        )


@router.get("/history", response_model=List[PMSignoffResponseDTO])
def get_signoff_history(
    portfolio_id: Optional[str] = Query(default=None),
    limit: int = Query(default=30, ge=1, le=100),
    service: PMReviewApplicationService = Depends(get_pm_review_service),
) -> List[PMSignoffResponseDTO]:
    """Retrieves historical morning review sign-offs."""
    history = service.list_history(limit=limit, portfolio_id=portfolio_id)
    return [
        PMSignoffResponseDTO(
            signoff_id=h["signoff_id"],
            portfolio_id=h.get("portfolio_id", "default"),
            review_date=h["review_date"],
            status="SIGNED_OFF",
            signed_off_by=h["signed_off_by"],
            signed_off_at_utc=h["signed_off_at_utc"],
            acknowledged_exceptions_count=h["acknowledged_exceptions_count"],
            notes=h.get("notes"),
            packet_sha256=h["packet_sha256"],
            verified_sample_ids=h.get("verified_sample_ids"),
            acknowledged_exception_ids=h.get("acknowledged_exception_ids"),
        )
        for h in history
    ]

