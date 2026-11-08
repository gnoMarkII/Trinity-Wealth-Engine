"""REST API router for Catalyst Exception Queue (Chapter 7 & Lab 7)."""
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status

from api.auth import require_session
from api.dependencies import get_catalyst_service
from api.schemas.thesis import CatalystExceptionDTO, ReviewCatalystRequestDTO
from application.thesis.catalyst_service import CatalystApplicationService

router = APIRouter()


@router.get("/catalyst-exceptions", response_model=List[CatalystExceptionDTO])
def get_catalyst_exceptions(
    user: dict = Depends(require_session),
    service: CatalystApplicationService = Depends(get_catalyst_service),
) -> List[CatalystExceptionDTO]:
    """Retrieves all active factual catalyst exceptions awaiting PM review."""
    exceptions = service.scan_exceptions()
    return [CatalystExceptionDTO.model_validate(e.to_dict()) for e in exceptions]


@router.post("/catalyst-exceptions/{exception_id}/review", response_model=CatalystExceptionDTO)
def review_catalyst_exception(
    exception_id: str,
    payload: ReviewCatalystRequestDTO,
    user: dict = Depends(require_session),
    service: CatalystApplicationService = Depends(get_catalyst_service),
) -> CatalystExceptionDTO:
    """PM enters human materiality assessment and acknowledges the factual exception."""
    reviewer_role = (user.get("user") if user and hasattr(user, "get") else None) or (user.get("role") if user and hasattr(user, "get") else None) or "PORTFOLIO_MANAGER_HERMES"
    reviewed = service.review_exception(
        exception_id=exception_id,
        human_assessment=payload.human_assessment,
        reviewer=str(reviewer_role),
    )
    if not reviewed:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Exception {exception_id} not found")
    return CatalystExceptionDTO.model_validate(reviewed.to_dict())
