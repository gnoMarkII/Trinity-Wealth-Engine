"""REST API router for Two-Sided Estimate Ledger & Variant Perception (Ch. 2 & 5)."""
from decimal import Decimal
import math
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.auth import require_session
from api.dependencies import get_estimate_ledger_service
from api.schemas.equity import EstimateRecordDTO, VariantPerceptionSnapshotDTO
from application.equity.estimate_service import EstimateLedgerApplicationService

router = APIRouter()


@router.get("/{ticker}/variant-perception", response_model=VariantPerceptionSnapshotDTO)
def get_variant_perception(
    ticker: str,
    period: str = Query("2026Q2", description="Earnings period e.g. '2026Q2'"),
    metric_name: str = Query("revenue", description="Financial metric name"),
    reported: Optional[str] = Query(None, description="Reported result (Decimal string). Required for surprise realization."),
    user: dict = Depends(require_session),
    service: EstimateLedgerApplicationService = Depends(get_estimate_ledger_service),
) -> VariantPerceptionSnapshotDTO:
    """Computes three-way surprise: Street Surprise vs Fund Variant Realization vs Prior Spread."""
    if not reported or not reported.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Query parameter 'reported' is required to evaluate variant realization against consensus.",
        )
    try:
        rep_val = Decimal(reported.strip())
        if rep_val.is_nan() or rep_val.is_infinite():
            raise ValueError("NaN or Infinite values are not permitted.")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid reported numerical value '{reported}': {exc}",
        )

    try:
        snapshot = service.get_variant_perception(
            ticker=ticker,
            period=period,
            metric_name=metric_name,
            reported=rep_val,
        )
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(val_err),
        )
    return VariantPerceptionSnapshotDTO.model_validate(snapshot.to_dict())


@router.get("/{ticker}/estimates-ledger", response_model=List[EstimateRecordDTO])
def get_estimates_ledger(
    ticker: str,
    user: dict = Depends(require_session),
    service: EstimateLedgerApplicationService = Depends(get_estimate_ledger_service),
) -> List[EstimateRecordDTO]:
    """Retrieves immutable chronological revision trail of estimates for a ticker."""
    records = service.list_estimates_for_ticker(ticker)
    return [EstimateRecordDTO.model_validate(r.to_dict()) for r in records]
