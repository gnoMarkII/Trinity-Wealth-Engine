from decimal import Decimal
from typing import Any, Optional

from pydantic import BaseModel, Field


class PvdAnalyzeResponse(BaseModel):
    scan_id: Optional[str] = None
    can_commit: bool
    status: str
    message: str
    statement: Optional[dict[str, Any]] = None
    holdings: list[dict[str, Any]] = Field(default_factory=list)
    current_nav: Optional[str] = None
    new_nav: Optional[str] = None
    delta_nav: Optional[str] = None
    vesting_percent: str = "100"
    vested_amount: Optional[str] = None
    archive_id: Optional[str] = None


class PvdCommitRequest(BaseModel):
    scan_id: str
    idempotency_key: str
    portfolio_id: str = "default"
    vesting_percent: Decimal = Decimal("100")
    selected_item_ids: Optional[list[str]] = None


class PvdCommitResponse(BaseModel):
    status: str
    message: str
    archive_id: Optional[str] = None
    state: dict[str, Any]
