"""ARR Gold Trade Confirmation Synchronization Schemas (Hexagonal Driving Adapter)."""
from typing import Any, List, Optional
from pydantic import BaseModel, Field

from api.schemas.portfolio import ActualPortfolioStateDTO


class ArrGoldEmailMetadataDTO(BaseModel):
    message_id: str
    attachment_id: Optional[str] = None
    subject: str
    sender: str
    received_at: str
    filename: Optional[str] = None
    size_bytes: int = 0


class ArrGoldEmailListResponseDTO(BaseModel):
    emails: List[ArrGoldEmailMetadataDTO] = []


class ArrGoldScanEmailRequestDTO(BaseModel):
    message_id: str
    attachment_id: Optional[str] = None
    subject: Optional[str] = None
    password: Optional[str] = None


class ArrGoldStagedItemFeeDTO(BaseModel):
    commission: str = "0.00"
    vat: str = "0.00"
    other_fees: str = "0.00"
    fee_currency: str = "USD"


class ArrGoldStagedItemDTO(BaseModel):
    item_id: str
    trade_date: str
    settlement_date: Optional[str] = None
    symbol: str
    action: str
    units: str
    price: str
    gross_amount: str
    fees: ArrGoldStagedItemFeeDTO = Field(default_factory=ArrGoldStagedItemFeeDTO)
    net_amount: str
    currency: str = "USD"
    exchange_rate: Optional[str] = None
    confirmation_no: str
    order_id: Optional[str] = None
    source: str = "ARR_GOLD"
    fingerprint: str
    line_index: int = 0
    cash_adjusted: bool = False
    asset_type: str = "Gold"


class ArrGoldScanResponseDTO(BaseModel):
    scan_id: str
    item_count: int
    items: List[ArrGoldStagedItemDTO] = []
    warnings: List[Any] = []


class ArrGoldBatchScanRequestDTO(BaseModel):
    password: Optional[str] = None
    force_rescan: bool = False
    portfolio_id: str = "default"


class ArrGoldCommitRequestDTO(BaseModel):
    portfolio_id: str = "default"
    selected_item_ids: Optional[List[str]] = None


class ArrGoldCommitResponseDTO(BaseModel):
    ok: bool = True
    imported_count: int
    state: ActualPortfolioStateDTO
