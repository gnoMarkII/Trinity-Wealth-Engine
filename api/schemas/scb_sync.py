"""SCBAM Fund Click Synchronization Schemas (Hexagonal Driving Adapter)."""
from typing import List, Optional
from pydantic import BaseModel, Field

from api.schemas.portfolio import ActualPortfolioStateDTO


class SCBAMEmailMetadataDTO(BaseModel):
    message_id: str
    attachment_id: str
    subject: str
    sender: str
    received_at: str
    filename: str
    size_bytes: int


class SCBAMEmailListResponseDTO(BaseModel):
    emails: List[SCBAMEmailMetadataDTO] = []


class SCBAMBatchScanRequestDTO(BaseModel):
    since_date: Optional[str] = None
    portfolio_id: str = "default"
    limit: Optional[int] = None


class SCBAMSingleScanEmailRequestDTO(BaseModel):
    message_id: str
    portfolio_id: str = "default"


class SCBAMStagedItemFeeDTO(BaseModel):
    commission: str = "0.00"
    vat: str = "0.00"
    other_fees: str = "0.00"
    fee_currency: str = "THB"


class SCBAMStagedItemDTO(BaseModel):
    item_id: str
    trade_date: str
    settlement_date: Optional[str] = None
    symbol: str
    action: str
    units: str
    price: str
    gross_amount: str
    fees: SCBAMStagedItemFeeDTO = Field(default_factory=SCBAMStagedItemFeeDTO)
    net_amount: str
    currency: str = "THB"
    exchange_rate: Optional[str] = None
    confirmation_no: str
    order_id: Optional[str] = None
    source: str = "SCB"
    fingerprint: str
    line_index: int = 0
    cash_adjusted: bool = True
    asset_type: str = "Fund"
    status: str = "NEW"


class SCBAMScanResponseDTO(BaseModel):
    scan_id: str
    item_count: int
    items: List[SCBAMStagedItemDTO] = []


class SCBAMCommitRequestDTO(BaseModel):
    portfolio_id: str = "default"
    selected_item_ids: Optional[List[str]] = None


class SCBAMCommitResponseDTO(BaseModel):
    ok: bool = True
    imported_count: int
    state: ActualPortfolioStateDTO


class SCBAMEmailHtmlResponseDTO(BaseModel):
    message_id: str
    html: str
