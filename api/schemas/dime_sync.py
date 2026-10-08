"""Dime Trade Confirmation Synchronization Schemas (Hexagonal Driving Adapter)."""
from typing import List, Optional
from pydantic import BaseModel, Field

from api.schemas.portfolio import ActualPortfolioStateDTO


class DimeEmailMetadataDTO(BaseModel):
    message_id: str
    attachment_id: str
    subject: str
    sender: str
    received_at: str
    filename: str
    size_bytes: int


class DimeEmailListResponseDTO(BaseModel):
    emails: List[DimeEmailMetadataDTO] = []


class DimeScanEmailRequestDTO(BaseModel):
    message_id: str
    attachment_id: str
    password: Optional[str] = None


class DimeStagedItemFeeDTO(BaseModel):
    commission: str = "0.00"
    vat: str = "0.00"
    other_fees: str = "0.00"
    fee_currency: str = "THB"


class DimeStagedItemDTO(BaseModel):
    item_id: str
    trade_date: str
    settlement_date: Optional[str] = None
    symbol: str
    action: str
    units: str
    price: str
    gross_amount: str
    fees: DimeStagedItemFeeDTO = Field(default_factory=DimeStagedItemFeeDTO)
    net_amount: str
    currency: str = "THB"
    exchange_rate: Optional[str] = None
    confirmation_no: str
    order_id: Optional[str] = None
    source: str = "DIME"
    fingerprint: str
    line_index: int = 0
    cash_adjusted: bool = True
    asset_type: str = "Stock"


class DimeScanResponseDTO(BaseModel):
    scan_id: str
    item_count: int
    items: List[DimeStagedItemDTO] = []


class DimeBatchScanRequestDTO(BaseModel):
    password: Optional[str] = None
    force_rescan: bool = False
    portfolio_id: str = "default"


class DimeCommitRequestDTO(BaseModel):
    portfolio_id: str = "default"
    selected_item_ids: Optional[List[str]] = None


class DimeCommitResponseDTO(BaseModel):
    ok: bool = True
    imported_count: int
    state: ActualPortfolioStateDTO


class DimeWarningItemDTO(BaseModel):
    message_id: str = ""
    attachment_id: str = ""
    subject: str = ""
    filename: str = ""
    received_at: str = ""
    reason: str
    can_preview: bool = False


class DimePdfTextPageDTO(BaseModel):
    page_number: int
    text: str


class DimePdfTextResponseDTO(BaseModel):
    message_id: str
    attachment_id: str
    filename: str
    page_count: int
    pages: List[DimePdfTextPageDTO] = []

