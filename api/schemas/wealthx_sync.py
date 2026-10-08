"""WealthX Trade Confirmation Synchronization Schemas (Hexagonal Driving Adapter)."""
from typing import List, Optional
from pydantic import BaseModel, Field

from api.schemas.portfolio import ActualPortfolioStateDTO


class WealthXEmailMetadataDTO(BaseModel):
    message_id: str
    attachment_id: str
    subject: str
    sender: str
    received_at: str
    filename: str
    size_bytes: int


class WealthXEmailListResponseDTO(BaseModel):
    emails: List[WealthXEmailMetadataDTO] = []


class WealthXScanEmailRequestDTO(BaseModel):
    message_id: str
    attachment_id: str
    password: Optional[str] = None


class WealthXStagedItemFeeDTO(BaseModel):
    commission: str = "0.00"
    vat: str = "0.00"
    other_fees: str = "0.00"
    fee_currency: str = "THB"


class WealthXStagedItemDTO(BaseModel):
    item_id: str
    trade_date: str
    settlement_date: Optional[str] = None
    symbol: str
    action: str
    units: str
    price: str
    gross_amount: str
    fees: WealthXStagedItemFeeDTO = Field(default_factory=WealthXStagedItemFeeDTO)
    net_amount: str
    currency: str = "THB"
    exchange_rate: Optional[str] = None
    confirmation_no: str
    order_id: Optional[str] = None
    source: str = "WEALTHX"
    fingerprint: str
    line_index: int = 0
    cash_adjusted: bool = True
    asset_type: str = "Fund"


class WealthXScanResponseDTO(BaseModel):
    scan_id: str
    item_count: int
    items: List[WealthXStagedItemDTO] = []


class WealthXBatchScanRequestDTO(BaseModel):
    password: Optional[str] = None
    force_rescan: bool = False
    portfolio_id: str = "default"


class WealthXCommitRequestDTO(BaseModel):
    portfolio_id: str = "default"
    selected_item_ids: Optional[List[str]] = None


class WealthXCommitResponseDTO(BaseModel):
    ok: bool = True
    imported_count: int
    state: ActualPortfolioStateDTO


class WealthXWarningItemDTO(BaseModel):
    message_id: str = ""
    attachment_id: str = ""
    subject: str = ""
    filename: str = ""
    received_at: str = ""
    reason: str
    can_preview: bool = False


class WealthXPdfTextPageDTO(BaseModel):
    page_number: int
    text: str


class WealthXPdfTextResponseDTO(BaseModel):
    message_id: str
    attachment_id: str
    filename: str
    page_count: int
    pages: List[WealthXPdfTextPageDTO] = []
