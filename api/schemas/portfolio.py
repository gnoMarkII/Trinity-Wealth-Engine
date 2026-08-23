"""Portfolio, Positions, Transactions, and Goals API Schemas."""
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field, field_validator

class DividendRoundDTO(BaseModel):
    symbol: str
    ex_date: str
    pay_date: Optional[str] = None
    dps: float
    currency: str
    units_held: float
    status: Literal["received", "upcoming"] = "received"
    gross_native: float = 0.0
    net_native: float = 0.0
    gross_thb: float = 0.0
    tax_rate: float = 0.0
    net_thb: float = 0.0
    fx_rate: float = 1.0


class ActualHoldingDTO(BaseModel):
    """12-Column Holding schema matching Frontend Holdings Table & Backend Holding Model"""
    symbol: str
    asset_type: str
    units: float
    status: str = "active"
    archived_at: Optional[str] = None
    bucket_id: Optional[str] = None
    avg_cost_usd: Optional[float] = None
    avg_cost_thb: Optional[float] = None
    current_price_usd: Optional[float] = None
    current_price_thb: Optional[float] = None
    fx_rate: Optional[float] = None
    market_value_thb: float = 0.0
    unrealized_pnl_percent: Optional[float] = None
    unrealized_pnl_value: Optional[float] = None
    market_cap_tier: Optional[str] = None  # 'Mega', 'Large', 'Mid', 'Small', 'N/A'
    yield_on_cost: Optional[float] = None
    company_name: Optional[str] = None
    pe_ratio: Optional[float] = None
    eps: Optional[float] = None
    payout_ratio: Optional[float] = None
    market_cap_value: Optional[float] = None
    dividend_per_share: Optional[float] = None
    dividend_yield: Optional[float] = None
    accumulated_dividend_thb: Optional[float] = None
    accumulated_dividend_native: Optional[float] = None
    upcoming_dividend_thb: Optional[float] = None
    upcoming_dividend_native: Optional[float] = None
    dividend_rounds: list[DividendRoundDTO] = Field(default_factory=list)
    dividend_source: Optional[Literal["synced", "manual"]] = None
    fundamentals_updated_at: Optional[float] = None


class ActualSummaryDTO(BaseModel):
    total_value_thb: float = 0.0
    total_cost_basis_thb: float = 0.0
    total_unrealized_profit: float = 0.0
    total_realized_profit_ytd: float = 0.0
    passive_income_ytd: float = 0.0
    total_accumulated_dividend: float = 0.0


class AllocationTargetDTO(BaseModel):
    bucket_id: str
    name: str
    target_percent: float = Field(ge=0, le=100)
    color: Optional[str] = None


class ActualPortfolioStateDTO(BaseModel):
    last_updated: Optional[str] = None
    fx_rates: dict[str, float] = {}
    summary: ActualSummaryDTO
    allocation_targets: list[AllocationTargetDTO] = []
    holdings: list[ActualHoldingDTO] = []
    price_refresh_info: Optional[dict[str, str]] = None


class BucketAllocationSummaryDTO(BaseModel):
    bucket_id: str
    name: str
    target_percent: float
    actual_value_thb: float
    actual_percent: float
    variance: float
    color: Optional[str] = None


class BucketAllocationResponseDTO(BaseModel):
    warning: Optional[str] = None
    summaries: list[BucketAllocationSummaryDTO] = []


class ActualWatchlistItemDTO(BaseModel):
    symbol: str
    asset_type: str
    target_price: Optional[float] = None
    added_date: str
    notes: Optional[str] = None


class ActualWatchlistStateDTO(BaseModel):
    last_updated: Optional[str] = None
    items: list[ActualWatchlistItemDTO] = []


class PortfolioMetaDTO(BaseModel):
    id: str
    name: str
    is_default: bool = False
    created_at: Optional[str] = None


class CreatePortfolioRequestDTO(BaseModel):
    name: str
    portfolio_id: Optional[str] = None


class RenamePortfolioRequestDTO(BaseModel):
    name: str



class ActualGoalItemDTO(BaseModel):
    name: str
    target_amount_thb: float
    goal_type: str  # 'nav_target', 'cash_target', 'passive_income_ytd', 'bucket_target'
    current_amount_thb: float = 0.0
    progress_pct: float = 0.0
    deadline: Optional[str] = None
    deadline_days_left: Optional[int] = None
    notes: Optional[str] = None
    portfolio_id: Optional[str] = "default"
    bucket_id: Optional[str] = None


class ActualGoalsResponseDTO(BaseModel):
    n_goals: int
    goals: list[ActualGoalItemDTO] = []
    generated_at: Optional[str] = None


class PerformanceSnapshotDTO(BaseModel):
    Date: str
    Total_NAV: float
    Total_Cost: float
    Unrealized_PnL: float
    Cash_Balance: float
    realized_pnl_ytd: Optional[float] = None
    passive_income_ytd: Optional[float] = None


class JournalEntryDTO(BaseModel):
    timestamp: str
    content: str


class UpsertAllocationTargetsRequestDTO(BaseModel):
    targets: list[AllocationTargetDTO] = []


class AssignBucketRequestDTO(BaseModel):
    bucket_id: Optional[str] = None


class BatchAssignBucketRequestDTO(BaseModel):
    symbols: list[str] = []
    bucket_id: Optional[str] = None


class BatchRemoveHoldingsRequestDTO(BaseModel):
    symbols: list[str] = []


class TradeRequestDTO(BaseModel):
    symbol: str
    asset_type: str
    action: Literal["buy", "sell"]
    units: float
    price: float
    currency: Literal["THB", "USD"] = "THB"
    exchange_rate: Optional[float] = None
    date: Optional[str] = None
    notes: str = ""
    bucket_id: Optional[str] = None


class TransactionItemDTO(BaseModel):
    transaction_id: str
    timestamp: str
    symbol: str
    action: str
    units: float
    price: float
    currency: str
    fx_rate: Optional[float] = None
    cost_thb: float = 0.0
    realized_pnl_thb: Optional[float] = None
    notes: str = ""

    @field_validator("fx_rate", "realized_pnl_thb", mode="before")
    @classmethod
    def _coerce_opt_float(cls, v):
        if v is None or v == "":
            return None
        return float(v)

    @field_validator("cost_thb", "units", "price", mode="before")
    @classmethod
    def _coerce_req_float(cls, v):
        if v is None or v == "":
            return 0.0
        return float(v)


class TransactionSummaryDTO(BaseModel):
    total_buy_count: int = 0
    total_sell_count: int = 0
    total_buy_thb: float = 0.0
    total_sell_thb: float = 0.0
    total_realized_pnl_thb: float = 0.0


class TransactionListResponseDTO(BaseModel):
    portfolio_id: str
    transactions: list[TransactionItemDTO] = []
    summary: TransactionSummaryDTO = Field(default_factory=TransactionSummaryDTO)


class UpdateTransactionNoteRequestDTO(BaseModel):
    notes: str = ""


class EditTransactionRequestDTO(BaseModel):
    timestamp: Optional[str] = None
    units: Optional[float] = None
    price: Optional[float] = None
    fx_rate: Optional[float] = None
    notes: Optional[str] = None
    adjust_cash: bool = True


class DeleteTransactionRequestDTO(BaseModel):
    adjust_cash: bool = True


class FXRateResponseDTO(BaseModel):
    date: str
    currency_pair: str = "USDTHB"
    rate: float
    source: Literal["historical", "live", "fallback"]


class SyncDividendsResponseDTO(BaseModel):
    synced_symbols: int
    total_rounds: int
    total_received_rounds: int = 0
    total_upcoming_rounds: int = 0
    total_dividend_thb: float
    total_upcoming_thb: float = 0.0
    skipped_manual: list[str] = Field(default_factory=list)
    details: dict[str, list[DividendRoundDTO]]


class CashFlowRequestDTO(BaseModel):
    amount: float
    action: Literal["deposit", "withdraw"]
    currency: Literal["THB", "USD"] = "THB"
    exchange_rate: Optional[float] = None
    date: Optional[str] = None
    notes: str = ""


class IncomeRequestDTO(BaseModel):
    income_type: Literal["Dividend", "Interest", "Rental", "Other"]
    amount_thb: float
    source_symbol: Optional[str] = None
    date: Optional[str] = None
    notes: str = ""


class EditHoldingRequestDTO(BaseModel):
    units: Optional[float] = None
    avg_cost: Optional[float] = None
    accumulated_dividend_thb: Optional[float] = None
    asset_type: Optional[str] = None
    reason: str = ""
    bucket_id: Optional[str] = None


class UpsertWatchlistItemRequestDTO(BaseModel):
    asset_type: str
    target_price: Optional[float] = None
    notes: str = ""


class UpsertGoalRequestDTO(BaseModel):
    goal_type: Literal["nav_target", "cash_target", "passive_income_ytd", "bucket_target"]
    target_amount_thb: float
    deadline: Optional[str] = None
    years_from_now: Optional[int] = None
    notes: Optional[str] = None
    portfolio_id: Optional[str] = "default"
    bucket_id: Optional[str] = None


class AppendJournalRequestDTO(BaseModel):
    entry: str




class CalendarEventDTO(BaseModel):
    ticker: str
    company_name: Optional[str] = None
    event_type: Literal["earnings", "ex_dividend"]
    event_date: str           # ISO date "2026-10-30"
    days_until: int           # -N = past, 0 = today, N = future
    bucket: Literal["holding", "watchlist"]
    eps_estimate: Optional[float] = None
    eps_low: Optional[float] = None
    eps_high: Optional[float] = None


class PortfolioCalendarDTO(BaseModel):
    generated_at: str
    events: list[CalendarEventDTO] = []
    tickers_fetched: int = 0
    tickers_failed: list[str] = []


