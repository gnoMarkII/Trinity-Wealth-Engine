from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Literal, Optional, List, Dict, Union, Tuple
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .constants import CASH_THB_SYMBOL, CASH_USD_SYMBOL, CASH_SYMBOL

UNITS_QUANTUM = Decimal("0.000001")   # 6 decimal places for fractional shares
PRICE_QUANTUM = Decimal("0.0001")     # 4 decimal places for share prices
MONEY_QUANTUM = Decimal("0.01")       # 2 decimal places for gross, fees, net, cash
FX_QUANTUM = Decimal("0.0001")        # 4 decimal places for FX rate


def quantize_decimal(val: Union[Decimal, float, str, int], quantum: Decimal = MONEY_QUANTUM) -> Decimal:
    """Quantize a value to standard financial decimal precision."""
    if val is None:
        return Decimal("0.00").quantize(quantum, rounding=ROUND_HALF_UP)
    d = Decimal(str(val)) if not isinstance(val, Decimal) else val
    return d.quantize(quantum, rounding=ROUND_HALF_UP)


class TradeFeeBreakdown(BaseModel):
    model_config = ConfigDict(extra="allow")

    commission: Decimal = Decimal("0.00")
    vat: Decimal = Decimal("0.00")
    other_fees: Decimal = Decimal("0.00")
    fee_currency: str = "THB"

    @property
    def total_fees(self) -> Decimal:
        return quantize_decimal(self.commission + self.vat + self.other_fees, MONEY_QUANTUM)


class TradeImportItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    item_id: str
    trade_date: str
    settlement_date: Optional[str] = None
    symbol: str
    action: Literal["BUY", "SELL"]
    units: Decimal
    price: Decimal
    gross_amount: Decimal
    fees: TradeFeeBreakdown = Field(default_factory=TradeFeeBreakdown)
    net_amount: Decimal
    currency: str = "THB"
    exchange_rate: Optional[Decimal] = None
    confirmation_no: str
    order_id: Optional[str] = None
    source: str = "DIME"
    fingerprint: str
    line_index: int = 0
    cash_adjusted: bool = True
    asset_type: str = "Stock"

    @property
    def transaction_identity(self) -> Tuple[str, str]:
        """Unique identity across Dime trades: (confirmation_no, order_id)."""
        if not self.order_id:
            raise ValueError(f"รายการ {self.symbol} ขาด order_id ไม่สามารถระบุตัวตนของธุรกรรมได้")
        return (self.confirmation_no.strip(), self.order_id.strip())


def _coerce_iso_string(v):
    """PyYAML implicit-types ISO 8601 strings -> datetime; coerce back to str."""
    if hasattr(v, "isoformat"):
        return v.isoformat(timespec="seconds") if hasattr(v, "hour") else v.isoformat()
    return v


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


class AllocationTarget(BaseModel):
    model_config = ConfigDict(extra="allow")

    bucket_id: str
    name: str
    target_percent: float = Field(ge=0, le=100)
    color: Optional[str] = None


def default_allocation_targets() -> List[AllocationTarget]:
    return [
        AllocationTarget(bucket_id="core_equities", name="Core Equities", target_percent=60.0, color="#3B82F6"),
        AllocationTarget(bucket_id="defensive", name="Defensive Assets", target_percent=20.0, color="#A855F7"),
        AllocationTarget(bucket_id="cash", name="💰 Cash & Equivalents", target_percent=20.0, color="#06B6D4"),
    ]


class DividendRound(BaseModel):
    model_config = ConfigDict(extra="allow")

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


class Holding(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_version: int = 1
    symbol: str
    asset_type: str
    units: float
    status: Literal["active", "archived"] = "active"
    archived_at: Optional[str] = None

    avg_cost_thb: Optional[float] = None
    avg_cost_usd: Optional[float] = None
    current_price_thb: Optional[float] = None
    current_price_usd: Optional[float] = None
    fx_rate: Optional[float] = None

    market_value_thb: float = 0.0
    unrealized_pnl_percent: Optional[float] = None
    accumulated_dividend_thb: Optional[float] = None
    accumulated_dividend_native: Optional[float] = None
    upcoming_dividend_thb: Optional[float] = None
    upcoming_dividend_native: Optional[float] = None
    dividend_rounds: List[DividendRound] = Field(default_factory=list)
    dividend_source: Optional[Literal["synced", "manual"]] = None
    bucket_id: Optional[str] = None
    fundamentals_updated_at: Optional[float] = None

    company_name: Optional[str] = None
    pe_ratio: Optional[float] = None
    eps: Optional[float] = None
    payout_ratio: Optional[float] = None
    market_cap_value: Optional[float] = None
    dividend_per_share: Optional[float] = None
    dividend_yield: Optional[float] = None

    @property
    def units_decimal(self) -> Decimal:
        return quantize_decimal(Decimal(str(self.units)), UNITS_QUANTUM)

    @property
    def units_str(self) -> str:
        d = Decimal(str(self.units))
        return f"{d:f}".rstrip("0").rstrip(".") if "." in f"{d:f}" else f"{d:f}"

    @property
    def avg_cost_thb_decimal(self) -> Optional[Decimal]:
        return quantize_decimal(Decimal(str(self.avg_cost_thb)), PRICE_QUANTUM) if self.avg_cost_thb is not None else None

    @property
    def avg_cost_usd_decimal(self) -> Optional[Decimal]:
        return quantize_decimal(Decimal(str(self.avg_cost_usd)), PRICE_QUANTUM) if self.avg_cost_usd is not None else None


class Summary(BaseModel):
    model_config = ConfigDict(extra="allow")

    total_value_thb: float = 0.0
    total_cost_basis_thb: float = 0.0
    total_unrealized_profit: float = 0.0
    total_realized_profit_ytd: float = 0.0
    passive_income_ytd: float = 0.0
    total_accumulated_dividend: float = 0.0


class PortfolioState(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_version: int = 1
    doc_type: Literal["portfolio_master"] = "portfolio_master"
    name: Optional[str] = None
    last_updated: str

    base_currency: str = "THB"
    summary: Summary = Field(default_factory=Summary)
    fx_rates: Dict[str, float] = Field(default_factory=lambda: {"USDTHB": 36.5})
    allocation_targets: List[AllocationTarget] = Field(default_factory=default_allocation_targets)
    holdings: List[Holding] = Field(default_factory=list)
    price_refresh_info: Optional[Dict[str, str]] = None

    @field_validator("last_updated", mode="before")
    @classmethod
    def _validate_last_updated(cls, v):
        return _coerce_iso_string(v)


class WatchlistItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_version: int = 1
    symbol: str
    asset_type: str
    target_price: Optional[float] = None
    notes: Optional[str] = None
    added_date: str


class WatchlistState(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_version: int = 1
    doc_type: Literal["watchlist"] = "watchlist"
    last_updated: str
    items: List[WatchlistItem] = Field(default_factory=list)

    @field_validator("last_updated", mode="before")
    @classmethod
    def _validate_last_updated(cls, v):
        return _coerce_iso_string(v)


class PortfolioMeta(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    name: str
    is_default: bool = False
    created_at: str = Field(default_factory=_now_iso)

    @field_validator("created_at", mode="before")
    @classmethod
    def _validate_created_at(cls, v):
        if v is None:
            return _now_iso()
        return _coerce_iso_string(v)


class GoalItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_version: int = 1
    name: str
    goal_type: Literal["nav_target", "cash_target", "passive_income_ytd", "bucket_target"]
    target_amount_thb: float
    deadline: Optional[str] = None
    years_from_now: Optional[int] = None
    notes: Optional[str] = None
    created_date: str
    portfolio_id: str = "default"
    bucket_id: Optional[str] = None


class GoalsState(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_version: int = 1
    doc_type: Literal["goals"] = "goals"
    last_updated: str
    goals: List[GoalItem] = Field(default_factory=list)

    @field_validator("last_updated", mode="before")
    @classmethod
    def _validate_last_updated(cls, v):
        return _coerce_iso_string(v)


class PerformanceSnapshot(BaseModel):
    model_config = ConfigDict(extra="allow")

    date: str
    total_nav: float
    total_cost: float
    unrealized_pnl: float
    cash_balance: float
    realized_pnl_ytd: float = 0.0
    passive_income_ytd: float = 0.0


class PerformanceState(BaseModel):
    model_config = ConfigDict(extra="allow")

    snapshots: List[PerformanceSnapshot] = Field(default_factory=list)


class JournalEntry(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: Optional[str] = None
    date: str
    content: str
    portfolio_id: str = "default"

