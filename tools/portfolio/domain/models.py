from datetime import datetime
from typing import Literal, Optional, List, Dict
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .constants import CASH_THB_SYMBOL, CASH_USD_SYMBOL, CASH_SYMBOL


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

