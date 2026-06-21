import concurrent.futures
import csv
import json
import os
import re
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import frontmatter
import yfinance as yf
from filelock import FileLock, Timeout
from langchain_core.tools import tool


def _coerce_iso_string(v):
    """PyYAML implicit-types ISO 8601 strings → datetime; coerce กลับเป็น str
    กัน ValidationError เมื่อไฟล์ถูกแก้ด้วยมือ (ไม่มี quotes รอบค่า)
    """
    if hasattr(v, "isoformat"):
        return v.isoformat(timespec="seconds") if hasattr(v, "hour") else v.isoformat()
    return v

def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")

from pydantic import BaseModel, ConfigDict, Field, field_validator

from core.logger import get_logger

log = get_logger(__name__)

from tools._atomic_io import _atomic_write_to


VAULT_PATH = Path(os.getenv("OBSIDIAN_VAULT_PATH", "./memories"))
PORTFOLIO_REL = os.getenv("PORTFOLIO_FILE", "20_Portfolio_Management/Current_Holdings/Portfolio_Holdings.md")
PORTFOLIO_PATH = VAULT_PATH / PORTFOLIO_REL
TRADING_JOURNAL_REL = os.getenv(
    "TRADING_JOURNAL_FILE",
    "20_Portfolio_Management/Journals_and_Reports/Trading_Journal.md",
)
TRADING_JOURNAL_PATH = VAULT_PATH / TRADING_JOURNAL_REL
WATCHLIST_REL = os.getenv(
    "WATCHLIST_FILE",
    "20_Portfolio_Management/Current_Holdings/Watchlist.md",
)
WATCHLIST_PATH = VAULT_PATH / WATCHLIST_REL
PERFORMANCE_LOG_REL = os.getenv(
    "PERFORMANCE_LOG_FILE",
    "20_Portfolio_Management/Journals_and_Reports/Performance_Log.csv",
)
PERFORMANCE_LOG_PATH = VAULT_PATH / PERFORMANCE_LOG_REL
_PERFORMANCE_LOG_HEADER = ["Date", "Total_NAV", "Total_Cost", "Unrealized_PnL", "Cash_Balance"]

# Derived sidecar folders (master = Portfolio_Holdings.md / Watchlist.md)
HOLDINGS_DIR = VAULT_PATH / "20_Portfolio_Management/Current_Holdings/Holdings"
WATCHLIST_ITEMS_DIR = VAULT_PATH / "20_Portfolio_Management/Current_Holdings/WatchlistItems"

CASH_THB_SYMBOL = "CASH_THB"
CASH_USD_SYMBOL = "CASH_USD"
_CASH_SYMBOLS = (CASH_THB_SYMBOL, CASH_USD_SYMBOL)
# Back-compat alias — call sites and tests still reference CASH_SYMBOL
CASH_SYMBOL = CASH_THB_SYMBOL

_FLOAT_EPS = 1e-6
_MONEY_DP = 2
_COST_DP = 4
_PCT_DP = 2

_LOCK_TIMEOUT = 15  # seconds — wait up to 15s for another process to release
_PRICE_FETCH_TIMEOUT = 6  # seconds per symbol when refreshing

_PORTFOLIO_LOCK_PATH = str(PORTFOLIO_PATH) + ".lock"
_portfolio_lock = FileLock(_PORTFOLIO_LOCK_PATH, timeout=_LOCK_TIMEOUT)



class Holding(BaseModel):
    model_config = ConfigDict(extra="allow")

    symbol: str
    asset_type: str
    units: float

    avg_cost_thb: float | None = None
    avg_cost_usd: float | None = None
    current_price_thb: float | None = None
    current_price_usd: float | None = None
    fx_rate: float | None = None

    market_value_thb: float = 0.0
    unrealized_pnl_percent: float | None = None
    accumulated_dividend_thb: float | None = None


class Summary(BaseModel):
    model_config = ConfigDict(extra="allow")

    total_value_thb: float = 0.0
    total_unrealized_profit: float = 0.0
    total_realized_profit_ytd: float = 0.0
    passive_income_ytd: float = 0.0
    total_accumulated_dividend: float = 0.0


class PortfolioState(BaseModel):
    model_config = ConfigDict(extra="allow")

    doc_type: Literal["portfolio_master"] = "portfolio_master"
    last_updated: str
    base_currency: str = "THB"
    summary: Summary = Field(default_factory=Summary)
    fx_rates: dict[str, float] = Field(default_factory=lambda: {"USDTHB": 36.5})
    holdings: list[Holding] = Field(default_factory=list)

    @field_validator("last_updated", mode="before")
    @classmethod
    def _validate_last_updated(cls, v):
        return _coerce_iso_string(v)


class WatchlistItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    symbol: str
    asset_type: str
    target_price: float | None = None
    notes: str | None = None
    added_date: str


class WatchlistState(BaseModel):
    model_config = ConfigDict(extra="allow")

    doc_type: Literal["watchlist"] = "watchlist"
    last_updated: str
    items: list[WatchlistItem] = Field(default_factory=list)

    @field_validator("last_updated", mode="before")
    @classmethod
    def _validate_last_updated(cls, v):
        return _coerce_iso_string(v)


class GoalItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    goal_type: Literal["nav_target", "cash_target", "passive_income_ytd"]
    target_amount_thb: float
    deadline: str | None = None
    notes: str | None = None
    created_date: str


class GoalsState(BaseModel):
    model_config = ConfigDict(extra="allow")

    doc_type: Literal["goals"] = "goals"
    last_updated: str
    goals: list[GoalItem] = Field(default_factory=list)

    @field_validator("last_updated", mode="before")
    @classmethod
    def _validate_last_updated(cls, v):
        return _coerce_iso_string(v)


