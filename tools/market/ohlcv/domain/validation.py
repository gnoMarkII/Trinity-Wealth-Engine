"""Domain validation and timeframe capabilities for OHLCV Market Data."""
import re
from typing import Dict, List, Set

TIMEFRAME_CAPABILITIES: dict[str, list[str]] = {
    "15m": ["5d", "1mo"],
    "1h": ["1mo", "3mo", "6mo", "1y", "2y"],
    "1d": ["1mo", "3mo", "6mo", "1y", "5y", "max"],
    "1wk": ["1y", "5y", "max"],
    "1mo": ["5y", "max"],
}

ALLOWED_RANGES: set[str] = {"5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"}
ALLOWED_INTERVALS: set[str] = {"15m", "1h", "1d", "1wk", "1mo"}


def validate_ticker(ticker: str) -> str:
    """Validate ticker string format and prevent path traversal."""
    ticker = ticker.upper().strip()
    if not re.match(r"^[A-Z0-9.\-_]+$", ticker):
        raise ValueError("Invalid ticker format")
    if ".." in ticker or "/" in ticker or "\\" in ticker:
        raise ValueError("Path traversal not allowed in ticker")
    return ticker


def validate_interval_and_range(interval: str, range_str: str) -> None:
    """Validate interval and range combination against capability matrix."""
    allowed_for_interval = TIMEFRAME_CAPABILITIES.get(interval)
    if allowed_for_interval is None:
        raise ValueError(
            f"Invalid interval '{interval}'. Allowed intervals are: {sorted(TIMEFRAME_CAPABILITIES.keys())}"
        )
    if range_str not in allowed_for_interval:
        raise ValueError(
            f"Invalid interval and range combination '{interval}' with '{range_str}'. Allowed ranges for '{interval}' are: {allowed_for_interval}"
        )
