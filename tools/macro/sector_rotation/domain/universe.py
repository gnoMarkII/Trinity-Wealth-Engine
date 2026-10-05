"""Versioned sector universe shared by ingestion, API and AI."""
from tools.macro.ticker_config import _US_SECTORS

UNIVERSE_VERSION = "spdr-select-sector-11-v1"
BENCHMARK = "SPY"
SECTOR_TICKERS: tuple[str, ...] = (
    "XLK", "XLC", "XLY", "XLF", "XLI", "XLB", "XLE", "XLV", "XLP", "XLU", "XLRE"
)


def sector_name(ticker: str) -> str:
    return _US_SECTORS.get(ticker, (ticker, ""))[0]

