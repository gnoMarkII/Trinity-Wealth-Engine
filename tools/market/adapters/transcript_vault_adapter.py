"""Transcript and Promise Ledger file persistence adapter.

Handles atomic file caching of raw transcript texts and structured promise ledgers
under data/cache/transcripts/{TICKER}/ using _atomic_io._atomic_write_to.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools._atomic_io import _atomic_write_to

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_TRANSCRIPT_CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "transcripts"


def get_ticker_dir(ticker: str, base_dir: Optional[Path] = None) -> Path:
    """Get the sanitized cache directory path for a ticker."""
    clean_ticker = ticker.strip().upper()
    root = base_dir or DEFAULT_TRANSCRIPT_CACHE_DIR
    target_dir = root / clean_ticker
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def save_transcript_text(
    ticker: str,
    quarter: str,
    text: str,
    base_dir: Optional[Path] = None,
) -> Path:
    """Save transcript plain text atomically."""
    clean_quarter = quarter.strip().upper()
    ticker_dir = get_ticker_dir(ticker, base_dir=base_dir)
    file_path = ticker_dir / f"{clean_quarter}.txt"
    _atomic_write_to(file_path, text)
    log.debug("Saved transcript for %s %s to %s", ticker, clean_quarter, file_path)
    return file_path


def load_transcript_text(
    ticker: str,
    quarter: str,
    base_dir: Optional[Path] = None,
) -> Optional[str]:
    """Load transcript plain text if present in cache."""
    clean_quarter = quarter.strip().upper()
    ticker_dir = get_ticker_dir(ticker, base_dir=base_dir)
    file_path = ticker_dir / f"{clean_quarter}.txt"
    if file_path.is_file():
        try:
            return file_path.read_text(encoding="utf-8")
        except Exception as exc:
            log.warning("Failed to read cached transcript at %s: %s", file_path, exc)
            return None
    return None


def save_promise_ledger(
    ticker: str,
    ledger_data: Dict[str, Any],
    base_dir: Optional[Path] = None,
) -> Path:
    """Save promise ledger JSON atomically."""
    ticker_dir = get_ticker_dir(ticker, base_dir=base_dir)
    file_path = ticker_dir / "promise_ledger.json"
    content = json.dumps(ledger_data, indent=2, ensure_ascii=False)
    _atomic_write_to(file_path, content)
    log.debug("Saved promise ledger for %s to %s", ticker, file_path)
    return file_path


def load_promise_ledger(
    ticker: str,
    base_dir: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Load promise ledger JSON if present in cache."""
    ticker_dir = get_ticker_dir(ticker, base_dir=base_dir)
    file_path = ticker_dir / "promise_ledger.json"
    if file_path.is_file():
        try:
            return json.loads(file_path.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("Failed to read promise ledger at %s: %s", file_path, exc)
            return None
    return None


def list_cached_transcripts(
    ticker: str,
    base_dir: Optional[Path] = None,
) -> List[str]:
    """List quarters for which transcripts are cached on disk."""
    ticker_dir = get_ticker_dir(ticker, base_dir=base_dir)
    quarters = []
    for f in ticker_dir.glob("*.txt"):
        quarters.append(f.stem)
    return sorted(quarters)





class ObsidianTranscriptVaultAdapter:
    """Concrete adapter for transcript and promise ledger persistence."""

    def __init__(self, base_dir: Optional[Path] = None):
        self._base_dir = base_dir

    def save_transcript(self, ticker: str, quarter: str, text: str) -> str:
        p = save_transcript_text(ticker, quarter, text, base_dir=self._base_dir)
        return str(p)

    def load_transcript(self, ticker: str, quarter: str) -> Optional[str]:
        return load_transcript_text(ticker, quarter, base_dir=self._base_dir)

    def list_cached(self, ticker: str) -> List[str]:
        return list_cached_transcripts(ticker, base_dir=self._base_dir)

    def save_ledger(self, ticker: str, ledger: dict) -> str:
        p = save_promise_ledger(ticker, ledger, base_dir=self._base_dir)
        return str(p)

    def load_ledger(self, ticker: str) -> Optional[dict]:
        return load_promise_ledger(ticker, base_dir=self._base_dir)


class StockAnalysisTranscriptScraperAdapter:
    """Concrete adapter for transcript discovery and scraping."""

    def discover_calls(self, ticker: str, limit: int = 8) -> List[dict]:
        from tools.market.transcript_scraper import discover_stockanalysis_calls
        return discover_stockanalysis_calls(ticker, limit=limit)

    def fetch_transcript(self, call: Any) -> tuple[str, int]:
        from tools.market.transcript_scraper import fetch_transcript_text
        return fetch_transcript_text(call)
