"""Point-in-Time (PIT) Fundamentals Ledger & Valuation Band Engine (Phase 4 & v3.1).

Manages point-in-time valuation multiples without look-ahead bias.
Enforces the hard dependency rule: If PIT ledger lacks sufficient historical depth,
historical multiple bands are marked 'unavailable' instead of backfilling from today's TTM.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

from core.logger import get_logger
from schemas.micro_quant_schemas import DataStatus
from tools._atomic_io import _atomic_write_to
from tools.archivist.core import VAULT_PATH
from tools.archivist.maintenance_guard import assert_write_allowed

log = get_logger(__name__)

_DEFAULT_PIT_DIR = VAULT_PATH / "30_Knowledge_Base" / ".pit_ledger"
_MIN_PIT_OBSERVATIONS_FOR_BANDS = 250  # ~1 trading year


class PITLedgerEntry(BaseModel):
    as_of_date: str  # YYYY-MM-DD
    ticker: str
    valuation_price: float  # Unadjusted/split-only price
    ttm_eps: Optional[float] = None
    ttm_sales_per_share: Optional[float] = None
    pe_multiple: Optional[float] = None
    ps_multiple: Optional[float] = None
    recorded_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class PITMultiplesBands(BaseModel):
    ticker: str
    observations_count: int
    pe_median_3y: Optional[float] = None
    pe_p25_3y: Optional[float] = None
    pe_p75_3y: Optional[float] = None
    current_pe_percentile: Optional[float] = None
    status: DataStatus = "available"
    flags: list[str] = Field(default_factory=list)


def append_pit_entry(
    entry: PITLedgerEntry,
    base_dir: Optional[Path] = None,
) -> Path:
    """Appends an immutable PIT snapshot to the ticker's ledger file."""
    target_dir = base_dir or _DEFAULT_PIT_DIR
    assert_write_allowed(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = target_dir / f"{entry.ticker.upper()}.json"
    
    existing_entries: list[dict] = []
    if file_path.exists():
        try:
            existing_entries = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception:
            existing_entries = []

    # Check for duplicate as_of_date
    existing_dates = {e.get("as_of_date") for e in existing_entries}
    if entry.as_of_date not in existing_dates:
        existing_entries.append(entry.model_dump())
        existing_entries.sort(key=lambda x: x.get("as_of_date", ""))

    _atomic_write_to(file_path, json.dumps(existing_entries, indent=2))
    return file_path


def compute_pit_multiple_bands(
    ticker: str,
    current_pe: Optional[float],
    base_dir: Optional[Path] = None,
) -> PITMultiplesBands:
    """Calculates Point-in-Time Multiple Bands from true historical ledger.

    Invariant:
        If observations < 250, returns status='unavailable' with flag
        'insufficient_pit_history'. NEVER backfills from current TTM.
    """
    target_dir = base_dir or _DEFAULT_PIT_DIR
    file_path = target_dir / f"{ticker.upper()}.json"

    if not file_path.exists():
        return PITMultiplesBands(
            ticker=ticker.upper(),
            observations_count=0,
            status="unavailable",
            flags=["pit_ledger_not_initialized:historical_bands_unavailable"],
        )

    try:
        entries = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("Failed to read PIT ledger for %s: %s", ticker, e)
        return PITMultiplesBands(
            ticker=ticker.upper(),
            observations_count=0,
            status="unavailable",
            flags=["pit_ledger_read_error"],
        )

    valid_pes = [
        float(e["pe_multiple"])
        for e in entries
        if e.get("pe_multiple") is not None and float(e["pe_multiple"]) > 0
    ]

    if len(valid_pes) < _MIN_PIT_OBSERVATIONS_FOR_BANDS:
        return PITMultiplesBands(
            ticker=ticker.upper(),
            observations_count=len(valid_pes),
            status="unavailable",
            flags=[f"insufficient_pit_history:{len(valid_pes)}_of_{_MIN_PIT_OBSERVATIONS_FOR_BANDS}_required"],
        )

    sorted_pes = sorted(valid_pes)
    n = len(sorted_pes)
    p25 = sorted_pes[int(n * 0.25)]
    median = sorted_pes[int(n * 0.50)]
    p75 = sorted_pes[int(n * 0.75)]

    percentile = None
    if current_pe is not None:
        count_below = sum(1 for p in sorted_pes if p <= current_pe)
        percentile = round((count_below / n) * 100.0, 1)

    return PITMultiplesBands(
        ticker=ticker.upper(),
        observations_count=n,
        pe_median_3y=round(median, 1),
        pe_p25_3y=round(p25, 1),
        pe_p75_3y=round(p75, 1),
        current_pe_percentile=percentile,
        status="available",
    )
