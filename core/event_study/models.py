"""Pure domain models for Event Study & Historical Scenario Analysis.

Pure Python primitives and dataclasses only — strictly zero external dependencies (no FastAPI, Pydantic,
SQLite, or yfinance) per Hexagonal Architecture AST Domain Layer Isolation rules.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class DailyPriceBar:
    """Historical daily price bar in chronological order."""
    date: str       # YYYY-MM-DD
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


@dataclass(frozen=True)
class EventPathPoint:
    """A single day's cumulative performance along the post-event trajectory."""
    day_offset: int                 # 1, 2, ..., N trading days after event
    date: str                       # YYYY-MM-DD
    close: float
    cumulative_return_pct: float    # e.g. +3.5 for +3.50%


@dataclass
class EventInstanceResult:
    """Individual event execution result and forward performance."""
    event_id: str
    event_date: str                 # YYYY-MM-DD of the event announcement
    entry_date: str                 # YYYY-MM-DD of the entry (next trading day t+1)
    entry_price: float              # Open price at t+1 (or fallback close)
    description: Optional[str] = None
    # Horizon in trading days (e.g. 5, 10, 20, 60) -> return percentage (e.g. 4.12)
    forward_returns: Dict[int, Optional[float]] = field(default_factory=dict)
    # Day-by-day cumulative returns from day 1 to max horizon
    daily_trajectory: List[EventPathPoint] = field(default_factory=list)
    # Win indicator by horizon: True if return > 0, False if return <= 0, None if insufficient bars
    is_win_by_horizon: Dict[int, Optional[bool]] = field(default_factory=dict)


@dataclass
class HorizonSummaryStats:
    """Statistical distribution profile for a specific forward horizon (e.g. 5d, 10d, 20d, 60d)."""
    horizon_days: int
    sample_size: int                # N events evaluated
    hit_rate_pct: float             # Win rate percentage (0.0 to 100.0)
    avg_return_pct: float           # Mean forward return %
    median_return_pct: float        # Median forward return %
    best_return_pct: Optional[float] = None
    worst_return_pct: Optional[float] = None
    avg_win_pct: Optional[float] = None
    avg_loss_pct: Optional[float] = None
    risk_reward_ratio: Optional[float] = None   # Avg Win / |Avg Loss|
    win_count: int = 0
    loss_count: int = 0


@dataclass(frozen=True)
class EventStudyGuardrailWarning:
    """Statistical or data completeness warning."""
    code: str                       # e.g. "UNRELIABLE_SAMPLE_SIZE", "INSUFFICIENT_BARS"
    message: str


@dataclass
class AverageTrajectoryPoint:
    """Average cumulative return at a given trading day offset across all events."""
    day_offset: int
    avg_return_pct: float
    sample_count: int


@dataclass
class EventStudyReport:
    """Complete institutional Event Study analysis report."""
    ticker: str
    event_name: str
    horizons: List[int]
    events: List[EventInstanceResult]
    horizon_stats: Dict[int, HorizonSummaryStats]
    average_trajectory: List[AverageTrajectoryPoint]
    warnings: List[EventStudyGuardrailWarning] = field(default_factory=list)
