"""Typed contracts for the deterministic US sector-rotation dataset."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


Quadrant = Literal["Leading", "Weakening", "Lagging", "Improving"]
Timeframe = Literal["daily", "weekly"]


class ReturnMetric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    absolute_return_pct: Optional[float] = None
    excess_return_pp: Optional[float] = None
    relative_return_pct: Optional[float] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    expected_sessions: int = 0
    valid_sessions: int = 0
    status: Literal["available", "partial", "unavailable"] = "unavailable"
    freshness: Literal["fresh", "stale", "unknown"] = "unknown"
    reason: Optional[str] = None


class RotationPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    as_of: str
    relative_trend: Optional[float] = None
    relative_momentum: Optional[float] = None
    quadrant: Optional[Quadrant] = None
    status: Literal["available", "unavailable"] = "available"
    reason: Optional[str] = None


class RelativePricePoint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    as_of: str
    sector_spy_rebased_100: Optional[float] = None
    status: Literal["available", "unavailable"] = "available"
    reason: Optional[str] = None


class QuadrantTransitionEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event_id: str
    timeframe: Timeframe
    previous_valid_at: str
    changed_at: Optional[str] = None
    confirmed_at: Optional[str] = None
    from_quadrant: Quadrant
    to_quadrant: Quadrant
    event_type: Literal["transition", "confirmed_transition"] = "confirmed_transition"


class SectorRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticker: str
    name: str
    status: Literal["available", "partial", "unavailable"]
    reason: Optional[str] = None
    returns_pct: dict[str, Optional[float]] = Field(default_factory=dict)
    return_metrics: dict[str, ReturnMetric] = Field(default_factory=dict)
    price_as_of: Optional[str] = None
    rotation_as_of: Optional[str] = None
    relative_strength: Optional[float] = None
    relative_price_base_date: Optional[str] = None
    relative_trend: Optional[float] = None
    relative_momentum: Optional[float] = None
    quadrant: Optional[Quadrant] = None
    quadrant_changed_at: Optional[str] = None
    momentum_direction: Literal["rising", "falling", "flat", "unavailable"] = "unavailable"
    history: dict[Timeframe, list[RotationPoint]] = Field(default_factory=dict)
    relative_price_history: dict[Timeframe, list[RelativePricePoint]] = Field(default_factory=dict)
    quadrant_transitions: list[QuadrantTransitionEvent] = Field(default_factory=list)


class SectorRotationSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "sector-rotation-snapshot-v2"
    formula_version: str = "relative-rotation-v2"
    calendar_version: str = "legacy"
    transition_rule_version: str = "legacy"
    formula_config: dict[str, Any] = Field(default_factory=dict)
    universe_version: str = "spdr-select-sector-11-v1"
    benchmark: str = "SPY"
    price_basis: str = "auto_adjusted_close"
    input_digest: str
    snapshot_id: str
    as_of_date: Optional[str] = None
    expected_session: Optional[str] = None
    expected_weekly_session: Optional[str] = None
    input_start_date: Optional[str] = None
    coverage: dict[str, int]
    expected_sectors: int = 11
    available_sectors: int = 0
    benchmark_status: Literal["available", "unavailable"]
    benchmark_reason: Optional[str] = None
    benchmark_returns_pct: dict[str, Optional[float]] = Field(default_factory=dict)
    rows: list[SectorRow]

    @model_validator(mode="after")
    def universe_is_complete(self) -> "SectorRotationSnapshot":
        tickers = [row.ticker for row in self.rows]
        if len(tickers) != len(set(tickers)):
            raise ValueError("sector rows must contain unique tickers")
        if self.available_sectors != sum(row.status != "unavailable" for row in self.rows):
            raise ValueError("available_sectors must match row statuses")
        return self


class SectorRotationEnvelope(BaseModel):
    """HTTP envelope separates refresh state from immutable snapshot content."""

    capability_status: Literal["enabled", "disabled"] = "enabled"
    refresh_state: Literal["idle", "running", "failed"] = "idle"
    retry_after_seconds: Optional[int] = None
    error_code: Optional[str] = None
    last_attempt_at: Optional[str] = None
    expected_session: Optional[str] = None
    freshness: Literal["fresh", "stale", "unknown"] = "unknown"
    missing_sessions: int = 0
    served_at: str
    snapshot: Optional[SectorRotationSnapshot] = None


class SectorFactClaim(BaseModel):
    """LLM-authored interpretation request; facts must be resolved in Python."""

    model_config = ConfigDict(extra="forbid")

    ticker: str
    claim_kind: Literal[
        "positive_excess", "negative_excess", "quadrant_membership",
        "quadrant_transition", "momentum_rising", "momentum_falling",
    ]
    metric_ref: str
    event_ref: Optional[str] = None
    macro_observable_refs: list[str] = Field(default_factory=list)
    interpretation_th: str


class ResolvedMetricClaim(BaseModel):
    """Resolved fact has either a numeric value or a category, never both."""

    model_config = ConfigDict(extra="forbid")

    ticker: str
    metric_ref: str
    numeric_value: Optional[float] = None
    categorical_value: Optional[str] = None
    unit: str
    horizon: str
    metric_as_of: str
    snapshot_id: str
    input_refs: list[str]

    @model_validator(mode="after")
    def exactly_one_value(self) -> "ResolvedMetricClaim":
        if (self.numeric_value is None) == (self.categorical_value is None):
            raise ValueError("exactly one of numeric_value or categorical_value is required")
        return self


class WatchCondition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric_ref: str
    operator: Literal[">", ">=", "<", "<=", "crosses_above", "crosses_below"]
    future_threshold: float
    unit: str
    horizon: str
    reason: str
