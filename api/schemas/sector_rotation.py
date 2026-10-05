"""HTTP response models for sector rotation; chart history is a requested view slice."""
from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict, Field
from schemas.sector_rotation_schemas import (
    Quadrant, QuadrantTransitionEvent, RelativePricePoint, RotationPoint, ReturnMetric,
)


class SectorRotationRowDTO(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ticker: str
    name: str
    status: Literal["available", "partial", "unavailable"]
    reason: Optional[str] = None
    returns_pct: dict[str, Optional[float]] = {}
    return_metrics: dict[str, ReturnMetric] = {}
    price_as_of: Optional[str] = None
    rotation_as_of: Optional[str] = None
    relative_strength: Optional[float] = None
    relative_price_base_date: Optional[str] = None
    relative_trend: Optional[float] = None
    relative_momentum: Optional[float] = None
    quadrant: Optional[Quadrant] = None
    quadrant_changed_at: Optional[str] = None
    momentum_direction: Literal["rising", "falling", "flat", "unavailable"]
    history: list[RotationPoint] = []
    relative_price_history: list[RelativePricePoint] = []
    quadrant_transitions: list[QuadrantTransitionEvent] = []


class SectorRotationSnapshotDTO(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str
    formula_version: str
    calendar_version: str = "legacy"
    transition_rule_version: str = "legacy"
    formula_config: dict = Field(default_factory=dict)
    universe_version: str
    benchmark: str
    price_basis: str
    input_digest: str
    snapshot_id: str
    as_of_date: Optional[str] = None
    expected_session: Optional[str] = None
    expected_weekly_session: Optional[str] = None
    input_start_date: Optional[str] = None
    coverage: dict[str, int]
    expected_sectors: int
    available_sectors: int
    benchmark_status: Literal["available", "unavailable"]
    benchmark_reason: Optional[str] = None
    benchmark_returns_pct: dict[str, Optional[float]] = {}
    rows: list[SectorRotationRowDTO]


class SectorRotationResponseDTO(BaseModel):
    model_config = ConfigDict(extra="forbid")
    capability_status: Literal["enabled", "disabled"]
    refresh_state: Literal["idle", "running", "failed"]
    retry_after_seconds: Optional[int] = None
    error_code: Optional[str] = None
    last_attempt_at: Optional[str] = None
    expected_session: Optional[str] = None
    freshness: Literal["fresh", "stale", "unknown"] = "unknown"
    missing_sessions: int = 0
    served_at: str
    timeframe: Literal["daily", "weekly"]
    tail: int
    summary: Optional["SectorRotationSummaryDTO"] = None
    snapshot: Optional[SectorRotationSnapshotDTO] = None


class SectorExcessRankDTO(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ticker: str
    name: str
    excess_return_pp: float
    as_of: Optional[str] = None
    status: Literal["available", "partial"]
    valid_sessions: int
    expected_sessions: int


class SectorBreadthDTO(BaseModel):
    model_config = ConfigDict(extra="forbid")
    outperforming: int
    valid_sectors: int
    expected_sectors: int
    status: Literal["complete", "partial"]
    as_of: Optional[str] = None


class SectorRotationSummaryDTO(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary_version: str
    timeframe: Literal["daily", "weekly"]
    rotation_as_of: Optional[str] = None
    ranked_by_excess_3m: list[SectorExcessRankDTO] = Field(default_factory=list)
    sector_breadth_3m: SectorBreadthDTO
    quadrant_members: dict[str, list[str]] = Field(default_factory=dict)
    periods_in_quadrant: dict[str, Optional[int]] = Field(default_factory=dict)
    elapsed_days_in_quadrant: dict[str, Optional[int]] = Field(default_factory=dict)
    momentum_delta: dict[str, Optional[float]] = Field(default_factory=dict)
    heading_deg: dict[str, Optional[float]] = Field(default_factory=dict)


class SectorRotationHistoryRowDTO(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ticker: str
    name: str
    status: Literal["available", "partial", "unavailable"]
    reason: Optional[str] = None
    relative_price_base_date: Optional[str] = None
    history: list[RotationPoint] = Field(default_factory=list)
    relative_price_history: list[RelativePricePoint] = Field(default_factory=list)
    quadrant_transitions: list[QuadrantTransitionEvent] = Field(default_factory=list)


class SectorRotationHistoryDTO(BaseModel):
    model_config = ConfigDict(extra="forbid")
    snapshot_id: str
    input_digest: str
    formula_version: str
    timeframe: Literal["daily", "weekly"]
    range: Literal["3m", "6m", "1y", "2y"]
    from_date: str
    to_date: Optional[str] = None
    rows: list[SectorRotationHistoryRowDTO]


SectorRotationResponseDTO.model_rebuild()
