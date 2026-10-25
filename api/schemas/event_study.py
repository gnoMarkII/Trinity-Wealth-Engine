"""Pydantic schemas and DTOs for Equity Event Study & Historical Scenario Analysis."""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class EventPresetItemDTO(BaseModel):
    preset_id: str
    name: str
    category: str
    description: str
    recommended_ticker: Optional[str] = None


class EventStudyRequestDTO(BaseModel):
    preset_id: str = Field(default="earnings", description="Preset ID (earnings, apple_iphone, nvidia_gtc, tesla_events, fed_fomc, custom)")
    custom_dates: Optional[List[str]] = Field(default=None, description="Custom event dates in YYYY-MM-DD")
    custom_event_name: Optional[str] = Field(default=None, description="Custom event title")
    horizons: Optional[List[int]] = Field(default_factory=lambda: [5, 10, 20, 60], description="Target forward horizons in trading days")


class EventPathPointDTO(BaseModel):
    day_offset: int
    date: str
    close: float
    cumulative_return_pct: float


class EventInstanceDTO(BaseModel):
    event_id: str
    event_date: str
    entry_date: str
    entry_price: float
    description: Optional[str] = None
    forward_returns: Dict[str, Optional[float]] = Field(default_factory=dict)
    daily_trajectory: List[EventPathPointDTO] = Field(default_factory=list)
    is_win_by_horizon: Dict[str, Optional[bool]] = Field(default_factory=dict)


class HorizonStatsDTO(BaseModel):
    horizon_days: int
    sample_size: int
    hit_rate_pct: float
    avg_return_pct: float
    median_return_pct: float
    best_return_pct: Optional[float] = None
    worst_return_pct: Optional[float] = None
    avg_win_pct: Optional[float] = None
    avg_loss_pct: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    win_count: int = 0
    loss_count: int = 0


class AverageTrajectoryPointDTO(BaseModel):
    day_offset: int
    avg_return_pct: float
    sample_count: int


class EventStudyGuardrailWarningDTO(BaseModel):
    code: str
    message: str


class EventStudyResponseDTO(BaseModel):
    ticker: str
    event_name: str
    horizons: List[int]
    events: List[EventInstanceDTO]
    horizon_stats: Dict[str, HorizonStatsDTO]
    average_trajectory: List[AverageTrajectoryPointDTO]
    warnings: List[EventStudyGuardrailWarningDTO] = Field(default_factory=list)
