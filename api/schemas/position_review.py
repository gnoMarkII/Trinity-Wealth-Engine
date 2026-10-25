"""Pydantic schemas and DTOs for Position Review & Decision Framework."""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class PriceMoveDecompositionDTO(BaseModel):
    horizon_label: str = Field(description="Horizon e.g. 1Y, 3Y, since_entry")
    start_date: str
    end_date: str
    start_price: float
    end_price: float
    total_return_pct: float
    start_eps: float
    end_eps: float
    eps_growth_pct: float
    start_pe: float
    end_pe: float
    pe_expansion_pct: float
    primary_driver: str
    diagnosis_tier: str
    diagnosis_summary: str


class ThenVsNowSnapshotDTO(BaseModel):
    is_held: bool
    period_label: str
    then_date: str
    now_date: str
    then_price: float
    now_price: float
    then_pe: Optional[float] = None
    now_pe: Optional[float] = None
    then_growth_pct: Optional[float] = None
    now_growth_pct: Optional[float] = None
    what_was_true_then: str
    what_is_true_now: str
    thesis_reflection_question: str


class PortfolioAwareTriggerDTO(BaseModel):
    trigger_id: str
    title: str
    condition_text: str
    category: str
    case_impact: str
    impact_description: str
    is_currently_triggered: bool
    metric_current_value: Optional[float] = None
    metric_threshold_value: Optional[float] = None


class HoldingSummaryDTO(BaseModel):
    is_held: bool = False
    units: float = 0.0
    avg_cost: Optional[float] = None
    unrealized_pnl_pct: Optional[float] = None
    current_weight_pct: float = 0.0
    bucket_id: Optional[str] = None
    entry_date: Optional[str] = None
    added_direction: Optional[str] = None  # "on_the_way_down", "on_the_way_up", "single_entry", None


class PositionReviewResponseDTO(BaseModel):
    ticker: str
    company_name: Optional[str] = None
    current_price: float
    as_of_date: str
    action_stance: str
    action_stance_reason: Optional[str] = None
    holding_summary: HoldingSummaryDTO
    decompositions: List[PriceMoveDecompositionDTO] = Field(default_factory=list)
    then_vs_now: Optional[ThenVsNowSnapshotDTO] = None
    triggers: List[PortfolioAwareTriggerDTO] = Field(default_factory=list)
    honest_gaps: List[str] = Field(default_factory=list)
