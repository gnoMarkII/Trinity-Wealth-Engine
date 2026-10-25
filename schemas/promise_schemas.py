"""Pydantic DTOs for Promise Check earnings commitments, resolution, and credibility scoring."""
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class TranscriptCallDTO(BaseModel):
    """Metadata for an individual earnings call in the rolling 8-quarter window."""
    quarter: str = Field(description="Fiscal quarter label, e.g. FY2026Q2")
    date: str = Field(description="Call date YYYY-MM-DD")
    status: Literal["ok", "missing"] = Field(default="ok", description="Availability status of the transcript")
    source: Optional[Literal["stockanalysis", "fool", "investing", "alphastreet", "edgar", "other"]] = Field(
        default="stockanalysis", description="Source provider of transcript"
    )
    source_url: Optional[str] = Field(default=None, description="Original source web URL")
    words: Optional[int] = Field(default=None, description="Word count of parsed transcript")
    cached_file: Optional[str] = Field(default=None, description="Relative path to cached transcript text")
    fetch_method: Optional[Literal["direct_http", "browser", "webfetch", "manual"]] = Field(
        default="direct_http", description="Ingestion mechanism"
    )
    retired: Optional[bool] = Field(default=False, description="True if rolled out of current 8-call window")


class PromiseItemDTO(BaseModel):
    """An individual checkable commitment made by corporate management."""
    id: str = Field(description="Canonical ID, e.g. P-NVDA-FY2026Q2-01")
    made_in: str = Field(description="Fiscal quarter call when commitment was made, e.g. FY2026Q2")
    speaker: str = Field(description="Name and title of speaking executive")
    section: Literal["prepared", "qa"] = Field(description="Call section where statement occurred")
    type: Literal["A_numeric", "B_milestone"] = Field(description="A: numeric guidance, B: strategic milestone")
    category: Literal[
        "revenue", "eps", "margin", "opex", "capex", "fcf", "units",
        "growth", "product", "capacity", "regulatory", "org", "other"
    ] = Field(description="Financial or operational category")
    scope: Literal["company", "segment", "n/a"] = Field(default="company", description="Organizational scope")
    quote: str = Field(description="Verbatim quote from transcript (max 40 words)")
    metric: str = Field(description="Headline summary of metric or deliverable in English")
    metric_th: Optional[str] = Field(default=None, description="Headline summary in natural Thai")
    
    target_kind: Literal["range", "point", "floor", "ceiling", "direction", "milestone"] = Field(
        description="Target structure"
    )
    target_low: Optional[float] = Field(default=None, description="Lower bound for range or floor")
    target_high: Optional[float] = Field(default=None, description="Upper bound for range or ceiling")
    target_point: Optional[float] = Field(default=None, description="Specific target point")
    direction: Optional[Literal["up", "down", "flat"]] = Field(default=None, description="Directional guidance")
    unit: Optional[str] = Field(default=None, description="Measurement unit, e.g. USD bn, %, pts")
    basis: Optional[str] = Field(
        default="n/a", description="Accounting basis (GAAP, non-GAAP, constant currency, organic, reported, cash, n/a)"
    )
    
    due: str = Field(description="Fiscal quarter when commitment is due, e.g. FY2026Q3")
    due_precision: Literal["explicit", "inferred"] = Field(
        default="explicit", description="Whether deadline was explicitly stated or inferred"
    )
    notes: Optional[str] = Field(default="", description="Explanatory notes in English")
    notes_th: Optional[str] = Field(default="", description="Explanatory notes in Thai")
    
    # Resolution state
    status: Literal["Delivered", "Partial", "Missed", "Restated", "Quietly dropped", "Pending", "Unverifiable"] = Field(
        default="Pending", description="Resolution outcome"
    )
    restated_direction: Optional[Literal["raised", "lowered"]] = Field(
        default=None, description="Direction if restated before due date"
    )
    resolved_in: Optional[str] = Field(default=None, description="Fiscal quarter call that settled the promise")
    evidence_quote: Optional[str] = Field(default=None, description="Verbatim resolving quote (max 40 words)")
    outcome_value: Optional[float] = Field(default=None, description="Reported actual outcome value")
    actual_numeric: Optional[float] = Field(default=None, description="Actual numeric outcome parsed from resolution")
    extraction_method: Optional[str] = Field(default=None, description="Extraction provenance method, e.g. direct_http")
    delta_pct: Optional[float] = Field(default=None, description="Fractional delta (actual - target) / target")
    slip_quarters: Optional[int] = Field(default=None, description="Quarters delayed past due date for milestones")
    
    reiterated_in: List[str] = Field(default_factory=list, description="Calls where target was reaffirmed unchanged")
    supersedes: Optional[str] = Field(default=None, description="ID of earlier promise superseded by this one")
    retired: Optional[bool] = Field(default=False, description="True if rolled out of 8-call scoring window")


class GuidanceBiasDTO(BaseModel):
    """Categorization of management guidance tendencies."""
    label: str = Field(description="Display label: Sandbagger, Straight shooter, Overpromiser, Mixed")
    code: str = Field(description="Short code: sandbagger, straight, overpromiser, mixed, insufficient")
    message: str = Field(description="Actionable guidance interpretation advice")
    median_delta_pct: Optional[float] = Field(default=None, description="Median percentage beat/miss (e.g. +3.5%)")
    beat_rate: Optional[float] = Field(default=None, description="Percentage of numeric guidance beaten")
    n: int = Field(default=0, description="Number of resolved numeric guidance items")
    min_n: int = Field(default=5, description="Minimum sample size required for confident bias classification")


class CohortTrendItemDTO(BaseModel):
    """Single-call cohort delivery performance."""
    quarter: str = Field(description="Fiscal quarter label, e.g. FY2026Q2")
    label: str = Field(description="Display label, e.g. FY26 Q2")
    date: Optional[str] = Field(default=None, description="Call date")
    status: str = Field(default="ok", description="Transcript status")
    made: int = Field(default=0, description="Total promises made in this call")
    resolved: int = Field(default=0, description="Number of promises resolved to date")
    delivered: int = Field(default=0, description="Number of promises delivered successfully")
    pending: int = Field(default=0, description="Number of promises still pending")
    hit_rate: Optional[float] = Field(default=None, description="Delivered / resolved ratio")


class PromiseScoresDTO(BaseModel):
    """Aggregated quantitative scores and credibility metrics."""
    framework_version: str = Field(default="0.1")
    ticker: str = Field(description="Ticker symbol")
    company: str = Field(description="Company name")
    run_date: str = Field(description="Date of evaluation YYYY-MM-DD")
    window: Dict[str, Any] = Field(description="Rolling window boundaries and missing call list")
    
    credibility: Optional[float] = Field(default=None, description="Overall Credibility Score 0-100")
    grade: Optional[str] = Field(default=None, description="Letter grade: A, B, C, D, F")
    confidence: Literal["Low", "Medium", "High"] = Field(description="Statistical confidence based on sample size")
    delivery: Optional[float] = Field(default=None, description="Delivery component score 0-100")
    transparency: float = Field(default=100.0, description="Transparency component score 0-100")
    
    n_promises: int = Field(description="Total active promises in 8-call window")
    n_resolved: int = Field(description="Total resolved promises")
    counts: Dict[str, int] = Field(description="Count per status")
    
    bias: GuidanceBiasDTO = Field(description="Guidance bias analysis")
    milestone_slip_quarters: Optional[float] = Field(default=None, description="Average quarters delayed for milestones")
    milestone_slip_n: int = Field(default=0, description="Count of resolved milestones")
    trend: Dict[str, Any] = Field(description="8-cohort delivery trend and verdict")
    
    top_kept: List[str] = Field(default_factory=list, description="IDs of top 3 standout kept promises")
    top_broken: List[str] = Field(default_factory=list, description="IDs of top 3 broken or dropped promises")


class PromiseLedgerDTO(BaseModel):
    """Full Promise Check ledger containing rolling 8-quarter calls and extracted promises."""
    ticker: str = Field(description="Ticker symbol")
    company: str = Field(description="Company legal name")
    fiscal_year_end_month: int = Field(default=12, description="Month of fiscal year end (1-12)")
    framework_version: str = Field(default="0.1")
    run_date: str = Field(description="Date of compilation YYYY-MM-DD")
    calls: List[TranscriptCallDTO] = Field(default_factory=list, description="Rolling 8 earnings calls")
    promises: List[PromiseItemDTO] = Field(default_factory=list, description="List of commitments")


class PromiseSyncResponseDTO(BaseModel):
    """Response returned when syncing or inspecting a company's Promise Check."""
    ledger: PromiseLedgerDTO = Field(description="Full ledger data")
    scores: PromiseScoresDTO = Field(description="Calculated credibility metrics")
    from_cache: bool = Field(default=False, description="True if served from local cache")
