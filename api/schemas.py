"""API response DTOs — แยกจาก schemas/macro_schemas.py โดยเด็ดขาด (ดู Rev.4/5 ข้อ 1)

Frontend ผูกกับ shape ที่นิยามในไฟล์นี้เท่านั้น ไม่ผูกกับ MacroStrategyDirection ภายใน
ถ้า macro_schemas.py เปลี่ยน field ในอนาคต (เพิ่ม guardrail ใหม่ ฯลฯ) แก้แค่ mapping
ในไฟล์นี้ ไม่ต้องแก้ frontend
"""
from typing import Any, Literal, Optional

from pydantic import BaseModel

from schemas.warning_registry import WarningMessage, translate_warning


class WarningDTO(BaseModel):
    """แสดง warning แบบ generic — code อาจเป็น ID ที่ registry ยังไม่มี Thai template ก็ยังโชว์ได้
    (ไม่ hardcode switch-case ตาม warning_registry.py วันนี้ ดู Rev.5 ข้อ 7)
    """
    code: Optional[str] = None
    message: str


def _to_warning_dto(raw: str) -> WarningDTO:
    parsed = WarningMessage.from_str(raw)
    return WarningDTO(code=parsed.id if parsed else None, message=translate_warning(raw))


def _warnings(raw_list: list[str] | None) -> list[WarningDTO]:
    return [_to_warning_dto(w) for w in (raw_list or [])]


class AssetAllocationDTO(BaseModel):
    asset_class: str
    asset_bucket: Optional[str] = None
    stance: str
    confidence: str
    rationale: str
    supporting_data: list[str] = []
    why_not_high: str = ""
    allocation_delta: str = ""
    invalidation_conditions: list[str] = []
    source_refs: list[str] = []
    observable_refs: list[str] = []
    warnings: list[WarningDTO] = []


class PairTradeDTO(BaseModel):
    long_leg: str
    short_leg: str
    thesis: str
    catalyst: str
    risk: str
    time_horizon: str = ""
    confidence: str
    sizing_guidance: str = ""
    instrument_proxy: str = ""
    hedge_ratio: str = ""
    implementation_idea: str = ""
    entry_trigger: str = ""
    stop_loss_trigger: str = ""
    target_gain_or_rebalance: str = ""
    supporting_data: list[str] = []
    source_refs: list[str] = []
    observable_refs: list[str] = []
    warnings: list[WarningDTO] = []


class RiskScenarioDTO(BaseModel):
    tail_risk: str
    probability: str
    impact: str
    trigger_to_activate: str
    hedge_instruments: list[str] = []
    unwind_or_cover_condition: str = ""
    early_warning_indicators: list[str] = []
    mitigation_strategy: str = ""
    cost_or_tradeoff: str = ""
    hedge_size: str = ""
    hedge_purpose: str = ""
    supporting_data: list[str] = []
    warnings: list[WarningDTO] = []


class RegimeEvidenceDTO(BaseModel):
    dimension: str
    signal: str
    evidence: str
    conflict: str = ""
    confidence: str
    source_refs: list[str] = []
    observable_refs: list[str] = []


class PortfolioDTO(BaseModel):
    evaluated_at: str
    overall_regime: str
    time_horizon: str
    conviction_level: str
    conviction_rationale: str
    quant_narrative_alignment: str
    divergence_note: str = ""
    focus_themes: list[str] = []
    asset_allocation: list[AssetAllocationDTO] = []
    pair_trades: list[PairTradeDTO] = []
    risk_scenarios: list[RiskScenarioDTO] = []
    warnings: list[WarningDTO] = []


class MacroIndicatorDTO(BaseModel):
    indicator_id: str
    series_key: str
    label: str
    value: float | None = None
    display_value: str = ""
    unit: str = ""
    observed_at: str = ""
    provider: str = ""
    source_file: str = ""
    is_valid: bool = True
    stale_reason: str = ""
    chart_available: bool = False


class MacroReferenceDTO(BaseModel):
    reference_id: str
    kind: Literal["news", "youtube"]
    title: str
    url: str
    publisher: str = ""
    published_at: str = ""
    age_hours: int | None = None
    summary: str = ""
    thumbnail_url: str = ""
    is_stale: bool = False
    related_observable_ids: list[str] = []


class MacroSeriesPointDTO(BaseModel):
    observed_at: str
    value: float


class MacroIndicatorSeriesDTO(BaseModel):
    indicator_id: str
    series_key: str
    label: str
    unit: str
    range: Literal["1m", "3m", "1y"]
    points: list[MacroSeriesPointDTO] = []


class MacroDashboardDTO(BaseModel):
    evaluated_at: str
    overall_regime: str
    time_horizon: str
    conviction_level: str = ""
    conviction_rationale: str = ""
    quant_narrative_alignment: str = ""
    divergence_note: str = ""
    focus_themes: list[str] = []
    key_assumptions: list[str] = []
    regime_probabilities: dict[str, Any] = {}
    regime_evidence: list[RegimeEvidenceDTO] = []
    asset_allocation: list[AssetAllocationDTO] = []
    pair_trades: list[PairTradeDTO] = []
    risk_scenarios: list[RiskScenarioDTO] = []
    source_files: list[str] = []
    generated_by: str = ""
    dashboard_indicators: list[MacroIndicatorDTO] = []
    report_references: list[MacroReferenceDTO] = []
    warnings: list[WarningDTO] = []



def _asset_dto(a: dict) -> AssetAllocationDTO:
    return AssetAllocationDTO(
        asset_class=a.get("asset_class", ""),
        asset_bucket=a.get("asset_bucket"),
        stance=a.get("stance", ""),
        confidence=a.get("confidence", ""),
        rationale=a.get("rationale", ""),
        supporting_data=a.get("supporting_data", []),
        why_not_high=a.get("why_not_high", ""),
        allocation_delta=a.get("allocation_delta", ""),
        invalidation_conditions=a.get("invalidation_conditions", []),
        source_refs=a.get("source_refs", []),
        observable_refs=a.get("observable_refs", []),
        warnings=_warnings(a.get("validation_warnings")),
    )


def _pair_trade_dto(pt: dict) -> PairTradeDTO:
    return PairTradeDTO(
        long_leg=pt.get("long_leg", ""),
        short_leg=pt.get("short_leg", ""),
        thesis=pt.get("thesis", ""),
        catalyst=pt.get("catalyst", ""),
        risk=pt.get("risk", ""),
        time_horizon=pt.get("time_horizon", ""),
        confidence=pt.get("confidence", ""),
        sizing_guidance=pt.get("sizing_guidance", ""),
        instrument_proxy=pt.get("instrument_proxy", ""),
        hedge_ratio=pt.get("hedge_ratio", ""),
        implementation_idea=pt.get("implementation_idea", ""),
        entry_trigger=pt.get("entry_trigger", ""),
        stop_loss_trigger=pt.get("stop_loss_trigger", ""),
        target_gain_or_rebalance=pt.get("target_gain_or_rebalance", ""),
        supporting_data=pt.get("supporting_data", []),
        source_refs=pt.get("source_refs", []),
        observable_refs=pt.get("observable_refs", []),
        warnings=_warnings(pt.get("validation_warnings")),
    )


def _risk_scenario_dto(rs: dict) -> RiskScenarioDTO:
    return RiskScenarioDTO(
        tail_risk=rs.get("tail_risk", ""),
        probability=rs.get("probability", ""),
        impact=rs.get("impact", ""),
        trigger_to_activate=rs.get("trigger_to_activate", ""),
        hedge_instruments=rs.get("hedge_instruments", []),
        unwind_or_cover_condition=rs.get("unwind_or_cover_condition", ""),
        early_warning_indicators=rs.get("early_warning_indicators", []),
        mitigation_strategy=rs.get("mitigation_strategy", ""),
        cost_or_tradeoff=rs.get("cost_or_tradeoff", ""),
        hedge_size=rs.get("hedge_size", ""),
        hedge_purpose=rs.get("hedge_purpose", ""),
        supporting_data=rs.get("supporting_data", []),
        warnings=_warnings(rs.get("validation_warnings")),
    )


def _regime_evidence_dto(re_: dict) -> RegimeEvidenceDTO:
    return RegimeEvidenceDTO(
        dimension=re_.get("dimension", ""),
        signal=re_.get("signal", ""),
        evidence=re_.get("evidence", ""),
        conflict=re_.get("conflict", ""),
        confidence=re_.get("confidence", ""),
        source_refs=re_.get("source_refs", []),
        observable_refs=re_.get("observable_refs", []),
    )


def _macro_indicator_dto(item: Any) -> MacroIndicatorDTO:
    raw = item if isinstance(item, dict) else {}
    value = raw.get("value")
    if not isinstance(value, (int, float)):
        value = None
    return MacroIndicatorDTO(
        indicator_id=str(raw.get("indicator_id", "")),
        series_key=str(raw.get("series_key", "")),
        label=str(raw.get("label", "")),
        value=float(value) if value is not None else None,
        display_value=str(raw.get("display_value", "")),
        unit=str(raw.get("unit", "")),
        observed_at=str(raw.get("observed_at", "")),
        provider=str(raw.get("provider", "")),
        source_file=str(raw.get("source_file", "")),
        is_valid=bool(raw.get("is_valid", True)),
        stale_reason=str(raw.get("stale_reason", "")),
        chart_available=bool(raw.get("chart_available", False)),
    )


def _macro_reference_dto(item: Any) -> MacroReferenceDTO | None:
    raw = item if isinstance(item, dict) else {}
    kind = raw.get("kind")
    url = str(raw.get("url", ""))
    if kind not in {"news", "youtube"} or not url.startswith("https://"):
        return None
    related_ids = raw.get("related_observable_ids", [])
    return MacroReferenceDTO(
        reference_id=str(raw.get("reference_id", "")),
        kind=kind,
        title=str(raw.get("title", "")),
        url=url,
        publisher=str(raw.get("publisher", "")),
        published_at=str(raw.get("published_at", "")),
        age_hours=raw.get("age_hours") if isinstance(raw.get("age_hours"), int) else None,
        summary=str(raw.get("summary", "")),
        thumbnail_url=str(raw.get("thumbnail_url", "")) if str(raw.get("thumbnail_url", "")).startswith("https://") else "",
        is_stale=bool(raw.get("is_stale", False)),
        related_observable_ids=[str(value) for value in related_ids] if isinstance(related_ids, list) else [],
    )


def portfolio_dto_from_raw(raw: dict) -> PortfolioDTO:
    """map จาก dict ที่ json.load มาตรงๆ จากไฟล์ sidecar — ไม่ re-instantiate ผ่าน
    MacroStrategyDirection(**raw) เพราะจะไป trigger guardrail validator ซ้ำโดยไม่จำเป็น
    (ข้อมูลผ่าน validation ครบแล้วตอนเขียนไฟล์)
    """
    return PortfolioDTO(
        evaluated_at=raw.get("evaluated_at", ""),
        overall_regime=raw.get("overall_regime", ""),
        time_horizon=raw.get("time_horizon", ""),
        conviction_level=raw.get("conviction_level", ""),
        conviction_rationale=raw.get("conviction_rationale", ""),
        quant_narrative_alignment=raw.get("quant_narrative_alignment", ""),
        divergence_note=raw.get("divergence_note", ""),
        focus_themes=raw.get("focus_themes", []),
        asset_allocation=[_asset_dto(a) for a in raw.get("asset_allocation", [])],
        pair_trades=[_pair_trade_dto(pt) for pt in raw.get("pair_trades", [])],
        risk_scenarios=[_risk_scenario_dto(rs) for rs in raw.get("risk_scenarios", [])],
        warnings=_warnings(raw.get("validation_warnings")) + _warnings(raw.get("stale_data_warnings")),
    )


def macro_dashboard_dto_from_raw(raw: dict) -> MacroDashboardDTO:
    report_references = [_macro_reference_dto(item) for item in raw.get("report_references", [])]
    return MacroDashboardDTO(
        evaluated_at=raw.get("evaluated_at", ""),
        overall_regime=raw.get("overall_regime", ""),
        time_horizon=raw.get("time_horizon", ""),
        conviction_level=raw.get("conviction_level", ""),
        conviction_rationale=raw.get("conviction_rationale", ""),
        quant_narrative_alignment=raw.get("quant_narrative_alignment", ""),
        divergence_note=raw.get("divergence_note", ""),
        focus_themes=raw.get("focus_themes", []),
        key_assumptions=raw.get("key_assumptions", []),
        regime_probabilities=raw.get("regime_probabilities", {}),
        regime_evidence=[_regime_evidence_dto(r) for r in raw.get("regime_evidence", [])],
        asset_allocation=[_asset_dto(a) for a in raw.get("asset_allocation", [])],
        pair_trades=[_pair_trade_dto(pt) for pt in raw.get("pair_trades", [])],
        risk_scenarios=[_risk_scenario_dto(rs) for rs in raw.get("risk_scenarios", [])],
        source_files=raw.get("source_files", []),
        generated_by=raw.get("generated_by", ""),
        dashboard_indicators=[_macro_indicator_dto(item) for item in raw.get("dashboard_indicators", [])],
        report_references=[item for item in report_references if item is not None],
        warnings=_warnings(raw.get("validation_warnings")) + _warnings(raw.get("stale_data_warnings")),
    )


class JobStatusDTO(BaseModel):
    job_id: str
    status: str
    card_id: Optional[str] = None
    error_message: Optional[str] = None
    current_node: Optional[str] = None
    interrupt_payload: Optional[dict] = None
    log_count: int = 0
    created_at: float = 0.0
    updated_at: float = 0.0


class ActiveAgentStatusDTO(BaseModel):
    running: bool
    flow: Optional[str] = None
    node: Optional[str] = None
    job_id: Optional[str] = None


class JobLogEntryDTO(BaseModel):
    seq: int
    node_name: Optional[str] = None
    content: str
    role: str = "reply"
    label: Optional[str] = None


class SpecialistOutputDTO(BaseModel):
    node_name: str
    label: str
    content: str
    seq: int
    created_at: float


class JobOutputsDTO(BaseModel):
    job_id: str
    status: str
    executive_summary: Optional[str] = None
    executive_summary_created_at: Optional[float] = None
    specialists: list[SpecialistOutputDTO] = []
    last_seq: int = 0
    error_message: Optional[str] = None


class KanbanCardDTO(BaseModel):
    card_id: str
    title: str
    prompt: Optional[str] = None
    column_name: str
    job_id: Optional[str] = None
    flow: str = "manager"
    scope: str = "both"
    display_seq: Optional[int] = None
    created_at: float
    updated_at: float


# ---------------------------------------------------------
# Actual Portfolio Hub DTOs (Phase 1 & Phase 2)
# ---------------------------------------------------------

class ActualHoldingDTO(BaseModel):
    """12-Column Holding schema matching Frontend Holdings Table & Backend Holding Model"""
    symbol: str
    asset_type: str
    units: float
    bucket_id: Optional[str] = None
    avg_cost_usd: Optional[float] = None
    avg_cost_thb: Optional[float] = None
    current_price_usd: Optional[float] = None
    current_price_thb: Optional[float] = None
    market_value_thb: float = 0.0
    unrealized_pnl_percent: Optional[float] = None
    unrealized_pnl_value: Optional[float] = None
    market_cap_tier: Optional[str] = None  # 'Mega', 'Large', 'Mid', 'Small', 'N/A'
    yield_on_cost: Optional[float] = None
    company_name: Optional[str] = None
    pe_ratio: Optional[float] = None
    eps: Optional[float] = None
    payout_ratio: Optional[float] = None
    market_cap_value: Optional[float] = None
    dividend_per_share: Optional[float] = None
    dividend_yield: Optional[float] = None
    accumulated_dividend_thb: Optional[float] = None
    fundamentals_updated_at: Optional[float] = None


class ActualSummaryDTO(BaseModel):
    total_value_thb: float = 0.0
    total_cost_basis_thb: float = 0.0
    total_unrealized_profit: float = 0.0
    passive_income_ytd: float = 0.0


class AllocationTargetDTO(BaseModel):
    bucket_id: str
    name: str
    target_percent: float
    color: Optional[str] = None


class ActualPortfolioStateDTO(BaseModel):
    last_updated: Optional[str] = None
    fx_rates: dict[str, float] = {}
    summary: ActualSummaryDTO
    allocation_targets: list[AllocationTargetDTO] = []
    holdings: list[ActualHoldingDTO] = []
    price_refresh_info: Optional[dict[str, str]] = None


class BucketAllocationSummaryDTO(BaseModel):
    bucket_id: str
    name: str
    target_percent: float
    actual_value_thb: float
    actual_percent: float
    variance: float
    color: Optional[str] = None


class BucketAllocationResponseDTO(BaseModel):
    warning: Optional[str] = None
    summaries: list[BucketAllocationSummaryDTO] = []


class ActualWatchlistItemDTO(BaseModel):
    symbol: str
    asset_type: str
    target_price: Optional[float] = None
    added_date: str
    notes: Optional[str] = None


class ActualWatchlistStateDTO(BaseModel):
    last_updated: Optional[str] = None
    items: list[ActualWatchlistItemDTO] = []


class ActualGoalItemDTO(BaseModel):
    name: str
    target_amount_thb: float
    goal_type: str  # 'nav_target', 'cash_target', 'passive_income_ytd'
    current_amount_thb: float = 0.0
    progress_pct: float = 0.0
    deadline: Optional[str] = None
    deadline_days_left: Optional[int] = None
    notes: Optional[str] = None


class ActualGoalsResponseDTO(BaseModel):
    n_goals: int
    goals: list[ActualGoalItemDTO] = []
    generated_at: Optional[str] = None


class PerformanceSnapshotDTO(BaseModel):
    Date: str
    Total_NAV: float
    Total_Cost: float
    Unrealized_PnL: float
    Cash_Balance: float
    realized_pnl_ytd: Optional[float] = None
    passive_income_ytd: Optional[float] = None


class JournalEntryDTO(BaseModel):
    timestamp: str
    content: str


class UpsertAllocationTargetsRequestDTO(BaseModel):
    targets: list[AllocationTargetDTO] = []


class AssignBucketRequestDTO(BaseModel):
    bucket_id: Optional[str] = None


class BatchAssignBucketRequestDTO(BaseModel):
    symbols: list[str] = []
    bucket_id: Optional[str] = None


class BatchRemoveHoldingsRequestDTO(BaseModel):
    symbols: list[str] = []


class TradeRequestDTO(BaseModel):
    symbol: str
    asset_type: str
    action: Literal["buy", "sell"]
    units: float
    price: float
    currency: Literal["THB", "USD"] = "THB"
    exchange_rate: Optional[float] = None
    date: Optional[str] = None
    notes: str = ""
    bucket_id: Optional[str] = None


class CashFlowRequestDTO(BaseModel):
    amount: float
    action: Literal["deposit", "withdraw"]
    currency: Literal["THB", "USD"] = "THB"
    exchange_rate: Optional[float] = None
    date: Optional[str] = None
    notes: str = ""


class IncomeRequestDTO(BaseModel):
    income_type: Literal["Dividend", "Interest", "Rental", "Other"]
    amount_thb: float
    source_symbol: Optional[str] = None
    date: Optional[str] = None
    notes: str = ""


class EditHoldingRequestDTO(BaseModel):
    units: Optional[float] = None
    avg_cost: Optional[float] = None
    accumulated_dividend_thb: Optional[float] = None
    asset_type: Optional[str] = None
    reason: str = ""
    bucket_id: Optional[str] = None


class UpsertWatchlistItemRequestDTO(BaseModel):
    asset_type: str
    target_price: Optional[float] = None
    notes: str = ""


class UpsertGoalRequestDTO(BaseModel):
    goal_type: Literal["nav_target", "cash_target", "passive_income_ytd"]
    target_amount_thb: float
    deadline: Optional[str] = None
    years_from_now: Optional[int] = None
    notes: Optional[str] = None


class AppendJournalRequestDTO(BaseModel):
    entry: str


class NewsFunnelPendingItemDTO(BaseModel):
    event_id: str
    canonical_title: str
    comprehensive_summary: str = ""
    macro_impact_score: int = 0
    asset_impact_score: int = 0
    extracted_tickers: list[str] = []
    extracted_themes: list[str] = []
    primary_tags: list[str] = []
    links: list[str] = []
    triage_source: Optional[str] = None


class NewsFunnelFilteredItemDTO(BaseModel):
    event_id: str
    canonical_title: str
    comprehensive_summary: str = ""
    macro_impact_score: int = 0
    asset_impact_score: int = 0
    extracted_tickers: list[str] = []
    extracted_themes: list[str] = []
    primary_tags: list[str] = []
    links: list[str] = []
    triage_source: Optional[str] = None
    status: str
    triage_reasoning: Optional[str] = None
    error_msg: Optional[str] = None
    ingested_at: Optional[str] = None

