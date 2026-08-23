"""Financial Domain Models (Pydantic Models)."""
from typing import Literal, Optional
from pydantic import BaseModel, Field


class LineItemMetaDTO(BaseModel):
    canonical_key: str
    display_label: str
    unit_type: Literal["currency", "per_share", "shares", "ratio", "percentage"] = "currency"
    is_primary_highlight: bool = False


class FinancialCellDTO(BaseModel):
    value: Optional[float] = None
    yoy_growth_pct: Optional[float] = None
    source_type: Literal["reported", "derived", "not_applicable", "unavailable"] = "reported"
    source_concept: Optional[str] = None
    source_filing_url: Optional[str] = None
    derivation: Optional[str] = None
    formula: Optional[str] = None
    input_items: Optional[list[str]] = None
    source_period: Optional[str] = None
    unavailable_reason: Optional[str] = None
    is_derived: bool = False


class FinancialPeriodDTO(BaseModel):
    period_key: str              # e.g. "2024-Q3", "2024-FY"
    fiscal_year: int             # e.g. 2024
    fiscal_quarter: Optional[int] = None # 1, 2, 3, 4, or None for annual
    period_end_date: str         # "YYYY-MM-DD"
    period_kind: Literal["instant", "duration"]
    duration_days: Optional[int] = None # 80..100 for Q, ~365 for FY, None for instant
    form_type: str               # "10-Q", "10-K", "yfinance"
    filing_url: Optional[str] = None
    is_derived: bool = False
    items: dict[str, FinancialCellDTO] = Field(default_factory=dict)


class FinancialStatementCategoryDTO(BaseModel):
    statement_type: Literal["income", "balance_sheet", "cash_flow"]
    period_kind: Literal["duration", "instant"] = "duration"
    periods: list[FinancialPeriodDTO] = Field(default_factory=list) # latest (left) -> historical (right)
    line_items: list[LineItemMetaDTO] = Field(default_factory=list)


class FinancialSummaryChartPointDTO(BaseModel):
    period_key: str
    date: str
    revenue: Optional[float] = None
    gross_profit: Optional[float] = None
    operating_income: Optional[float] = None
    net_income: Optional[float] = None
    free_cash_flow: Optional[float] = None              # Backwards-compatible calculated FCF
    calculated_free_cash_flow: Optional[float] = None   # OCF - |CapEx|
    reported_free_cash_flow: Optional[float] = None     # Company-reported Non-GAAP FCF (8-K)
    operating_margin_pct: Optional[float] = None
    net_margin_pct: Optional[float] = None


class FinancialRatioPointDTO(BaseModel):
    period_key: str
    period_end_date: str
    gross_margin_pct: Optional[float] = None
    operating_margin_pct: Optional[float] = None
    net_margin_pct: Optional[float] = None
    fcf_margin_pct: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None


class FinancialStatementsDTO(BaseModel):
    schema_version: int = 6
    ticker: str
    market: Literal["US", "TH"]
    currency: str
    provider: Optional[Literal["edgartools", "yfinance"]] = None
    provider_symbol: str
    data_status: Literal["ok", "partial", "empty", "stale"]
    coverage_status: Literal["complete", "partial"] = "complete"
    core_coverage_status: Literal["complete", "partial"] = "complete"
    expanded_coverage_status: Literal["complete", "partial", "not_available"] = "complete"
    expanded_data_status: Literal["complete", "partial", "not_available"] = "complete"
    core_coverage_pct: Optional[float] = None
    expanded_coverage_pct: Optional[float] = None
    missing_required_items: list[str] = Field(default_factory=list)
    missing_expanded_items: list[str] = Field(default_factory=list)
    validation_warnings: list[str] = Field(default_factory=list)
    expanded_validation_warnings: list[str] = Field(default_factory=list)
    expanded_error_count: int = 0
    error_code: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)
    quarterly: list[FinancialStatementCategoryDTO] = Field(default_factory=list)
    annual: list[FinancialStatementCategoryDTO] = Field(default_factory=list)
    summary_chart_quarterly: list[FinancialSummaryChartPointDTO] = Field(default_factory=list)
    summary_chart_annual: list[FinancialSummaryChartPointDTO] = Field(default_factory=list)
    ratios_quarterly: list[FinancialRatioPointDTO] = Field(default_factory=list)
    ratios_annual: list[FinancialRatioPointDTO] = Field(default_factory=list)
    synced_at: Optional[str] = None
