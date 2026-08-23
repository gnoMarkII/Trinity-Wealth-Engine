"""Financials Package Entrypoint preserving public API contract, legacy symbols, and feature flagging."""
import os
from typing import Literal, Optional

from tools.market.financials.adapters.sec_8k_subclient import Sec8KSubclient
from tools.market.financials.domain.calculations import (
    calc_yoy_growth as _calc_yoy_growth,
    finite_or_none as _finite_or_none,
)
from tools.market.financials.domain.constants import (
    CANONICAL_LINE_ITEMS,
    EDGAR_CONCEPT_SYNONYMS,
    REQUIRED_CORE_FIELDS,
    REQUIRED_EXPANDED_FIELDS,
    YFINANCE_CONCEPT_MAP,
)
from tools.market.financials.domain.models import (
    FinancialCellDTO,
    FinancialPeriodDTO,
    FinancialRatioPointDTO,
    FinancialStatementCategoryDTO,
    FinancialStatementsDTO,
    FinancialSummaryChartPointDTO,
    LineItemMetaDTO,
)
from tools.market.financials.domain.normalizer import (
    collect_distinct_authoritative_filings as _collect_distinct_authoritative_filings,
    extract_canonical_cell_from_df as _extract_canonical_cell_from_df,
    select_xbrl_column as _select_xbrl_column,
    table_to_grid as _table_to_grid,
    validate_sec_url as _validate_sec_url,
)
from tools.market.financials.domain.validator import (
    compute_coverage_metrics as _compute_coverage_metrics,
    validate_periods as _validate_periods,
)
from tools.market.financials.service import FinancialsService

# Legacy / Regression helper bindings
_sec_8k_inst = Sec8KSubclient()
_extract_q4_eps_from_8k = _sec_8k_inst.extract_q4_eps
_extract_8k_fcf_reconciliation = _sec_8k_inst.extract_fcf_reconciliation


def init_sec_edgar() -> bool:
    from tools.market.financials.adapters.edgar_subclient import EdgarSubclient
    return EdgarSubclient().init_identity()


def fetch_edgar_multi_period(ticker: str) -> Optional[FinancialStatementsDTO]:
    from tools.market.financials.adapters.composite_us_provider import CompositeUsFinancialProvider
    return CompositeUsFinancialProvider().fetch_statements(ticker, ticker)


def get_financial_statements(
    ticker: str,
    market: Literal["US", "TH"] = "US",
    provider_symbol: Optional[str] = None,
    force_refresh: bool = False,
) -> FinancialStatementsDTO:
    """ฟังก์ชันหลักสำหรับดึงงบการเงิน
    - รองรับ Positional Call เดิม: get_financial_statements(ticker, market, provider_symbol, force_refresh)
    - Intentional API Expansion: เพิ่ม default market="US" และ provider_symbol=None (fallback สู่ ticker)
    """
    resolved_symbol = provider_symbol or ticker
    return FinancialsService().get_financial_statements(
        ticker=ticker,
        market=market,
        provider_symbol=resolved_symbol,
        force_refresh=force_refresh,
    )


__all__ = [
    "get_financial_statements",
    "FinancialsService",
    "init_sec_edgar",
    "fetch_edgar_multi_period",
    "CANONICAL_LINE_ITEMS",
    "EDGAR_CONCEPT_SYNONYMS",
    "REQUIRED_CORE_FIELDS",
    "REQUIRED_EXPANDED_FIELDS",
    "YFINANCE_CONCEPT_MAP",
    "LineItemMetaDTO",
    "FinancialCellDTO",
    "FinancialPeriodDTO",
    "FinancialStatementCategoryDTO",
    "FinancialSummaryChartPointDTO",
    "FinancialRatioPointDTO",
    "FinancialStatementsDTO",
    "_collect_distinct_authoritative_filings",
    "_validate_sec_url",
    "_finite_or_none",
    "_calc_yoy_growth",
    "_select_xbrl_column",
    "_extract_canonical_cell_from_df",
    "_extract_q4_eps_from_8k",
    "_extract_8k_fcf_reconciliation",
    "_validate_periods",
    "_table_to_grid",
    "_compute_coverage_metrics",
]
