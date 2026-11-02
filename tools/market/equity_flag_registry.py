"""Taxonomy and classification registry for equity data quality flags.

Architecture Layer: tools/market/
Dependencies: Standard Library only. Complies with AST dependency rules.

Classifies each data quality flag into canonical categories:
- ISSUE: Actionable defect, calculation exception, or data inconsistency requiring attention.
- DATA_GAP: Market or financial statement data legitimately missing from data providers.
- ASSUMPTION: Transparent model assumptions, legal disclosures (e.g. 10b5-1), or default proxy variables.
- INFO: Informational status, healthy thresholds, or positive valuation confirmations.
"""
from enum import Enum
from typing import Dict


class EquityFlagCategory(str, Enum):
    ISSUE = "ISSUE"
    DATA_GAP = "DATA_GAP"
    ASSUMPTION = "ASSUMPTION"
    INFO = "INFO"


EQUITY_FLAG_CATEGORIES: Dict[str, EquityFlagCategory] = {
    # ── ASSUMPTION: Valuation assumptions, disclosures, proxies ──────
    "10b51_unfiltered:insider_signal": EquityFlagCategory.ASSUMPTION,
    "eps_proxy_base_growth:dcf": EquityFlagCategory.ASSUMPTION,
    "generic_base_growth_assumption:dcf": EquityFlagCategory.ASSUMPTION,
    "hardcoded_us_risk_free:dcf": EquityFlagCategory.ASSUMPTION,
    "hardcoded_th_risk_free:dcf": EquityFlagCategory.ASSUMPTION,
    "hardcoded_country_risk_premium:dcf": EquityFlagCategory.ASSUMPTION,
    "hardcoded_cost_of_debt:dcf": EquityFlagCategory.ASSUMPTION,
    "tax_policy:US_FEDERAL_STATUTORY_21PCT": EquityFlagCategory.ASSUMPTION,
    "rich_market_valuation_low_erp:dcf": EquityFlagCategory.ASSUMPTION,
    "operating_margin_proxy:dcf": EquityFlagCategory.ASSUMPTION,
    "cyclical_peak_margin_normalized:dcf": EquityFlagCategory.ASSUMPTION,
    "cyclical_trough_margin_normalized:dcf": EquityFlagCategory.ASSUMPTION,

    # ── INFO: Informational states, bounds, buy zone confirmations ────
    "in_margin_of_safety_buy_zone:dcf": EquityFlagCategory.INFO,
    "kd_clamped:dcf": EquityFlagCategory.INFO,
    "high_leverage_risk:solvency": EquityFlagCategory.INFO,
    "unsustainable_payout:dividend": EquityFlagCategory.INFO,
    "implied_growth_below_lower_bound:dcf": EquityFlagCategory.INFO,
    "implied_growth_above_upper_bound:dcf": EquityFlagCategory.INFO,
    "high_terminal_value_dependency:dcf": EquityFlagCategory.INFO,
    "aggressive_implied_exit_multiple:dcf": EquityFlagCategory.INFO,
    "extreme_terminal_exit_ebitda:dcf": EquityFlagCategory.INFO,
    "depressed_terminal_exit_ebitda:dcf": EquityFlagCategory.INFO,
    "valuation_not_actionable:scorecard": EquityFlagCategory.INFO,
    "non_actionable_dcf:valuation": EquityFlagCategory.INFO,
    "macro_regime_headwind:scorecard": EquityFlagCategory.INFO,

    # ── DATA_GAP: Legitimate data unavailable from upstream providers ──
    "negative_fcf_dcf_unavailable:dcf": EquityFlagCategory.DATA_GAP,
    "beta_unavailable_dcf_unavailable:dcf": EquityFlagCategory.DATA_GAP,
    "insider_date_unavailable:insider_signal": EquityFlagCategory.DATA_GAP,
    "macro_stale_publication:us_10y": EquityFlagCategory.DATA_GAP,
    "macro_stale_publication:erp": EquityFlagCategory.DATA_GAP,
    "macro_publication_future_leak:us_10y": EquityFlagCategory.DATA_GAP,
    "stale_ohlcv:target_eod_missing": EquityFlagCategory.DATA_GAP,
    "stale_ohlcv:stale_one_session": EquityFlagCategory.DATA_GAP,
    "stale_ohlcv:market_data": EquityFlagCategory.DATA_GAP,
    "missing_growth_data:growth": EquityFlagCategory.DATA_GAP,
    "missing_dividend_data:dividend": EquityFlagCategory.DATA_GAP,
    "missing_solvency_data:solvency": EquityFlagCategory.DATA_GAP,
    "missing_liquidity_data:liquidity": EquityFlagCategory.DATA_GAP,
    "missing_operating_margin:dcf": EquityFlagCategory.DATA_GAP,
    "unknown_financial_source_tier:dcf": EquityFlagCategory.DATA_GAP,
    "unknown_financial_period_type:dcf": EquityFlagCategory.DATA_GAP,
    "negative_earnings:pe_undefined": EquityFlagCategory.DATA_GAP,
    "invalidation:macro_inputs_stale_or_misaligned": EquityFlagCategory.DATA_GAP,
    "non_actionable_macro_anomaly:reverse_dcf": EquityFlagCategory.DATA_GAP,
    "quote_ohlcv_mismatch:market_data": EquityFlagCategory.DATA_GAP,

    # ── ISSUE: True defects, execution failures, or math invalidations ─
    "execution_exception": EquityFlagCategory.ISSUE,
    "wacc_below_terminal_growth:dcf": EquityFlagCategory.ISSUE,
    "wacc_below_terminal_growth:reverse_dcf": EquityFlagCategory.ISSUE,
}


def categorize_equity_flag(flag: str) -> EquityFlagCategory:
    """Classify an equity data quality flag into its canonical category."""
    clean = str(flag or "").strip()
    if clean in EQUITY_FLAG_CATEGORIES:
        return EQUITY_FLAG_CATEGORIES[clean]

    # Heuristic fallback for dynamic or composite flag keys
    clean_lower = clean.lower()
    if any(k in clean_lower for k in ("exception", "error", "mismatch")):
        return EquityFlagCategory.ISSUE
    if any(k in clean_lower for k in ("unavailable", "missing", "stale", "invalidation")):
        return EquityFlagCategory.DATA_GAP
    if any(k in clean_lower for k in ("assumption", "proxy", "hardcoded", "policy")):
        return EquityFlagCategory.ASSUMPTION

    return EquityFlagCategory.INFO

