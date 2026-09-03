"""Unit tests for DCF hard invalidation invariants, rate domain, US domestic tax fallback, and central target suppression."""
import pytest
from datetime import datetime, timezone
from schemas.macro_schemas import MarketObservable
from schemas.micro_quant_schemas import DCFResult
from tools.market.dcf_valuation import compute_dcf_valuation, compute_institutional_reverse_dcf


def _create_obs(obs_id: str, indicator: str, value: str, observed_at: str) -> MarketObservable:
    return MarketObservable(
        observable_id=obs_id,
        asset_bucket="fixed_income",
        region="USA",
        indicator=indicator,
        value=value,
        unit="%",
        observed_at=observed_at,
        source_file="test_macro.md",
        is_valid=True,
    )


def test_dcf_hard_invalidation_non_positive_erp():
    macro_reg = {
        "obs_dgs10": _create_obs("obs_dgs10", "DGS10", "4.25", "2026-08-20"),
        "obs_erp_gspc": _create_obs("obs_erp_gspc", "ERP", "-0.50", "2026-08-01"),
    }

    res, flags = compute_dcf_valuation(
        ticker="TEST",
        market="US",
        current_price=100.0,
        beta=1.1,
        fcf_per_share=5.0,
        market_cap=10000.0,
        total_debt=1000.0,
        interest_expense=50.0,
        tax_rate=0.21,
        fcf_cagr_3y=10.0,
        macro_registry=macro_reg,
    )

    assert res is not None
    assert res.is_actionable is False
    assert res.valuation_verdict == "unavailable"
    assert len(res.scenarios) == 0  # Targets suppressed
    assert "erp_non_positive" in res.invalidation_reasons


def test_dcf_rate_domain_error_wacc_spread():
    macro_reg = {
        "obs_dgs10": _create_obs("obs_dgs10", "DGS10", "2.00", "2026-08-20"),
        "obs_erp_gspc": _create_obs("obs_erp_gspc", "ERP", "0.10", "2026-08-01"),
    }

    res, flags = compute_dcf_valuation(
        ticker="TEST",
        market="US",
        current_price=100.0,
        beta=0.1,  # Ke = 2.0 + 0.01 = 2.01% -> WACC ~ 2.01% <= g_terminal (2.0%) + 0.005
        fcf_per_share=5.0,
        market_cap=10000.0,
        total_debt=0.0,
        interest_expense=0.0,
        tax_rate=0.21,
        fcf_cagr_3y=5.0,
        macro_registry=macro_reg,
    )

    assert res.is_actionable is False
    assert "wacc_terminal_spread_insufficient" in res.invalidation_reasons
    assert res.scenarios == {}


def test_reverse_dcf_target_suppression_when_non_actionable():
    macro_reg = {
        "obs_dgs10": _create_obs("obs_dgs10", "DGS10", "4.25", "2026-08-20"),
        "obs_erp_gspc": _create_obs("obs_erp_gspc", "ERP", "-1.00", "2026-08-01"),
    }

    rev_res, flags = compute_institutional_reverse_dcf(
        ticker="TEST",
        market="US",
        current_price=100.0,
        shares_outstanding=100.0,
        base_revenue=1000.0,
        base_ebit_margin_pct=30.0,
        tax_rate=0.21,
        reinvestment_rate_pct=10.0,
        beta=1.0,
        macro_registry=macro_reg,
    )

    assert rev_res.is_actionable is False
    assert rev_res.status == "unavailable"
    assert rev_res.valuation_verdict == "unavailable"
    assert rev_res.target_price_12m is None
    assert rev_res.upside_12m_pct is None
    assert rev_res.intrinsic_value_today is None
    assert len(rev_res.explicit_forecast_5y) == 0


def test_tax_policy_fallback_us_domestic_only():
    macro_reg = {
        "obs_dgs10": _create_obs("obs_dgs10", "DGS10", "4.25", "2026-08-20"),
        "obs_erp_gspc": _create_obs("obs_erp_gspc", "ERP", "4.50", "2026-08-01"),
    }

    # US Domestic company with missing tax_rate (0) -> applies 21% US Federal Statutory
    res_us, flags_us = compute_dcf_valuation(
        ticker="US_CO",
        market="US",
        current_price=100.0,
        beta=1.0,
        fcf_per_share=5.0,
        market_cap=10000.0,
        total_debt=1000.0,
        interest_expense=50.0,
        tax_rate=0.0,  # missing
        fcf_cagr_3y=5.0,
        macro_registry=macro_reg,
        is_us_domestic=True,
    )
    assert "tax_policy:US_FEDERAL_STATUTORY_21PCT" in flags_us
