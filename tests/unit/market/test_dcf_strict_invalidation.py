"""Unit tests for Strict DCF Domain Validation, Dynamic Invalidation, and Universal Target Suppression (Phase 2 / P0.2)."""
from datetime import datetime, timezone
from typing import Dict
import pytest

from schemas.macro_schemas import MarketObservable
from tools.market.dcf_valuation import (
    compute_dcf_valuation,
    compute_institutional_reverse_dcf,
)


def _mock_macro_registry(dgs10: float = 4.25, erp: float = 4.50) -> Dict[str, MarketObservable]:
    """Helper mock macro registry."""
    return {
        "obs_us_dgs10": MarketObservable(
            observable_id="obs_us_dgs10",
            asset_bucket="fixed_income",
            region="US",
            indicator="10-Year Treasury Yield",
            value=str(dgs10),
            unit="percent",
            observed_at="2026-09-01",
            source_file="FRED_DGS10.csv",
            is_valid=True,
        ),
        "obs_us_damodaran_erp": MarketObservable(
            observable_id="obs_us_damodaran_erp",
            asset_bucket="equities",
            region="US",
            indicator="Damodaran Implied ERP",
            value=str(erp),
            unit="percent",
            observed_at="2026-09-01",
            source_file="Damodaran_ERP.csv",
            is_valid=True,
        ),
    }


def test_dcf_valid_standard_execution():
    """Verify normal DCF calculation with healthy macro conditions."""
    macro = _mock_macro_registry(dgs10=4.25, erp=4.50)
    result, flags = compute_dcf_valuation(
        ticker="FTNT",
        market="US",
        current_price=172.78,
        beta=1.15,
        fcf_per_share=5.50,
        market_cap=133_000_000_000.0,
        total_debt=1_000_000_000.0,
        interest_expense=50_000_000.0,
        tax_rate=0.18,
        fcf_cagr_3y=18.0,
        macro_registry=macro,
    )
    assert result is not None
    assert result.is_actionable is True
    assert result.invalidation_reasons == []
    assert len(result.scenarios) == 3
    assert result.scenarios["base"].target_price is not None
    assert result.scenarios["base"].target_price > 0
    assert result.scenarios["bull"].target_price > result.scenarios["base"].target_price > result.scenarios["bear"].target_price


def test_dcf_upfront_domain_validation_out_of_scale():
    """Verify upfront domain rejection when rates are out of realistic economic bounds."""
    # Test 1: Excessive Risk-Free Rate > 20%
    macro_bad_rf = _mock_macro_registry(dgs10=25.0, erp=4.50)
    res_rf, flags_rf = compute_dcf_valuation(
        ticker="FTNT",
        market="US",
        current_price=172.78,
        beta=1.15,
        fcf_per_share=5.50,
        market_cap=133_000_000_000.0,
        total_debt=1_000_000_000.0,
        interest_expense=50_000_000.0,
        tax_rate=0.18,
        fcf_cagr_3y=18.0,
        macro_registry=macro_bad_rf,
    )
    assert res_rf.is_actionable is False
    assert "rate_out_of_domain:rf" in res_rf.invalidation_reasons
    assert res_rf.scenarios == {}

    # Test 2: Excessive Tax Rate > 60%
    macro_good = _mock_macro_registry(dgs10=4.25, erp=4.50)
    res_tax, flags_tax = compute_dcf_valuation(
        ticker="FTNT",
        market="US",
        current_price=172.78,
        beta=1.15,
        fcf_per_share=5.50,
        market_cap=133_000_000_000.0,
        total_debt=1_000_000_000.0,
        interest_expense=50_000_000.0,
        tax_rate=0.75,  # 75% tax rate
        fcf_cagr_3y=18.0,
        macro_registry=macro_good,
    )
    assert res_tax.is_actionable is False
    assert "rate_out_of_domain:tax_rate" in res_tax.invalidation_reasons
    assert res_tax.scenarios == {}


def test_dcf_non_positive_erp_invalidation():
    """Verify DCF invalidation when ERP is zero or negative."""
    macro_zero_erp = _mock_macro_registry(dgs10=4.25, erp=0.0)
    result, flags = compute_dcf_valuation(
        ticker="FTNT",
        market="US",
        current_price=172.78,
        beta=1.15,
        fcf_per_share=5.50,
        market_cap=133_000_000_000.0,
        total_debt=1_000_000_000.0,
        interest_expense=50_000_000.0,
        tax_rate=0.18,
        fcf_cagr_3y=18.0,
        macro_registry=macro_zero_erp,
    )
    assert result.is_actionable is False
    assert "erp_non_positive" in result.invalidation_reasons
    # UNIVERSAL TARGET SUPPRESSION: Must not contain target prices
    assert result.scenarios == {}
    assert result.valuation_verdict == "unavailable"


def test_reverse_dcf_universal_target_suppression_when_invalid():
    """Verify Reverse DCF suppresses all targets, upside, and projections when non-actionable."""
    macro_bad = _mock_macro_registry(dgs10=4.25, erp=-0.50)
    rev_res, flags = compute_institutional_reverse_dcf(
        ticker="FTNT",
        market="US",
        current_price=172.78,
        shares_outstanding=771_000_000.0,
        base_revenue=8_000_000_000.0,
        base_ebit_margin_pct=36.0,
        tax_rate=0.18,
        reinvestment_rate_pct=10.0,
        beta=1.15,
        total_debt=1_000_000_000.0,
        cash_and_equivalents=3_000_000_000.0,
        macro_registry=macro_bad,
    )
    assert rev_res.is_actionable is False
    assert rev_res.target_price_12m is None
    assert rev_res.upside_12m_pct is None
    assert rev_res.intrinsic_value_today is None
    assert rev_res.explicit_forecast_5y == []
    assert rev_res.valuation_verdict == "unavailable"
