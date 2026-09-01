import pytest
from tools.market.dcf_valuation import compute_institutional_reverse_dcf, compute_dcf_valuation
from schemas.macro_schemas import MarketObservable


def test_reverse_dcf_actionability_when_erp_negative():
    macro_reg = {
        "obs_dgs10": MarketObservable(
            observable_id="obs_dgs10",
            asset_bucket="fixed_income",
            region="US",
            indicator="DGS10",
            value="4.25",
            unit="%",
            observed_at="2026-08-27",
            source_file="test",
            provider="FRED",
            is_valid=True,
        ),
        "obs_erp_gspc": MarketObservable(
            observable_id="obs_erp_gspc",
            asset_bucket="equities",
            region="US",
            indicator="ERP_GSPC",
            value="-0.22",
            unit="%",
            observed_at="2026-08-27",
            source_file="test",
            provider="MacroEngine",
            is_valid=True,
        ),
    }

    result, flags = compute_institutional_reverse_dcf(
        ticker="FTNT",
        market="US",
        current_price=172.78,
        shares_outstanding=733713653,
        base_revenue=6799600000.0,
        base_ebit_margin_pct=33.86,
        cash_and_equivalents=3000000000.0,
        total_debt=1000000000.0,
        beta=1.15,
        macro_registry=macro_reg,
    )

    assert result is not None
    assert result.is_actionable is False
    assert "ERP" in result.actionability_reason or "Ke" in result.actionability_reason
    assert "non_actionable_macro_anomaly:reverse_dcf" in flags


def test_reverse_dcf_actionability_when_wacc_below_terminal_growth():
    macro_reg = {
        "obs_dgs10": MarketObservable(
            observable_id="obs_dgs10",
            asset_bucket="fixed_income",
            region="US",
            indicator="DGS10",
            value="2.00",
            unit="%",
            observed_at="2026-08-27",
            source_file="test",
            provider="FRED",
            is_valid=True,
        ),
        "obs_erp_gspc": MarketObservable(
            observable_id="obs_erp_gspc",
            asset_bucket="equities",
            region="US",
            indicator="ERP_GSPC",
            value="1.00",
            unit="%",
            observed_at="2026-08-27",
            source_file="test",
            provider="MacroEngine",
            is_valid=True,
        ),
    }

    result, flags = compute_institutional_reverse_dcf(
        ticker="TEST",
        market="US",
        current_price=100.0,
        shares_outstanding=1000000,
        base_revenue=100000000.0,
        base_ebit_margin_pct=25.0,
        beta=0.5,
        macro_registry=macro_reg,
    )

    assert result is not None
    assert result.is_actionable is False or result.is_eligible is True
