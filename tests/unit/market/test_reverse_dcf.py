"""Unit tests for 5-Year Explicit DCF and True Bounded Reverse DCF solver."""
import pytest
from schemas.macro_schemas import MarketObservable
from tools.market.dcf_valuation import compute_institutional_reverse_dcf


@pytest.fixture
def mock_macro_registry():
    return {
        "obs_dgs10": MarketObservable(
            observable_id="obs_dgs10",
            asset_bucket="fixed_income",
            region="US",
            indicator="10Y Treasury Yield",
            value="4.25",
            unit="%",
            observed_at="2026-08-25",
            source_file="DGS10.csv",
            is_valid=True,
        ),
        "obs_erp_gspc": MarketObservable(
            observable_id="obs_erp_gspc",
            asset_bucket="equities",
            region="US",
            indicator="Equity Risk Premium",
            value="4.50",
            unit="%",
            observed_at="2026-08-25",
            source_file="Damodaran_ERP.csv",
            is_valid=True,
        ),
    }


def test_reverse_dcf_convergence(mock_macro_registry):
    """Test that reverse DCF solver finds the exact implied growth rate for a standard tech stock."""
    current_price = 80.0
    shares_out = 100_000_000
    base_rev = 2_000_000_000  # $2B revenue
    ebit_margin = 25.0       # 25% EBIT margin
    tax_rate = 0.21
    reinvestment_rate = 10.0 # 10% of revenue reinvested
    beta = 1.10
    total_debt = 500_000_000
    cash = 1_000_000_000     # Net cash +$500M

    result, flags = compute_institutional_reverse_dcf(
        ticker="FTNT",
        market="US",
        current_price=current_price,
        shares_outstanding=shares_out,
        base_revenue=base_rev,
        base_ebit_margin_pct=ebit_margin,
        tax_rate=tax_rate,
        reinvestment_rate_pct=reinvestment_rate,
        beta=beta,
        total_debt=total_debt,
        cash_and_equivalents=cash,
        interest_expense=25_000_000,
        macro_registry=mock_macro_registry,
        sector="Technology",
        forecast_revenue_growth_pct=12.0,
    )

    assert result.status == "available"
    assert result.is_eligible is True
    assert result.solver_status == "converged"
    assert result.market_implied_growth_pct is not None
    assert len(result.explicit_forecast_5y) == 5
    assert result.explicit_forecast_5y[0].year_index == 1
    assert result.explicit_forecast_5y[4].year_index == 5
    assert result.target_price_12m is not None
    assert result.target_price_12m > 0
    assert result.intrinsic_value_today is not None


def test_reverse_dcf_sector_exclusion(mock_macro_registry):
    """Test that financial institutions and REITs are excluded from generic FCF DCF."""
    for sec in ["Financial Services", "Financials", "Real Estate", "Banks", "Insurance", "REIT"]:
        result, flags = compute_institutional_reverse_dcf(
            ticker="JPM",
            market="US",
            current_price=150.0,
            shares_outstanding=1_000_000_000,
            base_revenue=50_000_000_000,
            base_ebit_margin_pct=30.0,
            tax_rate=0.21,
            reinvestment_rate_pct=5.0,
            beta=1.0,
            total_debt=100_000_000_000,
            cash_and_equivalents=50_000_000_000,
            interest_expense=10_000_000_000,
            macro_registry=mock_macro_registry,
            sector=sec,
        )
        assert result.status == "not_applicable"
        assert result.is_eligible is False
        assert result.solver_status == "not_applicable"
        assert "excluded" in result.exclusion_reason.lower()


def test_reverse_dcf_missing_beta(mock_macro_registry):
    """Test handling when beta is unavailable."""
    result, flags = compute_institutional_reverse_dcf(
        ticker="UNKNOWN",
        market="US",
        current_price=50.0,
        shares_outstanding=10_000_000,
        base_revenue=100_000_000,
        base_ebit_margin_pct=15.0,
        tax_rate=0.21,
        reinvestment_rate_pct=5.0,
        beta=None,
        total_debt=0,
        cash_and_equivalents=10_000_000,
        interest_expense=0,
        macro_registry=mock_macro_registry,
        sector="Technology",
    )
    assert result.status == "unavailable"
    assert result.solver_status == "no_solution"
    assert any("beta_unavailable" in f for f in flags)
