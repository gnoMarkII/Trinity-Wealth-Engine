"""Unit tests for Look-Through Calculator integrity, precision, and coverage.

Verifies:
- F04: Cap comparison uses exact unrounded Decimal precision before rounding for display.
- F07: Asset classification across REIT, Gold, Bond, Cash, and Equity.
- P0-07: LookThroughCoverage calculations and reconciliation.
"""
from dataclasses import dataclass
from decimal import Decimal
import pytest

from core.portfolio.look_through_calculator import LookThroughCalculator
from core.portfolio.look_through_models import (
    FundConstituentHolding,
    FundConstituentSector,
    FundConstituentsDomain,
)


@dataclass
class MockHolding:
    symbol: str
    asset_type: str
    market_value_thb: float
    company_name: str = ""
    sector: str = ""


def test_unrounded_cap_breach_detection():
    """F04: 15.004% must be detected as cap breached even though rounded to 15.00%."""
    # NAV = 1,000,000 THB. 15% Cap = 150,000 THB.
    # Holding = 150,040 THB -> 15.004% -> rounded to 15.00%, but raw > 15.0%
    holdings = [
        MockHolding(symbol="MSFT", asset_type="EQUITY", market_value_thb=150040.0),
        MockHolding(symbol="CASH", asset_type="CASH", market_value_thb=849960.0),
    ]

    report = LookThroughCalculator.calculate(holdings, {}, total_nav_thb=Decimal("1000000.0"))

    msft = next(h for h in report.holdings_exposure if h.ticker == "MSFT")
    assert msft.total_weight_pct == Decimal("15.00")
    assert msft.is_cap_breached is True
    assert report.cap_breached_count == 1


def test_reit_and_gold_asset_classification():
    """F07: Gold and REITs are properly classified and not aggregated into equity caps."""
    holdings = [
        MockHolding(symbol="CPNREIT", asset_type="REIT", market_value_thb=100000.0, company_name="CPN Retail Growth Leasehold REIT"),
        MockHolding(symbol="GOLD", asset_type="GOLD", market_value_thb=100000.0),
        MockHolding(symbol="AAPL", asset_type="EQUITY", market_value_thb=200000.0),
        MockHolding(symbol="CASH", asset_type="CASH", market_value_thb=600000.0),
    ]

    report = LookThroughCalculator.calculate(holdings, {}, total_nav_thb=Decimal("1000000.0"))

    # Gold is in sectors/asset tracking but not in holdings_exposure (which tracks stocks/REITs)
    tickers = {h.ticker for h in report.holdings_exposure}
    assert "AAPL" in tickers
    assert "CPNREIT" in tickers
    assert "GOLD" not in tickers

    cpnreit = next(h for h in report.holdings_exposure if h.ticker == "CPNREIT")
    assert cpnreit.asset_type == "REIT"
    assert cpnreit.market == "TH"

    # Real Estate sector includes CPNREIT
    re_sector = next((s for s in report.sector_exposure if s.sector_name == "Real Estate"), None)
    assert re_sector is not None
    assert re_sector.total_value_thb == Decimal("100000.00")


def test_look_through_coverage_metrics():
    """P0-07: Calculates coverage metrics and identifies unresolved values."""
    holdings = [
        # Fund 1 with constituents resolved: 500,000 THB (Top holdings = 50%, sectors = 100%)
        MockHolding(symbol="RESOLVED_FUND", asset_type="FUND", market_value_thb=500000.0),
        # Fund 2 missing constituents: 500,000 THB
        MockHolding(symbol="UNRESOLVED_FUND", asset_type="FUND", market_value_thb=500000.0),
    ]

    fund_map = {
        "RESOLVED_FUND": FundConstituentsDomain(
            fund_id="F1",
            symbol="RESOLVED_FUND",
            as_of_date="2026-03-31",
            top_holdings=[
                FundConstituentHolding(name="Microsoft Corporation", percent=Decimal("50.0"), ticker="MSFT"),
            ],
            sector_breakdown=[
                FundConstituentSector(sector_name_th="เทคโนโลยี", sector_name_en="Information Technology", percent=Decimal("100.0")),
            ],
        )
    }

    report = LookThroughCalculator.calculate(holdings, fund_map, total_nav_thb=Decimal("1000000.0"))

    assert report.coverage is not None
    cov = report.coverage
    assert cov.total_fund_value_thb == Decimal("1000000.00")
    assert cov.resolved_fund_value_thb == Decimal("500000.00")
    assert cov.fund_fetch_coverage == Decimal("50.00")  # 500k / 1M
    # security coverage: 500k * 0.50 / 1M = 25%
    assert cov.security_look_through_coverage == Decimal("25.00")
    # sector coverage: 500k * 1.00 / 1M = 50%
    assert cov.sector_coverage == Decimal("50.00")
    assert cov.unresolved_value_thb == Decimal("500000.00")
