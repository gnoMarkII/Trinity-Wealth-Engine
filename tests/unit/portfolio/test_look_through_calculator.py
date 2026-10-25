from dataclasses import dataclass
from decimal import Decimal
import pytest

from core.portfolio.look_through_calculator import LookThroughCalculator
from core.portfolio_advisory.guardrails import check_look_through_concentration_cap
from tools.portfolio.ports.thai_fund_port import (
    FundConstituentsData,
    FundHoldingItem,
    FundSectorItem,
)


@dataclass
class MockHolding:
    symbol: str
    asset_type: str
    market_value_thb: float
    company_name: str = ""
    sector: str = ""


def test_look_through_calculator_exact_math():
    holdings = [
        # Direct stock: 50,000 THB (5%)
        MockHolding(symbol="MSFT", asset_type="EQUITY", market_value_thb=50000.0, company_name="Microsoft Corp", sector="Information Technology"),
        # Other stock: 350,000 THB
        MockHolding(symbol="AAPL", asset_type="EQUITY", market_value_thb=350000.0, company_name="Apple Inc", sector="Information Technology"),
        # Fund 1: 300,000 THB (contains MSFT 7.0%)
        MockHolding(symbol="SCBS&P500E", asset_type="FUND", market_value_thb=300000.0),
        # Fund 2: 300,000 THB (contains MSFT 8.0%)
        MockHolding(symbol="TLNDQINCOME", asset_type="FUND", market_value_thb=300000.0),
    ]

    fund_map = {
        "SCBS&P500E": FundConstituentsData(
            fund_id="F1",
            symbol="SCBS&P500E",
            as_of_date="2026-03-31",
            top_holdings=[
                FundHoldingItem(name="Microsoft Corporation", percent=Decimal("7.0"), ticker="MSFT"),
            ],
            sector_breakdown=[
                FundSectorItem(sector_name_th="เทคโนโลยี", sector_name_en="Information Technology", percent=Decimal("30.0")),
            ],
        ),
        "TLNDQINCOME": FundConstituentsData(
            fund_id="F2",
            symbol="TLNDQINCOME",
            as_of_date="2026-03-31",
            top_holdings=[
                FundHoldingItem(name="Microsoft Corporation", percent=Decimal("8.0"), ticker="MSFT"),
            ],
            sector_breakdown=[
                FundSectorItem(sector_name_th="เทคโนโลยี", sector_name_en="Information Technology", percent=Decimal("40.0")),
            ],
        ),
    }

    # Total NAV = 50k + 350k + 300k + 300k = 1,000,000 THB
    report = LookThroughCalculator.calculate(holdings, fund_map, total_nav_thb=Decimal("1000000.0"))

    assert report.total_nav_thb == Decimal("1000000.00")

    msft_exp = next(h for h in report.holdings_exposure if h.ticker == "MSFT")
    assert msft_exp.direct_value_thb == Decimal("50000.00")
    assert msft_exp.direct_weight_pct == Decimal("5.00")

    # Indirect = (300,000 * 0.07) + (300,000 * 0.08) = 21,000 + 24,000 = 45,000 THB (4.5%)
    assert msft_exp.indirect_value_thb == Decimal("45000.00")
    assert msft_exp.indirect_weight_pct == Decimal("4.50")

    # Total MSFT = 95,000 THB (9.5%)
    assert msft_exp.total_value_thb == Decimal("95000.00")
    assert msft_exp.total_weight_pct == Decimal("9.50")
    assert msft_exp.is_cap_breached is False
    assert msft_exp.is_hidden_concentration is False
    assert len(msft_exp.source_funds) == 2


def test_look_through_hidden_concentration_cap_warning():
    holdings = [
        # Direct NVDA = 120,000 THB (12.0%)
        MockHolding(symbol="NVDA", asset_type="EQUITY", market_value_thb=120000.0, company_name="NVIDIA Corporation"),
        # Other holdings = 280,000 THB
        MockHolding(symbol="CASH", asset_type="CASH", market_value_thb=280000.0),
        # Fund = 600,000 THB (contains NVDA 8.0% = 48,000 THB = 4.8%)
        MockHolding(symbol="SCBS&P500E", asset_type="FUND", market_value_thb=600000.0),
    ]

    fund_map = {
        "SCBS&P500E": FundConstituentsData(
            fund_id="F1",
            symbol="SCBS&P500E",
            as_of_date="2026-03-31",
            top_holdings=[
                FundHoldingItem(name="NVIDIA Corporation", percent=Decimal("8.0"), ticker="NVDA"),
            ],
            sector_breakdown=[
                FundSectorItem(sector_name_th="เทคโนโลยี", sector_name_en="Information Technology", percent=Decimal("35.0")),
            ],
        ),
    }

    # Total NAV = 1,000,000 THB
    report = LookThroughCalculator.calculate(holdings, fund_map, total_nav_thb=Decimal("1000000.0"))

    nvda_exp = next(h for h in report.holdings_exposure if h.ticker == "NVDA")
    # Total NVDA = 120,000 + 48,000 = 168,000 THB (16.8%)
    assert nvda_exp.total_weight_pct == Decimal("16.80")
    assert nvda_exp.direct_weight_pct == Decimal("12.00")
    assert nvda_exp.is_cap_breached is True
    assert nvda_exp.is_hidden_concentration is True

    assert len(report.hidden_concentration_warnings) == 1
    assert "NVDA" in report.hidden_concentration_warnings[0]
    assert "16.8%" in report.hidden_concentration_warnings[0]
    assert "15.0% Cap" in report.hidden_concentration_warnings[0]

    # Verify Guardrail integration
    breached, warning = check_look_through_concentration_cap(nvda_exp)
    assert breached is True
    assert warning is not None
    assert warning.guardrail_code == "ANOMALY_LOOK_THROUGH_CONCENTRATION_CAP"
    assert warning.ticker == "NVDA"
    assert "exceeding 15.0% Cap" in warning.message


def test_true_sector_crowding():
    holdings = [
        # Direct Tech: 200,000 THB
        MockHolding(symbol="MSFT", asset_type="EQUITY", market_value_thb=200000.0, sector="เทคโนโลยี"),
        # Fund with Tech: 400,000 THB (50% Tech = 200,000 THB)
        MockHolding(symbol="TLNDQINCOME", asset_type="FUND", market_value_thb=400000.0),
        # Non-tech: 400,000 THB
        MockHolding(symbol="CASH", asset_type="CASH", market_value_thb=400000.0),
    ]

    fund_map = {
        "TLNDQINCOME": FundConstituentsData(
            fund_id="F2",
            symbol="TLNDQINCOME",
            as_of_date="2026-03-31",
            top_holdings=[],
            sector_breakdown=[
                FundSectorItem(sector_name_th="เทคโนโลยี", sector_name_en="Information Technology", percent=Decimal("50.0")),
            ],
        ),
    }

    report = LookThroughCalculator.calculate(holdings, fund_map, total_nav_thb=Decimal("1000000.0"))

    tech_sec = next(s for s in report.sector_exposure if s.sector_name == "Information Technology")
    # Total Tech = 200k direct + 200k fund = 400k (40.0%)
    assert tech_sec.total_value_thb == Decimal("400000.00")
    assert tech_sec.weight_pct == Decimal("40.00")
    assert tech_sec.is_crowded is True
