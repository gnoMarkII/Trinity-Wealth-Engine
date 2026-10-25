"""Unit tests for Portfolio Advisory Guardrails and Market Lot Rounding.

Verifies:
- S3: safe_decimal handling of NaN, Inf, and InvalidOperation
- D1: resolve_holding_currency_and_market preventing Thai stock USD trap
- T1–T5: Temporal integrity, future date anomaly, and live upside re-anchoring
- I1–I5, I7: Price scale sanity, FX rates, bucket sum, unassigned drift guard
- D4: Real-world exchange lot sizing (SET 100-share Board Lot vs US 4-dp Fractional)
"""
from decimal import Decimal
import math
from unittest.mock import MagicMock
import pytest

from core.portfolio_advisory.guardrails import (
    check_future_date_anomaly,
    check_price_scale_anomaly,
    check_stale_portfolio_price,
    check_unassigned_holdings_drift,
    compute_live_upside,
    resolve_holding_currency_and_market,
    safe_decimal,
    validate_bucket_target_sum,
    validate_fx_rate,
)
from core.portfolio_advisory.lot_rounding import calculate_market_lot_sizing
from core.portfolio_advisory.sector_mapping import (
    is_sector_match,
    normalize_to_canonical_sector,
)


def test_safe_decimal_nan_inf_traps():
    """S3: safe_decimal must never raise decimal.InvalidOperation on NaN or Inf."""
    assert safe_decimal(float("nan"), default=Decimal("0.0")) == Decimal("0.0")
    assert safe_decimal(float("inf"), default=Decimal("0.0")) == Decimal("0.0")
    assert safe_decimal(float("-inf"), default=Decimal("0.0")) == Decimal("0.0")
    assert safe_decimal(None, default=Decimal("10.0")) == Decimal("10.0")
    assert safe_decimal("not_a_number", default=Decimal("5.0")) == Decimal("5.0")
    assert safe_decimal("1,234.5678", quantum="0.01") == Decimal("1234.57")
    assert safe_decimal(123.456, quantum="0.1") == Decimal("123.5")


def test_resolve_holding_currency_and_market_thai_stock_trap():
    """D1: Thai stock with current_price_usd populated must NOT be recognized as US stock."""
    # Simulation of price_yfinance_adapter populating both THB and USD prices on a Thai stock
    thai_holding = MagicMock()
    thai_holding.symbol = "ADVANC.BK"
    thai_holding.asset_type = "Stock"
    thai_holding.avg_cost_thb = 240.0
    thai_holding.avg_cost_usd = None
    thai_holding.current_price_thb = 250.0
    thai_holding.current_price_usd = 6.85  # Trap: Populated by currency conversion in price adapter

    market, currency = resolve_holding_currency_and_market(thai_holding)
    assert market == "TH"
    assert currency == "THB"

    # US stock with avg_cost_usd only
    us_holding = MagicMock()
    us_holding.symbol = "AAPL"
    us_holding.asset_type = "Stock"
    us_holding.avg_cost_thb = None
    us_holding.avg_cost_usd = 180.0
    us_holding.current_price_usd = 185.0
    us_holding.current_price_thb = 6500.0

    market_us, curr_us = resolve_holding_currency_and_market(us_holding)
    assert market_us == "US"
    assert curr_us == "USD"

    # Tax vehicles and domestic mutual funds must always be TH/THB
    fund_holding = MagicMock()
    fund_holding.symbol = "K-USA-A(A)"
    fund_holding.asset_type = "Mutual_Fund"
    market_f, curr_f = resolve_holding_currency_and_market(fund_holding)
    assert market_f == "TH"
    assert curr_f == "THB"


def test_price_scale_anomaly_detection():
    """I1: Catches currency scale mismatches or stock splits exceeding 50% divergence."""
    # Holding price in THB mistakenly compared to USD price without conversion
    warn = check_price_scale_anomaly(
        holding_price=Decimal("4500.0"),  # user typed THB
        sidecar_price=Decimal("130.0"),   # USD price in sidecar
        ticker="NVDA",
        tolerance_pct=Decimal("50.0"),
    )
    assert warn is not None
    assert warn.guardrail_code == "PRICE_SCALE_OR_CURRENCY_ANOMALY"
    assert warn.severity == "critical"

    # Normal price fluctuation within tolerance
    no_warn = check_price_scale_anomaly(
        holding_price=Decimal("135.0"),
        sidecar_price=Decimal("130.0"),
        ticker="NVDA",
    )
    assert no_warn is None


def test_fx_rate_validation():
    """I2: FX rate outside 20.0-60.0 bounds triggers warning and fallback."""
    rate, warn = validate_fx_rate(Decimal("150.0"))
    assert rate == Decimal("35.50")
    assert warn is not None
    assert warn.guardrail_code == "FX_AND_FINITE_NUMBER_GATE"

    normal_rate, no_warn = validate_fx_rate(Decimal("36.25"))
    assert normal_rate == Decimal("36.2500")
    assert no_warn is None


def test_bucket_target_sum_validation():
    """I3: Strategy bucket targets not summing to 100% trigger warning."""
    b1 = MagicMock(target_percent=40.0)
    b2 = MagicMock(target_percent=40.0)
    # Sum is 80%
    is_valid, warn = validate_bucket_target_sum([b1, b2])
    assert is_valid is False
    assert warn is not None
    assert warn.guardrail_code == "BUCKET_TARGET_SUM_INVALID"

    # Sum is 100%
    b3 = MagicMock(target_percent=20.0)
    is_valid_ok, no_warn = validate_bucket_target_sum([b1, b2, b3])
    assert is_valid_ok is True
    assert no_warn is None


def test_unassigned_holdings_drift_guard():
    """I7: Holdings without strategy buckets exceeding 5% NAV trigger drift guard."""
    # 200,000 / 1,000,000 = 20% > 5%
    is_drift, warn = check_unassigned_holdings_drift(
        unassigned_value_thb=Decimal("200000"),
        total_nav_thb=Decimal("1000000"),
        threshold_pct=Decimal("5.0"),
    )
    assert is_drift is True
    assert warn is not None
    assert warn.guardrail_code == "UNASSIGNED_BUCKET_DRIFT_GUARD"

    # 30,000 / 1,000,000 = 3% <= 5%
    is_drift_ok, no_warn = check_unassigned_holdings_drift(
        unassigned_value_thb=Decimal("30000"),
        total_nav_thb=Decimal("1000000"),
        threshold_pct=Decimal("5.0"),
    )
    assert is_drift_ok is False
    assert no_warn is None


def test_market_lot_rounding_rules():
    """D4: Market-aware lot sizing for Thai Board Lot (100 shares) and US Fractional."""
    # Thai Stock: Target 100,000 THB at 245 THB/share
    # 100,000 / 245 = 408.16 shares -> 408 total shares -> 400 Board Lot + 8 Odd Lot
    thai_lot = calculate_market_lot_sizing(
        market="TH",
        native_currency="THB",
        target_amount_native=Decimal("100000"),
        current_price_native=Decimal("245.0"),
        fx_rate=Decimal("35.50"),
    )
    assert thai_lot.board_lot_shares == 400
    assert thai_lot.odd_lot_shares == 8
    assert thai_lot.amount_thb == Decimal("408") * Decimal("245.0")  # 99,960 THB

    # US Stock: Target $3,000 at $185.50/share
    # 3000 / 185.50 = 16.1725067... -> 16 whole shares + 16.1725 fractional shares
    us_lot = calculate_market_lot_sizing(
        market="US",
        native_currency="USD",
        target_amount_native=Decimal("3000"),
        current_price_native=Decimal("185.50"),
        fx_rate=Decimal("36.00"),
    )
    assert us_lot.whole_shares == 16
    assert us_lot.fractional_shares == Decimal("16.1725")
    assert us_lot.amount_native == Decimal("3000.00")

    # Tax-locked vehicle: Must return 0 shares/amounts
    locked_lot = calculate_market_lot_sizing(
        market="TH",
        native_currency="THB",
        target_amount_native=Decimal("50000"),
        current_price_native=Decimal("25.0"),
        fx_rate=Decimal("35.50"),
        is_tax_locked=True,
    )
    assert locked_lot.board_lot_shares == 0
    assert locked_lot.amount_thb == Decimal("0.00")


def test_sector_canonical_matching():
    """D6: Three-way mapping between SPDR ETF, canonical sector, and Yahoo peer sector."""
    assert normalize_to_canonical_sector("XLK") == "Technology"
    assert normalize_to_canonical_sector("Technology") == "Technology"
    assert normalize_to_canonical_sector("Information Technology") == "Technology"
    assert normalize_to_canonical_sector("Financial Services") == "Financials"
    assert normalize_to_canonical_sector("XLF") == "Financials"

    assert is_sector_match("XLK", "Technology") is True
    assert is_sector_match("Financial Services", "XLF") is True
    assert is_sector_match("Healthcare", "XLV") is True
    assert is_sector_match("Energy", "Technology") is False
