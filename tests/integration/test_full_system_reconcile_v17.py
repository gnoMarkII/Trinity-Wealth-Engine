"""End-to-End Golden Fixture & Integration Test for v17 Institutional Hardening & Valuation Integrity."""
import json
import pytest
from decimal import Decimal
from pathlib import Path

from schemas.micro_quant_schemas import (
    AtomicMarketSnapshot,
    MarginMetricItem,
    QuantSignals,
)


def test_golden_fixture_precision_and_exact_cents():
    fixture_path = Path(__file__).resolve().parents[1] / "fixtures" / "expected_ftnt_reconcile.json"
    assert fixture_path.exists(), f"Golden fixture file missing at {fixture_path}"

    with open(fixture_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Market Snapshot Exact Cent Assertions
    raw_price = Decimal(data["raw_analysis_price_str"])
    shares = int(data["shares_outstanding_str"])
    market_cap_dec = raw_price * Decimal(shares)
    market_cap_cents = int(market_cap_dec * 100)

    assert market_cap_cents == data["market_cap_cents"]
    assert market_cap_cents == 11875155473805
    assert data["market_cap_cents"] == 11875155473805

    # 2. TTM Accounting Exact Assertions
    ttm = data["ttm_accounting"]
    rev_dec = Decimal(ttm["raw_ttm_revenue_usd_str"])
    op_inc_dec = Decimal(ttm["raw_ttm_operating_income_usd_str"])
    fcf_dec = Decimal(ttm["raw_ttm_fcf_usd_str"])
    ocf_dec = Decimal(ttm["raw_ttm_operating_cash_flow_usd_str"])
    ni_dec = Decimal(ttm["raw_ttm_net_income_usd_str"])

    assert rev_dec == Decimal("7527400000.00")
    assert op_inc_dec == Decimal("2442200000.00")

    # GAAP margin = 2442.2 / 7527.4 = 32.444137...% -> 32.44%
    gaap_margin_exact = (op_inc_dec / rev_dec) * 100
    assert round(float(gaap_margin_exact), 2) == 32.44
    assert ttm["standardized_ttm_gaap_operating_margin_pct"] == 32.44

    # FCF margin = 3117.0 / 7527.4 = 41.4087...% -> 41.41%
    fcf_margin_exact = (fcf_dec / rev_dec) * 100
    assert round(float(fcf_margin_exact), 2) == 41.41
    assert ttm["fcf_margin_pct"] == 41.41

    # FCF yield = 3,117,000,000 / 118,751,554,738.05 = 2.6248077...% -> 2.62%
    fcf_yield_exact = (fcf_dec / market_cap_dec) * 100
    assert round(float(fcf_yield_exact), 2) == 2.62
    assert ttm["fcf_yield_pct"] == 2.62

    # OCF / NI = 3396.1 / 2120.7 = 1.6014... -> 1.60
    ocf_to_ni = round(float(ocf_dec / ni_dec), 2)
    assert ocf_to_ni == 1.60
    assert ttm["ocf_to_net_income"] == 1.60

    # 3. Form 4 Insider Conviction Exact Assertions
    insider = data["insider_conviction"]
    assert insider["rule_10b5_1_s_value_cents"] == 2626687822
    assert insider["rule_10b5_1_s_value_usd_str"] == "26266878.22"
    assert insider["filing_count_90d"] == 1
    assert insider["transaction_lot_count_90d"] == 6
    assert insider["quarantined_filing_count_90d"] == 0

    # 4. 4-Tier Margin Taxonomy Integrity
    margins = data["margin_taxonomy"]
    assert margins["gaap_operating_margin_pct"] == 32.44
    assert margins["non_gaap_operating_margin_pct"] == 36.0
    assert margins["historical_gaap_operating_margin_pct"] == 30.66
    assert margins["provider_ebit_margin_pct"] == 30.66

    # 5. Guidance Midpoint Comparisons
    guidance = data["guidance"]
    assert guidance["computed_revenue_growth_yoy_pct"] == 19.1
    assert guidance["computed_billings_growth_yoy_pct"] == 25.1
    assert guidance["non_gaap_operating_margin_midpoint_pct"] == 36.0
