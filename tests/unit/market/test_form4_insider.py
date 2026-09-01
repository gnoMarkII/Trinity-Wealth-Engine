"""Unit tests for Canonical Form 4 Insider Conviction."""
import pytest
from tools.market.ownership import compute_canonical_insider_conviction


def test_form4_th_market_not_applicable():
    """Test that Thai market returns status not_applicable."""
    res, flags = compute_canonical_insider_conviction("PTT", market="TH")
    assert res.status == "not_applicable"
    assert res.data_status == "not_applicable"
    assert res.open_market_p_count_90d == 0


def test_form4_code_p_csuite_cluster():
    """Test C-suite cluster buying with Code P."""
    txs = [
        {
            "transaction_date": "2026-08-10",
            "transaction_code": "P",
            "shares": 10_000,
            "price_per_share": 75.0,
            "officer_title": "Chief Executive Officer (CEO)",
            "is_c_suite": True,
        },
        {
            "transaction_date": "2026-08-12",
            "transaction_code": "P",
            "shares": 5_000,
            "price_per_share": 76.0,
            "officer_title": "Chief Financial Officer (CFO)",
            "is_c_suite": True,
        },
        {
            "transaction_date": "2026-08-01",
            "transaction_code": "M",  # Option exercise - not counted as P
            "shares": 20_000,
            "price_per_share": 15.0,
            "officer_title": "Director",
        },
    ]

    res, flags = compute_canonical_insider_conviction("FTNT", market="US", canonical_transactions=txs)
    assert res.status == "bullish_cluster"
    assert res.data_status == "available"
    assert res.open_market_p_count_90d == 2
    assert res.c_suite_p_count == 2
    assert res.insider_buy_range_min == 75.0
    assert res.insider_buy_range_max == 76.0
    assert res.open_market_p_value_usd == (10_000 * 75.0 + 5_000 * 76.0)


def test_form4_code_s_contextual_selling():
    """Test that Code S is captured as contextual selling activity without error."""
    txs = [
        {
            "transaction_date": "2026-08-15",
            "transaction_code": "S",
            "shares": 50_000,
            "price_per_share": 85.0,
            "officer_title": "Director",
            "is_c_suite": False,
        }
    ]

    res, flags = compute_canonical_insider_conviction("AAPL", market="US", canonical_transactions=txs)
    assert res.status == "selling_activity"
    assert res.open_market_p_count_90d == 0
    assert res.open_market_s_count_90d == 1
    assert res.open_market_s_value_usd == (50_000 * 85.0)


def test_form4_neutral_no_signal():
    """Test empty transactions return neutral_no_signal."""
    res, flags = compute_canonical_insider_conviction("MSFT", market="US", canonical_transactions=[])
    assert res.status == "neutral_no_signal"
    assert res.data_status == "available"
    assert res.open_market_p_count_90d == 0
