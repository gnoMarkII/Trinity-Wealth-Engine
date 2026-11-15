"""Unit tests for Canonical Form 4 Insider Conviction."""
from datetime import datetime, timezone
import pytest
from tools.market.ownership import compute_canonical_insider_conviction


_PINNED_AS_OF = datetime(2026, 8, 20, tzinfo=timezone.utc)


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

    res, flags = compute_canonical_insider_conviction(
        "FTNT",
        market="US",
        canonical_transactions=txs,
        as_of=_PINNED_AS_OF,
    )
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

    res, flags = compute_canonical_insider_conviction(
        "AAPL",
        market="US",
        canonical_transactions=txs,
        as_of=_PINNED_AS_OF,
    )
    assert res.status == "selling_activity"
    assert res.open_market_p_count_90d == 0
    assert res.open_market_s_count_90d == 1
    assert res.open_market_s_value_usd == (50_000 * 85.0)


def test_form4_excludes_transactions_older_than_90_days():
    """Transactions strictly older than 90 days before as_of are excluded."""
    txs = [
        {
            "transaction_date": "2026-05-01",
            "transaction_code": "P",
            "shares": 10_000,
            "price_per_share": 75.0,
            "officer_title": "CEO",
            "is_c_suite": True,
        }
    ]
    res, _ = compute_canonical_insider_conviction(
        "FTNT",
        market="US",
        canonical_transactions=txs,
        as_of=_PINNED_AS_OF,
    )
    assert res.status == "neutral_no_signal"
    assert res.open_market_p_count_90d == 0


def test_form4_neutral_no_signal():
    """Test empty transactions return neutral_no_signal."""
    res, flags = compute_canonical_insider_conviction("MSFT", market="US", canonical_transactions=[])
    assert res.status == "neutral_no_signal"
    assert res.data_status == "available"
    assert res.open_market_p_count_90d == 0


def test_yf_insider_zero_dollar_awards_and_exercises_not_mislabeled_as_buys():
    """Option 2: $0 grants/RSUs/awards and option exercises must NEVER classify as open-market buy 'P'."""
    from unittest.mock import MagicMock, patch
    import pandas as pd
    from tools.market.adapters.insider_provider import YFinanceInsiderHistoryAdapter, _classify_yf_insider_transaction
    from tools.market.ownership import compute_smart_money_flags
    from tools.market.terminal_v2.adapters.sec_edgar_adapter import SecEdgarAdapter

    # Direct classifier checks
    assert _classify_yf_insider_transaction("Purchase", "Stock Award (Grant) RSU vesting", 0.0) == ("A", "A")
    assert _classify_yf_insider_transaction("Purchase", "Option Exercise", 0.0) == ("M", "A")
    assert _classify_yf_insider_transaction("Purchase", "", 0.0) == ("A", "A")
    assert _classify_yf_insider_transaction("Tax Withholding", "Surrender shares for tax", 50.0) == ("F", "D")
    assert _classify_yf_insider_transaction("Purchase", "Open market purchase", 65.0) == ("P", "A")

    # Adapter & ownership integration with mock yfinance table
    df = pd.DataFrame(
        [
            {"Insider": "Ken Xie", "Position": "CEO", "Transaction": "Purchase", "Text": "Stock Award(Grant) at $0", "Start Date": "2026-08-10", "Shares": 5000, "Value": 0.0},
            {"Insider": "John Whittle", "Position": "Officer", "Transaction": "Option Exercise", "Text": "Conversion or exercise", "Start Date": "2026-08-11", "Shares": 3000, "Value": 0.0},
            {"Insider": "Jane Doe", "Position": "Director", "Transaction": "Sale", "Text": "Automatic Sale under 10b5-1", "Start Date": "2026-08-12", "Shares": 1000, "Value": 80000.0},
        ]
    )
    fake_tk = MagicMock()
    fake_tk.insider_transactions = df
    fake_tk.info = {"heldPercentInsiders": 0.175}
    records = YFinanceInsiderHistoryAdapter(ticker_factory=lambda _: fake_tk).fetch("FTNT")
    assert [r["transactions"][0]["transaction_code"] for r in records] == ["A", "M", "S"]

    with patch("tools.market.ownership.yf.Ticker", return_value=fake_tk):
        sm_flags, _ = compute_smart_money_flags("FTNT", info_dict=fake_tk.info, as_of=_PINNED_AS_OF)
        assert sm_flags.insider_buy_count_90d == 0
        assert sm_flags.insider_sell_count_90d == 1
        assert sm_flags.insider_ownership_pct == 17.5

        with patch("tools.market.terminal_v2.bootstrap.get_terminal_data_service", side_effect=RuntimeError("offline")):
            conv, _ = compute_canonical_insider_conviction("FTNT", market="US", as_of=_PINNED_AS_OF)
        assert conv.open_market_p_count_90d == 0
        assert conv.open_market_s_count_90d == 1

    # SEC EDGAR CIK resolution for FTNT and dynamic fallback
    adapter = SecEdgarAdapter()
    assert adapter._resolve_cik("FTNT") == "0001262039"
    fake_resp = MagicMock()
    fake_resp.json.return_value = {"0": {"ticker": "DYNCO", "cik_str": 123456}}
    with patch.object(adapter, "_request_get", return_value=fake_resp):
        assert adapter._resolve_cik("DYNCO") == "0000123456"

