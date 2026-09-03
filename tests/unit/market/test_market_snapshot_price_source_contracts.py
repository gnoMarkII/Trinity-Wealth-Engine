"""Unit tests for AtomicMarketSnapshot PriceSource contracts, Decimal representations, and semantics."""
import pytest
from datetime import datetime, timezone
import pandas as pd

from schemas.micro_quant_schemas import AtomicMarketSnapshot, PriceSource
from tools.market.quant_engine import create_atomic_market_snapshot


def test_price_source_exact_eod_match():
    # Setup dataframe with matching target EOD bar
    now_dt = datetime.now(timezone.utc)
    dates = pd.date_range(end=now_dt, periods=50, freq="B")
    df = pd.DataFrame({
        "Close": [100.0 + i for i in range(len(dates))],
        "Open": [99.0 + i for i in range(len(dates))],
        "High": [101.0 + i for i in range(len(dates))],
        "Low": [98.0 + i for i in range(len(dates))],
        "Volume": [1000000] * len(dates)
    }, index=dates)

    info = {
        "currentPrice": 149.0,
        "sharesOutstanding": 733713653,
    }

    snapshot, flags = create_atomic_market_snapshot(
        provider_symbol="TEST",
        df_1y=df,
        info=info,
        force_intraday_mode=False
    )

    assert snapshot.price_source in ("ohlcv_close", "stale_eod")
    assert snapshot.is_provisional is False
    assert snapshot.raw_analysis_price_str is not None
    assert snapshot.market_cap_cents is not None
    assert isinstance(snapshot.market_cap_cents, int)


def test_price_source_force_intraday_mode():
    now_dt = datetime.now(timezone.utc)
    dates = pd.date_range(end=now_dt, periods=50, freq="B")
    df = pd.DataFrame({
        "Close": [100.0 + i for i in range(len(dates))],
    }, index=dates)

    info = {
        "currentPrice": 161.85,
        "sharesOutstanding": 733713653,
    }

    snapshot, flags = create_atomic_market_snapshot(
        provider_symbol="TEST",
        df_1y=df,
        info=info,
        force_intraday_mode=True
    )

    assert snapshot.price_source == "intraday_snapshot"
    assert snapshot.is_provisional is True
    assert snapshot.volume_confirmation == "unavailable"
    assert snapshot.raw_analysis_price_str == "161.8500"
    assert snapshot.market_cap_cents == 11875155473805  # 161.85 * 733713653


def test_price_source_unavailable_when_no_data():
    empty_df = pd.DataFrame()
    empty_info = {}

    snapshot, flags = create_atomic_market_snapshot(
        provider_symbol="NO_DATA",
        df_1y=empty_df,
        info=empty_info,
        force_intraday_mode=False
    )

    assert snapshot.price_source == "unavailable"
    assert snapshot.analysis_price is None
    assert snapshot.raw_analysis_price_str is None
    assert snapshot.market_cap is None
    assert snapshot.market_cap_cents is None
    assert snapshot.data_freshness_status == "unavailable"
