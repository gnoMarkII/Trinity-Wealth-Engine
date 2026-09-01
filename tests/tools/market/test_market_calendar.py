from datetime import datetime, timezone
from tools.market.quant_engine import is_us_trading_day, get_us_market_session_info


def test_is_us_trading_day_weekends_and_holidays():
    # Monday 2026-08-31 is a normal trading day
    assert is_us_trading_day(datetime(2026, 8, 31)) is True

    # Saturday 2026-08-29 and Sunday 2026-08-30 are not trading days
    assert is_us_trading_day(datetime(2026, 8, 29)) is False
    assert is_us_trading_day(datetime(2026, 8, 30)) is False

    # US Independence Day July 4th
    assert is_us_trading_day(datetime(2026, 7, 4)) is False

    # Christmas Dec 25th
    assert is_us_trading_day(datetime(2026, 12, 25)) is False


def test_market_session_and_freshness_calculation():
    # Simulate run on Monday 2026-08-31 at 08:00 ET (pre-market, 12:00 UTC)
    # Latest OHLCV is Thursday 2026-08-27
    run_dt = datetime(2026, 8, 31, 12, 0, 0, tzinfo=timezone.utc)
    res = get_us_market_session_info(run_dt, "2026-08-27")

    assert res["market_session_status"] == "pre_market"
    assert res["expected_latest_session_date"] == "2026-08-28"  # Friday
    assert res["actual_latest_session_date"] == "2026-08-27"    # Thursday
    assert res["missing_trading_sessions"] == 1
    assert res["data_freshness_status"] == "stale_one_session"


def test_market_session_fresh():
    # If run on Friday 2026-08-28 after close (21:00 UTC = 17:00 EDT) with today's OHLCV
    run_dt = datetime(2026, 8, 28, 21, 0, 0, tzinfo=timezone.utc)
    res = get_us_market_session_info(run_dt, "2026-08-28")

    assert res["market_session_status"] == "after_hours"
    assert res["expected_latest_session_date"] == "2026-08-28"
    assert res["missing_trading_sessions"] == 0
    assert res["data_freshness_status"] == "fresh"
