"""Unit tests for Institutional Market Calendar & Session Resolver (Phase 1 / P0.1)."""
from datetime import date, datetime, time
from zoneinfo import ZoneInfo
import pytest

from tools.market.market_calendar import (
    NY_TZ,
    get_last_completed_regular_session,
    get_nyse_close_time,
    get_us_early_close_days,
    get_us_market_holidays,
    get_us_market_session_state,
    is_us_trading_day,
)


def test_us_market_holidays_2026():
    """Verify standard 2026 US market holidays."""
    holidays = get_us_market_holidays(2026)
    # MLK Day: 3rd Monday in Jan 2026 = Jan 19, 2026
    assert date(2026, 1, 19) in holidays
    # Presidents Day: 3rd Monday in Feb 2026 = Feb 16, 2026
    assert date(2026, 2, 16) in holidays
    # Good Friday 2026: Easter is April 5, 2026 -> Good Friday is April 3, 2026
    assert date(2026, 4, 3) in holidays
    # Memorial Day: Last Monday in May 2026 = May 25, 2026
    assert date(2026, 5, 25) in holidays
    # Juneteenth 2026: June 19, 2026 (Friday)
    assert date(2026, 6, 19) in holidays
    # July 4th 2026 is Saturday -> observed Friday July 3, 2026
    assert date(2026, 7, 3) in holidays
    # Labor Day: 1st Monday in Sep 2026 = Sep 7, 2026
    assert date(2026, 9, 7) in holidays
    # Thanksgiving: 4th Thursday in Nov 2026 = Nov 26, 2026
    assert date(2026, 11, 26) in holidays
    # Christmas 2026: Dec 25, 2026 (Friday)
    assert date(2026, 12, 25) in holidays


def test_early_close_schedule_2026():
    """Verify NYSE 13:00 Early Close days."""
    early_closes = get_us_early_close_days(2026)
    # Black Friday: Nov 27, 2026
    assert date(2026, 11, 27) in early_closes
    assert get_nyse_close_time(date(2026, 11, 27)) == time(13, 0)
    # Regular trading day has 16:00 close
    assert get_nyse_close_time(date(2026, 8, 31)) == time(16, 0)


def test_last_completed_regular_session_regular_trading_day():
    """Verify session resolution on a regular trading day (Tuesday Sep 1, 2026)."""
    # 10:30 AM NY Time on Tuesday Sep 1, 2026 (Market open, not closed yet)
    as_of_morning = datetime(2026, 9, 1, 10, 30, tzinfo=NY_TZ)
    # Should roll back to Monday Aug 31, 2026
    assert get_last_completed_regular_session(as_of_morning) == date(2026, 8, 31)

    # 4:15 PM NY Time on Tuesday Sep 1, 2026 (Market has closed at 16:00)
    as_of_afternoon = datetime(2026, 9, 1, 16, 15, tzinfo=NY_TZ)
    # Should be Tuesday Sep 1, 2026
    assert get_last_completed_regular_session(as_of_afternoon) == date(2026, 9, 1)


def test_last_completed_regular_session_early_close_day():
    """Verify session resolution on an early close day (Black Friday Nov 27, 2026)."""
    # 12:30 PM NY Time on Black Friday (Close is at 13:00)
    as_of_morning = datetime(2026, 11, 27, 12, 30, tzinfo=NY_TZ)
    # Thanksgiving was Nov 26 (Holiday), so last completed was Wednesday Nov 25
    assert get_last_completed_regular_session(as_of_morning) == date(2026, 11, 25)

    # 1:15 PM NY Time on Black Friday (Market has closed at 13:00)
    as_of_after_close = datetime(2026, 11, 27, 13, 15, tzinfo=NY_TZ)
    assert get_last_completed_regular_session(as_of_after_close) == date(2026, 11, 27)


def test_last_completed_regular_session_weekend():
    """Verify session resolution on Sunday Sep 6, 2026."""
    sunday = datetime(2026, 9, 6, 14, 0, tzinfo=NY_TZ)
    # Last completed was Friday Sep 4, 2026
    assert get_last_completed_regular_session(sunday) == date(2026, 9, 4)


def test_market_session_states():
    """Verify session state classification."""
    # Pre-market: 06:00 AM NY on Sep 1, 2026
    assert get_us_market_session_state(datetime(2026, 9, 1, 6, 0, tzinfo=NY_TZ)) == "pre_market"
    # Regular open: 11:00 AM NY on Sep 1, 2026
    assert get_us_market_session_state(datetime(2026, 9, 1, 11, 0, tzinfo=NY_TZ)) == "regular_open"
    # After-hours: 17:30 NY on Sep 1, 2026
    assert get_us_market_session_state(datetime(2026, 9, 1, 17, 30, tzinfo=NY_TZ)) == "after_hours"
    # Closed: 22:00 NY on Sep 1, 2026
    assert get_us_market_session_state(datetime(2026, 9, 1, 22, 0, tzinfo=NY_TZ)) == "closed"


def test_timezone_dst_transitions():
    """Verify America/New_York DST offset handling (EDT UTC-4 vs EST UTC-5)."""
    # Summer (EDT UTC-4) on July 15, 2026
    summer_dt = datetime(2026, 7, 15, 12, 0, tzinfo=NY_TZ)
    assert summer_dt.utcoffset().total_seconds() == -4 * 3600

    # Winter (EST UTC-5) on December 15, 2026
    winter_dt = datetime(2026, 12, 15, 12, 0, tzinfo=NY_TZ)
    assert winter_dt.utcoffset().total_seconds() == -5 * 3600
