"""Institutional Market Calendar & Session Resolver for US Equities (NYSE/Nasdaq).

Uses America/New_York timezone with full Daylight Saving Time (EDT/EST) support,
holiday observation rules, and early close schedules (e.g. 13:00 NY close on Black Friday/Christmas Eve).
"""
from datetime import date, datetime, time, timedelta
from typing import Optional, Tuple, Union
from zoneinfo import ZoneInfo

NY_TZ = ZoneInfo("America/New_York")


def _easter_date(year: int) -> date:
    """Computes Western Easter Sunday using the Anonymous Gregorian algorithm."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _good_friday(year: int) -> date:
    """Returns Good Friday for the given year."""
    return _easter_date(year) - timedelta(days=2)


def _nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> date:
    """Finds the nth occurrence of a weekday in a month (1-indexed n, weekday: Monday=0)."""
    first_day = date(year, month, 1)
    day_offset = (weekday - first_day.weekday()) % 7
    return first_day + timedelta(days=day_offset + (n - 1) * 7)


def _last_weekday_of_month(year: int, month: int, weekday: int) -> date:
    """Finds the last occurrence of a weekday in a month."""
    if month == 12:
        last_day = date(year, 12, 31)
    else:
        last_day = date(year, month + 1, 1) - timedelta(days=1)
    day_offset = (last_day.weekday() - weekday) % 7
    return last_day - timedelta(days=day_offset)


def get_us_market_holidays(year: int) -> set[date]:
    """Returns the set of full-day NYSE/Nasdaq market holidays for a given year."""
    holidays: set[date] = set()

    # 1. New Year's Day (Jan 1, or observed Jan 2 if Sunday)
    nyd = date(year, 1, 1)
    if nyd.weekday() == 6:  # Sunday -> Monday
        holidays.add(date(year, 1, 2))
    elif nyd.weekday() < 5:  # Monday-Friday
        holidays.add(nyd)

    # 2. Martin Luther King Jr. Day (3rd Monday in January)
    holidays.add(_nth_weekday_of_month(year, 1, 0, 3))

    # 3. Washington's Birthday / Presidents' Day (3rd Monday in February)
    holidays.add(_nth_weekday_of_month(year, 2, 0, 3))

    # 4. Good Friday (Friday before Easter)
    holidays.add(_good_friday(year))

    # 5. Memorial Day (Last Monday in May)
    holidays.add(_last_weekday_of_month(year, 5, 0))

    # 6. Juneteenth National Independence Day (June 19, observed)
    june19 = date(year, 6, 19)
    if june19.weekday() == 6:  # Sunday -> Monday
        holidays.add(date(year, 6, 20))
    elif june19.weekday() == 5:  # Saturday -> Friday
        holidays.add(date(year, 6, 18))
    else:
        holidays.add(june19)

    # 7. Independence Day (July 4, observed)
    july4 = date(year, 7, 4)
    if july4.weekday() == 6:  # Sunday -> Monday
        holidays.add(date(year, 7, 5))
    elif july4.weekday() == 5:  # Saturday -> Friday
        holidays.add(date(year, 7, 3))
    else:
        holidays.add(july4)

    # 8. Labor Day (1st Monday in September)
    holidays.add(_nth_weekday_of_month(year, 9, 0, 1))

    # 9. Thanksgiving Day (4th Thursday in November)
    holidays.add(_nth_weekday_of_month(year, 11, 3, 4))

    # 10. Christmas Day (Dec 25, observed)
    xmas = date(year, 12, 25)
    if xmas.weekday() == 6:  # Sunday -> Monday
        holidays.add(date(year, 12, 26))
    elif xmas.weekday() == 5:  # Saturday -> Friday
        holidays.add(date(year, 12, 24))
    else:
        holidays.add(xmas)

    return holidays


def get_us_early_close_days(year: int) -> set[date]:
    """Returns the set of scheduled 13:00 NY Early Close days for a given year."""
    early_closes: set[date] = set()

    # 1. Day after Thanksgiving (Black Friday - 4th Friday in November)
    thanksgiving = _nth_weekday_of_month(year, 11, 3, 4)
    black_friday = thanksgiving + timedelta(days=1)
    early_closes.add(black_friday)

    # 2. July 3rd (Day before July 4, if weekday and July 4 is weekday other than Monday)
    july4 = date(year, 7, 4)
    if july4.weekday() in (1, 2, 3, 4):  # Tue, Wed, Thu, Fri
        early_closes.add(date(year, 7, 3))

    # 3. Christmas Eve (Dec 24, if weekday and Dec 25 is weekday other than Monday)
    dec24 = date(year, 12, 24)
    if dec24.weekday() in (0, 1, 2, 3):  # Mon, Tue, Wed, Thu
        early_closes.add(dec24)

    return early_closes


def is_us_trading_day(dt_or_date: Union[datetime, date]) -> bool:
    """Checks if the given date is a valid US equity trading day (Mon-Fri, non-holiday)."""
    d = dt_or_date.date() if isinstance(dt_or_date, datetime) else dt_or_date
    if d.weekday() >= 5:  # Saturday or Sunday
        return False
    holidays = get_us_market_holidays(d.year)
    return d not in holidays


def get_nyse_close_time(d: date) -> time:
    """Returns the scheduled market close time for NYSE on the given date (13:00 on early close, 16:00 regular)."""
    early_closes = get_us_early_close_days(d.year)
    if d in early_closes:
        return time(13, 0)
    return time(16, 0)


def get_last_completed_regular_session(as_of: Optional[datetime] = None) -> date:
    """Determines the exact calendar date of the last completed regular trading session.

    Takes current America/New_York local time and checks against the exact market close time
    for today (13:00 for early close, 16:00 for regular session).
    If market has not closed yet today, rolls backward to the preceding trading day.
    """
    if as_of is None:
        ny_dt = datetime.now(NY_TZ)
    elif as_of.tzinfo is not None:
        ny_dt = as_of.astimezone(NY_TZ)
    else:
        # Assume UTC if naive, convert to NY
        ny_dt = as_of.replace(tzinfo=ZoneInfo("UTC")).astimezone(NY_TZ)

    candidate_date = ny_dt.date()

    if is_us_trading_day(candidate_date):
        close_t = get_nyse_close_time(candidate_date)
        # If current NY time is before the scheduled close time, today's session is not complete
        if ny_dt.time() < close_t:
            candidate_date -= timedelta(days=1)
    else:
        candidate_date -= timedelta(days=1)

    while not is_us_trading_day(candidate_date):
        candidate_date -= timedelta(days=1)

    return candidate_date


def get_us_market_session_state(dt: Optional[datetime] = None) -> str:
    """Returns current market session state: 'pre_market', 'regular_open', 'after_hours', or 'closed'."""
    if dt is None:
        ny_dt = datetime.now(NY_TZ)
    elif dt.tzinfo is not None:
        ny_dt = dt.astimezone(NY_TZ)
    else:
        ny_dt = dt.replace(tzinfo=ZoneInfo("UTC")).astimezone(NY_TZ)

    d = ny_dt.date()
    t = ny_dt.time()

    if not is_us_trading_day(d):
        return "closed"

    close_t = get_nyse_close_time(d)
    # Pre-market: 04:00 to 09:30
    if time(4, 0) <= t < time(9, 30):
        return "pre_market"
    # Regular trading session: 09:30 to scheduled close (13:00 or 16:00)
    elif time(9, 30) <= t < close_t:
        return "regular_open"
    # After-hours: from close to 20:00 (or 13:00 to 17:00 on early close)
    elif close_t <= t < time(20, 0):
        return "after_hours"
    else:
        return "closed"


def count_business_days_between(d1: date, d2: date) -> int:
    """Counts trading days between d1 and d2 (exclusive of start, inclusive of end)."""
    if d1 > d2:
        d1, d2 = d2, d1
    cur = d1 + timedelta(days=1)
    count = 0
    while cur <= d2:
        if is_us_trading_day(cur):
            count += 1
        cur += timedelta(days=1)
    return count


def get_us_market_session_info(now_dt: datetime, latest_ohlcv_date_str: Optional[str]) -> dict:
    """Computes session status, expected target EOD, and freshness relative to latest available bar."""
    expected_eod = get_last_completed_regular_session(now_dt)
    expected_eod_str = expected_eod.strftime("%Y-%m-%d")
    market_session_status = get_us_market_session_state(now_dt)
    if market_session_status == "regular_open":
        market_session_status = "open"

    if not latest_ohlcv_date_str:
        return {
            "market_session_status": market_session_status,
            "data_freshness_status": "unavailable",
            "expected_latest_session_date": expected_eod_str,
            "actual_latest_session_date": None,
            "missing_trading_sessions": 999,
        }

    try:
        actual_date = datetime.strptime(latest_ohlcv_date_str[:10], "%Y-%m-%d").date()
    except ValueError:
        return {
            "market_session_status": market_session_status,
            "data_freshness_status": "unknown",
            "expected_latest_session_date": expected_eod_str,
            "actual_latest_session_date": latest_ohlcv_date_str,
            "missing_trading_sessions": 999,
        }

    missing_sessions = count_business_days_between(actual_date, expected_eod)
    if missing_sessions == 0:
        data_freshness_status = "fresh"
    elif missing_sessions == 1:
        data_freshness_status = "stale_one_session"
    else:
        data_freshness_status = "stale_multiple_sessions"

    return {
        "market_session_status": market_session_status,
        "data_freshness_status": data_freshness_status,
        "expected_latest_session_date": expected_eod_str,
        "actual_latest_session_date": latest_ohlcv_date_str,
        "missing_trading_sessions": missing_sessions,
    }
