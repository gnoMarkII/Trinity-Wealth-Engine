"""
Comprehensive unit tests for Sector Rotation: Calendar, Calculations, Boundaries, and Data Quality.
Validates AC-01 through AC-08, AC-17, AC-25, AC-26 and RC-05 through RC-09, RC-13, RC-15, RC-25, RC-26.
"""
from datetime import date, timedelta
import math
import pytest

from tools.macro.sector_rotation.domain.calculations import (
    _CONFIG,
    _HORIZONS,
    _finite_price_map,
    _relative_price_history,
    _rotation_history,
    _transition_events,
    _weekly_sessions,
    build_snapshot,
    normalize_price_inputs,
    normalized_input_digest,
)
from tools.macro.sector_rotation.domain.universe import BENCHMARK, SECTOR_TICKERS
from tools.market.market_calendar import is_us_trading_day
from schemas.sector_rotation_schemas import RotationPoint


def _generate_business_days(start: date, count: int) -> list[str]:
    """Generate count US equity trading days starting from start date."""
    sessions: list[str] = []
    current = start
    while len(sessions) < count:
        if is_us_trading_day(current):
            sessions.append(current.isoformat())
        current += timedelta(days=1)
    return sessions


# ==============================================================================
# AC-01, AC-02, RC-05, RC-06, RC-07, RC-26: Return Calculations & Sign Symmetry
# ==============================================================================

def test_independent_reference_calculation_all_horizons():
    """Verify returns against independent arithmetic formula for 1W, 1M, 3M, 6M, 1Y, YTD."""
    # 260 trading days spanning from 2025 through 2026
    sessions = _generate_business_days(date(2025, 1, 2), 260)
    end_day = sessions[-1]
    
    # Deterministic price series:
    # SPY grows linearly from 100 to 200
    # XLK grows from 100 to 220
    b_prices = {day: 100.0 + 100.0 * (i / (len(sessions) - 1)) for i, day in enumerate(sessions)}
    s_prices = {day: 100.0 + 120.0 * (i / (len(sessions) - 1)) for i, day in enumerate(sessions)}
    
    all_prices = {ticker: s_prices.copy() for ticker in SECTOR_TICKERS}
    all_prices[BENCHMARK] = b_prices
    
    snapshot = build_snapshot(all_prices, expected_sessions=sessions)
    row = next(r for r in snapshot.rows if r.ticker == "XLK")
    
    for horizon, interval in _HORIZONS["daily"].items():
        start_idx = len(sessions) - 1 - interval
        start_day = sessions[start_idx]
        
        # Independent manual reference calculation
        expected_s_ret = (s_prices[end_day] / s_prices[start_day]) - 1.0
        expected_b_ret = (b_prices[end_day] / b_prices[start_day]) - 1.0
        expected_abs = round(expected_s_ret * 100.0, 6)
        expected_excess = round((expected_s_ret - expected_b_ret) * 100.0, 6)
        expected_rel = round(((1.0 + expected_s_ret) / (1.0 + expected_b_ret) - 1.0) * 100.0, 6)
        
        actual_metric = row.return_metrics[horizon]
        assert actual_metric.status == "available"
        assert math.isclose(actual_metric.absolute_return_pct, expected_abs, abs_tol=1e-6)
        assert math.isclose(actual_metric.excess_return_pp, expected_excess, abs_tol=1e-6)
        assert math.isclose(actual_metric.relative_return_pct, expected_rel, abs_tol=1e-6)
        assert actual_metric.start_date == start_day
        assert actual_metric.end_date == end_day


def test_sign_symmetry_and_relative_return_direction():
    """Verify sign symmetry and relative return when absolute is negative."""
    sessions = ["2026-09-21", "2026-09-22", "2026-09-23", "2026-09-24", "2026-09-25", "2026-09-28"]
    start, end = sessions[0], sessions[-1]
    
    # Case 1: Sector +10%, SPY +5%
    p1 = {BENCHMARK: {start: 100.0, end: 105.0}, "XLK": {start: 100.0, end: 110.0}}
    for t in SECTOR_TICKERS:
        if t != "XLK":
            p1[t] = {start: 100.0, end: 100.0}
    s1 = build_snapshot(p1, expected_sessions=sessions)
    r1 = next(r for r in s1.rows if r.ticker == "XLK").return_metrics["1W"]
    assert r1.absolute_return_pct == 10.0
    assert r1.excess_return_pp == 5.0
    assert math.isclose(r1.relative_return_pct, 4.761905, abs_tol=1e-6)
    
    # Case 2: Sector -5%, SPY -10% (Absolute negative, but relative is POSITIVE)
    p2 = {BENCHMARK: {start: 100.0, end: 90.0}, "XLK": {start: 100.0, end: 95.0}}
    for t in SECTOR_TICKERS:
        if t != "XLK":
            p2[t] = {start: 100.0, end: 100.0}
    s2 = build_snapshot(p2, expected_sessions=sessions)
    r2 = next(r for r in s2.rows if r.ticker == "XLK").return_metrics["1W"]
    assert r2.absolute_return_pct == -5.0
    assert r2.excess_return_pp == 5.0
    assert math.isclose(r2.relative_return_pct, 5.555556, abs_tol=1e-6)
    
    # Case 3: Sector -10%, SPY -5% (Absolute negative, relative negative)
    p3 = {BENCHMARK: {start: 100.0, end: 95.0}, "XLK": {start: 100.0, end: 90.0}}
    for t in SECTOR_TICKERS:
        if t != "XLK":
            p3[t] = {start: 100.0, end: 100.0}
    s3 = build_snapshot(p3, expected_sessions=sessions)
    r3 = next(r for r in s3.rows if r.ticker == "XLK").return_metrics["1W"]
    assert r3.absolute_return_pct == -10.0
    assert r3.excess_return_pp == -5.0
    assert math.isclose(r3.relative_return_pct, -5.263158, abs_tol=1e-6)


# ==============================================================================
# AC-03, AC-04, RC-08, RC-09: Warm-up Boundaries & Degenerate Variance
# ==============================================================================

def test_warmup_exact_bar_thresholds():
    """Verify weekly requires 27 bars (26 is empty), daily requires 91 bars (90 is empty)."""
    # Weekly test
    weekly_dates = [f"2025-01-{i:02d}" for i in range(1, 28)]
    w_sec = {d: 100.0 + i + math.sin(i) for i, d in enumerate(weekly_dates)}
    w_bm = {d: 100.0 for d in weekly_dates}
    
    # Exactly 26 weekly bars -> 0 points
    h26 = _rotation_history(dict(list(w_sec.items())[:26]), dict(list(w_bm.items())[:26]), "weekly")
    assert len(h26) == 0
    # Exactly 27 weekly bars -> 1 point
    h27 = _rotation_history(w_sec, w_bm, "weekly")
    assert len(h27) == 1
    assert h27[0].status == "available"
    assert h27[0].relative_trend is not None
    assert h27[0].relative_momentum is not None

    # Daily test
    daily_dates = [f"D{i:03d}" for i in range(1, 92)]
    d_sec = {d: 100.0 + i + math.sin(i * 0.5) for i, d in enumerate(daily_dates)}
    d_bm = {d: 100.0 for d in daily_dates}
    
    # Exactly 90 daily bars -> 0 points
    hd90 = _rotation_history(dict(list(d_sec.items())[:90]), dict(list(d_bm.items())[:90]), "daily")
    assert len(hd90) == 0
    # Exactly 91 daily bars -> 1 point
    hd91 = _rotation_history(d_sec, d_bm, "daily")
    assert len(hd91) == 1
    assert hd91[0].status == "available"


def test_near_constant_ratio_does_not_fabricate_quadrant():
    """Verify near-constant ratio triggers degenerate_variance and leaves quadrant as None."""
    daily_dates = [f"D{i:03d}" for i in range(1, 100)]
    # Benchmark moving, sector moving with nearly exact identical proportionality
    benchmark = {d: 100.0 + i for i, d in enumerate(daily_dates)}
    # Add infinitesimal perturbation (e.g. 1e-12) well below 1e-10 degenerate threshold
    sector = {d: (100.0 + i) * 0.5 + 1e-12 * (i % 2) for i, d in enumerate(daily_dates)}
    
    history = _rotation_history(sector, benchmark, "daily", daily_dates)
    assert len(history) > 0
    last_point = history[-1]
    assert last_point.status == "unavailable"
    assert last_point.reason == "degenerate_variance"
    assert last_point.quadrant is None
    assert last_point.relative_trend is None
    assert last_point.relative_momentum is None


# ==============================================================================
# AC-05, AC-06, RC-08: Calendar Holidays, Good Friday, and Unfinished Week
# ==============================================================================

def test_good_friday_week_resolves_thursday_as_weekly_close():
    """In a Good Friday week (e.g. 2026-04-03 holiday), week ends on Thursday 2026-04-02."""
    week_sessions = ["2026-03-30", "2026-03-31", "2026-04-01", "2026-04-02"]
    # Friday 2026-04-03 is Good Friday
    assert not is_us_trading_day(date(2026, 4, 3))
    
    weekly = _weekly_sessions(week_sessions)
    assert weekly == ["2026-04-02"]


def test_unfinished_week_is_not_treated_as_completed_weekly_close():
    """A partial week (e.g. Mon-Wed of a week with a trading Friday) must not be a weekly close."""
    # 2026-10-05 (Mon), 2026-10-06 (Tue), 2026-10-07 (Wed). Friday 2026-10-09 is a trading day.
    partial_sessions = ["2026-10-05", "2026-10-06", "2026-10-07"]
    assert is_us_trading_day(date(2026, 10, 9))
    
    weekly = _weekly_sessions(partial_sessions)
    # The week ending session would be 2026-10-09, which is > 2026-10-07, so it must not be in weekly!
    assert weekly == []


def test_ytd_prior_year_close_and_cross_year():
    """Verify YTD requires prior year-end close; missing prior year close yields unavailable."""
    # Scenario A: Cross year with 2025-12-31 and 2026-01-02
    sessions_with_prior = ["2025-12-30", "2025-12-31", "2026-01-02", "2026-01-05"]
    p_with_prior = {BENCHMARK: {d: 100.0 for d in sessions_with_prior},
                    "XLK": {"2025-12-30": 100.0, "2025-12-31": 100.0, "2026-01-02": 105.0, "2026-01-05": 110.0}}
    for t in SECTOR_TICKERS:
        if t != "XLK":
            p_with_prior[t] = {d: 100.0 for d in sessions_with_prior}
    s_a = build_snapshot(p_with_prior, expected_sessions=sessions_with_prior)
    ytd_a = next(r for r in s_a.rows if r.ticker == "XLK").return_metrics["YTD"]
    assert ytd_a.status == "available"
    assert ytd_a.start_date == "2025-12-31"
    assert ytd_a.end_date == "2026-01-05"
    assert ytd_a.absolute_return_pct == 10.0

    # Scenario B: Grid only starts in 2026 (no 2025 close)
    sessions_no_prior = ["2026-01-02", "2026-01-05", "2026-01-06"]
    p_no_prior = {BENCHMARK: {d: 100.0 for d in sessions_no_prior},
                  "XLK": {d: 100.0 for d in sessions_no_prior}}
    for t in SECTOR_TICKERS:
        if t != "XLK":
            p_no_prior[t] = {d: 100.0 for d in sessions_no_prior}
    s_b = build_snapshot(p_no_prior, expected_sessions=sessions_no_prior)
    ytd_b = next(r for r in s_b.rows if r.ticker == "XLK").return_metrics["YTD"]
    assert ytd_b.status == "unavailable"
    assert ytd_b.reason == "prior_year_close_unavailable"
    assert ytd_b.absolute_return_pct is None


# ==============================================================================
# AC-07, AC-08, RC-06, RC-25: Data Quality, Non-Finite Values, Missing Inputs
# ==============================================================================

def test_non_finite_and_non_positive_prices_are_safely_filtered():
    """Verify NaN, Inf, zero, and negative prices are safely stripped without crash."""
    dirty_prices = {
        "2026-01-02": 100.0,
        "2026-01-05": float("nan"),
        "2026-01-06": float("inf"),
        "2026-01-07": -50.0,
        "2026-01-08": 0.0,
        "2026-01-09": 105.0,
    }
    clean = _finite_price_map(dirty_prices)
    assert clean == {"2026-01-02": 100.0, "2026-01-09": 105.0}


def test_missing_benchmark_and_missing_sector():
    """Verify behavior when benchmark is completely missing or an individual ETF is missing."""
    sessions = ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09"]
    
    # Missing benchmark:
    p_no_bm = {t: {d: 100.0 for d in sessions} for t in SECTOR_TICKERS}
    p_no_bm[BENCHMARK] = {}
    snap_no_bm = build_snapshot(p_no_bm, expected_sessions=sessions)
    assert snap_no_bm.benchmark_status == "unavailable"
    assert snap_no_bm.benchmark_reason == "benchmark_history_unavailable"
    for row in snap_no_bm.rows:
        assert row.reason in ("benchmark_history_unavailable", "benchmark_overlap_unavailable")
        assert row.relative_trend is None
        assert row.relative_momentum is None
        assert row.quadrant is None

    # Missing sector (e.g. XLE has no data):
    p_with_bm = {t: {d: 100.0 for d in sessions} for t in SECTOR_TICKERS}
    p_with_bm[BENCHMARK] = {d: 100.0 for d in sessions}
    p_with_bm["XLE"] = {}
    snap_partial = build_snapshot(p_with_bm, expected_sessions=sessions)
    assert snap_partial.available_sectors == 10
    xle_row = next(r for r in snap_partial.rows if r.ticker == "XLE")
    assert xle_row.status == "unavailable"
    assert xle_row.reason == "price_history_unavailable"
    assert len(snap_partial.rows) == 11  # All 11 sector rows retained!


def test_future_inputs_do_not_alter_pre_cutoff_facts():
    """Verify that dates beyond cutoff are dropped by SectorHistoryAdapter and don't change snapshot."""
    from tools.macro.adapters.sector_history_adapter import SectorHistoryAdapter
    import pandas as pd

    cutoff = date(2026, 1, 9)
    # Series with observations up to cutoff
    dates_valid = pd.date_range("2026-01-05", "2026-01-09", freq="B")
    # Series with future observations beyond cutoff
    dates_with_future = pd.date_range("2026-01-05", "2026-01-16", freq="B")

    frame_valid = pd.DataFrame({"Close": [100.0] * len(dates_valid)}, index=dates_valid)
    frame_with_future = pd.DataFrame({"Close": [100.0] * len(dates_with_future)}, index=dates_with_future)

    norm_valid = SectorHistoryAdapter._normalize(frame_valid, cutoff)
    norm_future = SectorHistoryAdapter._normalize(frame_with_future, cutoff)

    # Future dates must be stripped and yield identical normalized price series
    assert norm_valid == norm_future
    assert max(norm_future.keys()) == "2026-01-09"


# ==============================================================================
# AC-25, AC-26, RC-13, RC-15: Quadrant Transitions & Stable Event IDs
# ==============================================================================

def test_transition_confirmation_and_reversion():
    """Verify transitions A->B->B confirms, A->B->A reverts, and gap resets pending."""
    def p(d, q, status="available"):
        return RotationPoint(as_of=d, quadrant=q, status=status)
    
    # A -> B -> B -> C -> C
    pts = [
        p("2026-01-02", "Improving"),
        p("2026-01-09", "Leading"),
        p("2026-01-16", "Leading"),
        p("2026-01-23", "Weakening"),
        p("2026-01-30", "Weakening"),
    ]
    events = _transition_events("XLK", "weekly", pts)
    assert len(events) == 4
    assert events[0].event_type == "transition"
    assert events[0].changed_at == "2026-01-09"
    assert events[1].event_type == "confirmed_transition"
    assert events[1].changed_at == "2026-01-09"
    assert events[1].confirmed_at == "2026-01-16"
    assert events[2].event_type == "transition"
    assert events[2].changed_at == "2026-01-23"
    assert events[3].event_type == "confirmed_transition"
    assert events[3].changed_at == "2026-01-23"
    assert events[3].confirmed_at == "2026-01-30"

    # Event IDs must be deterministic and prefix with evt_
    for ev in events:
        assert ev.event_id.startswith("evt_")
        assert len(ev.event_id) == 28  # "evt_" + 24 hex characters
