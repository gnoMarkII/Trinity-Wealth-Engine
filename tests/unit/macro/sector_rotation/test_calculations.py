from datetime import date, timedelta
import math

from tools.macro.sector_rotation.domain.calculations import _rotation_history, _transition_events, build_snapshot
from tools.macro.sector_rotation.domain.universe import BENCHMARK, SECTOR_TICKERS
from schemas.sector_rotation_schemas import RotationPoint


def _sessions(count: int) -> list[str]:
    result: list[str] = []
    current = date(2025, 1, 1)
    while len(result) < count:
        if current.weekday() < 5:
            result.append(current.isoformat())
        current += timedelta(days=1)
    return result


def _prices(dates: list[str], sector_last: float, benchmark_last: float):
    benchmark = {day: 100.0 + (benchmark_last - 100.0) * i / (len(dates) - 1) for i, day in enumerate(dates)}
    sectors = {ticker: {day: 100.0 + (sector_last - 100.0) * i / (len(dates) - 1) for i, day in enumerate(dates)} for ticker in SECTOR_TICKERS}
    sectors["XLK"] = {day: 100.0 + (sector_last - 100.0) * i / (len(dates) - 1) for i, day in enumerate(dates)}
    return {**sectors, BENCHMARK: benchmark}


def test_absolute_excess_and_relative_returns_are_distinct():
    dates = _sessions(6)
    snapshot = build_snapshot(_prices(dates, 110.0, 105.0))
    row = next(item for item in snapshot.rows if item.ticker == "XLK")
    assert row.returns_pct["1W_absolute_pct"] == 10.0
    assert row.returns_pct["1W_excess_pp"] == 5.0
    assert math.isclose(row.returns_pct["1W_relative_pct"], 4.761905, abs_tol=1e-6)


def test_negative_absolute_return_can_have_positive_relative_return():
    dates = _sessions(6)
    snapshot = build_snapshot(_prices(dates, 95.0, 90.0))
    row = next(item for item in snapshot.rows if item.ticker == "XLK")
    assert row.returns_pct["1W_absolute_pct"] == -5.0
    assert row.returns_pct["1W_excess_pp"] == 5.0
    assert math.isclose(row.returns_pct["1W_relative_pct"], 5.555556, abs_tol=1e-6)


def test_return_endpoints_follow_expected_sessions_and_report_missing_coverage():
    sessions = ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08", "2025-01-09", "2025-01-10"]
    prices = _prices(sessions, 110.0, 105.0)
    prices["XLK"].pop("2025-01-06")
    snapshot = build_snapshot(prices, expected_sessions=sessions)
    metric = next(row for row in snapshot.rows if row.ticker == "XLK").return_metrics["1W"]

    assert metric.start_date == "2025-01-03"
    assert metric.end_date == "2025-01-10"
    assert metric.expected_sessions == 6
    assert metric.valid_sessions == 5
    assert metric.status == "partial"
    assert metric.freshness == "fresh"


def test_rotation_warmup_boundaries_match_formula():
    weekly_dates = [f"2025-01-{3 + 7 * i:02d}" for i in range(27)]
    weekly_sector = {day: 100.0 + i + math.sin(i * 0.7) for i, day in enumerate(weekly_dates)}
    weekly_benchmark = {day: 100.0 for day in weekly_dates}
    assert _rotation_history(dict(list(weekly_sector.items())[:26]), dict(list(weekly_benchmark.items())[:26]), "weekly") == []
    assert len(_rotation_history(weekly_sector, weekly_benchmark, "weekly")) == 1

    daily_dates = _sessions(91)
    daily_sector = {day: 100.0 + i + math.sin(i * 0.7) for i, day in enumerate(daily_dates)}
    daily_benchmark = {day: 100.0 for day in daily_dates}
    assert _rotation_history(dict(list(daily_sector.items())[:90]), dict(list(daily_benchmark.items())[:90]), "daily") == []
    assert len(_rotation_history(daily_sector, daily_benchmark, "daily")) == 1


def test_constant_relative_ratio_is_unavailable_instead_of_neutral():
    dates = _sessions(100)
    benchmark = {day: 100.0 + index for index, day in enumerate(dates)}
    prices = {ticker: {day: value * 0.5 for day, value in benchmark.items()} for ticker in SECTOR_TICKERS}
    snapshot = build_snapshot({**prices, BENCHMARK: benchmark})
    row = next(item for item in snapshot.rows if item.ticker == "XLK")
    assert row.quadrant is None
    assert row.history["daily"][-1].status == "unavailable"
    assert row.history["daily"][-1].reason == "degenerate_variance"


def test_missing_sector_is_retained_and_snapshot_identity_ignores_view_tail():
    dates = _sessions(6)
    prices = _prices(dates, 105.0, 103.0)
    prices["XLE"] = {}
    snapshot = build_snapshot(prices)
    row = next(item for item in snapshot.rows if item.ticker == "XLE")
    assert row.status == "unavailable"
    assert row.reason == "price_history_unavailable"
    assert len(snapshot.rows) == 11
    # Tail/timeframe are presentation inputs and aren't passed to the identity calculation.
    second = build_snapshot(prices)
    assert snapshot.snapshot_id == second.snapshot_id


def test_snapshot_identity_tracks_expected_session_grid_but_ignores_provider_error_text():
    sessions = ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"]
    prices = _prices(sessions, 105.0, 103.0)
    first = build_snapshot(prices, reasons={"XLK": "provider_error:Timeout"}, expected_sessions=sessions)
    same_facts = build_snapshot(prices, reasons={"XLK": "provider_error:ConnectionError"}, expected_sessions=sessions)
    changed_grid = build_snapshot(prices, expected_sessions=["2025-01-02", "2025-01-06", "2025-01-07"])

    assert first.snapshot_id == same_facts.snapshot_id
    assert first.model_dump(mode="json") == same_facts.model_dump(mode="json")
    assert first.snapshot_id != changed_grid.snapshot_id


def test_quadrant_transition_requires_two_valid_bars_and_does_not_cross_gaps():
    def point(day, quadrant, *, status="available"):
        return RotationPoint(as_of=day, quadrant=quadrant, status=status,
                             reason="missing_price_observation" if status == "unavailable" else None)

    confirmed = _transition_events("XLK", "weekly", [
        point("2025-01-03", "Lagging"),
        point("2025-01-10", "Improving"),
        point("2025-01-17", "Improving"),
    ])
    reverted = _transition_events("XLK", "weekly", [
        point("2025-01-03", "Lagging"),
        point("2025-01-10", "Improving"),
        point("2025-01-17", "Lagging"),
    ])
    after_gap = _transition_events("XLK", "weekly", [
        point("2025-01-03", "Lagging"),
        point("2025-01-10", "Improving", status="unavailable"),
        point("2025-01-17", "Improving"),
        point("2025-01-24", "Improving"),
    ])

    assert [(event.event_type, event.changed_at, event.confirmed_at) for event in confirmed] == [
        ("transition", "2025-01-10", None),
        ("confirmed_transition", "2025-01-10", "2025-01-17"),
    ]
    assert [event.event_type for event in reverted] == ["transition"]
    assert after_gap == []
