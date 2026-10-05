"""Versioned, provider-independent return and relative-rotation calculations."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from datetime import date, timedelta
from typing import Mapping, Optional, Sequence

from schemas.sector_rotation_schemas import (
    QuadrantTransitionEvent, RelativePricePoint, ReturnMetric, RotationPoint, SectorRow, SectorRotationSnapshot,
)
from .universe import BENCHMARK, SECTOR_TICKERS, UNIVERSE_VERSION, sector_name

SCHEMA_VERSION = "sector-rotation-snapshot-v2"
FORMULA_VERSION = "relative-rotation-v2"
CALENDAR_VERSION = "us-equity-sessions-v1"
TRANSITION_RULE_VERSION = "confirmed-after-two-bars-v1"
_CONFIG = {
    "daily": {"n": 63, "m": 21, "smooth": 5, "tail": 20},
    "weekly": {"n": 14, "m": 10, "smooth": 3, "tail": 12},
    "degenerate_sd_relative_tolerance": 1e-10,
    "rotation_center": 100.0,
}
FORMULA_CONFIG = _CONFIG
_HORIZONS = {"daily": {"1W": 5, "1M": 21, "3M": 63, "6M": 126, "1Y": 252},
             "weekly": {"1W": 1, "1M": 4, "3M": 13, "6M": 26, "1Y": 52}}


def is_current_snapshot(snapshot: SectorRotationSnapshot) -> bool:
    return (
        snapshot.formula_version == FORMULA_VERSION
        and snapshot.calendar_version == CALENDAR_VERSION
        and snapshot.transition_rule_version == TRANSITION_RULE_VERSION
        and snapshot.formula_config == FORMULA_CONFIG
    )


def _finite_price_map(values: Mapping[str, float] | None) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in (values or {}).items():
        try:
            parsed = float(value)
            normalized_date = date.fromisoformat(str(key)[:10]).isoformat()
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed) and parsed > 0:
            result[normalized_date] = round(parsed, 8)
    return dict(sorted(result.items()))


def normalized_input_digest(prices: Mapping[str, Mapping[str, float] | None]) -> str:
    canonical = normalize_price_inputs(prices)
    raw = json.dumps(canonical, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def normalize_price_inputs(prices: Mapping[str, Mapping[str, float] | None]) -> dict[str, dict[str, float]]:
    return {ticker: _finite_price_map(prices.get(ticker)) for ticker in sorted([*SECTOR_TICKERS, BENCHMARK])}


def _sample_z(values: Sequence[Optional[float]], window: int) -> list[Optional[float]]:
    out: list[Optional[float]] = [None] * len(values)
    for i in range(window - 1, len(values)):
        sample = values[i - window + 1:i + 1]
        if any(value is None or not math.isfinite(value) for value in sample):
            continue
        mean = statistics.fmean(sample)
        sd = statistics.stdev(sample) if window > 1 else 0.0
        if sd <= 1e-10 * max(abs(mean), 1e-300):
            continue
        out[i] = (values[i] - mean) / sd
    return out


def _sma_with_gaps(values: Sequence[Optional[float]], window: int) -> list[Optional[float]]:
    out: list[Optional[float]] = [None] * len(values)
    for i in range(window - 1, len(values)):
        sample = values[i - window + 1:i + 1]
        if all(value is not None and math.isfinite(value) for value in sample):
            out[i] = statistics.fmean(value for value in sample if value is not None)
    return out


def _quadrant(trend: float, momentum: float) -> str:
    if trend >= 100 and momentum >= 100:
        return "Leading"
    if trend >= 100 and momentum < 100:
        return "Weakening"
    if trend < 100 and momentum < 100:
        return "Lagging"
    return "Improving"


def _semantic_transition_id(
    ticker: str, timeframe: str, previous: RotationPoint, changed: RotationPoint,
    confirmed: Optional[RotationPoint], event_type: str,
) -> str:
    semantic = {
        "universe_version": UNIVERSE_VERSION,
        "ticker": ticker,
        "timeframe": timeframe,
        "rule_version": TRANSITION_RULE_VERSION,
        "event_type": event_type,
        "previous_valid_at": previous.as_of,
        "changed_at": changed.as_of,
        "confirmed_at": confirmed.as_of if confirmed else None,
        "from": previous.quadrant,
        "to": changed.quadrant,
    }
    raw = json.dumps(semantic, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return "evt_" + hashlib.sha256(raw).hexdigest()[:24]


def _weekly_sessions(daily_sessions: Sequence[str]) -> list[str]:
    """Return completed week-ending US sessions represented by a daily grid."""
    from tools.market.market_calendar import is_us_trading_day

    if not daily_sessions:
        return []
    first = date.fromisoformat(daily_sessions[0])
    last = date.fromisoformat(daily_sessions[-1])
    result: list[str] = []
    week_monday = first - timedelta(days=first.weekday())
    while week_monday <= last:
        friday = week_monday + timedelta(days=4)
        close = friday
        while not is_us_trading_day(close):
            close -= timedelta(days=1)
        if first <= close <= last:
            result.append(close.isoformat())
        week_monday += timedelta(days=7)
    return result


def _relative_price_history(
    sector: Mapping[str, float],
    benchmark: Mapping[str, float],
    timeframe: str,
    sessions: Optional[Sequence[str]] = None,
) -> list[RelativePricePoint]:
    grid = list(sessions or sorted(set(sector) | set(benchmark)))
    if timeframe == "weekly":
        weekly_grid = _weekly_sessions(grid)
        sector = _to_weekly(sector, weekly_grid)
        benchmark = _to_weekly(benchmark, weekly_grid)
        grid = weekly_grid
    first_day = next((day for day in grid if day in sector and day in benchmark), None)
    if first_day is None:
        return [RelativePricePoint(as_of=day, status="unavailable", reason="missing_price_observation") for day in grid]
    first_ratio = sector[first_day] / benchmark[first_day]
    result = []
    for day in grid:
        if day not in sector or day not in benchmark:
            result.append(RelativePricePoint(as_of=day, status="unavailable", reason="missing_price_observation"))
        else:
            result.append(RelativePricePoint(
                as_of=day,
                sector_spy_rebased_100=round((sector[day] / benchmark[day]) / first_ratio * 100, 6),
            ))
    return result


def _rotation_history(
    sector: Mapping[str, float],
    benchmark: Mapping[str, float],
    timeframe: str,
    sessions: Optional[Sequence[str]] = None,
) -> list[RotationPoint]:
    grid = list(sessions or sorted(set(sector) | set(benchmark)))
    if not grid:
        return []
    ratio = [100.0 * sector[day] / benchmark[day] if day in sector and day in benchmark else None for day in grid]
    cfg = _CONFIG[timeframe]
    minimum = cfg["n"] + cfg["m"] + 2 * cfg["smooth"] - 3
    trend_z = _sample_z(ratio, cfg["n"])
    trend_input = [100.0 + value if value is not None else None for value in trend_z]
    trend = _sma_with_gaps(trend_input, cfg["smooth"])
    momentum_z = _sample_z(trend, cfg["m"])
    momentum_input = [100.0 + value if value is not None else None for value in momentum_z]
    momentum = _sma_with_gaps(momentum_input, cfg["smooth"])
    points: list[RotationPoint] = []
    for i, session in enumerate(grid):
        x, y = trend[i], momentum[i]
        if x is None or y is None:
            if i + 1 >= minimum:
                if ratio[i] is None:
                    reason = "missing_price_observation"
                elif any(value is None for value in ratio[max(0, i - minimum + 1):i + 1]):
                    reason = "incomplete_rolling_window"
                else:
                    reason = "degenerate_variance"
                points.append(RotationPoint(as_of=session, status="unavailable", reason=reason))
            continue
        points.append(RotationPoint(
            as_of=session,
            relative_trend=round(x, 6),
            relative_momentum=round(y, 6),
            quadrant=_quadrant(x, y),
        ))
    return points


def _transition_events(
    ticker: str, timeframe: str, points: Sequence[RotationPoint],
) -> list[QuadrantTransitionEvent]:
    """Emit a transition only after two consecutive valid bars confirm it."""
    transitions: list[QuadrantTransitionEvent] = []
    previous: Optional[RotationPoint] = None
    pending: Optional[RotationPoint] = None
    for point in points:
        if point.status != "available" or point.quadrant is None:
            previous = None
            pending = None
            continue
        if previous is None:
            previous = point
            continue
        if point.quadrant == previous.quadrant:
            pending = None
            previous = point
            continue
        if pending is None or pending.quadrant != point.quadrant:
            pending = point
            transitions.append(QuadrantTransitionEvent(
                event_id=_semantic_transition_id(ticker, timeframe, previous, point, None, "transition"),
                timeframe=timeframe,
                previous_valid_at=previous.as_of,
                changed_at=point.as_of,
                from_quadrant=previous.quadrant,
                to_quadrant=point.quadrant,
                event_type="transition",
            ))
            continue
        transitions.append(QuadrantTransitionEvent(
            event_id=_semantic_transition_id(ticker, timeframe, previous, pending, point, "confirmed_transition"),
            timeframe=timeframe,
            previous_valid_at=previous.as_of,
            changed_at=pending.as_of,
            confirmed_at=point.as_of,
            from_quadrant=previous.quadrant,
            to_quadrant=point.quadrant,
        ))
        previous = point
        pending = None
    return transitions


def _return_metrics(
    sector: Mapping[str, float], benchmark: Optional[Mapping[str, float]], sessions: Sequence[str],
) -> tuple[dict[str, Optional[float]], dict[str, ReturnMetric]]:
    grid = list(sessions)
    labels = {**_HORIZONS["daily"], "YTD": None}
    values: dict[str, Optional[float]] = {}
    facts: dict[str, ReturnMetric] = {}
    end_index = next((i for i in range(len(grid) - 1, -1, -1)
                      if grid[i] in sector and (benchmark is None or grid[i] in benchmark)), None)
    for label, intervals in labels.items():
        if end_index is None:
            start_index = None
            reason = "price_history_unavailable"
        elif label == "YTD":
            end_year = date.fromisoformat(grid[end_index]).year
            prior_year_sessions = [i for i, session in enumerate(grid) if date.fromisoformat(session).year < end_year]
            start_index = prior_year_sessions[-1] if prior_year_sessions else None
            reason = "prior_year_close_unavailable" if start_index is None else None
        else:
            start_index = end_index - int(intervals)
            reason = "insufficient_history" if start_index < 0 else None
        expected = 0 if start_index is None or end_index is None else end_index - start_index + 1
        start_day = grid[start_index] if start_index is not None and start_index >= 0 else None
        end_day = grid[end_index] if end_index is not None else None
        start_valid = start_day is not None and start_day in sector and (benchmark is None or start_day in benchmark)
        end_valid = end_day is not None and end_day in sector and (benchmark is None or end_day in benchmark)
        if reason is None and not start_valid:
            reason = "missing_start_endpoint"
        if reason is None and not end_valid:
            reason = "missing_end_endpoint"
        valid = sum(
            1 for day in grid[start_index:end_index + 1]
            if day in sector and (benchmark is None or day in benchmark)
        ) if start_index is not None and end_index is not None and start_index >= 0 else 0
        absolute = excess = relative = None
        if reason is None and start_day and end_day:
            sector_return = sector[end_day] / sector[start_day] - 1.0
            absolute = round(sector_return * 100, 6)
            if benchmark is not None:
                benchmark_return = benchmark[end_day] / benchmark[start_day] - 1.0
                excess = round((sector_return - benchmark_return) * 100, 6)
                relative = round(((1.0 + sector_return) / (1.0 + benchmark_return) - 1.0) * 100, 6) if benchmark_return > -1 else None
        status = "unavailable" if absolute is None else "partial" if valid < expected else "available"
        if reason is None and status == "partial":
            reason = "missing_observations"
        freshness = "unknown" if end_day is None else "fresh" if end_day == grid[-1] else "stale"
        values[f"{label}_absolute_pct"] = absolute
        values[f"{label}_excess_pp"] = excess
        values[f"{label}_relative_pct"] = relative
        facts[label] = ReturnMetric(
            absolute_return_pct=absolute,
            excess_return_pp=excess,
            relative_return_pct=relative,
            start_date=start_day,
            end_date=end_day,
            expected_sessions=expected,
            valid_sessions=valid,
            status=status,
            freshness=freshness,
            reason=reason,
        )
    return values, facts


def _returns(
    sector: Mapping[str, float], benchmark: Mapping[str, float], sessions: Optional[Sequence[str]] = None,
) -> dict[str, Optional[float]]:
    grid = list(sessions or sorted(set(sector) | set(benchmark)))
    return _return_metrics(sector, benchmark, grid)[0]


def _absolute_returns(series: Mapping[str, float], sessions: Optional[Sequence[str]] = None) -> dict[str, Optional[float]]:
    grid = list(sessions or sorted(series))
    return _return_metrics(series, None, grid)[0]


def build_snapshot(prices: Mapping[str, Mapping[str, float] | None], *, reasons: Mapping[str, str] | None = None, expected_sessions: Optional[Sequence[str]] = None) -> SectorRotationSnapshot:
    """Build a stable snapshot from adjusted closes; dates must already be cutoff-filtered."""
    clean = normalize_price_inputs(prices)
    digest = normalized_input_digest(clean)
    benchmark = clean[BENCHMARK]
    common_as_of = max(benchmark, default=None)
    input_start = min(benchmark, default=None)
    observed_sessions = [session for values in clean.values() for session in values]
    grid = sorted(set(expected_sessions or observed_sessions))
    if common_as_of:
        grid = [session for session in grid if session <= common_as_of]
    if not grid and common_as_of:
        grid = [common_as_of]
    weekly_grid = _weekly_sessions(grid)
    quality_by_symbol = {
        ticker: {"missing_sessions": [session for session in grid if session not in clean[ticker]],
                 "last_observed": max(clean[ticker], default=None)}
        for ticker in (*SECTOR_TICKERS, BENCHMARK)
    }
    quality_fingerprint = hashlib.sha256(
        json.dumps({"expected_sessions": grid, "symbols": quality_by_symbol}, sort_keys=True,
                   separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    rows: list[SectorRow] = []
    coverage: dict[str, int] = {}
    for ticker in SECTOR_TICKERS:
        series = clean[ticker]
        common = sorted(set(series) & set(benchmark))
        coverage[ticker] = len(common)
        daily = _rotation_history(series, benchmark, "daily", grid)
        weekly_prices = _to_weekly(series, weekly_grid)
        weekly_benchmark = _to_weekly(benchmark, weekly_grid)
        weekly = _rotation_history(weekly_prices, weekly_benchmark, "weekly", weekly_grid)
        transitions = [
            *_transition_events(ticker, "daily", daily),
            *_transition_events(ticker, "weekly", weekly),
        ]
        selected = daily[-1] if daily else None
        old_momentum = daily[-2].relative_momentum if len(daily) > 1 and daily[-2].status == "available" else None
        if selected and selected.status == "available" and selected.relative_momentum is not None and old_momentum is not None:
            delta = selected.relative_momentum - old_momentum
            direction = "rising" if delta > 0.01 else "falling" if delta < -0.01 else "flat"
        else:
            direction = "unavailable"
        changed_at = next((event.confirmed_at for event in reversed(transitions)
                           if event.timeframe == "daily" and event.event_type == "confirmed_transition"), None)
        status = "unavailable" if not series else "available" if common else "partial"
        if series:
            if common and common[-1] != common_as_of:
                status = "partial"
            if common and not daily:
                status = "partial"
            if len(common) < 252:
                status = "partial"
        returns, return_metrics = _return_metrics(series, benchmark, grid)
        current = selected
        first = series.get(common[0]) if common else None
        last = series.get(common[-1]) if common else None
        first_ratio = first / benchmark[common[0]] if common else None
        relative_strength = (
            100.0 * (last / benchmark[common[-1]]) / first_ratio
            if last and first_ratio and common else None
        )
        rows.append(SectorRow(
            ticker=ticker,
            name=sector_name(ticker),
            status=status,
            reason=("price_history_unavailable" if not series else
                    "benchmark_history_unavailable" if not benchmark else
                    "benchmark_overlap_unavailable" if not common else
                    "stale_price_history" if common[-1] != common_as_of else
                    "insufficient_price_history" if len(common) < 252 else
                    selected.reason if selected and selected.status == "unavailable" else None),
            returns_pct=returns,
            return_metrics=return_metrics,
            price_as_of=max(series, default=None),
            rotation_as_of=selected.as_of if selected else None,
            relative_strength=round(relative_strength, 6) if relative_strength is not None else None,
            relative_price_base_date=next((p.as_of for p in _relative_price_history(series, benchmark, "daily", grid)
                                           if p.sector_spy_rebased_100 is not None), None),
            relative_trend=current.relative_trend if current and current.status == "available" else None,
            relative_momentum=current.relative_momentum if current and current.status == "available" else None,
            quadrant=current.quadrant if current and current.status == "available" else None,
            quadrant_changed_at=changed_at,
            momentum_direction=direction,
            history={"daily": daily, "weekly": weekly},
            relative_price_history={
                "daily": _relative_price_history(series, benchmark, "daily", grid),
                "weekly": _relative_price_history(weekly_prices, weekly_benchmark, "weekly", weekly_grid),
            },
            quadrant_transitions=transitions,
        ))
    benchmark_metrics = _returns(benchmark, benchmark, grid) if benchmark else {}
    identity = {
        "schema_version": SCHEMA_VERSION,
        "formula_version": FORMULA_VERSION,
        "calendar_version": CALENDAR_VERSION,
        "transition_rule_version": TRANSITION_RULE_VERSION,
        "universe_version": UNIVERSE_VERSION,
        "benchmark": BENCHMARK,
        "price_basis": "auto_adjusted_close",
        "input_digest": digest,
        "quality_fingerprint": quality_fingerprint,
        "input_start_date": input_start,
        "as_of_date": common_as_of,
        "expected_session": grid[-1] if grid else None,
        "config": _CONFIG,
    }
    snapshot_id = "sr_" + hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()[:32]
    available = sum(row.status != "unavailable" for row in rows)
    return SectorRotationSnapshot(
        input_digest=digest,
        snapshot_id=snapshot_id,
        schema_version=SCHEMA_VERSION,
        formula_version=FORMULA_VERSION,
        calendar_version=CALENDAR_VERSION,
        transition_rule_version=TRANSITION_RULE_VERSION,
        formula_config=_CONFIG,
        as_of_date=common_as_of,
        expected_session=grid[-1] if grid else None,
        expected_weekly_session=weekly_grid[-1] if weekly_grid else None,
        input_start_date=input_start,
        coverage=coverage,
        available_sectors=available,
        benchmark_status="available" if benchmark else "unavailable",
        benchmark_reason="benchmark_history_unavailable" if not benchmark else None,
        benchmark_returns_pct=benchmark_metrics,
        rows=rows,
    )


def _to_weekly(daily: Mapping[str, float], weekly_grid: Optional[Sequence[str]] = None) -> dict[str, float]:
    """Project daily observations onto a shared grid of completed weekly closes."""
    grid = list(weekly_grid or _weekly_sessions(sorted(daily)))
    return {week_end: daily[week_end] for week_end in grid if week_end in daily}
