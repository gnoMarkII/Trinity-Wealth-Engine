"""Resolve and validate LLM-authored sector references against Python facts."""
from __future__ import annotations

import math
from typing import Any

from schemas.sector_rotation_schemas import (
    ResolvedMetricClaim,
    SectorFactClaim,
    SectorRotationSnapshot,
    WatchCondition,
)
from tools.macro.sector_rotation.domain.calculations import is_current_snapshot


def _return_fact_is_eligible(row, metric: str) -> tuple[bool, str | None]:
    horizon = metric.split("_", 1)[0]
    fact = row.return_metrics.get(horizon)
    if fact is None:
        return False, None
    value = None
    if metric.endswith("_absolute_pct"):
        value = fact.absolute_return_pct
    elif metric.endswith("_excess_pp"):
        value = fact.excess_return_pp
    elif metric.endswith("_relative_pct"):
        value = fact.relative_return_pct
    return bool(fact.status in {"available", "partial"} and fact.freshness == "fresh" and value is not None), fact.end_date
def _latest_weekly_point(row):
    points = row.history.get("weekly", [])
    if not points or points[-1].status != "available":
        return None
    return points[-1]


def _weekly_momentum_direction(row) -> str:
    points = row.history.get("weekly", [])
    if len(points) < 2 or points[-1].status != "available" or points[-2].status != "available":
        return "unavailable"
    current = points[-1].relative_momentum
    previous = points[-2].relative_momentum
    if current is None or previous is None:
        return "unavailable"
    delta = current - previous
    return "rising" if delta > 0.01 else "falling" if delta < -0.01 else "flat"


def compact_ai_context(snapshot: SectorRotationSnapshot) -> dict[str, Any]:
    sectors = []
    for row in snapshot.rows:
        weekly_point = _latest_weekly_point(row)
        latest_transition = next(
            (event for event in reversed(row.quadrant_transitions)
             if event.timeframe == "weekly" and event.event_type == "confirmed_transition"),
            None,
        )
        latest_event = next((event for event in reversed(row.quadrant_transitions) if event.timeframe == "weekly"), None)
        latest_weekly_point = _latest_weekly_point(row)
        latest_pending = (
            latest_event
            if latest_event and latest_event.event_type == "transition" and latest_weekly_point
            and snapshot.expected_weekly_session is not None
            and latest_weekly_point.as_of == snapshot.expected_weekly_session
            and latest_event.changed_at == latest_weekly_point.as_of
            and latest_event.to_quadrant == latest_weekly_point.quadrant
            else None
        )
        current_returns = {}
        return_quality = {}
        for metric_ref, value in row.returns_pct.items():
            horizon = metric_ref.split("_", 1)[0]
            metric = row.return_metrics.get(horizon)
            eligible = bool(metric and metric.status in {"available", "partial"}
                            and metric.freshness == "fresh" and value is not None)
            current_returns[metric_ref] = value if eligible else None
            if metric:
                return_quality[horizon] = {
                    "start_date": metric.start_date,
                    "end_date": metric.end_date,
                    "status": metric.status,
                    "freshness": metric.freshness,
                    "reason": metric.reason,
                    "valid_sessions": metric.valid_sessions,
                    "expected_sessions": metric.expected_sessions,
                }
        sectors.append({
            "ticker": row.ticker,
            "name": row.name,
            "status": row.status,
            "reason": row.reason,
            "returns_pct": current_returns,
            "return_quality": return_quality,
            "price_as_of": row.price_as_of,
            "rotation_as_of": row.rotation_as_of,
            "rotation_timeframe": "weekly",
            "rotation_as_of_date": weekly_point.as_of if weekly_point else None,
            "relative_trend": weekly_point.relative_trend if weekly_point else None,
            "relative_momentum": weekly_point.relative_momentum if weekly_point else None,
            "quadrant": weekly_point.quadrant if weekly_point else None,
            "quadrant_changed_at": latest_transition.confirmed_at if latest_transition else None,
            "momentum_direction": _weekly_momentum_direction(row),
            "latest_transition": {
                "from": latest_transition.from_quadrant,
                "to": latest_transition.to_quadrant,
                "confirmed_at": latest_transition.confirmed_at,
                "event_ref": latest_transition.event_id,
            } if latest_transition else None,
            "pending_transition": {
                "from": latest_pending.from_quadrant,
                "to": latest_pending.to_quadrant,
                "observed_at": latest_pending.changed_at,
                "event_ref": latest_pending.event_id,
            } if latest_pending else None,
        })
    return {
        "snapshot_id": snapshot.snapshot_id,
        "as_of_date": snapshot.as_of_date,
        "expected_session": snapshot.expected_session,
        "expected_weekly_session": snapshot.expected_weekly_session,
        "input_digest": snapshot.input_digest,
        "benchmark": snapshot.benchmark,
        "formula_version": snapshot.formula_version,
        "coverage": {"available_sectors": snapshot.available_sectors, "expected_sectors": snapshot.expected_sectors},
        "interpretation_limit": "Relative adjusted-price performance is not evidence of fund flows or macro causality.",
        "sectors": sectors,
    }


def resolve_sector_claims(
    snapshot: SectorRotationSnapshot,
    claims: list[SectorFactClaim],
    watch_conditions: list[WatchCondition],
    *,
    valid_macro_refs: set[str] | None = None,
) -> tuple[list[ResolvedMetricClaim], list[WatchCondition], list[SectorFactClaim], list[str]]:
    """Reject unsupported signs, unavailable facts, and hallucinated source refs."""
    if not is_current_snapshot(snapshot):
        return [], [], [], ["legacy_snapshot_not_eligible_for_current_claims"]
    rows = {row.ticker: row for row in snapshot.rows}
    valid_macro_refs = valid_macro_refs or set()
    resolved: list[ResolvedMetricClaim] = []
    valid_conditions: list[WatchCondition] = []
    valid_claims: list[SectorFactClaim] = []
    rejected: list[str] = []
    for claim in claims:
        row = rows.get(claim.ticker)
        if row is None:
            rejected.append(f"unknown_ticker:{claim.ticker}")
            continue
        unknown_refs = set(claim.macro_observable_refs) - valid_macro_refs
        if unknown_refs:
            rejected.append(f"unknown_macro_refs:{claim.ticker}:{','.join(sorted(unknown_refs))}")
            continue
        metric = claim.metric_ref.rsplit(".", 1)[-1]
        if not (claim.metric_ref.startswith(f"{claim.ticker}.") or claim.metric_ref.startswith(f"sector:{claim.ticker}.")):
            rejected.append(f"metric_ticker_mismatch:{claim.ticker}")
            continue
        numeric: float | None = None
        categorical: str | None = None
        unit = ""
        horizon = ""
        metric_as_of = snapshot.expected_session or snapshot.as_of_date or ""
        if claim.claim_kind in {"positive_excess", "negative_excess"}:
            if not metric.endswith("_excess_pp"):
                rejected.append(f"wrong_metric_for_excess:{claim.ticker}")
                continue
            value = row.returns_pct.get(metric)
            eligible, observed_as_of = _return_fact_is_eligible(row, metric)
            metric_as_of = observed_as_of or metric_as_of
            if not eligible or value is None or (claim.claim_kind == "positive_excess" and value <= 0) or (claim.claim_kind == "negative_excess" and value >= 0):
                rejected.append(f"excess_sign_or_availability_mismatch:{claim.ticker}:{metric}")
                continue
            numeric, unit = value, "percentage_points"
            horizon = metric.split("_", 1)[0]
        elif claim.claim_kind == "quadrant_membership":
            weekly_point = _latest_weekly_point(row)
            if (metric != "quadrant" or not weekly_point or not weekly_point.quadrant
                    or (snapshot.expected_weekly_session and weekly_point.as_of != snapshot.expected_weekly_session)):
                rejected.append(f"quadrant_unavailable:{claim.ticker}")
                continue
            categorical, unit, horizon = weekly_point.quadrant, "category", "current_weekly"
            metric_as_of = weekly_point.as_of
        elif claim.claim_kind == "quadrant_transition":
            if metric != "quadrant_transition":
                rejected.append(f"transition_unavailable:{claim.ticker}")
                continue
            transition = next(
                (event for event in reversed(row.quadrant_transitions)
                 if event.timeframe == "weekly" and event.event_type == "confirmed_transition"),
                None,
            )
            if (not transition or not transition.confirmed_at
                    or (snapshot.expected_weekly_session and transition.confirmed_at > snapshot.expected_weekly_session)):
                rejected.append(f"transition_history_unavailable:{claim.ticker}")
                continue
            categorical = f"{transition.from_quadrant}->{transition.to_quadrant}"
            if claim.event_ref != transition.event_id:
                rejected.append(f"transition_event_ref_mismatch:{claim.ticker}")
                continue
            unit, horizon = "transition", "since_previous_valid_weekly_point"
            metric_as_of = transition.confirmed_at
        elif claim.claim_kind in {"momentum_rising", "momentum_falling"}:
            expected = "rising" if claim.claim_kind == "momentum_rising" else "falling"
            if metric != "momentum_direction" or _weekly_momentum_direction(row) != expected:
                rejected.append(f"momentum_direction_mismatch:{claim.ticker}")
                continue
            categorical, unit, horizon = expected, "direction", "latest_weekly_interval"
            weekly_point = _latest_weekly_point(row)
            metric_as_of = weekly_point.as_of if weekly_point else metric_as_of
        else:
            rejected.append(f"unsupported_claim_kind:{claim.claim_kind}")
            continue
        valid_claims.append(claim)
        resolved.append(ResolvedMetricClaim(
            ticker=claim.ticker,
            metric_ref=claim.metric_ref,
            numeric_value=numeric,
            categorical_value=categorical,
            unit=unit,
            horizon=horizon,
            metric_as_of=metric_as_of,
            snapshot_id=snapshot.snapshot_id,
            input_refs=[snapshot.snapshot_id, claim.metric_ref, *claim.macro_observable_refs]
            + ([claim.event_ref] if claim.event_ref else []),
        ))

    for condition in watch_conditions:
        # Watch thresholds are hypotheses about a future observation; they are never resolved metrics.
        reference = condition.metric_ref.removeprefix("sector:")
        ticker = reference.split(".", 1)[0]
        metric = reference.rsplit(".", 1)[-1]
        if ticker not in rows:
            rejected.append(f"watch_unknown_ticker:{ticker}")
            continue
        if metric.endswith("_excess_pp"):
            expected_unit = "percentage_points"
            expected_horizon = metric.split("_", 1)[0]
        elif metric.endswith("_absolute_pct") or metric.endswith("_relative_pct"):
            expected_unit = "percent"
            expected_horizon = metric.split("_", 1)[0]
        elif metric in {"relative_trend", "relative_momentum"}:
            expected_unit = "index"
            expected_horizon = "current_weekly"
        else:
            rejected.append(f"watch_metric_not_observable:{ticker}:{metric}")
            continue
        if condition.unit != expected_unit or condition.horizon != expected_horizon:
            rejected.append(f"watch_unit_or_horizon_mismatch:{ticker}:{metric}")
            continue
        if not math.isfinite(condition.future_threshold):
            rejected.append(f"watch_threshold_not_finite:{ticker}:{metric}")
            continue
        if metric.endswith("_excess_pp") or metric.endswith("_absolute_pct") or metric.endswith("_relative_pct"):
            eligible, _end_date = _return_fact_is_eligible(rows[ticker], metric)
            if rows[ticker].returns_pct.get(metric) is None or not eligible:
                rejected.append(f"watch_metric_unavailable:{ticker}:{metric}")
                continue
        elif metric in {"relative_trend", "relative_momentum"}:
            weekly_point = _latest_weekly_point(rows[ticker])
            if (weekly_point is None or getattr(weekly_point, metric, None) is None
                    or (snapshot.expected_weekly_session and weekly_point.as_of != snapshot.expected_weekly_session)):
                rejected.append(f"watch_metric_unavailable:{ticker}:{metric}")
                continue
        elif getattr(rows[ticker], metric, None) is None:
            rejected.append(f"watch_metric_unavailable:{ticker}:{metric}")
            continue
        valid_conditions.append(condition)
    return resolved, valid_conditions, valid_claims, rejected
