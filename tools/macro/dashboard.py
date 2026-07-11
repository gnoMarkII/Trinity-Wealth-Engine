"""Build and persist dashboard-specific data from validated macro observables."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from schemas.macro_schemas import MacroStrategyDirection, MarketObservable
from tools._atomic_io import _atomic_write_to


_SERIES_SUBDIR = "30_Knowledge_Base/Strategies/Macro_Indicator_Series"
_SAFE_SERIES_KEY = re.compile(r"^[a-z0-9][a-z0-9_-]{0,119}$")
_NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
_DATE_SUFFIX = re.compile(r"_\d{8}(?:_\d+)?$")
_RANGE_DAYS = {"1m": 31, "3m": 92, "1y": 366}


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _series_key(observable: MarketObservable) -> str:
    configured = str(observable.metadata.get("series_key", "")).strip().lower()
    if _SAFE_SERIES_KEY.fullmatch(configured):
        return configured

    stable_id = _DATE_SUFFIX.sub("", observable.observable_id.lower())
    stable_id = _slug(stable_id)
    if stable_id:
        return stable_id[:120]

    return f"{_slug(observable.provider)}_{_slug(observable.indicator)}"[:120]


def _numeric_value(observable: MarketObservable) -> float | None:
    candidate = observable.metadata.get("numeric_value")
    if isinstance(candidate, (int, float)) and math.isfinite(float(candidate)):
        return float(candidate)

    match = _NUMBER.search(str(observable.value))
    if not match:
        return None
    try:
        value = float(match.group(0).replace(",", ""))
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _cited_observable_ids(direction: MacroStrategyDirection) -> set[str]:
    cited_ids: set[str] = set()
    for evidence in direction.regime_evidence:
        cited_ids.update(evidence.observable_refs)
    for allocation in direction.asset_allocation:
        cited_ids.update(allocation.observable_refs)
    for pair_trade in direction.pair_trades:
        cited_ids.update(pair_trade.observable_refs)
    return cited_ids


def _coerce_observable(value: MarketObservable | Mapping[str, Any]) -> MarketObservable | None:
    try:
        return value if isinstance(value, MarketObservable) else MarketObservable.model_validate(value)
    except Exception:
        return None


def build_dashboard_indicators(
    direction: MacroStrategyDirection,
    observable_registry: Mapping[str, MarketObservable | Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Return cited, serializable indicators for the public macro dashboard."""
    if not observable_registry:
        return []

    cited_ids = _cited_observable_ids(direction)
    dashboard_indicators: list[dict[str, Any]] = []
    for observable_id in sorted(cited_ids):
        raw_observable = observable_registry.get(observable_id)
        if raw_observable is None:
            continue
        observable = _coerce_observable(raw_observable)
        if observable is None:
            continue

        numeric_value = _numeric_value(observable)
        dashboard_indicators.append(
            {
                "indicator_id": observable.observable_id,
                "series_key": _series_key(observable),
                "label": observable.indicator,
                "value": numeric_value,
                "display_value": observable.value,
                "unit": observable.unit,
                "observed_at": observable.observed_at,
                "provider": observable.provider,
                "source_file": observable.source_file,
                "is_valid": observable.is_valid,
                "stale_reason": observable.stale_reason,
                "chart_available": numeric_value is not None,
            }
        )
    return dashboard_indicators


def _series_path(vault_path: Path, series_key: str) -> Path:
    if not _SAFE_SERIES_KEY.fullmatch(series_key):
        raise ValueError("Invalid macro indicator series key")
    return vault_path / _SERIES_SUBDIR / f"{series_key}.json"


def persist_indicator_series(vault_path: Path, indicators: list[dict[str, Any]]) -> None:
    """Append one validated report snapshot per indicator date to its local series."""
    for indicator in indicators:
        series_key = str(indicator.get("series_key", ""))
        value = indicator.get("value")
        observed_at = str(indicator.get("observed_at", ""))
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            continue
        try:
            datetime.strptime(observed_at, "%Y-%m-%d")
            path = _series_path(vault_path, series_key)
        except (ValueError, TypeError):
            continue

        series: dict[str, Any] = {
            "series_key": series_key,
            "label": indicator.get("label", ""),
            "unit": indicator.get("unit", ""),
            "points": [],
        }
        if path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict) and isinstance(loaded.get("points"), list):
                    series.update(loaded)
            except (OSError, json.JSONDecodeError):
                pass

        point = {"observed_at": observed_at, "value": float(value)}
        existing_points = {
            str(item.get("observed_at")): item
            for item in series["points"]
            if isinstance(item, dict) and isinstance(item.get("value"), (int, float))
        }
        existing_points[observed_at] = point
        series["points"] = [existing_points[key] for key in sorted(existing_points)[-2_000:]]
        series["series_key"] = series_key
        series["label"] = indicator.get("label", series.get("label", ""))
        series["unit"] = indicator.get("unit", series.get("unit", ""))
        _atomic_write_to(path, json.dumps(series, ensure_ascii=False, separators=(",", ":")))


def load_indicator_series(vault_path: Path, series_key: str, range_key: str) -> list[dict[str, Any]]:
    """Load a bounded range of previously persisted report snapshots."""
    if range_key not in _RANGE_DAYS:
        raise ValueError("Unsupported macro indicator range")
    path = _series_path(vault_path, series_key)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    points = payload.get("points", []) if isinstance(payload, dict) else []
    valid_points: list[dict[str, Any]] = []
    for point in points:
        if not isinstance(point, dict):
            continue
        observed_at = str(point.get("observed_at", ""))
        value = point.get("value")
        try:
            datetime.strptime(observed_at, "%Y-%m-%d")
        except ValueError:
            continue
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            valid_points.append({"observed_at": observed_at, "value": float(value)})

    if not valid_points:
        return []
    valid_points.sort(key=lambda point: point["observed_at"])
    cutoff = date.fromisoformat(valid_points[-1]["observed_at"]) - timedelta(days=_RANGE_DAYS[range_key])
    return [point for point in valid_points if date.fromisoformat(point["observed_at"]) >= cutoff]
