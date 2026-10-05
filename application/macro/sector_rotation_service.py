"""Use case for immutable sector-rotation snapshots and non-blocking refresh."""
from __future__ import annotations

import os
import hashlib
import json
import threading
import calendar as calendar_module
import math
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Optional

from filelock import FileLock, Timeout

from application.macro.sector_rotation_ports import SectorEvidencePort, SectorHistoryPort, SectorRunBindingPort, SectorSnapshotStorePort
from schemas.sector_rotation_schemas import SectorRotationSnapshot
from tools.market.market_calendar import get_last_completed_regular_session, is_us_trading_day
from tools.macro.sector_rotation.domain.calculations import (
    CALENDAR_VERSION,
    FORMULA_VERSION,
    FORMULA_CONFIG,
    TRANSITION_RULE_VERSION,
    build_snapshot,
    is_current_snapshot,
    normalize_price_inputs,
)


class SectorRotationApplicationService:
    """Serves warm reads quickly and coalesces cold/stale refreshes in one worker."""

    def __init__(
        self,
        history: SectorHistoryPort,
        store: SectorSnapshotStorePort,
        evidence: SectorEvidencePort,
        *,
        calendar=get_last_completed_regular_session,
        data_enabled: Optional[bool] = None,
        ai_enabled: Optional[bool] = None,
        run_bindings: Optional[SectorRunBindingPort] = None,
    ) -> None:
        self._history = history
        self._store = store
        self._evidence = evidence
        self._calendar = calendar
        self._enabled_override = data_enabled
        self._ai_enabled_override = ai_enabled
        self._run_bindings = run_bindings
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sector-rotation-refresh")
        self._future: Optional[Future] = None
        self._verified_snapshot_fingerprints: dict[str, str] = {}

    @property
    def data_enabled(self) -> bool:
        if self._enabled_override is not None:
            return self._enabled_override
        return os.getenv("SECTOR_ROTATION_DATA_ENABLED", "true").strip().lower() not in {"0", "false", "off", "no"}

    @property
    def ai_enabled(self) -> bool:
        if self._ai_enabled_override is not None:
            return self._ai_enabled_override
        return os.getenv("SECTOR_ROTATION_AI_ENABLED", "false").strip().lower() not in {"0", "false", "off", "no"}

    def latest(self, *, timeframe: str = "weekly", tail: int = 12) -> dict[str, Any]:
        if timeframe not in {"daily", "weekly"}:
            raise ValueError("timeframe must be daily or weekly")
        if not 1 <= tail <= 60:
            raise ValueError("tail must be between 1 and 60")
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        if not self.data_enabled:
            return {"capability_status": "disabled", "refresh_state": "idle", "served_at": now,
                    "timeframe": timeframe, "tail": tail, "snapshot": None}

        try:
            cached = self._load_latest()
        except Exception:
            cached = None
            self._store.update_state(refresh_state="failed", error_code="sector_snapshot_integrity_failure")
        expected = self._calendar().isoformat()
        is_cold = cached is None
        is_stale = bool(cached and (
            cached[0].as_of_date != expected or not is_current_snapshot(cached[0])
        ))
        if is_cold or is_stale:
            self.request_refresh()
        state = self._store.read_state()
        refresh_state = str(state.get("refresh_state") or "idle")
        missing_sessions = _missing_completed_sessions(cached[0].as_of_date if cached else None, expected) if cached else 0
        current_schema = bool(cached and is_current_snapshot(cached[0]))
        within_fallback = bool(cached and current_schema and missing_sessions <= 3)
        snapshot = cached[0] if within_fallback else None
        return {
            "capability_status": "enabled",
            "refresh_state": refresh_state,
            "retry_after_seconds": 2 if refresh_state == "running" and is_cold else None,
            "error_code": state.get("error_code"),
            "last_attempt_at": state.get("last_attempt_at"),
            "expected_session": expected,
            "freshness": "fresh" if snapshot and not is_stale else "stale" if snapshot else "unknown",
            "missing_sessions": missing_sessions,
            "served_at": now,
            "timeframe": timeframe,
            "tail": tail,
            "summary": self.summary(snapshot, timeframe=timeframe) if snapshot else None,
            "snapshot": self._view(snapshot, timeframe, tail) if snapshot else None,
        }

    def get_snapshot(self, snapshot_id: str) -> Optional[SectorRotationSnapshot]:
        cached = self._store.load(snapshot_id)
        if cached:
            return self._verified_snapshot(snapshot_id, cached)[0]
        archived = self._evidence.load(snapshot_id)
        if not archived:
            return None
        snapshot, _prices = archived
        return snapshot

    def summary(self, snapshot: SectorRotationSnapshot, *, timeframe: str = "weekly") -> dict[str, Any]:
        if timeframe not in {"daily", "weekly"}:
            raise ValueError("timeframe must be daily or weekly")
        quadrant_members = {name: [] for name in ("Leading", "Weakening", "Lagging", "Improving")}
        periods: dict[str, Optional[int]] = {}
        elapsed_days: dict[str, Optional[int]] = {}
        headings: dict[str, Optional[float]] = {}
        momentum_delta: dict[str, Optional[float]] = {}
        expected_rotation_session = snapshot.expected_weekly_session if timeframe == "weekly" else snapshot.expected_session
        for row in snapshot.rows:
            points = row.history.get(timeframe, [])
            latest = points[-1] if points else None
            if (latest and latest.status == "available" and latest.quadrant
                    and (expected_rotation_session is None or latest.as_of == expected_rotation_session)):
                quadrant_members[latest.quadrant].append(row.ticker)
                count = 0
                for point in reversed(points):
                    if point.status != "available" or point.quadrant != latest.quadrant:
                        break
                    count += 1
                periods[row.ticker] = count
                first_in_run = points[-count]
                elapsed_days[row.ticker] = (date.fromisoformat(latest.as_of) - date.fromisoformat(first_in_run.as_of)).days
            else:
                periods[row.ticker] = None
                elapsed_days[row.ticker] = None

            if (len(points) >= 2 and points[-1].status == "available" and points[-2].status == "available"
                    and points[-1].relative_momentum is not None and points[-2].relative_momentum is not None):
                momentum_delta[row.ticker] = round(points[-1].relative_momentum - points[-2].relative_momentum, 6)
            else:
                momentum_delta[row.ticker] = None

            if (latest and latest.status == "available" and latest.as_of == expected_rotation_session
                    and latest.relative_trend is not None
                    and latest.relative_momentum is not None and len(points) >= 4
                    and all(point.status == "available" and point.relative_trend is not None
                            and point.relative_momentum is not None for point in points[-4:])):
                prior = points[-4]
                dx = float(latest.relative_trend) - float(prior.relative_trend)
                dy = float(latest.relative_momentum) - float(prior.relative_momentum)
                headings[row.ticker] = None if math.hypot(dx, dy) <= 1e-10 else round(math.degrees(math.atan2(dx, dy)) % 360, 2)
            else:
                headings[row.ticker] = None

        eligible = []
        positive_count = 0
        partial_count = 0
        for row in snapshot.rows:
            metric = row.return_metrics.get("3M")
            if not metric or metric.freshness != "fresh" or metric.status == "unavailable" or metric.excess_return_pp is None:
                continue
            eligible.append({
                "ticker": row.ticker,
                "name": row.name,
                "excess_return_pp": metric.excess_return_pp,
                "as_of": metric.end_date,
                "status": metric.status,
                "valid_sessions": metric.valid_sessions,
                "expected_sessions": metric.expected_sessions,
            })
            positive_count += int(metric.excess_return_pp > 0)
            partial_count += int(metric.status == "partial")
        eligible.sort(key=lambda item: (-item["excess_return_pp"], item["ticker"]))
        return {
            "summary_version": "sector-summary-v1",
            "timeframe": timeframe,
            "rotation_as_of": expected_rotation_session,
            "ranked_by_excess_3m": eligible,
            "sector_breadth_3m": {
                "outperforming": positive_count,
                "valid_sectors": len(eligible),
                "expected_sectors": snapshot.expected_sectors,
                "status": "complete" if len(eligible) == snapshot.expected_sectors and partial_count == 0 else "partial",
                "as_of": eligible[0]["as_of"] if eligible else None,
            },
            "quadrant_members": quadrant_members,
            "periods_in_quadrant": periods,
            "elapsed_days_in_quadrant": elapsed_days,
            "momentum_delta": momentum_delta,
            "heading_deg": headings,
        }

    def history(self, snapshot_id: str, *, timeframe: str = "weekly", range_name: str = "1y") -> Optional[dict[str, Any]]:
        if timeframe not in {"daily", "weekly"}:
            raise ValueError("timeframe must be daily or weekly")
        months = {"3m": 3, "6m": 6, "1y": 12, "2y": 24}
        if range_name not in months:
            raise ValueError("range must be one of 3m, 6m, 1y, 2y")
        snapshot = self.get_snapshot(snapshot_id)
        if snapshot is None:
            return None
        end = date.fromisoformat(snapshot.expected_session or snapshot.as_of_date or "")
        month_index = end.year * 12 + end.month - 1 - months[range_name]
        year, month = divmod(month_index, 12)
        start_day = min(end.day, calendar_module.monthrange(year, month + 1)[1])
        start = date(year, month + 1, start_day).isoformat()
        rows = []
        for row in snapshot.rows:
            history = [point.model_dump(mode="json") for point in row.history.get(timeframe, []) if point.as_of >= start]
            relative = [point.model_dump(mode="json") for point in row.relative_price_history.get(timeframe, []) if point.as_of >= start]
            transitions = [event.model_dump(mode="json") for event in row.quadrant_transitions
                           if event.timeframe == timeframe and (event.confirmed_at or event.changed_at or "") >= start]
            rows.append({
                "ticker": row.ticker,
                "name": row.name,
                "status": row.status,
                "reason": row.reason,
                "relative_price_base_date": row.relative_price_base_date,
                "history": history,
                "relative_price_history": relative,
                "quadrant_transitions": transitions,
            })
        return {
            "snapshot_id": snapshot.snapshot_id,
            "input_digest": snapshot.input_digest,
            "formula_version": snapshot.formula_version,
            "timeframe": timeframe,
            "range": range_name,
            "from_date": start,
            "to_date": snapshot.expected_session or snapshot.as_of_date,
            "rows": rows,
        }

    def prepare_for_analysis(self) -> SectorRotationSnapshot:
        """Synchronous path for a Macro job that must pin evidence before LLM calls."""
        if not self.data_enabled:
            raise RuntimeError("sector_rotation_data_disabled")
        latest = self._load_latest()
        expected = self._calendar().isoformat()
        if latest and latest[0].as_of_date == expected and is_current_snapshot(latest[0]) and _has_committed_evidence(latest[1]):
            return latest[0]
        return self._refresh_once()

    def pin_for_run(
        self,
        run_id: str,
        *,
        job_id: str | None = None,
        task_id: str | None = None,
        preferred_snapshot_id: str | None = None,
    ) -> tuple[Optional[SectorRotationSnapshot], dict[str, Any]]:
        """Persist one immutable selection per logical Macro run before it reaches an LLM."""
        stable_run_id = str(run_id or "").strip()
        if not stable_run_id:
            raise ValueError("sector_run_id_required")
        if self._run_bindings is None:
            if not self.ai_enabled:
                return None, {"status": "unavailable", "reason": "ai_capability_disabled"}
            raise RuntimeError("sector_run_binding_store_unavailable")

        with FileLock(
            str(self._run_bindings.lock_path(stable_run_id)),
            timeout=_timeout_from_env("SECTOR_ROTATION_RUN_BINDING_LOCK_TIMEOUT_SECONDS", 120.0),
        ):
            existing = self._run_bindings.load(stable_run_id)
            if existing:
                if not self.ai_enabled or not self.data_enabled:
                    return None, {"status": "unavailable",
                                  "reason": "ai_capability_disabled" if not self.ai_enabled else "data_capability_disabled",
                                  "snapshot_id": existing.get("snapshot_id")}
                return self._resume_binding(existing)

            if preferred_snapshot_id:
                preferred = self.get_snapshot(str(preferred_snapshot_id))
                if preferred is None or not is_current_snapshot(preferred):
                    unavailable = {
                        "run_id": stable_run_id, "job_id": job_id, "logical_task_id": task_id,
                        "snapshot_id": str(preferred_snapshot_id), "data_revision": None,
                        "selected_at": _now(), "evaluation_as_of": self._calendar().isoformat(),
                        "eligibility_at_selection": {}, "publication_status": "unavailable",
                        "unavailable_reason": "legacy_or_missing_snapshot_not_eligible",
                    }
                    self._run_bindings.create(stable_run_id, unavailable)
                    return None, {"status": "unavailable", "reason": unavailable["unavailable_reason"],
                                  "snapshot_id": str(preferred_snapshot_id)}

            if not self.ai_enabled or not self.data_enabled:
                reason = "ai_capability_disabled" if not self.ai_enabled else "data_capability_disabled"
                unavailable = {
                    "run_id": stable_run_id, "job_id": job_id, "logical_task_id": task_id,
                    "snapshot_id": None, "data_revision": None, "selected_at": _now(),
                    "evaluation_as_of": self._calendar().isoformat(), "eligibility_at_selection": {},
                    "publication_status": "unavailable", "unavailable_reason": reason,
                }
                self._run_bindings.create(stable_run_id, unavailable)
                return None, {"status": "unavailable", "reason": reason}

            if preferred_snapshot_id:
                preferred = self.get_snapshot(str(preferred_snapshot_id))
                if preferred is not None:
                    cached = self._store.load(preferred.snapshot_id)
                    receipt = cached[1] if cached else {"status": "recovered_from_evidence"}
                    record = self._binding_record(stable_run_id, preferred, job_id, task_id, receipt=receipt)
                    self._run_bindings.create(stable_run_id, record)
                    return preferred, record

            selected_record: Optional[dict[str, Any]] = None

            def stage(snapshot: SectorRotationSnapshot, prices: dict[str, dict[str, float]], sessions: tuple[str, ...]) -> None:
                nonlocal selected_record
                selected_record = self._binding_record(
                    stable_run_id, snapshot, job_id, task_id,
                    pending_snapshot=snapshot.model_dump(mode="json"),
                    pending_prices=normalize_price_inputs(prices),
                    pending_expected_sessions=list(sessions),
                )
                self._run_bindings.create(stable_run_id, selected_record)

            try:
                snapshot = self._refresh_once(before_publish=stage, require_current_formula=True)
                if selected_record is None:
                    archived = self._evidence.load(snapshot.snapshot_id)
                    if not archived:
                        raise RuntimeError(f"committed_sector_snapshot_evidence_missing:{snapshot.snapshot_id}")
                    cached = self._store.load(snapshot.snapshot_id)
                    receipt = cached[1] if cached else {"status": "recovered_from_evidence"}
                    selected_record = self._binding_record(stable_run_id, snapshot, job_id, task_id, receipt=receipt)
                    self._run_bindings.create(stable_run_id, selected_record)
                else:
                    receipt = (self._store.load(snapshot.snapshot_id) or (None, {}))[1]
                    selected_record = self._run_bindings.update_publication(
                        stable_run_id, str(receipt.get("status") or "committed"), receipt,
                    )
                return snapshot, selected_record
            except Exception as exc:
                existing = self._run_bindings.load(stable_run_id)
                if existing and existing.get("publication_status") == "pending":
                    return None, {"status": "unavailable", "reason": f"snapshot_publication_pending:{type(exc).__name__}",
                                  "snapshot_id": existing.get("snapshot_id")}
                if existing:
                    return self._resume_binding(existing)
                unavailable = {
                    "run_id": stable_run_id,
                    "job_id": job_id,
                    "logical_task_id": task_id,
                    "snapshot_id": None,
                    "data_revision": None,
                    "selected_at": _now(),
                    "evaluation_as_of": self._calendar().isoformat(),
                    "eligibility_at_selection": {},
                    "publication_status": "unavailable",
                    "unavailable_reason": f"snapshot_prepare_failed:{type(exc).__name__}",
                }
                self._run_bindings.create(stable_run_id, unavailable)
                return None, {"status": "unavailable", "reason": unavailable["unavailable_reason"]}

    def _binding_record(
        self,
        run_id: str,
        snapshot: SectorRotationSnapshot,
        job_id: Optional[str],
        task_id: Optional[str],
        *,
        receipt: Optional[dict[str, Any]] = None,
        pending_snapshot: Optional[dict[str, Any]] = None,
        pending_prices: Optional[dict[str, dict[str, float]]] = None,
        pending_expected_sessions: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "job_id": job_id,
            "logical_task_id": task_id,
            "snapshot_id": snapshot.snapshot_id,
            "data_revision": snapshot.input_digest,
            "formula_version": snapshot.formula_version,
            "calendar_version": snapshot.calendar_version,
            "transition_rule_version": snapshot.transition_rule_version,
            "formula_config": snapshot.formula_config,
            "selected_at": _now(),
            "evaluation_as_of": snapshot.expected_session or snapshot.as_of_date,
            "eligibility_at_selection": {
                "return_metrics": {
                    row.ticker: {
                        label: metric.freshness == "fresh" and metric.status != "unavailable"
                        for label, metric in row.return_metrics.items()
                    }
                    for row in snapshot.rows
                },
                "weekly_rotation": {
                    row.ticker: bool(
                        row.history.get("weekly")
                        and row.history["weekly"][-1].status == "available"
                        and row.history["weekly"][-1].as_of == snapshot.expected_weekly_session
                    )
                    for row in snapshot.rows
                },
                "weekly_rotation_as_of": snapshot.expected_weekly_session,
            },
            "publication_status": str((receipt or {}).get("status") or "pending"),
            "publication_receipt": receipt,
            "pending_snapshot": pending_snapshot,
            "pending_prices": pending_prices,
            "pending_expected_sessions": pending_expected_sessions,
        }

    def _resume_binding(self, binding: dict[str, Any]) -> tuple[Optional[SectorRotationSnapshot], dict[str, Any]]:
        status = str(binding.get("publication_status") or "unavailable")
        if status == "unavailable" or not binding.get("snapshot_id"):
            return None, {"status": "unavailable", "reason": binding.get("unavailable_reason") or "binding_unavailable"}
        snapshot_id = str(binding["snapshot_id"])
        if ((binding.get("formula_version") != FORMULA_VERSION
             or binding.get("calendar_version") != CALENDAR_VERSION
             or binding.get("transition_rule_version") != TRANSITION_RULE_VERSION
             or binding.get("formula_config") != FORMULA_CONFIG)
                and status != "pending"):
            return None, {"status": "unavailable", "reason": "legacy_snapshot_formula_not_current", "snapshot_id": snapshot_id}
        if status == "pending":
            raw = binding.get("pending_snapshot")
            prices = binding.get("pending_prices")
            sessions = binding.get("pending_expected_sessions")
            if not isinstance(raw, dict) or not isinstance(prices, dict):
                raise RuntimeError("pending_sector_binding_payload_missing")
            snapshot = SectorRotationSnapshot.model_validate(raw)
            receipt = self._evidence.publish(snapshot, prices, expected_sessions=sessions)
            binding = self._run_bindings.update_publication(str(binding["run_id"]), str(receipt["status"]), receipt)
        snapshot = self.get_snapshot(snapshot_id)
        if snapshot is None or not is_current_snapshot(snapshot):
            return None, {"status": "unavailable", "reason": "pinned_sector_snapshot_unavailable", "snapshot_id": snapshot_id}
        return snapshot, binding

    def request_refresh(self, *, force: bool = False) -> bool:
        if not self.data_enabled:
            return False
        with self._lock:
            if self._future is not None and not self._future.done():
                return False
            state = self._store.read_state()
            if not force and state.get("refresh_state") == "failed" and _seconds_since(state.get("last_attempt_at")) < 60:
                return False
            if state.get("refresh_state") == "running":
                started = state.get("last_attempt_at")
                lease_seconds = _timeout_from_env("SECTOR_ROTATION_REFRESH_LEASE_SECONDS", 120.0)
                if _seconds_since(started) < lease_seconds:
                    return False
            requested_at = _now()
            self._store.update_state(refresh_state="running", last_attempt_at=requested_at,
                                     refresh_requested_at=requested_at, error_code=None)
            self._future = self._executor.submit(self._refresh_once, force, requested_at)
            return True

    def refresh_now(
        self,
        *,
        allow_stale_provider_data: bool = False,
        max_stale_sessions: int = 1,
    ) -> SectorRotationSnapshot:
        """Synchronously refresh; EOD jobs may explicitly accept bounded provider lag."""
        if not self.data_enabled:
            raise RuntimeError("sector_rotation_data_capability_disabled")
        if max_stale_sessions < 0:
            raise ValueError("max_stale_sessions must be non-negative")
        return self._refresh_once(
            allow_stale_provider_data=allow_stale_provider_data,
            max_stale_sessions=max_stale_sessions,
        )

    def _refresh_once(
        self,
        force: bool = False,
        requested_at: Optional[str] = None,
        *,
        before_publish: Optional[Callable[[SectorRotationSnapshot, dict[str, dict[str, float]], tuple[str, ...]], None]] = None,
        require_current_formula: bool = False,
        allow_stale_provider_data: bool = False,
        max_stale_sessions: int = 1,
    ) -> SectorRotationSnapshot:
        with FileLock(
            str(self._store.refresh_lock_path),
            timeout=_timeout_from_env("SECTOR_ROTATION_REFRESH_LOCK_TIMEOUT_SECONDS", 60.0),
        ):
            expected = self._calendar()
            cached = self._load_latest()
            state = self._store.read_state()
            completed_after_request = (
                requested_at is not None
                and _parse_utc(state.get("last_success_at")) is not None
                and _parse_utc(state.get("last_success_at")) >= _parse_utc(requested_at)
            )
            if (cached and cached[0].as_of_date == expected.isoformat() and is_current_snapshot(cached[0])
                    and _has_committed_evidence(cached[1])
                    and (not require_current_formula or is_current_snapshot(cached[0]))
                    and (not force or completed_after_request)):
                self._store.update_state(refresh_state="idle", error_code=None)
                return cached[0]
            self._store.update_state(refresh_state="running", last_attempt_at=_now(), expected_session=expected.isoformat(), error_code=None)
            try:
                batch = self._history.fetch()
                snapshot = build_snapshot(batch.prices, reasons=batch.reasons, expected_sessions=batch.expected_sessions)
                if snapshot.benchmark_status != "available":
                    raise RuntimeError("benchmark_history_unavailable")
                provider_lag = _missing_completed_sessions(snapshot.as_of_date, batch.cutoff.isoformat())
                if provider_lag:
                    if not allow_stale_provider_data:
                        raise RuntimeError("provider_history_missing_last_completed_session")
                    if provider_lag > max_stale_sessions:
                        raise RuntimeError("provider_history_lag_exceeds_eod_limit")
                if before_publish is not None:
                    before_publish(snapshot, batch.prices, batch.expected_sessions)
                evidence_ref = self._evidence.publish(snapshot, batch.prices, expected_sessions=batch.expected_sessions)
                self._store.save(snapshot, evidence_ref)
                self._store.update_state(refresh_state="idle", last_attempt_at=_now(), last_success_at=_now(),
                                         error_code=None, expected_session=expected.isoformat())
                return snapshot
            except Exception as exc:
                self._store.update_state(refresh_state="failed", last_attempt_at=_now(),
                                         error_code=f"refresh_failed:{type(exc).__name__}",
                                         expected_session=expected.isoformat())
                raise

    def _load_latest(self) -> Optional[tuple[SectorRotationSnapshot, dict[str, Any]]]:
        cached = self._store.latest()
        if cached:
            return self._verified_snapshot(cached[0].snapshot_id, cached)
        snapshot_id = str(self._store.read_state().get("latest_snapshot_id") or "")
        if snapshot_id:
            archived = self._evidence.load(snapshot_id)
            if archived:
                snapshot, _prices = archived
                return snapshot, {"status": "recovered_from_evidence", "snapshot_id": snapshot_id}
        return None

    def _verified_snapshot(
        self,
        snapshot_id: str,
        cached: tuple[SectorRotationSnapshot, dict[str, Any]],
    ) -> tuple[SectorRotationSnapshot, dict[str, Any]]:
        fingerprint = _snapshot_fingerprint(cached[0])
        if self._verified_snapshot_fingerprints.get(snapshot_id) == fingerprint:
            return cached
        archived = self._evidence.load(snapshot_id)
        if not archived:
            raise RuntimeError(f"committed_sector_snapshot_evidence_missing:{snapshot_id}")
        snapshot, _prices = archived
        if snapshot.input_digest != cached[0].input_digest or snapshot.snapshot_id != cached[0].snapshot_id:
            raise RuntimeError(f"sector_snapshot_runtime_archive_mismatch:{snapshot_id}")
        self._verified_snapshot_fingerprints[snapshot_id] = _snapshot_fingerprint(snapshot)
        return snapshot, cached[1]

    @staticmethod
    def _view(snapshot: SectorRotationSnapshot, timeframe: str, tail: int) -> dict[str, Any]:
        result = snapshot.model_dump(mode="json")
        for row in result["rows"]:
            full_history = row.get("history", {}).get(timeframe, [])
            history = full_history[-tail:]
            row["history"] = history
            current = full_history[-1] if full_history else None
            if current and current.get("status") == "available":
                row["relative_trend"] = current.get("relative_trend")
                row["relative_momentum"] = current.get("relative_momentum")
                row["quadrant"] = current.get("quadrant")
            else:
                row["relative_trend"] = None
                row["relative_momentum"] = None
                row["quadrant"] = None
            previous = full_history[-2] if len(full_history) >= 2 else None
            if (
                current and previous
                and current.get("status") == previous.get("status") == "available"
                and current.get("relative_momentum") is not None
                and previous.get("relative_momentum") is not None
            ):
                delta = float(current["relative_momentum"]) - float(previous["relative_momentum"])
                row["momentum_direction"] = "rising" if delta > 0.01 else "falling" if delta < -0.01 else "flat"
            else:
                row["momentum_direction"] = "unavailable"
            full_relative_history = row.get("relative_price_history", {}).get(timeframe, [])
            base_points = full_relative_history
            row["relative_price_base_date"] = next(
                (point.get("as_of") for point in base_points if point.get("sector_spy_rebased_100") is not None),
                None,
            )
            row["relative_price_history"] = full_relative_history[-tail:]
            start_date = history[0]["as_of"] if history else ""
            transitions = [
                event for event in row.get("quadrant_transitions", [])
                if event.get("timeframe") == timeframe
            ]
            confirmed = [event for event in transitions if event.get("event_type") == "confirmed_transition"]
            row["quadrant_changed_at"] = confirmed[-1].get("confirmed_at") if confirmed else None
            row["quadrant_transitions"] = [
                event for event in transitions
                if not start_date or (event.get("confirmed_at") or event.get("changed_at") or "") >= start_date
            ][-tail:]
        return result


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _has_committed_evidence(receipt: dict[str, Any]) -> bool:
    """Accept the receipt shape actually persisted by SectorSnapshotStore."""
    return receipt.get("status") in {"committed", "duplicate_reused", "recovered_from_evidence"}


def _snapshot_fingerprint(snapshot: SectorRotationSnapshot) -> str:
    body = json.dumps(snapshot.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _missing_completed_sessions(as_of: Optional[str], expected: str) -> int:
    try:
        current = date.fromisoformat(str(as_of)[:10]) + timedelta(days=1) if as_of else None
        end = date.fromisoformat(expected[:10])
    except ValueError:
        return 10_000
    if current is None:
        return 10_000
    count = 0
    while current <= end:
        count += int(is_us_trading_day(current))
        current += timedelta(days=1)
    return count


def _seconds_since(value: Any) -> float:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - parsed).total_seconds())
    except (TypeError, ValueError):
        return float("inf")


def _timeout_from_env(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = float(default)
    if not math.isfinite(value):
        value = float(default)
    return min(900.0, max(1.0, value))


def _parse_utc(value: Any) -> Optional[datetime]:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None
