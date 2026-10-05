from datetime import date
from hashlib import sha256

from schemas.sector_rotation_schemas import (
    QuadrantTransitionEvent,
    ReturnMetric,
    RotationPoint,
    SectorFactClaim,
    SectorRow,
    SectorRotationSnapshot,
)
from tools.macro.sector_rotation.domain.claims import compact_ai_context, resolve_sector_claims
from tools.macro.sector_rotation.domain.calculations import CALENDAR_VERSION, FORMULA_CONFIG, TRANSITION_RULE_VERSION
from application.macro.sector_rotation_service import SectorRotationApplicationService
from application.macro.sector_rotation_ports import SectorHistoryBatch
from application.macro.run_identity import next_macro_task_run_id
from api.schemas.sector_rotation import SectorRotationHistoryDTO, SectorRotationResponseDTO


def _snapshot():
    weekly_transition = QuadrantTransitionEvent(
        event_id="evt_weekly_xlk",
        timeframe="weekly",
        previous_valid_at="2026-09-11",
        confirmed_at="2026-09-18",
        from_quadrant="Improving",
        to_quadrant="Leading",
    )
    daily_transition = QuadrantTransitionEvent(
        event_id="evt_daily_xlk",
        timeframe="daily",
        previous_valid_at="2026-09-28",
        confirmed_at="2026-09-29",
        from_quadrant="Improving",
        to_quadrant="Lagging",
    )
    row = SectorRow(
        ticker="XLK",
        name="Technology",
        status="available",
        returns_pct={"1M_absolute_pct": -1.0, "1M_excess_pp": 3.5},
        return_metrics={"1M": ReturnMetric(absolute_return_pct=-1.0, excess_return_pp=3.5,
                                            relative_return_pct=3.8, start_date="2026-08-28",
                                            end_date="2026-09-29", expected_sessions=22,
                                            valid_sessions=22, status="available", freshness="fresh")},
        relative_trend=82.0,
        relative_momentum=80.0,
        quadrant="Lagging",
        momentum_direction="falling",
        history={
            "daily": [
                RotationPoint(as_of="2026-09-28", relative_trend=101, relative_momentum=102, quadrant="Improving"),
                RotationPoint(as_of="2026-09-29", relative_trend=82, relative_momentum=80, quadrant="Lagging"),
            ],
            "weekly": [
                RotationPoint(as_of="2026-09-11", relative_trend=99, relative_momentum=98, quadrant="Improving"),
                RotationPoint(as_of="2026-09-18", relative_trend=103, relative_momentum=101, quadrant="Leading"),
            ],
        },
        quadrant_transitions=[weekly_transition, daily_transition],
    )
    return SectorRotationSnapshot(
        input_digest="digest",
        snapshot_id="sr_test",
        as_of_date="2026-09-29",
        calendar_version=CALENDAR_VERSION,
        transition_rule_version=TRANSITION_RULE_VERSION,
        formula_config=FORMULA_CONFIG,
        expected_session="2026-09-29",
        expected_weekly_session="2026-09-18",
        input_start_date="2025-01-01",
        coverage={"XLK": 300},
        available_sectors=1,
        benchmark_status="available",
        rows=[row],
    )


def test_api_view_uses_requested_timeframe_for_current_marker_and_transition():
    snapshot = _snapshot()

    weekly = SectorRotationApplicationService._view(snapshot, "weekly", 1)
    daily = SectorRotationApplicationService._view(snapshot, "daily", 1)

    assert weekly["rows"][0]["quadrant"] == "Leading"
    assert weekly["rows"][0]["relative_trend"] == 103
    assert weekly["rows"][0]["quadrant_changed_at"] == "2026-09-18"
    assert [event["event_id"] for event in weekly["rows"][0]["quadrant_transitions"]] == ["evt_weekly_xlk"]
    assert daily["rows"][0]["quadrant"] == "Lagging"
    assert daily["rows"][0]["quadrant_changed_at"] == "2026-09-29"


def test_ai_context_and_claim_resolver_use_weekly_rotation_and_daily_returns():
    snapshot = _snapshot()
    context = compact_ai_context(snapshot)
    sector = context["sectors"][0]
    claims = [
        SectorFactClaim(ticker="XLK", claim_kind="quadrant_membership", metric_ref="XLK.quadrant", interpretation_th="weekly leader"),
        SectorFactClaim(ticker="XLK", claim_kind="quadrant_transition", metric_ref="XLK.quadrant_transition", event_ref="evt_weekly_xlk", interpretation_th="weekly rotation confirmed"),
        SectorFactClaim(ticker="XLK", claim_kind="positive_excess", metric_ref="XLK.1M_excess_pp", interpretation_th="outperformed SPY"),
    ]

    resolved, _conditions, accepted, rejected = resolve_sector_claims(snapshot, claims, [])

    assert sector["rotation_timeframe"] == "weekly"
    assert sector["quadrant"] == "Leading"
    assert sector["rotation_as_of_date"] == "2026-09-18"
    assert sector["returns_pct"]["1M_absolute_pct"] == -1.0
    assert {item.claim_kind for item in accepted} == {"quadrant_membership", "quadrant_transition", "positive_excess"}
    assert next(item for item in resolved if item.metric_ref == "XLK.quadrant").metric_as_of == "2026-09-18"
    assert next(item for item in resolved if item.metric_ref == "XLK.quadrant_transition").input_refs[-1] == "evt_weekly_xlk"
    assert rejected == []


def test_ai_claims_with_daily_or_invalid_transition_refs_are_rejected():
    snapshot = _snapshot()
    claims = [
        SectorFactClaim(ticker="XLK", claim_kind="quadrant_transition", metric_ref="XLK.quadrant_transition", event_ref="evt_daily_xlk", interpretation_th="wrong timeframe"),
        SectorFactClaim(ticker="XLK", claim_kind="negative_excess", metric_ref="XLK.1M_excess_pp", interpretation_th="wrong sign"),
    ]

    resolved, _conditions, accepted, rejected = resolve_sector_claims(snapshot, claims, [])

    assert resolved == []
    assert accepted == []
    assert "transition_event_ref_mismatch:XLK" in rejected
    assert "excess_sign_or_availability_mismatch:XLK:1M_excess_pp" in rejected


def test_stale_return_fact_is_redacted_from_ai_context_and_claims_are_rejected():
    snapshot = _snapshot()
    snapshot.rows[0].return_metrics["1M"].freshness = "stale"
    claim = SectorFactClaim(
        ticker="XLK", claim_kind="positive_excess", metric_ref="XLK.1M_excess_pp",
        interpretation_th="positive excess return",
    )

    resolved, _conditions, accepted, rejected = resolve_sector_claims(snapshot, [claim], [])

    assert compact_ai_context(snapshot)["sectors"][0]["returns_pct"]["1M_excess_pp"] is None
    assert resolved == []
    assert accepted == []
    assert "excess_sign_or_availability_mismatch:XLK:1M_excess_pp" in rejected


class _MemoryStore:
    def __init__(self, snapshot, receipt):
        self.snapshot = snapshot
        self.receipt = receipt
        self.refresh_lock_path = ".sector-review-refresh.lock"

    def load(self, _snapshot_id):
        return self.snapshot, self.receipt

    def latest(self):
        return self.snapshot, self.receipt

    def read_state(self):
        return {"latest_snapshot_id": self.snapshot.snapshot_id, "refresh_state": "idle"}

    def update_state(self, **_changes):
        return {}

    def save(self, snapshot, receipt):
        self.snapshot = (snapshot, receipt)


class _MemoryEvidence:
    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.loads = 0

    def load(self, _snapshot_id):
        self.loads += 1
        return self.snapshot, {}

    def publish(self, snapshot, _prices):
        return {"status": "committed", "note_id": "note", "revision_id": "rev"}


class _NoFetchHistory:
    def fetch(self):
        raise AssertionError("fresh committed cache must not refetch prices")


class _MemoryBindings:
    def __init__(self, root):
        self.root = root
        self.records = {}

    def lock_path(self, run_id):
        return self.root / f"{sha256(run_id.encode()).hexdigest()}.lock"

    def load(self, run_id):
        record = self.records.get(run_id)
        return dict(record) if record else None

    def create(self, run_id, record):
        if run_id in self.records:
            raise RuntimeError("sector_run_binding_already_exists")
        self.records[run_id] = dict(record)

    def update_publication(self, run_id, status, receipt=None):
        record = self.records[run_id]
        record["publication_status"] = status
        record["publication_receipt"] = receipt
        for key in ("pending_snapshot", "pending_prices", "pending_expected_sessions"):
            record.pop(key, None)
        return dict(record)


class _EmptyStore:
    def __init__(self):
        self.snapshots = {}
        self.state = {}
        self.refresh_lock_path = ".sector-run-refresh.lock"

    def latest(self):
        snapshot_id = self.state.get("latest_snapshot_id")
        return self.load(snapshot_id) if snapshot_id else None

    def load(self, snapshot_id):
        return self.snapshots.get(snapshot_id)

    def read_state(self):
        return dict(self.state)

    def update_state(self, **changes):
        self.state.update(changes)
        return dict(self.state)

    def save(self, snapshot, receipt):
        self.snapshots[snapshot.snapshot_id] = (snapshot, receipt)
        self.state["latest_snapshot_id"] = snapshot.snapshot_id


class _RetryEvidence:
    def __init__(self):
        self.failures_remaining = 1
        self.published = []
        self.archived = {}

    def publish(self, snapshot, prices, *, expected_sessions=None):
        self.published.append((snapshot.snapshot_id, dict(prices), tuple(expected_sessions or ())))
        if self.failures_remaining:
            self.failures_remaining -= 1
            raise RuntimeError("simulated evidence write failure")
        self.archived[snapshot.snapshot_id] = (snapshot, prices)
        return {"status": "committed", "note_id": "note", "revision_id": "rev"}

    def load(self, snapshot_id):
        return self.archived.get(snapshot_id)


class _CountingHistory:
    def __init__(self):
        self.calls = 0
        self.days = ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08"]

    def fetch(self):
        self.calls += 1
        from tools.macro.sector_rotation.domain.universe import BENCHMARK, SECTOR_TICKERS
        prices = {
            ticker: {day: 100.0 + index for index, day in enumerate(self.days)}
            for ticker in (*SECTOR_TICKERS, BENCHMARK)
        }
        return SectorHistoryBatch(prices, {}, date.fromisoformat(self.days[-1]), tuple(self.days))


class _OneSessionLagHistory(_CountingHistory):
    def fetch(self):
        batch = super().fetch()
        expected = (*batch.expected_sessions, "2025-01-09")
        return SectorHistoryBatch(batch.prices, {}, date.fromisoformat(expected[-1]), expected)


class _ArchiveEvidence:
    def __init__(self):
        self.archived = {}

    def publish(self, snapshot, prices, *, expected_sessions=None):
        receipt = {"status": "committed", "note_id": "note", "revision_id": "rev"}
        self.archived[snapshot.snapshot_id] = (snapshot, dict(prices))
        return receipt

    def load(self, snapshot_id):
        return self.archived.get(snapshot_id)


def test_cached_snapshot_lookup_returns_snapshot_model_not_store_tuple():
    snapshot = _snapshot()
    receipt = {"status": "committed", "note_id": "note", "revision_id": "rev"}
    evidence = _MemoryEvidence(snapshot)
    service = SectorRotationApplicationService(
        _NoFetchHistory(), _MemoryStore(snapshot, receipt), evidence,
        calendar=lambda: date.fromisoformat(snapshot.as_of_date), data_enabled=True,
    )

    try:
        result = service.get_snapshot(snapshot.snapshot_id)
        assert isinstance(result, SectorRotationSnapshot)
        assert result.snapshot_id == snapshot.snapshot_id
        assert SectorRotationApplicationService._view(result, "weekly", 1)["snapshot_id"] == snapshot.snapshot_id
    finally:
        service._executor.shutdown(wait=True)


def test_summary_and_history_are_sliced_from_the_pinned_snapshot_revision():
    snapshot = _snapshot()
    receipt = {"status": "committed", "note_id": "note", "revision_id": "rev"}
    service = SectorRotationApplicationService(
        _NoFetchHistory(), _MemoryStore(snapshot, receipt), _MemoryEvidence(snapshot),
        calendar=lambda: date.fromisoformat(snapshot.as_of_date), data_enabled=True,
    )

    try:
        summary = service.summary(snapshot, timeframe="weekly")
        history = service.history(snapshot.snapshot_id, timeframe="daily", range_name="3m")
        latest_response = SectorRotationResponseDTO.model_validate(service.latest(timeframe="weekly", tail=12))
        history_response = SectorRotationHistoryDTO.model_validate(history)

        assert summary["sector_breadth_3m"] == {
            "outperforming": 0, "valid_sectors": 0, "expected_sectors": 11,
            "status": "partial", "as_of": None,
        }
        assert summary["quadrant_members"]["Leading"] == ["XLK"]
        assert summary["periods_in_quadrant"]["XLK"] == 1
        assert history["snapshot_id"] == snapshot.snapshot_id
        assert history["input_digest"] == snapshot.input_digest
        assert [point["as_of"] for point in history["rows"][0]["history"]] == ["2026-09-28", "2026-09-29"]
        assert latest_response.summary.sector_breadth_3m.expected_sectors == 11
        assert history_response.range == "3m"
    finally:
        service._executor.shutdown(wait=True)


def test_fresh_cached_receipt_is_reused_without_fetching_history(tmp_path):
    snapshot = _snapshot()
    receipt = {"status": "duplicate_reused", "note_id": "note", "revision_id": "rev"}
    evidence = _MemoryEvidence(snapshot)
    store = _MemoryStore(snapshot, receipt)
    store.refresh_lock_path = tmp_path / "refresh.lock"
    service = SectorRotationApplicationService(
        _NoFetchHistory(), store, evidence,
        calendar=lambda: date.fromisoformat(snapshot.as_of_date), data_enabled=True,
    )

    try:
        assert service.prepare_for_analysis().snapshot_id == snapshot.snapshot_id
        assert evidence.loads == 1
        assert service.prepare_for_analysis().snapshot_id == snapshot.snapshot_id
        assert evidence.loads == 1
        assert service.refresh_now().snapshot_id == snapshot.snapshot_id
        assert evidence.loads == 1
    finally:
        service._executor.shutdown(wait=True)


def test_eod_can_archive_one_session_of_provider_lag_but_macro_analysis_stays_strict(tmp_path):
    history = _OneSessionLagHistory()
    evidence = _ArchiveEvidence()
    store = _EmptyStore()
    store.refresh_lock_path = tmp_path / "refresh.lock"
    service = SectorRotationApplicationService(
        history,
        store,
        evidence,
        calendar=lambda: date(2025, 1, 9),
        data_enabled=True,
    )

    try:
        try:
            service.refresh_now()
        except RuntimeError as exc:
            assert str(exc) == "provider_history_missing_last_completed_session"
        else:
            raise AssertionError("normal refresh must reject provider lag")

        snapshot = service.refresh_now(allow_stale_provider_data=True, max_stale_sessions=1)

        assert snapshot.as_of_date == "2025-01-08"
        assert store.latest()[0].snapshot_id == snapshot.snapshot_id
        assert evidence.load(snapshot.snapshot_id) is not None
        try:
            service.prepare_for_analysis()
        except RuntimeError as exc:
            assert str(exc) == "provider_history_missing_last_completed_session"
        else:
            raise AssertionError("Macro analysis must not receive the stale EOD snapshot")
    finally:
        service._executor.shutdown(wait=True)


def test_runtime_snapshot_tampering_after_first_read_is_rechecked():
    snapshot = _snapshot()
    receipt = {"status": "committed", "note_id": "note", "revision_id": "rev"}
    evidence = _MemoryEvidence(snapshot)
    store = _MemoryStore(snapshot, receipt)
    service = SectorRotationApplicationService(
        _NoFetchHistory(), store, evidence,
        calendar=lambda: date.fromisoformat(snapshot.as_of_date), data_enabled=True,
    )

    try:
        assert service.get_snapshot(snapshot.snapshot_id).rows[0].name == "Technology"
        tampered = snapshot.model_copy(deep=True)
        tampered.rows[0].name = "modified runtime payload"
        store.snapshot = tampered
        assert service.get_snapshot(snapshot.snapshot_id).rows[0].name == "Technology"
        assert evidence.loads == 2
    finally:
        service._executor.shutdown(wait=True)


def test_pending_run_binding_retries_same_snapshot_without_refetching(tmp_path):
    history = _CountingHistory()
    evidence = _RetryEvidence()
    bindings = _MemoryBindings(tmp_path)
    service = SectorRotationApplicationService(
        history, _EmptyStore(), evidence,
        calendar=lambda: date.fromisoformat(history.days[-1]),
        data_enabled=True, ai_enabled=True, run_bindings=bindings,
    )

    try:
        first_snapshot, first_status = service.pin_for_run("macro-task-1")
        pending = bindings.load("macro-task-1")
        second_snapshot, second_status = service.pin_for_run("macro-task-1")

        assert first_snapshot is None
        assert first_status["status"] == "unavailable"
        assert pending["publication_status"] == "pending"
        assert second_snapshot is not None
        assert second_status["publication_status"] == "committed"
        assert history.calls == 1
        assert evidence.published[0] == evidence.published[1]
        assert bindings.load("macro-task-1")["snapshot_id"] == second_snapshot.snapshot_id
    finally:
        service._executor.shutdown(wait=True)


def test_macro_task_identity_reuses_retries_and_separates_new_queued_tasks():
    first_run, sequence = next_macro_task_run_id("job-1", "turn-1", 0)
    retry_run, retry_sequence = next_macro_task_run_id(
        "job-1", "turn-1", sequence, retry_run_id=first_run,
    )
    next_run, next_sequence = next_macro_task_run_id("job-1", "turn-1", sequence)

    assert first_run == retry_run == "job-1:turn-1:1"
    assert retry_sequence == sequence == 1
    assert next_run == "job-1:turn-1:2"
    assert next_sequence == 2
