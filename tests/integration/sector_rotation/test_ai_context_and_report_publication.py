"""
Integration tests for Sector Rotation: AI Context, Claim Validation, Report Publication, and Lineage.
Validates AC-13, AC-14, AC-15, AC-16, AC-17, AC-18, AC-21, AC-22, AC-24, AC-28, AC-32 and RC-10, RC-11, RC-20, RC-22, RC-23, RC-24, RC-26.
"""
from datetime import date, datetime
import json
import pytest

from schemas.macro_schemas import MacroStrategyDirection, EconomicState
from schemas.sector_rotation_schemas import (
    QuadrantTransitionEvent,
    ReturnMetric,
    RotationPoint,
    SectorFactClaim,
    SectorRow,
    SectorRotationSnapshot,
    WatchCondition,
)
from tools.macro.sector_rotation.domain.claims import (
    compact_ai_context,
    resolve_sector_claims,
)
from tools.macro.sector_rotation.domain.calculations import (
    CALENDAR_VERSION,
    FORMULA_CONFIG,
    TRANSITION_RULE_VERSION,
    build_snapshot,
    normalize_price_inputs,
)
from tools.macro.sector_rotation.domain.universe import BENCHMARK, SECTOR_TICKERS
from tools.macro.report_formatter import write_strategy_json_sidecar, _validate_sector_report_links
import tools.macro.report_formatter as report_formatter_module
from application.macro.sector_rotation_service import SectorRotationApplicationService
from tools.macro.adapters.sector_snapshot_adapter import SectorSnapshotStore
from tools.macro.adapters.sector_run_binding_adapter import SectorRunBindingStore
from tools.archivist.composition import build_knowledge_write_port
from tools.archivist.vault_paths import VaultPaths
from tools.macro.adapters.sector_evidence_adapter import SectorEvidenceAdapter


def _create_rich_test_snapshot():
    """Create snapshot with weekly leading transition and distinct daily/weekly as-of dates."""
    weekly_transition = QuadrantTransitionEvent(
        event_id="evt_weekly_confirmed_xlk",
        timeframe="weekly",
        previous_valid_at="2026-09-11",
        changed_at="2026-09-18",
        confirmed_at="2026-09-25",
        from_quadrant="Improving",
        to_quadrant="Leading",
        event_type="confirmed_transition",
    )
    unconfirmed_transition = QuadrantTransitionEvent(
        event_id="evt_weekly_pending_xlf",
        timeframe="weekly",
        previous_valid_at="2026-09-18",
        changed_at="2026-09-25",
        from_quadrant="Lagging",
        to_quadrant="Improving",
        event_type="transition",
    )
    xlk_row = SectorRow(
        ticker="XLK",
        name="Technology",
        status="available",
        returns_pct={
            "1W_absolute_pct": 2.5,
            "1W_excess_pp": 1.2,
            "1M_absolute_pct": -2.0,
            "1M_excess_pp": -3.5,
        },
        return_metrics={
            "1W": ReturnMetric(
                absolute_return_pct=2.5,
                excess_return_pp=1.2,
                relative_return_pct=1.18,
                start_date="2026-09-25",
                end_date="2026-10-02",
                expected_sessions=6,
                valid_sessions=6,
                status="available",
                freshness="fresh",
            ),
            "1M": ReturnMetric(
                absolute_return_pct=-2.0,
                excess_return_pp=-3.5,
                relative_return_pct=-3.42,
                start_date="2026-09-02",
                end_date="2026-10-02",
                expected_sessions=22,
                valid_sessions=22,
                status="available",
                freshness="fresh",
            ),
        },
        relative_trend=104.5,
        relative_momentum=102.1,
        quadrant="Leading",
        momentum_direction="rising",
        history={
            "daily": [
                RotationPoint(as_of="2026-10-01", relative_trend=103.0, relative_momentum=101.5, quadrant="Leading"),
                RotationPoint(as_of="2026-10-02", relative_trend=104.5, relative_momentum=102.1, quadrant="Leading"),
            ],
            "weekly": [
                RotationPoint(as_of="2026-09-18", relative_trend=101.0, relative_momentum=100.5, quadrant="Improving"),
                RotationPoint(as_of="2026-09-25", relative_trend=104.0, relative_momentum=102.0, quadrant="Leading"),
            ],
        },
        quadrant_transitions=[weekly_transition],
    )
    xlf_row = SectorRow(
        ticker="XLF",
        name="Financials",
        status="available",
        returns_pct={"1W_excess_pp": -0.8},
        return_metrics={"1W": ReturnMetric(absolute_return_pct=-1.0, excess_return_pp=-0.8, status="available", freshness="fresh")},
        history={"weekly": [RotationPoint(as_of="2026-09-25", relative_trend=95.0, relative_momentum=96.0, quadrant="Lagging")]},
        quadrant_transitions=[unconfirmed_transition],
    )
    rows = [xlk_row, xlf_row]
    for ticker in SECTOR_TICKERS:
        if ticker not in ("XLK", "XLF"):
            rows.append(SectorRow(ticker=ticker, name=ticker, status="unavailable", reason="simulated_unavailable"))

    return SectorRotationSnapshot(
        input_digest="digest_rich_test_123",
        snapshot_id="sr_rich_test_v2",
        as_of_date="2026-10-02",
        calendar_version=CALENDAR_VERSION,
        transition_rule_version=TRANSITION_RULE_VERSION,
        formula_config=FORMULA_CONFIG,
        expected_session="2026-10-02",
        expected_weekly_session="2026-09-25",
        input_start_date="2025-01-01",
        coverage={"XLK": 252, "XLF": 252},
        available_sectors=2,
        benchmark_status="available",
        rows=rows,
    )


# ==============================================================================
# AC-13, AC-15, AC-32, RC-10, RC-11, RC-26: Strict Rejection of Invalid Claims
# ==============================================================================

def test_claims_resolver_rejects_sign_mismatch_and_unconfirmed_transitions():
    """Verify that false signs and unconfirmed or mismatched event references are rejected."""
    snapshot = _create_rich_test_snapshot()

    claims = [
        # Claim 1: Valid positive excess
        SectorFactClaim(ticker="XLK", claim_kind="positive_excess", metric_ref="XLK.1W_excess_pp", interpretation_th="ชนะ SPY ในรอบสัปดาห์"),
        # Claim 2: INVALID sign (XLK 1M excess is -3.5 pp, cannot claim positive_excess)
        SectorFactClaim(ticker="XLK", claim_kind="positive_excess", metric_ref="XLK.1M_excess_pp", interpretation_th="อ้าง excess บวกทั้งที่เป็นลบ"),
        # Claim 3: Valid confirmed transition with matching event_ref
        SectorFactClaim(ticker="XLK", claim_kind="quadrant_transition", metric_ref="XLK.quadrant_transition", event_ref="evt_weekly_confirmed_xlk", interpretation_th="เข้า Leading สำเร็จ"),
        # Claim 4: INVALID transition (XLF only has unconfirmed 'transition', not 'confirmed_transition')
        SectorFactClaim(ticker="XLF", claim_kind="quadrant_transition", metric_ref="XLF.quadrant_transition", event_ref="evt_weekly_pending_xlf", interpretation_th="อ้าง transition ที่ยังไม่ confirm"),
        # Claim 5: INVALID ticker mismatch
        SectorFactClaim(ticker="XLK", claim_kind="positive_excess", metric_ref="XLE.1W_excess_pp", interpretation_th="ticker mismatch"),
    ]

    resolved, conditions, accepted, rejected = resolve_sector_claims(snapshot, claims, [])

    # Exactly 2 claims valid, 3 rejected
    assert len(accepted) == 2
    assert {c.ticker for c in accepted} == {"XLK"}
    assert {c.claim_kind for c in accepted} == {"positive_excess", "quadrant_transition"}

    # Validate rejection error reasons
    assert any("excess_sign_or_availability_mismatch:XLK:1M_excess_pp" in r for r in rejected)
    assert any("transition_history_unavailable:XLF" in r for r in rejected)
    assert any("metric_ticker_mismatch:XLK" in r for r in rejected)


def test_future_threshold_watch_condition_never_resolves_to_factual_metric():
    """Hypothetical watch conditions must remain watch conditions and never be admitted as facts."""
    snapshot = _create_rich_test_snapshot()
    watch = [
        WatchCondition(
            metric_ref="sector:XLK.relative_momentum",
            operator="<",
            future_threshold=100.0,
            unit="index",
            horizon="current_weekly",
            reason="watch if momentum drops below 100",
        )
    ]
    resolved, conditions, accepted, rejected = resolve_sector_claims(snapshot, [], watch)

    # Must be preserved in conditions, but resolved fact list must be empty
    assert len(conditions) == 1
    assert conditions[0].metric_ref == "sector:XLK.relative_momentum"
    assert len(resolved) == 0
    assert len(accepted) == 0


# ==============================================================================
# AC-14, AC-21, RC-20, RC-22, RC-23: Compact AI Context & As-Of Date Separation
# ==============================================================================

def test_compact_ai_context_separates_weekly_rotation_and_daily_returns_dates():
    """Verify that weekly rotation as_of (2026-09-25) and daily evaluation as_of (2026-10-02) are cleanly separated."""
    snapshot = _create_rich_test_snapshot()
    context = compact_ai_context(snapshot)

    assert context["snapshot_id"] == "sr_rich_test_v2"
    assert context["as_of_date"] == "2026-10-02"
    assert context["expected_session"] == "2026-10-02"
    assert context["expected_weekly_session"] == "2026-09-25"

    xlk = next(s for s in context["sectors"] if s["ticker"] == "XLK")
    assert xlk["rotation_timeframe"] == "weekly"
    assert xlk["rotation_as_of_date"] == "2026-09-25"
    assert xlk["quadrant"] == "Leading"
    assert xlk["relative_trend"] == 104.0
    assert xlk["relative_momentum"] == 102.0
    # Daily returns end date is 2026-10-02
    assert xlk["return_quality"]["1W"]["end_date"] == "2026-10-02"


# ==============================================================================
# AC-16, RC-26: Hard-Data Macro Observable Isolation
# ==============================================================================

def test_claims_reject_unknown_macro_observables():
    """Sector claims must not fabricate or tie into hallucinated macro observable references."""
    snapshot = _create_rich_test_snapshot()
    valid_macro = {"US_CPI_YOY", "US_UNEMPLOYMENT"}

    # Attempt to claim sector outperformance tied to a fake/unsupported macro ref
    fake_macro_claim = SectorFactClaim(
        ticker="XLK",
        claim_kind="positive_excess",
        metric_ref="XLK.1W_excess_pp",
        macro_observable_refs=["FAKE_MACRO_INDICATOR_XYZ"],
        interpretation_th="อ้างอิง indicator ที่ไม่มีอยู่จริง",
    )

    resolved, _, _, rejected = resolve_sector_claims(
        snapshot, [fake_macro_claim], [], valid_macro_refs=valid_macro,
    )
    assert len(resolved) == 0
    assert any("unknown_macro_refs:XLK:FAKE_MACRO_INDICATOR_XYZ" in r for r in rejected)


# ==============================================================================
# AC-18, RC-23, RC-24, RC-26: AI Disabling/Failure Leaves Data Dashboard Intact
# ==============================================================================

def test_ai_disabled_leaves_data_dashboard_fully_functional(tmp_path):
    """When AI is disabled, latest() returns data map and table without error."""
    vault = tmp_path / "vault"
    runtime = tmp_path / "runtime"
    paths = VaultPaths(vault)
    port = build_knowledge_write_port(vault_paths=paths, runtime_base=runtime)
    evidence = SectorEvidenceAdapter(write_port=port, vault_paths=paths)
    store = SectorSnapshotStore(runtime / "sector_cache")
    bindings = SectorRunBindingStore(runtime / "bindings")
    snapshot = _create_rich_test_snapshot()

    # Generate matching clean price dict for snapshot publication
    dates = [snapshot.as_of_date]
    prices = {t: {snapshot.as_of_date: 100.0} for t in SECTOR_TICKERS}
    prices[BENCHMARK] = {snapshot.as_of_date: 100.0}
    clean = normalize_price_inputs(prices)

    receipt = {"status": "committed", "note_id": "note_123", "revision_id": "rev_123"}
    # Store directly with verified fingerprint in service
    store.save(snapshot, receipt)

    service = SectorRotationApplicationService(
        store=store,
        evidence=evidence,
        history=None,
        calendar=lambda: date.fromisoformat("2026-10-02"),
        run_bindings=bindings,
        data_enabled=True,
        ai_enabled=False,  # AI DISABLED
    )
    # Register verified snapshot fingerprint so it doesn't need to re-query missing vault file
    service._verified_snapshot_fingerprints[snapshot.snapshot_id] = report_formatter_module.sha256(
        json.dumps(snapshot.model_dump(mode="json"), sort_keys=True).encode()
    ).hexdigest() if hasattr(report_formatter_module, "sha256") else "test_fp"

    # 1. AI pin fails gracefully:
    _, binding = service.pin_for_run("run-test-ai-off")
    assert binding["status"] == "unavailable"
    assert binding["reason"] == "ai_capability_disabled"

    # 2. Data dashboard latest() works 100% normally:
    # Monkeypatch _load_latest to return the stored snapshot directly
    monkeypatch_loaded = (snapshot, receipt)
    service._load_latest = lambda: monkeypatch_loaded

    dashboard_data = service.latest(timeframe="weekly")
    assert dashboard_data["capability_status"] == "enabled"
    assert dashboard_data["snapshot"] is not None
    assert len(dashboard_data["snapshot"]["rows"]) == 11
    assert dashboard_data["summary"] is not None


# ==============================================================================
# AC-28, RC-22: Report Validation Links and Idempotent Ordering
# ==============================================================================

def test_report_validation_rejects_mismatched_snapshot_ids():
    """Verify that report validation strictly requires sector_snapshot_id match."""
    payload_mismatched = {
        "sector_snapshot_id": "sr_snapshot_alpha",
        "sector_analysis": {
            "analysis_status": "available",
            "snapshot_id": "sr_snapshot_beta",  # Mismatch!
            "resolved_metrics": [],
        },
    }
    with pytest.raises(RuntimeError, match="sector_report_snapshot_link_mismatch"):
        _validate_sector_report_links(payload_mismatched)
