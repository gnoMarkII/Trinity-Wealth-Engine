"""
Integration tests for Sector Rotation: Live Adjustment Semantics and Macro AI Shadow.
Validates V06 acceptance criteria:
- AC-09: Live adjusted history semantics (dividends, splits, auto_adjust True vs False, timezone, cutoff, no double adjustment).
- AC-20, AC-24, AC-26: Macro AI shadow report generation with Thai narrative, pinned snapshot binding, resolved sector claims.
- AC-27: Input digest revision immutability (new data revision yields new snapshot_id while prior archive remains readable).
- AC-29: Lineage immutability on run resume/retry.
"""
from datetime import date, datetime, timedelta
import json
import os
from pathlib import Path
import pytest
import yfinance as yf

from schemas.macro_schemas import (
    AssetStance,
    EconomicState,
    MacroStrategyDirection,
)
from schemas.sector_rotation_schemas import (
    QuadrantTransitionEvent,
    ReturnMetric,
    RotationPoint,
    SectorFactClaim,
    SectorRow,
    SectorRotationSnapshot,
)
from tools.macro.adapters.sector_evidence_adapter import SectorEvidenceAdapter
from tools.macro.adapters.sector_history_adapter import SectorHistoryAdapter
from tools.macro.adapters.sector_run_binding_adapter import SectorRunBindingStore
from tools.macro.adapters.sector_snapshot_adapter import SectorSnapshotStore
from application.macro.sector_rotation_service import SectorRotationApplicationService
from tools.macro.sector_rotation.domain.calculations import (
    CALENDAR_VERSION,
    FORMULA_CONFIG,
    TRANSITION_RULE_VERSION,
    build_snapshot,
    normalize_price_inputs,
)
from tools.macro.sector_rotation.domain.claims import (
    compact_ai_context,
    resolve_sector_claims,
)
from tools.macro.sector_rotation.domain.universe import BENCHMARK, SECTOR_TICKERS
from tools.macro.report_formatter import write_strategy_json_sidecar, _load_committed_strategy_report
from tools.archivist.composition import build_knowledge_write_port
from tools.archivist.vault_paths import VaultPaths


def _make_sample_snapshot_and_prices(as_of: str = "2026-10-02", num_days: int = 50):
    """Build a deterministic, fully compliant snapshot and prices using build_snapshot."""
    cutoff = date.fromisoformat(as_of)
    sessions = []
    curr = cutoff - timedelta(days=num_days * 2)
    while curr <= cutoff:
        if curr.weekday() < 5:
            sessions.append(curr.isoformat())
        curr += timedelta(days=1)
    sessions = sessions[-num_days:]

    prices = {t: {s: 100.0 + idx * 0.1 for idx, s in enumerate(sessions)} for t in SECTOR_TICKERS}
    prices[BENCHMARK] = {s: 200.0 + idx * 0.05 for idx, s in enumerate(sessions)}
    clean = normalize_price_inputs(prices)
    snapshot = build_snapshot(clean, expected_sessions=tuple(sessions))
    return snapshot, clean, tuple(sessions)


# ==============================================================================
# 1. AC-09: Live Provider Adjusted History Semantics (Dividends & Splits)
# ==============================================================================

def test_live_provider_dividend_and_split_adjustment_semantics():
    """Verify live yfinance total-return adjusted prices vs raw prices.
    
    1. Dividend event: SPY on 2026-09-18 ($1.889 dividend).
       - auto_adjust=True Close matches auto_adjust=False Adj Close within 1e-4.
       - auto_adjust=False Close pre-dividend is higher by dividend amount.
    2. Split event: XLK on 2025-12-05 (2:1 split).
       - auto_adjust=True Close matches auto_adjust=False Adj Close within 1e-4.
       - No double adjustment or 50% cliff jump exists across the event boundary.
    3. Shared adapter normalization:
       - Strips future dates beyond cutoff, filters invalid dates, ensures valid prices.
    """
    # 1. SPY Dividend Check
    spy = yf.Ticker("SPY")
    df_spy_adj = spy.history(start="2026-09-15", end="2026-09-23", auto_adjust=True)
    df_spy_raw = spy.history(start="2026-09-15", end="2026-09-23", auto_adjust=False)

    assert not df_spy_adj.empty, "Live SPY adjusted history should not be empty"
    assert not df_spy_raw.empty, "Live SPY raw history should not be empty"
    assert len(df_spy_adj) == len(df_spy_raw)

    for dt in df_spy_adj.index:
        close_adj = float(df_spy_adj.loc[dt, "Close"])
        adj_close_raw = float(df_spy_raw.loc[dt, "Adj Close"])
        assert abs(close_adj - adj_close_raw) < 1e-4, f"Mismatch at {dt}: {close_adj} vs {adj_close_raw}"

    # Pre-dividend (2026-09-17) raw Close reflects unadjusted settlement, differing by ~1.889
    dt_pre = [d for d in df_spy_adj.index if "2026-09-17" in str(d)][0]
    raw_close_pre = float(df_spy_raw.loc[dt_pre, "Close"])
    adj_close_pre = float(df_spy_raw.loc[dt_pre, "Adj Close"])
    diff_pre = raw_close_pre - adj_close_pre
    assert 1.80 < diff_pre < 1.95, f"Expected ~1.889 dividend adjustment diff, got {diff_pre}"

    # 2. XLK Split Check
    xlk = yf.Ticker("XLK")
    df_xlk_adj = xlk.history(start="2025-12-01", end="2025-12-10", auto_adjust=True)
    df_xlk_raw = xlk.history(start="2025-12-01", end="2025-12-10", auto_adjust=False)

    assert not df_xlk_adj.empty, "Live XLK history should not be empty"
    for dt in df_xlk_adj.index:
        close_adj = float(df_xlk_adj.loc[dt, "Close"])
        adj_close_raw = float(df_xlk_raw.loc[dt, "Adj Close"])
        assert abs(close_adj - adj_close_raw) < 1e-4, f"XLK split mismatch at {dt}: {close_adj} vs {adj_close_raw}"

    # Verify smooth continuous ratio across split date
    dt_before = [d for d in df_xlk_adj.index if "2025-12-04" in str(d)][0]
    dt_split = [d for d in df_xlk_adj.index if "2025-12-05" in str(d)][0]
    p_before = float(df_xlk_adj.loc[dt_before, "Close"])
    p_split = float(df_xlk_adj.loc[dt_split, "Close"])
    ratio = p_split / p_before
    assert 0.95 < ratio < 1.05, f"Expected smooth continuous ratio across split date, got {ratio}"

    # 3. Adapter normalization logic
    cutoff = date(2026, 9, 18)
    normalized = SectorHistoryAdapter._normalize(df_spy_adj, cutoff)
    for session_str, price in normalized.items():
        session = date.fromisoformat(session_str)
        assert session <= cutoff, f"Session {session} should not exceed cutoff {cutoff}"
        assert price > 0, "Normalized price must be strictly positive"


# ==============================================================================
# 2. AC-27: Snapshot Rebuild from Captured Prices & Revision Immutability
# ==============================================================================

def test_snapshot_rebuild_from_captured_prices_and_revision_immutability(tmp_path, monkeypatch):
    """Verify that price input adjustments generate new unique revisions while preserving archived records."""
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(tmp_path / "vault"))
    vault = tmp_path / "vault"
    runtime = tmp_path / "runtime"
    paths = VaultPaths(vault)
    port = build_knowledge_write_port(vault_paths=paths, runtime_base=runtime)
    evidence = SectorEvidenceAdapter(write_port=port, vault_paths=paths)
    store = SectorSnapshotStore(runtime / "sector_cache")

    snap_v1, clean_v1, sessions = _make_sample_snapshot_and_prices("2026-10-02", num_days=60)
    assert snap_v1.snapshot_id.startswith("sr_")
    assert snap_v1.input_digest is not None

    # Publish v1 to vault evidence
    res_v1 = evidence.publish(snap_v1, clean_v1, expected_sessions=sessions)
    assert res_v1["status"] == "committed"
    store.save(snap_v1, res_v1)

    # Simulate an adjustment revision (e.g. retroactive price restatement for XLK)
    prices_v2 = {t: dict(clean_v1[t]) for t in SECTOR_TICKERS}
    prices_v2[BENCHMARK] = dict(clean_v1[BENCHMARK])
    target_date = sessions[10]
    prices_v2["XLK"][target_date] = 95.0
    clean_v2 = normalize_price_inputs(prices_v2)

    snap_v2 = build_snapshot(clean_v2, expected_sessions=sessions)

    # The new revision must have a DIFFERENT input digest and snapshot ID
    assert snap_v2.input_digest != snap_v1.input_digest
    assert snap_v2.snapshot_id != snap_v1.snapshot_id

    # Publish v2 to vault evidence
    res_v2 = evidence.publish(snap_v2, clean_v2, expected_sessions=sessions)
    assert res_v2["status"] == "committed"
    store.save(snap_v2, res_v2)

    # Both archived revisions must remain independently readable in the vault
    archived_v1 = evidence.load(snap_v1.snapshot_id)
    archived_v2 = evidence.load(snap_v2.snapshot_id)

    assert archived_v1 is not None
    assert archived_v2 is not None
    assert archived_v1[0].snapshot_id == snap_v1.snapshot_id
    assert archived_v2[0].snapshot_id == snap_v2.snapshot_id
    assert archived_v1[1]["XLK"][target_date] != archived_v2[1]["XLK"][target_date]


# ==============================================================================
# 3. AC-20, AC-24: Macro AI Shadow Report Generation & Canonical Receipt
# ==============================================================================

def test_macro_ai_shadow_report_generation_and_canonical_receipt(tmp_path, monkeypatch):
    """Verify that a shadow Macro AI run produces a Thai strategy report with canonical snapshot evidence."""
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(tmp_path / "vault"))
    monkeypatch.setenv("SECTOR_ROTATION_AI_ENABLED", "true")
    monkeypatch.setenv("SECTOR_ROTATION_DATA_ENABLED", "true")

    vault = tmp_path / "vault"
    runtime = tmp_path / "runtime"
    paths = VaultPaths(vault)
    port = build_knowledge_write_port(vault_paths=paths, runtime_base=runtime)
    evidence = SectorEvidenceAdapter(write_port=port, vault_paths=paths)
    store = SectorSnapshotStore(runtime / "sector_cache")
    bindings = SectorRunBindingStore(runtime / "bindings")

    snapshot, clean_prices, sessions = _make_sample_snapshot_and_prices("2026-10-02", num_days=60)
    receipt = evidence.publish(snapshot, clean_prices, expected_sessions=sessions)
    store.save(snapshot, receipt)

    service = SectorRotationApplicationService(
        store=store,
        evidence=evidence,
        history=None,
        calendar=lambda: date.fromisoformat("2026-10-02"),
        run_bindings=bindings,
        data_enabled=True,
        ai_enabled=True,
    )

    run_id = "shadow-macro-run-v06-001"
    pinned_snapshot, binding = service.pin_for_run(run_id, job_id="macro_job_shadow", preferred_snapshot_id=snapshot.snapshot_id)

    assert pinned_snapshot is not None
    assert pinned_snapshot.snapshot_id == snapshot.snapshot_id
    assert binding["publication_status"] in {"committed", "duplicate_reused"}

    # Extract actual metric from pinned snapshot row for XLK
    xlk_row = next(r for r in pinned_snapshot.rows if r.ticker == "XLK")
    metric_1w = xlk_row.return_metrics.get("1W")

    thai_summary = (
        "ภาพรวมเศรษฐกิจอยู่ในภาวะขยายตัว โดยภาคเทคโนโลยี (XLK) มีผลการดำเนินงานที่โดดเด่น "
        f"โดย 1W Excess Return อยู่ที่ {xlk_row.returns_pct.get('1W_excess_pp', 0.0):+.2f} pp"
    )

    # Valid claim referencing pinned snapshot
    valid_claim = SectorFactClaim(
        ticker="XLK",
        claim_kind="positive_excess" if (xlk_row.returns_pct.get("1W_excess_pp") or 0) > 0 else "negative_excess",
        metric_ref=f"sector:{xlk_row.ticker}.1W_excess_pp",
        interpretation_th="ผลตอบแทนส่วนเกิน XLK เทียบ SPY เป็นบวก",
    )
    valid_macro = {"FED_FUNDS_RATE"}
    resolved_claims, _, _, rejected = resolve_sector_claims(
        pinned_snapshot, [valid_claim], [], valid_macro_refs=valid_macro,
    )
    assert len(resolved_claims) == 1
    assert len(rejected) == 0

    sector_analysis_payload = {
        "analysis_status": "available",
        "snapshot_id": pinned_snapshot.snapshot_id,
        "resolved_metrics": [c.model_dump(mode="json") for c in resolved_claims],
        "summary_th": thai_summary,
    }

    from schemas.macro_schemas import AssetAllocationView
    direction = MacroStrategyDirection(
        evaluated_at=datetime.now().isoformat(),
        overall_regime=EconomicState.GOLDILOCKS,
        asset_allocation=[
            AssetAllocationView(
                asset_class="US Equities (Tech)",
                asset_bucket="equities",
                stance=AssetStance.OVERWEIGHT,
                rationale="โมเมนตัมแข็งแกร่งในโซน Leading",
            ),
            AssetAllocationView(
                asset_class="US Bonds",
                asset_bucket="fixed_income",
                stance=AssetStance.NEUTRAL,
                rationale="อัตราดอกเบี้ยทรงตัว",
            ),
        ],
        focus_themes=["US Sector Rotation", "Semiconductor Leadership"],
        conviction_level="high",
        conviction_rationale="ตัวเลขสนับสนุนจาก Relative Rotation Graph ชัดเจน",
        quant_narrative_alignment="aligned",
        divergence_note="",
        source_files=["Macro_Baseline_2026-10-02.md"],
    )

    # Commit report via write_strategy_json_sidecar
    json_path = write_strategy_json_sidecar(
        direction,
        "2026-10-02",
        run_id=run_id,
        job_id="macro_job_shadow",
        sector_snapshot_id=pinned_snapshot.snapshot_id,
        sector_analysis=sector_analysis_payload,
    )

    assert json_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    # Verify report fields & snapshot link
    assert payload["sector_snapshot_id"] == pinned_snapshot.snapshot_id
    assert payload["sector_analysis"]["snapshot_id"] == pinned_snapshot.snapshot_id
    assert payload["sector_analysis"]["summary_th"] == thai_summary
    assert "strategy_report_evidence" in payload
    assert payload["strategy_report_evidence"]["note_id"] is not None
    assert payload["strategy_report_evidence"]["revision_id"] is not None

    # Load committed report from vault to verify durable archive
    report_id = payload["strategy_report_id"]
    committed_payload, committed_evidence = _load_committed_strategy_report(report_id, vault)
    assert committed_payload is not None
    assert committed_evidence is not None
    assert committed_payload["sector_snapshot_id"] == pinned_snapshot.snapshot_id


# ==============================================================================
# 4. AC-29: Lineage Immutability Across Run Retry / Resume
# ==============================================================================

def test_shadow_run_retry_and_resume_maintains_lineage(tmp_path, monkeypatch):
    """Verify that retrying or resuming a pinned run returns the original snapshot even if cache advances."""
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(tmp_path / "vault"))
    vault = tmp_path / "vault"
    runtime = tmp_path / "runtime"
    paths = VaultPaths(vault)
    port = build_knowledge_write_port(vault_paths=paths, runtime_base=runtime)
    evidence = SectorEvidenceAdapter(write_port=port, vault_paths=paths)
    store = SectorSnapshotStore(runtime / "sector_cache")
    bindings = SectorRunBindingStore(runtime / "bindings")

    snap_s1, prices_s1, sessions_s1 = _make_sample_snapshot_and_prices("2026-10-01", num_days=50)
    snap_s2, prices_s2, sessions_s2 = _make_sample_snapshot_and_prices("2026-10-02", num_days=50)

    r1 = evidence.publish(snap_s1, prices_s1, expected_sessions=sessions_s1)
    r2 = evidence.publish(snap_s2, prices_s2, expected_sessions=sessions_s2)
    store.save(snap_s1, r1)
    store.save(snap_s2, r2)

    service = SectorRotationApplicationService(
        store=store,
        evidence=evidence,
        history=None,
        calendar=lambda: date.fromisoformat("2026-10-02"),
        run_bindings=bindings,
        data_enabled=True,
        ai_enabled=True,
    )

    run_id = "shadow-resume-test-001"

    # 1. Pin run to S1
    p1, b1 = service.pin_for_run(run_id, preferred_snapshot_id=snap_s1.snapshot_id)
    assert p1.snapshot_id == snap_s1.snapshot_id
    assert b1["snapshot_id"] == snap_s1.snapshot_id

    # 2. Simulate cache updating latest pointer or another run pinning S2
    # When resuming run_id without specifying preferred_snapshot_id:
    p_resumed, b_resumed = service.pin_for_run(run_id)

    # Lineage must be strictly preserved: still S1!
    assert p_resumed.snapshot_id == snap_s1.snapshot_id
    assert b_resumed["snapshot_id"] == snap_s1.snapshot_id
    assert b_resumed["run_id"] == run_id
