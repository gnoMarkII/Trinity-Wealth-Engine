"""Unit tests for EquityNotebookLMExportService."""
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
import pytest

from application.equity.notebooklm_export_ports import (
    EquityCorpusSnapshot,
    EquityExportBlockedError,
    EquityExportRecord,
)
from application.equity.notebooklm_export_service import EquityNotebookLMExportService


@pytest.fixture
def sample_snapshot() -> EquityCorpusSnapshot:
    return EquityCorpusSnapshot(
        ticker="NFLX",
        market="US",
        snapshot_at="2026-10-16T12:00:00Z",
        latest_analysis={
            "ticker": "NFLX",
            "market": "US",
            "company_name": "Netflix Inc.",
            "analysis_date": "2026-10-16",
            "quant_signals": {
                "scorecard": {
                    "action_stance": "HOLD_WAIT",
                    "core_conviction_score": 5.5,
                }
            },
        },
        metadata={"counts": {"historical_analyses": 1}, "warnings": []},
    )


@pytest.fixture
def empty_snapshot() -> EquityCorpusSnapshot:
    return EquityCorpusSnapshot(
        ticker="UNKNOWN",
        market="US",
        snapshot_at="2026-10-16T12:00:00Z",
        latest_analysis=None,
        metadata={"counts": {}, "warnings": ["No analysis found"]},
    )


def test_request_export_blocked_when_no_analysis(empty_snapshot, tmp_path: Path):
    reader = MagicMock()
    reader.capture_snapshot.return_value = empty_snapshot
    builder = MagicMock()
    repo = MagicMock()
    dispatcher = MagicMock()

    service = EquityNotebookLMExportService(
        corpus_reader=reader,
        bundle_builder=builder,
        repo=repo,
        dispatcher=dispatcher,
    )

    with pytest.raises(EquityExportBlockedError) as exc_info:
        service.request_export(ticker="UNKNOWN", market="US")

    assert "No baseline analysis found" in str(exc_info.value)
    builder.build_bundle.assert_not_called()
    dispatcher.dispatch.assert_not_called()


def test_request_export_success_creates_and_dispatches(sample_snapshot, tmp_path: Path):
    reader = MagicMock()
    reader.capture_snapshot.return_value = sample_snapshot

    bundle_dir = tmp_path / "bundle_123"
    bundle_dir.mkdir()
    inventory = {"total_sources": 10, "sources": []}
    builder = MagicMock()
    builder.build_bundle.return_value = (bundle_dir, "hash_abc", inventory)

    repo = MagicMock()
    repo.get_by_content_hash.return_value = None  # No existing
    created_rec = EquityExportRecord(
        export_id="export_nflx_123",
        request_key="equity_notebooklm:US:NFLX:hash_abc",
        ticker="NFLX",
        market="US",
        content_hash="hash_abc",
        job_id=None,
        state="queued",
        stage="bundle_frozen",
    )
    repo.create.return_value = created_rec
    repo.update_state.return_value = EquityExportRecord(
        export_id="export_nflx_123",
        request_key="equity_notebooklm:US:NFLX:hash_abc",
        ticker="NFLX",
        market="US",
        content_hash="hash_abc",
        job_id="job_999",
        state="queued",
        stage="dispatched",
    )

    dispatcher = MagicMock()
    dispatcher.dispatch.return_value = "job_999"

    service = EquityNotebookLMExportService(
        corpus_reader=reader,
        bundle_builder=builder,
        repo=repo,
        dispatcher=dispatcher,
    )

    record = service.request_export(ticker="NFLX", market="US")

    assert record.export_id == "export_nflx_123"
    assert record.job_id == "job_999"
    dispatcher.dispatch.assert_called_once_with(
        instruction=record.export_id,
        card_id=None,
        flow="equity_notebooklm",
        scope="both",
    )


def test_request_export_idempotent_reuses_active_record(sample_snapshot, tmp_path: Path):
    reader = MagicMock()
    reader.capture_snapshot.return_value = sample_snapshot

    builder = MagicMock()
    bundle_dir = tmp_path / "bundle_123"
    builder.build_bundle.return_value = (bundle_dir, "hash_abc", {})

    existing = EquityExportRecord(
        export_id="export_existing_456",
        request_key="equity_notebooklm:US:NFLX:hash_abc",
        ticker="NFLX",
        market="US",
        content_hash="hash_abc",
        job_id="job_active_123",
        state="uploading",
        stage="uploading_sources",
    )
    repo = MagicMock()
    repo.get_by_content_hash.return_value = existing

    dispatcher = MagicMock()

    service = EquityNotebookLMExportService(
        corpus_reader=reader,
        bundle_builder=builder,
        repo=repo,
        dispatcher=dispatcher,
    )

    record = service.request_export(ticker="NFLX", market="US")

    assert record.export_id == "export_existing_456"
    assert record.state == "uploading"
    repo.create.assert_not_called()
    dispatcher.dispatch.assert_not_called()


@pytest.mark.asyncio
async def test_execute_worker_job_success(tmp_path: Path):
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text("{}", encoding="utf-8")

    record = EquityExportRecord(
        export_id="export_nflx_123",
        request_key="equity_notebooklm:US:NFLX:hash_abc",
        ticker="NFLX",
        market="US",
        content_hash="hash_abc",
        job_id="job_999",
        state="queued",
        stage="dispatched",
        manifest_path=str(manifest_file),
    )

    repo = MagicMock()
    repo.get_by_id.return_value = record

    pipeline = MagicMock()
    pipeline.execute_export = AsyncMock(return_value={
        "status": "ready",
        "notebook_id": "nb_456",
        "notebook_url": "https://notebooklm.google.com/notebook/nb_456",
    })

    service = EquityNotebookLMExportService(
        corpus_reader=MagicMock(),
        bundle_builder=MagicMock(),
        repo=repo,
        dispatcher=MagicMock(),
        pipeline=pipeline,
    )

    await service.execute_worker_job(job_id="job_999", instruction="export_nflx_123")

    pipeline.execute_export.assert_called_once()
    repo.update_state.assert_any_call(
        export_id="export_nflx_123",
        state="ready",
        stage="completed",
        notebook_id="nb_456",
        notebook_url="https://notebooklm.google.com/notebook/nb_456",
    )
