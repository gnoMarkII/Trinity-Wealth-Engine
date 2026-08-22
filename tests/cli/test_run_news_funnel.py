"""Unit tests สำหรับ cli/run_news_funnel.py: Global Pipeline Lock, Synthesis Recovery Fallback และ CLI Routing"""
import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from filelock import FileLock

from cli import run_news_funnel


def test_pipeline_lock_path(tmp_path):
    store = str(tmp_path / "news_state.json")
    lock_path = run_news_funnel._pipeline_lock_path(store)
    assert lock_path.endswith(".json.pipeline.lock")


def test_synthesize_with_recovery_triggers_ingest_when_no_pending_and_has_raw(monkeypatch):
    """เมื่อไม่มี eligible pending แต่มี raw candidates ตกค้าง -> ต้อง trigger recovery ingest ก่อน synthesize"""
    mock_get_pending = MagicMock(return_value=[])
    mock_get_raw = MagicMock(return_value=[{"title": "Raw 1"}])
    mock_ingest = MagicMock(return_value={"status": "success", "high_impact_count": 1})
    mock_synth = MagicMock(return_value={"status": "require_kanban_approval", "pending_events": [{"event_id": "e1"}]})
    mock_handle = MagicMock()

    monkeypatch.setattr("cli.run_news_funnel.get_pending_high_impact_events", mock_get_pending)
    monkeypatch.setattr("cli.run_news_funnel.get_raw_candidates", mock_get_raw)
    monkeypatch.setattr("cli.run_news_funnel.run_news_funnel_ingest", mock_ingest)
    monkeypatch.setattr("cli.run_news_funnel.run_news_funnel_synthesize", mock_synth)
    monkeypatch.setattr("cli.run_news_funnel._handle_synthesize_result", mock_handle)

    args = argparse.Namespace(
        store_path="/fake/store.json",
        vault_root="/fake/vault",
        force_autonomous=False,
    )

    res = run_news_funnel._run_synthesize_with_recovery(args, period="evening")

    mock_ingest.assert_called_once_with(store_path="/fake/store.json", fetch_only=False)
    mock_synth.assert_called_once_with(
        period="evening",
        store_path="/fake/store.json",
        vault_root="/fake/vault",
        allow_autonomous=False,
    )
    mock_handle.assert_called_once_with(mock_synth.return_value)
    assert res == mock_synth.return_value


def test_synthesize_with_recovery_skips_ingest_when_already_has_pending(monkeypatch):
    """เมื่อมี eligible pending อยู่แล้ว -> ต้องไม่รัน recovery ingest"""
    mock_get_pending = MagicMock(return_value=[{"event_id": "e1"}])
    mock_get_raw = MagicMock(return_value=[{"title": "Raw 1"}])
    mock_ingest = MagicMock()
    mock_synth = MagicMock(return_value={"status": "require_kanban_approval"})
    mock_handle = MagicMock()

    monkeypatch.setattr("cli.run_news_funnel.get_pending_high_impact_events", mock_get_pending)
    monkeypatch.setattr("cli.run_news_funnel.get_raw_candidates", mock_get_raw)
    monkeypatch.setattr("cli.run_news_funnel.run_news_funnel_ingest", mock_ingest)
    monkeypatch.setattr("cli.run_news_funnel.run_news_funnel_synthesize", mock_synth)
    monkeypatch.setattr("cli.run_news_funnel._handle_synthesize_result", mock_handle)

    args = argparse.Namespace(
        store_path="/fake/store.json",
        vault_root=None,
        force_autonomous=False,
    )

    res = run_news_funnel._run_synthesize_with_recovery(args, period="morning")

    mock_ingest.assert_not_called()
    mock_synth.assert_called_once()
    mock_handle.assert_called_once()


def test_synthesize_with_recovery_skips_ingest_when_raw_is_empty(monkeypatch):
    """เมื่อไม่มี pending และ raw ก็ว่าง -> ต้องไม่รัน recovery ingest และได้ no_pending_events"""
    mock_get_pending = MagicMock(return_value=[])
    mock_get_raw = MagicMock(return_value=[])
    mock_ingest = MagicMock()
    mock_synth = MagicMock(return_value={"status": "no_pending_events"})
    mock_handle = MagicMock()

    monkeypatch.setattr("cli.run_news_funnel.get_pending_high_impact_events", mock_get_pending)
    monkeypatch.setattr("cli.run_news_funnel.get_raw_candidates", mock_get_raw)
    monkeypatch.setattr("cli.run_news_funnel.run_news_funnel_ingest", mock_ingest)
    monkeypatch.setattr("cli.run_news_funnel.run_news_funnel_synthesize", mock_synth)
    monkeypatch.setattr("cli.run_news_funnel._handle_synthesize_result", mock_handle)

    args = argparse.Namespace(
        store_path=None,
        vault_root=None,
        force_autonomous=False,
    )

    res = run_news_funnel._run_synthesize_with_recovery(args, period="evening")

    mock_ingest.assert_not_called()
    mock_synth.assert_called_once()
    assert res["status"] == "no_pending_events"


def test_synthesize_with_recovery_handles_ingest_exception_gracefully(monkeypatch):
    """หาก recovery ingest ล้มเหลว -> ต้องไม่แครช และยังคงเรียก synthesize ตามปกติ"""
    mock_get_pending = MagicMock(return_value=[])
    mock_get_raw = MagicMock(return_value=[{"title": "Raw 1"}])
    mock_ingest = MagicMock(side_effect=RuntimeError("API quota exceeded"))
    mock_synth = MagicMock(return_value={"status": "no_pending_events"})
    mock_handle = MagicMock()

    monkeypatch.setattr("cli.run_news_funnel.get_pending_high_impact_events", mock_get_pending)
    monkeypatch.setattr("cli.run_news_funnel.get_raw_candidates", mock_get_raw)
    monkeypatch.setattr("cli.run_news_funnel.run_news_funnel_ingest", mock_ingest)
    monkeypatch.setattr("cli.run_news_funnel.run_news_funnel_synthesize", mock_synth)
    monkeypatch.setattr("cli.run_news_funnel._handle_synthesize_result", mock_handle)

    args = argparse.Namespace(
        store_path=None,
        vault_root=None,
        force_autonomous=False,
    )

    res = run_news_funnel._run_synthesize_with_recovery(args, period="evening")

    mock_ingest.assert_called_once()
    mock_synth.assert_called_once()
    assert res["status"] == "no_pending_events"


def test_pipeline_lock_concurrency(tmp_path):
    """ทดสอบว่า FileLock ป้องกันการรัน pipeline พร้อมกันได้จริง"""
    lock_file = str(tmp_path / "test.pipeline.lock")
    lock1 = FileLock(lock_file)
    lock2 = FileLock(lock_file)

    with lock1:
        # Lock2 ต้อง timeout ทันทีเมื่อ timeout=0.1
        with pytest.raises(Exception):
            lock2.acquire(timeout=0.1)
