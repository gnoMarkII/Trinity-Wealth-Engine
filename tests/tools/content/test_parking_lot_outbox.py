"""Unit tests สำหรับ tools/content/parking_lot_outbox.py"""
import json
from pathlib import Path
import pytest

from tools.content.parking_lot_outbox import (
    write_parking_lot_outbox_atomic,
    mark_outbox_synced_atomic,
    reconcile_parking_lot_outbox,
    _get_outbox_dir,
    _get_quarantine_dir,
)
from api import state_db


def test_write_and_sync_parking_lot_outbox(tmp_path):
    vault_root = tmp_path / "vault"
    ideas = ["ไอเดียที่ 1  ", "ไอเดียที่ 2"]
    
    outbox_file = write_parking_lot_outbox_atomic(
        vault_root=vault_root,
        job_id="job-101",
        pitch_id="pitch-202",
        ideas=ideas,
    )
    
    assert outbox_file.exists()
    with open(outbox_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["schema_version"] == "1.0"
    assert data["job_id"] == "job-101"
    assert data["pitch_id"] == "pitch-202"
    assert data["sync_status"] == "pending"
    assert len(data["ideas"]) == 2
    assert data["ideas"][0]["text"] == "ไอเดียที่ 1"

    # Mark synced
    mark_outbox_synced_atomic(outbox_file)
    with open(outbox_file, "r", encoding="utf-8") as f:
        data_synced = json.load(f)
    assert data_synced["sync_status"] == "synced"
    assert "synced_at" in data_synced


def test_reconcile_parking_lot_outbox(tmp_path):
    vault_root = tmp_path / "vault"
    db_path = str(tmp_path / "state.db")
    
    # 1. Write a pending outbox record
    write_parking_lot_outbox_atomic(
        vault_root=vault_root,
        job_id="job-pending",
        pitch_id="pitch-pending",
        ideas=["ไอเดียต่อยอด A", "ไอเดียต่อยอด B"],
    )

    # 2. Write a corrupt file
    outbox_dir = _get_outbox_dir(vault_root)
    corrupt_file = outbox_dir / "parking_corrupted.json"
    with open(corrupt_file, "w", encoding="utf-8") as f:
        f.write("{ invalid json")

    # 3. Run reconciliation
    summary = reconcile_parking_lot_outbox(vault_root=vault_root, db_path=db_path)
    
    assert summary["scanned_count"] == 2
    assert summary["reconciled_count"] == 2
    assert summary["quarantined_count"] == 1

    # Verify corrupt file moved to quarantine
    quarantine_dir = _get_quarantine_dir(vault_root)
    quarantined = list(quarantine_dir.iterdir())
    assert len(quarantined) == 1
    assert "corrupted" in quarantined[0].name

    # Verify SQLite kanban cards created
    conn = state_db.get_connection(db_path)
    cards = state_db.list_kanban_cards(conn)
    titles = [c["title"] for c in cards]
    assert "ไอเดียต่อยอด A" in titles
    assert "ไอเดียต่อยอด B" in titles
