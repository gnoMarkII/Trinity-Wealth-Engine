from __future__ import annotations

import json
from pathlib import Path

import pytest

from application.knowledge.write_models import KnowledgeWriteCommand
from tools.archivist.maintenance_guard import acquire_maintenance_lease, release_maintenance_lease
from tools.archivist.vault_paths import VaultPaths
from tools.archivist.write_broker import KnowledgeWriteBroker


def _command(key: str, *, title: str = "Broker R8", body: str = "Body", metadata: dict | None = None) -> KnowledgeWriteCommand:
    note_metadata = {
        "schema_version": 2,
        "entity_type": "concept",
        "title": title,
        **(metadata or {}),
    }
    return KnowledgeWriteCommand(
        operation="upsert_note",
        idempotency_key=key,
        producer="r8-test",
        producer_version="1",
        payload={"metadata": note_metadata, "body": body},
    )


def test_maintenance_lease_blocks_before_canonical_side_effect(tmp_path: Path) -> None:
    vault = tmp_path / "memories"
    vault_paths = VaultPaths(vault)
    lease = acquire_maintenance_lease(vault, owner="migration", purpose="test", ttl_seconds=60)
    broker = KnowledgeWriteBroker(vault_paths=vault_paths, runtime_root=tmp_path / "runtime", broker_id="r8")
    try:
        receipt = broker.submit(_command("maintenance-block"))
        assert receipt.status == "retry_wait"
        assert receipt.error_code == "maintenance_lease"
        assert not list(vault.rglob("*.md"))
        assert broker.health()["queue_depth"] == 1
    finally:
        release_maintenance_lease(vault, owner=lease.owner)


def test_expected_revision_rejects_stale_application_update(tmp_path: Path) -> None:
    vault = tmp_path / "memories"
    broker = KnowledgeWriteBroker(vault_paths=VaultPaths(vault), runtime_root=tmp_path / "runtime", broker_id="r8")
    first = broker.submit(_command("revision-1", body="A"))
    assert first.status == "committed"
    second = broker.submit(_command("revision-2", body="B"))
    assert second.status == "committed"

    path = vault / str(second.relative_path)
    metadata = json.loads("{}")
    from tools.archivist.metadata import parse_note

    metadata, _, issues = parse_note(path.read_text(encoding="utf-8"))
    assert not issues
    stale = KnowledgeWriteCommand(
        operation="upsert_note",
        idempotency_key="revision-stale",
        producer="r8-test",
        expected_revision_id=first.revision_id,
        payload={"metadata": metadata, "body": "C"},
    )
    receipt = broker.submit(stale)
    assert receipt.status == "conflict"
    assert receipt.conflict_code == "stale_write"


def test_dead_letter_retry_is_audited(tmp_path: Path) -> None:
    class FailingExecutor:
        def commit(self, command, *, fencing_token=0):
            raise RuntimeError("provider unavailable")

    broker = KnowledgeWriteBroker(
        vault_paths=VaultPaths(tmp_path / "memories"),
        runtime_root=tmp_path / "runtime",
        executor=FailingExecutor(),
        broker_id="r8",
        max_attempts=1,
        retry_backoff_seconds=0,
    )
    command = _command("dead-letter")
    receipt = broker.submit(command)
    assert receipt.status == "dead_letter"
    retried = broker.retry(command.command_id)
    assert retried is not None and retried.status == "dead_letter"
    with broker._connect() as conn:  # audit read only
        events = conn.execute(
            "SELECT to_status, detail_json FROM broker_events WHERE command_id=? ORDER BY event_id",
            (command.command_id,),
        ).fetchall()
    statuses = [row["to_status"] for row in events]
    assert "accepted" in statuses[statuses.index("dead_letter") + 1 :]
    assert statuses[-1] == "dead_letter"
    assert any(json.loads(row["detail_json"]).get("operator_retry") for row in events)


def test_application_update_cannot_silently_overwrite_unreconciled_human_edit(tmp_path: Path) -> None:
    from tools.archivist.metadata import dump_note, parse_note

    vault = tmp_path / "memories"
    broker = KnowledgeWriteBroker(vault_paths=VaultPaths(vault), runtime_root=tmp_path / "runtime", broker_id="r8")
    first = broker.submit(_command("human-race", body="broker baseline"))
    path = vault / str(first.relative_path)
    metadata, _body, issues = parse_note(path.read_text(encoding="utf-8"))
    assert not issues
    path.write_text(dump_note(metadata, "human edit"), encoding="utf-8")
    app_update = KnowledgeWriteCommand(
        operation="upsert_note",
        idempotency_key="human-race-app-update",
        producer="r8-test",
        expected_revision_id=first.revision_id,
        payload={"metadata": metadata, "body": "app update"},
    )
    receipt = broker.submit(app_update)
    assert receipt.status == "conflict"
    assert receipt.conflict_code == "stale_write"
    assert "human edit" in path.read_text(encoding="utf-8")
