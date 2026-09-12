from __future__ import annotations

from pathlib import Path

from application.knowledge.write_models import KnowledgeWriteCommand
from tools.archivist.vault_paths import VaultPaths
from tools.archivist.write_broker import KnowledgeWriteBroker


def _command(*, key: str = "article-1", body: str = "# Hello") -> KnowledgeWriteCommand:
    return KnowledgeWriteCommand(
        operation="upsert_note",
        idempotency_key=key,
        producer="test-app",
        producer_version="1.0",
        payload={
            "metadata": {
                "schema_version": 2,
                "entity_type": "concept",
                "title": "Boundary test",
            },
            "body": body,
        },
    )


def test_command_fingerprint_is_stable_for_same_payload() -> None:
    first = _command()
    second = KnowledgeWriteCommand.from_dict(first.to_dict())
    assert first.payload_hash == second.payload_hash
    assert first.command_fingerprint == second.command_fingerprint


def test_broker_commits_and_reuses_idempotent_write(tmp_path: Path) -> None:
    vault = tmp_path / "memories"
    broker = KnowledgeWriteBroker(
        vault_paths=VaultPaths(vault),
        runtime_root=tmp_path / "runtime",
        broker_id="test-broker",
    )
    command = _command()

    first = broker.submit(command)
    replay = broker.submit(command)

    assert first.status == "committed"
    assert replay.status == "duplicate_reused"
    assert first.note_id
    assert first.revision_id
    assert first.relative_path and first.relative_path.endswith(".md")
    assert broker.pending_count() == 0
    assert (vault / first.relative_path).is_file()


def test_idempotency_key_cannot_change_command_meaning(tmp_path: Path) -> None:
    broker = KnowledgeWriteBroker(
        vault_paths=VaultPaths(tmp_path / "memories"),
        runtime_root=tmp_path / "runtime",
        broker_id="test-broker",
    )
    assert broker.submit(_command(body="first")).status == "committed"
    conflict = broker.submit(_command(body="different"))
    assert conflict.status == "conflict"
    assert conflict.conflict_code == "idempotency_key_reused"


def test_retirement_is_event_sourced_and_restore_reopens_identity(tmp_path: Path) -> None:
    vault = tmp_path / "memories"
    broker = KnowledgeWriteBroker(
        vault_paths=VaultPaths(vault),
        runtime_root=tmp_path / "runtime",
        broker_id="test-broker",
    )
    retire = KnowledgeWriteCommand(
        operation="retire_note",
        idempotency_key="retire-1",
        document_key="concept:boundary-test",
        producer="test-app",
        payload={"document_key": "concept:boundary-test", "reason": "test"},
    )
    restore = KnowledgeWriteCommand(
        operation="restore_note",
        idempotency_key="restore-1",
        document_key="concept:boundary-test",
        producer="test-app",
        payload={"document_key": "concept:boundary-test", "reason": "test"},
    )
    assert broker.submit(retire).status == "committed"
    from tools.archivist.vault_policy import is_retired_note

    assert is_retired_note(vault, document_key="concept:boundary-test") is True
    assert broker.submit(restore).status == "committed"
    assert is_retired_note(vault, document_key="concept:boundary-test") is False
