"""Regression tests converted from 2026-09-07 behavior probes.

These tests capture the concrete failures identified during the Vault V2 review:
1. Sidecar-only change overwriting previous frozen revision without creating a new revision ID.
2. Different report dates for the same ticker incorrectly receiving the same note_id.
3. Companion write failure leading to partial commit without pending recovery journals.
4. Earnings calls with different transcripts in the same period overwriting each other without note_id.
5. Corrupt NotebookLM manifest returning None and allowing unintended new runs.

All tests run strictly in isolated temporary directories with disabled network.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tools.archivist import artifact_writer as aw
from tools.archivist.vault_paths import VaultPaths
from tools.content.earnings_call.adapters import obsidian_adapter as ea
from tools.content.notebooklm import manifest as nlm_manifest


@pytest.fixture(autouse=True)
def isolate_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensures test runs against tmp_path and network is completely blocked."""
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(tmp_path))
    monkeypatch.setenv("WEBUI_STATE_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("ENABLE_BACKGROUND_WORKERS", "false")
    monkeypatch.setenv("SCHEDULER_ENABLED", "false")

    # Disable socket connect to guarantee zero external calls
    def forbidden_network(*args, **kwargs):
        raise RuntimeError("External network call forbidden in isolated tests")

    import socket
    monkeypatch.setattr(socket.socket, "connect", forbidden_network)
    monkeypatch.setattr(socket, "create_connection", forbidden_network)


def test_probe_1_sidecar_only_change_must_create_new_revision(tmp_path: Path) -> None:
    """Probe 1: Changing companion sidecar (e.g. data.json) MUST produce a new revision

    and MUST NOT overwrite the previous revision's frozen sidecar.
    """
    (tmp_path / ".system").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".system" / "vault_config.json").write_text('{"layout_version": 2}', encoding="utf-8")

    writer = aw.ArtifactWriter(vault_paths=VaultPaths(tmp_path))
    meta = {
        "entity_type": "equity_analysis",
        "ticker": "FTNT",
        "title": "Review Analysis",
        "date": "2026-09-01",
    }

    first = writer.write_note(
        meta,
        "Same analysis body",
        filename="Day1",
        companion_artifacts={"data.json": '{"score":10}'},
    )
    frozen_sidecar_1 = tmp_path / "40_Archive" / "Revisions" / first.note_id / first.revision_id / "data.json"
    assert frozen_sidecar_1.exists()
    frozen_before = frozen_sidecar_1.read_bytes()

    # Second write with changed companion sidecar
    revised = writer.write_note(
        meta,
        "Same analysis body",
        filename="Day1",
        companion_artifacts={"data.json": '{"score":20}'},
    )

    # Correct invariant: Revision MUST NOT be reused, revision_id MUST be different,
    # and first frozen revision MUST remain untouched!
    assert revised.revision_id != first.revision_id, (
        f"Sidecar change must produce distinct revision_id, got same: {first.revision_id}"
    )
    assert revised.is_reused is False, "Sidecar change must not be marked as reused"
    assert frozen_sidecar_1.read_bytes() == frozen_before, (
        "Previous frozen sidecar was overwritten by revised write!"
    )


def test_probe_2_different_report_dates_must_get_distinct_note_ids(tmp_path: Path) -> None:
    """Probe 2: Equity analyses on different dates for same ticker MUST have distinct note_ids."""
    (tmp_path / ".system").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".system" / "vault_config.json").write_text('{"layout_version": 2}', encoding="utf-8")

    writer = aw.ArtifactWriter(vault_paths=VaultPaths(tmp_path))
    meta_day1 = {
        "entity_type": "equity_analysis",
        "ticker": "FTNT",
        "title": "FTNT 2026-09-01",
        "date": "2026-09-01",
    }
    first = writer.write_note(meta_day1, "Day 1 analysis body", filename="Day1")

    meta_day2 = {
        "entity_type": "equity_analysis",
        "ticker": "FTNT",
        "title": "FTNT 2026-09-02",
        "date": "2026-09-02",
    }
    second = writer.write_note(meta_day2, "Day 2 analysis body", filename="Day2")

    # Correct invariant: Different dates represent distinct document identities
    assert first.note_id != second.note_id, (
        f"Different report dates received identical note_id: {first.note_id}"
    )
    assert first.primary_file != second.primary_file
    assert first.primary_file.exists() and second.primary_file.exists()


def test_probe_3_companion_failure_must_prevent_partial_commit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Probe 3: Companion write failure must not leave partial commit and must preserve recovery journal."""
    (tmp_path / ".system").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".system" / "vault_config.json").write_text('{"layout_version": 2}', encoding="utf-8")

    writer = aw.ArtifactWriter(vault_paths=VaultPaths(tmp_path))
    meta = {
        "entity_type": "equity_analysis",
        "ticker": "FTNT",
        "title": "Review Analysis",
        "date": "2026-09-01",
    }
    first = writer.write_note(meta, "Initial body", filename="Day1", companion_artifacts={"data.json": '{"score":10}'})
    initial_content = first.primary_file.read_text(encoding="utf-8")

    # Inject failure when writing companion
    real_atomic_write = aw._atomic_write_text

    def fail_companion(path, content, *args, **kwargs):
        if Path(path) == first.primary_file.parent / "data.json":
            raise OSError("Injected companion commit failure")
        return real_atomic_write(path, content, *args, **kwargs)

    monkeypatch.setattr(aw, "_atomic_write_text", fail_companion)

    with pytest.raises(OSError, match="Injected companion commit failure"):
        writer.write_note(
            meta,
            "Changed primary before companion failure",
            filename="Day1",
            companion_artifacts={"data.json": '{"score":30}'},
        )

    # Invariant: Either primary was not updated in-place, OR pending journal is retained for crash recovery
    pending_journals = list((tmp_path / ".system" / "pending_writes").rglob("journal.json"))
    current_primary = first.primary_file.read_text(encoding="utf-8")

    if current_primary != initial_content:
        # If primary was swapped, recovery journal MUST be retained to allow recovery/rollback
        assert len(pending_journals) > 0, "Partial commit occurred without retaining recovery journal"
    else:
        # Clean rollback on failure
        assert current_primary == initial_content


def test_probe_4_earnings_same_period_transcripts_preserve_both(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Probe 4: Multiple earnings transcripts in same period must not overwrite and must have note_id."""
    (tmp_path / ".system").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".system" / "vault_config.json").write_text('{"layout_version": 2}', encoding="utf-8")

    monkeypatch.setattr(ea, "_index_upsert", lambda *args, **kwargs: None)
    adapter = ea.ObsidianEarningsCallAdapter(tmp_path)

    path_a = adapter.write_note("FTNT", "Q2 2026", "Transcript A content", "Highlights A")
    path_b = adapter.write_note("FTNT", "Q2 2026", "Transcript B content", "Highlights B")

    file_a = tmp_path / path_a
    file_b = tmp_path / path_b

    assert file_a.exists(), "Original transcript file must exist"
    assert "Transcript A content" in file_a.read_text(encoding="utf-8"), "Transcript A was overwritten!"

    assert file_b.exists(), "Second transcript file must exist"
    assert "Transcript B content" in file_b.read_text(encoding="utf-8")

    # Both must have note_id
    assert "note_id:" in file_a.read_text(encoding="utf-8")
    assert "note_id:" in file_b.read_text(encoding="utf-8")


def test_probe_5_notebooklm_corrupt_manifest_fail_closed(tmp_path: Path) -> None:
    """Probe 5: Corrupt manifest must return a typed corrupt/error result and never silently fall back to new_manifest()."""
    bad_manifest = tmp_path / "corrupt_manifest.json"
    bad_manifest.write_text("{invalid_json", encoding="utf-8")

    # In the updated system, load_manifest must return typed result or raise, NOT return None
    result = nlm_manifest.load_manifest(bad_manifest)
    assert result is not None, "Corrupt manifest must not return None (which triggers silent new_manifest())"
    assert getattr(result, "status", "") in ("corrupt", "error") or getattr(result, "is_corrupt", False), (
        "Corrupt manifest must be explicitly marked as corrupt"
    )


# ──────────────────────────────────────────────────────────────────────────────
# C02 — Import conflict must fail-closed; durable store must not be overwritten
# Gate: A01 (identity preservation)
# ──────────────────────────────────────────────────────────────────────────────

def test_c02_import_conflict_fails_closed_without_overwrite(tmp_path: Path) -> None:
    """C02: Importing a different note_id for an already-registered document_key must raise a
    typed conflict error. The durable store bytes must be identical before and after the attempt.

    This verifies F02: import conflict fail-closed — no silent overwrite of existing mappings.
    """
    from tools.archivist.identity_store import DurableIdentityStore

    store = DurableIdentityStore(root=tmp_path)
    doc_key = "v2:concept:DeFi:primary"

    # First allocation creates the canonical mapping
    original = store.reserve_note_identity(doc_key, entity_id=None)
    assert original.note_id.startswith("note_")

    store_file = tmp_path / ".system" / "identity_allocations.json"
    bytes_before = store_file.read_bytes()

    # Attempt to import a DIFFERENT note_id for the same document_key
    conflicting_id = "note_aabbcc112233"
    assert conflicting_id != original.note_id, "Test setup: conflicting_id must differ from original"

    with pytest.raises(Exception) as exc_info:
        store.import_note_identity(
            note_id=conflicting_id,
            document_key=doc_key,
        )

    # Must raise a conflict-typed exception (not just log and overwrite)
    assert exc_info.type.__name__ in (
        "IdentityConflictError", "DuplicateNoteIdError", "ValueError", "RuntimeError"
    ), f"Expected a conflict exception, got {exc_info.type.__name__}: {exc_info.value}"

    # Store bytes MUST be unchanged after the failed import
    bytes_after = store_file.read_bytes()
    assert bytes_before == bytes_after, (
        "Store was mutated during a failed import — original mapping was overwritten!"
    )

    # The original note_id must still be resolvable
    retrieved = store.get_note_identity(doc_key)
    assert retrieved is not None
    assert retrieved.note_id == original.note_id, (
        f"Original note_id was replaced: expected {original.note_id}, got {retrieved.note_id}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# C03 — Malformed registry + concurrent allocate: no reset to {}, no duplicate IDs
# Gate: A01 (identity preservation), A03 (recovery)
# ──────────────────────────────────────────────────────────────────────────────

def test_c03_malformed_store_does_not_reset_to_empty_dict(tmp_path: Path) -> None:
    """C03: When the identity store file is malformed (truncated / invalid JSON), reading it
    MUST NOT silently return {} which would allow re-allocation of previously assigned IDs.

    Verifies: F02 — corrupt store quarantine; A01 — no allocation overwrite.
    """
    from tools.archivist.identity_store import DurableIdentityStore

    store = DurableIdentityStore(root=tmp_path)
    doc_key = "v2:article_note:some-url:primary"

    # Allocate original ID
    original = store.reserve_note_identity(doc_key)

    # Corrupt the store file
    store_file = tmp_path / ".system" / "identity_allocations.json"
    store_file.write_text("{INVALID_JSON_CORRUPT", encoding="utf-8")

    # A new store instance must detect corruption — must NOT return {} silently
    store2 = DurableIdentityStore(root=tmp_path)

    # Attempting to reserve the same key must either:
    # (a) raise a StoreCorruptError / similar, OR
    # (b) correctly quarantine and preserve the corrupt evidence without losing it
    try:
        result = store2.reserve_note_identity(doc_key)
        # If it succeeds, it MUST return a new/different ID (not silently reuse empty mapping)
        # AND the original corrupt file must have been quarantined, not simply overwritten silently
        quarantine_exists = any(
            (tmp_path / ".system").rglob("*corrupt*")
        ) or any(
            (tmp_path / ".system").rglob("*quarantine*")
        )
        # Either quarantine OR the returned ID matches original (recovery from corrupt was partial)
        # The key invariant: returned ID must not be a fresh allocation if original existed
        assert quarantine_exists or result.note_id == original.note_id, (
            "Corrupt store was silently reset to {} and a new ID was allocated without quarantine evidence"
        )
    except Exception as e:
        # A typed error is acceptable — the point is: do NOT silently reset
        assert "corrupt" in str(e).lower() or "quarantine" in str(e).lower() or "parse" in str(e).lower(), (
            f"Expected corrupt-related error, got: {e}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# C04 — Missing manifest for frozen revision must be rejected (not returned as success)
# Gate: A02 (immutable revision), A04 (history continuity)
# ──────────────────────────────────────────────────────────────────────────────

def test_c04_missing_manifest_must_be_rejected_not_success(tmp_path: Path) -> None:
    """C04: DurableArtifactStore.get_revision_artifact() with a missing manifest.json must
    raise ArtifactNotFoundError. It must never return a successful StoredArtifact with manifest=None.

    Verifies: F03.1 — strict reader; manifest is required, not optional.
    """
    from tools.archivist.artifact_store import DurableArtifactStore, ArtifactNotFoundError
    from tools.archivist.vault_paths import VaultPaths

    (tmp_path / ".system").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".system" / "vault_config.json").write_text('{"layout_version": 2}', encoding="utf-8")

    # Create a revision directory with ONLY the markdown file — no manifest
    note_id = "note_c04testfixture"
    rev_id = "rev_abc123def456"
    rev_dir = tmp_path / "40_Archive" / "Revisions" / note_id / rev_id
    rev_dir.mkdir(parents=True, exist_ok=True)

    md_content = "---\ntitle: C04 Test\nentity_type: concept\n---\n# C04 Test\nBody."
    (rev_dir / "note.md").write_text(md_content, encoding="utf-8")
    # Deliberately NO manifest.json

    store = DurableArtifactStore(vault_paths=VaultPaths(tmp_path))

    with pytest.raises((ArtifactNotFoundError, Exception)) as exc_info:
        store.get_revision_artifact(note_id, rev_id, verify_digests=True)

    # Must raise an error — NOT return success with manifest=None
    exc_msg = str(exc_info.value).lower()
    assert any(kw in exc_msg for kw in ("manifest", "missing", "not found", "required")), (
        f"Expected manifest-related error, got: {exc_info.value}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# C05 — Missing required companion must be rejected (not returned with empty companions)
# Gate: A02 (immutable revision), A04 (history continuity)
# ──────────────────────────────────────────────────────────────────────────────

def test_c05_missing_required_companion_must_be_rejected(tmp_path: Path) -> None:
    """C05: When a manifest declares data.json as a required companion but the file is absent,
    get_revision_artifact() must raise an error. It must never return a result with companions={}.

    Verifies: F03.1 — strict reader; required companions must be validated from manifest list.
    """
    import json
    import hashlib
    from tools.archivist.artifact_store import DurableArtifactStore, ArtifactNotFoundError
    from tools.archivist.vault_paths import VaultPaths

    (tmp_path / ".system").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".system" / "vault_config.json").write_text('{"layout_version": 2}', encoding="utf-8")

    note_id = "note_c05testfixture"
    rev_id = "rev_c05deadbeef"
    rev_dir = tmp_path / "40_Archive" / "Revisions" / note_id / rev_id
    rev_dir.mkdir(parents=True, exist_ok=True)

    md_content = "---\ntitle: C05 Test\nentity_type: concept\n---\n# C05 Test\nBody."
    md_bytes = md_content.encode("utf-8")
    (rev_dir / "note.md").write_text(md_content, encoding="utf-8")

    # Create manifest that declares data.json as required companion
    manifest_data = {
        "note_id": note_id,
        "revision_id": rev_id,
        "revision": 1,
        "primary_sha256": hashlib.sha256(md_bytes).hexdigest(),
        "required_companions": ["data.json"],
        "companions": {"data.json": "deadbeef" * 8},  # hash for a file that does NOT exist
        "algorithm": "sha256",
        "manifest_version": 1,
    }
    (rev_dir / "manifest.json").write_text(json.dumps(manifest_data), encoding="utf-8")
    # Deliberately NO data.json

    store = DurableArtifactStore(vault_paths=VaultPaths(tmp_path))

    with pytest.raises((ArtifactNotFoundError, Exception)) as exc_info:
        store.get_revision_artifact(note_id, rev_id, verify_digests=True)

    # Must raise an error about missing required companion — not succeed with empty companions
    exc_msg = str(exc_info.value).lower()
    assert any(kw in exc_msg for kw in ("companion", "missing", "required", "data.json")), (
        f"Expected companion-related error, got: {exc_info.value}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# C08 — A→B→A: three distinct revision_ids; returned/current/frozen ref = revision 3
# Gate: A02 (immutable revision)
# ──────────────────────────────────────────────────────────────────────────────

def test_c08_a_b_a_cycle_gets_distinct_revision_ids(tmp_path: Path) -> None:
    """C08: Writing content A, then B, then A again must produce three distinct revision_ids.
    The returned result for the third write must report revision=3 and its revision_id
    must NOT match the revision_id of the first write.

    Verifies: F03.2 — no content-addressed revision_id reuse across A→B→A cycle.
    """
    from tools.archivist import artifact_writer as aw
    from tools.archivist.vault_paths import VaultPaths

    (tmp_path / ".system").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".system" / "vault_config.json").write_text('{"layout_version": 2}', encoding="utf-8")

    writer = aw.ArtifactWriter(vault_paths=VaultPaths(tmp_path))
    meta = {"entity_type": "concept", "title": "ABA Cycle Test"}

    # Write A
    rev_a1 = writer.write_note(meta, "Body content A", filename="ABATest")
    assert rev_a1.revision == 1

    # Write B
    rev_b = writer.write_note(meta, "Body content B — different", filename="ABATest")
    assert rev_b.revision == 2
    assert rev_b.revision_id != rev_a1.revision_id, "B must have a different revision_id than A"

    # Write A again (same as first)
    rev_a2 = writer.write_note(meta, "Body content A", filename="ABATest")

    # Invariant: A→B→A must be revision 3, NOT a reuse of revision 1
    assert rev_a2.revision == 3, (
        f"A→B→A must produce revision=3, got revision={rev_a2.revision}"
    )
    assert rev_a2.revision_id != rev_a1.revision_id, (
        f"A→B→A third write must get a NEW revision_id, not reuse revision 1's id: {rev_a1.revision_id}"
    )
    assert rev_a2.revision_id != rev_b.revision_id, (
        "Third write revision_id must also differ from B's revision_id"
    )
    assert rev_a2.is_reused is False, "A→B→A third write must not be marked as reused"
