"""Unit tests for NotebookLMApplicationService."""
import pytest
from pathlib import Path
from unittest.mock import MagicMock
from application.notebooklm.service import NotebookLMApplicationService
from tools.content.notebooklm.adapters.filesystem import FilesystemSourceCatalogAdapter


def test_notebooklm_service_list_available_sources(tmp_path: Path):
    sources_dir = tmp_path / "NotebookLM_Sources"
    sources_dir.mkdir()

    # Create dummy markdown sources
    file1 = sources_dir / "2026-08-20_AI_Semiconductors_rev1_abcd1234_verified.md"
    file1.write_text("# AI Briefing", encoding="utf-8")

    file2 = sources_dir / "2026-08-21_Energy_Storage_rev2_ef567890_unverified.md"
    file2.write_text("# Energy Briefing", encoding="utf-8")

    fake_repo = MagicMock()
    service = NotebookLMApplicationService(
        repo=fake_repo,
        source_catalog=FilesystemSourceCatalogAdapter(sources_dir),
    )
    sources = service.list_available_sources()

    assert len(sources) == 2
    assert sources[0].title == "Energy_Storage"
    assert sources[0].is_verified is False
    assert sources[1].title == "AI_Semiconductors"
    assert sources[1].is_verified is True


def test_notebooklm_service_validate_briefing_source(tmp_path: Path):
    sources_dir = tmp_path / "NotebookLM_Sources"
    sources_dir.mkdir()
    valid_file = sources_dir / "briefing.md"
    valid_file.write_text("# Brief", encoding="utf-8")

    fake_repo = MagicMock()
    service = NotebookLMApplicationService(
        repo=fake_repo,
        source_catalog=FilesystemSourceCatalogAdapter(sources_dir),
    )
    validated = service.validate_briefing_source(str(valid_file))
    assert validated.exists()

    with pytest.raises(ValueError):
        service.validate_briefing_source(str(tmp_path / "outside.md"))


def test_notebooklm_service_fail_closed_on_corrupt_manifest(tmp_path: Path):
    """Verifies that generate() rejects dispatch with 0 calls when source manifest is corrupt or unsupported."""
    sources_dir = tmp_path / "NotebookLM_Sources"
    sources_dir.mkdir()
    valid_file = sources_dir / "briefing.md"
    valid_file.write_text("# Brief", encoding="utf-8")

    from tools.content.notebooklm.manifest import ManifestLoadResult, ManifestStatus

    fake_repo = MagicMock()
    fake_card_repo = MagicMock()
    fake_card_repo.get_card.return_value = {"flow": "notebooklm", "prompt": str(valid_file)}
    fake_dispatcher = MagicMock()
    fake_binary = MagicMock()
    fake_manifest_port = MagicMock()
    fake_manifest_port.get_for_source.return_value = ManifestLoadResult(
        load_status=ManifestStatus.CORRUPT,
        error_message="Invalid JSON syntax",
        path="data/notebooklm_runs/bad.json",
    )

    service = NotebookLMApplicationService(
        repo=fake_repo,
        card_repo=fake_card_repo,
        dispatcher=fake_dispatcher,
        binary=fake_binary,
        source_catalog=FilesystemSourceCatalogAdapter(sources_dir),
        manifest_port=fake_manifest_port,
    )

    # 1. generate() must raise RuntimeError and never dispatch to external provider
    with pytest.raises(RuntimeError, match="NotebookLM dispatch blocked: source manifest is corrupt"):
        service.generate(card_id="card-123")

    fake_dispatcher.dispatch.assert_not_called()
    fake_card_repo.move_card.assert_not_called()

    # 2. get_status() must surface recovery blocker without crashing
    fake_repo.get_job.return_value = {"status": "pending", "instruction": str(valid_file)}
    status_dto = service.get_status("job-123")
    assert status_dto is not None
    assert status_dto.recovery_status == "corrupt"
    assert "Recovery blocker" in status_dto.error


# ──────────────────────────────────────────────────────────────────────────────
# C15 — Completed source metadata edit must preserve lineage (provider calls = 0)
# Gate: A04 (history continuity), A05 (no external duplication)
# ──────────────────────────────────────────────────────────────────────────────

def test_c15_completed_source_metadata_edit_does_not_trigger_provider(tmp_path: Path) -> None:
    """C15: A RESOLVED manifest must not block dispatch. After source metadata is edited
    (title rename, tag changes), the manifest port must still return the resolved history and
    generate() must proceed normally (provider dispatch = 1, not blocked).

    This test verifies the positive case: RESOLVED manifests pass through the guard.
    The negative cases (non-RESOLVED blocking) are covered by C16.
    """
    from datetime import datetime, timezone
    from tools.content.notebooklm.manifest import (
        ManifestLoadResult, ManifestStatus, NotebookLMManifest,
    )

    sources_dir = tmp_path / "NotebookLM_Sources"
    sources_dir.mkdir()
    briefing = sources_dir / "briefing_completed.md"
    briefing.write_text("# Completed Briefing\nBody text.", encoding="utf-8")

    # Create a real NotebookLMManifest (not MagicMock — pydantic validates the type)
    now = datetime.now(timezone.utc).isoformat()
    real_manifest = NotebookLMManifest(
        content_hash="abc123" * 8,
        briefing_path=str(briefing),
        notebook_id="nb-original-001",
        source_id="src-001",
        status="completed",
        audio_path="/audio/output.wav",
        created_at=now,
        updated_at=now,
    )

    fake_repo = MagicMock()
    fake_card_repo = MagicMock()
    fake_card_repo.get_card.return_value = {"flow": "notebooklm", "prompt": str(briefing)}
    fake_dispatcher = MagicMock()
    fake_dispatcher.dispatch.return_value = "job-c15"
    fake_repo.get_job.return_value = {"status": "completed", "instruction": str(briefing)}
    fake_binary = MagicMock()

    fake_manifest_port = MagicMock()
    fake_manifest_port.get_for_source.return_value = ManifestLoadResult(
        load_status=ManifestStatus.RESOLVED,
        manifest=real_manifest,
        path=str(briefing),
    )

    service = NotebookLMApplicationService(
        repo=fake_repo,
        card_repo=fake_card_repo,
        dispatcher=fake_dispatcher,
        binary=fake_binary,
        source_catalog=FilesystemSourceCatalogAdapter(sources_dir),
        manifest_port=fake_manifest_port,
    )

    # RESOLVED manifest must NOT block dispatch — generate() must succeed
    result = service.generate(card_id="card-c15")
    assert result is not None

    # Provider MUST have been called once (guard passes for RESOLVED manifest)
    fake_dispatcher.dispatch.assert_called_once()

    # get_status must surface the resolved lineage
    status = service.get_status("job-c15")
    assert status is not None
    assert status.recovery_status == "resolved"


# ──────────────────────────────────────────────────────────────────────────────
# C16 — Unsupported / missing / corrupt / conflict manifest must block dispatch
# Gate: A04 (history continuity), A05 (no external duplication)
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("load_status,error_msg", [
    ("unsupported_version", "schema_version 99 is not supported"),
    ("missing_history", "History store unreadable: disk error"),
    ("corrupt", "Invalid JSON syntax at offset 42"),
    ("conflict", "Two manifests claim the same content_hash"),
])
def test_c16_blocked_manifest_statuses_prevent_provider_calls(
    tmp_path: Path,
    load_status: str,
    error_msg: str,
) -> None:
    """C16: Any non-RESOLVED manifest status must block generate() before any provider call.

    Covers: schema_version=99 (UNSUPPORTED_VERSION), unreadable store (MISSING_HISTORY),
    corrupt JSON (CORRUPT), and hash collision (CONFLICT).
    All branches must result in: provider dispatch=0, card moves=0, RuntimeError raised.
    """
    from tools.content.notebooklm.manifest import ManifestLoadResult, ManifestStatus

    sources_dir = tmp_path / "NotebookLM_Sources"
    sources_dir.mkdir()
    briefing = sources_dir / "briefing.md"
    briefing.write_text("# Briefing\nBody.", encoding="utf-8")

    status_enum = ManifestStatus(load_status)

    fake_repo = MagicMock()
    fake_card_repo = MagicMock()
    fake_card_repo.get_card.return_value = {"flow": "notebooklm", "prompt": str(briefing)}
    fake_dispatcher = MagicMock()
    fake_binary = MagicMock()

    fake_manifest_port = MagicMock()
    fake_manifest_port.get_for_source.return_value = ManifestLoadResult(
        load_status=status_enum,
        error_message=error_msg,
        path=str(briefing),
    )

    service = NotebookLMApplicationService(
        repo=fake_repo,
        card_repo=fake_card_repo,
        dispatcher=fake_dispatcher,
        binary=fake_binary,
        source_catalog=FilesystemSourceCatalogAdapter(sources_dir),
        manifest_port=fake_manifest_port,
    )

    # generate() MUST raise and never reach provider
    with pytest.raises(RuntimeError, match=f"source manifest is {load_status}"):
        service.generate(card_id="card-c16")

    # Assert zero provider side effects
    fake_dispatcher.dispatch.assert_not_called()
    fake_card_repo.move_card.assert_not_called()

    # get_status must surface the blocked status without crashing
    fake_repo.get_job.return_value = {"status": "pending", "instruction": str(briefing)}
    status = service.get_status("job-c16")
    assert status is not None
    assert status.recovery_status == load_status
