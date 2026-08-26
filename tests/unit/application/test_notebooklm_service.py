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
