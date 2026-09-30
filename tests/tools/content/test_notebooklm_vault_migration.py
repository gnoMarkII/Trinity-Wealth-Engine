import json
from pathlib import Path
import pytest

from schemas.briefing_book_schemas import (
    PublishableBriefingResult,
    ResearchQualityReport,
)
from tests.fixtures.briefing_fixtures import (
    make_valid_briefing_draft,
    make_valid_evidence_bundle,
)
from tools.content.briefing_artifacts import save_briefing_artifact
from tools.content.notebooklm.adapters.filesystem import (
    FilesystemSourceCatalogAdapter,
    FilesystemManifestAdapter,
)
from tools.content.notebooklm import manifest


def test_save_briefing_artifact_uses_canonical_v2_routing(tmp_path):
    # R9 freezes all new canonical writes on the V2 path policy, even when a
    # legacy vault has no layout config yet.
    v1_vault = tmp_path / "v1_vault"
    v1_vault.mkdir()

    synthesis = PublishableBriefingResult(
        content="# Briefing V1",
        draft=make_valid_briefing_draft(),
        quality_report=ResearchQualityReport(score=100, status="pass", publishable=True),
        evidence_bundle=make_valid_evidence_bundle(),
    )

    art_v1 = save_briefing_artifact(synthesis, "V1 Report", vault_root=v1_vault, date_str="2026-09-06")
    assert "NotebookLM_Sources" in str(art_v1.path)
    assert art_v1.path.parent.name == "09"
    assert art_v1.path.parent.parent.name == "2026"
    assert art_v1.path.exists()
    assert art_v1.quality_path.exists()

    # Configured V2 vault follows the same canonical path.
    v2_vault = tmp_path / "v2_vault"
    (v2_vault / ".system").mkdir(parents=True)
    (v2_vault / ".system" / "vault_config.json").write_text(json.dumps({"layout_version": 2}), encoding="utf-8")

    synthesis_v2 = PublishableBriefingResult(
        content="# Briefing V2",
        draft=make_valid_briefing_draft(),
        quality_report=ResearchQualityReport(score=100, status="pass", publishable=True),
        evidence_bundle=make_valid_evidence_bundle(),
    )

    art_v2 = save_briefing_artifact(synthesis_v2, "V2 Report", vault_root=v2_vault, date_str="2026-09-06")
    assert "NotebookLM_Sources" in str(art_v2.path)
    assert art_v2.path.parent.name == "09"
    assert art_v2.path.parent.parent.name == "2026"
    assert art_v2.path.parent.parent.parent.name == "NotebookLM_Sources"
    assert art_v2.path.exists()
    assert art_v2.quality_path.exists()


def test_save_briefing_artifact_unverified_draft_maps_trust_tier_to_t3(tmp_path):
    from schemas.briefing_book_schemas import (
        UnverifiedBriefingDraftResult,
        UnverifiedDraftOverrideAudit,
        QualityIssueRecord,
    )
    from tools.archivist.metadata import parse_note

    vault = tmp_path / "vault"
    draft_result = UnverifiedBriefingDraftResult(
        content="# Unverified Draft Content\n\nBody details",
        draft=make_valid_briefing_draft(),
        quality_report=ResearchQualityReport(
            score=70,
            status="degraded",
            publishable=False,
            issues=[
                QualityIssueRecord(
                    code="SRC_UNVERIFIED",
                    category="provenance",
                    severity="blocker",
                    description="Missing news source",
                    bypassable=True,
                )
            ],
        ),
        evidence_bundle=make_valid_evidence_bundle(),
        override_audit=UnverifiedDraftOverrideAudit(
            job_id="job-123",
            thread_id="th-123",
            pitch_id="p-001",
            policy_version="v1",
            reason="User override incomplete provenance",
            server_timestamp="2026-09-06T12:00:00",
            token_hash="fake_token_hash",
            source_readiness_snapshot=["SRC_UNVERIFIED"],
        ),
    )

    art = save_briefing_artifact(draft_result, "Draft Report", vault_root=vault, date_str="2026-09-06")
    assert art.path.exists()
    assert art.quality_path.exists()

    meta, body, issues = parse_note(art.path.read_text(encoding="utf-8"))
    assert meta.get("trust_tier") == "T3"
    assert meta.get("production_eligible") is False
    assert meta.get("artifact_status") == "unverified_draft"
    assert "Unverified Draft Content" in body


def test_filesystem_source_catalog_discovers_nested_and_flat(tmp_path):
    sources_dir = tmp_path / "NotebookLM_Sources"
    sources_dir.mkdir()

    # Legacy flat file
    flat_file = sources_dir / "2026-08-01_Old Briefing_pitch-1_rev1_abcd1234_verified.md"
    flat_file.write_text("# Old Briefing", encoding="utf-8")

    # V2 nested file
    nested_dir = sources_dir / "2026" / "09"
    nested_dir.mkdir(parents=True)
    nested_file = nested_dir / "2026-09-06_New Briefing_pitch-2_rev1_efgh5678_verified.md"
    nested_file.write_text("# New Briefing", encoding="utf-8")

    # Ignored outbox / quarantine / hidden file
    outbox_dir = sources_dir / "outbox"
    outbox_dir.mkdir()
    (outbox_dir / "parking.md").write_text("Ignored", encoding="utf-8")

    catalog = FilesystemSourceCatalogAdapter(sources_dir)
    items = catalog.list_sources()

    assert len(items) == 2
    paths = [Path(item["file_path"]).resolve() for item in items]
    assert flat_file.resolve() in paths
    assert nested_file.resolve() in paths

    # Test resolve_source
    resolved_flat = catalog.resolve_source(str(flat_file))
    assert resolved_flat["file_path"] == str(flat_file.resolve())

    resolved_nested = catalog.resolve_source(str(nested_file))
    assert resolved_nested["file_path"] == str(nested_file.resolve())

    # Rejection outside sources_dir
    outside_file = tmp_path / "outside.md"
    outside_file.write_text("outside", encoding="utf-8")
    with pytest.raises(ValueError):
        catalog.resolve_source(str(outside_file))


def test_manifest_continuity_across_relocation(tmp_path, monkeypatch):
    monkeypatch.setattr(manifest, "MANIFEST_DIR", tmp_path / "manifests")

    # Original file in flat directory
    flat_dir = tmp_path / "sources"
    flat_dir.mkdir()
    source_file = flat_dir / "briefing.md"
    content = "# Important Briefing Content"
    source_file.write_text(content, encoding="utf-8")

    # Compute hash and save initial manifest
    c_hash = manifest.compute_content_hash(source_file)
    mf = manifest.new_manifest(content_hash=c_hash, briefing_path=source_file)
    mf.notebook_id = "nb-12345"
    mf.status = "notebook_created"
    manifest.save_manifest(mf, base_dir=tmp_path / "manifests")

    # Verify adapter loads manifest
    adapter = FilesystemManifestAdapter()
    loaded_mf = adapter.get_for_source(str(source_file))
    assert loaded_mf is not None
    assert loaded_mf.notebook_id == "nb-12345"

    # Now simulate file moved to YYYY/MM nested location
    nested_dir = flat_dir / "2026" / "09"
    nested_dir.mkdir(parents=True)
    moved_file = nested_dir / "briefing.md"
    moved_file.write_text(content, encoding="utf-8")

    # Manifest lookup on moved file should STILL find the exact same manifest!
    loaded_moved_mf = adapter.get_for_source(str(moved_file))
    assert loaded_moved_mf is not None
    assert loaded_moved_mf.notebook_id == "nb-12345"
    assert loaded_moved_mf.content_hash == c_hash
