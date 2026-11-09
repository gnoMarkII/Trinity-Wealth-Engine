"""Tests for Obsidian Vault V2 Audit Engine, Fixtures, and Streaming Benchmark Generator."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tests.tools.archivist.vault_v2_fixtures import create_comprehensive_test_vault
from tools.archivist.vault_audit import scan_vault, write_audit_report
from tools.archivist.vault_benchmark import generate_corpus
from tools.archivist.vault_v2_cli import main as cli_main


def _collect_vault_hashes(root: Path) -> dict[str, str]:
    """Computes SHA-256 for every file in root, keyed by relative POSIX path."""
    hashes: dict[str, str] = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            hasher = hashlib.sha256()
            with p.open("rb") as f:
                while chunk := f.read(65536):
                    hasher.update(chunk)
            hashes[p.relative_to(root).as_posix()] = hasher.hexdigest()
    return hashes


def test_audit_zero_mutation(tmp_path: Path) -> None:
    """Audit must NEVER modify, move, or delete any file in the scanned vault."""
    vault_dir = tmp_path / "vault"
    output_dir = tmp_path / "audit_output"

    # Setup synthetic fixture vault
    create_comprehensive_test_vault(vault_dir)

    # 1. Snapshot hashes before audit
    hashes_before = _collect_vault_hashes(vault_dir)
    assert len(hashes_before) > 0

    # 2. Run scan and write audit report
    result = scan_vault(vault_dir)
    inv_file, audit_file = write_audit_report(result, output_dir)

    # 3. Snapshot hashes after audit
    hashes_after = _collect_vault_hashes(vault_dir)

    # Assert 100% identical hashes and file counts
    assert hashes_before == hashes_after
    assert inv_file.exists()
    assert audit_file.exists()


def test_audit_inventory_contents(tmp_path: Path) -> None:
    """Verifies all files are recorded, excluded directories are flagged, and companions detected."""
    vault_dir = tmp_path / "vault"
    create_comprehensive_test_vault(vault_dir)

    result = scan_vault(vault_dir)

    assert result.total_files > 15
    assert result.active_files_count > 0
    assert result.excluded_files_count >= 3  # .obsidian, .trash, .pre_migration_backup...

    inv_map = {r.relative_path: r for r in result.inventory}

    # Verify excluded files
    assert ".obsidian/app.json" in inv_map
    assert inv_map[".obsidian/app.json"].is_excluded is True

    assert ".trash/Old Note.md" in inv_map
    assert inv_map[".trash/Old Note.md"].is_excluded is True

    assert ".pre_migration_backup_20_Portfolio_Management_v2/Portfolio_Holdings_Old.md" in inv_map
    assert inv_map[".pre_migration_backup_20_Portfolio_Management_v2/Portfolio_Holdings_Old.md"].is_excluded is True

    # Verify active files
    ftnt_hub_rel = "30_Knowledge_Base/Stocks/FTNT/FTNT.md"
    assert ftnt_hub_rel in inv_map
    hub_rec = inv_map[ftnt_hub_rel]
    assert hub_rec.is_excluded is False
    assert hub_rec.has_frontmatter is True
    assert hub_rec.properties.get("entity_type") == "stock_hub"
    assert hub_rec.properties.get("ticker") == "FTNT"
    assert "2026-09-05 FTNT Equity Analysis" in hub_rec.links

    # Verify sidecar detection
    ana_rel = "30_Knowledge_Base/Stocks/FTNT/Analysis/2026-09-05 FTNT Equity Analysis.md"
    assert ana_rel in inv_map
    ana_rec = inv_map[ana_rel]
    assert any("2026-09-05 FTNT Equity Analysis.json" in c for c in ana_rec.artifact_candidates)

    # Verify briefing companion detection
    briefing_rel = "30_Knowledge_Base/NotebookLM_Sources/2026/09/2026-09-06 Daily Briefing.md"
    assert briefing_rel in inv_map
    briefing_rec = inv_map[briefing_rel]
    assert any(".quality.json" in c for c in briefing_rec.artifact_candidates)


def test_audit_excludes_archive_revisions_from_active_knowledge(tmp_path: Path) -> None:
    vault_dir = tmp_path / "vault"
    active = vault_dir / "30_Knowledge_Base" / "Concepts" / "Current.md"
    frozen = vault_dir / "40_Archive" / "Revisions" / "note_1" / "rev_1" / "Current.md"
    active.parent.mkdir(parents=True)
    frozen.parent.mkdir(parents=True)
    active.write_text("# Current\n", encoding="utf-8")
    frozen.write_text("# Frozen\n", encoding="utf-8")

    result = scan_vault(vault_dir)
    records = {record.relative_path: record for record in result.inventory}
    assert records["30_Knowledge_Base/Concepts/Current.md"].is_excluded is False
    assert records["40_Archive/Revisions/note_1/rev_1/Current.md"].is_excluded is True


def test_audit_issues_detection(tmp_path: Path) -> None:
    """Verifies duplicate stems, broken links, malformed YAML, and missing metadata are detected."""
    vault_dir = tmp_path / "vault"
    create_comprehensive_test_vault(vault_dir)

    result = scan_vault(vault_dir)

    issue_types = {i.issue_type for i in result.issues}

    # 1. Duplicate filename detection
    assert "duplicate_filename" in issue_types
    dup_details = [i.details for i in result.issues if i.issue_type == "duplicate_filename"]
    assert any("ftnt" in d.lower() for d in dup_details)
    assert any("macro direction" in d.lower() for d in dup_details)

    # 2. Broken link detection
    assert "broken_link" in issue_types
    broken_links = [i.details for i in result.issues if i.issue_type == "broken_link"]
    assert any("Completely_Missing_File_12345" in b for b in broken_links)

    # 3. Malformed YAML parse error
    assert "parse_error" in issue_types
    parse_errors = [i for i in result.issues if i.issue_type == "parse_error"]
    assert any("Malformed Note.md" in pe.relative_path for pe in parse_errors)

    # 4. Missing metadata (info level)
    assert "missing_metadata" in issue_types
    missing_meta = [i for i in result.issues if i.issue_type == "missing_metadata"]
    # The Concept FTNT.md only has title: FTNT Concept, missing note_id, date, entity_type
    assert any("Concepts/FTNT.md" in mm.relative_path for mm in missing_meta)


def test_benchmark_streaming_memory_budget(tmp_path: Path) -> None:
    """Streaming synthetic corpus generator must stay strictly within the 64 MiB RAM budget."""
    out_dir = tmp_path / "synthetic_corpus"
    
    # Generate 1,000 synthetic notes
    meta = generate_corpus(out_dir, count=1000, seed=123)

    assert meta["count"] == 1000
    assert meta["peak_memory_overhead_mib"] < 64.0  # Must be < 64 MiB
    assert (out_dir / "corpus_manifest.json").exists()

    # Verify generated notes exist on disk and have valid structure
    md_files = list(out_dir.rglob("*.md"))
    assert len(md_files) == 1000


def test_cli_audit_execution(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """End-to-end CLI execution for vault audit subcommand."""
    vault_dir = tmp_path / "vault"
    output_dir = tmp_path / "baseline_out"
    create_comprehensive_test_vault(vault_dir)

    exit_code = cli_main(["audit", "--vault", str(vault_dir), "--output", str(output_dir)])
    assert exit_code == 0

    # Ensure output files exist and are populated
    inv_file = output_dir / "inventory.json"
    audit_file = output_dir / "audit.md"

    assert inv_file.exists()
    assert audit_file.exists()

    with inv_file.open("r", encoding="utf-8") as f:
        inv_data = json.load(f)
    assert inv_data["total_files"] > 10
    assert "inventory" in inv_data

    audit_text = audit_file.read_text(encoding="utf-8")
    assert "# Vault V2 Audit Report" in audit_text
    assert "File Breakdown by Extension" in audit_text


def test_cli_extended_subcommands_end_to_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Exercises newly wired vault_v2_cli subcommands on an isolated temp vault."""
    vault_dir = tmp_path / "vault"
    runtime_base = tmp_path / "runtime_base"
    create_comprehensive_test_vault(vault_dir)

    # Clear test-mode maintenance locks if any, and ensure writes target tmp_path only
    monkeypatch.setenv("VAULT_RUNTIME_BASE", str(runtime_base))

    # 1. nav-index
    assert cli_main(["nav-index", "--vault", str(vault_dir)]) == 0
    assert (vault_dir / "00_Index" / "Home.md").is_file()

    # 2. archive (dry-run and apply on dated news note)
    old_news = vault_dir / "30_Knowledge_Base" / "News" / "2020-01-01 Old Macro News.md"
    old_news.parent.mkdir(parents=True, exist_ok=True)
    old_news.write_text("# Old News\n", encoding="utf-8")
    assert cli_main(["archive", "--vault", str(vault_dir), "--max-age-days", "30"]) == 0
    assert old_news.exists()  # dry-run keeps file in place
    assert cli_main(["archive", "--vault", str(vault_dir), "--max-age-days", "30", "--apply"]) == 0
    assert not old_news.exists()
    assert (vault_dir / "40_Archive" / "News" / "2020" / old_news.name).is_file()

    # 3. query-notes (populates empty catalog if missing + resolves stock_hub path via link resolver)
    capsys.readouterr()
    assert cli_main(["query-notes", "--vault", str(vault_dir), "--wikilink", "[[FTNT]]"]) == 0
    q_out = json.loads(capsys.readouterr().out)
    assert q_out["resolved_wikilink"] == "30_Knowledge_Base/Stocks/FTNT/FTNT.md"

    # 4. maintain (runs against the initialized SQLite catalog)
    assert cli_main(["maintain", "--vault", str(vault_dir), "--no-vacuum"]) == 0
    m_out = json.loads(capsys.readouterr().out)
    assert "maintenance" in m_out and "orphaned_sidecars" in m_out

    # 5. reconcile (shadow scan)
    assert cli_main(["reconcile", "--vault", str(vault_dir), "--runtime-base", str(runtime_base)]) == 0
    r_out = json.loads(capsys.readouterr().out)
    assert "scanned" in r_out and "findings" in r_out

    # 6. runtime-migrate (dry-run pass)
    mig_report = tmp_path / "mig_report.json"
    assert (
        cli_main(
            [
                "runtime-migrate",
                "--vault",
                str(vault_dir),
                "--runtime-base",
                str(runtime_base),
                "--output",
                str(mig_report),
            ]
        )
        == 0
    )
    assert mig_report.is_file()
    mig_out = json.loads(capsys.readouterr().out)
    assert mig_out["status"] == "PASS"

    # 7. projection-rebuild
    proj_input = tmp_path / "proj_input.json"
    proj_input.write_text(
        json.dumps(
            {
                "source_state": {"nav": 100000.0},
                "projections": [
                    {
                        "entity_type": "holding",
                        "title": "PTT Projection",
                        "generated_text": "PTT 100 shares",
                        "stable_key": "PTT",
                        "metadata": {"document_key": "projection:default:PTT"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    assert (
        cli_main(
            [
                "projection-rebuild",
                "--vault",
                str(vault_dir),
                "--portfolio-id",
                "default",
                "--input-json",
                str(proj_input),
                "--runtime-base",
                str(runtime_base),
            ]
        )
        == 0
    )
    p_out = json.loads(capsys.readouterr().out)
    assert len(p_out) == 1
    assert p_out[0]["status"] in {"committed", "applied", "replayed"}

    # 8. concepts-cleanup argument validation + actual policy-only cleanup
    assert cli_main(["concepts-cleanup", "--vault", str(vault_dir)]) == 2
    from tools.archivist.concepts_cleanup import scan_concepts
    snap = scan_concepts(vault_dir)
    plan_file = tmp_path / "concepts_plan.json"
    plan_file.write_text(json.dumps(snap), encoding="utf-8")
    quarantine_dir = tmp_path / "quarantine"
    assert (
        cli_main(
            [
                "concepts-cleanup",
                "--vault",
                str(vault_dir),
                "--plan",
                str(plan_file),
                "--quarantine",
                str(quarantine_dir),
                "--run-id",
                "run_test_cli",
                "--policy-only",
            ]
        )
        == 0
    )
    c_out = json.loads(capsys.readouterr().out)
    assert c_out["status"] == "PASS"

    # 9. readiness-audit (write into tmp_path so repo files remain untouched)
    readiness_out = tmp_path / "readiness-decision.md"
    assert (
        cli_main(
            [
                "readiness-audit",
                "--repo-root",
                ".",
                "--output",
                str(readiness_out),
            ]
        )
        == 0
    )
    assert readiness_out.is_file()
    audit_summary = json.loads(capsys.readouterr().out)
    assert audit_summary["total"] == 10
    assert "sha256_checksum" in audit_summary
