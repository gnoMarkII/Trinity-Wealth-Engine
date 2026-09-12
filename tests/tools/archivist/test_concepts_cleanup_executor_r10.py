from __future__ import annotations

import json
from pathlib import Path

from tools.archivist.concepts_cleanup import build_cleanup_plan, scan_concepts
from tools.archivist.concepts_cleanup_executor import apply_cleanup, rollback_cleanup


def test_cleanup_apply_is_quarantined_and_rollback_restores_preimage(tmp_path: Path) -> None:
    vault = tmp_path / "memories"
    concept = vault / "30_Knowledge_Base" / "Concepts" / "Unused.md"
    concept.parent.mkdir(parents=True)
    concept.write_text(
        "---\ntitle: Unused\nentity_type: concept\nnote_id: note-unused\n"
        "document_key: concept:unused\nsearch_scope: excluded\nlifecycle_status: stub\n---\n\n"
        "<!-- Concept Stub created automatically -->\n",
        encoding="utf-8",
    )
    snapshot = scan_concepts(vault)
    plan = build_cleanup_plan(snapshot)
    plan_path = tmp_path / "cleanup-plan.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    quarantine_root = tmp_path / "quarantine"

    result = apply_cleanup(
        vault,
        plan_path,
        quarantine_root,
        run_id="r10-test",
        owner="r10-test",
    )

    assert result["status"] == "PASS"
    manifest_path = quarantine_root / "r10-test" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "APPLIED"
    assert not concept.exists()
    assert (quarantine_root / "r10-test" / "files" / concept.relative_to(vault)).is_file()

    rollback = rollback_cleanup(vault, manifest_path, owner="r10-test-rollback")
    assert rollback["status"] == "PASS"
    assert concept.is_file()
    assert "Concept Stub created automatically" in concept.read_text(encoding="utf-8")


def test_cleanup_converts_active_stub_links_to_plain_text_before_retire(tmp_path: Path) -> None:
    vault = tmp_path / "memories"
    concept = vault / "30_Knowledge_Base" / "Concepts" / "Unused.md"
    source = vault / "30_Knowledge_Base" / "News" / "Source.md"
    concept.parent.mkdir(parents=True)
    source.parent.mkdir(parents=True)
    concept.write_text(
        "---\ntitle: Unused\nentity_type: concept\nnote_id: note-unused\n"
        "document_key: concept:unused\nsearch_scope: excluded\nlifecycle_status: stub\n---\n\n"
        "<!-- Concept Stub created automatically -->\n",
        encoding="utf-8",
    )
    source.write_text(
        "---\ntitle: Source\nentity_type: company_news\nnote_id: note-source\n"
        "document_key: news:source\nsearch_scope: included\ndate: 2026-09-12\n---\n\n"
        "See [Unused](../Concepts/Unused.md) for context.\n",
        encoding="utf-8",
    )
    snapshot = scan_concepts(vault)
    plan = build_cleanup_plan(snapshot)
    assert plan["disposition_counts"] == {"RETIRE": 1}
    plan_path = tmp_path / "cleanup-plan.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")

    result = apply_cleanup(vault, plan_path, tmp_path / "quarantine", run_id="r10-link-retire", owner="r10-test")

    assert result["status"] == "PASS"
    assert not concept.exists()
    updated_source = source.read_text(encoding="utf-8")
    assert "See Unused for context." in updated_source
    assert "../Concepts/Unused.md" not in updated_source

    manifest_path = tmp_path / "quarantine" / "r10-link-retire" / "manifest.json"
    rollback = rollback_cleanup(vault, manifest_path, owner="r10-test-rollback")
    assert rollback["status"] == "PASS"
    assert concept.is_file()
    assert "[Unused](../Concepts/Unused.md)" in source.read_text(encoding="utf-8")
