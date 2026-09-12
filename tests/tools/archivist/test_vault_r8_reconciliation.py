from __future__ import annotations

from pathlib import Path

from application.knowledge.write_models import KnowledgeWriteCommand
from tools.archivist.human_edit_reconciler import HumanEditReconciler
from tools.archivist.managed_blocks import annotation_hash
from tools.archivist.metadata import dump_note, parse_note
from tools.archivist.vault_paths import VaultPaths
from tools.archivist.write_broker import KnowledgeWriteBroker


def _write_initial(tmp_path: Path):
    vault = tmp_path / "memories"
    broker = KnowledgeWriteBroker(vault_paths=VaultPaths(vault), runtime_root=tmp_path / "runtime", broker_id="r8")
    receipt = broker.submit(
        KnowledgeWriteCommand(
            operation="upsert_note",
            idempotency_key="reconcile-initial",
            producer="r8-test",
            payload={
                "metadata": {"schema_version": 2, "entity_type": "concept", "title": "Human edit"},
                "body": "Original body",
            },
        )
    )
    path = vault / str(receipt.relative_path)
    return vault, broker, receipt, path


def test_human_body_edit_is_imported_as_new_revision(tmp_path: Path) -> None:
    vault, broker, initial, path = _write_initial(tmp_path)
    metadata, _, issues = parse_note(path.read_text(encoding="utf-8"))
    assert not issues
    path.write_text(dump_note(metadata, "Human annotation that must survive"), encoding="utf-8")

    reconciler = HumanEditReconciler(vault_paths=VaultPaths(vault), broker=broker, runtime_root=tmp_path / "runtime")
    shadow = reconciler.reconcile_once()
    assert any(f.kind == "manual_edit" and f.status == "manual_edit" for f in shadow.findings)
    imported = reconciler.reconcile_once(write_enabled=True)
    finding = next(f for f in imported.findings if f.kind == "manual_edit")
    assert finding.status == "imported"
    assert finding.receipt and finding.receipt["revision_id"] != initial.revision_id
    assert "Human annotation that must survive" in path.read_text(encoding="utf-8")


def test_system_field_and_malformed_yaml_are_never_overwritten(tmp_path: Path) -> None:
    vault, broker, _, path = _write_initial(tmp_path)
    metadata, body, issues = parse_note(path.read_text(encoding="utf-8"))
    assert not issues
    metadata["revision"] = 999
    path.write_text(dump_note(metadata, body), encoding="utf-8")
    reconciler = HumanEditReconciler(vault_paths=VaultPaths(vault), broker=broker, runtime_root=tmp_path / "runtime")
    report = reconciler.reconcile_once(write_enabled=True)
    finding = next(f for f in report.findings if f.kind == "system_field_edit")
    assert finding.status == "conflict"
    assert "system-owned" in " ".join(finding.issues)

    path.write_text("---\ntitle: broken\n: invalid\n---\nBody", encoding="utf-8")
    malformed = reconciler.reconcile_once(write_enabled=True)
    assert any(f.kind == "malformed_yaml" and f.status == "conflict" for f in malformed.findings)
    assert "title: broken" in path.read_text(encoding="utf-8")


def test_managed_projection_preserves_human_annotation_hash() -> None:
    from tools.portfolio.projection_boundary import render_projection_body

    existing = (
        "<!-- managed:start portfolio-summary -->\nGenerated old\n"
        "<!-- managed:end portfolio-summary -->\n\n## Human Notes\nKeep this exact.\n"
    )
    before = annotation_hash(existing, "portfolio-summary")
    refreshed = render_projection_body("Generated new", existing_body=existing)
    assert "Generated new" in refreshed
    assert "Keep this exact." in refreshed
    assert annotation_hash(refreshed, "portfolio-summary") == before
