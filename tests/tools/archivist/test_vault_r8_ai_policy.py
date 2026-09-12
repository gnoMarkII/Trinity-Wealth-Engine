from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from application.knowledge.write_models import KnowledgeWriteCommand
from tools.archivist.ai_answer_contract import collect_evidence, validate_answer
from tools.archivist.catalog_adapter import SqliteNoteCatalogAdapter
from tools.archivist.catalog_runtime import catalog_runtime_root
from tools.archivist.schema_registry import load_default_registry
from tools.archivist.vault_paths import VaultPaths
from tools.archivist.write_broker import KnowledgeWriteBroker


def test_ai_policy_excludes_generated_and_restricted_from_primary_namespace(tmp_path: Path) -> None:
    vault = tmp_path / "memories"
    broker = KnowledgeWriteBroker(vault_paths=VaultPaths(vault), runtime_root=tmp_path / "runtime", broker_id="r8")
    published = broker.submit(
        KnowledgeWriteCommand(
            operation="upsert_note",
            idempotency_key="ai-published",
            producer="r8-test",
            payload={
                "metadata": {
                    "schema_version": 2,
                    "entity_type": "concept",
                    "title": "Published evidence",
                    "content_status": "published",
                    "sensitivity": "internal",
                    "production_eligible": True,
                    "trust_tier": "T1",
                    "source_verification_status": "verified",
                    "content_verification_status": "verified",
                },
                "body": "Grounded body",
            },
        )
    )
    generated = broker.submit(
        KnowledgeWriteCommand(
            operation="upsert_note",
            idempotency_key="ai-generated",
            producer="r8-test",
            payload={
                "metadata": {
                    "schema_version": 2,
                    "entity_type": "concept",
                    "title": "Generated view",
                    "content_status": "generated",
                    "sensitivity": "internal",
                },
                "body": "Do not retrieve",
            },
        )
    )
    restricted = broker.submit(
        KnowledgeWriteCommand(
            operation="upsert_note",
            idempotency_key="ai-restricted",
            producer="r8-test",
            payload={
                "metadata": {
                    "schema_version": 2,
                    "entity_type": "concept",
                    "title": "Restricted note",
                    "content_status": "published",
                    "sensitivity": "restricted",
                },
                "body": "Do not leak",
            },
        )
    )
    catalog = SqliteNoteCatalogAdapter(db_path=tmp_path / "runtime" / "catalog.db", vault_root=vault)
    catalog.sync_from_vault(vault)
    docs = [
        SimpleNamespace(metadata={"relative_path": published.relative_path}),
        SimpleNamespace(metadata={"relative_path": generated.relative_path}),
        SimpleNamespace(metadata={"relative_path": restricted.relative_path}),
    ]
    evidence = collect_evidence(vault, catalog, docs)
    assert [item["note_id"] for item in evidence] == [published.note_id]
    assert evidence[0]["revision_id"] == published.revision_id
    result = validate_answer("answer", evidence, cited_paths=[published.relative_path], production_mode=True)
    assert result["status"] == "PASS"


def test_registry_vector_policy_excludes_sensitive_lifecycle() -> None:
    registry = load_default_registry()
    assert registry.is_index_eligible({"entity_type": "concept", "search_scope": "included", "content_status": "published", "sensitivity": "internal"}, vector=True)
    for metadata in (
        {"entity_type": "concept", "search_scope": "included", "content_status": "generated", "sensitivity": "internal"},
        {"entity_type": "concept", "search_scope": "included", "content_status": "published", "sensitivity": "restricted"},
    ):
        assert registry.is_index_eligible(metadata, vector=True) is False
