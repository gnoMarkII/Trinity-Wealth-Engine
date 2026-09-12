"""Incremental, manifest-backed vector indexing for the Vault V2 read model.

Indexing is deliberately a worker concern. Search only opens the active
validated generation and performs a query embedding; it never calls this
module's catalog sync, outbox, mutation, or state persistence paths.
"""
from __future__ import annotations

import json
import os
import hashlib
from pathlib import Path
from typing import Any, Callable, Optional, Union

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.logger import get_logger
from tools.archivist.catalog_adapter import SqliteNoteCatalogAdapter
from tools.archivist.maintenance_guard import assert_write_allowed
from tools.archivist.parser import _strip_frontmatter
from tools.archivist.schema_registry import SchemaRegistry, load_default_registry
from tools.archivist.vault_paths import VaultPaths
from tools.archivist.vector_generation import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_CHUNKER_CONFIG,
    DEFAULT_CHUNKER_VERSION,
    VectorGenerationError,
    activate_manifest,
    build_manifest,
    generation_id_for,
    load_active_manifest,
    save_manifest,
    describe_chunker,
    describe_embeddings,
    vector_runtime_path,
)

logger = get_logger(__name__)


class FakeEmbeddings:
    """Deterministic fast embedding provider for tests and benchmarks only."""

    def __init__(self, size: int = 384) -> None:
        self.size = size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        import hashlib

        h = hashlib.sha256(text.encode("utf-8")).digest()
        vec = [((h[i % len(h)] + i) % 100) / 100.0 for i in range(self.size)]
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        return [x / norm for x in vec]


class IndexingWorker:
    """Synchronize the catalog into a bounded, recoverable Chroma generation."""

    MAX_BATCH_BYTES = 2 * 1024 * 1024

    def __init__(
        self,
        catalog: SqliteNoteCatalogAdapter,
        vault_root: Optional[Union[str, Path]] = None,
        chroma_dir: Optional[Union[str, Path]] = None,
        embeddings: Optional[Any] = None,
        sync_catalog: bool = True,
    ) -> None:
        self._catalog = catalog
        vp = VaultPaths(vault_root)
        self._vault_root = vp.root
        assert_write_allowed(self._vault_root)
        self._runtime_root = (
            Path(chroma_dir).resolve()
            if chroma_dir
            else vector_runtime_path(self._vault_root)
        )
        self._runtime_root.mkdir(parents=True, exist_ok=True)
        self._chroma_dir = self._runtime_root
        self._state_file = self._runtime_root / "vector_index_state.json"
        self._embeddings = embeddings
        self._sync_catalog = bool(sync_catalog)
        self._schema_registry: SchemaRegistry = load_default_registry()
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(DEFAULT_CHUNKER_CONFIG["chunk_size"]),
            chunk_overlap=int(DEFAULT_CHUNKER_CONFIG["chunk_overlap"]),
            separators=list(DEFAULT_CHUNKER_CONFIG["separators"]),
        )

    def _load_state(self) -> dict[str, Any]:
        if not self._state_file.exists():
            return {"version": 2, "indexed_files": {}}
        try:
            data = json.loads(self._state_file.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("vector index state must be an object")
            indexed = data.get("indexed_files", {})
            if not isinstance(indexed, dict):
                raise ValueError("vector index indexed_files must be an object")
            data["indexed_files"] = {str(k): str(v) for k, v in indexed.items()}
            return data
        except Exception as exc:
            logger.warning("Vector index state is unreadable; rebuilding deltas: %s", exc)
            return {"version": 2, "indexed_files": {}, "state_error": str(exc)}

    def _load_indexed_hashes(self) -> dict[str, str]:
        return dict(self._load_state().get("indexed_files", {}))

    def _save_index_state(
        self,
        indexed_files: dict[str, str],
        *,
        indexed_chunks: Optional[dict[str, dict[str, Any]]] = None,
        generation: Optional[dict[str, Any]] = None,
    ) -> None:
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        state: dict[str, Any] = {
            "version": 2,
            "indexed_files": dict(indexed_files),
            "indexed_chunks": dict(indexed_chunks or {}),
        }
        if generation:
            state.update(
                {
                    "generation_id": generation.get("generation_id"),
                    "collection_name": generation.get("collection_name"),
                    "model_fingerprint": generation.get("model_fingerprint"),
                    "chunker_fingerprint": generation.get("chunker_fingerprint"),
                    "registry_digest": generation.get("registry_digest"),
                    "policy_digest": generation.get("policy_digest"),
                    "corpus_fingerprint": generation.get("corpus_fingerprint"),
                    "eligible_note_count": generation.get("eligible_note_count", 0),
                    "eligible_chunk_count": generation.get("eligible_chunk_count", 0),
                }
            )
        tmp = self._state_file.with_suffix(self._state_file.suffix + ".tmp")
        tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._state_file)

    def _iter_catalog_entries(self):
        def eligible(entry):
            try:
                metadata = json.loads(str(getattr(entry, "metadata_json", "{}") or "{}"))
            except (TypeError, ValueError, json.JSONDecodeError):
                return False
            return self._schema_registry.is_index_eligible(metadata, vector=True)

        iterator = getattr(self._catalog, "iter_notes", None)
        if iterator is not None:
            for entry in iterator(page_size=500):
                if eligible(entry):
                    yield entry
            return
        # Compatibility with a small test double or an older adapter. Keep the
        # fallback bounded and avoid the former 1,000,000-row materialization.
        offset = 0
        while True:
            page = self._catalog.find_notes(limit=500, offset=offset)
            if not page:
                return
            for entry in page:
                if eligible(entry):
                    yield entry
            if len(page) < 500:
                return
            offset += len(page)

    def _resolve_embeddings(self) -> Any:
        embeddings = self._embeddings
        if embeddings is None:
            from tools.archivist.search import get_embeddings

            embeddings = get_embeddings()
        is_test_env = os.getenv("VAULT_ALLOW_TEST_EMBEDDINGS") == "1" or "PYTEST_CURRENT_TEST" in os.environ
        if isinstance(embeddings, FakeEmbeddings) and not is_test_env:
            raise RuntimeError(
                "FakeEmbeddings is prohibited in production vector indexing. "
                "Use production embeddings or export VAULT_ALLOW_TEST_EMBEDDINGS=1 for unit benchmarks."
            )
        return embeddings

    def _generation_for(self, embeddings: Any, entries) -> dict[str, Any]:
        model = describe_embeddings(embeddings)
        chunker = describe_chunker(version=DEFAULT_CHUNKER_VERSION, config=DEFAULT_CHUNKER_CONFIG)
        try:
            active = load_active_manifest(self._vault_root)
        except VectorGenerationError as exc:
            logger.warning("Active vector generation is unavailable; preparing a replacement: %s", exc)
            active = None

        # Any non-no-op build is a new immutable generation, even when the
        # model/chunker are compatible with the active one.  Reusing the active
        # manifest path would turn a failed build into a failed active pointer.
        base_generation_id = generation_id_for(model, chunker, DEFAULT_COLLECTION_NAME)
        if active is None:
            generation_id = base_generation_id
            collection_name = DEFAULT_COLLECTION_NAME
        else:
            nonce = hashlib.sha256(os.urandom(16)).hexdigest()[:8]
            generation_id = f"{base_generation_id}_{nonce}"
            collection_name = f"{DEFAULT_COLLECTION_NAME}_{nonce}"

        return build_manifest(
            vault_root=self._vault_root,
            embeddings=embeddings,
            collection_name=collection_name,
            chunker_version=DEFAULT_CHUNKER_VERSION,
            chunker_config=DEFAULT_CHUNKER_CONFIG,
            corpus_scope="active_searchable_notes",
            corpus_entries=entries,
            eligible_chunk_count=0,
            status="building",
            generation_id=generation_id,
            dimension=getattr(embeddings, "size", None),
        )

    def _add_bounded(
        self,
        vectorstore: Any,
        documents: list[Document],
        document_ids: list[str],
        *,
        batch_size: int,
    ) -> None:
        """Add documents while respecting both chunk and UTF-8 byte limits."""
        batch: list[Document] = []
        batch_ids: list[str] = []
        byte_count = 0
        for document, document_id in zip(documents, document_ids):
            doc_bytes = len(document.page_content.encode("utf-8"))
            if batch and (len(batch) >= batch_size or byte_count + doc_bytes > self.MAX_BATCH_BYTES):
                vectorstore.add_documents(batch, ids=batch_ids)
                batch = []
                batch_ids = []
                byte_count = 0
            batch.append(document)
            batch_ids.append(document_id)
            byte_count += doc_bytes
        if batch:
            vectorstore.add_documents(batch, ids=batch_ids)

    def sync_index(
        self,
        batch_size: int = 128,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        limit: Optional[int] = None,
    ) -> dict[str, int]:
        """Synchronize catalog entries into a validated active generation."""
        assert_write_allowed(self._vault_root)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        from langchain_chroma import Chroma

        # Catalog reconciliation is worker-owned maintenance.  Published R5
        # catalog generations are immutable; callers that already rebuilt and
        # activated a generation pass a read-only catalog with sync_catalog=False.
        if self._sync_catalog:
            self._catalog.sync_from_vault(self._vault_root)

        state_payload = self._load_state()
        indexed_state = dict(state_payload.get("indexed_files", {}))
        indexed_chunks: dict[str, dict[str, Any]] = {
            str(path): value
            for path, value in (state_payload.get("indexed_chunks", {}) or {}).items()
            if isinstance(value, dict)
        }
        # First pass is metadata-only and streaming.  Keep only the indexed
        # path keys needed to detect deletions; note rows themselves are
        # revisited from the catalog after the no-op decision.
        current_count = 0
        changed_count = 0
        present_indexed_paths: set[str] = set()
        for entry in self._iter_catalog_entries():
            rel = str(entry.relative_path).replace("\\", "/")
            current_count += 1
            if rel in indexed_state:
                present_indexed_paths.add(rel)
            if indexed_state.get(rel) != entry.content_sha256:
                changed_count += 1
        to_delete_paths = [rel for rel in indexed_state if rel not in present_indexed_paths]
        work_limit = max(0, limit) if limit is not None else None
        total_work = min(changed_count, work_limit) if work_limit is not None else changed_count

        # Nothing changed and a valid active pointer exists: this is a true
        # no-op, so avoid loading the model or opening Chroma.
        try:
            active = load_active_manifest(self._vault_root)
        except VectorGenerationError:
            active = None
        if total_work == 0 and not to_delete_paths and active is not None:
            return {"added": 0, "updated": 0, "deleted": 0, "total_indexed": len(indexed_state)}
        if current_count == 0 and not to_delete_paths:
            return {"added": 0, "updated": 0, "deleted": 0, "total_indexed": 0}

        embeddings = self._resolve_embeddings()
        # A fresh iterator keeps generation corpus accounting streaming and
        # avoids retaining all catalog rows while embeddings are built.
        manifest = self._generation_for(embeddings, self._iter_catalog_entries())
        save_manifest(self._vault_root, manifest)
        vectorstore = Chroma(
            collection_name=str(manifest["collection_name"]),
            embedding_function=embeddings,
            persist_directory=str(self._chroma_dir),
        )

        deleted_count = 0
        pending_deletes: list[str] = []
        for rel_path in to_delete_paths:
            # Defer destructive deletes until every replacement has been
            # embedded successfully.  A failed build must leave the currently
            # active generation/query results intact.
            pending_deletes.append(rel_path)

        added_count = 0
        updated_count = 0
        indexed_chunk_count = 0
        failed_count = 0
        failures: list[str] = []
        pending_replacements: list[tuple[str, list[str], list[str]]] = []
        staged_new_ids: list[str] = []
        original_indexed_count = len(indexed_state)
        processed_candidates = 0
        completed_work = 0
        for entry in self._iter_catalog_entries():
            rel = str(entry.relative_path).replace("\\", "/")
            if indexed_state.get(rel) == entry.content_sha256:
                continue
            if work_limit is not None and processed_candidates >= work_limit:
                break
            processed_candidates += 1
            file_path = self._vault_root / str(entry.relative_path)
            if not file_path.exists():
                continue
            try:
                raw_text = file_path.read_text(encoding="utf-8")
                body = _strip_frontmatter(raw_text)
                try:
                    entry_metadata = json.loads(str(getattr(entry, "metadata_json", "{}") or "{}"))
                except (TypeError, ValueError, json.JSONDecodeError):
                    entry_metadata = {}
                chunks = self._text_splitter.split_text(body) or [entry.title]
                documents = [
                    Document(
                        page_content=chunk,
                        metadata={
                            "note_id": entry.note_id,
                            "document_key": entry.document_key,
                            "relative_path": str(entry.relative_path).replace("\\", "/"),
                            "content_sha256": entry.content_sha256,
                            "current_revision_id": getattr(entry, "current_revision_id", None),
                            "current_revision": getattr(entry, "current_revision", None),
                            "entity_type": entry.entity_type,
                            "title": entry.title,
                            "ticker": entry.ticker or "",
                            "search_scope": str(entry_metadata.get("search_scope") or "included"),
                            "content_status": str(entry_metadata.get("content_status") or "published"),
                            "sensitivity": str(entry_metadata.get("sensitivity") or "internal"),
                            "verification_status": entry_metadata.get("verification_status")
                            or entry_metadata.get("content_verification_status"),
                            "registry_digest": self._schema_registry.digest(),
                            "policy_digest": self._schema_registry.policy_digest(),
                            "chunk_index": chunk_idx,
                        },
                    )
                    for chunk_idx, chunk in enumerate(chunks)
                ]
                old_hash = indexed_state.get(rel)
                old_info = indexed_chunks.get(rel) or {}
                old_count = int(old_info.get("chunk_count", 0) or 0)
                old_note_id = str(old_info.get("note_id") or entry.note_id)
                old_ids = [f"{old_note_id}:{old_hash}:{n}" for n in range(old_count)] if old_hash and old_count else []

                # Add with deterministic ids. This lets a replacement be
                # published before deleting the old ids; an embedding failure
                # therefore leaves the previous indexed content available.
                new_ids = [f"{entry.note_id}:{entry.content_sha256}:{n}" for n in range(len(documents))]
                # Register the complete intended set before the first Chroma
                # call so a provider that fails after a partial insert can be
                # cleaned up on the failure path as well.
                staged_new_ids.extend(new_ids)
                self._add_bounded(vectorstore, documents, new_ids, batch_size=batch_size)
                if rel in indexed_state:
                    pending_replacements.append((rel, old_ids, new_ids))
                    updated_count += 1
                else:
                    added_count += 1
                indexed_state[rel] = entry.content_sha256
                indexed_chunks[rel] = {
                    "note_id": entry.note_id,
                    "content_sha256": entry.content_sha256,
                    "chunk_count": len(documents),
                }
                indexed_chunk_count += len(documents)
                completed_work += 1
                if progress_callback:
                    progress_callback(completed_work, total_work)
            except Exception as exc:
                failed_count += 1
                failures.append(f"{entry.relative_path}: {exc}")
                logger.warning("Failed indexing note %s: %s", entry.relative_path, exc)

        if failed_count:
            # Remove successfully staged replacement IDs where possible and do
            # not publish or persist the partial state.  The prior active
            # pointer and its indexed state remain the source of truth.
            # Remove every ID staged by this build, including brand-new notes
            # that have no predecessor tuple.  The previous active generation
            # remains the only readable source of truth after a failed build.
            for new_id in staged_new_ids:
                try:
                    vectorstore.delete(ids=[new_id])
                except Exception as exc:
                    logger.warning("Could not clean failed staged id %s: %s", new_id, exc)
            manifest["build_status"] = "failed"
            manifest["validated_at"] = None
            manifest["error"] = "; ".join(failures[:8])
            save_manifest(self._vault_root, manifest)
            return {
                "added": 0,
                "updated": 0,
                "deleted": 0,
                "failed": failed_count,
                "total_indexed": original_indexed_count,
            }

        # Commit deferred destructive operations only after all additions have
        # succeeded, keeping one complete set visible to the active reader.
        for rel, old_ids, _new_ids in pending_replacements:
            try:
                if old_ids:
                    vectorstore.delete(ids=old_ids)
                else:
                    vectorstore.delete(where={"relative_path": rel})
            except Exception as exc:
                logger.warning("Could not remove superseded chunks for %s: %s", rel, exc)
        for rel_path in pending_deletes:
            try:
                vectorstore.delete(where={"relative_path": rel_path})
                indexed_state.pop(rel_path, None)
                indexed_chunks.pop(rel_path, None)
                deleted_count += 1
            except Exception as exc:
                logger.warning("Error deleting %s from Chroma: %s", rel_path, exc)

        manifest["eligible_chunk_count"] = sum(
            int(value.get("chunk_count", 0) or 0) for value in indexed_chunks.values()
        )
        manifest["build_status"] = "validated"
        manifest["validated_at"] = manifest.get("built_at")
        manifest["error"] = None
        save_manifest(self._vault_root, manifest)
        activate_manifest(self._vault_root, manifest)
        self._save_index_state(indexed_state, indexed_chunks=indexed_chunks, generation=manifest)

        return {
            "added": added_count,
            "updated": updated_count,
            "deleted": deleted_count,
            "total_indexed": len(indexed_state),
        }
