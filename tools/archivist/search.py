from langsmith import traceable
import json
import os
import re
import shutil
import sqlite3
import tempfile
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import frontmatter as fm
from filelock import FileLock
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.logger import get_logger
from schemas.pkm_models import MemoryEntry

log = get_logger(__name__)
from .core import VAULT_PATH, INDEX_PATH, INDEX_LOCK, _atomic_write_text, _VAULT_SYSTEM_FILES, _INDEX_EXCLUDE, _LINKED_CONTENT_LIMIT
from .maintenance_guard import assert_write_allowed
from .parser import _chunk_file
from .vector_generation import vector_runtime_path
from .runtime_layout import runtime_layout
from .portable_links import (
    iter_internal_markdown_links,
    resolve_vault_target_detailed,
)
CHROMA_PATH = VAULT_PATH / ".chroma_index"
try:
    CHROMA_PATH = vector_runtime_path(VAULT_PATH)
except Exception:
    # Keep import-time compatibility for tests that monkeypatch CHROMA_PATH.
    CHROMA_PATH = runtime_layout(VAULT_PATH, create=True).vector_root
_CHROMA_MTIME_FILE = CHROMA_PATH / "legacy_index_state.json"
_vs_cache: dict = {}  # {"cache_key": str, "vs": Chroma}


def get_query_vector_runtime(vault_root: Path, active_runtime: Path, manifest: dict | None) -> Path:
    """Return an isolated query copy for Chroma's write-lock bookkeeping.

    Chroma updates its ``acquire_write`` bookkeeping table even for a
    similarity read.  The published vector generation must remain immutable,
    so copy the SQLite catalog and the active HNSW segment once per generation
    into an external query cache.  User notes and the published generation are
    never modified by a read query.
    """
    if not manifest:
        return active_runtime
    generation_id = str(manifest.get("generation_id") or "").strip()
    collection_name = str(manifest.get("collection_name") or "").strip()
    source_db = Path(active_runtime).resolve() / "chroma.sqlite3"
    if not generation_id or not collection_name or not source_db.is_file():
        return active_runtime

    query_root = Path(active_runtime).resolve() / "query_cache" / generation_id
    marker = query_root / "query-cache-manifest.json"
    if marker.is_file() and (query_root / "chroma.sqlite3").is_file():
        try:
            marker_payload = json.loads(marker.read_text(encoding="utf-8"))
            segment_ids = [str(item) for item in marker_payload.get("segment_ids") or []]
            source_matches = str(marker_payload.get("source_runtime") or "") in {"", str(Path(active_runtime).resolve())}
            if source_matches and segment_ids and all((query_root / item).is_dir() for item in segment_ids):
                return query_root
        except (OSError, UnicodeDecodeError, ValueError):
            pass

    query_root.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(query_root.with_suffix(".lock")))
    with lock:
        if marker.is_file() and (query_root / "chroma.sqlite3").is_file():
            try:
                marker_payload = json.loads(marker.read_text(encoding="utf-8"))
                segment_ids = [str(item) for item in marker_payload.get("segment_ids") or []]
                source_matches = str(marker_payload.get("source_runtime") or "") in {"", str(Path(active_runtime).resolve())}
                if source_matches and segment_ids and all((query_root / item).is_dir() for item in segment_ids):
                    return query_root
            except (OSError, UnicodeDecodeError, ValueError):
                pass

        uri = f"file:{source_db.as_posix()}?mode=ro&immutable=1"
        with sqlite3.connect(uri, uri=True) as conn:
            row = conn.execute("SELECT id FROM collections WHERE name = ?", (collection_name,)).fetchone()
            segment_rows = (
                conn.execute(
                    "SELECT id, type FROM segments WHERE collection = ?",
                    (str(row[0]),),
                ).fetchall()
                if row
                else []
            )
        if not row:
            return active_runtime
        collection_id = str(row[0])
        segment_ids = [
            str(item[0])
            for item in segment_rows
            if "vector" in str(item[1]).lower() and (Path(active_runtime).resolve() / str(item[0])).is_dir()
        ]
        if not segment_ids:
            return active_runtime

        staging = query_root.parent / f".{generation_id}.{os.getpid()}.staging"
        assert_write_allowed(query_root)
        if staging.exists():
            shutil.rmtree(staging)
        try:
            staging.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_db, staging / "chroma.sqlite3")
            for segment_id in segment_ids:
                shutil.copytree(Path(active_runtime).resolve() / segment_id, staging / segment_id)
            (staging / "query-cache-manifest.json").write_text(
                json.dumps({
                    "generation_id": generation_id,
                    "collection_name": collection_name,
                    "collection_id": collection_id,
                    "segment_ids": segment_ids,
                    "source_runtime": str(Path(active_runtime).resolve()),
                }, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            if query_root.exists():
                shutil.rmtree(query_root)
            os.replace(staging, query_root)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    return query_root


class OfflineHashEmbeddings:
    """Small deterministic local embedding backend for offline Vault queries."""

    model_name = "local-hash-ngram-v1"
    revision = "1"

    def __init__(self, size: int = 384) -> None:
        self.size = int(size)

    def _embed(self, text: str) -> list[float]:
        import hashlib
        import math

        tokens = re.findall(r"[\w\u0E00-\u0E7F]+", str(text).lower())
        features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        values = [0.0] * self.size
        for feature in features or [""]:
            digest = hashlib.sha256(feature.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.size
            sign = 1.0 if digest[4] & 1 else -1.0
            values[index] += sign
        norm = math.sqrt(sum(value * value for value in values)) or 1.0
        return [value / norm for value in values]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@lru_cache(maxsize=1)
def _get_offline_embeddings() -> OfflineHashEmbeddings:
    return OfflineHashEmbeddings()





VAULT_PATH = Path(os.getenv("OBSIDIAN_VAULT_PATH", "./memories"))
INDEX_PATH = VAULT_PATH / ".system" / "master_index.json"
INDEX_LOCK = str(INDEX_PATH) + ".lock"


@lru_cache(maxsize=1)
def get_embeddings():
    backend = os.getenv("VAULT_EMBEDDINGS_BACKEND", "").strip().lower()
    model_name = os.getenv("VAULT_EMBEDDING_MODEL", "").strip()
    policy_path = VAULT_PATH / ".system" / "ai_retrieval_policy.json"
    if not backend and policy_path.is_file():
        try:
            policy = json.loads(policy_path.read_text(encoding="utf-8"))
            backend = str(policy.get("embedding_backend") or "").strip().lower()
            model_name = model_name or str(policy.get("model_identifier") or "").strip()
        except (OSError, UnicodeDecodeError, ValueError):
            backend = ""
    backend = backend or "offline"
    if backend in {"offline", "local", "hash"}:
        log.info("ใช้ local offline embedding backend สำหรับ Semantic Search")
        return _get_offline_embeddings()
    log.info("กำลังโหลด embedding model สำหรับ Semantic Search")
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=model_name or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"local_files_only": True},
    )


def _load_index_state() -> dict:
    """โหลด per-file state: {rel: {mtime, chunks}}"""
    if not _CHROMA_MTIME_FILE.exists():
        return {}
    try:
        data = json.loads(_CHROMA_MTIME_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "files" in data:
            return data["files"]
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def _save_index_state(files: dict) -> None:
    _atomic_write_text(
        _CHROMA_MTIME_FILE,
        json.dumps({"version": 1, "files": files}, ensure_ascii=False),
    )


def _searchable_files() -> list[Path]:
    from tools.archivist.catalog_runtime import resolve_catalog_path
    try:
        catalog_db = resolve_catalog_path(VAULT_PATH, require_exists=True)
    except FileNotFoundError:
        catalog_db = None
    if catalog_db is not None:
        try:
            from tools.archivist.catalog_adapter import SqliteNoteCatalogAdapter
            cat = SqliteNoteCatalogAdapter(
                db_path=catalog_db, vault_root=VAULT_PATH, read_only=True
            )
            iterator = getattr(cat, "iter_notes", None)
            entries = iterator(page_size=500) if iterator else iter(cat.find_notes(limit=500))
            return [VAULT_PATH / e.relative_path for e in entries if (VAULT_PATH / e.relative_path).exists()]
        except Exception:
            pass

    return [
        f for f in VAULT_PATH.rglob("*.md")
        if f.name not in _VAULT_SYSTEM_FILES
        and not any(excl in f.parts for excl in _INDEX_EXCLUDE)
    ]


def _format_search_results(keyword: str, results: list) -> str:
    if not results:
        return f"ไม่พบความจำที่เกี่ยวข้องกับ '{keyword}'"
    parts = [f"ผลการค้นหาเชิงความหมายสำหรับ '{keyword}' ({len(results)} ผลลัพธ์):\n"]
    for i, doc in enumerate(results, 1):
        metadata = getattr(doc, "metadata", {}) or {}
        source = metadata.get("relative_path") or metadata.get("source", "ไม่ทราบแหล่งที่มา")
        parts.append(f"--- ผลลัพธ์ที่ {i} | แหล่งที่มา: [{source}] ---\n{doc.page_content}\n")
    return "\n".join(parts)


def _current_vector_results(keyword: str, catalog) -> list:
    """Read current vector chunks and join them to the immutable catalog."""
    from tools.archivist.vector_generation import (
        DEFAULT_COLLECTION_NAME,
        VectorGenerationError,
        load_active_manifest,
    )

    try:
        active_manifest = load_active_manifest(VAULT_PATH)
    except VectorGenerationError as exc:
        raise RuntimeError(f"active vector generation unavailable: {exc}") from exc
    collection_name = (
        str(active_manifest["collection_name"])
        if active_manifest is not None
        else DEFAULT_COLLECTION_NAME
    )
    query_runtime = get_query_vector_runtime(VAULT_PATH, CHROMA_PATH, active_manifest)
    active_generation_id = str(active_manifest.get("generation_id") or "") if active_manifest else ""
    cache_root = _vs_cache.get("vault_root")
    if cache_root is not None and cache_root != str(VAULT_PATH.resolve()):
        vectorstore = None
    else:
        vectorstore = _vs_cache.get("vs")
    if vectorstore is not None and _vs_cache.get("generation_id", active_generation_id) != active_generation_id:
        vectorstore = None
    if vectorstore is None:
        vectorstore = Chroma(
            collection_name=collection_name,
            persist_directory=str(query_runtime),
            embedding_function=get_embeddings(),
        )
        _vs_cache["vs"] = vectorstore
        _vs_cache["vault_root"] = str(VAULT_PATH.resolve())
        _vs_cache["generation_id"] = active_generation_id

    results = vectorstore.similarity_search(keyword, k=5)
    filtered = []
    for doc in results:
        metadata = getattr(doc, "metadata", {}) or {}
        source = metadata.get("relative_path") or metadata.get("source")
        source_rel = str(source).replace("\\", "/").strip("/") if source else ""
        entry = None
        try:
            note_id = metadata.get("note_id")
            if note_id:
                entry = catalog.get_by_id(str(note_id))
            if entry is None and source_rel:
                entry = catalog.get_by_path(source_rel)
        except Exception:
            entry = None
        if not source and not metadata.get("note_id"):
            filtered.append(doc)
            continue
        if entry is None:
            continue
        entry_rel = str(entry.relative_path).replace("\\", "/").strip("/")
        if source_rel and source_rel != entry_rel:
            continue
        if str(getattr(entry, "record_state", "active")) != "active":
            continue
        if str(getattr(entry, "storage_scope", "active")) != "active":
            continue
        indexed_hash = metadata.get("content_sha256")
        if indexed_hash and str(indexed_hash) != str(entry.content_sha256):
            continue
        filtered.append(doc)
    return filtered


def _search_registered_v2(keyword: str) -> str:
    """Read-only V2 query path.

    Catalog construction is read-only and the vector index is prepared by the
    background worker. A cold query may load the existing collection and create
    one query embedding, but it must never process outbox entries, add/delete
    documents, write index state, or rebuild the collection.
    """
    try:
        from tools.archivist.catalog_runtime import resolve_catalog_path
        from tools.archivist.catalog_adapter import SqliteNoteCatalogAdapter
        catalog_db = resolve_catalog_path(VAULT_PATH, require_exists=True)
        catalog = SqliteNoteCatalogAdapter(
            db_path=catalog_db, vault_root=VAULT_PATH, read_only=True
        )
        if catalog.count_notes() == 0:
            return "ยังไม่มีไฟล์ความจำใดใน Vault"
    except Exception as exc:
        return f"ไม่สามารถค้นหาได้ในขณะนี้: catalog read-only ไม่พร้อมใช้งาน ({exc})"

    try:
        from tools.archivist.vector_generation import (
            DEFAULT_COLLECTION_NAME,
            VectorGenerationError,
            load_active_manifest,
        )
        active_manifest = load_active_manifest(VAULT_PATH)
    except VectorGenerationError as exc:
        return f"เกิดข้อผิดพลาดในการเปิด vectorstore: active generation unavailable ({exc})"
    collection_name = (
        str(active_manifest["collection_name"])
        if active_manifest is not None
        else DEFAULT_COLLECTION_NAME
    )

    query_runtime = get_query_vector_runtime(VAULT_PATH, CHROMA_PATH, active_manifest)
    active_generation_id = str(active_manifest.get("generation_id") or "") if active_manifest else ""
    cache_root = _vs_cache.get("vault_root")
    # Older callers/tests seed `_vs_cache` without a vault_root marker. Keep
    # that explicit object as a warm cache; newly created caches are tagged so a
    # process switching vault roots cannot reuse the wrong collection.
    if cache_root is not None and cache_root != str(VAULT_PATH.resolve()):
        vectorstore = None
    else:
        vectorstore = _vs_cache.get("vs")
    if vectorstore is not None and _vs_cache.get("generation_id", active_generation_id) != active_generation_id:
        vectorstore = None
    if vectorstore is None:
        try:
            vectorstore = Chroma(
                collection_name=collection_name,
                persist_directory=str(query_runtime),
                embedding_function=get_embeddings(),
            )
        except Exception as exc:
            log.error("V2 vectorstore unavailable at %s: %s", CHROMA_PATH, exc)
            return f"เกิดข้อผิดพลาดในการเปิด vectorstore: {exc}"
        _vs_cache["vs"] = vectorstore
        _vs_cache["vault_root"] = str(VAULT_PATH.resolve())
        _vs_cache["generation_id"] = active_generation_id

    try:
        results = vectorstore.similarity_search(keyword, k=5)
    except Exception as exc:
        return f"เกิดข้อผิดพลาดในการค้นหา: {exc}"

    # Join only the selected chunks back to the current catalog row. Avoid
    # materializing/scanning the whole catalog on every request while still
    # hiding stale/deleted chunks from the current read model.
    filtered = []
    for doc in results:
        metadata = getattr(doc, "metadata", {}) or {}
        source = metadata.get("relative_path") or metadata.get("source")
        source_rel = str(source).replace("\\", "/").strip("/") if source else ""
        entry = None
        try:
            note_id = metadata.get("note_id")
            if note_id:
                entry = catalog.get_by_id(str(note_id))
            if entry is None and source_rel:
                entry = catalog.get_by_path(source_rel)
        except Exception:
            entry = None
        # Older stores may not carry source metadata; preserve those results
        # for compatibility. Once a source/identity is present it must join
        # to an active current catalog row.
        if not source and not metadata.get("note_id"):
            filtered.append(doc)
            continue
        if entry is None:
            continue
        entry_rel = str(entry.relative_path).replace("\\", "/").strip("/")
        if source_rel and source_rel != entry_rel:
            continue
        if str(getattr(entry, "record_state", "active")) != "active":
            continue
        if str(getattr(entry, "storage_scope", "active")) != "active":
            continue
        indexed_hash = metadata.get("content_sha256")
        if indexed_hash and str(indexed_hash) != str(entry.content_sha256):
            continue
        filtered.append(doc)
    try:
        from tools.archivist.hybrid_retriever import hybrid_search

        ranked = hybrid_search(
            VAULT_PATH,
            catalog,
            catalog_db,
            keyword,
            vector_results=filtered,
            k=5,
        )
    except Exception as exc:
        log.warning("Hybrid retrieval fallback failed for %s: %s", keyword, exc)
        ranked = filtered
    return _format_search_results(keyword, ranked)


def _search_legacy_index(keyword: str) -> str:
    """Compatibility path for pre-catalog vaults.

    V2 production vaults never enter this function. It remains for callers that
    have not created a catalog yet and for the legacy tests; all future managed
    writes are routed through the catalog/worker path above.
    """
    # A caller may provide a pre-warmed legacy vector store (for example a
    # read-only compatibility adapter).  Do not turn that query into an
    # indexing request merely because no legacy snapshot accompanies it.
    if (
        "vs" in _vs_cache
        and "cache_signature" not in _vs_cache
        and "vault_root" not in _vs_cache
    ):
        try:
            results = _vs_cache["vs"].similarity_search(keyword, k=5)
        except Exception as exc:
            return f"เกิดข้อผิดพลาดในการค้นหา: {exc}"
        return _format_search_results(keyword, results)

    stored = _load_index_state()
    md_files = _searchable_files()
    if not md_files:
        return "ยังไม่มีไฟล์ความจำใดใน Vault"
    current = {
        str(f.relative_to(VAULT_PATH)): {"mtime": f.stat().st_mtime}
        for f in md_files
    }
    added_or_changed = [
        rel for rel, info in current.items()
        if rel not in stored or abs(stored[rel].get("mtime", 0) - info["mtime"]) > 1e-3
    ]
    removed = [rel for rel in stored if rel not in current]
    cache_signature = (
        len(current),
        tuple(sorted((rel, info["mtime"]) for rel, info in current.items())),
    )
    needs_update = bool(added_or_changed or removed) or _vs_cache.get("cache_signature") != cache_signature
    if "vs" in _vs_cache and not needs_update:
        vectorstore = _vs_cache["vs"]
    else:
        try:
            vectorstore = _vs_cache.get("vs") or Chroma(
                collection_name="obsidian_vault_v2",
                persist_directory=str(CHROMA_PATH),
                embedding_function=get_embeddings(),
            )
        except Exception as exc:
            return f"เกิดข้อผิดพลาดในการเปิด vectorstore: {exc}"
        ids_to_delete: list[str] = []
        for rel in [*removed, *added_or_changed]:
            ids_to_delete.extend(f"{rel}::{i}" for i in range(stored.get(rel, {}).get("chunks", 0)))
        if ids_to_delete:
            try:
                vectorstore.delete(ids=ids_to_delete)
            except Exception as exc:
                log.warning("Legacy Chroma delete failed: %s", exc)
        if added_or_changed:
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            all_texts: list[str] = []
            all_metas: list[dict] = []
            all_ids: list[str] = []
            for rel in added_or_changed:
                fp = VAULT_PATH / rel
                if not fp.exists():
                    continue
                texts, metas, ids = _chunk_file(fp, splitter)
                all_texts.extend(texts)
                all_metas.extend(metas)
                all_ids.extend(ids)
                current[rel]["chunks"] = len(texts)
            if all_texts:
                try:
                    vectorstore.add_texts(texts=all_texts, metadatas=all_metas, ids=all_ids)
                except Exception as exc:
                    return f"เกิดข้อผิดพลาดในการเพิ่ม vectorstore: {exc}"
        # Persist the snapshot only after all legacy index mutations complete;
        # otherwise every invocation appears to be a first index and deleted
        # or changed chunk ids cannot be removed on the next run.
        _save_index_state(current)
        _vs_cache["vs"] = vectorstore
        _vs_cache["cache_signature"] = cache_signature
    try:
        results = vectorstore.similarity_search(keyword, k=5)
    except Exception as exc:
        return f"เกิดข้อผิดพลาดในการค้นหา: {exc}"
    return _format_search_results(keyword, results)


@tool
def search_all_memories(keyword: str) -> str:
    """ค้นหาความจำด้วย semantic search โดย query path ของ V2 เป็น read-only."""
    from tools.archivist.catalog_runtime import resolve_catalog_path
    try:
        catalog_exists = resolve_catalog_path(VAULT_PATH, require_exists=True).is_file()
    except (FileNotFoundError, OSError):
        catalog_exists = False
    if catalog_exists:
        return _search_registered_v2(keyword)
    return _search_legacy_index(keyword)


@tool
def search_memories_with_evidence(
    keyword: str,
    retrieval_namespace: str = "primary",
    production_mode: bool = False,
) -> str:
    """Return current Vault passages joined to identity/hash evidence.

    This is the production-facing retrieval boundary for callers that may
    ground an AI answer. It intentionally joins the current catalog to the
    read-only hybrid vector/lexical layer, then applies the evidence contract
    before any passage is returned. A preview is marked as such; callers
    should use ``read_note_chunk`` for a complete long note.
    """
    from tools.archivist.ai_answer_contract import collect_evidence, validate_answer
    from tools.archivist.catalog_adapter import SqliteNoteCatalogAdapter
    from tools.archivist.catalog_runtime import resolve_catalog_path

    namespace = str(retrieval_namespace or "primary").strip().lower()
    try:
        catalog_path = resolve_catalog_path(VAULT_PATH, require_exists=True)
        catalog = SqliteNoteCatalogAdapter(
            db_path=catalog_path,
            vault_root=VAULT_PATH,
            read_only=True,
        )
    except Exception as exc:
        return json.dumps(
            {
                "status": "BLOCKED",
                "keyword": keyword,
                "retrieval_namespace": namespace,
                "error": f"catalog_unavailable: {exc}",
                "results": [],
                "evidence": [],
            },
            ensure_ascii=False,
        )

    try:
        from tools.archivist.hybrid_retriever import hybrid_search

        documents = hybrid_search(
            VAULT_PATH,
            catalog,
            catalog_path,
            keyword,
            vector_results=_current_vector_results(keyword, catalog),
            k=5,
        )
    except Exception as exc:
        return json.dumps(
            {
                "status": "BLOCKED",
                "keyword": keyword,
                "retrieval_namespace": namespace,
                "error": f"retrieval_failed: {exc}",
                "results": [],
                "evidence": [],
            },
            ensure_ascii=False,
        )

    evidence = collect_evidence(
        VAULT_PATH,
        catalog,
        documents,
        retrieval_namespace=namespace,
    )
    evidence_by_path = {
        str(item.get("relative_path") or ""): item
        for item in evidence
    }
    results: list[dict] = []
    for document in documents:
        metadata = getattr(document, "metadata", {}) or {}
        relative_path = str(
            metadata.get("relative_path") or metadata.get("source") or ""
        ).replace("\\", "/").strip("/")
        item = evidence_by_path.get(relative_path)
        if item is None:
            continue
        result = dict(item)
        result.update(
            {
                "content": str(getattr(document, "page_content", "") or ""),
                "retrieval_score": metadata.get("retrieval_score"),
            }
        )
        results.append(result)

    cited_paths = [str(item["relative_path"]) for item in evidence]
    contract = validate_answer(
        "retrieval_context" if evidence else "",
        evidence,
        cited_paths=cited_paths,
        production_mode=bool(production_mode),
        retrieval_namespace=namespace,
    )
    limitations = list(contract.get("limitations") or [])
    if any(item.get("content_truncated") for item in evidence):
        limitations.append("One or more results are previews; page the note before making a completeness claim.")
    return json.dumps(
        {
            "status": contract["status"],
            "keyword": keyword,
            "retrieval_namespace": namespace,
            "production_mode": bool(production_mode),
            "results": results,
            "evidence": evidence,
            "contract": contract,
            "limitations": sorted(set(limitations)),
        },
        ensure_ascii=False,
    )


def _find_file_by_name(name: str, all_files: list[Path]) -> Path | None:
    """ค้นหาไฟล์โดย stem ตรงทั้งหมดก่อน แล้ว fallback ไป partial match"""
    name_lower = name.lower()
    exact = next((f for f in all_files if f.stem.lower() == name_lower), None)
    if exact:
        return exact
    return next((f for f in all_files if name_lower in f.stem.lower()), None)


def _graph_eligible_files(all_files: list[Path]) -> list[Path]:
    """Filter GraphRAG candidates by the same public/internal read policy.

    ``iter_notes`` is intentionally broader than vector retrieval because it
    is also used by catalog maintenance. GraphRAG must apply its own
    fail-closed lifecycle and sensitivity boundary before it chooses either a
    main entity or a backlink.
    """
    from tools.archivist.metadata import parse_note
    from tools.archivist.schema_registry import load_default_registry
    from tools.archivist.catalog_runtime import resolve_catalog_path

    registry = load_default_registry()
    try:
        catalog_exists = resolve_catalog_path(VAULT_PATH, require_exists=True).is_file()
    except (FileNotFoundError, OSError):
        catalog_exists = False
    eligible: list[Path] = []
    for path in all_files:
        try:
            metadata, _, issues = parse_note(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError):
            continue
        if not metadata and not catalog_exists:
            eligible.append(path)
            continue
        if issues:
            # Legacy test/adapter notes without frontmatter remain readable
            # only when no catalog exists. Production catalog-backed notes
            # are required to carry valid metadata.
            if catalog_exists:
                continue
            eligible.append(path)
            continue
        if not registry.is_index_eligible(metadata, vector=False):
            continue
        sensitivity = str(metadata.get("sensitivity") or "internal").lower()
        if sensitivity not in {"public", "internal"}:
            continue
        content_status = str(metadata.get("content_status") or "published").lower()
        lifecycle_status = str(metadata.get("lifecycle_status") or "active").lower()
        if content_status in {"draft", "generated", "retired", "superseded"}:
            continue
        if lifecycle_status in {"stub", "retired", "superseded"}:
            continue
        eligible.append(path)
    return eligible


@tool
def search_graph_context(entity_name: str) -> str:
    """ค้นหาข้อมูล Entity พร้อมดึงเนื้อหาจาก Linked Entities ที่เชื่อมโยงกัน (GraphRAG)

    [Usage/When to use]
    ใช้เมื่อต้องการวิเคราะห์บริษัท, บุคคล, กลยุทธ์, หรือเหตุการณ์แบบเจาะลึก 360 องศา
    - เครื่องมือนี้จะดึงเนื้อหาจาก "ไฟล์เป้าหมาย" และ "ไฟล์ทั้งหมดที่เป้าหมายนั้นทำ Wikilink โยงไปหา" มาให้ในครั้งเดียว
    - เหมาะสำหรับการดูภาพรวมเครือข่ายความสัมพันธ์ของ Entity ใด Entity หนึ่ง

    [Caution]
    - ไม่เหมาะสำหรับการค้นหากว้างๆ หรือ Semantic Search (ให้ใช้ `search_all_memories` แทน)
    - ต้องระบุชื่อ Entity ที่มีแนวโน้มเป็นชื่อไฟล์จริงๆ ในระบบ

    Args:
        entity_name (str): ชื่อ Entity หรือชื่อไฟล์เป้าหมายที่ต้องการเจาะลึก เช่น 'PTT', 'Somchai', 'Interest_Rate_Hike'

    Returns:
        str: เนื้อหาของ Entity หลัก พร้อมกับเนื้อหาแบบตัดทอนของไฟล์ทั้งหมดที่เชื่อมโยงอยู่ (หรือแจ้งเตือนหากไม่พบไฟล์)
    """
    # The module-level path is intentionally kept compatible with legacy
    # callers/tests and may be relative (for example ``memories``).  Normalize
    # once at the GraphRAG boundary so every catalog/path operation uses the
    # same absolute Vault root on Windows and POSIX.
    vault_root = VAULT_PATH.resolve()
    all_files = [path.resolve() for path in _graph_eligible_files(_searchable_files())]
    if not all_files:
        return "ยังไม่มีไฟล์ความจำใดใน Vault"

    # Step 1-2: หาและอ่านไฟล์หลัก
    main_file = _find_file_by_name(entity_name, all_files)
    if main_file is None:
        return f"ไม่พบไฟล์สำหรับ entity '{entity_name}' ใน Vault"

    eligible = {path.resolve() for path in all_files}
    main_content = main_file.read_text(encoding="utf-8")
    output = f"--- Main Entity: {main_file.stem} ---\n{main_content}\n"

    def _context(path: Path) -> str:
        content = path.read_text(encoding="utf-8")
        if len(content) > _LINKED_CONTENT_LIMIT:
            return content[:_LINKED_CONTENT_LIMIT] + "\n...[ตัดทอน]"
        return content

    def _linked_targets(path: Path, content: str) -> list[tuple[str, Path | None, str]]:
        """Resolve portable Markdown edges and legacy wikilinks from a note."""
        raw_links: list[tuple[str, str]] = [
            (link.destination, "markdown")
            for link in iter_internal_markdown_links(content)
        ]
        raw_links.extend(
            (raw.split("|", 1)[0].strip(), "wikilink")
            for raw in re.findall(r"\[\[(.*?)\]\]", content)
        )
        result: list[tuple[str, Path | None, str]] = []
        seen: set[tuple[str, str]] = set()
        for target, link_type in raw_links:
            key = (target, link_type)
            if not target or key in seen:
                continue
            seen.add(key)
            resolution = resolve_vault_target_detailed(
                vault_root,
                target,
                source=path,
            )
            target_path = resolution.target if resolution.status == "resolved" else None
            if target_path is not None and target_path.resolve() not in eligible:
                # Excluded/restricted notes are not GraphRAG evidence.
                target_path = None
            result.append((target, target_path, link_type))
        return result

    outgoing: list[tuple[str, Path | None, str]] = []
    backlinks: list[tuple[str, Path]] = []
    used_catalog_graph = False
    try:
        from tools.archivist.catalog_adapter import SqliteNoteCatalogAdapter
        from tools.archivist.catalog_runtime import resolve_catalog_path

        catalog_path = resolve_catalog_path(vault_root, require_exists=True)
        catalog = SqliteNoteCatalogAdapter(
            db_path=catalog_path,
            vault_root=vault_root,
            read_only=True,
        )
        main_entry = catalog.get_by_path(main_file.relative_to(vault_root).as_posix())
        if main_entry is not None:
            catalog_outgoing = catalog.iter_link_edges(note_id=main_entry.note_id, direction="outgoing")
            catalog_incoming = catalog.iter_link_edges(note_id=main_entry.note_id, direction="incoming")
            if catalog_outgoing or catalog_incoming:
                used_catalog_graph = True
                for edge in catalog_outgoing:
                    target = vault_root / str(edge["target_relative_path"])
                    target_entry = catalog.get_by_path(str(edge["target_relative_path"]))
                    if (
                        target.resolve() in eligible
                        and target_entry is not None
                        and str(edge.get("target_content_sha256") or "") == str(target_entry.content_sha256)
                    ):
                        outgoing.append((str(edge["raw_target"]), target, "markdown"))
                for edge in catalog_incoming:
                    source = vault_root / str(edge["source_relative_path"])
                    source_entry = catalog.get_by_path(str(edge["source_relative_path"]))
                    if (
                        source.resolve() in eligible
                        and source_entry is not None
                        and str(edge.get("source_content_sha256") or "") == str(source_entry.content_sha256)
                    ):
                        backlinks.append((source.stem, source))
    except Exception:
        # A legacy generation may not yet contain note_links.  The filesystem
        # fallback below is read-only and uses the exact same resolver.
        used_catalog_graph = False

    if not used_catalog_graph:
        outgoing = _linked_targets(main_file, main_content)
        for candidate in all_files:
            if candidate.resolve() == main_file.resolve():
                continue
            try:
                candidate_content = candidate.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for _, resolved, _ in _linked_targets(candidate, candidate_content):
                if resolved is not None and resolved.resolve() == main_file.resolve():
                    backlinks.append((candidate.stem, candidate))
                    break

    if outgoing or backlinks:
        output += "\n--- Linked Connections (portable Markdown + legacy adapter) ---\n"
        seen_paths: set[Path] = set()
        for target, linked_file, link_type in outgoing:
            if linked_file is not None:
                linked_file = linked_file.resolve()
                if linked_file in seen_paths:
                    continue
                seen_paths.add(linked_file)
                output += (
                    f"\n- outgoing [{target}] ({link_type}) -> "
                    f"{linked_file.relative_to(vault_root).as_posix()}:\n"
                    f"{_context(linked_file)}\n"
                )
            else:
                output += f"\n- outgoing [{target}]: (ไม่พบไฟล์ใน Vault; excluded from searchable GraphRAG)\n"
        for title, linked_file in backlinks:
            linked_file = linked_file.resolve()
            if linked_file in seen_paths:
                continue
            seen_paths.add(linked_file)
            output += (
                f"\n- incoming [{title}] <- "
                f"{linked_file.relative_to(vault_root).as_posix()}:\n"
                f"{_context(linked_file)}\n"
            )

    return output


