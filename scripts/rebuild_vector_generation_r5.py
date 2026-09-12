"""Batch-build and atomically activate the R5 multilingual vector generation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.archivist.catalog_adapter import SqliteNoteCatalogAdapter  # noqa: E402
from tools.archivist.catalog_runtime import resolve_catalog_path  # noqa: E402
from tools.archivist.parser import _strip_frontmatter  # noqa: E402
from tools.archivist.search import get_embeddings  # noqa: E402
from tools.archivist.vector_generation import (  # noqa: E402
    DEFAULT_CHUNKER_CONFIG,
    DEFAULT_CHUNKER_VERSION,
    activate_manifest,
    build_manifest,
    corpus_fingerprint,
    describe_embeddings,
    generation_id_for,
    save_manifest,
    vector_runtime_path,
)
from tools.archivist.maintenance_guard import assert_write_allowed  # noqa: E402


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def _save_state(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".r5.tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def rebuild(vault: Path, run_dir: Path, *, batch_size: int = 256) -> dict[str, Any]:
    vault = vault.resolve()
    run_dir = run_dir.resolve()
    assert_write_allowed(vault)
    catalog_path = resolve_catalog_path(vault, require_exists=True)
    catalog = SqliteNoteCatalogAdapter(vault_root=vault, read_only=True)
    entries = list(catalog.iter_notes(page_size=500))
    if not entries:
        raise RuntimeError("active catalog has no searchable notes")

    embeddings = get_embeddings()
    dimension = getattr(embeddings, "size", None) or len(embeddings.embed_query("dimension probe"))
    model = describe_embeddings(embeddings, dimension=dimension)
    chunker = {
        "version": DEFAULT_CHUNKER_VERSION,
        "config": dict(DEFAULT_CHUNKER_CONFIG),
    }
    chunker["fingerprint"] = hashlib.sha256(
        json.dumps(chunker, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    base_id = generation_id_for(model, chunker, "obsidian_vault_v2")
    nonce = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    generation_id = f"{base_id}_r5_{nonce}"
    collection_name = f"obsidian_vault_v2_r5_{nonce.replace('T', '').replace('Z', '')}"
    manifest = build_manifest(
        vault_root=vault,
        embeddings=embeddings,
        collection_name=collection_name,
        chunker_version=DEFAULT_CHUNKER_VERSION,
        chunker_config=DEFAULT_CHUNKER_CONFIG,
        corpus_scope="active_searchable_notes",
        corpus_entries=entries,
        eligible_chunk_count=0,
        status="building",
        generation_id=generation_id,
        dimension=dimension,
    )
    save_manifest(vault, manifest)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(DEFAULT_CHUNKER_CONFIG["chunk_size"]),
        chunk_overlap=int(DEFAULT_CHUNKER_CONFIG["chunk_overlap"]),
        separators=list(DEFAULT_CHUNKER_CONFIG["separators"]),
    )
    runtime = vector_runtime_path(vault)
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(runtime),
    )
    batch: list[Document] = []
    ids: list[str] = []
    indexed_files: dict[str, str] = {}
    indexed_chunks: dict[str, dict[str, Any]] = {}
    chunk_count = 0
    added_notes = 0

    def flush() -> None:
        nonlocal batch, ids
        if batch:
            vectorstore.add_documents(batch, ids=ids)
            batch = []
            ids = []

    try:
        for entry in entries:
            rel = str(entry.relative_path).replace("\\", "/")
            path = vault / rel
            if not path.is_file():
                continue
            raw = path.read_text(encoding="utf-8")
            body = _strip_frontmatter(raw)
            chunks = splitter.split_text(body) or [entry.title]
            note_ids: list[str] = []
            for index, chunk in enumerate(chunks):
                document_id = f"{entry.note_id}:{entry.content_sha256}:{index}"
                batch.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "note_id": entry.note_id,
                            "document_key": entry.document_key,
                            "relative_path": rel,
                            "content_sha256": entry.content_sha256,
                            "current_revision_id": getattr(entry, "current_revision_id", None),
                            "current_revision": getattr(entry, "current_revision", None),
                            "entity_type": entry.entity_type,
                            "title": entry.title,
                            "ticker": entry.ticker or "",
                            "chunk_index": index,
                        },
                    )
                )
                ids.append(document_id)
                note_ids.append(document_id)
                chunk_count += 1
                if len(batch) >= batch_size:
                    flush()
            indexed_files[rel] = str(entry.content_sha256)
            indexed_chunks[rel] = {
                "note_id": entry.note_id,
                "content_sha256": entry.content_sha256,
                "chunk_count": len(chunks),
                "ids": note_ids,
            }
            added_notes += 1
        flush()
    except Exception as exc:
        manifest["build_status"] = "failed"
        manifest["error"] = str(exc)
        save_manifest(vault, manifest)
        raise

    manifest["eligible_chunk_count"] = chunk_count
    manifest["build_status"] = "validated"
    manifest["validated_at"] = datetime.now(timezone.utc).isoformat()
    manifest["error"] = None
    save_manifest(vault, manifest)
    activate_manifest(vault, manifest)
    _save_state(
        runtime / "vector_index_state.json",
        {
            "version": 2,
            "indexed_files": indexed_files,
            "indexed_chunks": indexed_chunks,
            "generation_id": generation_id,
            "collection_name": collection_name,
            "model_fingerprint": manifest["model_fingerprint"],
            "chunker_fingerprint": manifest["chunker_fingerprint"],
            "corpus_fingerprint": manifest["corpus_fingerprint"],
            "eligible_note_count": manifest["eligible_note_count"],
            "eligible_chunk_count": chunk_count,
        },
    )
    result = {
        "status": "PASS",
        "generation_id": generation_id,
        "collection_name": collection_name,
        "catalog_path": str(catalog_path),
        "runtime": str(runtime),
        "model": model,
        "note_count": added_notes,
        "chunk_count": chunk_count,
        "corpus_fingerprint": manifest["corpus_fingerprint"],
    }
    _write_json(run_dir / "vector-generation-r5.json", {**manifest, "result": result})
    print(json.dumps(result, ensure_ascii=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", type=Path, default=Path("memories"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    rebuild(args.vault, args.run_dir, batch_size=args.batch_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
