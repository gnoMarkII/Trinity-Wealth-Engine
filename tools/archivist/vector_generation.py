"""Durable manifest and activation helpers for Vault V2 vector generations.

The vector store is a derived read model.  A query may only use a generation
whose manifest was validated and atomically published as active; a worker may
build a replacement without disturbing the last valid generation.  Keeping
the contract in this small module prevents the worker and search code from
silently choosing different collection/model/chunker defaults.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from tools.archivist.core import _atomic_write_text
from tools.archivist.runtime_layout import runtime_layout
from tools.archivist.schema_registry import load_default_registry

GENERATION_MANIFEST_VERSION = 2
DEFAULT_COLLECTION_NAME = "obsidian_vault_v2"
DEFAULT_CHUNKER_VERSION = "recursive-character-text-splitter-v1"
DEFAULT_CHUNKER_CONFIG: dict[str, Any] = {
    "chunk_size": 1000,
    "chunk_overlap": 150,
    "separators": ["\n## ", "\n### ", "\n\n", "\n", " "],
}


class VectorGenerationError(RuntimeError):
    """Raised when a generation manifest is invalid or cannot be activated."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def generation_dir(vault_root: Path) -> Path:
    # Generation manifests are derived runtime state and must live beside the
    # external vector database.  The vault keeps only the small active pointer.
    return vector_runtime_path(Path(vault_root)) / "generations"


def vector_runtime_path(vault_root: Path) -> Path:
    """Return rebuildable vector storage outside the Obsidian user-content tree."""
    root = Path(vault_root).resolve()
    try:
        runtime = runtime_layout(root, create=True).vector_root
    except ValueError as exc:
        raise VectorGenerationError(str(exc)) from exc
    runtime.mkdir(parents=True, exist_ok=True)
    return runtime


def active_pointer_path(vault_root: Path) -> Path:
    return Path(vault_root).resolve() / ".system" / "vector_generation_active.json"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")


def fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def describe_embeddings(embeddings: Any, *, dimension: Optional[int] = None) -> dict[str, Any]:
    """Return stable model metadata without downloading or mutating a model."""
    cls = type(embeddings)
    identifier = (
        getattr(embeddings, "model_name", None)
        or getattr(embeddings, "model", None)
        or getattr(embeddings, "model_id", None)
        or f"{cls.__module__}.{cls.__qualname__}"
    )
    if not isinstance(identifier, str):
        identifier = str(identifier)
    revision = getattr(embeddings, "revision", None) or getattr(embeddings, "model_revision", None) or "unknown"
    inferred_dimension = dimension
    if inferred_dimension is None:
        value = getattr(embeddings, "size", None) or getattr(embeddings, "dimension", None)
        try:
            inferred_dimension = int(value) if value is not None else None
        except (TypeError, ValueError):
            inferred_dimension = None
    descriptor = {
        "identifier": identifier,
        "revision": str(revision),
        "dimension": inferred_dimension,
    }
    descriptor["fingerprint"] = fingerprint(descriptor)
    return descriptor


def describe_chunker(*, version: str = DEFAULT_CHUNKER_VERSION, config: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    payload = {"version": version, "config": dict(config or DEFAULT_CHUNKER_CONFIG)}
    payload["fingerprint"] = fingerprint(payload)
    return payload


def corpus_fingerprint(entries: Iterable[Any]) -> tuple[str, int, int]:
    """Fingerprint identity/body scope while streaming catalog entries.

    Entries may expose ``note_id``, ``relative_path``, ``content_sha256`` and
    ``metadata_json``.  The function intentionally does not read note files or
    retain the whole corpus in memory.  Catalog callers provide a stable
    keyset order, so the resulting digest is deterministic for a generation.
    """
    digest = hashlib.sha256()
    count = 0
    for entry in entries:
        count += 1
        row = {
            "note_id": str(getattr(entry, "note_id", "")),
            "document_key": str(getattr(entry, "document_key", "")),
            "relative_path": str(getattr(entry, "relative_path", "")),
            "content_sha256": str(getattr(entry, "content_sha256", "")),
        }
        encoded = _canonical_json(row)
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest(), count, 0


def generation_id_for(model: Mapping[str, Any], chunker: Mapping[str, Any], collection_name: str) -> str:
    basis = {
        "model_fingerprint": model.get("fingerprint"),
        "chunker_fingerprint": chunker.get("fingerprint"),
        "collection_name": collection_name,
    }
    return "gen_" + fingerprint(basis)[:24]


def manifest_path(vault_root: Path, generation_id: str) -> Path:
    return generation_dir(vault_root) / f"{generation_id}.json"


def build_manifest(
    *,
    vault_root: Path,
    embeddings: Any,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    chunker_version: str = DEFAULT_CHUNKER_VERSION,
    chunker_config: Optional[Mapping[str, Any]] = None,
    corpus_scope: str = "active_searchable_notes",
    corpus_entries: Optional[Iterable[Any]] = None,
    eligible_chunk_count: int = 0,
    status: str = "building",
    dimension: Optional[int] = None,
    generation_id: Optional[str] = None,
    error: Optional[str] = None,
    registry_digest: Optional[str] = None,
    policy_digest: Optional[str] = None,
) -> dict[str, Any]:
    model = describe_embeddings(embeddings, dimension=dimension)
    chunker = describe_chunker(version=chunker_version, config=chunker_config)
    corpus_hash = ""
    note_count = 0
    if corpus_entries is not None:
        corpus_hash, note_count, _ = corpus_fingerprint(corpus_entries)
    gid = generation_id or generation_id_for(model, chunker, collection_name)
    registry = load_default_registry()
    now = _utc_now()
    return {
        "manifest_version": GENERATION_MANIFEST_VERSION,
        "registry_digest": registry_digest or registry.digest(),
        "policy_digest": policy_digest or registry.policy_digest(),
        "generation_id": gid,
        "model_identifier": model["identifier"],
        "model_revision": model["revision"],
        "model_fingerprint": model["fingerprint"],
        "dimension": model["dimension"],
        "chunker_version": chunker["version"],
        "chunker_config": chunker["config"],
        "chunker_fingerprint": chunker["fingerprint"],
        "collection_name": collection_name,
        "corpus_scope": corpus_scope,
        "corpus_fingerprint": corpus_hash,
        "eligible_set_fingerprint": corpus_hash,
        "eligible_note_count": note_count,
        "eligible_chunk_count": int(eligible_chunk_count),
        "build_status": status,
        "built_at": now,
        "validated_at": now if status == "validated" else None,
        "error": error,
    }


def validate_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "manifest_version",
        "generation_id",
        "model_identifier",
        "model_revision",
        "model_fingerprint",
        "dimension",
        "chunker_version",
        "chunker_config",
        "chunker_fingerprint",
        "collection_name",
        "corpus_scope",
        "corpus_fingerprint",
        "eligible_note_count",
        "eligible_chunk_count",
        "build_status",
        "built_at",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise VectorGenerationError(f"generation manifest missing fields: {', '.join(missing)}")
    version = payload.get("manifest_version")
    if version not in {1, GENERATION_MANIFEST_VERSION}:
        raise VectorGenerationError(f"unsupported generation manifest version: {version}")
    if version >= 2:
        for field in ("registry_digest", "policy_digest", "eligible_set_fingerprint"):
            if not isinstance(payload.get(field), str) or not payload.get(field):
                raise VectorGenerationError(f"generation manifest {field} is required for version {version}")
    if payload.get("build_status") not in {"building", "validated", "failed", "blocked"}:
        raise VectorGenerationError(f"invalid generation build_status: {payload.get('build_status')}")
    if not isinstance(payload.get("chunker_config"), dict):
        raise VectorGenerationError("generation chunker_config must be an object")
    if not payload.get("collection_name"):
        raise VectorGenerationError("generation collection_name is required")
    return dict(payload)


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise VectorGenerationError(f"generation manifest not found: {path}") from exc
    except (OSError, ValueError) as exc:
        raise VectorGenerationError(f"generation manifest unreadable: {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise VectorGenerationError(f"generation manifest must be an object: {path}")
    return validate_manifest(raw)


def save_manifest(vault_root: Path, payload: Mapping[str, Any]) -> Path:
    data = validate_manifest(payload)
    target = manifest_path(vault_root, str(data["generation_id"]))
    target.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(target, json.dumps(data, indent=2, ensure_ascii=False))
    return target


def activate_manifest(vault_root: Path, payload: Mapping[str, Any]) -> Path:
    data = validate_manifest(payload)
    if data.get("build_status") != "validated":
        raise VectorGenerationError("only a validated generation may be activated")
    target = save_manifest(vault_root, data)
    pointer = {
        "pointer_version": 1,
        "generation_id": data["generation_id"],
        "manifest_relative_path": target.relative_to(vector_runtime_path(Path(vault_root))).as_posix(),
        "collection_name": data["collection_name"],
        "activated_at": _utc_now(),
    }
    pointer_path = active_pointer_path(vault_root)
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(pointer_path, json.dumps(pointer, indent=2, ensure_ascii=False))
    return pointer_path


def load_active_manifest(vault_root: Path) -> Optional[dict[str, Any]]:
    pointer_path = active_pointer_path(vault_root)
    if not pointer_path.is_file():
        return None
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        if not isinstance(pointer, dict) or not pointer.get("generation_id"):
            raise VectorGenerationError("active generation pointer is malformed")
        payload = load_manifest(manifest_path(Path(vault_root), str(pointer["generation_id"])))
        if payload.get("build_status") != "validated":
            raise VectorGenerationError("active generation is not validated")
        if pointer.get("collection_name") != payload.get("collection_name"):
            raise VectorGenerationError("active generation collection mismatch")
        return payload
    except (OSError, ValueError, VectorGenerationError) as exc:
        raise VectorGenerationError(f"active vector generation unavailable: {exc}") from exc
