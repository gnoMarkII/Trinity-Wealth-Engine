"""External runtime location and generation pointer for the note catalog.

Markdown, durable identities, and Obsidian-facing control metadata stay in
the vault.  SQLite generations are rebuildable read-model state and live in
``data/vault_runtime/<vault-id>`` (or the explicitly configured runtime
directory).  Query paths never create directories or mutate the pointer.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from tools.archivist.core import _atomic_write_text
from tools.archivist.runtime_layout import RuntimeLayoutError, runtime_root_for


CATALOG_POINTER_VERSION = 1
CATALOG_FILENAME = "vault_catalog.db"
CATALOG_OUTBOX_FILENAME = "catalog_outbox.jsonl"
_GENERATION_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class CatalogRuntimeError(RuntimeError):
    """Raised when an external catalog generation is unsafe or invalid."""


def catalog_runtime_root(vault_root: str | Path, *, create: bool = False) -> Path:
    try:
        return runtime_root_for(vault_root, create=create)
    except RuntimeLayoutError as exc:
        raise CatalogRuntimeError(str(exc)) from exc


def catalog_pointer_path(vault_root: str | Path) -> Path:
    return Path(vault_root).resolve() / ".system" / "catalog_generation_active.json"


def legacy_catalog_path(vault_root: str | Path) -> Path:
    return Path(vault_root).resolve() / ".system" / CATALOG_FILENAME


def catalog_outbox_path(vault_root: str | Path, *, create: bool = False) -> Path:
    """Return the durable catalog outbox location for this vault.

    Once an external generation is published, catalog mutations must not be
    written beside the immutable active database.  Keep the legacy in-vault
    location only for pre-generation/test vaults so older callers remain
    compatible while the live R5 vault uses external runtime storage.
    """
    root = Path(vault_root).resolve()
    pointer = load_catalog_pointer(root)
    if pointer is None:
        return root / ".system" / CATALOG_OUTBOX_FILENAME
    runtime = catalog_runtime_root(root, create=create)
    outbox = runtime / "outbox" / CATALOG_OUTBOX_FILENAME
    if create:
        outbox.parent.mkdir(parents=True, exist_ok=True)
    return outbox


def load_catalog_pointer(vault_root: str | Path) -> Optional[dict[str, Any]]:
    path = catalog_pointer_path(vault_root)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise CatalogRuntimeError(f"cannot read catalog pointer {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CatalogRuntimeError(f"catalog pointer must be an object: {path}")
    if payload.get("pointer_version") != CATALOG_POINTER_VERSION:
        raise CatalogRuntimeError(f"unsupported catalog pointer version: {payload.get('pointer_version')}")
    generation_id = str(payload.get("generation_id") or "")
    if not _GENERATION_RE.fullmatch(generation_id):
        raise CatalogRuntimeError(f"invalid catalog generation id: {generation_id!r}")
    return payload


def catalog_generation_path(
    vault_root: str | Path,
    generation_id: str,
    *,
    create: bool = False,
) -> Path:
    if not _GENERATION_RE.fullmatch(str(generation_id)):
        raise CatalogRuntimeError(f"invalid catalog generation id: {generation_id!r}")
    path = catalog_runtime_root(vault_root, create=create) / "catalog" / str(generation_id) / CATALOG_FILENAME
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def resolve_catalog_path(
    vault_root: str | Path,
    *,
    require_exists: bool = False,
    allow_legacy_fallback: bool = True,
) -> Path:
    """Resolve the active database without creating a runtime or touching SQLite."""
    root = Path(vault_root).resolve()
    pointer = load_catalog_pointer(root)
    if pointer is not None:
        path = catalog_generation_path(root, str(pointer["generation_id"]))
        if pointer.get("database_relative_path"):
            rel = Path(str(pointer["database_relative_path"]))
            candidate = (catalog_runtime_root(root) / rel).resolve()
            runtime = catalog_runtime_root(root)
            if not candidate.is_relative_to(runtime):
                raise CatalogRuntimeError(f"catalog pointer escapes runtime root: {candidate}")
            path = candidate
        if require_exists and not path.is_file():
            raise FileNotFoundError(f"active catalog generation does not exist: {path}")
        return path

    external_default = catalog_runtime_root(root) / "catalog" / "active" / CATALOG_FILENAME
    if external_default.is_file():
        return external_default
    legacy = legacy_catalog_path(root)
    if allow_legacy_fallback:
        if require_exists and not legacy.is_file():
            raise FileNotFoundError(f"catalog does not exist: {legacy}")
        return legacy
    if require_exists and not external_default.is_file():
        raise FileNotFoundError(f"external catalog does not exist: {external_default}")
    return external_default


def write_catalog_pointer(
    vault_root: str | Path,
    *,
    generation_id: str,
    database_path: Path,
    manifest_path: Optional[Path] = None,
) -> Path:
    root = Path(vault_root).resolve()
    runtime = catalog_runtime_root(root)
    database = database_path.resolve()
    if not database.is_relative_to(runtime):
        raise CatalogRuntimeError(f"catalog database must be inside runtime root: {database}")
    if not database.is_file():
        raise FileNotFoundError(database)
    payload: dict[str, Any] = {
        "pointer_version": CATALOG_POINTER_VERSION,
        "generation_id": generation_id,
        "database_relative_path": database.relative_to(runtime).as_posix(),
    }
    if manifest_path is not None:
        manifest = manifest_path.resolve()
        if manifest.is_relative_to(runtime):
            payload["manifest_relative_path"] = manifest.relative_to(runtime).as_posix()
    pointer = catalog_pointer_path(root)
    _atomic_write_text(pointer, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return pointer
