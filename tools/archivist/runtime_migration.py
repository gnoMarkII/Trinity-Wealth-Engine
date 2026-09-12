"""Idempotent migration from legacy runtime locations to the R9 layout."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable, Optional

from tools.archivist.recovery_bundle import sha256_file, sqlite_online_backup
from tools.archivist.runtime_layout import VaultRuntimeLayout, runtime_base_for, runtime_layout


class RuntimeMigrationError(RuntimeError):
    """Raised when runtime migration would overwrite divergent state."""


_DB_SUFFIXES = {".sqlite3", ".sqlite", ".db"}
_TRANSIENT_SUFFIXES = ("-wal", "-shm", "-journal")


def _same_file(left: Path, right: Path) -> bool:
    return left.is_file() and right.is_file() and left.stat().st_size == right.stat().st_size and sha256_file(left) == sha256_file(right)


def _sqlite_is_empty(path: Path) -> bool:
    import sqlite3

    try:
        with sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True, timeout=30.0) as conn:
            tables = [str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
            for table in tables:
                quoted = '"' + table.replace('"', '""') + '"'
                if int(conn.execute(f"SELECT COUNT(*) FROM {quoted}").fetchone()[0]) > 0:
                    return False
        return True
    except Exception:
        return False


def _legacy_sources(vault: Path, base: Path, layout: VaultRuntimeLayout) -> list[tuple[str, Path, Path]]:
    candidates = [
        ("legacy-broker", base / "broker", layout.root / "broker"),
        ("legacy-portfolio", base / "portfolio", layout.root / "portfolio"),
        ("legacy-reconciliation", base / "reconciliation", layout.root / "reconciliation"),
        ("legacy-catalog", base / "catalog", layout.root / "catalog"),
        ("legacy-vector", vault.parent / "data" / "vector_runtime" / "vault_v2", layout.vector_root),
    ]
    return [(label, source.resolve(), destination.resolve()) for label, source, destination in candidates if source.is_dir() and source.resolve() != destination.resolve()]


def _iter_files(source: Path) -> Iterable[Path]:
    for path in sorted(source.rglob("*")):
        if path.is_file() and not any(path.name.endswith(suffix) for suffix in _TRANSIENT_SUFFIXES):
            yield path


def _copy_one(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() in _DB_SUFFIXES and destination.is_file() and _sqlite_is_empty(destination):
        # Windows may keep an empty SQLite file open briefly after a read-only
        # probe.  An online backup into the validated empty destination avoids
        # a rename over an open handle while preserving the same source data.
        sqlite_online_backup(source, destination)
        return {"source": str(source), "destination": str(destination), "status": "copied", "sha256": sha256_file(destination), "size": destination.stat().st_size}
    temporary = destination.with_name(f".{destination.name}.r9-migration.tmp")
    temporary.unlink(missing_ok=True)
    try:
        if source.suffix.lower() in _DB_SUFFIXES:
            sqlite_online_backup(source, temporary)
        else:
            shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {"source": str(source), "destination": str(destination), "status": "copied", "sha256": sha256_file(destination), "size": destination.stat().st_size}


def migrate_runtime_layout(
    *,
    vault_root: str | Path,
    runtime_base: str | Path | None = None,
    apply: bool = False,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Plan or apply a non-destructive runtime migration.

    Existing divergent destination files are never overwritten.  Legacy
    sources remain in place so rollback can select them until the operator
    removes the retention hold.
    """

    vault = Path(vault_root).resolve()
    base = runtime_base_for(vault, runtime_base)
    layout = runtime_layout(vault, base, create=apply)
    sources = _legacy_sources(vault, base, layout)
    entries: list[dict[str, Any]] = []
    conflicts: list[dict[str, str]] = []
    for label, source_root, destination_root in sources:
        for source in _iter_files(source_root):
            relative = source.relative_to(source_root)
            destination = destination_root / relative
            if destination.is_file():
                if _same_file(source, destination):
                    entries.append({"label": label, "source": str(source), "destination": str(destination), "status": "already_present", "sha256": sha256_file(destination)})
                elif source.suffix.lower() in _DB_SUFFIXES and _sqlite_is_empty(destination):
                    if apply:
                        entries.append({"label": label, **_copy_one(source, destination), "status": "replaced_empty_destination"})
                    else:
                        entries.append({"label": label, "source": str(source), "destination": str(destination), "status": "planned_replace_empty", "sha256": sha256_file(source)})
                else:
                    conflict = {"label": label, "source": str(source), "destination": str(destination), "status": "divergent_destination"}
                    conflicts.append(conflict)
                    entries.append(conflict)
            elif destination.exists():
                conflict = {"label": label, "source": str(source), "destination": str(destination), "status": "destination_not_file"}
                conflicts.append(conflict)
                entries.append(conflict)
            elif apply:
                entries.append({"label": label, **_copy_one(source, destination)})
            else:
                entries.append({"label": label, "source": str(source), "destination": str(destination), "status": "planned", "sha256": sha256_file(source)})
    if conflicts:
        result = {"status": "CONFLICT", "vault_root": str(vault), "runtime_base": str(base), "runtime_layout": layout.as_dict(), "entries": entries, "conflicts": conflicts, "applied": False}
    else:
        config_path = vault / ".system" / "vault_config.json"
        config_changed = False
        if apply:
            try:
                payload = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
            except (OSError, UnicodeDecodeError, ValueError) as exc:
                raise RuntimeMigrationError(f"cannot read Vault config: {config_path}: {exc}") from exc
            if payload.get("vault_id") != layout.vault_id:
                payload["vault_id"] = layout.vault_id
                temporary = config_path.with_name(f".{config_path.name}.r9-migration.tmp")
                temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                os.replace(temporary, config_path)
                config_changed = True
        result = {"status": "PASS", "vault_root": str(vault), "runtime_base": str(base), "runtime_layout": layout.as_dict(), "entries": entries, "conflicts": [], "applied": apply, "config_vault_id_changed": config_changed}
    if output is not None:
        destination = Path(output).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result
