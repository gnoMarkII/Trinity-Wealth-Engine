"""Point-in-time backup and safe staging restore for the Vault platform.

The bundle contains the canonical Vault snapshot, online SQLite backups for
durable runtime stores, and hashes/metadata for rebuildable read models.  It
never copies SQLite WAL/SHM sidecars as if they were an independent backup.
Restore is staging-only unless a caller explicitly performs a separately
reviewed activation step.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from tools.archivist.runtime_layout import VaultRuntimeLayout, runtime_layout, vault_id
from tools.archivist.schema_registry import load_default_registry
from tools.archivist.vault_backup import create_vault_snapshot, restore_vault_snapshot


BUNDLE_VERSION = 1
_SQLITE_SUFFIXES = {".sqlite3", ".sqlite", ".db"}
_TRANSIENT_SUFFIXES = {"-wal", "-shm", "-journal"}


class RecoveryBundleError(RuntimeError):
    """Raised when a recovery bundle is incomplete or unsafe."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise RecoveryBundleError(f"path escapes recovery root: {path}") from exc


def sqlite_online_backup(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        source_conn = sqlite3.connect(f"file:{source.resolve().as_posix()}?mode=ro", uri=True, timeout=30.0)
    except sqlite3.Error as exc:
        raise RecoveryBundleError(f"cannot open SQLite source for backup: {source}: {exc}") from exc
    target_conn = sqlite3.connect(str(destination), timeout=30.0)
    try:
        source_conn.backup(target_conn, pages=1000, sleep=0.05)
        target_conn.commit()
        integrity = str(target_conn.execute("PRAGMA integrity_check").fetchone()[0])
        if integrity.lower() != "ok":
            raise RecoveryBundleError(f"SQLite integrity check failed for {source}: {integrity}")
        table_counts: dict[str, int] = {}
        for row in target_conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
            table = str(row[0])
            quoted = '"' + table.replace('"', '""') + '"'
            table_counts[table] = int(target_conn.execute(f"SELECT COUNT(*) FROM {quoted}").fetchone()[0])
    except sqlite3.Error as exc:
        raise RecoveryBundleError(f"SQLite online backup failed for {source}: {exc}") from exc
    finally:
        source_conn.close()
        target_conn.close()
    return {"source": str(source), "destination": str(destination), "sha256": sha256_file(destination), "table_counts": table_counts}


def _copy_runtime_tree(source_root: Path, destination_root: Path) -> dict[str, Any]:
    if not source_root.is_dir():
        return {"source_root": str(source_root), "present": False, "files": [], "databases": []}
    files: list[dict[str, Any]] = []
    databases: list[dict[str, Any]] = []
    for source in sorted(source_root.rglob("*")):
        if not source.is_file():
            continue
        if any(source.name.endswith(suffix) for suffix in _TRANSIENT_SUFFIXES):
            continue
        relative = Path(_safe_relative(source, source_root))
        destination = destination_root / relative
        if source.suffix.lower() in _SQLITE_SUFFIXES:
            databases.append(sqlite_online_backup(source, destination))
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            files.append({"relative_path": relative.as_posix(), "sha256": sha256_file(destination), "size": destination.stat().st_size})
    return {"source_root": str(source_root), "present": True, "files": files, "databases": databases}


def _read_only_connection(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(
        f"file:{path.resolve().as_posix()}?mode=ro",
        uri=True,
        timeout=30.0,
    )


def _broker_runtime_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"present": False}
    with _read_only_connection(path) as conn:
        try:
            status_counts = {
                str(row[0]): int(row[1])
                for row in conn.execute(
                    "SELECT status, COUNT(*) FROM broker_commands GROUP BY status ORDER BY status"
                ).fetchall()
            }
            command_count = int(conn.execute("SELECT COUNT(*) FROM broker_commands").fetchone()[0])
            event_count, max_event_id = conn.execute(
                "SELECT COUNT(*), COALESCE(MAX(event_id), 0) FROM broker_events"
            ).fetchone()
            last_updated = conn.execute("SELECT MAX(updated_at) FROM broker_commands").fetchone()[0]
        except sqlite3.Error:
            tables = [str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()]
            return {"present": True, "schema_compatible": False, "tables": tables}
    return {
        "present": True,
        "command_count": command_count,
        "receipt_count": command_count,
        "event_count": int(event_count or 0),
        "event_high_water_mark": int(max_event_id or 0),
        "status_counts": status_counts,
        "last_updated_at": str(last_updated) if last_updated else None,
        "pending_count": sum(status_counts.get(item, 0) for item in ("accepted", "retry_wait", "leased", "committing")),
    }


def _portfolio_runtime_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"present": False}
    with _read_only_connection(path) as conn:
        try:
            rows = conn.execute(
                "SELECT event_id, portfolio_id, sequence, event_type, payload_json, occurred_at "
                "FROM portfolio_events ORDER BY portfolio_id, sequence"
            ).fetchall()
        except sqlite3.Error:
            tables = [str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()]
            return {"present": True, "schema_compatible": False, "tables": tables}
    streams: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        streams.setdefault(str(row[1]), []).append(
            {
                "event_id": str(row[0]),
                "portfolio_id": str(row[1]),
                "sequence": int(row[2]),
                "event_type": str(row[3]),
                "payload": json.loads(str(row[4])),
                "occurred_at": str(row[5]),
            }
        )
    stream_state: dict[str, Any] = {}
    for portfolio_id, events in streams.items():
        state: dict[str, Any] = {}
        for event in events:
            payload = event["payload"]
            event_type = event["event_type"]
            if event_type == "state_snapshot":
                state = dict(payload.get("state") or {})
            elif event_type in {"state_patch", "transaction"}:
                state.update(dict(payload.get("state_patch") or payload.get("state") or {}))
            elif event_type == "delete_field":
                state.pop(str(payload.get("field") or ""), None)
        stream_json = json.dumps(events, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        state_json = json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        stream_state[portfolio_id] = {
            "event_count": len(events),
            "sequence_high_water_mark": int(events[-1]["sequence"]) if events else 0,
            "stream_hash": hashlib.sha256(stream_json.encode("utf-8")).hexdigest(),
            "state_hash": hashlib.sha256(state_json.encode("utf-8")).hexdigest(),
        }
    return {
        "present": True,
        "event_count": sum(len(items) for items in streams.values()),
        "portfolio_count": len(streams),
        "streams": stream_state,
    }


def _pointer_payload(vault: Path, filename: str) -> dict[str, Any]:
    path = vault / ".system" / filename
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _catalog_runtime_state(vault: Path, layout: VaultRuntimeLayout) -> dict[str, Any]:
    pointer = _pointer_payload(vault, "catalog_generation_active.json")
    relative = str(pointer.get("database_relative_path") or "").strip()
    path = (layout.root / relative).resolve() if relative else layout.catalog_root / "active" / "vault_catalog.db"
    if not path.is_file():
        return {"present": False, "generation_id": pointer.get("generation_id")}
    with _read_only_connection(path) as conn:
        integrity = str(conn.execute("PRAGMA integrity_check").fetchone()[0])
        counts: dict[str, int] = {}
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall():
            table = str(row[0])
            quoted = '"' + table.replace('"', '""') + '"'
            counts[table] = int(conn.execute(f"SELECT COUNT(*) FROM {quoted}").fetchone()[0])
    manifest_path = path.parent / "manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                manifest = raw
        except (OSError, UnicodeDecodeError, ValueError):
            manifest = {}
    return {
        "present": True,
        "path": str(path),
        "generation_id": str(pointer.get("generation_id") or manifest.get("generation_id") or ""),
        "integrity_check": integrity,
        "counts": counts,
        "registry_digest": manifest.get("registry_digest"),
        "policy_digest": manifest.get("policy_digest"),
        "eligible_set_fingerprint": manifest.get("eligible_set_fingerprint"),
        "eligible_note_count": manifest.get("eligible_note_count"),
    }


def _vector_runtime_state(vault: Path, layout: VaultRuntimeLayout) -> dict[str, Any]:
    pointer = _pointer_payload(vault, "vector_generation_active.json")
    generation_id = str(pointer.get("generation_id") or "")
    relative = str(pointer.get("manifest_relative_path") or "").strip()
    manifest_path = (layout.vector_root / relative).resolve() if relative else layout.vector_root / "generations" / f"{generation_id}.json"
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                manifest = raw
        except (OSError, UnicodeDecodeError, ValueError):
            manifest = {}
    state_path = layout.vector_root / "vector_index_state.json"
    state: dict[str, Any] = {}
    if state_path.is_file():
        try:
            raw = json.loads(state_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                state = raw
        except (OSError, UnicodeDecodeError, ValueError):
            state = {}
    return {
        "present": bool(manifest or state or generation_id),
        "generation_id": generation_id or manifest.get("generation_id") or state.get("generation_id"),
        "collection_name": pointer.get("collection_name") or manifest.get("collection_name") or state.get("collection_name"),
        "registry_digest": manifest.get("registry_digest") or state.get("registry_digest"),
        "policy_digest": manifest.get("policy_digest") or state.get("policy_digest"),
        "eligible_set_fingerprint": manifest.get("eligible_set_fingerprint") or manifest.get("corpus_fingerprint") or state.get("corpus_fingerprint"),
        "eligible_note_count": manifest.get("eligible_note_count") or state.get("eligible_note_count"),
        "eligible_chunk_count": manifest.get("eligible_chunk_count") or state.get("eligible_chunk_count"),
        "manifest_path": str(manifest_path),
    }


def capture_runtime_state(
    vault_root: str | Path,
    *,
    runtime_base: str | Path | None = None,
    runtime_root: str | Path | None = None,
) -> dict[str, Any]:
    """Capture high-water marks and derived generation metadata read-only."""

    vault = Path(vault_root).resolve()
    if runtime_root is not None and runtime_base is not None:
        raise RecoveryBundleError("provide either runtime_root or runtime_base")
    if runtime_root is None:
        layout = runtime_layout(vault, runtime_base, create=False)
    else:
        root = Path(runtime_root).resolve()
        layout = VaultRuntimeLayout(vault_root=vault, root=root, vault_id=vault_id(vault))
    registry = load_default_registry()
    return {
        "captured_at": _utc_now(),
        "runtime_layout": layout.as_dict(),
        "registry_digest": registry.digest(),
        "policy_digest": registry.policy_digest(),
        "broker": _broker_runtime_state(layout.broker_db),
        "portfolio": _portfolio_runtime_state(layout.portfolio_db),
        "catalog": _catalog_runtime_state(vault, layout),
        "vector": _vector_runtime_state(vault, layout),
    }


def _bundle_files(bundle_root: Path, excluded: Iterable[Path] = ()) -> list[dict[str, Any]]:
    excluded_set = {path.resolve() for path in excluded}
    result: list[dict[str, Any]] = []
    for path in sorted(bundle_root.rglob("*")):
        if not path.is_file() or path.resolve() in excluded_set:
            continue
        result.append({"path": path.relative_to(bundle_root).as_posix(), "sha256": sha256_file(path), "size": path.stat().st_size})
    return result


def _copy_contract_snapshot(vault: Path, metadata_root: Path) -> dict[str, Any]:
    workspace_root = Path(__file__).resolve().parents[2]
    candidates = [
        vault / ".system" / "storage_contract.json",
        workspace_root / "schemas" / "vault" / "registry.json",
        workspace_root / "schemas" / "vault" / "policies" / "indexing.json",
        workspace_root / "schemas" / "vault" / "policies" / "field-ownership.json",
        workspace_root / "schemas" / "vault" / "policies" / "retention.json",
    ]
    copied: list[dict[str, Any]] = []
    for source in candidates:
        if not source.is_file():
            continue
        name = source.name if source.parent.name == ".system" else f"{source.parent.name}-{source.name}"
        destination = metadata_root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append({"source": str(source.resolve()), "path": destination.relative_to(metadata_root).as_posix(), "sha256": sha256_file(destination)})
    registry = load_default_registry()
    return {"files": copied, "registry_digest": registry.digest(), "policy_digest": registry.policy_digest()}


def create_recovery_bundle(
    *,
    vault_root: str | Path,
    output_dir: str | Path,
    runtime_base: str | Path | None = None,
    extra_runtime_roots: Optional[Mapping[str, str | Path]] = None,
    run_id: Optional[str] = None,
) -> dict[str, Any]:
    """Create a hashed Vault-platform recovery bundle without mutating source state."""

    vault = Path(vault_root).resolve()
    if not vault.is_dir():
        raise RecoveryBundleError(f"Vault does not exist: {vault}")
    layout = runtime_layout(vault, runtime_base, create=False)
    bundle_root = Path(output_dir).resolve() / (run_id or f"bundle-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    if bundle_root.exists():
        raise RecoveryBundleError(f"recovery bundle destination already exists: {bundle_root}")
    bundle_root.mkdir(parents=True, exist_ok=False)
    snapshot_path, snapshot_sha = create_vault_snapshot(vault_root=vault, backup_dir=bundle_root / "vault")
    runtime_records: dict[str, Any] = {
        "canonical": _copy_runtime_tree(layout.root, bundle_root / "runtime" / layout.vault_id),
    }
    for label, source in (extra_runtime_roots or {}).items():
        safe_label = "".join(character if character.isalnum() or character in "-_" else "_" for character in str(label)) or "legacy"
        runtime_records[safe_label] = _copy_runtime_tree(Path(source).resolve(), bundle_root / "runtime" / safe_label)
    contract = _copy_contract_snapshot(vault, bundle_root / "metadata")
    runtime_state = capture_runtime_state(vault, runtime_base=runtime_base)
    manifest: dict[str, Any] = {
        "bundle_version": BUNDLE_VERSION,
        "created_at": _utc_now(),
        "vault_root": str(vault),
        "vault_id": layout.vault_id,
        "runtime_layout": layout.as_dict(),
        "vault_snapshot": {
            "path": snapshot_path.relative_to(bundle_root).as_posix(),
            "sha256": snapshot_sha,
            "size": snapshot_path.stat().st_size,
        },
        "runtime": runtime_records,
        "runtime_state": runtime_state,
        "contract": contract,
        "derived_rebuild": {
            "catalog": "rebuild from restored Vault and registry/policy digests",
            "vector": "rebuild from restored Vault and eligible-set policy digest",
        },
    }
    manifest_path = bundle_root / "bundle-manifest.json"
    manifest["files"] = _bundle_files(bundle_root, excluded=(manifest_path,))
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"status": "PASS", "bundle_root": str(bundle_root), "manifest": str(manifest_path), "manifest_sha256": sha256_file(manifest_path), "vault_snapshot_sha256": snapshot_sha}


def _validate_bundle_files(bundle_root: Path, manifest: Mapping[str, Any]) -> None:
    for record in manifest.get("files") or []:
        relative = Path(str(record.get("path") or ""))
        if relative.is_absolute() or ".." in relative.parts:
            raise RecoveryBundleError(f"unsafe bundle member: {relative}")
        path = (bundle_root / relative).resolve()
        if not path.is_relative_to(bundle_root) or not path.is_file():
            raise RecoveryBundleError(f"bundle member is missing: {relative}")
        actual = sha256_file(path)
        if actual != str(record.get("sha256") or ""):
            raise RecoveryBundleError(f"bundle member hash mismatch: {relative}")


def restore_recovery_bundle(*, bundle_root: str | Path, restore_root: str | Path) -> dict[str, Any]:
    """Restore a bundle into a new staging root and verify all recorded hashes."""

    bundle = Path(bundle_root).resolve()
    manifest_path = bundle / "bundle-manifest.json"
    if not manifest_path.is_file():
        raise RecoveryBundleError(f"bundle manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("bundle_version") != BUNDLE_VERSION:
        raise RecoveryBundleError(f"unsupported recovery bundle version: {manifest.get('bundle_version')}")
    _validate_bundle_files(bundle, manifest)
    destination = Path(restore_root).resolve()
    if destination.exists():
        raise RecoveryBundleError(f"restore destination already exists: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    snapshot = bundle / Path(str(manifest["vault_snapshot"]["path"]))
    restored_vault = destination / "vault"
    restored_count = restore_vault_snapshot(snapshot, restored_vault, verify_checksum=True)
    runtime_source = bundle / "runtime"
    restored_runtime = destination / "runtime"
    if runtime_source.is_dir():
        shutil.copytree(runtime_source, restored_runtime)
    metadata_source = bundle / "metadata"
    restored_metadata = destination / "metadata"
    if metadata_source.is_dir():
        shutil.copytree(metadata_source, restored_metadata)

    # Verify the restored external runtime and contract snapshot against the
    # same manifest records that were validated before the copy.  The Vault
    # archive itself is verified by restore_vault_snapshot; this closes the
    # remaining gap for broker/portfolio/catalog/vector state.
    restored_hashes: list[dict[str, Any]] = []
    for record in manifest.get("files") or []:
        relative = Path(str(record.get("path") or ""))
        if not relative.parts or relative.parts[0] not in {"runtime", "metadata"}:
            continue
        target = (destination / relative).resolve()
        if not target.is_file():
            raise RecoveryBundleError(f"restored bundle member is missing: {relative}")
        actual = sha256_file(target)
        expected = str(record.get("sha256") or "")
        if actual != expected:
            raise RecoveryBundleError(f"restored bundle member hash mismatch: {relative}")
        restored_hashes.append({"path": relative.as_posix(), "sha256": actual})
    result = {
        "status": "PASS",
        "bundle_root": str(bundle),
        "restore_root": str(destination),
        "restored_vault": str(restored_vault),
        "restored_runtime": str(restored_runtime),
        "restored_metadata": str(restored_metadata),
        "restored_vault_files": restored_count,
        "restored_external_files_verified": len(restored_hashes),
        "vault_snapshot_sha256": str(manifest["vault_snapshot"]["sha256"]),
        "registry_digest": str((manifest.get("contract") or {}).get("registry_digest") or ""),
        "policy_digest": str((manifest.get("contract") or {}).get("policy_digest") or ""),
    }
    (destination / "restore-report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result
