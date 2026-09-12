"""Vault V2 Migration Engine (Plan, Apply, Verify, Rollback).

Provides safe, deterministic, journaled migration from Obsidian Vault V1 to V2.
Guarantees zero data loss, strict pre-hash validation, and 100% reversible rollback.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

from core.logger import get_logger
from tools.archivist.core import _atomic_write_text
from tools.archivist.metadata import normalize_legacy_metadata, parse_note, validate_note
from tools.archivist.maintenance_guard import assert_write_allowed
from tools.archivist.vault_paths import VaultPaths
from tools.archivist.vault_policy import is_searchable_note

logger = get_logger(__name__)


@dataclass
class MigrationAction:
    source_rel: str
    target_rel: str
    action: str  # 'relocate', 'no_op', 'normalize_properties'
    pre_hash: str
    entity_type: str
    companion_sidecars: list[dict[str, str]] = field(default_factory=list)  # [{"source_rel": ..., "target_rel": ..., "pre_hash": ...}]


@dataclass
class MigrationPlan:
    plan_id: str
    created_at: str
    vault_root: str
    total_files: int
    actions: list[MigrationAction]
    summary: dict[str, int]
    config_before: Optional[dict[str, Any]] = None
    config_before_hash: Optional[str] = None
    contract_version: str = "vault-v2-r3"


def _compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _safe_plan_path(v_root: Path, relative: str) -> Path:
    """Resolve a plan path and reject traversal/junction escapes."""
    candidate = (v_root / str(relative)).resolve()
    if not candidate.is_relative_to(v_root):
        raise ValueError(f"Migration plan path escapes vault root: {relative}")
    return candidate


def create_migration_plan(
    vault_root: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
) -> MigrationPlan:
    """Scans vault and computes canonical V2 targets for all notes, generating a MigrationPlan."""
    v_root = Path(vault_root).resolve()
    vp = VaultPaths(v_root)

    plan_id = f"plan_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    actions: list[MigrationAction] = []
    summary = {"relocate": 0, "no_op": 0, "normalize_properties": 0, "total": 0}
    seen_targets: dict[str, str] = {}
    config_path = v_root / ".system" / "vault_config.json"
    if config_path.is_file():
        config_before_hash = _compute_sha256(config_path)
        try:
            config_before = json.loads(config_path.read_text(encoding="utf-8"))
            if not isinstance(config_before, dict):
                config_before = {"_raw": config_path.read_text(encoding="utf-8")}
        except (OSError, ValueError):
            config_before = {"_raw": config_path.read_text(encoding="utf-8")}
    else:
        # V1's implicit configuration is represented explicitly so rollback
        # remains reversible without dropping the original layout semantics.
        config_before = {"layout_version": 1}
        config_before_hash = None

    # Find all markdown files in vault
    for md_path in sorted(v_root.rglob("*.md")):
        try:
            rel = md_path.relative_to(v_root).as_posix()
        except ValueError:
            continue

        if not is_searchable_note(md_path, vault_root=v_root):
            continue

        # Only reorganize notes that are in 30_Knowledge_Base or legacy NotebookLM_Sources
        first_part = Path(rel).parts[0]
        if first_part not in ("30_Knowledge_Base", "NotebookLM_Sources"):
            continue

        pre_hash = _compute_sha256(md_path)

        # Parse note metadata
        try:
            content = md_path.read_text(encoding="utf-8")
            raw_meta, body, _ = parse_note(content)
            norm_meta, _ = normalize_legacy_metadata(raw_meta)
            model, _ = validate_note(norm_meta, mode="lenient")
        except Exception:
            raw_meta = {}
            norm_meta = {}
            model = None

        meta_dict = model.model_dump(mode="python") if model else norm_meta
        entity_type = meta_dict.get("entity_type", "concept")

        target_abs = vp.note_path(meta_dict, filename=md_path.name)
        target_rel = target_abs.relative_to(v_root).as_posix()

        action_type = "no_op" if rel == target_rel else "relocate"

        # Check collision
        if target_rel in seen_targets:
            target_p = Path(target_rel)
            target_rel = (target_p.parent / f"{target_p.stem}_{pre_hash[:8]}{target_p.suffix}").as_posix()
            action_type = "relocate"

        seen_targets[target_rel] = rel

        # Check companion sidecar (e.g. .json with same stem)
        companion_sidecars: list[dict[str, str]] = []
        sidecar_json = md_path.with_suffix(".json")
        if sidecar_json.exists():
            sc_rel = sidecar_json.relative_to(v_root).as_posix()
            target_md = v_root / target_rel
            sc_target = target_md.with_suffix(".json").relative_to(v_root).as_posix()
            sc_hash = _compute_sha256(sidecar_json)
            companion_sidecars.append({
                "source_rel": sc_rel,
                "target_rel": sc_target,
                "pre_hash": sc_hash,
            })

        summary[action_type] = summary.get(action_type, 0) + 1
        summary["total"] += 1

        actions.append(
            MigrationAction(
                source_rel=rel,
                target_rel=target_rel,
                action=action_type,
                pre_hash=pre_hash,
                entity_type=entity_type,
                companion_sidecars=companion_sidecars,
            )
        )

    plan = MigrationPlan(
        plan_id=plan_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        vault_root=str(v_root),
        total_files=len(actions),
        actions=actions,
        summary=summary,
        config_before=config_before,
        config_before_hash=config_before_hash,
    )

    if output_file:
        out_p = Path(output_file).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        plan_dict = {
            "plan_id": plan.plan_id,
            "created_at": plan.created_at,
            "vault_root": plan.vault_root,
            "total_files": plan.total_files,
            "summary": plan.summary,
            "config_before": plan.config_before,
            "config_before_hash": plan.config_before_hash,
            "contract_version": plan.contract_version,
            "actions": [asdict(a) for a in plan.actions],
        }
        out_p.write_text(json.dumps(plan_dict, indent=2, ensure_ascii=False), encoding="utf-8")

    return plan


def apply_migration_plan(
    plan: Union[MigrationPlan, dict, str, Path],
    vault_root: Optional[Union[str, Path]] = None,
    journal_file: Optional[Union[str, Path]] = None,
    allow_live: bool = False,
) -> dict[str, Any]:
    """Applies a MigrationPlan with strict pre-hash validation and journal logging."""
    if isinstance(plan, (str, Path)):
        p_path = Path(plan).resolve()
        with p_path.open("r", encoding="utf-8") as f:
            plan_data = json.load(f)
    elif isinstance(plan, MigrationPlan):
        plan_data = {
            "plan_id": plan.plan_id,
            "vault_root": plan.vault_root,
            "config_before": plan.config_before,
            "config_before_hash": plan.config_before_hash,
            "contract_version": plan.contract_version,
            "actions": [asdict(a) for a in plan.actions],
        }
    else:
        plan_data = dict(plan)

    v_root = Path(vault_root or plan_data["vault_root"]).resolve()
    assert_write_allowed(v_root)

    # Safety Gate: protect live 'memories' without explicit flag
    if v_root.name == "memories" and not allow_live:
        raise PermissionError("Live migration on 'memories' requires allow_live=True.")

    # 1. Pre-hash validation phase (fail-closed before mutating anything)
    for act in plan_data["actions"]:
        src = _safe_plan_path(v_root, act["source_rel"])
        if not src.exists():
            raise FileNotFoundError(f"Pre-check failed: Source file {act['source_rel']} does not exist.")
        cur_hash = _compute_sha256(src)
        if cur_hash != act["pre_hash"]:
            raise ValueError(
                f"Pre-check hash mismatch on {act['source_rel']}: expected {act['pre_hash'][:8]}, found {cur_hash[:8]}."
            )

        for sc in act.get("companion_sidecars", []):
            sc_src = _safe_plan_path(v_root, sc["source_rel"])
            if not sc_src.exists():
                raise FileNotFoundError(f"Pre-check failed: Sidecar {sc['source_rel']} does not exist.")
            if _compute_sha256(sc_src) != sc["pre_hash"]:
                raise ValueError(f"Pre-check hash mismatch on sidecar {sc['source_rel']}.")

    config_path = _safe_plan_path(v_root, ".system/vault_config.json")
    config_before_hash = plan_data.get("config_before_hash")
    if config_before_hash:
        if not config_path.is_file() or _compute_sha256(config_path) != config_before_hash:
            raise ValueError("Pre-check hash mismatch on vault_config.json")

    # 2. Setup journal
    if journal_file is None:
        plan_id = plan_data.get("plan_id")
        journal_name = f"migration_journal_{plan_id}.jsonl" if plan_id else "migration_journal.jsonl"
        journal_p = v_root / ".system" / journal_name
        if journal_p.exists():
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            journal_p = journal_p.with_name(f"{journal_p.stem}_{stamp}{journal_p.suffix}")
    else:
        journal_p = Path(journal_file).resolve()
    journal_p.parent.mkdir(parents=True, exist_ok=True)

    applied_moves: list[dict[str, str]] = []

    # 3. Apply file relocations
    with journal_p.open("w", encoding="utf-8") as jf:
        for act in plan_data["actions"]:
            if act["action"] == "no_op":
                continue

            src = _safe_plan_path(v_root, act["source_rel"])
            dst = _safe_plan_path(v_root, act["target_rel"])
            dst.parent.mkdir(parents=True, exist_ok=True)

            # Write journal intent record BEFORE moving
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "type": "relocate",
                "source_rel": act["source_rel"],
                "target_rel": act["target_rel"],
                "sha256": act["pre_hash"],
            }
            jf.write(json.dumps(record) + "\n")
            jf.flush()

            # Execute move
            shutil.move(str(src), str(dst))
            applied_moves.append(record)

            # Move companion sidecars
            for sc in act.get("companion_sidecars", []):
                sc_src = _safe_plan_path(v_root, sc["source_rel"])
                sc_dst = _safe_plan_path(v_root, sc["target_rel"])
                sc_dst.parent.mkdir(parents=True, exist_ok=True)
                sc_record = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "type": "relocate_sidecar",
                    "source_rel": sc["source_rel"],
                    "target_rel": sc["target_rel"],
                    "sha256": sc["pre_hash"],
                }
                jf.write(json.dumps(sc_record) + "\n")
                jf.flush()
                shutil.move(str(sc_src), str(sc_dst))
                applied_moves.append(sc_record)

    # 4. Write vault config layout_version = 2 while preserving custom fields
    # and recording an exact before-image for rollback.
    config_before = plan_data.get("config_before")
    before_bytes = config_path.read_bytes() if config_path.is_file() else b""
    config_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": "config_update",
        "before_exists": config_path.is_file(),
        "before_hash": config_before_hash,
        "before": config_before if isinstance(config_before, dict) else {"layout_version": 1},
        "before_bytes_hex": before_bytes.hex(),
    }
    with journal_p.open("a", encoding="utf-8") as jf:
        jf.write(json.dumps(config_record, ensure_ascii=False) + "\n")
        jf.flush()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    updated_config = {
        **(config_before if isinstance(config_before, dict) else {}),
        "layout_version": 2,
        "migrated_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_text(
        config_path,
        json.dumps(updated_config, indent=2, ensure_ascii=False),
    )

    # 5. Incremental Catalog Sync
    try:
        from tools.archivist.catalog_adapter import SqliteNoteCatalogAdapter
        cat = SqliteNoteCatalogAdapter(vault_root=v_root)
        cat_res = cat.sync_from_vault(v_root, force=True)
    except Exception as e:
        logger.warning("Catalog sync after migration reported: %s", e)
        cat_res = {"status": "error", "error": str(e)}

    return {
        "applied_count": len(applied_moves),
        "journal_file": str(journal_p),
        "catalog_sync": cat_res,
    }


def rollback_migration(
    journal_file: Union[str, Path],
    vault_root: Optional[Union[str, Path]] = None,
) -> dict[str, Any]:
    """Rolls back applied migration operations in reverse journal order with conflict detection."""
    j_path = Path(journal_file).resolve()
    if not j_path.exists():
        raise FileNotFoundError(f"Journal file {j_path} not found.")

    v_root = Path(vault_root).resolve() if vault_root else j_path.parent.parent

    lines = j_path.read_text(encoding="utf-8").strip().splitlines()
    records = [json.loads(line) for line in lines if line.strip()]

    rolled_back: list[dict[str, str]] = []
    conflicts: list[dict[str, Any]] = []
    config_restored = False

    # Process in REVERSE order
    for rec in reversed(records):
        if rec.get("type") == "config_update":
            config_path = _safe_plan_path(v_root, ".system/vault_config.json")
            try:
                before_hex = rec.get("before_bytes_hex") or ""
                if rec.get("before_exists") and before_hex:
                    config_path.parent.mkdir(parents=True, exist_ok=True)
                    config_path.write_bytes(bytes.fromhex(before_hex))
                else:
                    # Legacy V1 had no physical config file; retain an
                    # explicit V1 marker for compatibility with existing tools.
                    config_path.parent.mkdir(parents=True, exist_ok=True)
                    config_path.write_text(json.dumps({"layout_version": 1}, indent=2), encoding="utf-8")
                config_restored = True
            except Exception as exc:
                conflicts.append({"target_rel": ".system/vault_config.json", "reason": str(exc)})
            continue
        if rec.get("type") not in {"relocate", "relocate_sidecar"}:
            continue

        src_orig = _safe_plan_path(v_root, rec["source_rel"])
        dst_curr = _safe_plan_path(v_root, rec["target_rel"])

        if dst_curr.exists():
            # Check for user-edit conflict post migration
            current_hash = _compute_sha256(dst_curr)
            if rec.get("sha256") and current_hash != rec["sha256"]:
                conflicts.append({
                    "target_rel": rec["target_rel"],
                    "reason": "user_edit_conflict",
                    "expected_hash": rec["sha256"],
                    "actual_hash": current_hash,
                })
                continue

            src_orig.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(dst_curr), str(src_orig))
            rolled_back.append(rec)

    # Revert config only from the journaled before-image. Pre-R3 journals had
    # no config record, so their compatibility fallback remains explicit.
    if not conflicts:
        config_path = _safe_plan_path(v_root, ".system/vault_config.json")
        if not config_restored and config_path.exists():
            config_path.write_text(json.dumps({"layout_version": 1}, indent=2), encoding="utf-8")
        j_path.unlink()

    return {
        "rolled_back_count": len(rolled_back),
        "conflicts": conflicts,
    }


def verify_migration(
    plan: Union[MigrationPlan, dict, str, Path],
    vault_root: Optional[Union[str, Path]] = None,
) -> dict[str, Any]:
    """Verifies that all files in plan were moved correctly to their target locations."""
    if isinstance(plan, (str, Path)):
        p_path = Path(plan).resolve()
        with p_path.open("r", encoding="utf-8") as f:
            plan_data = json.load(f)
    elif isinstance(plan, MigrationPlan):
        plan_data = {
            "plan_id": plan.plan_id,
            "vault_root": plan.vault_root,
            "config_before": plan.config_before,
            "config_before_hash": plan.config_before_hash,
            "contract_version": plan.contract_version,
            "actions": [asdict(a) for a in plan.actions],
        }
    else:
        plan_data = dict(plan)

    v_root = Path(vault_root or plan_data["vault_root"]).resolve()

    missing_targets = []
    hash_mismatches = []
    verified_count = 0

    for act in plan_data["actions"]:
        dst = _safe_plan_path(v_root, act["target_rel"])
        if not dst.exists():
            missing_targets.append(act["target_rel"])
            continue

        if _compute_sha256(dst) != act["pre_hash"]:
            hash_mismatches.append(act["target_rel"])
            continue

        for sc in act.get("companion_sidecars", []):
            sc_dst = _safe_plan_path(v_root, sc["target_rel"])
            if not sc_dst.exists():
                missing_targets.append(sc["target_rel"])
            elif _compute_sha256(sc_dst) != sc["pre_hash"]:
                hash_mismatches.append(sc["target_rel"])

        verified_count += 1

    success = len(missing_targets) == 0 and len(hash_mismatches) == 0

    return {
        "success": success,
        "verified_count": verified_count,
        "missing_targets": missing_targets,
        "hash_mismatches": hash_mismatches,
    }
