"""Durable Outbox & Reconciliation for YouTube Pitch Parking Lot Ideas.

This module provides atomic file persistence in Vault and a reconciliation workflow
independent of SQLite database availability.
"""
import hashlib
import json
import logging
import os
import re
import sqlite3
import unicodedata
import uuid
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

_SYNC_HANDLER: Optional[Callable[[List[str], str, Optional[str]], int]] = None


def set_sync_handler(handler: Optional[Callable[[List[str], str, Optional[str]], int]]) -> None:
    """Configure the outbound parking-lot persistence port at composition time."""
    global _SYNC_HANDLER
    _SYNC_HANDLER = handler


def _get_outbox_dir(vault_root: Path) -> Path:
    """Return the canonical outbox directory in Vault."""
    root = Path(vault_root)
    if (root / "30_Knowledge_Base").is_dir():
        root = root / "30_Knowledge_Base"
    outbox_dir = root / "NotebookLM_Sources" / "outbox"
    outbox_dir.mkdir(parents=True, exist_ok=True)
    return outbox_dir


def _get_quarantine_dir(vault_root: Path) -> Path:
    """Return quarantine directory for corrupt outbox records."""
    q_dir = _get_outbox_dir(vault_root) / "quarantine"
    q_dir.mkdir(parents=True, exist_ok=True)
    return q_dir


def write_parking_lot_outbox_atomic(
    vault_root: Path,
    job_id: str,
    pitch_id: str,
    ideas: List[str],
) -> Path:
    """Atomically write parking lot ideas to Vault outbox with pending status."""
    outbox_dir = _get_outbox_dir(vault_root)
    clean_job = (job_id or "job").replace("/", "_").replace("\\", "_")
    clean_pitch = (pitch_id or "pitch").replace("/", "_").replace("\\", "_")
    target_file = outbox_dir / f"parking_{clean_job}_{clean_pitch}.json"
    temp_file = outbox_dir / f".tmp_parking_{clean_job}_{clean_pitch}_{uuid.uuid4().hex[:8]}.json"

    idea_records = []
    for idea in ideas:
        norm = unicodedata.normalize("NFC", str(idea)).strip()
        if not norm:
            continue
        idea_hash = hashlib.sha256(norm.casefold().encode("utf-8")).hexdigest()
        idea_records.append({
            "hash": idea_hash,
            "text": norm,
            "source_pitch_id": pitch_id,
        })

    payload = {
        "schema_version": "1.0",
        "job_id": job_id,
        "pitch_id": pitch_id,
        "created_at": datetime.now().isoformat(),
        "sync_status": "pending",
        "ideas": idea_records,
    }

    try:
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        temp_file.replace(target_file)
        logger.info("Saved atomic outbox record: %s", target_file)
        return target_file
    except Exception as e:
        if temp_file.exists():
            try:
                temp_file.unlink()
            except OSError:
                pass
        logger.error("Failed writing parking lot outbox record: %s", e)
        raise


def mark_outbox_synced_atomic(outbox_file: Path) -> None:
    """Atomically update outbox record to synced status."""
    if not outbox_file.exists():
        return
    parent = outbox_file.parent
    temp_file = parent / f".tmp_sync_{uuid.uuid4().hex[:8]}_{outbox_file.name}"

    try:
        with open(outbox_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        data["sync_status"] = "synced"
        data["synced_at"] = datetime.now().isoformat()

        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        temp_file.replace(outbox_file)
        logger.debug("Marked outbox record synced: %s", outbox_file)
    except Exception as e:
        if temp_file.exists():
            try:
                temp_file.unlink()
            except OSError:
                pass
        logger.warning("Failed to mark outbox record as synced for %s: %s", outbox_file, e)


def _sync_ideas_to_sqlite(ideas: List[str], source_pitch_id: str, db_path: Optional[str] = None) -> int:
    if _SYNC_HANDLER is not None:
        return _SYNC_HANDLER(ideas, source_pitch_id, db_path)

    path = db_path or os.getenv("WEBUI_STATE_DB_PATH", "./data/webui_state.db")
    created_count = 0
    now = datetime.now().timestamp()
    try:
        with closing(sqlite3.connect(path, timeout=30)) as conn:
            conn.isolation_level = None
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS kanban_cards (
                    card_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    column_name TEXT NOT NULL,
                    job_id TEXT,
                    flow TEXT DEFAULT 'agent_orchestrator',
                    display_seq INTEGER,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    prompt TEXT,
                    scope TEXT DEFAULT 'both',
                    is_verified INTEGER DEFAULT 0,
                    discord_notify INTEGER DEFAULT 1,
                    discord_sent_events TEXT DEFAULT '[]'
                )
                """
            )
            conn.execute("BEGIN IMMEDIATE")
            try:
                cur_seq_row = conn.execute("SELECT COALESCE(MAX(display_seq), 0) FROM kanban_cards").fetchone()
                cur_seq = cur_seq_row[0] if cur_seq_row else 0
                for raw_idea in ideas[:5]:
                    norm = unicodedata.normalize("NFC", str(raw_idea)).strip()
                    norm = re.sub(r"\s+", " ", norm)
                    if not norm:
                        continue
                    card_id = f"parking:{hashlib.sha256(norm.casefold().encode('utf-8')).hexdigest()}"
                    prompt = f"ไอเดียต่อยอดจาก YouTube Pitch ({source_pitch_id}): {norm}"
                    cur = conn.execute(
                        "INSERT OR IGNORE INTO kanban_cards (card_id, title, column_name, job_id, flow, display_seq, prompt, scope, is_verified, created_at, updated_at) "
                        "VALUES (?, ?, 'backlog', NULL, 'youtube_pitch', ?, ?, 'both', 1, ?, ?)",
                        (card_id, norm, cur_seq + 1, prompt, now, now),
                    )
                    if cur.rowcount > 0:
                        cur_seq += 1
                        created_count += 1
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
    except Exception as exc:
        logger.warning("Could not sync ideas to sqlite directly: %s", exc)
        raise
    return created_count


def reconcile_parking_lot_outbox(
    vault_root: Path,
    sync_fn: Optional[Callable[[List[str], str, Optional[str]], int]] = None,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Scan outbox directory for pending records, sync to SQLite, and quarantine corrupted files."""
    sync_handler = sync_fn or (lambda ids, pid, path: _sync_ideas_to_sqlite(ids, pid, path))

    outbox_dir = _get_outbox_dir(vault_root)
    quarantine_dir = _get_quarantine_dir(vault_root)
    
    summary: Dict[str, Any] = {
        "scanned_count": 0,
        "reconciled_count": 0,
        "already_synced_count": 0,
        "quarantined_count": 0,
        "warnings": [],
    }

    if not outbox_dir.exists():
        return summary

    for entry in outbox_dir.iterdir():
        if entry.is_dir() or entry.name.startswith(".tmp_") or not entry.name.endswith(".json"):
            continue

        summary["scanned_count"] += 1
        try:
            with open(entry, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as err:
            # Corrupted JSON file -> move to quarantine
            corrupt_dest = quarantine_dir / f"{entry.name}.corrupted_{int(datetime.now().timestamp())}"
            try:
                entry.replace(corrupt_dest)
                summary["quarantined_count"] += 1
                msg = f"Quarantined corrupt outbox file {entry.name} -> {corrupt_dest.name} (error: {err})"
                logger.warning(msg)
                summary["warnings"].append(msg)
            except Exception as move_err:
                msg = f"Failed to quarantine corrupt file {entry.name}: {move_err}"
                logger.error(msg)
                summary["warnings"].append(msg)
            continue

        sync_status = data.get("sync_status")
        if sync_status == "synced":
            summary["already_synced_count"] += 1
            continue

        ideas = [item.get("text", "") for item in data.get("ideas", []) if isinstance(item, dict) and item.get("text")]
        pitch_id = data.get("pitch_id", "reconcile")

        if ideas:
            try:
                sync_handler(ideas, pitch_id, db_path)
                mark_outbox_synced_atomic(entry)
                summary["reconciled_count"] += len(ideas)
            except Exception as insert_err:
                msg = f"Failed to reconcile parking lot outbox file {entry.name}: {insert_err}"
                logger.warning(msg)
                summary["warnings"].append(msg)

    return summary
