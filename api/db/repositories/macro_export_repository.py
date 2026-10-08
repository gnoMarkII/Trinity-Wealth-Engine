"""Raw SQLite DAO for Macro NotebookLM research export records.

The DAO intentionally does not commit or rollback. A caller-owned
DbUnitOfWork manages the transaction boundary.
"""
from __future__ import annotations

import json
import sqlite3
import time
from typing import Any, Dict, List, Optional


def insert_export(
    conn: sqlite3.Connection,
    *,
    export_id: str,
    request_key: str,
    content_hash: str,
    snapshot_at: str,
    strategy_report_id: Optional[str] = None,
    job_id: Optional[str] = None,
    state: str = "queued",
    stage: str = "initialized",
    manifest_path: Optional[str] = None,
    inventory: Optional[Dict[str, Any]] = None,
    warnings: Optional[List[str]] = None,
) -> sqlite3.Row:
    now = time.time()
    conn.execute(
        """INSERT INTO macro_notebooklm_exports
           (export_id, request_key, content_hash, job_id, state, stage, snapshot_at,
            strategy_report_id, notebook_id, notebook_url, manifest_path,
            inventory_json, warnings_json, error_code, error_message, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?, ?, NULL, NULL, ?, ?)""",
        (
            export_id,
            request_key,
            content_hash,
            job_id,
            state,
            stage,
            snapshot_at,
            strategy_report_id,
            manifest_path,
            json.dumps(inventory or {}, ensure_ascii=False),
            json.dumps(warnings or [], ensure_ascii=False),
            now,
            now,
        ),
    )
    row = get_export_by_id(conn, export_id)
    if row is None:
        raise RuntimeError("Failed to retrieve inserted macro export record")
    return row


def get_export_by_id(conn: sqlite3.Connection, export_id: str) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM macro_notebooklm_exports WHERE export_id = ?",
        (export_id,),
    ).fetchone()


def get_export_by_request_key(conn: sqlite3.Connection, request_key: str) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM macro_notebooklm_exports WHERE request_key = ?",
        (request_key,),
    ).fetchone()


def get_export_by_content_hash(conn: sqlite3.Connection, content_hash: str) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM macro_notebooklm_exports WHERE content_hash = ? ORDER BY created_at DESC LIMIT 1",
        (content_hash,),
    ).fetchone()


def get_latest_export(conn: sqlite3.Connection) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM macro_notebooklm_exports ORDER BY created_at DESC LIMIT 1"
    ).fetchone()


def update_export_state(
    conn: sqlite3.Connection,
    export_id: str,
    *,
    state: str,
    stage: str,
    job_id: Optional[str] = None,
    notebook_id: Optional[str] = None,
    notebook_url: Optional[str] = None,
    manifest_path: Optional[str] = None,
    inventory: Optional[Dict[str, Any]] = None,
    warnings: Optional[List[str]] = None,
    error_code: Optional[str] = None,
    error_message: Optional[str] = None,
) -> Optional[sqlite3.Row]:
    now = time.time()
    fields = ["state = ?", "stage = ?", "updated_at = ?"]
    params: List[Any] = [state, stage, now]

    if job_id is not None:
        fields.append("job_id = ?")
        params.append(job_id)
    if notebook_id is not None:
        fields.append("notebook_id = ?")
        params.append(notebook_id)
    if notebook_url is not None:
        fields.append("notebook_url = ?")
        params.append(notebook_url)
    if manifest_path is not None:
        fields.append("manifest_path = ?")
        params.append(manifest_path)
    if inventory is not None:
        fields.append("inventory_json = ?")
        params.append(json.dumps(inventory, ensure_ascii=False))
    if warnings is not None:
        fields.append("warnings_json = ?")
        params.append(json.dumps(warnings, ensure_ascii=False))
    if error_code is not None:
        fields.append("error_code = ?")
        params.append(error_code)
    if error_message is not None:
        fields.append("error_message = ?")
        params.append(error_message)

    params.append(export_id)
    query = f"UPDATE macro_notebooklm_exports SET {', '.join(fields)} WHERE export_id = ?"
    conn.execute(query, tuple(params))
    return get_export_by_id(conn, export_id)
