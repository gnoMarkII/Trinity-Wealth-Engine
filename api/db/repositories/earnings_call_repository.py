"""Earnings Call Runs SQLite DAO Repository."""
import sqlite3
import time
import uuid
from typing import Optional

from application.earnings_call.dto import ClaimDTO, EarningsCallRunDTO
from application.earnings_call.errors import EarningsCallLeaseExpiredError
from application.earnings_call.workflow import EarningsCallRunStatus, EarningsCallKanbanStatus


def _row_to_run_dto(row: sqlite3.Row) -> EarningsCallRunDTO:
    keys = row.keys() if hasattr(row, "keys") else []
    return EarningsCallRunDTO(
        run_id=row["run_id"],
        source_key=row["source_key"],
        ticker=row["ticker"],
        period=row["period"],
        transcript_hash=row["transcript_hash"],
        prompt_version=row["prompt_version"],
        status=EarningsCallRunStatus(row["status"]),
        kanban_status=EarningsCallKanbanStatus(row["kanban_status"]),
        highlights=row["highlights"],
        vault_path=row["vault_path"],
        kanban_card_id=row["kanban_card_id"],
        execution_token=row["execution_token"],
        execution_expires_at=row["execution_expires_at"],
        attempt_count=row["attempt_count"],
        last_error_code=row["last_error_code"],
        revision_ref=row["revision_ref"] if "revision_ref" in keys else None,
        content_sha256=row["content_sha256"] if "content_sha256" in keys else None,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def claim_or_resume(
    conn: sqlite3.Connection,
    source_key: str,
    ticker: str,
    period: str,
    transcript_hash: str,
    prompt_version: str,
    lease_seconds: int = 60,
) -> ClaimDTO:
    now = time.time()
    execution_token = uuid.uuid4().hex
    execution_expires_at = now + lease_seconds
    run_id = uuid.uuid4().hex

    # Atomic insert if not exists
    conn.execute(
        "INSERT INTO earnings_call_runs ("
        "  run_id, source_key, ticker, period, transcript_hash, prompt_version,"
        "  status, kanban_status, execution_token, execution_expires_at, attempt_count, created_at, updated_at"
        ") VALUES (?, ?, ?, ?, ?, ?, 'new', 'none', ?, ?, 1, ?, ?) "
        "ON CONFLICT(source_key) DO NOTHING",
        (run_id, source_key, ticker, period, transcript_hash, prompt_version, execution_token, execution_expires_at, now, now),
    )

    cur = conn.execute("SELECT * FROM earnings_call_runs WHERE source_key = ?", (source_key,))
    row = cur.fetchone()
    if row is None:
        raise RuntimeError("Failed to retrieve or insert earnings call run")

    run_dto = _row_to_run_dto(row)

    # If this thread/process inserted the new row
    if run_dto.run_id == run_id:
        return ClaimDTO(
            run=run_dto,
            owns_execution=True,
            execution_token=execution_token,
            execution_expires_at=execution_expires_at,
        )

    # Existing row: check if completed
    if run_dto.status == EarningsCallRunStatus.COMPLETED:
        return ClaimDTO(run=run_dto, owns_execution=False)

    # Existing row: check if lease is expired or unassigned
    is_expired = run_dto.execution_expires_at is None or run_dto.execution_expires_at <= now
    if is_expired:
        cur = conn.execute(
            "UPDATE earnings_call_runs SET "
            "  execution_token = ?, execution_expires_at = ?, attempt_count = attempt_count + 1, updated_at = ? "
            "WHERE source_key = ? AND (execution_expires_at IS NULL OR execution_expires_at <= ?)",
            (execution_token, execution_expires_at, now, source_key, now),
        )
        if cur.rowcount > 0:
            updated_row = conn.execute("SELECT * FROM earnings_call_runs WHERE source_key = ?", (source_key,)).fetchone()
            return ClaimDTO(
                run=_row_to_run_dto(updated_row),
                owns_execution=True,
                execution_token=execution_token,
                execution_expires_at=execution_expires_at,
            )

    return ClaimDTO(run=run_dto, owns_execution=False)


def renew_execution_lease(
    conn: sqlite3.Connection,
    run_id: str,
    execution_token: str,
    extension_seconds: int = 60,
) -> Optional[ClaimDTO]:
    now = time.time()
    new_expires_at = now + extension_seconds
    cur = conn.execute(
        "UPDATE earnings_call_runs SET execution_expires_at = ?, updated_at = ? "
        "WHERE run_id = ? AND execution_token = ? AND (execution_expires_at IS NULL OR execution_expires_at > ?)",
        (new_expires_at, now, run_id, execution_token, now),
    )
    if cur.rowcount == 0:
        return None
    row = conn.execute("SELECT * FROM earnings_call_runs WHERE run_id = ?", (run_id,)).fetchone()
    if not row:
        return None
    return ClaimDTO(
        run=_row_to_run_dto(row),
        owns_execution=True,
        execution_token=execution_token,
        execution_expires_at=new_expires_at,
    )


def save_summary(
    conn: sqlite3.Connection,
    run_id: str,
    execution_token: str,
    highlights: str,
) -> EarningsCallRunDTO:
    now = time.time()
    cur = conn.execute(
        "UPDATE earnings_call_runs SET status = 'summarized', highlights = ?, updated_at = ? "
        "WHERE run_id = ? AND execution_token = ? AND (execution_expires_at IS NULL OR execution_expires_at > ?)",
        (highlights, now, run_id, execution_token, now),
    )
    if cur.rowcount == 0:
        raise EarningsCallLeaseExpiredError(f"Execution lease for run '{run_id}' has expired or is invalid")
    row = conn.execute("SELECT * FROM earnings_call_runs WHERE run_id = ?", (run_id,)).fetchone()
    return _row_to_run_dto(row)


def mark_note_written(
    conn: sqlite3.Connection,
    run_id: str,
    execution_token: str,
    vault_path: str,
    revision_ref: Optional[str] = None,
    content_sha256: Optional[str] = None,
) -> EarningsCallRunDTO:
    now = time.time()
    cur = conn.execute(
        "UPDATE earnings_call_runs SET status = 'note_written', vault_path = ?, "
        "  revision_ref = COALESCE(?, revision_ref), content_sha256 = COALESCE(?, content_sha256), updated_at = ? "
        "WHERE run_id = ? AND execution_token = ? AND (execution_expires_at IS NULL OR execution_expires_at > ?)",
        (vault_path, revision_ref, content_sha256, now, run_id, execution_token, now),
    )
    if cur.rowcount == 0:
        raise EarningsCallLeaseExpiredError(f"Execution lease for run '{run_id}' has expired or is invalid")
    row = conn.execute("SELECT * FROM earnings_call_runs WHERE run_id = ?", (run_id,)).fetchone()
    return _row_to_run_dto(row)


def complete_kanban_delivery(
    conn: sqlite3.Connection,
    run_id: str,
    card_id: str,
    is_existing: bool = False,
) -> EarningsCallRunDTO:
    now = time.time()
    kanban_status = "existing" if is_existing else "created"
    conn.execute(
        "UPDATE earnings_call_runs SET status = 'completed', kanban_status = ?, "
        "  kanban_card_id = ?, last_error_code = NULL, execution_token = NULL, "
        "  execution_expires_at = NULL, updated_at = ? "
        "WHERE run_id = ?",
        (kanban_status, card_id, now, run_id),
    )
    row = conn.execute("SELECT * FROM earnings_call_runs WHERE run_id = ?", (run_id,)).fetchone()
    return _row_to_run_dto(row)


def schedule_kanban_retry(
    conn: sqlite3.Connection,
    run_id: str,
    error_code: str,
) -> EarningsCallRunDTO:
    now = time.time()
    conn.execute(
        "UPDATE earnings_call_runs SET status = 'kanban_pending', kanban_status = 'pending', "
        "  last_error_code = ?, execution_token = NULL, execution_expires_at = NULL, updated_at = ? "
        "WHERE run_id = ?",
        (error_code, now, run_id),
    )
    row = conn.execute("SELECT * FROM earnings_call_runs WHERE run_id = ?", (run_id,)).fetchone()
    return _row_to_run_dto(row)


def mark_terminal_failure(
    conn: sqlite3.Connection,
    run_id: str,
    error_code: str,
) -> EarningsCallRunDTO:
    now = time.time()
    conn.execute(
        "UPDATE earnings_call_runs SET status = 'failed', kanban_status = 'failed', "
        "  last_error_code = ?, execution_token = NULL, execution_expires_at = NULL, updated_at = ? "
        "WHERE run_id = ?",
        (error_code, now, run_id),
    )
    row = conn.execute("SELECT * FROM earnings_call_runs WHERE run_id = ?", (run_id,)).fetchone()
    return _row_to_run_dto(row)


def get_run(conn: sqlite3.Connection, run_id: str) -> Optional[EarningsCallRunDTO]:
    cur = conn.execute("SELECT * FROM earnings_call_runs WHERE run_id = ?", (run_id,))
    row = cur.fetchone()
    return _row_to_run_dto(row) if row else None


def get_run_by_source_key(conn: sqlite3.Connection, source_key: str) -> Optional[EarningsCallRunDTO]:
    cur = conn.execute("SELECT * FROM earnings_call_runs WHERE source_key = ?", (source_key,))
    row = cur.fetchone()
    return _row_to_run_dto(row) if row else None
