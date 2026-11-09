import api.db.repositories.advisor_chat_repository as advisor_chat_repo_dao
from application.manager.ports import AdvisorSessionRepoPort, PMReviewSignoffRepoPort
from application.thesis.ports import ThesisExceptionStorePort
from core.sandbox.models import AdvisorChatMessage, AdvisorSessionSummary
import tools.thesis.thesis_sqlite_dao as thesis_sqlite_dao
import uuid
"""SQLite driven adapters for application repository ports.

The adapters deliberately support two lifecycles:

* standalone calls (the historical API) open and commit their own connection;
* a connection-bound instance is created by :class:`DbUnitOfWork` and never
  commits or rolls back, so several repositories can participate in one
  transaction.

Raw SQL remains in ``api.db.repositories``.  This module is the outbound
adapter boundary and contains only connection/row mapping concerns.
"""

from contextlib import closing, contextmanager
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterator, List, Optional, Dict, Sequence

from application.jobs.ports import JobRepositoryPort
from application.kanban.ports import KanbanRepositoryPort
from application.notebooklm.ports import (
    NotebookLMJobRepositoryPort,
    NotebookLMCardRepositoryPort,
    NotebookLMNotificationOutboxPort,
)
from application.equity.ports import (
    AnalystCachePort,
    InsiderHistoryProviderPort,
    InsiderLedgerPort,
    ValuationLedgerPort,
    InsiderSyncPort,
)
from api.db.connection import get_connection
from api.config import get_state_db_path
from tools._atomic_io import _atomic_write_to
import api.db.repositories.institutional_ledger_repository as institutional_ledger_dao
import api.db.repositories.job_repository as job_repo_dao
import api.db.repositories.kanban_repository as kanban_repo_dao
import api.db.repositories.cache_repository as cache_repo_dao
import api.db.repositories.dcf_repository as dcf_repo_dao
import api.db.repositories.insider_repository as insider_repo_dao
import api.db.repositories.outbox_repository as outbox_repo_dao
import api.db.repositories.earnings_call_repository as earnings_call_repo_dao
import api.db.repositories.earnings_call_outbox_repository as earnings_call_outbox_repo_dao
from application.earnings_call.ports import EarningsCallWorkflowPort
from application.earnings_call.errors import (
    EarningsCallLeaseExpiredError,
    EarningsCallRunNotReadyError,
)
from application.earnings_call.dto import (
    ClaimDTO,
    EarningsCallRunDTO,
    EarningsCallOutboxEventDTO,
    LeaseDTO,
)


class _SqliteAdapterBase:
    """Shared connection lifecycle for SQLite driven adapters."""

    def __init__(
        self,
        db_path: Optional[str] = None,
        *,
        conn: Optional[sqlite3.Connection] = None,
    ) -> None:
        if conn is not None and db_path is not None:
            raise ValueError("Provide either conn or db_path, not both")
        self._db_path = db_path
        self._conn = conn

    @property
    def is_connection_bound(self) -> bool:
        return self._conn is not None

    @contextmanager
    def _connection(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        """Yield a connection and commit only when this adapter owns it."""
        if self._conn is not None:
            yield self._conn
            return

        with closing(get_connection(self._db_path)) as conn:
            try:
                if write and not conn.in_transaction:
                    conn.execute("BEGIN IMMEDIATE")
                yield conn
                if write:
                    conn.commit()
            except Exception:
                if write:
                    conn.rollback()
                raise


class SqliteJobRepositoryAdapter(_SqliteAdapterBase, JobRepositoryPort):
    """Concrete SQLite adapter for :class:`JobRepositoryPort`."""

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._connection() as conn:
            row = job_repo_dao.get_job(conn, job_id)
            return dict(row) if row else None

    def find_job_by_idempotency_key(self, idempotency_key: str) -> Optional[Dict[str, Any]]:
        with self._connection() as conn:
            row = job_repo_dao.find_job_by_idempotency_key(conn, idempotency_key)
            return dict(row) if row else None

    def create_job(
        self,
        job_id: str,
        thread_id: str,
        card_id: Optional[str],
        idempotency_key: str,
        instruction: str,
        status: str = "queued",
        flow: str = "manager",
        scope: str = "both",
    ) -> None:
        with self._connection(write=True) as conn:
            job_repo_dao.create_job(
                conn,
                job_id,
                thread_id,
                card_id,
                idempotency_key,
                instruction,
                status=status,
                flow=flow,
                scope=scope,
            )

    def update_job_status(self, job_id: str, status: str, error_message: Optional[str] = None) -> None:
        with self._connection(write=True) as conn:
            job_repo_dao.update_job_status(conn, job_id, status, error_message)

    def cas_job_status(self, job_id: str, old_status: str, new_status: str) -> bool:
        with self._connection(write=True) as conn:
            return bool(job_repo_dao.cas_job_status(conn, job_id, old_status, new_status))

    def set_job_awaiting_approval(self, job_id: str, interrupt_payload_json: str) -> None:
        with self._connection(write=True) as conn:
            job_repo_dao.set_job_awaiting_approval(conn, job_id, interrupt_payload_json)

    def clear_job_resume_value(self, job_id: str) -> None:
        with self._connection(write=True) as conn:
            job_repo_dao.clear_job_resume_value(conn, job_id)

    def append_job_log(
        self,
        job_id: str,
        node_name: str,
        content: str,
        role: str = "reply",
        label: Optional[str] = None,
    ) -> None:
        with self._connection(write=True) as conn:
            job_repo_dao.append_job_log(conn, job_id, node_name, content, role, label)

    def get_job_reply_logs(self, job_id: str) -> List[Dict[str, Any]]:
        with self._connection() as conn:
            rows = job_repo_dao.get_job_reply_logs(conn, job_id)
            return [dict(r) for r in rows]

    def get_job_logs_since(self, job_id: str, after_seq: int) -> List[Dict[str, Any]]:
        with self._connection() as conn:
            rows = job_repo_dao.get_job_logs_since(conn, job_id, after_seq)
            return [dict(r) for r in rows]

    def claim_job_resume(
        self,
        job_id: str,
        resume_value_json: str,
        token_uses: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        with self._connection(write=True) as conn:
            job_repo_dao.claim_job_resume(
                conn=conn,
                job_id=job_id,
                resume_value_json=resume_value_json,
                token_uses=token_uses,
            )

    def get_latest_job_log_node(self, job_id: str) -> Optional[str]:
        with self._connection() as conn:
            return job_repo_dao.get_latest_job_log_node(conn, job_id)

    def get_job_log_count(self, job_id: str) -> int:
        with self._connection() as conn:
            return job_repo_dao.get_job_log_count(conn, job_id)

    def list_jobs_by_status(
        self, statuses: List[str], flows: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        with self._connection() as conn:
            rows = job_repo_dao.list_jobs_by_status(conn, statuses, flows=flows)
            return [dict(row) for row in rows]


class SqliteKanbanRepositoryAdapter(_SqliteAdapterBase, KanbanRepositoryPort):
    """Concrete SQLite adapter for :class:`KanbanRepositoryPort`."""

    def list_kanban_cards(self) -> List[Dict[str, Any]]:
        with self._connection() as conn:
            rows = kanban_repo_dao.list_kanban_cards(conn)
            return [dict(r) for r in rows]

    def get_kanban_card(self, card_id: str) -> Optional[Dict[str, Any]]:
        with self._connection() as conn:
            row = kanban_repo_dao.get_kanban_card(conn, card_id)
            return dict(row) if row else None

    def find_kanban_card_by_title_in_column(
        self, title: str, column_name: str, prompt: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        with self._connection() as conn:
            row = kanban_repo_dao.find_kanban_card_by_title_in_column(
                conn, title, column_name, prompt=prompt
            )
            return dict(row) if row else None

    def find_kanban_card_by_source_key(
        self, source_key: str
    ) -> Optional[Dict[str, Any]]:
        with self._connection() as conn:
            row = kanban_repo_dao.find_kanban_card_by_source_key(conn, source_key)
            return dict(row) if row else None

    def create_kanban_card(
        self,
        card_id: str,
        title: str,
        column_name: str,
        flow: str = "manager",
        prompt: Optional[str] = None,
        source_key: Optional[str] = None,
        scope: str = "both",
        discord_notify: bool = True,
        is_verified: bool = True,
    ) -> None:
        with self._connection(write=True) as conn:
            kanban_repo_dao.create_kanban_card(
                conn=conn,
                card_id=card_id,
                title=title,
                column_name=column_name,
                flow=flow,
                prompt=prompt,
                source_key=source_key,
                scope=scope,
                is_verified=is_verified,
            )

    def update_kanban_card(
        self,
        card_id: str,
        title: str,
        flow: str,
        prompt: Optional[str] = None,
        scope: str = "both",
        discord_notify: Optional[bool] = None,
    ) -> None:
        with self._connection(write=True) as conn:
            kanban_repo_dao.update_kanban_card(
                conn=conn,
                card_id=card_id,
                title=title,
                flow=flow,
                prompt=prompt,
                scope=scope,
            )

    def delete_kanban_card(self, card_id: str) -> bool:
        with self._connection(write=True) as conn:
            existing = kanban_repo_dao.get_kanban_card(conn, card_id)
            if existing is None:
                return False
            kanban_repo_dao.delete_kanban_card(conn, card_id)
            return True

    def move_kanban_card(
        self,
        card_id: str,
        target_column: str,
        job_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        with self._connection(write=True) as conn:
            kanban_repo_dao.move_kanban_card(conn, card_id, target_column, job_id=job_id)
            row = kanban_repo_dao.get_kanban_card(conn, card_id)
            return dict(row) if row else None

    def update_kanban_display_seq(self, card_id: str, new_seq: int) -> None:
        with self._connection(write=True) as conn:
            kanban_repo_dao.update_kanban_display_seq(conn, card_id, new_seq)

    def toggle_discord(
        self, card_id: str, enabled: Optional[bool] = None
    ) -> Optional[Dict[str, Any]]:
        with self._connection(write=True) as conn:
            kanban_repo_dao.toggle_kanban_card_discord(
                conn, card_id, enabled=enabled if enabled is not None else True
            )
            row = kanban_repo_dao.get_kanban_card(conn, card_id)
            return dict(row) if row else None


class SqliteNotebookLMJobRepositoryAdapter(_SqliteAdapterBase, NotebookLMJobRepositoryPort):
    """Concrete SQLite adapter for NotebookLM job status queries."""

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._connection() as conn:
            row = job_repo_dao.get_job(conn, job_id)
            return dict(row) if row else None


class SqliteNotebookLMCardRepositoryAdapter(_SqliteAdapterBase, NotebookLMCardRepositoryPort):
    """Connection adapter for the small Kanban surface used by NotebookLM."""

    def get_card(self, card_id: str) -> Optional[Dict[str, Any]]:
        with self._connection() as conn:
            row = kanban_repo_dao.get_kanban_card(conn, card_id)
            return dict(row) if row else None

    def set_source(self, card_id: str, prompt: str, is_verified: bool) -> None:
        with self._connection(write=True) as conn:
            kanban_repo_dao.set_kanban_card_source(conn, card_id, prompt, is_verified)

    def move_card(self, card_id: str, column_name: str, job_id: Optional[str] = None) -> None:
        with self._connection(write=True) as conn:
            kanban_repo_dao.move_kanban_card(conn, card_id, column_name, job_id=job_id)


class SqliteNotificationOutboxAdapter(_SqliteAdapterBase, NotebookLMNotificationOutboxPort):
    """Durable notification outbox adapter used by post-production workers."""

    def enqueue(
        self,
        *,
        event_id: str,
        idempotency_key: str,
        aggregate_type: str,
        aggregate_id: str,
        event_type: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        with self._connection(write=True) as conn:
            return dict(
                outbox_repo_dao.enqueue_event(
                    conn,
                    event_id=event_id,
                    idempotency_key=idempotency_key,
                    aggregate_type=aggregate_type,
                    aggregate_id=aggregate_id,
                    event_type=event_type,
                    payload=payload,
                )
            )

    def mark_sent(self, idempotency_key: str) -> None:
        with self._connection(write=True) as conn:
            outbox_repo_dao.mark_sent(conn, idempotency_key)

    def mark_failed(self, idempotency_key: str, error: str) -> None:
        with self._connection(write=True) as conn:
            outbox_repo_dao.mark_failed(conn, idempotency_key, error)

    def get(self, idempotency_key: str) -> Optional[Dict[str, Any]]:
        with self._connection() as conn:
            row = outbox_repo_dao.get_event(conn, idempotency_key)
            return dict(row) if row else None

    def list_pending(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._connection() as conn:
            return [dict(row) for row in outbox_repo_dao.list_pending(conn, limit=limit)]


class SqliteAnalystCacheAdapter(_SqliteAdapterBase, AnalystCachePort):
    """SQLite adapter for analyst context cache reads and writes."""

    def get(self, ticker: str) -> Optional[Dict[str, Any]]:
        with self._connection() as conn:
            return cache_repo_dao.get_analyst_context_cache(conn, ticker)

    def upsert(self, ticker: str, data: Dict[str, Any]) -> None:
        payload = dict(data)
        synced_at = payload.get("synced_at")
        if isinstance(synced_at, str):
            from datetime import datetime

            payload["synced_at"] = datetime.fromisoformat(synced_at.replace("Z", "+00:00")).timestamp()
        with self._connection(write=True) as conn:
            cache_repo_dao.upsert_analyst_context_cache(conn, ticker, payload)


class SqliteValuationLedgerAdapter(_SqliteAdapterBase, ValuationLedgerPort):
    """SQLite adapter for the immutable DCF evaluation ledger."""

    def latest(self, ticker: str) -> Optional[Dict[str, Any]]:
        with self._connection() as conn:
            row = dcf_repo_dao.get_latest_dcf_evaluation(conn, ticker)
            return dict(row) if row else None

    def record(self, **kwargs: Any) -> None:
        with self._connection(write=True) as conn:
            dcf_repo_dao.record_dcf_evaluation(conn, **kwargs)


class SqliteInsiderLedgerAdapter(_SqliteAdapterBase, InsiderLedgerPort):
    """SQLite adapter for canonical SEC Form 4 records."""

    def list_records(self, ticker: str, since_date: str) -> List[Dict[str, Any]]:
        with self._connection() as conn:
            return insider_repo_dao.get_sec_insider_filings_and_transactions(
                conn, ticker, since_date=since_date
            )


class SqliteInsiderSyncAdapter(InsiderSyncPort):
    """Persist normalized insider history atomically through the SQLite UoW."""

    def __init__(
        self,
        db_path: Optional[str] = None,
        *,
        provider: InsiderHistoryProviderPort,
    ) -> None:
        self._db_path = db_path
        self._provider = provider

    def sync(self, ticker: str) -> None:
        from api.db.uow import DbUnitOfWork
        from tools.market.sec_form4_pipeline import derive_canonical_transactions

        records = self._provider.fetch(ticker)
        has_amendment = any(isinstance(parsed, dict) and parsed.get("is_amendment") for parsed in records)
        canonical_accessions = None
        if has_amendment:
            canonical_lots, q_meta = derive_canonical_transactions(records)
            if not q_meta.get("requires_review"):
                canonical_accessions = {
                    lot.get("accession_number")
                    for lot in canonical_lots
                    if lot.get("accession_number")
                }

        with DbUnitOfWork(db_path=self._db_path) as uow:
            for parsed in records:
                payload = parsed
                if has_amendment and isinstance(parsed, dict) and not parsed.get("is_amendment"):
                    acc = parsed.get("accession_number")
                    if canonical_accessions is None or (acc and acc not in canonical_accessions):
                        payload = {**parsed, "transactions": []}
                insider_repo_dao.record_parsed_filing(uow.conn, payload)


class SqliteEarningsCallWorkflowAdapter(_SqliteAdapterBase, EarningsCallWorkflowPort):
    """SQLite driven adapter for managing Earnings Call Saga state and Transactional Outbox."""

    def claim_or_resume(
        self,
        source_key: str,
        ticker: str,
        period: str,
        transcript_hash: str,
        prompt_version: str,
        lease_seconds: int = 60,
    ) -> ClaimDTO:
        with self._connection(write=True) as conn:
            return earnings_call_repo_dao.claim_or_resume(
                conn=conn,
                source_key=source_key,
                ticker=ticker,
                period=period,
                transcript_hash=transcript_hash,
                prompt_version=prompt_version,
                lease_seconds=lease_seconds,
            )

    def renew_execution_lease(
        self, run_id: str, execution_token: str, extension_seconds: int = 60
    ) -> Optional[ClaimDTO]:
        with self._connection(write=True) as conn:
            return earnings_call_repo_dao.renew_execution_lease(
                conn=conn,
                run_id=run_id,
                execution_token=execution_token,
                extension_seconds=extension_seconds,
            )

    def save_summary(
        self, run_id: str, execution_token: str, highlights: str
    ) -> EarningsCallRunDTO:
        with self._connection(write=True) as conn:
            return earnings_call_repo_dao.save_summary(
                conn=conn,
                run_id=run_id,
                execution_token=execution_token,
                highlights=highlights,
            )

    def record_note_and_enqueue(
        self,
        run_id: str,
        execution_token: str,
        vault_path: str,
        revision_ref: Optional[str] = None,
        content_sha256: Optional[str] = None,
        outbox_lease_seconds: int = 60,
    ) -> tuple[EarningsCallRunDTO, EarningsCallOutboxEventDTO, LeaseDTO]:
        with self._connection(write=True) as conn:
            run_dto = earnings_call_repo_dao.mark_note_written(
                conn=conn,
                run_id=run_id,
                execution_token=execution_token,
                vault_path=vault_path,
                revision_ref=revision_ref,
                content_sha256=content_sha256,
            )
            event_dto, lease_dto = earnings_call_outbox_repo_dao.enqueue_event(
                conn=conn,
                run_id=run_id,
                source_key=run_dto.source_key,
                event_type="deliver_kanban",
                outbox_lease_seconds=outbox_lease_seconds,
            )
            return run_dto, event_dto, lease_dto

    def complete_kanban_delivery(
        self,
        run_id: str,
        event_id: str,
        lease_token: str,
        card_id: str,
        is_existing: bool,
    ) -> EarningsCallRunDTO:
        with self._connection(write=True) as conn:
            completed = earnings_call_outbox_repo_dao.complete_event(
                conn=conn,
                event_id=event_id,
                lease_token=lease_token,
            )
            if not completed:
                raise EarningsCallLeaseExpiredError(
                    f"Outbox lease for event '{event_id}' has expired or is invalid"
                )
            return earnings_call_repo_dao.complete_kanban_delivery(
                conn=conn,
                run_id=run_id,
                card_id=card_id,
                is_existing=is_existing,
            )

    def schedule_kanban_retry(
        self,
        run_id: str,
        event_id: str,
        lease_token: str,
        error_code: str,
        retry_delay_seconds: int,
    ) -> EarningsCallRunDTO:
        with self._connection(write=True) as conn:
            scheduled = earnings_call_outbox_repo_dao.schedule_retry(
                conn=conn,
                event_id=event_id,
                lease_token=lease_token,
                error_code=error_code,
                retry_delay_seconds=retry_delay_seconds,
            )
            if not scheduled:
                raise EarningsCallLeaseExpiredError(
                    f"Outbox lease for event '{event_id}' has expired or is invalid"
                )
            return earnings_call_repo_dao.schedule_kanban_retry(
                conn=conn,
                run_id=run_id,
                error_code=error_code,
            )

    def mark_terminal_failure(
        self,
        run_id: str,
        event_id: str,
        lease_token: str,
        error_code: str,
    ) -> EarningsCallRunDTO:
        with self._connection(write=True) as conn:
            dead_lettered = earnings_call_outbox_repo_dao.mark_dead_letter(
                conn=conn,
                event_id=event_id,
                lease_token=lease_token,
                error_code=error_code,
            )
            if not dead_lettered:
                raise EarningsCallLeaseExpiredError(
                    f"Outbox lease for event '{event_id}' has expired or is invalid"
                )
            return earnings_call_repo_dao.mark_terminal_failure(
                conn=conn,
                run_id=run_id,
                error_code=error_code,
            )

    def get_run(self, run_id: str) -> Optional[EarningsCallRunDTO]:
        with self._connection() as conn:
            return earnings_call_repo_dao.get_run(conn=conn, run_id=run_id)

    def list_pending_outbox(self, limit: int = 10) -> list[EarningsCallOutboxEventDTO]:
        with self._connection() as conn:
            return earnings_call_outbox_repo_dao.list_pending(conn=conn, limit=limit)

    def lease_outbox_event(
        self, event_id: str, lease_seconds: int = 60
    ) -> Optional[LeaseDTO]:
        with self._connection(write=True) as conn:
            return earnings_call_outbox_repo_dao.lease_event(
                conn=conn,
                event_id=event_id,
                lease_seconds=lease_seconds,
            )

    def reset_run_for_manual_retry(
        self, run_id: str, lease_seconds: int = 60
    ) -> tuple[EarningsCallRunDTO, EarningsCallOutboxEventDTO, LeaseDTO]:
        with self._connection(write=True) as conn:
            run_dto = earnings_call_repo_dao.get_run(conn=conn, run_id=run_id)
            if not run_dto:
                raise EarningsCallRunNotReadyError(f"Run '{run_id}' not found")
            try:
                event_dto, lease_dto = earnings_call_outbox_repo_dao.reset_for_manual_retry(
                    conn=conn,
                    run_id=run_id,
                    source_key=run_dto.source_key,
                    lease_seconds=lease_seconds,
                )
            except RuntimeError as exc:
                # A live lease belongs to another worker (or the event is
                # already completed).  Surface a typed application error so
                # the inbound adapter can return a deterministic 409 rather
                # than leaking an infrastructure failure as HTTP 500.
                raise EarningsCallRunNotReadyError(str(exc)) from exc
            return run_dto, event_dto, lease_dto


from application.macro.notebooklm_export_ports import (
    MacroExportRecord,
    MacroExportRepositoryPort,
)
import api.db.repositories.macro_export_repository as macro_export_repo_dao


def _row_to_macro_export_record(row: sqlite3.Row) -> MacroExportRecord:
    return MacroExportRecord(
        export_id=row["export_id"],
        request_key=row["request_key"],
        content_hash=row["content_hash"],
        job_id=row["job_id"],
        state=row["state"],
        stage=row["stage"],
        snapshot_at=row["snapshot_at"],
        strategy_report_id=row["strategy_report_id"],
        notebook_id=row["notebook_id"],
        notebook_url=row["notebook_url"],
        manifest_path=row["manifest_path"],
        inventory=json.loads(row["inventory_json"]) if row["inventory_json"] else {},
        warnings=json.loads(row["warnings_json"]) if row["warnings_json"] else [],
        error_code=row["error_code"],
        error_message=row["error_message"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


class SqliteMacroExportRepositoryAdapter(MacroExportRepositoryPort):
    """Adapter implementing MacroExportRepositoryPort backed by SQLite."""

    def __init__(self, conn: Optional[sqlite3.Connection] = None, db_path: Optional[str] = None) -> None:
        self._conn = conn
        self._db_path = db_path

    @contextmanager
    def _connection(self, write: bool = False):
        if self._conn is not None:
            yield self._conn
        else:
            with get_connection(self._db_path) as conn:
                if write:
                    with conn:
                        yield conn
                else:
                    yield conn

    def create(self, record: MacroExportRecord) -> MacroExportRecord:
        with self._connection(write=True) as conn:
            row = macro_export_repo_dao.insert_export(
                conn=conn,
                export_id=record.export_id,
                request_key=record.request_key,
                content_hash=record.content_hash,
                snapshot_at=record.snapshot_at,
                strategy_report_id=record.strategy_report_id,
                job_id=record.job_id,
                state=record.state,
                stage=record.stage,
                manifest_path=record.manifest_path,
                inventory=record.inventory,
                warnings=record.warnings,
            )
            return _row_to_macro_export_record(row)

    def get_by_id(self, export_id: str) -> Optional[MacroExportRecord]:
        with self._connection() as conn:
            row = macro_export_repo_dao.get_export_by_id(conn=conn, export_id=export_id)
            return _row_to_macro_export_record(row) if row else None

    def get_by_request_key(self, request_key: str) -> Optional[MacroExportRecord]:
        with self._connection() as conn:
            row = macro_export_repo_dao.get_export_by_request_key(conn=conn, request_key=request_key)
            return _row_to_macro_export_record(row) if row else None

    def get_by_content_hash(self, content_hash: str) -> Optional[MacroExportRecord]:
        with self._connection() as conn:
            row = macro_export_repo_dao.get_export_by_content_hash(conn=conn, content_hash=content_hash)
            return _row_to_macro_export_record(row) if row else None

    def get_latest(self) -> Optional[MacroExportRecord]:
        with self._connection() as conn:
            row = macro_export_repo_dao.get_latest_export(conn=conn)
            return _row_to_macro_export_record(row) if row else None

    def update_state(
        self,
        export_id: str,
        state: str,
        stage: str,
        *,
        job_id: Optional[str] = None,
        notebook_id: Optional[str] = None,
        notebook_url: Optional[str] = None,
        manifest_path: Optional[str] = None,
        inventory: Optional[Dict[str, Any]] = None,
        warnings: Optional[List[str]] = None,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> Optional[MacroExportRecord]:
        with self._connection(write=True) as conn:
            row = macro_export_repo_dao.update_export_state(
                conn=conn,
                export_id=export_id,
                state=state,
                stage=stage,
                job_id=job_id,
                notebook_id=notebook_id,
                notebook_url=notebook_url,
                manifest_path=manifest_path,
                inventory=inventory,
                warnings=warnings,
                error_code=error_code,
                error_message=error_message,
            )
            return _row_to_macro_export_record(row) if row else None


from application.equity.notebooklm_export_ports import (
    EquityExportRecord,
    EquityExportRepositoryPort,
)
import api.db.repositories.equity_export_repository as equity_export_repo_dao


def _row_to_equity_export_record(row: sqlite3.Row) -> EquityExportRecord:
    return EquityExportRecord(
        export_id=row["export_id"],
        request_key=row["request_key"],
        ticker=row["ticker"],
        market=row["market"],
        content_hash=row["content_hash"],
        job_id=row["job_id"],
        state=row["state"],
        stage=row["stage"],
        snapshot_at=row["snapshot_at"],
        notebook_id=row["notebook_id"],
        notebook_url=row["notebook_url"],
        manifest_path=row["manifest_path"],
        inventory=json.loads(row["inventory_json"]) if row["inventory_json"] else {},
        warnings=json.loads(row["warnings_json"]) if row["warnings_json"] else [],
        error_code=row["error_code"],
        error_message=row["error_message"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


class SqliteEquityExportRepositoryAdapter(EquityExportRepositoryPort):
    """Adapter implementing EquityExportRepositoryPort backed by SQLite."""

    def __init__(self, conn: Optional[sqlite3.Connection] = None, db_path: Optional[str] = None) -> None:
        self._conn = conn
        self._db_path = db_path

    @contextmanager
    def _connection(self, write: bool = False):
        if self._conn is not None:
            yield self._conn
        else:
            with get_connection(self._db_path) as conn:
                if write:
                    with conn:
                        yield conn
                else:
                    yield conn

    def create(self, record: EquityExportRecord) -> EquityExportRecord:
        with self._connection(write=True) as conn:
            row = equity_export_repo_dao.insert_export(
                conn=conn,
                export_id=record.export_id,
                request_key=record.request_key,
                ticker=record.ticker,
                market=record.market,
                content_hash=record.content_hash,
                snapshot_at=record.snapshot_at,
                job_id=record.job_id,
                state=record.state,
                stage=record.stage,
                manifest_path=record.manifest_path,
                inventory=record.inventory,
                warnings=record.warnings,
            )
            return _row_to_equity_export_record(row)

    def get_by_id(self, export_id: str) -> Optional[EquityExportRecord]:
        with self._connection() as conn:
            row = equity_export_repo_dao.get_export_by_id(conn=conn, export_id=export_id)
            return _row_to_equity_export_record(row) if row else None

    def get_by_request_key(self, request_key: str) -> Optional[EquityExportRecord]:
        with self._connection() as conn:
            row = equity_export_repo_dao.get_export_by_request_key(conn=conn, request_key=request_key)
            return _row_to_equity_export_record(row) if row else None

    def get_by_content_hash(
        self, ticker: str, market: str, content_hash: str
    ) -> Optional[EquityExportRecord]:
        with self._connection() as conn:
            row = equity_export_repo_dao.get_export_by_content_hash(
                conn=conn, ticker=ticker, market=market, content_hash=content_hash
            )
            return _row_to_equity_export_record(row) if row else None

    def get_latest(self, ticker: str, market: str = "US") -> Optional[EquityExportRecord]:
        with self._connection() as conn:
            row = equity_export_repo_dao.get_latest_export(conn=conn, ticker=ticker, market=market)
            return _row_to_equity_export_record(row) if row else None

    def update_state(
        self,
        export_id: str,
        state: str,
        stage: str,
        *,
        job_id: Optional[str] = None,
        notebook_id: Optional[str] = None,
        notebook_url: Optional[str] = None,
        manifest_path: Optional[str] = None,
        inventory: Optional[Dict[str, Any]] = None,
        warnings: Optional[List[str]] = None,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> Optional[EquityExportRecord]:
        with self._connection(write=True) as conn:
            row = equity_export_repo_dao.update_export_state(
                conn=conn,
                export_id=export_id,
                state=state,
                stage=stage,
                job_id=job_id,
                notebook_id=notebook_id,
                notebook_url=notebook_url,
                manifest_path=manifest_path,
                inventory=inventory,
                warnings=warnings,
                error_code=error_code,
                error_message=error_message,
            )
            return _row_to_equity_export_record(row) if row else None


class SqliteAdvisorSessionRepositoryAdapter(_SqliteAdapterBase, AdvisorSessionRepoPort):
    """Concrete SQLite adapter for AdvisorSessionRepoPort."""

    def list_sessions(self, limit: int = 30) -> list[AdvisorSessionSummary]:
        with self._connection() as conn:
            return advisor_chat_repo_dao.list_sessions(conn, limit=limit)

    def create_session(
        self, title: str = "บทสนทนาใหม่", session_id: Optional[str] = None
    ) -> AdvisorSessionSummary:
        actual_id = session_id or f"session_{uuid.uuid4().hex[:12]}"
        with self._connection(write=True) as conn:
            return advisor_chat_repo_dao.create_session(conn, actual_id, title)

    def update_session_title(self, session_id: str, title: str) -> None:
        with self._connection(write=True) as conn:
            advisor_chat_repo_dao.update_session_title(conn, session_id, title)

    def get_session(self, session_id: str) -> Optional[AdvisorSessionSummary]:
        with self._connection() as conn:
            return advisor_chat_repo_dao.get_session(conn, session_id)

    def get_messages(self, session_id: str) -> list[AdvisorChatMessage]:
        with self._connection() as conn:
            return advisor_chat_repo_dao.get_messages(conn, session_id)

    def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        citations: Sequence[str] = (),
    ) -> AdvisorChatMessage:
        msg_id = f"msg_{uuid.uuid4().hex[:12]}"
        citations_json = json.dumps(list(citations), ensure_ascii=False)
        with self._connection(write=True) as conn:
            return advisor_chat_repo_dao.append_message(
                conn, msg_id, session_id, role, content, citations_json
            )


class SqlitePMReviewSignoffRepositoryAdapter(_SqliteAdapterBase, PMReviewSignoffRepoPort):
    """Concrete SQLite adapter for PMReviewSignoffRepoPort."""

    def get_signoff(
        self,
        review_date: str,
        portfolio_id: str = "default",
        packet_sha256: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        with self._connection() as conn:
            return thesis_sqlite_dao.get_signoff(
                conn,
                review_date=review_date,
                portfolio_id=portfolio_id,
                packet_sha256=packet_sha256,
            )

    def save_signoff(
        self,
        signoff_id: str,
        review_date: str,
        signed_off_by: str,
        signed_off_at_utc: str,
        acknowledged_exceptions_count: int,
        notes: Optional[str],
        packet_sha256: str,
        portfolio_id: str = "default",
        verified_sample_ids: Optional[Sequence[str]] = None,
        acknowledged_exception_ids: Optional[Sequence[str]] = None,
        packet_id: Optional[str] = None,
        request_digest: Optional[str] = None,
        principal_subject_id: Optional[str] = None,
        principal_session_id: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        with self._connection(write=True) as conn:
            return thesis_sqlite_dao.insert_signoff(
                conn,
                signoff_id=signoff_id,
                review_date=review_date,
                signed_off_by=signed_off_by,
                signed_off_at_utc=signed_off_at_utc,
                acknowledged_exceptions_count=acknowledged_exceptions_count,
                notes=notes,
                packet_sha256=packet_sha256,
                portfolio_id=portfolio_id,
                verified_sample_ids=verified_sample_ids,
                acknowledged_exception_ids=acknowledged_exception_ids,
                packet_id=packet_id,
                request_digest=request_digest,
                principal_subject_id=principal_subject_id,
                principal_session_id=principal_session_id,
            )

    def save_packet_snapshot(
        self,
        *,
        packet_id: str,
        portfolio_id: str,
        review_date: str,
        cutoff_utc: str,
        input_fingerprint_sha256: str,
        policy_version: str,
        packet_sha256: str,
        payload: Dict[str, Any],
    ) -> dict[str, Any]:
        with self._connection(write=True) as conn:
            return thesis_sqlite_dao.save_packet_snapshot(
                conn,
                packet_id=packet_id,
                portfolio_id=portfolio_id,
                review_date=review_date,
                cutoff_utc=cutoff_utc,
                input_fingerprint_sha256=input_fingerprint_sha256,
                policy_version=policy_version,
                packet_sha256=packet_sha256,
                payload=payload,
            )

    def get_packet_snapshot(self, portfolio_id: str, packet_sha256: str) -> Optional[dict[str, Any]]:
        with self._connection() as conn:
            return thesis_sqlite_dao.get_packet_snapshot(conn, portfolio_id, packet_sha256)

    def list_signoffs(self, limit: int = 30, portfolio_id: Optional[str] = None) -> list[dict[str, Any]]:
        with self._connection() as conn:
            return thesis_sqlite_dao.list_signoffs(conn, limit=limit, portfolio_id=portfolio_id)


class SqliteThesisExceptionRepositoryAdapter(_SqliteAdapterBase, ThesisExceptionStorePort):
    """Concrete SQLite adapter for ThesisExceptionStorePort."""

    def insert_exception(self, exc: Any) -> None:
        with self._connection(write=True) as conn:
            thesis_sqlite_dao.insert_exception(conn, exc)

    def list_pending_exceptions(self, ticker: Optional[str] = None, limit: int = 50) -> list[Any]:
        with self._connection() as conn:
            return thesis_sqlite_dao.list_pending_exceptions(conn, ticker=ticker, limit=limit)

    def acknowledge_exception(self, exception_id: str) -> None:
        with self._connection(write=True) as conn:
            thesis_sqlite_dao.acknowledge_exception(conn, exception_id)



class SqliteInstitutionalLedgerRepositoryAdapter(_SqliteAdapterBase):
    """Production SQLite adapter for durable routine, catalyst, source and estimate state."""

    def __init__(self, db_path: Optional[str] = None, *, conn: Optional[sqlite3.Connection] = None, artifact_root: Optional[str | Path] = None) -> None:
        super().__init__(db_path, conn=conn)
        resolved_db = Path(db_path or get_state_db_path()).resolve()
        self.artifact_root = Path(artifact_root).resolve() if artifact_root else resolved_db.parent / "source-artifacts"

    def append_routine_event(self, execution_id: str, event_type: str, payload: Dict[str, Any], portfolio_id: Optional[str] = None) -> dict[str, Any]:
        with self._connection(write=True) as conn:
            return institutional_ledger_dao.append_routine_event(
                conn, execution_id=execution_id, event_type=event_type, payload=payload, portfolio_id=portfolio_id
            )

    def list_routine_events(self) -> list[dict[str, Any]]:
        with self._connection() as conn:
            return institutional_ledger_dao.list_routine_events(conn)

    def save_catalyst_revision(self, catalyst: Dict[str, Any], portfolio_id: Optional[str] = None) -> dict[str, Any]:
        with self._connection(write=True) as conn:
            return institutional_ledger_dao.append_catalyst_revision(
                conn, catalyst, portfolio_id=portfolio_id
            )

    def list_latest_catalysts(self) -> list[dict[str, Any]]:
        with self._connection() as conn:
            return institutional_ledger_dao.list_latest_catalysts(conn)

    def _materialize_source(self, source: Dict[str, Any]) -> dict[str, Any]:
        content = str(source.get("content") or source.get("content_snippet") or "")
        if not content.strip():
            raise ValueError("source capture body is required; a locator alone is not evidence")
        content_sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
        supplied_sha = str(source.get("content_sha256") or source.get("content_hash") or "").strip()
        if supplied_sha and supplied_sha != content_sha:
            raise ValueError("source capture content hash does not match the supplied body")
        relative_ref = Path("sha256") / content_sha[:2] / f"{content_sha}.txt"
        target = (self.artifact_root / relative_ref).resolve()
        if not target.is_relative_to(self.artifact_root):
            raise ValueError("source artifact path escaped the configured artifact root")
        if target.exists():
            existing = target.read_bytes()
            if hashlib.sha256(existing).hexdigest() != content_sha:
                raise ValueError("content-addressed source artifact is corrupt")
        else:
            _atomic_write_to(target, content)
            if hashlib.sha256(target.read_bytes()).hexdigest() != content_sha:
                raise IOError("source artifact failed post-write hash verification")
        captured_at = str(source.get("captured_at_utc") or "")
        source_id = str(source.get("source_id") or "").strip()
        if not source_id or not captured_at:
            raise ValueError("source_id and captured_at_utc are required")
        revision = str(source.get("revision") or source.get("source_revision") or content_sha)
        locator = str(source.get("source_locator") or source.get("source_url") or "").strip()
        if not locator:
            raise ValueError("source_locator is required")
        metadata = {
            "entity_id": source.get("entity_id"),
            "source_type": source.get("source_type"),
            "effective_period": source.get("effective_period"),
            "is_untrusted_instruction": bool(source.get("is_untrusted_instruction", False)),
        }
        return {
            "source_id": source_id,
            "revision": revision,
            "issuer_id": source.get("issuer_id") or source.get("entity_id"),
            "ticker": source.get("ticker") or source.get("entity_id"),
            "document_type": str(source.get("document_type") or source.get("source_type") or "UNKNOWN"),
            "fiscal_period": source.get("fiscal_period") or source.get("effective_period"),
            "source_locator": locator,
            "artifact_ref": relative_ref.as_posix(),
            "content_sha256": content_sha,
            "captured_at_utc": captured_at,
            "available_at_utc": source.get("available_at_utc") or captured_at,
            "metadata": metadata,
            "content_snippet": content,
        }

    def capture_source(self, source: Dict[str, Any]) -> dict[str, Any]:
        materialized = self._materialize_source(dict(source))
        stored_fields = {k: v for k, v in materialized.items() if k != "content_snippet"}
        with self._connection(write=True) as conn:
            stored = institutional_ledger_dao.insert_source_capture(conn, stored_fields)
        return {**stored, **materialized}

    def list_source_captures(self) -> list[dict[str, Any]]:
        with self._connection() as conn:
            rows = institutional_ledger_dao.list_source_captures(conn)
        result = []
        for row in rows:
            target = (self.artifact_root / Path(row["artifact_ref"])).resolve()
            if not target.is_relative_to(self.artifact_root) or not target.is_file():
                raise FileNotFoundError(f"source artifact unavailable: {row['artifact_ref']}")
            content = target.read_bytes()
            if hashlib.sha256(content).hexdigest() != row["content_sha256"]:
                raise ValueError(f"source artifact hash mismatch: {row['artifact_ref']}")
            metadata = row.get("metadata") or {}
            result.append({
                **row,
                "entity_id": metadata.get("entity_id") or row.get("ticker") or row.get("issuer_id") or "",
                "source_type": metadata.get("source_type") or row["document_type"],
                "effective_period": metadata.get("effective_period") or row.get("fiscal_period") or "",
                "is_untrusted_instruction": bool(metadata.get("is_untrusted_instruction", False)),
                "content_snippet": content.decode("utf-8"),
                "content_hash": row["content_sha256"],
            })
        return result

    def save_catalyst_assessment(self, assessment: Dict[str, Any]) -> dict[str, Any]:
        with self._connection(write=True) as conn:
            return institutional_ledger_dao.insert_catalyst_assessment(conn, assessment)

    def list_catalyst_assessments(self) -> list[dict[str, Any]]:
        with self._connection() as conn:
            return institutional_ledger_dao.list_catalyst_assessments(conn)

    def save_estimate_record(self, record: Dict[str, Any]) -> dict[str, Any]:
        with self._connection(write=True) as conn:
            return institutional_ledger_dao.insert_estimate_record(conn, record)

    def list_estimate_records(self) -> list[dict[str, Any]]:
        with self._connection() as conn:
            return institutional_ledger_dao.list_estimate_records(conn)

    def import_legacy_records(self, source_digest: str, record_family: str, records: list[Dict[str, Any]]) -> dict[str, int]:
        """Idempotently import a validated legacy JSON list in one DB transaction."""
        if record_family not in {"routine", "catalyst", "estimate", "source"}:
            raise ValueError(f"unsupported legacy record family: {record_family}")
        prepared_sources = []
        if record_family == "source":
            prepared_sources = [self._materialize_source(dict(row)) for row in records]
        imported = skipped = 0
        with self._connection(write=True) as conn:
            for index, original in enumerate(records):
                record = dict(original)
                if record_family == "routine":
                    legacy_id = str(
                        record.get("event_id") or record.get("routine_event_id")
                        or f"{record.get('execution_id', '')}:{record.get('event_sequence', index)}"
                    )
                    record.setdefault(
                        "event_id",
                        "LEGACY-" + hashlib.sha256(
                            json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
                        ).hexdigest()[:32],
                    )
                elif record_family == "source":
                    legacy_id = f"{record.get('source_id', '')}:{record.get('revision') or record.get('source_revision') or index}"
                elif record_family == "catalyst":
                    legacy_id = str(record.get("catalyst_id") or index)
                else:
                    legacy_id = str(record.get("estimate_id") or index)
                payload_json = json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
                payload_sha = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
                existing = conn.execute(
                    """SELECT payload_sha256 FROM state_import_receipts
                       WHERE source_digest=? AND record_family=? AND legacy_record_id=?""",
                    (source_digest, record_family, legacy_id),
                ).fetchone()
                if existing:
                    existing_sha = existing[0]
                    if existing_sha != payload_sha:
                        raise ValueError(f"import conflict for {record_family}:{legacy_id}")
                    skipped += 1
                    continue
                if record_family == "routine":
                    institutional_ledger_dao.append_routine_event(
                        conn, execution_id=str(record["execution_id"]),
                        event_type=str(record.get("event_type") or record.get("status") or "legacy_import"), payload=record
                    )
                    target_id = legacy_id
                elif record_family == "catalyst":
                    row = institutional_ledger_dao.append_catalyst_revision(conn, record)
                    target_id = str(row["catalyst_id"])
                elif record_family == "estimate":
                    row = institutional_ledger_dao.insert_estimate_record(conn, record)
                    target_id = str(row["estimate_id"])
                else:
                    materialized = prepared_sources[index]
                    fields = {k: v for k, v in materialized.items() if k != "content_snippet"}
                    row = institutional_ledger_dao.insert_source_capture(conn, fields)
                    target_id = f"{row['source_id']}:{row['revision']}"
                institutional_ledger_dao.insert_import_receipt(
                    conn, source_digest=source_digest, record_family=record_family,
                    legacy_record_id=legacy_id, payload_sha256=payload_sha,
                    target_record_id=target_id,
                )
                imported += 1
        return {"imported": imported, "skipped": skipped}
