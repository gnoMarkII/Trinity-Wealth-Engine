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
import sqlite3
from typing import Any, Iterator, List, Optional, Dict

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
import api.db.repositories.job_repository as job_repo_dao
import api.db.repositories.kanban_repository as kanban_repo_dao
import api.db.repositories.cache_repository as cache_repo_dao
import api.db.repositories.dcf_repository as dcf_repo_dao
import api.db.repositories.insider_repository as insider_repo_dao
import api.db.repositories.outbox_repository as outbox_repo_dao
import api.db.repositories.earnings_call_repository as earnings_call_repo_dao
import api.db.repositories.earnings_call_outbox_repository as earnings_call_outbox_repo_dao
from application.earnings_call.ports import EarningsCallWorkflowPort
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
        provider: Optional[InsiderHistoryProviderPort] = None,
    ) -> None:
        self._db_path = db_path
        if provider is None:
            from tools.market.adapters.insider_provider import YFinanceInsiderHistoryAdapter

            provider = YFinanceInsiderHistoryAdapter()
        self._provider = provider

    def sync(self, ticker: str) -> None:
        from api.db.uow import DbUnitOfWork

        records = self._provider.fetch(ticker)
        with DbUnitOfWork(db_path=self._db_path) as uow:
            for parsed in records:
                insider_repo_dao.record_parsed_filing(uow.conn, parsed)


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
        outbox_lease_seconds: int = 60,
    ) -> tuple[EarningsCallRunDTO, EarningsCallOutboxEventDTO, LeaseDTO]:
        with self._connection(write=True) as conn:
            run_dto = earnings_call_repo_dao.mark_note_written(
                conn=conn,
                run_id=run_id,
                execution_token=execution_token,
                vault_path=vault_path,
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
            earnings_call_outbox_repo_dao.complete_event(
                conn=conn,
                event_id=event_id,
                lease_token=lease_token,
            )
            return earnings_call_repo_dao.complete_kanban_delivery(
                conn=conn,
                run_id=run_id,
                card_id=card_id,
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
            earnings_call_outbox_repo_dao.schedule_retry(
                conn=conn,
                event_id=event_id,
                lease_token=lease_token,
                error_code=error_code,
                retry_delay_seconds=retry_delay_seconds,
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
            earnings_call_outbox_repo_dao.mark_dead_letter(
                conn=conn,
                event_id=event_id,
                lease_token=lease_token,
                error_code=error_code,
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
                raise RuntimeError(f"Run '{run_id}' not found")
            event_dto, lease_dto = earnings_call_outbox_repo_dao.reset_for_manual_retry(
                conn=conn,
                run_id=run_id,
                lease_seconds=lease_seconds,
            )
            return run_dto, event_dto, lease_dto
