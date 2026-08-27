"""SQLite store สำหรับ job log + kanban state — Facade Re-export Layer.

Decomposed into DAO repositories under `api.db.*`:
  - `api.db.connection`: SQLite Connection, Schema Management & Migrations
  - `api.db.uow`: Unit of Work transaction manager
  - `api.db.repositories.job_repository`: Job Lifecycle & Job Logs
  - `api.db.repositories.kanban_repository`: Kanban Cards & Parking Lot Atomicity
  - `api.db.repositories.dcf_repository`: Canonical DCF Evaluations Ledger
  - `api.db.repositories.insider_repository`: SEC Form 4 Filings & Transactions
  - `api.db.repositories.cache_repository`: Analyst Context & Financial Statements Caches

This facade provides 100% backward compatibility for all existing legacy imports and tests.
"""
import sqlite3
from contextlib import closing
from typing import Optional, List, Dict, Any

from api.db.connection import (
    _SCHEMA,
    _COLUMN_MIGRATIONS,
    _INITIALIZED_DB_PATHS,
    _INIT_LOCK,
    _migrate_columns,
    _migrate_dispatcher_column_cards,
    _backfill_kanban_display_seq,
    init_schema,
    get_connection as _get_connection,
)
from api.db.uow import DbUnitOfWork
import api.db.repositories.job_repository as _job_repo
import api.db.repositories.kanban_repository as _kanban_repo
import api.db.repositories.dcf_repository as _dcf_repo
import api.db.repositories.insider_repository as _insider_repo
import api.db.repositories.cache_repository as _cache_repo


def get_connection(db_path: str | None = None) -> sqlite3.Connection:
    """Compatibility connection factory preserving monkeypatchable init hook."""
    return _get_connection(db_path, schema_initializer=init_schema)


# ---------------------------------------------------------
# Job Repository Facade
# ---------------------------------------------------------

def create_job(
    conn: sqlite3.Connection,
    job_id: str,
    thread_id: str,
    card_id: str | None,
    idempotency_key: str,
    instruction: str,
    status: str = "queued",
    flow: str = "manager",
    scope: str = "both",
) -> None:
    _job_repo.create_job(
        conn=conn,
        job_id=job_id,
        thread_id=thread_id,
        card_id=card_id,
        idempotency_key=idempotency_key,
        instruction=instruction,
        status=status,
        flow=flow,
        scope=scope,
    )
    conn.commit()


def set_job_awaiting_approval(conn: sqlite3.Connection, job_id: str, interrupt_payload_json: str) -> None:
    _job_repo.set_job_awaiting_approval(conn, job_id, interrupt_payload_json)
    conn.commit()


def set_job_resume_value(conn: sqlite3.Connection, job_id: str, resume_value_json: str) -> None:
    _job_repo.set_job_resume_value(conn, job_id, resume_value_json)
    conn.commit()


def claim_job_resume(
    conn: sqlite3.Connection,
    *,
    job_id: str,
    resume_value_json: str,
    token_uses: list[dict[str, str | int]] | None = None,
) -> None:
    _job_repo.claim_job_resume(
        conn=conn,
        job_id=job_id,
        resume_value_json=resume_value_json,
        token_uses=token_uses,
    )
    conn.commit()


def clear_job_resume_value(conn: sqlite3.Connection, job_id: str) -> None:
    _job_repo.clear_job_resume_value(conn, job_id)
    conn.commit()


def find_job_by_idempotency_key(conn: sqlite3.Connection, idempotency_key: str) -> sqlite3.Row | None:
    return _job_repo.find_job_by_idempotency_key(conn, idempotency_key)


def get_job(conn: sqlite3.Connection, job_id: str) -> sqlite3.Row | None:
    return _job_repo.get_job(conn, job_id)


def update_job_status(
    conn: sqlite3.Connection,
    job_id: str,
    status: str,
    error_message: str | None = None,
) -> None:
    _job_repo.update_job_status(conn, job_id, status, error_message)
    conn.commit()


def cas_job_status(
    conn: sqlite3.Connection,
    job_id: str,
    old_status: str,
    new_status: str,
) -> bool:
    res = _job_repo.cas_job_status(conn, job_id, old_status, new_status)
    conn.commit()
    return res


def list_jobs_by_status(
    conn: sqlite3.Connection,
    statuses: list,
    flows: list[str] | None = None,
) -> list:
    """flows=None (default) = ทุก flow เหมือนเดิมทุกประการ — ใส่ให้ JobQueue ที่แชร์ DB เดียวกันกับ
คิวอื่น (เช่น notebooklm_job_queue) กรองเฉพาะ flow ของตัวเอง กัน reenqueue_pending() ข้ามคิวไปกวาด
งานคนละ flow มาประมวลผลผิดที่"""
    return _job_repo.list_jobs_by_status(conn, statuses, flows)


def append_job_log(
    conn: sqlite3.Connection,
    job_id: str,
    node_name: str,
    content: str,
    role: str = "reply",
    label: str | None = None,
) -> None:
    _job_repo.append_job_log(conn, job_id, node_name, content, role, label)
    conn.commit()


def get_job_logs_since(
    conn: sqlite3.Connection,
    job_id: str,
    after_seq: int = 0,
) -> list[sqlite3.Row]:
    return _job_repo.get_job_logs_since(conn, job_id, after_seq)


def get_job_reply_logs(
    conn: sqlite3.Connection,
    job_id: str,
) -> list[sqlite3.Row]:
    return _job_repo.get_job_reply_logs(conn, job_id)


def get_latest_job_log_node(
    conn: sqlite3.Connection,
    job_id: str,
) -> str | None:
    return _job_repo.get_latest_job_log_node(conn, job_id)


def get_job_log_count(
    conn: sqlite3.Connection,
    job_id: str,
) -> int:
    return _job_repo.get_job_log_count(conn, job_id)


# ---------------------------------------------------------
# Kanban Repository Facade
# ---------------------------------------------------------

def list_kanban_cards(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return _kanban_repo.list_kanban_cards(conn)


def create_kanban_card(
    conn: sqlite3.Connection,
    card_id: str,
    title: str,
    column_name: str = "backlog",
    flow: str = "manager",
    prompt: str | None = None,
    scope: str = "both",
    is_verified: bool = True,
) -> None:
    _kanban_repo.create_kanban_card(
        conn=conn,
        card_id=card_id,
        title=title,
        column_name=column_name,
        flow=flow,
        prompt=prompt,
        scope=scope,
        is_verified=is_verified,
    )
    conn.commit()


def create_parking_lot_cards_atomic(
    ideas: list[str],
    source_pitch_id: str,
    db_path: str | None = None,
) -> int:
    # The compatibility facade owns the historical standalone transaction;
    # the inner DAO only executes SQL on this connection.
    with closing(get_connection(db_path)) as conn:
        conn.isolation_level = None
        conn.execute("BEGIN IMMEDIATE")
        try:
            created = _kanban_repo.create_parking_lot_cards(
                conn=conn,
                ideas=ideas,
                source_pitch_id=source_pitch_id,
            )
            conn.execute("COMMIT")
            return created
        except Exception:
            conn.execute("ROLLBACK")
            raise


def update_kanban_card(
    conn: sqlite3.Connection, card_id: str, title: str, prompt: str | None, flow: str, scope: str
) -> None:
    _kanban_repo.update_kanban_card(conn, card_id, title, prompt, flow, scope)
    conn.commit()


def set_kanban_card_source(conn: sqlite3.Connection, card_id: str, prompt: str, is_verified: bool) -> None:
    _kanban_repo.set_kanban_card_source(conn, card_id, prompt, is_verified)
    conn.commit()


def toggle_kanban_card_discord(conn: sqlite3.Connection, card_id: str, enabled: bool) -> None:
    _kanban_repo.toggle_kanban_card_discord(conn, card_id, enabled)
    conn.commit()


def mark_discord_events_sent(conn: sqlite3.Connection, card_id: str, event_ids: list[str]) -> None:
    _kanban_repo.mark_discord_events_sent(conn, card_id, event_ids)
    conn.commit()


def move_kanban_card(conn: sqlite3.Connection, card_id: str, column_name: str, job_id: str | None = None) -> None:
    _kanban_repo.move_kanban_card(conn, card_id, column_name, job_id)
    conn.commit()


def get_kanban_card(conn: sqlite3.Connection, card_id: str) -> sqlite3.Row | None:
    return _kanban_repo.get_kanban_card(conn, card_id)


def find_kanban_card_by_title_in_column(
    conn: sqlite3.Connection, title: str, column_name: str, prompt: str | None = None
) -> sqlite3.Row | None:
    return _kanban_repo.find_kanban_card_by_title_in_column(conn, title, column_name, prompt)


def delete_kanban_card(conn: sqlite3.Connection, card_id: str) -> None:
    _kanban_repo.delete_kanban_card(conn, card_id)
    conn.commit()


# ---------------------------------------------------------
# DCF Repository Facade
# ---------------------------------------------------------

def record_dcf_evaluation(
    conn: sqlite3.Connection,
    evaluation_id: str,
    ticker: str,
    market: str,
    evaluated_at: str,
    scenarios: dict,
    model_version: str = "dcf_v1.0",
    valuation_price_basis: str = "split_adjusted_only",
    current_price_at_eval: float | None = None,
    wacc_pct: float | None = None,
    valuation_verdict: str = "unknown",
    corporate_action_evidence: list | None = None,
    input_snapshot: dict | None = None,
) -> None:
    _dcf_repo.record_dcf_evaluation(
        conn=conn,
        evaluation_id=evaluation_id,
        ticker=ticker,
        market=market,
        evaluated_at=evaluated_at,
        scenarios=scenarios,
        model_version=model_version,
        valuation_price_basis=valuation_price_basis,
        current_price_at_eval=current_price_at_eval,
        wacc_pct=wacc_pct,
        valuation_verdict=valuation_verdict,
        corporate_action_evidence=corporate_action_evidence,
        input_snapshot=input_snapshot,
    )
    conn.commit()


def get_latest_dcf_evaluation(conn: sqlite3.Connection, ticker: str) -> sqlite3.Row | None:
    return _dcf_repo.get_latest_dcf_evaluation(conn, ticker)


def get_dcf_evaluation_by_id(conn: sqlite3.Connection, evaluation_id: str) -> sqlite3.Row | None:
    return _dcf_repo.get_dcf_evaluation_by_id(conn, evaluation_id)


# ---------------------------------------------------------
# Insider Repository Facade
# ---------------------------------------------------------

def record_sec_form4_filing(
    conn: sqlite3.Connection,
    accession_number: str,
    issuer_cik: str,
    ticker: str,
    filing_url: str,
    filed_at: str,
    reporting_owner_cik: str | None = None,
    reporting_owner_name: str | None = None,
    is_director: bool = False,
    is_officer: bool = False,
    is_ten_percent_owner: bool = False,
    officer_title: str | None = None,
    raw_xml_payload: str | None = None,
    is_amendment: bool = False,
    amends_accession_number: str | None = None,
) -> None:
    _insider_repo.record_sec_form4_filing(
        conn=conn,
        accession_number=accession_number,
        issuer_cik=issuer_cik,
        ticker=ticker,
        filing_url=filing_url,
        filed_at=filed_at,
        reporting_owner_cik=reporting_owner_cik,
        reporting_owner_name=reporting_owner_name,
        is_director=is_director,
        is_officer=is_officer,
        is_ten_percent_owner=is_ten_percent_owner,
        officer_title=officer_title,
        raw_xml_payload=raw_xml_payload,
        is_amendment=is_amendment,
        amends_accession_number=amends_accession_number,
    )
    conn.commit()


def record_sec_insider_transaction(
    conn: sqlite3.Connection,
    transaction_id: str,
    accession_number: str,
    ticker: str,
    transaction_date: str,
    transaction_code: str,
    shares: float,
    price_per_share: float,
    acquired_or_disposed: str,
    shares_owned_following: float | None = None,
    ownership_nature: str | None = None,
    is_derivative: bool = False,
    normalized_weight: float = 1.0,
) -> None:
    _insider_repo.record_sec_insider_transaction(
        conn=conn,
        transaction_id=transaction_id,
        accession_number=accession_number,
        ticker=ticker,
        transaction_date=transaction_date,
        transaction_code=transaction_code,
        shares=shares,
        price_per_share=price_per_share,
        acquired_or_disposed=acquired_or_disposed,
        shares_owned_following=shares_owned_following,
        ownership_nature=ownership_nature,
        is_derivative=is_derivative,
        normalized_weight=normalized_weight,
    )
    conn.commit()


def get_sec_insider_filings_and_transactions(
    conn: sqlite3.Connection, ticker: str, since_date: str | None = None
) -> list[dict]:
    return _insider_repo.get_sec_insider_filings_and_transactions(conn, ticker, since_date)


# ---------------------------------------------------------
# Cache Repository Facade
# ---------------------------------------------------------

def get_analyst_context_cache(conn: sqlite3.Connection, ticker: str) -> dict | None:
    return _cache_repo.get_analyst_context_cache(conn, ticker)


def upsert_analyst_context_cache(conn: sqlite3.Connection, ticker: str, data: dict) -> None:
    _cache_repo.upsert_analyst_context_cache(conn, ticker, data)
    conn.commit()


def get_financial_statements_cache(
    conn: sqlite3.Connection, market: str, provider_symbol: str
) -> dict | None:
    return _cache_repo.get_financial_statements_cache(conn, market, provider_symbol)


def get_raw_financial_statements_cache(
    conn: sqlite3.Connection, market: str, provider_symbol: str
) -> dict | None:
    return _cache_repo.get_raw_financial_statements_cache(conn, market, provider_symbol)


def upsert_financial_statements_cache(
    conn: sqlite3.Connection,
    market: str,
    provider_symbol: str,
    provider: str,
    data_json: str,
    synced_at: float,
) -> None:
    _cache_repo.upsert_financial_statements_cache(
        conn=conn,
        market=market,
        provider_symbol=provider_symbol,
        provider=provider,
        data_json=data_json,
        synced_at=synced_at,
    )
    conn.commit()


__all__ = [
    "_SCHEMA",
    "_COLUMN_MIGRATIONS",
    "_INITIALIZED_DB_PATHS",
    "_INIT_LOCK",
    "_migrate_columns",
    "_migrate_dispatcher_column_cards",
    "_backfill_kanban_display_seq",
    "init_schema",
    "get_connection",
    "DbUnitOfWork",
    "create_job",
    "set_job_awaiting_approval",
    "set_job_resume_value",
    "claim_job_resume",
    "clear_job_resume_value",
    "find_job_by_idempotency_key",
    "get_job",
    "update_job_status",
    "cas_job_status",
    "list_jobs_by_status",
    "append_job_log",
    "get_job_logs_since",
    "get_job_reply_logs",
    "get_latest_job_log_node",
    "get_job_log_count",
    "list_kanban_cards",
    "create_kanban_card",
    "create_parking_lot_cards_atomic",
    "update_kanban_card",
    "set_kanban_card_source",
    "toggle_kanban_card_discord",
    "mark_discord_events_sent",
    "move_kanban_card",
    "get_kanban_card",
    "find_kanban_card_by_title_in_column",
    "delete_kanban_card",
    "record_dcf_evaluation",
    "get_latest_dcf_evaluation",
    "get_dcf_evaluation_by_id",
    "record_sec_form4_filing",
    "record_sec_insider_transaction",
    "get_sec_insider_filings_and_transactions",
    "get_analyst_context_cache",
    "upsert_analyst_context_cache",
    "get_financial_statements_cache",
    "get_raw_financial_statements_cache",
    "upsert_financial_statements_cache",
]
