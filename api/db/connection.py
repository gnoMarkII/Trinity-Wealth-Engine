"""SQLite connection, schema initialization, and schema migrations."""
import os
import sqlite3
import threading

from api.config import get_state_db_path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL,
    card_id TEXT,
    idempotency_key TEXT UNIQUE,
    instruction TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    error_message TEXT,
    flow TEXT NOT NULL DEFAULT 'manager',
    interrupt_payload TEXT,
    resume_value TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS job_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    node_name TEXT,
    content TEXT,
    role TEXT NOT NULL DEFAULT 'reply',
    label TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_job_logs_job_id ON job_logs(job_id, seq);

CREATE TABLE IF NOT EXISTS used_eligibility_tokens (
    token_hash TEXT PRIMARY KEY,
    jti TEXT NOT NULL UNIQUE,
    job_id TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    pitch_id TEXT NOT NULL,
    approval_revision INTEGER NOT NULL,
    used_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_used_eligibility_tokens_job ON used_eligibility_tokens(job_id);

CREATE TABLE IF NOT EXISTS kanban_cards (
    card_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    column_name TEXT NOT NULL DEFAULT 'backlog',
    job_id TEXT,
    flow TEXT NOT NULL DEFAULT 'manager',
    display_seq INTEGER,
    is_verified INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS dcf_evaluations_ledger (
    evaluation_id TEXT PRIMARY KEY,
    ticker TEXT NOT NULL,
    market TEXT NOT NULL,
    evaluated_at TEXT NOT NULL,
    model_version TEXT NOT NULL DEFAULT 'dcf_v1.0',
    valuation_price_basis TEXT NOT NULL DEFAULT 'split_adjusted_only',
    current_price_at_eval REAL,
    wacc_pct REAL,
    valuation_verdict TEXT,
    scenarios_json TEXT NOT NULL,
    corporate_action_evidence_json TEXT,
    input_snapshot_json TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_dcf_evaluations_ticker ON dcf_evaluations_ledger(ticker, evaluated_at DESC);

CREATE TABLE IF NOT EXISTS sec_form4_raw_ledger (
    accession_number TEXT PRIMARY KEY,
    issuer_cik TEXT NOT NULL,
    ticker TEXT NOT NULL,
    filing_url TEXT NOT NULL,
    filed_at TEXT NOT NULL,
    reporting_owner_cik TEXT,
    reporting_owner_name TEXT,
    is_director INTEGER NOT NULL DEFAULT 0,
    is_officer INTEGER NOT NULL DEFAULT 0,
    is_ten_percent_owner INTEGER NOT NULL DEFAULT 0,
    officer_title TEXT,
    raw_xml_payload TEXT,
    is_amendment INTEGER NOT NULL DEFAULT 0,
    amends_accession_number TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sec_raw_ticker ON sec_form4_raw_ledger(ticker, filed_at DESC);

CREATE TABLE IF NOT EXISTS sec_insider_transactions (
    transaction_id TEXT PRIMARY KEY,
    accession_number TEXT NOT NULL,
    ticker TEXT NOT NULL,
    transaction_date TEXT NOT NULL,
    transaction_code TEXT NOT NULL,
    shares REAL NOT NULL,
    price_per_share REAL NOT NULL,
    acquired_or_disposed TEXT NOT NULL,
    shares_owned_following REAL,
    ownership_nature TEXT,
    is_derivative INTEGER NOT NULL DEFAULT 0,
    normalized_weight REAL NOT NULL DEFAULT 1.0,
    created_at REAL NOT NULL,
    FOREIGN KEY (accession_number) REFERENCES sec_form4_raw_ledger(accession_number)
);
CREATE INDEX IF NOT EXISTS idx_insider_tx_ticker ON sec_insider_transactions(ticker, transaction_date DESC);

CREATE TABLE IF NOT EXISTS analyst_context_cache (
    ticker           TEXT PRIMARY KEY,
    provider_symbol  TEXT NOT NULL,
    market           TEXT NOT NULL,
    currency         TEXT NOT NULL,
    exchange_tz      TEXT NOT NULL,
    target_mean      REAL,
    target_high      REAL,
    target_low       REAL,
    num_analysts     INTEGER,
    next_earnings_date TEXT,
    eps_history_json TEXT NOT NULL DEFAULT '[]',
    source_as_of     TEXT,
    data_status      TEXT NOT NULL DEFAULT 'ok',
    synced_at        REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_analyst_context_ticker ON analyst_context_cache(ticker);

CREATE TABLE IF NOT EXISTS financial_statements_cache (
    market           TEXT NOT NULL,
    provider_symbol  TEXT NOT NULL,
    provider         TEXT NOT NULL,
    data_json        TEXT NOT NULL,
    synced_at        REAL NOT NULL,
    PRIMARY KEY (market, provider_symbol)
);
CREATE INDEX IF NOT EXISTS idx_financial_statements_cache_pk ON financial_statements_cache(market, provider_symbol);

CREATE TABLE IF NOT EXISTS notification_outbox (
    event_id TEXT PRIMARY KEY,
    idempotency_key TEXT NOT NULL UNIQUE,
    aggregate_type TEXT NOT NULL,
    aggregate_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    sent_at REAL
);
CREATE INDEX IF NOT EXISTS idx_notification_outbox_pending
    ON notification_outbox(status, created_at ASC);

CREATE TABLE IF NOT EXISTS earnings_call_runs (
    run_id TEXT PRIMARY KEY,
    source_key TEXT NOT NULL UNIQUE,
    ticker TEXT NOT NULL,
    period TEXT NOT NULL,
    transcript_hash TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    status TEXT NOT NULL,
    kanban_status TEXT NOT NULL DEFAULT 'none',
    highlights TEXT,
    vault_path TEXT,
    kanban_card_id TEXT,
    execution_token TEXT,
    execution_expires_at REAL,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    last_error_code TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_earnings_call_runs_ticker ON earnings_call_runs(ticker, created_at DESC);

CREATE TABLE IF NOT EXISTS earnings_call_outbox (
    event_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    source_key TEXT NOT NULL,
    event_type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    available_at REAL NOT NULL,
    lease_token TEXT,
    lease_expires_at REAL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE(run_id, event_type)
);
CREATE INDEX IF NOT EXISTS idx_earnings_call_outbox_pending
    ON earnings_call_outbox(status, available_at ASC);
"""

_INITIALIZED_DB_PATHS: set[str] = set()
_INIT_LOCK = threading.Lock()

_COLUMN_MIGRATIONS: dict[str, dict[str, str]] = {
    "jobs": {
        "flow": "flow TEXT NOT NULL DEFAULT 'manager'",
        "interrupt_payload": "interrupt_payload TEXT",
        "resume_value": "resume_value TEXT",
        "scope": "scope TEXT NOT NULL DEFAULT 'both'",
    },
    "job_logs": {
        "role": "role TEXT NOT NULL DEFAULT 'reply'",
        "label": "label TEXT",
    },
    "kanban_cards": {
        "flow": "flow TEXT NOT NULL DEFAULT 'manager'",
        "display_seq": "display_seq INTEGER",
        "prompt": "prompt TEXT",
        "source_key": "source_key TEXT",
        "scope": "scope TEXT NOT NULL DEFAULT 'both'",
        "discord_notify": "discord_notify INTEGER NOT NULL DEFAULT 1",
        "discord_sent_events": "discord_sent_events TEXT",
        "is_verified": "is_verified INTEGER NOT NULL DEFAULT 1",
    },
}


def _migrate_columns(conn: sqlite3.Connection) -> None:
    """เพิ่มคอลัมน์ใหม่ให้ตารางเก่าที่มีอยู่แล้วในไฟล์ SQLite จริง — `CREATE TABLE IF NOT EXISTS`
    ไม่แก้ตารางที่มีอยู่แล้ว ถ้า schema เปลี่ยนหลังจากไฟล์ .sqlite ถูกสร้างไปแล้ว (เช่น
    เพิ่ม flow/interrupt_payload ตอนทำ HITL) คอลัมน์ใหม่จะไม่มีอยู่จริง ทำให้ INSERT/SELECT
    พังด้วย "table X has no column named Y" — พบเจอจริงตอน dispatch งานจาก Kanban
    """
    for table, columns in _COLUMN_MIGRATIONS.items():
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        for col_name, col_def in columns.items():
            if col_name not in existing:
                try:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_def}")
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e).lower():
                        raise
    conn.commit()


def _migrate_dispatcher_column_cards(conn: sqlite3.Connection) -> None:
    """คอลัมน์ 'dispatcher' ถูกตัดออกจาก UI แล้ว (เหลือ backlog/approval/executing/done) —
    การ์ดเก่าที่ยังค้างอยู่ใน 'dispatcher' ต้องย้ายกลับ backlog ไม่งั้นจะไม่โผล่ในหน้าเว็บเลย
    เพราะ frontend ไม่มีคอลัมน์นั้นให้ render อีกต่อไป
    """
    conn.execute("UPDATE kanban_cards SET column_name = 'backlog' WHERE column_name = 'dispatcher'")
    conn.commit()


def _backfill_kanban_display_seq(conn: sqlite3.Connection) -> None:
    """การ์ดเก่าที่มีอยู่ก่อน Rev.2 (ก่อนมีคอลัมน์ display_seq) จะมีค่า NULL — เติมเลขให้
    ตามลำดับ created_at เพื่อให้ Linear-style #AG-N ID เรียงลำดับสร้างจริง ไม่ใช่เลขสุ่ม
    """
    cur = conn.execute("SELECT COUNT(*) FROM kanban_cards WHERE display_seq IS NULL")
    if cur.fetchone()[0] == 0:
        return
    cur = conn.execute("SELECT COALESCE(MAX(display_seq), 0) FROM kanban_cards")
    next_seq = cur.fetchone()[0] + 1
    rows = conn.execute(
        "SELECT card_id FROM kanban_cards WHERE display_seq IS NULL ORDER BY created_at ASC"
    ).fetchall()
    for row in rows:
        conn.execute("UPDATE kanban_cards SET display_seq = ? WHERE card_id = ?", (next_seq, row["card_id"]))
        next_seq += 1
    conn.commit()


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    conn.commit()
    _migrate_columns(conn)
    _migrate_dispatcher_column_cards(conn)
    _backfill_kanban_display_seq(conn)
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_kanban_cards_source_key "
        "ON kanban_cards(source_key) WHERE source_key IS NOT NULL"
    )
    conn.commit()


def get_connection(
    db_path: str | None = None,
    *,
    schema_initializer=None,
) -> sqlite3.Connection:
    path = db_path or get_state_db_path()
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
    try:
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA journal_mode=WAL")
    except sqlite3.OperationalError:
        pass
    conn.row_factory = sqlite3.Row
    with _INIT_LOCK:
        if path not in _INITIALIZED_DB_PATHS:
            (schema_initializer or init_schema)(conn)
            _INITIALIZED_DB_PATHS.add(path)
    return conn
