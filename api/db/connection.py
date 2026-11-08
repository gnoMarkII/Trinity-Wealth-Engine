"""SQLite connection, schema initialization, and schema migrations."""
import hashlib
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
    revision_ref TEXT,
    content_sha256 TEXT,
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

CREATE TABLE IF NOT EXISTS macro_notebooklm_exports (
    export_id TEXT PRIMARY KEY,
    request_key TEXT UNIQUE,
    content_hash TEXT NOT NULL,
    job_id TEXT,
    state TEXT NOT NULL DEFAULT 'queued',
    stage TEXT NOT NULL DEFAULT 'initialized',
    snapshot_at TEXT NOT NULL,
    strategy_report_id TEXT,
    notebook_id TEXT,
    notebook_url TEXT,
    manifest_path TEXT,
    inventory_json TEXT,
    warnings_json TEXT,
    error_code TEXT,
    error_message TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_macro_export_hash ON macro_notebooklm_exports(content_hash);
CREATE INDEX IF NOT EXISTS idx_macro_export_created ON macro_notebooklm_exports(created_at DESC);

CREATE TABLE IF NOT EXISTS equity_notebooklm_exports (
    export_id TEXT PRIMARY KEY,
    request_key TEXT UNIQUE,
    ticker TEXT NOT NULL,
    market TEXT NOT NULL DEFAULT 'US',
    content_hash TEXT NOT NULL,
    job_id TEXT,
    state TEXT NOT NULL DEFAULT 'queued',
    stage TEXT NOT NULL DEFAULT 'initialized',
    snapshot_at TEXT NOT NULL,
    notebook_id TEXT,
    notebook_url TEXT,
    manifest_path TEXT,
    inventory_json TEXT,
    warnings_json TEXT,
    error_code TEXT,
    error_message TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_equity_export_hash ON equity_notebooklm_exports(ticker, market, content_hash);
CREATE INDEX IF NOT EXISTS idx_equity_export_created ON equity_notebooklm_exports(ticker, market, created_at DESC);

-- Investor Essence Runtime Tables
CREATE TABLE IF NOT EXISTS investor_sessions (
    session_id TEXT PRIMARY KEY,
    scope TEXT NOT NULL DEFAULT 'workspace',
    status TEXT NOT NULL,
    active_branch_id TEXT NOT NULL DEFAULT 'main',
    revision INTEGER NOT NULL DEFAULT 1,
    schema_version INTEGER NOT NULL DEFAULT 1,
    interview_version TEXT NOT NULL DEFAULT '1.0',
    context_hash TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS investor_questions (
    question_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    branch_id TEXT NOT NULL,
    sequence_no INTEGER NOT NULL,
    text TEXT NOT NULL,
    options_json TEXT NOT NULL,
    evidence_type TEXT NOT NULL,
    coverage_topics_json TEXT NOT NULL,
    is_clarification INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_investor_questions_seq ON investor_questions(session_id, branch_id, sequence_no);

CREATE TABLE IF NOT EXISTS investor_answers (
    answer_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    question_id TEXT NOT NULL,
    branch_id TEXT NOT NULL,
    answer_kind TEXT NOT NULL,
    option_id TEXT,
    free_text TEXT,
    evidence_type TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_investor_answers_q ON investor_answers(session_id, question_id);

CREATE TABLE IF NOT EXISTS investor_summary_drafts (
    session_id TEXT PRIMARY KEY,
    statement TEXT NOT NULL,
    claims_json TEXT NOT NULL,
    unresolved_topics_json TEXT NOT NULL,
    coverage_report_json TEXT NOT NULL,
    evidence_snapshot_hash TEXT NOT NULL,
    revision INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS investor_context_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    portfolio_id TEXT NOT NULL,
    data_json TEXT NOT NULL,
    is_ready INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS investor_plan_drafts (
    draft_id TEXT PRIMARY KEY,
    draft_type TEXT NOT NULL,
    portfolio_id TEXT NOT NULL,
    data_json TEXT NOT NULL,
    revision INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_investor_plan_drafts_port ON investor_plan_drafts(portfolio_id, draft_type);

CREATE TABLE IF NOT EXISTS investor_confirmed_refs (
    pointer_key TEXT PRIMARY KEY,
    scope TEXT NOT NULL,
    portfolio_id TEXT,
    ref_type TEXT NOT NULL,
    artifact_ref_json TEXT NOT NULL,
    snapshot_json TEXT NOT NULL,
    revision INTEGER NOT NULL DEFAULT 1,
    confirmed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS investor_confirmed_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    scope TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    source_id TEXT NOT NULL,
    source_revision INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    snapshot_json TEXT NOT NULL,
    confirmed_at_iso TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_investor_confirmed_snapshots_lookup 
ON investor_confirmed_snapshots(scope, scope_id, kind, confirmed_at_iso DESC);

CREATE TABLE IF NOT EXISTS investor_operations (
    operation_id TEXT PRIMARY KEY,
    task_type TEXT NOT NULL,
    resource_id TEXT NOT NULL,
    status TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    result_json TEXT,
    error_message TEXT,
    lease_owner TEXT,
    lease_expires_at REAL,
    fencing_token INTEGER NOT NULL DEFAULT 0,
    attempts INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_investor_operations_status ON investor_operations(status, lease_expires_at);

CREATE TABLE IF NOT EXISTS investor_intents (
    intent_id TEXT PRIMARY KEY,
    intent_type TEXT NOT NULL,
    resource_id TEXT NOT NULL,
    status TEXT NOT NULL,
    request_fingerprint TEXT NOT NULL,
    receipt_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS investor_command_receipts (
    receipt_key TEXT PRIMARY KEY,
    scope TEXT NOT NULL,
    use_case TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    result_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS advisor_sessions (
    session_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS advisor_messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    citations_json TEXT NOT NULL DEFAULT '[]',
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_advisor_messages_session ON advisor_messages(session_id, created_at);

CREATE TABLE IF NOT EXISTS thesis_claims (
    claim_id TEXT PRIMARY KEY,
    ticker TEXT NOT NULL,
    version INTEGER NOT NULL,
    prior_claim_id TEXT,
    as_of_utc TEXT NOT NULL,
    claim_statement TEXT NOT NULL,
    falsifiers_json TEXT NOT NULL DEFAULT '[]',
    catalysts_json TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'ACTIVE',
    evidence_tier TEXT NOT NULL DEFAULT 'DOCUMENTED',
    author_role TEXT NOT NULL DEFAULT 'HUMAN_PM',
    checksum_sha256 TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_thesis_claims_ticker ON thesis_claims(ticker, version DESC);

CREATE TABLE IF NOT EXISTS thesis_exceptions (
    exception_id TEXT PRIMARY KEY,
    claim_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    detected_at_utc TEXT NOT NULL,
    exception_type TEXT NOT NULL,
    metric_name TEXT,
    expected_value TEXT,
    observed_value TEXT,
    source_citation TEXT,
    human_action TEXT NOT NULL DEFAULT 'REVIEW_REQUIRED',
    status TEXT NOT NULL DEFAULT 'PENDING'
);
CREATE INDEX IF NOT EXISTS idx_thesis_exceptions_ticker ON thesis_exceptions(ticker, status);

CREATE TABLE IF NOT EXISTS pm_review_signoffs (
    signoff_id TEXT PRIMARY KEY,
    portfolio_id TEXT NOT NULL DEFAULT 'default',
    review_date TEXT NOT NULL,
    signed_off_by TEXT NOT NULL,
    signed_off_at_utc TEXT NOT NULL,
    acknowledged_exceptions_count INTEGER NOT NULL,
    notes TEXT,
    packet_sha256 TEXT NOT NULL,
    verified_sample_ids_json TEXT
);
"""

_INITIALIZED_DB_PATHS: set[str] = set()
_INIT_LOCK = threading.Lock()

_COLUMN_MIGRATIONS: dict[str, dict[str, str]] = {
    "pm_review_signoffs": {
        "portfolio_id": "portfolio_id TEXT NOT NULL DEFAULT 'default'",
        "verified_sample_ids_json": "verified_sample_ids_json TEXT",
    },
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
    # Earnings Call tables were introduced before execution/outbox leasing was
    # finalized.  Keep these additive migrations so an existing deployment
    # can be upgraded in place instead of failing on SELECT/UPDATE of a new
    # column.
    "earnings_call_runs": {
        "kanban_status": "kanban_status TEXT NOT NULL DEFAULT 'none'",
        "execution_token": "execution_token TEXT",
        "execution_expires_at": "execution_expires_at REAL",
        "attempt_count": "attempt_count INTEGER NOT NULL DEFAULT 0",
        "last_error_code": "last_error_code TEXT",
        "revision_ref": "revision_ref TEXT",
        "content_sha256": "content_sha256 TEXT",
    },
    "earnings_call_outbox": {
        "source_key": "source_key TEXT NOT NULL DEFAULT ''",
        "event_type": "event_type TEXT NOT NULL DEFAULT 'deliver_kanban'",
        "status": "status TEXT NOT NULL DEFAULT 'pending'",
        "attempts": "attempts INTEGER NOT NULL DEFAULT 0",
        "last_error": "last_error TEXT",
        "available_at": "available_at REAL NOT NULL DEFAULT 0",
        "lease_token": "lease_token TEXT",
        "lease_expires_at": "lease_expires_at REAL",
    },
}


def _migrate_columns(conn: sqlite3.Connection) -> None:
    """เพิ่มคอลัมน์ใหม่ให้ตารางเก่าที่มีอยู่แล้วในไฟล์ SQLite จริง — `CREATE TABLE IF NOT EXISTS`
    ไม่แก้ตารางที่มีอยู่แล้ว ถ้า schema เปลี่ยนหลังจากไฟล์ .sqlite ถูกสร้างไปแล้ว (เช่น
    เพิ่ม flow/interrupt_payload ตอนทำ HITL) คอลัมน์ใหม่จะไม่มีอยู่จริง ทำให้ INSERT/SELECT
    พังด้วย "table X has no column named Y" — พบเจอจริงตอน dispatch งานจาก Kanban
    """
    for table, columns in _COLUMN_MIGRATIONS.items():
        existing = {
            (row["name"] if (hasattr(row, "keys") or isinstance(row, dict)) else row[1])
            for row in conn.execute(f"PRAGMA table_info({table})")
        }
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



def _migrate_pm_review_signoffs(conn: sqlite3.Connection) -> None:
    """Safely upgrades pm_review_signoffs table schema and indexes across legacy databases."""
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='pm_review_signoffs'")
    if not cur.fetchone():
        return

    existing = {
        (row["name"] if (hasattr(row, "keys") or isinstance(row, dict)) else row[1])
        for row in conn.execute("PRAGMA table_info(pm_review_signoffs)")
    }
    if "portfolio_id" not in existing:
        conn.execute("ALTER TABLE pm_review_signoffs ADD COLUMN portfolio_id TEXT NOT NULL DEFAULT 'default'")
    if "verified_sample_ids_json" not in existing:
        conn.execute("ALTER TABLE pm_review_signoffs ADD COLUMN verified_sample_ids_json TEXT")

    # Drop legacy single-column date-only index that breaks portfolio isolation
    conn.execute("DROP INDEX IF EXISTS idx_pm_signoff_date")
    conn.execute("DROP INDEX IF EXISTS idx_pm_signoffs_review_date")
    # Drop legacy unique index on (portfolio_id, review_date) to allow multi-revision append-only history
    conn.execute("DROP INDEX IF EXISTS idx_pm_signoff_port_date")

    # Recreate composite indexes:
    # 1. Unique index on (portfolio_id, packet_sha256) for append-only packet revision signoff
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_pm_signoff_port_packet ON pm_review_signoffs(portfolio_id, packet_sha256)")
    # 2. Non-unique index on (portfolio_id, review_date) for fast date-based lookup
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pm_signoff_port_date ON pm_review_signoffs(portfolio_id, review_date)")
    conn.commit()



_ROUND5_MIGRATION_ID = "20261107_institutional_storage_v2"
_ROUND5_DDL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    migration_id TEXT PRIMARY KEY,
    checksum_sha256 TEXT NOT NULL,
    applied_at_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS pm_review_signoffs (
    signoff_id TEXT PRIMARY KEY,
    review_date TEXT NOT NULL,
    signed_off_by TEXT NOT NULL,
    signed_off_at_utc TEXT NOT NULL,
    acknowledged_exceptions_count INTEGER NOT NULL,
    notes TEXT,
    packet_sha256 TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS pm_review_packets (
    packet_id TEXT PRIMARY KEY,
    portfolio_id TEXT NOT NULL,
    review_date TEXT NOT NULL,
    revision INTEGER NOT NULL CHECK (revision > 0),
    cutoff_utc TEXT NOT NULL,
    input_fingerprint_sha256 TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    packet_sha256 TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    UNIQUE (portfolio_id, packet_sha256),
    UNIQUE (portfolio_id, review_date, revision),
    UNIQUE (portfolio_id, review_date, input_fingerprint_sha256, policy_version)
);
CREATE INDEX IF NOT EXISTS idx_pm_packet_date
    ON pm_review_packets(portfolio_id, review_date, revision DESC);
CREATE TABLE IF NOT EXISTS routine_execution_events (
    event_id TEXT PRIMARY KEY,
    execution_id TEXT NOT NULL,
    event_sequence INTEGER NOT NULL CHECK (event_sequence > 0),
    portfolio_id TEXT,
    event_type TEXT NOT NULL,
    scheduled_at_utc TEXT,
    started_at_utc TEXT,
    completed_at_utc TEXT,
    payload_json TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    UNIQUE (execution_id, event_sequence)
);
CREATE INDEX IF NOT EXISTS idx_routine_execution_stream
    ON routine_execution_events(execution_id, event_sequence);
CREATE TABLE IF NOT EXISTS catalyst_revisions (
    catalyst_id TEXT NOT NULL,
    revision INTEGER NOT NULL CHECK (revision > 0),
    portfolio_id TEXT,
    entity_id TEXT NOT NULL,
    linked_claim_id TEXT,
    event_name TEXT NOT NULL,
    expected_date TEXT,
    observed_date TEXT,
    source_id TEXT,
    source_revision TEXT,
    prior_revision INTEGER,
    payload_json TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    recorded_at_utc TEXT NOT NULL,
    PRIMARY KEY (catalyst_id, revision)
);
CREATE INDEX IF NOT EXISTS idx_catalyst_entity_revision
    ON catalyst_revisions(portfolio_id, entity_id, revision);
CREATE TABLE IF NOT EXISTS source_captures (
    source_id TEXT NOT NULL,
    revision TEXT NOT NULL,
    issuer_id TEXT,
    ticker TEXT,
    document_type TEXT NOT NULL,
    fiscal_period TEXT,
    source_locator TEXT NOT NULL,
    artifact_ref TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    captured_at_utc TEXT NOT NULL,
    available_at_utc TEXT,
    metadata_json TEXT NOT NULL,
    PRIMARY KEY (source_id, revision)
);
CREATE INDEX IF NOT EXISTS idx_source_ticker_period_available
    ON source_captures(ticker, fiscal_period, available_at_utc);
CREATE TABLE IF NOT EXISTS catalyst_assessments (
    assessment_id TEXT PRIMARY KEY,
    idempotency_key TEXT NOT NULL UNIQUE,
    portfolio_id TEXT,
    catalyst_id TEXT NOT NULL,
    exception_fingerprint TEXT NOT NULL,
    source_id TEXT,
    source_revision TEXT,
    reviewer_subject_id TEXT NOT NULL,
    reviewer_session_id TEXT,
    assessment_text TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    recorded_at_utc TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_catalyst_assessment_scope
    ON catalyst_assessments(portfolio_id, catalyst_id, exception_fingerprint);
CREATE TABLE IF NOT EXISTS estimate_records (
    estimate_id TEXT PRIMARY KEY,
    prior_record_id TEXT,
    ticker TEXT NOT NULL,
    fiscal_period TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    estimate_type TEXT NOT NULL,
    value_decimal_text TEXT NOT NULL,
    unit TEXT NOT NULL,
    source_id TEXT,
    source_revision TEXT,
    source_as_of_utc TEXT,
    available_at_utc TEXT NOT NULL,
    approved_by_subject_id TEXT,
    payload_json TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    recorded_at_utc TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_estimate_ticker_period_metric_available
    ON estimate_records(ticker, fiscal_period, metric_name, available_at_utc);
CREATE TABLE IF NOT EXISTS state_import_receipts (
    receipt_id TEXT PRIMARY KEY,
    source_digest TEXT NOT NULL,
    record_family TEXT NOT NULL,
    legacy_record_id TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    target_record_id TEXT NOT NULL,
    import_status TEXT NOT NULL,
    imported_at_utc TEXT NOT NULL,
    UNIQUE (source_digest, record_family, legacy_record_id)
);
"""
_ROUND5_IMMUTABLE_TABLES = (
    "pm_review_packets",
    "routine_execution_events",
    "catalyst_revisions",
    "source_captures",
    "catalyst_assessments",
    "estimate_records",
    "state_import_receipts",
)
_PM_SIGNOFF_NEW_COLUMNS = {
    "portfolio_id": "portfolio_id TEXT NOT NULL DEFAULT 'default'",
    "verified_sample_ids_json": "verified_sample_ids_json TEXT",
    "packet_id": "packet_id TEXT",
    "acknowledged_exception_ids_json": "acknowledged_exception_ids_json TEXT",
    "request_digest": "request_digest TEXT",
    "principal_subject_id": "principal_subject_id TEXT",
    "principal_session_id": "principal_session_id TEXT",
}


def _apply_round5_storage_migration(conn: sqlite3.Connection) -> None:
    """Apply the approved append-only institutional storage migration atomically."""
    if conn.in_transaction:
        raise RuntimeError("round-5 storage migration requires an idle connection")
    checksum = hashlib.sha256(_ROUND5_DDL.encode("utf-8")).hexdigest()
    try:
        conn.executescript("BEGIN IMMEDIATE;\n" + _ROUND5_DDL)
        existing = {
            row["name"] if hasattr(row, "keys") else row[1]
            for row in conn.execute("PRAGMA table_info(pm_review_signoffs)")
        }
        for name, definition in _PM_SIGNOFF_NEW_COLUMNS.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE pm_review_signoffs ADD COLUMN {definition}")
        conn.execute("DROP INDEX IF EXISTS idx_pm_signoff_date")
        conn.execute("DROP INDEX IF EXISTS idx_pm_signoffs_review_date")
        conn.execute("DROP INDEX IF EXISTS idx_pm_signoff_port_date")
        duplicates = conn.execute(
            """
            SELECT portfolio_id, packet_sha256, COUNT(*) AS duplicate_count
            FROM pm_review_signoffs
            GROUP BY portfolio_id, packet_sha256
            HAVING COUNT(*) > 1
            """
        ).fetchall()
        if duplicates:
            raise RuntimeError(
                "Cannot migrate PM sign-offs: duplicate (portfolio_id, packet_sha256) rows require manual review."
            )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_pm_signoff_port_packet "
            "ON pm_review_signoffs(portfolio_id, packet_sha256)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_pm_signoff_port_date "
            "ON pm_review_signoffs(portfolio_id, review_date)"
        )
        conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS trg_pm_signoff_packet_scope
            BEFORE INSERT ON pm_review_signoffs
            WHEN NEW.packet_id IS NOT NULL AND NOT EXISTS (
                SELECT 1 FROM pm_review_packets p
                WHERE p.packet_id = NEW.packet_id
                  AND p.portfolio_id = NEW.portfolio_id
                  AND p.packet_sha256 = NEW.packet_sha256
                  AND p.review_date = NEW.review_date
            )
            BEGIN SELECT RAISE(ABORT, 'PM sign-off packet scope mismatch'); END
            """
        )
        for table in _ROUND5_IMMUTABLE_TABLES:
            conn.execute(
                f"""
                CREATE TRIGGER IF NOT EXISTS trg_{table}_immutable_update
                BEFORE UPDATE ON {table}
                BEGIN SELECT RAISE(ABORT, '{table} is append-only'); END
                """
            )
            conn.execute(
                f"""
                CREATE TRIGGER IF NOT EXISTS trg_{table}_immutable_delete
                BEFORE DELETE ON {table}
                BEGIN SELECT RAISE(ABORT, '{table} is append-only'); END
                """
            )
        conn.execute(
            "INSERT OR IGNORE INTO schema_migrations(migration_id, checksum_sha256, applied_at_utc) "
            "VALUES (?, ?, strftime('%Y-%m-%dT%H:%M:%fZ','now'))",
            (_ROUND5_MIGRATION_ID, checksum),
        )
        stored = conn.execute(
            "SELECT checksum_sha256 FROM schema_migrations WHERE migration_id = ?",
            (_ROUND5_MIGRATION_ID,),
        ).fetchone()
        stored_checksum = stored["checksum_sha256"] if hasattr(stored, "keys") else stored[0]
        if stored_checksum != checksum:
            raise RuntimeError("round-5 migration checksum differs from the recorded migration.")
        conn.execute("PRAGMA user_version = 6")
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def migrate_round5_storage(conn: sqlite3.Connection) -> None:
    """Explicit migration entry point; does not run unrelated schema initialization."""
    _apply_round5_storage_migration(conn)


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    conn.commit()
    _migrate_columns(conn)
    _migrate_pm_review_signoffs(conn)
    _migrate_dispatcher_column_cards(conn)
    _backfill_kanban_display_seq(conn)
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_kanban_cards_source_key "
        "ON kanban_cards(source_key) WHERE source_key IS NOT NULL"
    )
    conn.commit()
    _apply_round5_storage_migration(conn)


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
