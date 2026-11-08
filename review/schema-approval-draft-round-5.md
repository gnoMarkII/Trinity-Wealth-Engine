# Round 5 schema and storage approval draft

**Status:** Draft for explicit approval. No DDL, migration, event-layout, or database writes have been applied by this review.  
**Prepared:** 2026-11-07  
**Scope:** K01, K03, K04, K06, K07 and the storage-dependent parts of K02.

## 1. Read-only inventory

The configured working-tree database data/webui_state.sqlite was opened with SQLite read-only mode.

- PRAGMA quick_check: ok
- PRAGMA user_version: 0
- Row counts: jobs=357, kanban_cards=284, thesis_claims=4, thesis_exceptions=18, pm_review_signoffs=0, portfolio_state_mirror=1
- The live pm_review_signoffs table has only signoff_id, review_date, signed_off_by, signed_off_at_utc, acknowledged_exceptions_count, notes, and packet_sha256. It does not have portfolio scope, verified sample IDs, acknowledged exception IDs, or a packet reference. It has only its primary-key index.
- The live thesis tables have the required idx_thesis_claims_ticker_status index. The PM packet uniqueness index is absent.
- Current cash flow events write action/currency to journal metadata and the amount only to message text. Performance_Log.csv has eight columns and no structured external-flow amount, FX, flow coverage, snapshot identity, or revision.
- No live row was printed or changed.

The current dirty checkout also contains a pm_review_signoffs migration called by init_schema. This review did not call init_schema, open a writable connection, or invoke the application. Do not start a process that initializes this database until the migration scope below is approved or that pending migration is otherwise explicitly handled.

## 2. DB-A: immutable PM packet and approval records

### Proposed packet table

    CREATE TABLE pm_review_packets (
        packet_id TEXT PRIMARY KEY,
        portfolio_id TEXT NOT NULL,
        review_date TEXT NOT NULL,
        revision INTEGER NOT NULL CHECK (revision > 0),
        cutoff_utc TEXT NOT NULL,
        policy_version TEXT NOT NULL,
        input_fingerprint_sha256 TEXT NOT NULL,
        packet_sha256 TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        source_refs_json TEXT NOT NULL DEFAULT '[]',
        created_at_utc TEXT NOT NULL,
        UNIQUE (portfolio_id, packet_sha256),
        UNIQUE (portfolio_id, review_date, revision),
        UNIQUE (portfolio_id, review_date, input_fingerprint_sha256, policy_version)
    );
    CREATE INDEX idx_pm_packet_date
        ON pm_review_packets(portfolio_id, review_date, revision DESC);

The stored canonical payload freezes portfolio/date/revision/cutoff, source and rule revisions, policy version, required sample IDs, critical exception IDs, quality/block reasons, and canonical decimal strings. A packet row is append-only. The service reuses the same revision only when its input fingerprint and policy version match; any material input change creates a new revision.

### Proposed additive sign-off columns

    ALTER TABLE pm_review_signoffs
        ADD COLUMN portfolio_id TEXT NOT NULL DEFAULT 'default';
    ALTER TABLE pm_review_signoffs
        ADD COLUMN packet_id TEXT;
    ALTER TABLE pm_review_signoffs
        ADD COLUMN verified_sample_ids_json TEXT;
    ALTER TABLE pm_review_signoffs
        ADD COLUMN acknowledged_exception_ids_json TEXT;
    ALTER TABLE pm_review_signoffs
        ADD COLUMN request_digest TEXT;
    ALTER TABLE pm_review_signoffs
        ADD COLUMN principal_subject_id TEXT;
    ALTER TABLE pm_review_signoffs
        ADD COLUMN principal_session_id TEXT;

    CREATE UNIQUE INDEX idx_pm_signoff_port_packet
        ON pm_review_signoffs(portfolio_id, packet_sha256);
    CREATE INDEX idx_pm_signoff_port_date
        ON pm_review_signoffs(portfolio_id, review_date);

packet_id is nullable to preserve legacy rows; new writes must reference an existing packet in the same portfolio and match its hash. Enforce that scope in the repository transaction and enable/check SQLite foreign keys before adding a foreign-key constraint. The service will normalize submitted sample and exception IDs, validate them against the exact frozen packet, reject missing required samples and critical acknowledgements, and persist the normalized IDs and request digest. acknowledged_exceptions_count for new rows is the count of persisted distinct acknowledged IDs, not the number of exceptions in the packet.

Legacy values for packet, sample IDs, acknowledged IDs, digest, and principal remain NULL (unknown). Do not infer them from notes or existing counts. Keep existing signer, timestamp, notes, and checksum values unchanged.

### Required preflight and migration behavior

1. Back up the target database and record its size/hash, schema inventory, row counts, indexes, and quick_check.
2. Run the duplicate preflight below against the actual target immediately before creating the unique index. Abort on any row returned; never delete or merge approvals automatically.
3. Rehearse from a byte-for-byte copy of the target and from a legacy fixture, then verify original rows and hashes.
4. Apply in an explicit transaction through the migration command, not from readiness checks. Record migration ID/checksum and result.
5. Read the packet and approval back through production repositories in a fresh process; verify exact IDs, actor, scope, hash, and revision.
6. Roll back by restoring the verified pre-migration backup if any post-migration invariant fails. Do not downgrade by deleting approval history.

    SELECT portfolio_id, packet_sha256, COUNT(*) AS duplicate_count
    FROM pm_review_signoffs
    GROUP BY portfolio_id, packet_sha256
    HAVING COUNT(*) > 1;

The new SQLite schema version/migration ledger must be recorded only after all DDL and indexes pass. Readiness remains read-only and must report missing schema as not ready.

## 3. DB-B: durable events, source captures, estimates, and imports

No colliding table names were found in the live database for the record families below. Re-run that inventory against every deployment before migration.

    CREATE TABLE routine_execution_events (
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

    CREATE TABLE catalyst_revisions (
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

    CREATE TABLE source_captures (
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

    CREATE TABLE catalyst_assessments (
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
        payload_sha256 TEXT NOT NULL,
        recorded_at_utc TEXT NOT NULL
    );

    CREATE TABLE estimate_records (
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
        payload_sha256 TEXT NOT NULL,
        recorded_at_utc TEXT NOT NULL
    );

    CREATE TABLE state_import_receipts (
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

Add indexes only after query plans are confirmed: routine (execution_id, event_sequence), catalyst (portfolio_id, entity_id, revision), sources (ticker, fiscal_period, available_at_utc), assessments (portfolio_id, catalyst_id, exception_fingerprint), and estimates (ticker, fiscal_period, metric_name, available_at_utc). Unique identity and idempotency constraints are part of the contract.

All JSON state files are imported by an explicit dry-run/apply command. It hashes and validates the source, preserves actor/time, compares duplicate IDs by payload hash, aborts conflicting payloads, writes import receipts, and leaves the original files untouched. Large source bodies live as content-addressed immutable artifacts; the database row is committed only after the artifact hash and file are verified. Source locators alone never establish evidence.

Repository writes use the existing UnitOfWork boundary. DAOs do not commit. A failed commit propagates as failure; success returns the canonical committed row. Restart and two-instance interleaving tests must verify read-back and preserve both valid revisions.

## 4. Portfolio event and performance storage contract

### Cash flow event v2

Persist every deposit and withdrawal as a structured event with:

- event_id, portfolio_id, action, signed amount_native_decimal, currency
- fx_rate_to_thb_decimal, fx_source_id, and signed amount_thb_decimal; FX fields are required for USD-to-THB flow conversion and unavailable otherwise
- occurred_at_utc, recorded_at_utc, source/state sequence or hash, and idempotency key
- flow_kind and explicit flow_coverage_status

Deposits are positive external flows and withdrawals negative external flows. Dividends, interest, and other investment income are not external contributions. Legacy note-only events are MISSING_FLOW_HISTORY; never backfill zero or derive an amount from prose. Backfill only from a reconciled source, preserving original provenance.

### Performance snapshot v2

Append versioned fields to the CSV contract:

Snapshot_ID, Revision, Cutoff_UTC, Recorded_At_UTC, Source_State_Sequence, Source_State_SHA256, Total_NAV_THB, Net_External_Flow_THB, Flow_Coverage_Status, Cash_Flow_Event_IDs_JSON

Use Decimal strings for stored values. Preserve legacy columns for existing readers during transition. Keep all revisions; do not overwrite a same-date snapshot with changed content. Legacy rows receive an explicit MISSING_FLOW_HISTORY status, not a zero flow. The reader selects snapshots available at or before the requested cutoff, rejects future observations, detects conflicting revisions, and returns unavailable P&L unless both NAV snapshots and complete flow coverage are present.

The final DTO must distinguish 0 from unavailable (null plus quality/reason). Regenerate contract manifests only after the separate approval required by AGENTS.md.

## 5. Approval requested

Please approve or reject these independently scoped storage changes:

1. DB-A: immutable PM packet table and additive sign-off columns/indexes.
2. DB-B: durable event/source/estimate/import tables, explicit legacy import, and content-addressed source references.
3. Portfolio storage v2: structured cash-flow events and versioned performance CSV/readers.

Approval to work on the code already performed does not authorize a live database migration. No dependencies or golden manifests are included in this request.

## 6. Interim implementation and verification status

The current K01 service/API/UI now require IDs for every pending packet exception. This is deliberately stricter than the final critical-only policy until severity is represented by a reviewed packet field. IDs are validated and normalized in packet order. Existing sign-off evidence is read only from repository fields; notes are not treated as proof. If the configured repository rejects or cannot read back acknowledged IDs, the API returns 503 and does not create a sign-off. The current SQLite adapter does not yet persist that field, so sign-off of a packet with pending exceptions remains unavailable until DB-A is approved and implemented.

Targeted verification on the current working tree:

- Backend: 63 passed, 2 warnings across the round-5 K01-K06 and API suites.
- Frontend: full Vitest suite, 624 passed across 123 files.
- TypeScript typecheck: passed.
- Isolated K01 SQLite concurrency test: passed when rerun alone.

Required full backend command, .venv\Scripts\pytest tests/ -v:

- 2,534 passed, 21 failed, 18 skipped, 9 errors in 997.36 seconds.
- Failures include CLI/news-funnel, sector-rotation live providers, archivist writer/layout, market snapshot/quant, and replay-manifest tests.
- The K01 concurrent sign-off test appeared as an error in the full run but passed alone. The eight archivist fixture errors consistently report that legacy V1 Markdown writes are disabled in the V2-only writer.
- These failures/errors are recorded as full-suite blockers; the passing targeted result does not close the full gate.

After the full suite, the live database was rechecked read-only: quick_check=ok, user_version=0, pm_review_signoffs still has 7 legacy columns, 0 rows, and only the primary-key index. Its recorded file timestamp was unchanged. No database migration was applied by this work.

K01, K02, K03, K04, K06, and K07 remain IN_PROGRESS or WAITING_APPROVAL because immutable packet/source state, durable acknowledgements, cash-flow event history, or production repositories are not yet available. K05, K08, and K09 have targeted regression evidence but are not marked CLOSED until the full required gates and remaining plan criteria pass.
