"""SQLite-backed Note Catalog Adapter.

Provides fast O(1) indexed lookups by note_id, document_key, relative_path,
ticker, and entity_type without scanning the filesystem.
"""
from __future__ import annotations

import json
import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Union

from application.knowledge.ports import NoteCatalogEntry, NoteCatalogPort
from core.logger import get_logger
from tools.archivist.metadata import (
    normalize_legacy_metadata,
    parse_note,
    validate_note,
)
from tools.archivist.vault_paths import VaultPaths
from tools.archivist.vault_policy import is_searchable_note
from tools.archivist.portable_links import (
    iter_internal_markdown_links,
    resolve_vault_target_detailed,
)
from tools.archivist.schema_registry import SchemaRegistry, load_default_registry
from tools.archivist.catalog_runtime import (
    CatalogRuntimeError,
    catalog_outbox_path,
    load_catalog_pointer,
    resolve_catalog_path,
)

logger = get_logger(__name__)


class _ClosingConnection(sqlite3.Connection):
    """Connection whose context-manager exit also releases OS file handles.

    ``sqlite3.Connection.__exit__`` commits or rolls back but deliberately
    leaves the connection open.  On Windows that keeps the catalog file locked
    across a clone/rehearsal cleanup, so every adapter operation must close its
    short-lived connection after the transaction boundary.
    """

    def __exit__(self, exc_type, exc, traceback):  # type: ignore[override]
        try:
            return super().__exit__(exc_type, exc, traceback)
        finally:
            self.close()


class SqliteNoteCatalogAdapter(NoteCatalogPort):
    """SQLite implementation of the NoteCatalogPort."""

    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = None,
        vault_root: Optional[Union[str, Path]] = None,
        read_only: bool = False,
    ) -> None:
        vp = VaultPaths(vault_root)
        self._vault_root = vp.root
        if db_path is None:
            self._db_path = resolve_catalog_path(self._vault_root)
        else:
            self._db_path = Path(db_path).resolve()
        self._read_only = read_only
        if not self._read_only and load_catalog_pointer(self._vault_root) is not None:
            active_path = resolve_catalog_path(self._vault_root, require_exists=False)
            if self._db_path == active_path:
                raise CatalogRuntimeError(
                    "published catalog generations are immutable; build a staging generation instead"
                )
        self._immutable_read_only = bool(
            self._read_only
            and not self._db_path.is_relative_to(self._vault_root)
            and load_catalog_pointer(self._vault_root) is not None
        )
        self._retired_note_ids_cache: Optional[set[str]] = None
        self._schema_registry: SchemaRegistry = load_default_registry()
        if self._read_only:
            if not self._db_path.is_file():
                raise FileNotFoundError(f"Read-only catalog does not exist: {self._db_path}")
        else:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if self._read_only:
            # URI mode=ro prevents accidental journal/WAL/schema mutation from
            # a query constructor or a read-only request path.  A published
            # external generation is immutable, so SQLite must not even probe
            # for sidecar WAL/SHM files beside it.
            flags = "mode=ro&immutable=1" if self._immutable_read_only else "mode=ro"
            uri = f"file:{self._db_path.as_posix()}?{flags}"
            conn = sqlite3.connect(uri, uri=True, timeout=30.0, factory=_ClosingConnection)
        else:
            conn = sqlite3.connect(str(self._db_path), timeout=30.0, factory=_ClosingConnection)
        conn.row_factory = sqlite3.Row
        if not self._read_only:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        return conn

    def _init_db(self) -> None:
        with self._get_conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS note_catalog (
                    note_id TEXT PRIMARY KEY,
                    document_key TEXT UNIQUE NOT NULL,
                    relative_path TEXT UNIQUE NOT NULL,
                    entity_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    date TEXT,
                    ticker TEXT,
                    source_key TEXT,
                    mtime REAL NOT NULL,
                    file_size INTEGER NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    current_revision_id TEXT,
                    current_revision INTEGER,
                    manifest_digest TEXT,
                    artifact_set_hash TEXT,
                    body_sha256 TEXT,
                    record_state TEXT NOT NULL DEFAULT 'active',
                    storage_scope TEXT NOT NULL DEFAULT 'active',
                    updated_at TEXT NOT NULL
                );
                """
            )
            # Existing deployments predate the revision/read-model columns.
            # Add them in place so rebuilding the derived catalog never drops
            # durable identities or custom metadata.
            existing_columns = {row["name"] for row in conn.execute("PRAGMA table_info(note_catalog)")}
            migrations = {
                "current_revision_id": "TEXT",
                "current_revision": "INTEGER",
                "manifest_digest": "TEXT",
                "artifact_set_hash": "TEXT",
                "body_sha256": "TEXT",
                "record_state": "TEXT NOT NULL DEFAULT 'active'",
                "storage_scope": "TEXT NOT NULL DEFAULT 'active'",
            }
            for column, definition in migrations.items():
                if column not in existing_columns:
                    conn.execute(f"ALTER TABLE note_catalog ADD COLUMN {column} {definition}")

            # Create indexes after the inline column migration.  Older live
            # catalogs do not have the revision/state columns yet, and SQLite
            # rejects an index over a column that has not been added.
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cat_doc_key ON note_catalog(document_key);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cat_rel_path ON note_catalog(relative_path);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cat_entity ON note_catalog(entity_type);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cat_ticker ON note_catalog(ticker);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cat_date ON note_catalog(date);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cat_revision ON note_catalog(current_revision_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cat_state_scope ON note_catalog(record_state, storage_scope);")

            # Sidecar catalog table for sub-millisecond JSON query resolution
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sidecar_catalog (
                    sidecar_id TEXT PRIMARY KEY,
                    ticker TEXT NOT NULL,
                    evaluation_date TEXT NOT NULL,
                    relative_path TEXT UNIQUE NOT NULL,
                    storage_tier TEXT NOT NULL, -- 'user' or 'system'
                    mtime REAL NOT NULL,
                    file_size INTEGER NOT NULL,
                    sha256 TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sidecar_ticker ON sidecar_catalog(ticker);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sidecar_date ON sidecar_catalog(evaluation_date);")

            # Derived, path-portable graph edges.  The table is deliberately
            # separate from note identity so a path move can be projected
            # without changing note_id/document_key.
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS note_links (
                    source_note_id TEXT NOT NULL,
                    source_relative_path TEXT NOT NULL,
                    target_note_id TEXT NOT NULL,
                    target_relative_path TEXT NOT NULL,
                    raw_target TEXT NOT NULL,
                    link_type TEXT NOT NULL DEFAULT 'markdown',
                    fragment TEXT,
                    source_content_sha256 TEXT NOT NULL,
                    target_content_sha256 TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (source_note_id, target_note_id, raw_target, link_type)
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_links_source ON note_links(source_note_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_links_target ON note_links(target_note_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_links_source_path ON note_links(source_relative_path);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_links_target_path ON note_links(target_relative_path);")

    def _retired_note_ids(self) -> set[str]:
        """Return durable retired identities so a deleted projection cannot reappear."""
        if self._retired_note_ids_cache is not None:
            return self._retired_note_ids_cache
        path = self._vault_root / ".system" / "retired_notes.jsonl"
        retired: set[str] = set()
        if path.is_file():
            try:
                for line in path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    if isinstance(value, dict) and value.get("note_id"):
                        note_id = str(value["note_id"])
                        status = str(value.get("status") or "retired").lower()
                        if status == "restored":
                            retired.discard(note_id)
                        else:
                            retired.add(note_id)
            except (OSError, UnicodeDecodeError, ValueError):
                logger.warning("Unable to read retired note tombstones from %s", path)
        self._retired_note_ids_cache = retired
        return retired

    def _row_to_entry(self, row: sqlite3.Row) -> NoteCatalogEntry:
        return NoteCatalogEntry(
            note_id=row["note_id"],
            document_key=row["document_key"],
            relative_path=row["relative_path"],
            entity_type=row["entity_type"],
            title=row["title"],
            date=row["date"],
            ticker=row["ticker"],
            source_key=row["source_key"],
            mtime=row["mtime"],
            file_size=row["file_size"],
            content_sha256=row["content_sha256"],
            metadata_json=row["metadata_json"],
            updated_at=row["updated_at"],
            current_revision_id=row["current_revision_id"] if "current_revision_id" in row.keys() else None,
            current_revision=row["current_revision"] if "current_revision" in row.keys() else None,
            manifest_digest=row["manifest_digest"] if "manifest_digest" in row.keys() else None,
            artifact_set_hash=row["artifact_set_hash"] if "artifact_set_hash" in row.keys() else None,
            body_sha256=row["body_sha256"] if "body_sha256" in row.keys() else None,
            record_state=row["record_state"] if "record_state" in row.keys() else "active",
            storage_scope=row["storage_scope"] if "storage_scope" in row.keys() else "active",
        )

    def _artifact_fields(self, note_id: str) -> dict[str, Any]:
        """Read the durable current-head projection without allocating identity.

        The catalog is a derived read model.  It may enrich a row from an
        already committed artifact head, but it must never create a head,
        revision, or identity while rebuilding itself.
        """
        if not note_id:
            return {}
        head_path = self._vault_root / ".system" / "artifacts" / "heads" / f"{note_id}.json"
        try:
            head = json.loads(head_path.read_text(encoding="utf-8"))
            if not isinstance(head, dict):
                return {}
            revision_id = head.get("revision_id")
            if not isinstance(revision_id, str) or not revision_id:
                return {}
            manifest_path = VaultPaths(self._vault_root).revision_path(
                note_id, revision_id, "manifest.json"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                return {}
            if manifest.get("note_id") != note_id or manifest.get("revision_id") != revision_id:
                return {}
            primary_name = manifest.get("primary_file")
            primary_path = manifest_path.parent / str(primary_name) if primary_name else None
            body_sha256 = None
            if primary_path and primary_path.is_file():
                try:
                    _, body, issues = parse_note(primary_path.read_text(encoding="utf-8"))
                    if not issues:
                        body_sha256 = hashlib.sha256(body.encode("utf-8")).hexdigest()
                except (OSError, UnicodeDecodeError, ValueError):
                    body_sha256 = None
            manifest_digest = hashlib.sha256(
                json.dumps(
                    manifest,
                    sort_keys=True,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            return {
                "current_revision_id": revision_id,
                "current_revision": int(manifest.get("revision", head.get("revision", 0)))
                if str(manifest.get("revision", head.get("revision", 0))).isdigit()
                else None,
                "manifest_digest": head.get("manifest_digest") or manifest_digest,
                "artifact_set_hash": manifest.get("artifact_set_hash") or head.get("artifact_set_hash"),
                "body_sha256": body_sha256,
                "record_state": "active",
                "storage_scope": "active",
            }
        except (OSError, UnicodeDecodeError, ValueError, TypeError):
            return {}

    def upsert_notes(self, entries: list[NoteCatalogEntry]) -> None:
        if not entries:
            return
        retired = self._retired_note_ids()
        entries = [entry for entry in entries if str(entry.note_id) not in retired]
        if not entries:
            return
        now = datetime.now(timezone.utc).isoformat()
        params = [
            (
                entry.note_id,
                entry.document_key,
                entry.relative_path,
                entry.entity_type,
                entry.title,
                entry.date,
                entry.ticker,
                entry.source_key,
                entry.mtime,
                entry.file_size,
                entry.content_sha256,
                entry.metadata_json,
                entry.current_revision_id,
                entry.current_revision,
                entry.manifest_digest,
                entry.artifact_set_hash,
                entry.body_sha256,
                entry.record_state,
                entry.storage_scope,
                now,
            )
            for entry in entries
        ]
        with self._get_conn() as conn:
            conn.executemany(
                """
                INSERT INTO note_catalog (
                    note_id, document_key, relative_path, entity_type,
                    title, date, ticker, source_key, mtime, file_size,
                    content_sha256, metadata_json, current_revision_id,
                    current_revision, manifest_digest, artifact_set_hash,
                    body_sha256, record_state, storage_scope, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(note_id) DO UPDATE SET
                    document_key=excluded.document_key,
                    relative_path=excluded.relative_path,
                    entity_type=excluded.entity_type,
                    title=excluded.title,
                    date=excluded.date,
                    ticker=excluded.ticker,
                    source_key=excluded.source_key,
                    mtime=excluded.mtime,
                    file_size=excluded.file_size,
                    content_sha256=excluded.content_sha256,
                    metadata_json=excluded.metadata_json,
                    current_revision_id=excluded.current_revision_id,
                    current_revision=excluded.current_revision,
                    manifest_digest=excluded.manifest_digest,
                    artifact_set_hash=excluded.artifact_set_hash,
                    body_sha256=excluded.body_sha256,
                    record_state=excluded.record_state,
                    storage_scope=excluded.storage_scope,
                    updated_at=excluded.updated_at;
                """,
                params,
            )

    def upsert_note(self, entry: NoteCatalogEntry) -> None:
        self.upsert_notes([entry])

    def delete_notes(self, note_ids: list[str]) -> None:
        if not note_ids:
            return
        with self._get_conn() as conn:
            conn.executemany("DELETE FROM note_catalog WHERE note_id = ?;", [(nid,) for nid in note_ids])
            conn.executemany(
                "DELETE FROM note_links WHERE source_note_id = ? OR target_note_id = ?;",
                [(nid, nid) for nid in note_ids],
            )

    def delete_note(self, note_id: str) -> None:
        self.delete_notes([note_id])

    def replace_link_edges(self, edges: Iterable[Mapping[str, Any]]) -> int:
        """Replace the derived portable-link projection for this catalog."""
        if self._read_only:
            raise CatalogRuntimeError("read-only catalog cannot replace link edges")
        now = datetime.now(timezone.utc).isoformat()
        rows: list[tuple[Any, ...]] = []
        for edge in edges:
            required = (
                "source_note_id", "source_relative_path", "target_note_id",
                "target_relative_path", "raw_target", "source_content_sha256",
                "target_content_sha256",
            )
            if any(not str(edge.get(field) or "").strip() for field in required):
                raise ValueError(f"link edge missing required fields: {edge}")
            rows.append(
                (
                    str(edge["source_note_id"]),
                    str(edge["source_relative_path"]).replace("\\", "/"),
                    str(edge["target_note_id"]),
                    str(edge["target_relative_path"]).replace("\\", "/"),
                    str(edge["raw_target"]),
                    str(edge.get("link_type") or "markdown"),
                    str(edge.get("fragment") or "") or None,
                    str(edge["source_content_sha256"]),
                    str(edge["target_content_sha256"]),
                    now,
                )
            )
        with self._get_conn() as conn:
            conn.execute("DELETE FROM note_links;")
            if rows:
                conn.executemany(
                    """
                    INSERT INTO note_links (
                        source_note_id, source_relative_path, target_note_id,
                        target_relative_path, raw_target, link_type, fragment,
                        source_content_sha256, target_content_sha256, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(source_note_id, target_note_id, raw_target, link_type)
                    DO UPDATE SET
                        source_relative_path=excluded.source_relative_path,
                        target_relative_path=excluded.target_relative_path,
                        fragment=excluded.fragment,
                        source_content_sha256=excluded.source_content_sha256,
                        target_content_sha256=excluded.target_content_sha256,
                        updated_at=excluded.updated_at;
                    """,
                    rows,
                )
        return len(rows)

    def iter_link_edges(
        self,
        *,
        note_id: Optional[str] = None,
        relative_path: Optional[str] = None,
        direction: str = "outgoing",
    ) -> list[dict[str, Any]]:
        """Read portable graph edges in outgoing or incoming direction."""
        direction = str(direction or "outgoing").lower()
        if direction not in {"outgoing", "incoming", "all"}:
            raise ValueError("direction must be outgoing, incoming, or all")
        clauses: list[str] = []
        params: list[Any] = []
        if direction == "outgoing":
            if note_id:
                clauses.append("source_note_id = ?")
                params.append(str(note_id))
            if relative_path:
                clauses.append("source_relative_path = ?")
                params.append(str(relative_path).replace("\\", "/").strip("/"))
        elif direction == "incoming":
            if note_id:
                clauses.append("target_note_id = ?")
                params.append(str(note_id))
            if relative_path:
                clauses.append("target_relative_path = ?")
                params.append(str(relative_path).replace("\\", "/").strip("/"))
        else:
            if note_id:
                clauses.append("(source_note_id = ? OR target_note_id = ?)")
                params.extend([str(note_id), str(note_id)])
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        try:
            with self._get_conn() as conn:
                rows = conn.execute(
                    "SELECT * FROM note_links" + where
                    + " ORDER BY source_relative_path, target_relative_path, raw_target;",
                    params,
                ).fetchall()
        except sqlite3.OperationalError as exc:
            # Old read-only generations may predate the optional graph table.
            if "no such table" in str(exc).lower():
                return []
            raise
        return [dict(row) for row in rows]

    def rebuild_link_edges(self, vault_root: Optional[Union[str, Path]] = None) -> int:
        """Derive current source-relative Markdown edges into ``note_links``."""
        if self._read_only:
            raise CatalogRuntimeError("read-only catalog cannot rebuild link edges")
        root = Path(vault_root).resolve() if vault_root else self._vault_root
        with self._get_conn() as conn:
            catalog_rows = conn.execute(
                "SELECT note_id, relative_path, content_sha256 FROM note_catalog "
                "WHERE record_state = 'active' AND storage_scope = 'active';"
            ).fetchall()
        by_path = {
            str(row["relative_path"]).replace("\\", "/"): row
            for row in catalog_rows
        }
        edges: list[dict[str, Any]] = []
        for source_rel, source_row in sorted(by_path.items()):
            source_path = root / source_rel
            if not source_path.is_file() or not is_searchable_note(source_path, vault_root=root):
                continue
            try:
                content = source_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            seen: set[tuple[str, str]] = set()
            for link in iter_internal_markdown_links(content):
                result = resolve_vault_target_detailed(root, link.destination, source=source_rel)
                if result.status != "resolved" or result.target is None:
                    continue
                target_rel = result.target.relative_to(root).as_posix()
                target_row = by_path.get(target_rel)
                if target_row is None:
                    continue
                key = (target_rel, link.destination)
                if key in seen:
                    continue
                seen.add(key)
                edges.append(
                    {
                        "source_note_id": source_row["note_id"],
                        "source_relative_path": source_rel,
                        "target_note_id": target_row["note_id"],
                        "target_relative_path": target_rel,
                        "raw_target": link.destination,
                        "link_type": "markdown",
                        "fragment": link.fragment,
                        "source_content_sha256": source_row["content_sha256"],
                        "target_content_sha256": target_row["content_sha256"],
                    }
                )
        return self.replace_link_edges(edges)

    def get_by_id(self, note_id: str) -> Optional[NoteCatalogEntry]:
        with self._get_conn() as conn:
            cur = conn.execute("SELECT * FROM note_catalog WHERE note_id = ?;", (note_id,))
            row = cur.fetchone()
            return self._row_to_entry(row) if row else None

    def get_by_path(self, relative_path: str) -> Optional[NoteCatalogEntry]:
        clean_rel = relative_path.replace("\\", "/").strip("/")
        with self._get_conn() as conn:
            cur = conn.execute("SELECT * FROM note_catalog WHERE relative_path = ?;", (clean_rel,))
            row = cur.fetchone()
            return self._row_to_entry(row) if row else None

    def get_by_document_key(self, document_key: str) -> Optional[NoteCatalogEntry]:
        with self._get_conn() as conn:
            cur = conn.execute("SELECT * FROM note_catalog WHERE document_key = ?;", (document_key,))
            row = cur.fetchone()
            return self._row_to_entry(row) if row else None

    def find_notes(
        self,
        *,
        entity_type: Optional[str] = None,
        ticker: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        include_non_searchable: bool = False,
    ) -> list[NoteCatalogEntry]:
        clauses = [] if include_non_searchable else [
            "record_state = 'active'",
            "storage_scope = 'active'",
            "COALESCE(json_extract(metadata_json, '$.search_scope'), 'main') IN ('main', 'included')",
            "COALESCE(json_extract(metadata_json, '$.lifecycle_status'), '') <> 'stub'",
        ]
        params: list[Any] = []

        if entity_type:
            clauses.append("entity_type = ?")
            params.append(entity_type.lower())
        if ticker:
            clauses.append("ticker = ?")
            params.append(ticker.upper())
        if date_from:
            clauses.append("date >= ?")
            params.append(date_from)
        if date_to:
            clauses.append("date <= ?")
            params.append(date_to)

        where_stmt = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"SELECT * FROM note_catalog {where_stmt} ORDER BY date DESC, note_id ASC LIMIT ? OFFSET ?;"
        params.extend([limit, offset])

        with self._get_conn() as conn:
            cur = conn.execute(query, params)
            return [self._row_to_entry(r) for r in cur.fetchall()]

    def count_notes(self, entity_type: Optional[str] = None) -> int:
        query = (
            "SELECT COUNT(*) FROM note_catalog WHERE record_state = 'active' "
            "AND storage_scope = 'active' "
            "AND COALESCE(json_extract(metadata_json, '$.search_scope'), 'main') IN ('main', 'included') "
            "AND COALESCE(json_extract(metadata_json, '$.lifecycle_status'), '') <> 'stub'"
        )
        params: list[Any] = []
        if entity_type:
            query += " AND entity_type = ?"
            params.append(entity_type.lower())
        with self._get_conn() as conn:
            cur = conn.execute(query, params)
            row = cur.fetchone()
            return row[0] if row else 0

    def iter_notes(
        self,
        *,
        entity_type: Optional[str] = None,
        ticker: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        page_size: int = 500,
        include_non_searchable: bool = False,
    ):
        """Yield catalog rows with a stable keyset cursor.

        Request and indexing paths use this iterator instead of asking SQLite
        for an arbitrarily large ``LIMIT`` and materializing the whole vault.
        Ordering matches :meth:`find_notes`: dated notes first (newest first),
        then undated notes, with ``note_id`` as the unique tie breaker.
        """
        if page_size <= 0:
            raise ValueError("page_size must be positive")

        filters: list[str] = [] if include_non_searchable else [
            "record_state = 'active'",
            "storage_scope = 'active'",
            "COALESCE(json_extract(metadata_json, '$.search_scope'), 'main') IN ('main', 'included')",
            "COALESCE(json_extract(metadata_json, '$.lifecycle_status'), '') <> 'stub'",
        ]
        filter_params: list[Any] = []
        if entity_type:
            filters.append("entity_type = ?")
            filter_params.append(entity_type.lower())
        if ticker:
            filters.append("ticker = ?")
            filter_params.append(ticker.upper())
        if date_from:
            filters.append("date >= ?")
            filter_params.append(date_from)
        if date_to:
            filters.append("date <= ?")
            filter_params.append(date_to)

        date_rank = "CASE WHEN date IS NULL THEN 1 ELSE 0 END"
        last_rank: int | None = None
        last_date: str | None = None
        last_note_id: str | None = None

        while True:
            clauses = list(filters)
            params = list(filter_params)
            if last_rank is not None and last_note_id is not None:
                clauses.append(
                    f"({date_rank} > ? OR ("
                    f"{date_rank} = ? AND ("
                    f"(? = 0 AND (date < ? OR (date = ? AND note_id > ?))) "
                    f"OR (? = 1 AND note_id > ?)"
                    ")))"
                )
                params.extend(
                    [
                        last_rank,
                        last_rank,
                        last_rank,
                        last_date,
                        last_date,
                        last_note_id,
                        last_rank,
                        last_note_id,
                    ]
                )
            where_stmt = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            query = (
                f"SELECT * FROM note_catalog {where_stmt} "
                f"ORDER BY {date_rank} ASC, date DESC, note_id ASC LIMIT ?;"
            )
            params.append(page_size)
            with self._get_conn() as conn:
                rows = conn.execute(query, params).fetchall()
            if not rows:
                return
            for row in rows:
                yield self._row_to_entry(row)
            last = rows[-1]
            last_rank = 1 if last["date"] is None else 0
            last_date = last["date"]
            last_note_id = last["note_id"]

    def get_modified_since(self, timestamp: float) -> list[NoteCatalogEntry]:
        """Returns notes modified strictly after given timestamp, ordered by mtime ascending."""
        query = "SELECT * FROM note_catalog WHERE mtime > ? ORDER BY mtime ASC;"
        with self._get_conn() as conn:
            cur = conn.execute(query, (timestamp,))
            return [self._row_to_entry(r) for r in cur.fetchall()]

    def upsert_note_from_file(self, file_path: Union[str, Path]) -> Optional[NoteCatalogEntry]:
        """Directly parses and upserts a single note into SQLite catalog."""
        fp = Path(file_path).resolve()
        if not fp.exists() or not fp.is_file():
            return None
        if not is_searchable_note(fp, vault_root=self._vault_root):
            return None
        try:
            rel = fp.relative_to(self._vault_root).as_posix()
        except ValueError:
            return None

        stat = fp.stat()
        content = fp.read_text(encoding="utf-8")
        raw_meta, body, _ = parse_note(content)
        norm_meta, _ = normalize_legacy_metadata(raw_meta)
        if not self._schema_registry.is_index_eligible(norm_meta, vector=False):
            existing = self.get_by_path(rel)
            if existing:
                self.delete_note(existing.note_id)
            return None
        model, _ = validate_note(norm_meta, mode="lenient")

        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        title = (getattr(model, "title", None) if model else None) or norm_meta.get("title") or fp.stem
        entity_type = (getattr(model, "entity_type", None) if model else None) or norm_meta.get("entity_type", "concept")
        incoming_note_id = (getattr(model, "note_id", None) if model else None) or norm_meta.get("note_id")
        incoming_doc_key = (getattr(model, "document_key", None) if model else None) or norm_meta.get("document_key")
        if incoming_note_id and str(incoming_note_id) in self._retired_note_ids():
            logger.warning("Refusing to reinsert retired note identity %s from %s", incoming_note_id, fp)
            return None
        existing = self.get_by_path(rel)
        identity_conflict = bool(
            existing
            and incoming_note_id
            and str(incoming_note_id) != str(existing.note_id)
        )
        # A catalog rebuild is not an identity allocator.  Preserve a proven
        # row identity; otherwise retain an explicitly supplied ID, and mark a
        # legacy record without either as an unresolved content surrogate
        # rather than deriving an authoritative ID from its path.
        if existing:
            note_id = existing.note_id
            doc_key = existing.document_key
        elif incoming_note_id:
            note_id = str(incoming_note_id)
            doc_key = str(incoming_doc_key or f"legacy-import:v1:{note_id}")
        else:
            note_id = f"legacy_unresolved_{content_hash[:24]}"
            doc_key = f"legacy-unresolved:v1:{content_hash}"
        date_val = (getattr(model, "date", None) if model else None) or norm_meta.get("date")
        ticker_val = (getattr(model, "ticker", None) if model else None) or norm_meta.get("ticker")
        source_key = (getattr(model, "source_key", None) if model else None) or norm_meta.get("source_key")
        meta_json = json.dumps(model.model_dump(mode="json") if model else norm_meta, default=str, ensure_ascii=False)
        artifact_fields = self._artifact_fields(note_id)
        if identity_conflict:
            artifact_fields["record_state"] = "identity_conflict"
            artifact_fields["storage_scope"] = "legacy"
        elif not existing and not incoming_note_id:
            artifact_fields["record_state"] = "unresolved"
            artifact_fields["storage_scope"] = "legacy"

        entry = NoteCatalogEntry(
            note_id=note_id,
            document_key=doc_key,
            relative_path=rel,
            entity_type=entity_type,
            title=title,
            date=str(date_val) if date_val else None,
            ticker=str(ticker_val) if ticker_val else None,
            source_key=source_key,
            mtime=stat.st_mtime,
            file_size=stat.st_size,
            content_sha256=content_hash,
            metadata_json=meta_json,
            **artifact_fields,
        )
        self.upsert_note(entry)
        return entry

    def upsert_sidecar(
        self,
        ticker: str,
        evaluation_date: str,
        relative_path: str,
        storage_tier: str,
        mtime: float,
        file_size: int,
        sha256: str,
    ) -> None:
        """Upserts a JSON sidecar file into the sidecar_catalog."""
        now = datetime.now(timezone.utc).isoformat()
        rel = relative_path.replace("\\", "/")
        # A ticker/date/tier tuple is not unique: a day may contain several
        # analysis or news companions. Bind the primary key to the exact path
        # so the sidecar read model cannot silently overwrite siblings.
        sidecar_id = "sc_" + hashlib.sha256(rel.encode("utf-8")).hexdigest()[:24]
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO sidecar_catalog (
                    sidecar_id, ticker, evaluation_date, relative_path,
                    storage_tier, mtime, file_size, sha256, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(sidecar_id) DO UPDATE SET
                    relative_path=excluded.relative_path,
                    mtime=excluded.mtime,
                    file_size=excluded.file_size,
                    sha256=excluded.sha256,
                    updated_at=excluded.updated_at;
                """,
                (sidecar_id, ticker.upper(), evaluation_date, rel, storage_tier, mtime, file_size, sha256, now),
            )

    def get_sidecars(self, ticker: Optional[str] = None) -> list[str]:
        """Returns relative paths of registered sidecars, ordered by evaluation_date DESC."""
        query = "SELECT relative_path FROM sidecar_catalog"
        params: list[Any] = []
        if ticker:
            query += " WHERE ticker = ?"
            params.append(ticker.upper())
        query += " ORDER BY evaluation_date DESC, storage_tier ASC;"
        with self._get_conn() as conn:
            cur = conn.execute(query, params)
            return [row["relative_path"] for row in cur.fetchall()]

    def delete_sidecar(self, relative_path: str) -> None:
        """Removes a sidecar entry by relative path."""
        rel = relative_path.replace("\\", "/")
        with self._get_conn() as conn:
            conn.execute("DELETE FROM sidecar_catalog WHERE relative_path = ?;", (rel,))

    def enqueue_outbox(self, relative_path: str, action: str = "upsert", error_detail: str = "") -> None:
        """Append an idempotent catalog operation to the durable outbox."""
        rel = relative_path.replace("\\", "/")
        outbox_file = catalog_outbox_path(self._vault_root, create=True)
        pending: list[dict[str, Any]] = []
        if outbox_file.is_file():
            try:
                pending = [
                    json.loads(line)
                    for line in outbox_file.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            except (OSError, UnicodeDecodeError, ValueError):
                pending = []
        target = self._vault_root / rel
        try:
            content = target.read_text(encoding="utf-8") if target.is_file() else ""
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest() if content else None
        except (OSError, UnicodeDecodeError):
            content_hash = None
        idempotency_key = f"catalog:{action}:{rel}:{content_hash or 'missing'}"
        event_id = "evt_" + hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()[:24]
        if any(str(item.get("idempotency_key")) == idempotency_key for item in pending if isinstance(item, dict)):
            return
        record = {
            "event_id": event_id,
            "idempotency_key": idempotency_key,
            "state": "pending",
            "attempts": 0,
            "relative_path": rel,
            "action": action,
            "error": error_detail,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(outbox_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def process_outbox(self) -> int:
        """Processes and retries pending outbox operations."""
        if load_catalog_pointer(self._vault_root) is not None and not self._read_only:
            active_path = resolve_catalog_path(self._vault_root, require_exists=False)
            if self._db_path == active_path:
                raise CatalogRuntimeError(
                    "cannot process outbox by mutating the active generation; rebuild and publish a new generation"
                )
        outbox_file = catalog_outbox_path(self._vault_root)
        if not outbox_file.exists():
            return 0
        try:
            lines = [line.strip() for line in outbox_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        except OSError:
            return 0
        if not lines:
            return 0

        unresolved = []
        resolved_count = 0
        for line in lines:
            try:
                data = json.loads(line)
                rel = data.get("relative_path")
                action = data.get("action", "upsert")
                target_fp = self._vault_root / rel
                if action == "upsert" and target_fp.exists():
                    self.upsert_note_from_file(target_fp)
                    resolved_count += 1
                elif action == "delete":
                    existing = self.get_by_path(rel)
                    if existing:
                        self.delete_note(existing.note_id)
                    resolved_count += 1
                elif not target_fp.exists():
                    existing = self.get_by_path(rel)
                    if existing:
                        self.delete_note(existing.note_id)
                    resolved_count += 1
            except Exception as e:
                logger.warning("Failed processing outbox record %s: %s", line, e)
                unresolved.append(line)

        if unresolved:
            outbox_file.write_text("\n".join(unresolved) + "\n", encoding="utf-8")
        else:
            outbox_file.unlink(missing_ok=True)

        return resolved_count

    def reconcile_missing(self) -> dict[str, int]:
        """Detects and prunes catalog entries where the markdown file was deleted externally."""
        with self._get_conn() as conn:
            cur = conn.execute("SELECT note_id, relative_path FROM note_catalog;")
            rows = cur.fetchall()

        missing_ids = []
        for r in rows:
            fp = self._vault_root / r["relative_path"]
            if not fp.exists():
                missing_ids.append(r["note_id"])

        if missing_ids:
            self.delete_notes(missing_ids)

        return {"scanned": len(rows), "pruned": len(missing_ids)}

    def sync_from_vault(
        self,
        vault_root: Optional[Union[str, Path]] = None,
        force: bool = False,
    ) -> dict[str, int]:
        """Performs incremental synchronization between vault markdown files and SQLite catalog."""
        v_root = Path(vault_root).resolve() if vault_root else self._vault_root
        if not v_root.exists():
            return {"scanned": 0, "added": 0, "updated": 0, "deleted": 0, "unchanged": 0}

        # 1. Fetch current catalog state.  Rebuilds keep these rows available so
        # a proven identity can survive a metadata repair, rename, or restart.
        # ``force`` means re-read every file; it does not authorize truncating
        # durable identities from the derived table.
        existing: dict[str, tuple[str, float, int]] = {}
        existing_rows: dict[str, sqlite3.Row] = {}
        with self._get_conn() as conn:
            cur = conn.execute("SELECT * FROM note_catalog;")
            for row in cur.fetchall():
                existing[row["relative_path"]] = (row["note_id"], row["mtime"], row["file_size"])
                existing_rows[row["relative_path"]] = row
        retired_note_ids = self._retired_note_ids()

        scanned = 0
        added = 0
        updated = 0
        unchanged = 0
        seen_paths: set[str] = set()
        seen_note_ids: set[str] = set()
        batch_entries: list[NoteCatalogEntry] = []
        BATCH_SIZE = 500

        for md_path in v_root.rglob("*.md"):
            try:
                rel = md_path.relative_to(v_root).as_posix()
            except ValueError:
                continue

            if not is_searchable_note(md_path, vault_root=v_root):
                continue

            scanned += 1
            seen_paths.add(rel)

            try:
                stat = md_path.stat()
                cur_mtime = stat.st_mtime
                cur_size = stat.st_size
            except OSError:
                continue

            # Check if unchanged
            if not force and rel in existing:
                note_id, prev_mtime, prev_size = existing[rel]
                if abs(cur_mtime - prev_mtime) < 1e-4 and cur_size == prev_size:
                    unchanged += 1
                    seen_note_ids.add(note_id)
                    continue

            # Parse note metadata
            try:
                content = md_path.read_text(encoding="utf-8")
                raw_meta, body, _ = parse_note(content)
                norm_meta, _ = normalize_legacy_metadata(raw_meta)
                if not self._schema_registry.is_index_eligible(norm_meta, vector=False):
                    continue
                model, _ = validate_note(norm_meta, mode="lenient")
            except Exception as e:
                logger.warning("Failed to parse %s for catalog: %s", rel, e)
                continue

            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            title = (getattr(model, "title", None) if model else None) or norm_meta.get("title") or md_path.stem
            entity_type = (getattr(model, "entity_type", None) if model else None) or norm_meta.get("entity_type", "concept")
            incoming_note_id = (getattr(model, "note_id", None) if model else None) or norm_meta.get("note_id")
            incoming_doc_key = (getattr(model, "document_key", None) if model else None) or norm_meta.get("document_key")
            if incoming_note_id and str(incoming_note_id) in retired_note_ids:
                logger.warning("Skipping retired note projection during catalog sync: %s", rel)
                continue
            prior_row = existing_rows.get(rel)
            identity_conflict = bool(
                prior_row
                and incoming_note_id
                and str(incoming_note_id) != str(prior_row["note_id"])
            )
            if prior_row:
                note_id = str(prior_row["note_id"])
                doc_key = str(prior_row["document_key"])
            elif incoming_note_id:
                note_id = str(incoming_note_id)
                doc_key = str(incoming_doc_key or f"legacy-import:v1:{note_id}")
            else:
                note_id = f"legacy_unresolved_{content_hash[:24]}"
                doc_key = f"legacy-unresolved:v1:{content_hash}"
            date_val = (getattr(model, "date", None) if model else None) or norm_meta.get("date")
            ticker_val = (getattr(model, "ticker", None) if model else None) or norm_meta.get("ticker")
            source_key = (getattr(model, "source_key", None) if model else None) or norm_meta.get("source_key")
            meta_json = json.dumps(model.model_dump(mode="json") if model else norm_meta, default=str, ensure_ascii=False)
            artifact_fields = self._artifact_fields(note_id)
            if not artifact_fields and prior_row:
                for field_name in (
                    "current_revision_id", "current_revision", "manifest_digest",
                    "artifact_set_hash", "body_sha256", "record_state", "storage_scope",
                ):
                    if field_name in prior_row.keys() and prior_row[field_name] is not None:
                        artifact_fields[field_name] = prior_row[field_name]
            if identity_conflict:
                artifact_fields["record_state"] = "identity_conflict"
                artifact_fields["storage_scope"] = "legacy"
            elif not prior_row and not incoming_note_id:
                artifact_fields["record_state"] = "unresolved"
                artifact_fields["storage_scope"] = "legacy"

            entry = NoteCatalogEntry(
                note_id=note_id,
                document_key=doc_key,
                relative_path=rel,
                entity_type=entity_type,
                title=title,
                date=str(date_val) if date_val else None,
                ticker=str(ticker_val) if ticker_val else None,
                source_key=source_key,
                mtime=cur_mtime,
                file_size=cur_size,
                content_sha256=content_hash,
                metadata_json=meta_json,
                **artifact_fields,
            )
            seen_note_ids.add(note_id)

            if rel in existing:
                updated += 1
            else:
                added += 1

            batch_entries.append(entry)
            if len(batch_entries) >= BATCH_SIZE:
                self.upsert_notes(batch_entries)
                batch_entries = []

        if batch_entries:
            self.upsert_notes(batch_entries)

        # 2. Prune deleted notes from catalog
        deleted_ids = [
            note_id
            for stored_rel, (note_id, _, _) in existing.items()
            if stored_rel not in seen_paths and note_id not in seen_note_ids
        ]
        if deleted_ids:
            self.delete_notes(deleted_ids)
        deleted = len(deleted_ids)

        # Keep the graph read model in the same maintenance pass as the note
        # catalog.  This projection is derived and can be rebuilt at any time;
        # it never changes durable note identity.
        try:
            self.rebuild_link_edges(v_root)
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            logger.warning("Unable to rebuild portable link projection: %s", exc)

        return {
            "scanned": scanned,
            "added": added,
            "updated": updated,
            "deleted": deleted,
            "unchanged": unchanged,
        }
