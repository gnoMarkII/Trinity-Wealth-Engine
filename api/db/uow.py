"""Unit of Work for SQLite database transaction management.

The UoW is the only object that owns a cross-repository transaction.  DAO
functions and connection-bound adapters never commit themselves; the context
manager decides whether the whole unit is committed or rolled back.
"""
import sqlite3
from typing import TYPE_CHECKING, Optional

from api.db.connection import get_connection

if TYPE_CHECKING:
    from application.earnings_call.ports import EarningsCallWorkflowPort


class DbUnitOfWork:
    """Manages SQLite transaction boundary with automatic commit and rollback."""

    def __init__(self, conn: Optional[sqlite3.Connection] = None, db_path: Optional[str] = None) -> None:
        self._conn = conn
        self._db_path = db_path
        self._owns_conn = False
        self._owns_transaction = False
        self._jobs = None
        self._kanban = None
        self._notebooklm = None
        self._outbox = None
        self._earnings_call_workflow = None

    def __enter__(self) -> "DbUnitOfWork":
        if self._conn is None:
            self._conn = get_connection(self._db_path)
            self._owns_conn = True
        # Start an explicit transaction so a read followed by several writes
        # is part of the same unit even before the first INSERT/UPDATE.
        if not self._conn.in_transaction:
            self._conn.execute("BEGIN IMMEDIATE")
            self._owns_transaction = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> Optional[bool]:
        if self._conn is not None:
            if not self._owns_transaction:
                if self._owns_conn:
                    self._conn.close()
                    self._conn = None
                return None
            if exc_type is not None:
                try:
                    self._conn.rollback()
                except Exception:
                    pass
            else:
                self._conn.commit()
            if self._owns_conn:
                self._conn.close()
                self._conn = None
            self._owns_transaction = False
        return None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = get_connection(self._db_path)
            self._owns_conn = True
        return self._conn

    def commit(self) -> None:
        if self._conn is not None:
            self._conn.commit()
            self._owns_transaction = False

    def rollback(self) -> None:
        if self._conn is not None:
            self._conn.rollback()
            self._owns_transaction = False

    @property
    def jobs(self):
        """Connection-bound :class:`JobRepositoryPort` adapter."""
        if self._jobs is None:
            from api.db.adapters import SqliteJobRepositoryAdapter

            self._jobs = SqliteJobRepositoryAdapter(conn=self.conn)
        return self._jobs

    @property
    def kanban(self):
        """Connection-bound :class:`KanbanRepositoryPort` adapter."""
        if self._kanban is None:
            from api.db.adapters import SqliteKanbanRepositoryAdapter

            self._kanban = SqliteKanbanRepositoryAdapter(conn=self.conn)
        return self._kanban

    @property
    def notebooklm(self):
        """Connection-bound NotebookLM job repository adapter."""
        if self._notebooklm is None:
            from api.db.adapters import SqliteNotebookLMJobRepositoryAdapter

            self._notebooklm = SqliteNotebookLMJobRepositoryAdapter(conn=self.conn)
        return self._notebooklm

    @property
    def outbox(self):
        """Connection-bound durable notification outbox adapter."""
        if self._outbox is None:
            from api.db.adapters import SqliteNotificationOutboxAdapter

            self._outbox = SqliteNotificationOutboxAdapter(conn=self.conn)
        return self._outbox

    @property
    def earnings_call_workflow(self) -> "EarningsCallWorkflowPort":
        """Connection-bound :class:`EarningsCallWorkflowPort` adapter."""
        if self._earnings_call_workflow is None:
            from api.db.adapters import SqliteEarningsCallWorkflowAdapter

            self._earnings_call_workflow = SqliteEarningsCallWorkflowAdapter(
                conn=self.conn
            )
        return self._earnings_call_workflow

    @property
    def macro_exports(self):
        """Connection-bound :class:`MacroExportRepositoryPort` adapter."""
        if getattr(self, "_macro_exports", None) is None:
            from api.db.adapters import SqliteMacroExportRepositoryAdapter

            self._macro_exports = SqliteMacroExportRepositoryAdapter(conn=self.conn)
        return self._macro_exports
