"""SQLite-backed Unit of Work and Factory for Investor Essence runtime."""
from __future__ import annotations

import sqlite3
from typing import Optional

from api.db.connection import get_connection
from api.db.repositories.investor_essence_repository import (
    SqliteIntentRepository,
    SqliteOperationRepository,
    SqlitePlanningRepository,
    SqliteSessionRepository,
)
from application.investor_essence.ports import (
    InvestorRuntimeUow,
    InvestorRuntimeUowFactory,
)


class SqliteInvestorRuntimeUow(InvestorRuntimeUow):
    """Transaction-owning Unit of Work for Investor Essence SQLite runtime."""

    def __init__(self, conn: Optional[sqlite3.Connection] = None, db_path: Optional[str] = None) -> None:
        self._conn = conn
        self._db_path = db_path
        self._owns_conn = False
        self._owns_transaction = False
        self._sessions: Optional[SqliteSessionRepository] = None
        self._planning: Optional[SqlitePlanningRepository] = None
        self._operations: Optional[SqliteOperationRepository] = None
        self._intents: Optional[SqliteIntentRepository] = None

    def __enter__(self) -> "SqliteInvestorRuntimeUow":
        if self._conn is None:
            self._conn = get_connection(self._db_path)
            self._owns_conn = True
        if not self._conn.in_transaction:
            self._conn.execute("BEGIN IMMEDIATE")
            self._owns_transaction = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._conn is not None:
            if not self._owns_transaction:
                if self._owns_conn:
                    self._conn.close()
                    self._conn = None
                return
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

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = get_connection(self._db_path)
            self._owns_conn = True
        return self._conn

    @property
    def sessions(self) -> SqliteSessionRepository:
        if self._sessions is None:
            self._sessions = SqliteSessionRepository(self.conn)
        return self._sessions

    @property
    def planning(self) -> SqlitePlanningRepository:
        if self._planning is None:
            self._planning = SqlitePlanningRepository(self.conn)
        return self._planning

    @property
    def operations(self) -> SqliteOperationRepository:
        if self._operations is None:
            self._operations = SqliteOperationRepository(self.conn)
        return self._operations

    @property
    def intents(self) -> SqliteIntentRepository:
        if self._intents is None:
            self._intents = SqliteIntentRepository(self.conn)
        return self._intents

    def commit(self) -> None:
        if self._conn is not None:
            self._conn.commit()
            self._owns_transaction = False

    def rollback(self) -> None:
        if self._conn is not None:
            self._conn.rollback()
            self._owns_transaction = False


class SqliteInvestorRuntimeUowFactory(InvestorRuntimeUowFactory):
    """Factory to create a new SqliteInvestorRuntimeUow."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self._db_path = db_path

    def open(self) -> InvestorRuntimeUow:
        return SqliteInvestorRuntimeUow(db_path=self._db_path)
