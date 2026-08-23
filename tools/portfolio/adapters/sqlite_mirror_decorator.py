import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Optional, List, Dict, Set

from core.logger import get_logger
from api.config import get_state_db_path
from tools.portfolio.domain.models import PortfolioState, PortfolioMeta
from tools.portfolio.domain.ledger_change import LedgerChange
from tools.portfolio.domain.validator import validate_portfolio_id
from tools.portfolio.ports.repository_port import PortfolioRepositoryPort, PortfolioUnitOfWork
from .markdown.paths import get_portfolio_filepath, get_trades_log_filepath

log = get_logger(__name__)

_SCHEMA_VERSION = 1
_MIRROR_VERSION = 1
_DIRTY_PORTFOLIOS: Set[str] = set()

_INIT_SQL = """
CREATE TABLE IF NOT EXISTS portfolio_state_mirror (
    portfolio_id TEXT PRIMARY KEY,
    schema_version INTEGER NOT NULL,
    mirror_version INTEGER NOT NULL,
    state_json TEXT NOT NULL,
    source_fingerprint TEXT NOT NULL,
    source_master_sha256 TEXT NOT NULL,
    source_trades_sha256 TEXT NOT NULL,
    source_mtime_ns INTEGER NOT NULL,
    synced_at REAL NOT NULL,
    status TEXT NOT NULL
);
"""


def _get_connection(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=10.0)
    conn.execute(_INIT_SQL)
    conn.commit()
    return conn


def _compute_disk_fingerprint(portfolio_id: str) -> str:
    master_file = get_portfolio_filepath(portfolio_id)
    ledger_file = get_trades_log_filepath(portfolio_id)

    mtime_master = master_file.stat().st_mtime_ns if master_file.exists() else 0
    size_master = master_file.stat().st_size if master_file.exists() else 0
    mtime_ledger = ledger_file.stat().st_mtime_ns if ledger_file.exists() else 0
    size_ledger = ledger_file.stat().st_size if ledger_file.exists() else 0

    return f"{mtime_master}:{size_master}:{mtime_ledger}:{size_ledger}"


def _compute_sha256(filepath: Path) -> str:
    if not filepath.exists():
        return ""
    h = hashlib.sha256()
    with filepath.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class MirroredPortfolioUnitOfWork(PortfolioUnitOfWork):
    """Unit of Work decorator forwarding to underlying Markdown UoW and writing through to SQLite."""

    def __init__(
        self,
        underlying_uow: PortfolioUnitOfWork,
        decorator: "SqliteMirroredPortfolioRepository",
        portfolio_id: str,
    ):
        self.underlying_uow = underlying_uow
        self.decorator = decorator
        self.portfolio_id = portfolio_id

    def __enter__(self) -> "MirroredPortfolioUnitOfWork":
        self.underlying_uow.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> Optional[bool]:
        return self.underlying_uow.__exit__(exc_type, exc_val, exc_tb)

    def load_state(self) -> PortfolioState:
        return self.underlying_uow.load_state()

    def read_trade_log_locked(self) -> List[Dict]:
        return self.underlying_uow.read_trade_log_locked()

    def commit(self, state: PortfolioState, ledger_change: Optional[LedgerChange] = None) -> None:
        # 1. Authoritative Markdown Commit
        self.underlying_uow.commit(state, ledger_change)

        # 2. Write-through to SQLite Mirror (Best-effort)
        self.decorator._write_mirror(self.portfolio_id, state)

    def rollback(self) -> None:
        self.underlying_uow.rollback()


class SqliteMirroredPortfolioRepository(PortfolioRepositoryPort):
    """Cache/Mirror Decorator on top of Authoritative Markdown Repository."""

    def __init__(
        self,
        underlying_repo: PortfolioRepositoryPort,
        db_path: Optional[str] = None,
    ):
        self.underlying_repo = underlying_repo
        self.db_path = db_path or get_state_db_path()

    def unit_of_work(self, portfolio_id: str = "default") -> PortfolioUnitOfWork:
        pid = validate_portfolio_id(portfolio_id)
        uow = self.underlying_repo.unit_of_work(pid)
        return MirroredPortfolioUnitOfWork(uow, self, pid)

    def load_state(self, portfolio_id: str = "default") -> PortfolioState:
        pid = validate_portfolio_id(portfolio_id)

        # Fast path check: if marked dirty in-memory -> reload from Markdown
        if pid in _DIRTY_PORTFOLIOS:
            return self._reload_and_update_mirror(pid)

        disk_fingerprint = _compute_disk_fingerprint(pid)

        try:
            with _get_connection(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT schema_version, mirror_version, state_json, source_fingerprint, status
                    FROM portfolio_state_mirror WHERE portfolio_id = ?
                    """,
                    (pid,),
                )
                row = cursor.fetchone()
                if row:
                    s_ver, m_ver, state_json, src_fp, status = row
                    if (
                        s_ver == _SCHEMA_VERSION
                        and m_ver == _MIRROR_VERSION
                        and status == "OK"
                        and src_fp == disk_fingerprint
                    ):
                        data = json.loads(state_json)
                        return PortfolioState.model_validate(data)
        except Exception as e:
            log.warning("SQLite mirror read failed for %s (%s) -> Falling back to Markdown", pid, e)
            _DIRTY_PORTFOLIOS.add(pid)

        return self._reload_and_update_mirror(pid)

    def _reload_and_update_mirror(self, portfolio_id: str) -> PortfolioState:
        state = self.underlying_repo.load_state(portfolio_id)
        self._write_mirror(portfolio_id, state)
        _DIRTY_PORTFOLIOS.discard(portfolio_id)
        return state

    def _write_mirror(self, portfolio_id: str, state: PortfolioState) -> None:
        """Write state DTO and fingerprint into SQLite mirror table."""
        try:
            pid = validate_portfolio_id(portfolio_id)
            master_file = get_portfolio_filepath(pid)
            ledger_file = get_trades_log_filepath(pid)

            disk_fingerprint = _compute_disk_fingerprint(pid)
            master_sha = _compute_sha256(master_file)
            ledger_sha = _compute_sha256(ledger_file)
            mtime_ns = master_file.stat().st_mtime_ns if master_file.exists() else 0

            state_json = json.dumps(state.model_dump(exclude_none=True), ensure_ascii=False)
            synced_at = time.time()

            with _get_connection(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO portfolio_state_mirror (
                        portfolio_id, schema_version, mirror_version, state_json,
                        source_fingerprint, source_master_sha256, source_trades_sha256,
                        source_mtime_ns, synced_at, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(portfolio_id) DO UPDATE SET
                        schema_version=excluded.schema_version,
                        mirror_version=excluded.mirror_version,
                        state_json=excluded.state_json,
                        source_fingerprint=excluded.source_fingerprint,
                        source_master_sha256=excluded.source_master_sha256,
                        source_trades_sha256=excluded.source_trades_sha256,
                        source_mtime_ns=excluded.source_mtime_ns,
                        synced_at=excluded.synced_at,
                        status=excluded.status
                    """,
                    (
                        pid,
                        _SCHEMA_VERSION,
                        _MIRROR_VERSION,
                        state_json,
                        disk_fingerprint,
                        master_sha,
                        ledger_sha,
                        mtime_ns,
                        synced_at,
                        "OK",
                    ),
                )
                conn.commit()
            _DIRTY_PORTFOLIOS.discard(pid)
        except Exception as e:
            log.warning("Failed to write SQLite mirror for %s: %s -> Marking DIRTY in memory", portfolio_id, e)
            _DIRTY_PORTFOLIOS.add(portfolio_id)

    # Delegated direct methods to underlying repository
    def read_trade_log(self, portfolio_id: str = "default", symbol: Optional[str] = None) -> List[Dict]:
        return self.underlying_repo.read_trade_log(portfolio_id, symbol=symbol)

    def backup_and_reset_clean_slate(self, portfolio_id: str = "default") -> PortfolioState:
        state = self.underlying_repo.backup_and_reset_clean_slate(portfolio_id)
        self._write_mirror(portfolio_id, state)
        return state

    def list_portfolios(self) -> List[PortfolioMeta]:
        return self.underlying_repo.list_portfolios()

    def create_portfolio(self, name: str, portfolio_id: Optional[str] = None) -> PortfolioMeta:
        meta = self.underlying_repo.create_portfolio(name, portfolio_id=portfolio_id)
        # Update mirror for newly created portfolio
        state = self.underlying_repo.load_state(meta.id)
        self._write_mirror(meta.id, state)
        return meta

    def delete_portfolio(self, portfolio_id: str) -> None:
        pid = validate_portfolio_id(portfolio_id)
        self.underlying_repo.delete_portfolio(pid)
        try:
            with _get_connection(self.db_path) as conn:
                conn.execute("DELETE FROM portfolio_state_mirror WHERE portfolio_id = ?", (pid,))
                conn.commit()
        except Exception as e:
            log.warning("Failed to delete SQLite mirror for %s: %s", pid, e)
        _DIRTY_PORTFOLIOS.discard(pid)

    def rename_portfolio(self, portfolio_id: str, new_name: str) -> PortfolioMeta:
        meta = self.underlying_repo.rename_portfolio(portfolio_id, new_name)
        state = self.underlying_repo.load_state(meta.id)
        self._write_mirror(meta.id, state)
        return meta

    def portfolio_exists(self, portfolio_id: str) -> bool:
        return self.underlying_repo.portfolio_exists(portfolio_id)
