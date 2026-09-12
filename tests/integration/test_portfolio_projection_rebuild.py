from __future__ import annotations

import os
from pathlib import Path

from tools.portfolio.adapters.markdown.repository_adapter import MarkdownVaultRepositoryAdapter
from tools.portfolio.adapters.sqlite_mirror_decorator import SqliteMirroredPortfolioRepository
from tools.portfolio.domain.ledger_change import LedgerChange
from tools.archivist.vault_paths import VaultPaths
from tools.portfolio.transaction_store import PortfolioTransactionStore
from tools.portfolio.transactional_repository import TransactionalPortfolioRepository


def _repo(tmp_path: Path) -> tuple[TransactionalPortfolioRepository, PortfolioTransactionStore]:
    vault = tmp_path / "memories"
    os.environ["OBSIDIAN_VAULT_PATH"] = str(vault)
    store = PortfolioTransactionStore(vault_paths=VaultPaths(vault), runtime_root=tmp_path / "runtime")
    legacy = SqliteMirroredPortfolioRepository(
        underlying_repo=MarkdownVaultRepositoryAdapter(),
        db_path=str(tmp_path / "mirror.sqlite3"),
    )
    return TransactionalPortfolioRepository(underlying_repo=legacy, store=store), store


def test_transaction_source_bootstrap_and_projection_commit(tmp_path: Path) -> None:
    repo, store = _repo(tmp_path)
    initial = repo.load_state("default")
    assert store.checkpoint("default").sequence == 1
    with repo.unit_of_work("default") as uow:
        state = uow.load_state()
        state.name = "External source"
        uow.commit(state, LedgerChange(kind="unchanged"))
    assert store.checkpoint("default").sequence == 2
    assert repo.load_state("default").name == "External source"
    assert store.replay("default")["name"] == "External source"
    assert initial.last_updated


def test_transaction_source_ledger_replay_is_not_markdown_read(tmp_path: Path) -> None:
    repo, store = _repo(tmp_path)
    repo.load_state("default")
    row = {"Transaction_ID": "tx-1", "Symbol": "AAPL", "Action": "BUY"}
    with repo.unit_of_work("default") as uow:
        state = uow.load_state()
        uow.commit(state, LedgerChange(kind="append", row=row))
    assert repo.read_trade_log("default", symbol="aapl") == [row]
    assert store.replay_ledger("default") == [row]
