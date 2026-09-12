"""Bootstrap external portfolio transaction streams without rewriting Markdown."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.archivist.vault_paths import VaultPaths  # noqa: E402
from tools.portfolio.adapters.markdown.repository_adapter import MarkdownVaultRepositoryAdapter  # noqa: E402
from tools.portfolio.adapters.sqlite_mirror_decorator import SqliteMirroredPortfolioRepository  # noqa: E402
from tools.portfolio.transaction_store import PortfolioTransactionStore  # noqa: E402
from tools.portfolio.transactional_repository import TransactionalPortfolioRepository  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", type=Path, default=Path("memories"))
    parser.add_argument("--runtime-base", "--runtime", dest="runtime_base", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("scratch/vault-r8/f07-transaction-source-migration.json"))
    args = parser.parse_args()
    vault = args.vault.resolve()
    os.environ["OBSIDIAN_VAULT_PATH"] = str(vault)
    paths = VaultPaths(vault)
    runtime_base = args.runtime_base.resolve() if args.runtime_base else None
    store = PortfolioTransactionStore(vault_paths=paths, runtime_base=runtime_base)
    runtime = store.runtime_root
    legacy = SqliteMirroredPortfolioRepository(
        underlying_repo=MarkdownVaultRepositoryAdapter(),
        db_path=str(runtime / "portfolio" / "markdown-mirror.sqlite3"),
    )
    repo = TransactionalPortfolioRepository(underlying_repo=legacy, store=store)
    portfolios = repo.list_portfolios()
    migrated = []
    for portfolio in portfolios:
        state = repo.load_state(portfolio.id)
        checkpoint = store.checkpoint(portfolio.id)
        migrated.append({"portfolio_id": portfolio.id, "state_hash": checkpoint.state_hash, "sequence": checkpoint.sequence, "holding_count": len(state.holdings)})
    report = {
        "schema": "vault-r8-portfolio-transaction-source-migration-v1",
        "status": "PASS",
        "vault": str(vault),
        "runtime": str(runtime),
        "portfolio_count": len(migrated),
        "portfolios": migrated,
        "canonical_markdown_rewritten": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
