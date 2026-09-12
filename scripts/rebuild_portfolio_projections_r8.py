"""Replay the external portfolio event store and publish projections."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.archivist.vault_paths import VaultPaths
from tools.archivist.write_broker import KnowledgeWriteBroker
from tools.portfolio.projection_boundary import PortfolioProjectionBoundary
from tools.portfolio.transaction_store import PortfolioTransactionStore


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", type=Path, default=Path("memories"))
    parser.add_argument("--runtime-base", "--runtime", dest="runtime_base", type=Path)
    parser.add_argument("--portfolio-id", required=True)
    parser.add_argument("--projections-json", type=Path)
    parser.add_argument("--output", type=Path, default=Path("scratch/vault-r8/f07-projection-rebuild.json"))
    parser.add_argument("--write-enabled", action="store_true")
    args = parser.parse_args()
    paths = VaultPaths(args.vault.resolve())
    store = PortfolioTransactionStore(vault_paths=paths, runtime_base=args.runtime_base)
    checkpoint = store.checkpoint(args.portfolio_id)
    payload: dict[str, Any] = {
        "schema": "vault-r8-portfolio-rebuild-v1",
        "mode": "write_enabled" if args.write_enabled else "preflight",
        "portfolio_id": args.portfolio_id,
        "checkpoint": checkpoint.__dict__,
        "state": store.replay(args.portfolio_id),
        "receipts": [],
    }
    if args.write_enabled:
        if args.projections_json is None:
            raise SystemExit("--projections-json is required with --write-enabled")
        projections = json.loads(args.projections_json.read_text(encoding="utf-8"))
        if not isinstance(projections, list):
            raise SystemExit("projection input must be a JSON array")
        broker = KnowledgeWriteBroker(vault_paths=paths, runtime_base=args.runtime_base)
        receipts = PortfolioProjectionBoundary(broker).rebuild_from_transaction_store(
            store=store,
            portfolio_id=args.portfolio_id,
            projections=projections,
        )
        payload["receipts"] = [receipt.to_dict() for receipt in receipts]
    payload["status"] = "PASS"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
