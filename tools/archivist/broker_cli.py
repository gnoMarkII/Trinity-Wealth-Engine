"""Small local operator CLI for the external knowledge write broker."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.archivist.vault_paths import VaultPaths
from tools.archivist.write_broker import KnowledgeWriteBroker


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect and operate the external Vault write broker")
    parser.add_argument("--vault", type=Path, default=Path("memories"))
    parser.add_argument("--runtime-base", "--runtime", dest="runtime_base", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("health")
    drain = sub.add_parser("drain")
    drain.add_argument("--limit", type=int, default=100)
    retry = sub.add_parser("retry")
    retry.add_argument("command_id")
    args = parser.parse_args(argv)
    broker = KnowledgeWriteBroker(vault_paths=VaultPaths(args.vault), runtime_base=args.runtime_base)
    if args.command == "health":
        result = broker.health()
    elif args.command == "drain":
        result = {"processed": broker.recover_expired(limit=args.limit), **broker.health()}
    else:
        receipt = broker.retry(args.command_id)
        result = receipt.to_dict() if receipt is not None else {"status": "not_found"}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
