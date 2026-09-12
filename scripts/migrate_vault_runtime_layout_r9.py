"""Plan or apply the final unified external runtime layout migration."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.archivist.maintenance_guard import acquire_maintenance_lease, release_maintenance_lease
from tools.archivist.runtime_migration import migrate_runtime_layout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", type=Path, default=Path("memories"))
    parser.add_argument("--runtime-base", type=Path, default=Path("data/vault_runtime"))
    parser.add_argument("--output", type=Path, default=Path("scratch/vault-r9/runtime-migration.json"))
    parser.add_argument("--apply", action="store_true", help="apply the migration; default is read-only dry-run")
    parser.add_argument("--owner", default="vault-r9-migration")
    parser.add_argument("--lease-ttl-seconds", type=int, default=900)
    args = parser.parse_args()
    lease = None
    if args.apply:
        lease = acquire_maintenance_lease(
            args.vault,
            owner=args.owner,
            purpose="R9 unified external runtime migration",
            ttl_seconds=args.lease_ttl_seconds,
        )
        os.environ["VAULT_MAINTENANCE_OWNER"] = args.owner
    try:
        result = migrate_runtime_layout(
            vault_root=args.vault,
            runtime_base=args.runtime_base,
            apply=args.apply,
            output=args.output,
        )
    finally:
        if lease is not None:
            release_maintenance_lease(args.vault, owner=args.owner, reason="R9 runtime migration completed")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
