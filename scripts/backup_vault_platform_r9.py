"""Create a complete, point-in-time recovery bundle for the Vault platform."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.archivist.recovery_bundle import create_recovery_bundle
from tools.archivist.maintenance_guard import acquire_maintenance_lease, release_maintenance_lease
from tools.archivist.vault_paths import VaultPaths
from tools.archivist.write_broker import KnowledgeWriteBroker


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", type=Path, default=Path("memories"))
    parser.add_argument("--output-dir", type=Path, default=Path("scratch/vault-r9/recovery"))
    parser.add_argument("--runtime-base", type=Path, default=None)
    parser.add_argument("--extra-runtime", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--owner", default="vault-r9-backup")
    parser.add_argument("--lease-ttl-seconds", type=int, default=900)
    parser.add_argument("--drain-limit", type=int, default=1000)
    args = parser.parse_args()
    extras: dict[str, Path] = {}
    for item in args.extra_runtime:
        if "=" not in item:
            parser.error(f"--extra-runtime must be LABEL=PATH: {item}")
        label, raw_path = item.split("=", 1)
        if not label.strip() or not raw_path.strip():
            parser.error(f"--extra-runtime must be LABEL=PATH: {item}")
        extras[label.strip()] = Path(raw_path.strip())
    lease = None
    result: dict[str, object]
    try:
        lease = acquire_maintenance_lease(
            args.vault,
            owner=args.owner,
            purpose="R9 point-in-time Vault platform backup",
            ttl_seconds=args.lease_ttl_seconds,
        )
        os.environ["VAULT_MAINTENANCE_OWNER"] = args.owner
        broker = KnowledgeWriteBroker(
            vault_paths=VaultPaths(args.vault.resolve()),
            runtime_base=args.runtime_base,
            broker_id=f"{args.owner}-broker",
        )
        before = broker.health()
        recovered = broker.recover_expired(limit=max(0, args.drain_limit))
        drained = broker.drain(limit=max(0, args.drain_limit))
        after = broker.health()
        if int(after.get("queue_depth", 0)) != 0:
            raise RuntimeError(f"broker is not quiescent before backup: {after}")
        result = create_recovery_bundle(
            vault_root=args.vault,
            output_dir=args.output_dir,
            runtime_base=args.runtime_base,
            extra_runtime_roots=extras,
            run_id=args.run_id,
        )
        result["broker_quiescence"] = {
            "before": before,
            "after": after,
            "recovered": recovered,
            "drained": drained,
        }
    except Exception as exc:  # noqa: BLE001 - operational evidence boundary
        result = {"status": "FAIL", "error": str(exc), "owner": args.owner}
    finally:
        if lease is not None:
            release_maintenance_lease(args.vault, owner=args.owner, reason="R9 backup completed")
        os.environ.pop("VAULT_MAINTENANCE_OWNER", None)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
