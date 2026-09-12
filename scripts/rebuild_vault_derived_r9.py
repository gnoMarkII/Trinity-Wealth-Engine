"""Rebuild the canonical catalog and vector read models under one R9 lease."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.archivist.maintenance_guard import acquire_maintenance_lease, release_maintenance_lease  # noqa: E402
from tools.archivist.runtime_layout import CANONICAL_RUNTIME_BASE_ENV  # noqa: E402
from tools.archivist.vault_paths import VaultPaths  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", type=Path, default=Path("memories"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--runtime-base", type=Path, default=None)
    parser.add_argument("--owner", default="vault-r9-derived-rebuild")
    parser.add_argument("--lease-ttl-seconds", type=int, default=3600)
    args = parser.parse_args()

    vault = args.vault.resolve()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.runtime_base is not None:
        os.environ[CANONICAL_RUNTIME_BASE_ENV] = str(args.runtime_base.resolve())
    lease = acquire_maintenance_lease(
        vault,
        owner=args.owner,
        purpose="R9 canonical catalog/vector rebuild",
        ttl_seconds=args.lease_ttl_seconds,
    )
    os.environ["VAULT_MAINTENANCE_OWNER"] = args.owner
    result: dict[str, object] = {
        "status": "FAIL",
        "vault": str(vault),
        "runtime_base": os.getenv(CANONICAL_RUNTIME_BASE_ENV),
        "lease_id": lease.lease_id,
    }
    try:
        from scripts.rebuild_catalog_generation_r5 import rebuild as rebuild_catalog
        from scripts.rebuild_vector_generation_r5 import rebuild as rebuild_vector

        catalog = rebuild_catalog(vault, run_dir / "catalog")
        vector = rebuild_vector(vault, run_dir / "vector")
        result.update({"status": "PASS", "catalog": catalog, "vector": vector})
    except Exception as exc:  # noqa: BLE001 - evidence boundary
        result["error"] = str(exc)
    finally:
        released = release_maintenance_lease(vault, owner=args.owner, reason="R9 derived rebuild completed")
        result["lease_status"] = released.status
    (run_dir / "derived-rebuild-r9.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
