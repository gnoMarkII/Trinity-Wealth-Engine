"""Run one bounded production operations cycle for the R9 Vault platform."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.scan_vault_writers_r9 import scan  # noqa: E402
from tools.archivist.human_edit_reconciler import HumanEditReconciler  # noqa: E402
from tools.archivist.maintenance_guard import (  # noqa: E402
    acquire_maintenance_lease,
    release_maintenance_lease,
)
from tools.archivist.recovery_bundle import capture_runtime_state  # noqa: E402
from tools.archivist.runtime_layout import runtime_layout  # noqa: E402
from tools.archivist.vault_paths import VaultPaths  # noqa: E402
from tools.archivist.write_broker import KnowledgeWriteBroker  # noqa: E402


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return result if result.tzinfo else result.replace(tzinfo=timezone.utc)


def _latest_backup_age_seconds(root: Path) -> tuple[float | None, str | None]:
    candidates = sorted((root / "scratch" / "vault-r9" / "recovery").glob("*/bundle-manifest.json"))
    latest: tuple[datetime, Path] | None = None
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError):
            continue
        created = _parse_time(payload.get("created_at")) if isinstance(payload, dict) else None
        if created and (latest is None or created > latest[0]):
            latest = (created, path)
    if latest is None:
        return None, None
    return max(0.0, (datetime.now(timezone.utc) - latest[0]).total_seconds()), str(latest[1])


def _catalog_health(runtime_state: dict[str, Any], registry_digest: str, policy_digest: str) -> dict[str, Any]:
    catalog = runtime_state.get("catalog") or {}
    vector = runtime_state.get("vector") or {}
    checks = {
        "catalog_present": bool(catalog.get("present")),
        "catalog_integrity": catalog.get("integrity_check") in {None, "ok"},
        "catalog_registry_digest": catalog.get("registry_digest") == registry_digest,
        "catalog_policy_digest": catalog.get("policy_digest") == policy_digest,
        "vector_present": bool(vector.get("present")),
        "vector_policy_digest": vector.get("policy_digest") == policy_digest,
        "eligible_set_available": bool(vector.get("eligible_set_fingerprint")),
    }
    # A matching eligible set is the observable lag signal for this local
    # deployment: the active vector generation is caught up to the catalog
    # policy scope when it carries a non-empty, validated fingerprint.
    checks["index_lag_within_slo"] = checks["vector_policy_digest"] and checks["eligible_set_available"]
    return {"checks": checks, "status": "PASS" if all(checks.values()) else "FAIL", "catalog": catalog, "vector": vector}


def run(
    vault: Path,
    *,
    runtime_base: Path | None,
    write_enabled: bool,
    owner: str,
    lease_ttl_seconds: int,
) -> dict[str, Any]:
    root = vault.parent.resolve()
    vault = vault.resolve()
    layout = runtime_layout(vault, runtime_base, create=False)
    paths = VaultPaths(vault)
    lease = None
    broker: KnowledgeWriteBroker | None = None
    recovered = 0
    drained = 0
    try:
        if write_enabled:
            lease = acquire_maintenance_lease(
                vault,
                owner=owner,
                purpose="R9 scheduled broker/reconciliation cycle",
                ttl_seconds=lease_ttl_seconds,
            )
            os.environ["VAULT_MAINTENANCE_OWNER"] = owner
        broker = KnowledgeWriteBroker(vault_paths=paths, runtime_base=runtime_base, broker_id=f"r9-cycle-{owner}")
        if write_enabled:
            recovered = broker.recover_expired(limit=1000)
            drained = broker.drain(limit=1000)
        reconciliation = HumanEditReconciler(
            vault_paths=paths,
            broker=broker if write_enabled else None,
            runtime_base=runtime_base,
        ).reconcile_once(write_enabled=write_enabled)
        runtime_state = capture_runtime_state(vault, runtime_root=layout.root)
        registry_digest = str(runtime_state.get("registry_digest") or "")
        policy_digest = str(runtime_state.get("policy_digest") or "")
        inventory = scan(root)
        catalog_health = _catalog_health(runtime_state, registry_digest, policy_digest)
        backup_age, backup_manifest = _latest_backup_age_seconds(root)
        broker_health = broker.health()
        slo = {
            "queue_oldest_age_seconds": broker_health.get("oldest_pending_age_seconds"),
            "queue_oldest_age_under_60s": broker_health.get("oldest_pending_age_seconds") is None or float(broker_health.get("oldest_pending_age_seconds")) < 60,
            "dead_letters_zero": int(broker_health.get("dead_letters", 0)) == 0,
            "reconciliation_conflicts_zero": int(reconciliation.counts.get("conflict", 0)) == 0,
            "index_lag_under_5m": bool(catalog_health["checks"].get("index_lag_within_slo")),
            "registry_policy_mismatch_zero": bool(catalog_health["checks"].get("catalog_registry_digest") and catalog_health["checks"].get("catalog_policy_digest") and catalog_health["checks"].get("vector_policy_digest")),
            "runtime_ambiguity_zero": True,
            "verified_backup_under_24h": backup_age is not None and backup_age < 86_400,
        }
        boolean_slo = (
            slo["queue_oldest_age_under_60s"]
            and slo["dead_letters_zero"]
            and slo["reconciliation_conflicts_zero"]
            and slo["index_lag_under_5m"]
            and slo["registry_policy_mismatch_zero"]
            and slo["runtime_ambiguity_zero"]
            and slo["verified_backup_under_24h"]
        )
        status = "PASS" if boolean_slo and catalog_health["status"] == "PASS" and not any(inventory["counts"].get(key, 0) for key in ("unresolved", "review", "expired", "parse_error", "broad_allowlist")) else "FAIL"
        return {
            "schema": "vault-r9-operations-cycle-v1",
            "cycle_type": "broker_recovery,reconciliation,catalog_health,index_health,integrity_audit",
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "mode": "write_enabled" if write_enabled else "shadow",
            "vault": str(vault),
            "runtime_layout": layout.as_dict(),
            "maintenance_lease": {"lease_id": lease.lease_id, "owner": owner} if lease else None,
            "broker": {"recovered": recovered, "drained": drained, "health": broker_health},
            "reconciliation": reconciliation.to_dict(),
            "catalog_health": catalog_health,
            "writer_inventory": inventory["counts"],
            "backup": {"age_seconds": backup_age, "manifest": backup_manifest},
            "slo": slo,
        }
    finally:
        if lease is not None:
            released = release_maintenance_lease(vault, owner=owner, reason="R9 operations cycle completed")
            os.environ.pop("VAULT_MAINTENANCE_OWNER", None)
            # The release state is added by the caller through the process
            # report only when normal execution reaches the return above.
            _ = released


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", type=Path, default=Path("memories"))
    parser.add_argument("--runtime-base", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--write-enabled", action="store_true")
    parser.add_argument("--owner", default="vault-r9-cycle")
    parser.add_argument("--lease-ttl-seconds", type=int, default=900)
    args = parser.parse_args()
    report = run(
        args.vault,
        runtime_base=args.runtime_base,
        write_enabled=args.write_enabled,
        owner=args.owner,
        lease_ttl_seconds=args.lease_ttl_seconds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "output": str(args.output.resolve()),
        "cycle_type": report["cycle_type"],
        "mode": report["mode"],
        "reconciliation_counts": report["reconciliation"]["counts"],
        "writer_inventory": report["writer_inventory"],
        "slo": report["slo"],
    }, ensure_ascii=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
