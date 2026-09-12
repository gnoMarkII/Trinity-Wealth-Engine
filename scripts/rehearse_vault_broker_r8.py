"""Exercise idempotency, collision, retry, and receipt recovery on a clone."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from application.knowledge.write_models import KnowledgeWriteCommand
from tools.archivist.vault_paths import VaultPaths
from tools.archivist.write_broker import KnowledgeWriteBroker


def _command(key: str, body: str = "R8 rehearsal") -> KnowledgeWriteCommand:
    return KnowledgeWriteCommand(
        operation="upsert_note",
        idempotency_key=key,
        producer="r8-rehearsal",
        producer_version="1",
        payload={"metadata": {"schema_version": 2, "entity_type": "concept", "title": "R8 rehearsal"}, "body": body},
    )


def run() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="vault-r8-rehearsal-", ignore_cleanup_errors=True) as temporary:
        root = Path(temporary)
        vault = root / "memories"
        runtime = root / "runtime"
        broker = KnowledgeWriteBroker(vault_paths=VaultPaths(vault), runtime_root=runtime, broker_id="rehearsal")
        first = broker.submit(_command("same-command"))
        replay = broker.submit(_command("same-command"))
        collision = broker.submit(_command("same-command", body="different"))

        class FailingExecutor:
            def commit(self, command: KnowledgeWriteCommand, *, fencing_token: int = 0):
                raise RuntimeError("rehearsal provider failure")

        failing = KnowledgeWriteBroker(
            vault_paths=VaultPaths(root / "dead-letter-vault"),
            runtime_root=root / "dead-letter-runtime",
            executor=FailingExecutor(),
            broker_id="rehearsal-failing",
            max_attempts=1,
            retry_backoff_seconds=0,
        )
        dead_letter = failing.submit(_command("dead-letter"))
        retry = failing.retry(dead_letter.command_id)
        with failing._connect() as conn:  # read-only evidence query
            audit = [dict(row) for row in conn.execute(
                "SELECT to_status, detail_json FROM broker_events WHERE command_id=? ORDER BY event_id",
                (dead_letter.command_id,),
            ).fetchall()]

        checks = {
            "first_commit": first.status == "committed",
            "same_payload_reuses_receipt": replay.status == "duplicate_reused" and replay.revision_id == first.revision_id,
            "different_payload_conflicts": collision.status == "conflict" and collision.conflict_code == "idempotency_key_reused",
            "dead_letter": dead_letter.status == "dead_letter",
            "operator_retry_audited": bool(retry and retry.status == "dead_letter" and any(json.loads(row["detail_json"]).get("operator_retry") for row in audit)),
            "no_duplicate_revision": len(list((vault / ".system" / "artifacts" / "heads").glob("*.json"))) == 1,
        }
        return {
            "schema": "vault-r8-broker-rehearsal-v1",
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "receipts": {"first": first.to_dict(), "replay": replay.to_dict(), "collision": collision.to_dict(), "dead_letter": dead_letter.to_dict(), "retry": retry.to_dict() if retry else None},
            "audit_event_count": len(audit),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("scratch/vault-r8/f03-broker-rehearsal.json"))
    args = parser.parse_args()
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
