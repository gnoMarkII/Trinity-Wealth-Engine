"""Apply the guarded R10 Concepts cleanup after an explicit --apply flag."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.archivist.concepts_cleanup_executor import apply_cleanup  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", type=Path, default=Path("memories"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--quarantine-root",
        type=Path,
        default=Path("data/vault_quarantine/memories"),
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--owner", default="codex-r10")
    parser.add_argument("--policy-only", action="store_true")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="perform the guarded mutation; without this flag only print the selected count",
    )
    args = parser.parse_args()
    plan = json.loads(args.plan.resolve().read_text(encoding="utf-8"))
    selected = [
        row
        for row in plan.get("concepts", [])
        if row.get("apply_eligible")
        and row.get("approval_state") == "approved"
        and row.get("disposition") in {"RETIRE", "RELOCATE"}
    ]
    if not args.apply:
        print(
            json.dumps(
                {
                    "status": "DRY_RUN",
                    "policy_only": args.policy_only,
                    "approved_apply_items": len(selected),
                    "message": "pass --apply to invoke the guarded maintenance executor",
                },
                ensure_ascii=False,
            )
        )
        return 0
    result = apply_cleanup(
        args.vault,
        args.plan,
        args.quarantine_root,
        run_id=args.run_id,
        owner=args.owner,
        policy_only=args.policy_only,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
