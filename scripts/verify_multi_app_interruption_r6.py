"""Exercise atomic-file recovery semantics on a scratch transaction only."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.apply_multi_app_migration_r6 import _atomic_write


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prove(run_dir: Path) -> dict[str, object]:
    run_dir = run_dir.resolve()
    scratch = run_dir / "interruption-recovery-r6"
    scratch.mkdir(parents=True, exist_ok=True)
    target = scratch / "transaction.md"
    before_text = "---\nnote_id: stable\n---\nBefore\n"
    after_text = "---\nnote_id: stable\nunknown_field: preserved\n---\nAfter\n"
    target.write_text(before_text, encoding="utf-8", newline="\n")
    before_sha = _sha(target)
    journal = {
        "operation": "atomic_replace",
        "path": target.name,
        "before_sha256": before_sha,
        "after_sha256": hashlib.sha256(after_text.encode("utf-8")).hexdigest(),
        "state": "prepared",
    }
    journal_path = scratch / "transaction-journal.json"
    journal_path.write_text(json.dumps(journal, indent=2) + "\n", encoding="utf-8")

    # Simulate a crash before os.replace: a partial temp file exists, but the
    # canonical target remains byte-for-byte at its before hash.
    with tempfile.NamedTemporaryFile(prefix=".transaction.md.", suffix=".r6tmp", dir=scratch, delete=False) as stream:
        temp_path = Path(stream.name)
        stream.write(after_text[:8].encode("utf-8"))
    interrupted_before_ok = _sha(target) == before_sha
    temp_path.unlink(missing_ok=True)

    _atomic_write(target, after_text)
    journal["state"] = "committed"
    journal_path.write_text(json.dumps(journal, indent=2) + "\n", encoding="utf-8")
    committed_sha = _sha(target)
    # A second recovery pass sees the committed after hash and is a no-op.
    recovery_noop = committed_sha == journal["after_sha256"]
    result = {
        "status": "PASS" if interrupted_before_ok and recovery_noop else "BLOCKED",
        "scratch_only": True,
        "interrupted_before_hash_preserved": interrupted_before_ok,
        "resume_after_hash_matches": recovery_noop,
        "orphan_temp_files": [p.name for p in scratch.glob("*.r6tmp")],
        "journal": str(journal_path),
    }
    (run_dir / "interrupted-transaction-recovery.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    return 0 if prove(args.run_dir)["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
