"""Rollback one applied R10 Concepts cleanup manifest."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.archivist.concepts_cleanup_executor import rollback_cleanup  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", type=Path, default=Path("memories"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--owner", default="codex-r10-rollback")
    args = parser.parse_args()
    result = rollback_cleanup(args.vault, args.manifest, owner=args.owner)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
