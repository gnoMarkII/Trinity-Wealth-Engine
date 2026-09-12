"""Restore a Vault platform recovery bundle into a new staging root."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.archivist.recovery_bundle import restore_recovery_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--restore-root", type=Path, required=True)
    args = parser.parse_args()
    result = restore_recovery_bundle(bundle_root=args.bundle, restore_root=args.restore_root)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
