"""Worker entry point for draining the external Vault write queue."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from tools.archivist.vault_paths import VaultPaths
from tools.archivist.write_broker import KnowledgeWriteBroker


def build_broker(vault: Optional[str | Path] = None, runtime_base: Optional[str | Path] = None) -> KnowledgeWriteBroker:
    return KnowledgeWriteBroker(vault_paths=VaultPaths(Path(vault) if vault else None), runtime_base=runtime_base)


def run_once(*, vault: Optional[str | Path] = None, runtime_base: Optional[str | Path] = None, limit: int = 100) -> dict[str, Any]:
    """Recover expired work, drain ready commands, and return safe telemetry."""
    broker = build_broker(vault, runtime_base)
    recovered = broker.recover_expired(limit=limit)
    drained = broker.drain(limit=limit)
    return {"recovered": recovered, "drained": drained, "health": broker.health()}


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", type=Path, default=Path("memories"))
    parser.add_argument("--runtime-base", "--runtime", dest="runtime_base", type=Path, default=Path("data/vault_runtime") )
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    print(json.dumps(run_once(vault=args.vault, runtime_base=args.runtime_base, limit=args.limit), ensure_ascii=False, indent=2))
