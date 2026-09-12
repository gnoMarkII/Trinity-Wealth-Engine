"""Canonical external runtime layout for one Obsidian Vault.

The Vault is a document store.  Queue, transaction, catalog, vector and
reconciliation state live beside it under one deterministic, vault-scoped
runtime root.  Every production composition root should use this module
instead of appending ``data/vault_runtime`` on its own.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


CANONICAL_RUNTIME_BASE_ENV = "INVEST_VAULT_RUNTIME_BASE"
LEGACY_RUNTIME_ENV = "OBSIDIAN_CATALOG_RUNTIME_PATH"
DEFAULT_RUNTIME_DIRECTORY = "vault_runtime"
_SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


class RuntimeLayoutError(ValueError):
    """Raised when runtime configuration is ambiguous or unsafe."""


def vault_id(vault_root: str | Path) -> str:
    """Return the stable identity used to namespace external runtime state."""

    root = Path(vault_root).resolve()
    config_path = root / ".system" / "vault_config.json"
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        configured = str(payload.get("vault_id") or "").strip()
        if configured and _SAFE_ID.fullmatch(configured):
            return configured
    except (OSError, UnicodeDecodeError, ValueError, AttributeError):
        pass
    fallback = re.sub(r"[^A-Za-z0-9_.-]+", "-", root.name).strip("-._")
    return fallback or "vault"


def _configured_base() -> Optional[Path]:
    canonical = os.getenv(CANONICAL_RUNTIME_BASE_ENV, "").strip()
    legacy = os.getenv(LEGACY_RUNTIME_ENV, "").strip()
    if canonical and legacy:
        canonical_path = Path(canonical).expanduser().resolve()
        legacy_path = Path(legacy).expanduser().resolve()
        if canonical_path != legacy_path:
            raise RuntimeLayoutError(
                f"conflicting runtime configuration: {CANONICAL_RUNTIME_BASE_ENV}={canonical_path} "
                f"and {LEGACY_RUNTIME_ENV}={legacy_path}"
            )
        return canonical_path
    configured = canonical or legacy
    return Path(configured).expanduser().resolve() if configured else None


def runtime_base_for(vault_root: str | Path, runtime_base: str | Path | None = None) -> Path:
    """Resolve the un-namespaced runtime base without creating directories."""

    root = Path(vault_root).resolve()
    base = Path(runtime_base).expanduser().resolve() if runtime_base is not None else _configured_base()
    if base is None:
        base = root.parent / "data" / DEFAULT_RUNTIME_DIRECTORY
    if base.is_relative_to(root):
        raise RuntimeLayoutError(f"runtime base must be outside the Vault: {base}")
    return base


def runtime_root_for(vault_root: str | Path, runtime_base: str | Path | None = None, *, create: bool = False) -> Path:
    """Resolve the one canonical vault-scoped runtime root."""

    root = Path(vault_root).resolve()
    runtime = runtime_base_for(root, runtime_base) / vault_id(root)
    if runtime.is_relative_to(root):
        raise RuntimeLayoutError(f"runtime root must be outside the Vault: {runtime}")
    if create:
        runtime.mkdir(parents=True, exist_ok=True)
    return runtime


@dataclass(frozen=True)
class VaultRuntimeLayout:
    """Typed paths for all durable and rebuildable state for one Vault."""

    vault_root: Path
    root: Path
    vault_id: str

    @property
    def broker_db(self) -> Path:
        return self.root / "broker" / "knowledge_write.sqlite3"

    @property
    def portfolio_db(self) -> Path:
        return self.root / "portfolio" / "transactions.sqlite3"

    @property
    def catalog_root(self) -> Path:
        return self.root / "catalog"

    @property
    def vector_root(self) -> Path:
        return self.root / "vector"

    @property
    def reconciliation_root(self) -> Path:
        return self.root / "reconciliation"

    @property
    def logs_root(self) -> Path:
        return self.root / "logs"

    @property
    def checkpoints_root(self) -> Path:
        return self.root / "checkpoints"

    def as_dict(self) -> dict[str, str]:
        return {
            "vault_root": str(self.vault_root),
            "vault_id": self.vault_id,
            "runtime_root": str(self.root),
            "broker_db": str(self.broker_db),
            "portfolio_db": str(self.portfolio_db),
            "catalog_root": str(self.catalog_root),
            "vector_root": str(self.vector_root),
            "reconciliation_root": str(self.reconciliation_root),
            "logs_root": str(self.logs_root),
            "checkpoints_root": str(self.checkpoints_root),
        }


def runtime_layout(
    vault_root: str | Path,
    runtime_base: str | Path | None = None,
    *,
    create: bool = False,
) -> VaultRuntimeLayout:
    root = Path(vault_root).resolve()
    identity = vault_id(root)
    runtime = runtime_root_for(root, runtime_base, create=create)
    return VaultRuntimeLayout(vault_root=root, root=runtime, vault_id=identity)


def assert_runtime_parity(vault_root: str | Path, *runtime_roots: str | Path) -> Path:
    """Fail closed if callers point at different runtime roots."""

    expected = runtime_root_for(vault_root)
    resolved = {Path(value).resolve() for value in runtime_roots if value is not None}
    if resolved and (resolved != {expected}):
        raise RuntimeLayoutError(
            f"runtime root mismatch for Vault {Path(vault_root).resolve()}: expected {expected}, got {sorted(map(str, resolved))}"
        )
    return expected
