import json
from pathlib import Path

import pytest

from tools.archivist.runtime_layout import (
    CANONICAL_RUNTIME_BASE_ENV,
    LEGACY_RUNTIME_ENV,
    RuntimeLayoutError,
    runtime_layout,
    runtime_root_for,
    vault_id,
)
from tools.archivist.catalog_runtime import catalog_runtime_root
from tools.archivist.vector_generation import vector_runtime_path
from tools.archivist.vault_paths import VaultPaths
from tools.archivist.write_broker import KnowledgeWriteBroker
from tools.portfolio.transaction_store import PortfolioTransactionStore


def _vault(tmp_path: Path, configured_id: str = "demo-vault") -> Path:
    root = tmp_path / "memories"
    (root / ".system").mkdir(parents=True)
    (root / ".system" / "vault_config.json").write_text(
        json.dumps({"layout_version": 2, "vault_id": configured_id}), encoding="utf-8"
    )
    return root


def test_runtime_layout_is_vault_scoped_and_typed(tmp_path: Path) -> None:
    vault = _vault(tmp_path)
    layout = runtime_layout(vault, runtime_base=tmp_path / "data" / "vault_runtime")

    assert layout.vault_id == "demo-vault"
    assert layout.root == (tmp_path / "data" / "vault_runtime" / "demo-vault").resolve()
    assert layout.broker_db == layout.root / "broker" / "knowledge_write.sqlite3"
    assert layout.portfolio_db == layout.root / "portfolio" / "transactions.sqlite3"
    assert layout.vector_root == layout.root / "vector"


def test_runtime_root_rejects_a_path_inside_the_vault(tmp_path: Path) -> None:
    vault = _vault(tmp_path)

    with pytest.raises(RuntimeLayoutError):
        runtime_root_for(vault, runtime_base=vault / ".system" / "runtime")


def test_conflicting_canonical_and_legacy_environment_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    vault = _vault(tmp_path)
    monkeypatch.setenv(CANONICAL_RUNTIME_BASE_ENV, str(tmp_path / "canonical"))
    monkeypatch.setenv(LEGACY_RUNTIME_ENV, str(tmp_path / "legacy"))

    with pytest.raises(RuntimeLayoutError):
        runtime_root_for(vault)


def test_vault_id_is_stable_when_the_vault_directory_is_renamed(tmp_path: Path) -> None:
    original = _vault(tmp_path, configured_id="stable-id")
    renamed = tmp_path / "renamed-vault"
    original.rename(renamed)

    assert vault_id(renamed) == "stable-id"


def test_production_runtime_components_share_one_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    vault = _vault(tmp_path, configured_id="shared-vault")
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(vault))
    paths = VaultPaths(vault)
    expected = runtime_root_for(vault)

    broker = KnowledgeWriteBroker(vault_paths=paths)
    portfolio = PortfolioTransactionStore(vault_paths=paths)

    assert broker.runtime_root == expected
    assert portfolio.runtime_root == expected
    assert catalog_runtime_root(vault) == expected
    assert vector_runtime_path(vault) == expected / "vector"
