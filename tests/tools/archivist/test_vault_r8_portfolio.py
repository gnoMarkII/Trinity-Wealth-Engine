from __future__ import annotations

from pathlib import Path

from tools.archivist.metadata import parse_note
from tools.archivist.vault_paths import VaultPaths
from tools.archivist.write_broker import KnowledgeWriteBroker
from tools.portfolio.projection_boundary import PortfolioProjectionBoundary
from tools.portfolio.transaction_store import PortfolioTransactionStore


def test_transaction_replay_and_projection_refresh_are_deterministic(tmp_path: Path) -> None:
    vault = tmp_path / "memories"
    vault_paths = VaultPaths(vault)
    store = PortfolioTransactionStore(vault_paths=vault_paths, runtime_root=tmp_path / "runtime")
    store.append_state("default", {"cash": 100, "holdings": {"FTNT": 2}}, event_id="evt-1")
    store.append_event(
        "default",
        "state_patch",
        {"state_patch": {"holdings": {"FTNT": 3}}},
        event_id="evt-2",
        expected_sequence=1,
    )
    assert store.replay("default")["holdings"] == {"FTNT": 3}
    checkpoint = store.checkpoint("default")
    assert checkpoint.sequence == 2

    broker = KnowledgeWriteBroker(vault_paths=vault_paths, runtime_root=tmp_path / "runtime", broker_id="r8")
    boundary = PortfolioProjectionBoundary(broker)
    projections = [
        {
            "entity_type": "holding",
            "title": "FTNT holding",
            "stable_key": "FTNT",
            "generated_text": "Units: 3",
            "metadata": {"portfolio_id": "default"},
        }
    ]
    first = boundary.rebuild_from_transaction_store(store=store, portfolio_id="default", projections=projections)
    assert first[0].status == "committed"
    path = vault / str(first[0].relative_path)
    metadata, body, issues = parse_note(path.read_text(encoding="utf-8"))
    assert not issues and metadata["search_scope"] == "excluded"
    path.write_text(path.read_text(encoding="utf-8") + "\nHuman annotation.\n", encoding="utf-8")

    store.append_state("default", {"cash": 99, "holdings": {"FTNT": 4}}, event_id="evt-3", expected_sequence=2)
    projections[0]["generated_text"] = "Units: 4"
    projections[0]["existing_body"] = path.read_text(encoding="utf-8").split("---", 2)[-1].strip()
    second = boundary.rebuild_from_transaction_store(store=store, portfolio_id="default", projections=projections)
    assert second[0].status == "committed"
    refreshed = path.read_text(encoding="utf-8")
    assert "Units: 4" in refreshed
    assert "Human annotation." in refreshed
    assert second[0].revision_id != first[0].revision_id
