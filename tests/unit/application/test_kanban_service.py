"""Unit tests for KanbanApplicationService."""
import pytest
from pathlib import Path
from api.db.connection import get_connection, init_schema
from api.db.adapters import SqliteKanbanRepositoryAdapter
from application.kanban.service import KanbanApplicationService


@pytest.fixture
def test_db(tmp_path: Path) -> str:
    db_file = str(tmp_path / "test_kanban_service.db")
    conn = get_connection(db_file)
    init_schema(conn)
    conn.close()
    return db_file


def test_kanban_service_create_and_deduplicate(test_db: str):
    service = KanbanApplicationService(repo=SqliteKanbanRepositoryAdapter(db_path=test_db))
    card1, created1 = service.create_card(title="Research TSMC", flow="manager", prompt="Deep dive")
    assert created1 is True
    assert card1.title == "Research TSMC"
    assert card1.column_name == "backlog"

    # Attempt duplicate creation in backlog
    card2, created2 = service.create_card(title="Research TSMC", flow="manager", prompt="Deep dive")
    assert created2 is False
    assert card2.card_id == card1.card_id


def test_kanban_service_lifecycle_move_and_delete(test_db: str):
    service = KanbanApplicationService(repo=SqliteKanbanRepositoryAdapter(db_path=test_db))
    card, _ = service.create_card(title="Task Alpha", flow="manager")

    moved = service.move_card(card.card_id, "executing", job_id="job_123")
    assert moved is not None
    assert moved.column_name == "executing"
    assert moved.job_id == "job_123"

    toggled = service.toggle_discord(card.card_id, False)
    assert toggled is not None
    assert toggled.discord_notify is False

    deleted = service.delete_card(card.card_id)
    assert deleted is True
    assert service.get_card(card.card_id) is None
