"""Integration tests for Earnings Call Saga & Outbox delivery end-to-end."""
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from api.db.connection import get_connection
from api.db.adapters import (
    SqliteEarningsCallWorkflowAdapter,
    SqliteKanbanRepositoryAdapter,
)
from application.earnings_call.bootstrap import build_earnings_call_service
from application.earnings_call.dto import EarningsCallSummarizeRequestDTO
from application.earnings_call.workflow import EarningsCallRunStatus, EarningsCallKanbanStatus
from application.kanban.service import KanbanApplicationService
from tools.content.earnings_call.adapters.kanban_adapter import KanbanEarningsCallAdapter
from tools.content.earnings_call.adapters.obsidian_adapter import ObsidianEarningsCallAdapter


@pytest.fixture
def temp_environment(tmp_path):
    db_file = tmp_path / "integration_state.sqlite"
    vault_dir = tmp_path / "test_vault"
    vault_dir.mkdir(parents=True, exist_ok=True)

    conn = get_connection(str(db_file))
    yield conn, str(db_file), vault_dir
    conn.close()


def test_earnings_call_delivery_end_to_end(temp_environment):
    conn, db_path, vault_dir = temp_environment

    llm_port = MagicMock()
    llm_port.summarize.return_value = (
        "### 1. 📊 Key Financial Highlights\n"
        "- Revenue: $19.5B (Beat by 4%)\n"
        "- Gross Margin: 53.2%"
    )

    writer_adapter = ObsidianEarningsCallAdapter(vault_path=vault_dir)
    workflow_adapter = SqliteEarningsCallWorkflowAdapter(db_path=db_path)
    kanban_repo = SqliteKanbanRepositoryAdapter(db_path=db_path)
    kanban_service = KanbanApplicationService(repo=kanban_repo)
    kanban_adapter = KanbanEarningsCallAdapter(kanban_service=kanban_service)

    service = build_earnings_call_service(
        llm_port=llm_port,
        writer_port=writer_adapter,
        workflow_port=workflow_adapter,
        kanban_port=kanban_adapter,
    )

    req = EarningsCallSummarizeRequestDTO(
        ticker="TSM",
        period="Q4 2024",
        transcript="Taiwan Semiconductor Manufacturing Company Limited Fourth Quarter 2024 Earnings Conference Call.",
    )

    # 1. First execution
    run1 = service.summarize_and_store(req)

    assert run1.status == EarningsCallRunStatus.COMPLETED
    assert run1.kanban_status == EarningsCallKanbanStatus.CREATED
    assert run1.reused_existing_run is False
    assert run1.kanban_card_id is not None

    # Verify note was written to temp vault
    note_path = vault_dir / run1.vault_path
    assert note_path.exists()
    note_content = note_path.read_text(encoding="utf-8")
    assert "TSM Earnings Call Q4 2024" in note_content
    assert "Revenue: $19.5B" in note_content

    # Verify Kanban card in DB
    kanban_cards = kanban_service.list_cards()
    assert len(kanban_cards) == 1
    assert kanban_cards[0].card_id == run1.kanban_card_id
    assert kanban_cards[0].title == "[TSM] Earnings Call Q4 2024"

    # 2. Second execution (Idempotent Replay)
    run2 = service.summarize_and_store(req)

    assert run2.run_id == run1.run_id
    assert run2.status == EarningsCallRunStatus.COMPLETED
    assert run2.reused_existing_run is True
    assert run2.kanban_card_id == run1.kanban_card_id

    # LLM should have only been invoked ONCE across both calls
    assert llm_port.summarize.call_count == 1
    # Kanban cards count should still be 1 (no duplicate card)
    assert len(kanban_service.list_cards()) == 1
