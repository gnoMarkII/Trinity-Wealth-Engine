"""Tests for atomic News Funnel card persistence."""
from pathlib import Path

from api.db.connection import get_connection
from api.db.legacy_adapter import LegacyNewsFunnelCardAdapter
from application.macro.card_service import NewsFunnelCardApplicationService


class _Prompt:
    def format_prompt(self, period, events):
        return f"{period}:{len(events)}"


def test_news_funnel_upsert_reuses_one_open_card(tmp_path: Path, monkeypatch):
    db_path = str(tmp_path / "news-funnel.sqlite")
    monkeypatch.setenv("WEBUI_STATE_DB_PATH", db_path)

    service = NewsFunnelCardApplicationService(
        storage=LegacyNewsFunnelCardAdapter(),
        prompt=_Prompt(),
    )
    service.upsert("morning", [{"event_id": "one"}])
    service.upsert("morning", [{"event_id": "one"}, {"event_id": "two"}])

    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT card_id, title, prompt FROM kanban_cards WHERE flow = 'news_funnel'"
        ).fetchall()
    finally:
        conn.close()

    assert len(rows) == 1
    assert rows[0]["prompt"] == "morning:2"
