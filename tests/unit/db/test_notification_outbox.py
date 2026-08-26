"""Contract tests for the durable notification outbox adapter."""
from api.db.adapters import SqliteNotificationOutboxAdapter


def test_notification_outbox_is_idempotent_and_tracks_delivery(tmp_path):
    adapter = SqliteNotificationOutboxAdapter(db_path=str(tmp_path / "state.sqlite"))
    payload = {"audio_path": "briefing.mp3", "title": "Briefing"}

    first = adapter.enqueue(
        event_id="notebooklm:hash-1",
        idempotency_key="notebooklm:hash-1",
        aggregate_type="notebooklm_briefing",
        aggregate_id="job-1",
        event_type="discord_audio_ready",
        payload=payload,
    )
    second = adapter.enqueue(
        event_id="different-event-id",
        idempotency_key="notebooklm:hash-1",
        aggregate_type="notebooklm_briefing",
        aggregate_id="job-1",
        event_type="discord_audio_ready",
        payload={"changed": True},
    )

    assert first["event_id"] == second["event_id"] == "notebooklm:hash-1"
    assert second["status"] == "pending"
    assert adapter.get("notebooklm:hash-1")["payload_json"]

    adapter.mark_failed("notebooklm:hash-1", "temporary failure")
    assert adapter.get("notebooklm:hash-1")["status"] == "failed"
    assert adapter.list_pending()[0]["attempts"] == 1

    adapter.mark_sent("notebooklm:hash-1")
    assert adapter.get("notebooklm:hash-1")["status"] == "sent"
    assert adapter.list_pending() == []
