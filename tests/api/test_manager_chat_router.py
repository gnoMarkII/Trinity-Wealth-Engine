"""Tests for api/routers/manager_chat_router.py."""
import pytest
from fastapi.testclient import TestClient


def test_unauthenticated_request_rejected(client: TestClient):
    resp = client.get("/api/manager/sessions")
    assert resp.status_code == 401


def test_list_and_create_sessions(authed_client: TestClient):
    # 1. Create a session
    resp = authed_client.post("/api/manager/sessions", json={"title": "พอร์ตเติบโต 2026"})
    assert resp.status_code == 201
    created = resp.json()
    assert created["title"] == "พอร์ตเติบโต 2026"
    assert created["id"].startswith("session_")
    session_id = created["id"]

    # 2. List sessions
    resp_list = authed_client.get("/api/manager/sessions")
    assert resp_list.status_code == 200
    sessions = resp_list.json()
    assert any(s["id"] == session_id for s in sessions)


def test_get_messages_for_new_session_is_empty(authed_client: TestClient):
    resp = authed_client.post("/api/manager/sessions", json={"title": "เทสต์ประวัติ"})
    session_id = resp.json()["id"]

    resp_msgs = authed_client.get(f"/api/manager/sessions/{session_id}/messages")
    assert resp_msgs.status_code == 200
    assert resp_msgs.json() == []


def test_get_messages_404_for_missing_session(authed_client: TestClient):
    resp = authed_client.get("/api/manager/sessions/non_existent_session_id/messages")
    assert resp.status_code == 404


def test_get_sandbox_state_returns_valid_structure(authed_client: TestClient):
    resp = authed_client.get("/api/manager/sandbox/state")
    assert resp.status_code == 200
    data = resp.json()
    assert "memory_files" in data
    assert "active_skills" in data
    assert "quarantined_skills" in data
    assert "scratch_files_count" in data


def test_get_hermes_sync_status(authed_client: TestClient):
    resp = authed_client.get("/api/manager/hermes/sync-status")
    assert resp.status_code == 200
    data = resp.json()
    assert "is_installed" in data
    assert "current_commit" in data


def test_stream_chat_emits_sse_events(authed_client: TestClient, monkeypatch):
    from application.manager.manager_chat_service import ManagerChatService

    async def mock_stream_turn(self, session_id, user_message, context_refs=()):
        yield "สวัสดีครับ "
        yield "แนะนำหุ้น NVDA ครับ"

    monkeypatch.setattr(ManagerChatService, "stream_turn", mock_stream_turn)

    # Create session
    resp = authed_client.post("/api/manager/sessions", json={"title": "เทสต์สตรีม"})
    session_id = resp.json()["id"]

    # Stream turn
    stream_resp = authed_client.post(
        f"/api/manager/sessions/{session_id}/stream",
        json={"message": "สวัสดีครับ แนะนำหุ้นให้หน่อย", "context_refs": ["@Portfolio"]},
    )
    assert stream_resp.status_code == 200
    assert "text/event-stream" in stream_resp.headers["content-type"]
    body = stream_resp.text
    assert "event: text_delta" in body
    assert "NVDA" in body
    assert "event: done" in body


def test_promote_skill_endpoint_handles_missing_skill(authed_client: TestClient):
    resp = authed_client.post("/api/manager/sandbox/skills/non_existent_skill_xyz/promote")
    assert resp.status_code == 400
    assert "Cannot promote skill" in resp.json()["detail"]


def test_stream_turn_multi_turn_history(authed_client: TestClient, monkeypatch):
    from application.manager.manager_chat_service import ManagerChatService
    from core.sandbox.models import AdvisorChatMessage

    recorded_histories = []

    async def mock_stream_reply(self, session_id, user_message, context_refs=(), history=()):
        recorded_histories.append(list(history))
        yield "ข้อความตอบกลับ"

    from tools.hermes_bridge.gemini_stream_engine import GeminiStreamEngine
    monkeypatch.setattr(GeminiStreamEngine, "stream_reply", mock_stream_reply)

    # Create session
    resp = authed_client.post("/api/manager/sessions", json={"title": "บทสนทนาใหม่"})
    session_id = resp.json()["id"]

    # Turn 1
    stream_resp1 = authed_client.post(
        f"/api/manager/sessions/{session_id}/stream",
        json={"message": "สวัสดีครับ แนะนำพอร์ตให้หน่อย", "context_refs": ["@Portfolio"]},
    )
    assert stream_resp1.status_code == 200

    # Verify session was automatically renamed from "บทสนทนาใหม่" to user's first prompt
    sess_info = authed_client.get(f"/api/manager/sessions")
    assert any(s["id"] == session_id and "สวัสดีครับ" in s["title"] for s in sess_info.json())

    # Turn 2: history should contain turn 1 messages
    stream_resp2 = authed_client.post(
        f"/api/manager/sessions/{session_id}/stream",
        json={"message": "แล้วความเสี่ยงล่ะ?", "context_refs": []},
    )
    assert stream_resp2.status_code == 200

    # Ensure second turn received turn 1 history
    assert len(recorded_histories) == 2
    assert len(recorded_histories[1]) >= 2  # user message 1 + assistant reply 1



def test_stream_chat_emits_job_dispatched_and_done_payload(authed_client: TestClient, monkeypatch):
    """Verify that SSE stream emits event: job_dispatched and complete done payload."""
    from application.manager.manager_chat_service import ManagerChatService

    async def mock_stream_turn(self, session_id, user_message, context_refs=()):
        # Simulate text stream
        yield "กำลังส่งงานวิเคราะห์หุ้น INTC..."
        # Simulate job dispatch recorded in chat_service
        self._sandbox_adapter # access adapter

    def mock_pop_jobs(self, session_id):
        return [{"job_id": "job-test-123", "ticker": "INTC", "job_type": "equity_intel"}]

    monkeypatch.setattr(ManagerChatService, "stream_turn", mock_stream_turn)
    monkeypatch.setattr(ManagerChatService, "pop_dispatched_jobs", mock_pop_jobs)

    # Create session
    resp = authed_client.post("/api/manager/sessions", json={"title": "เทสต์งานวิจัย"})
    session_id = resp.json()["id"]

    # Stream turn
    stream_resp = authed_client.post(
        f"/api/manager/sessions/{session_id}/stream",
        json={"message": "วิเคราะห์ INTC ด่วน", "context_refs": ["$INTC"]},
    )
    assert stream_resp.status_code == 200
    body = stream_resp.text
    assert "event: text_delta" in body
    assert "event: job_dispatched" in body
    assert "job-test-123" in body
    assert "INTC" in body
    assert "event: done" in body
    assert "status" in body
