"""ทดสอบ human-in-the-loop approval flow (youtube_pitch) แบบ end-to-end ผ่าน default_run_fn จริง"""
import json
from contextlib import closing
from unittest.mock import MagicMock

from api import state_db
from api.jobs import default_run_fn
from schemas.youtube_pitch_schemas import YouTubeContentPitchBatch, YouTubeContentPitchItem


def _fake_pitches_batch():
    item = YouTubeContentPitchItem(
        pitch_id="pitch-abc",
        working_titles=["หัวข้อคำถาม?", "หัวข้อวิเคราะห์", "หัวข้อเตือนภัย"],
        target_audience="นักลงทุนไทย",
        core_hook="Hook เปิดคลิป",
        key_questions_to_answer=["คำถาม 1", "คำถาม 2", "คำถาม 3"],
        research_hypotheses=["สมมติฐาน 1", "สมมติฐาน 2"],
        source_event_ids=["ev-1", "ev-2"],
        source_links=["https://example.com/news", "https://second.example.com/news"],
        source_titles=["ข่าวเศรษฐกิจสำคัญ", "ข่าวยืนยันจากอีกสำนัก"],
        recommended_format="Deep Dive 15m",
        estimated_impact="ผลกระทบสูง",
    )
    return YouTubeContentPitchBatch(
        pitches=[item],
        date_range_summary="ย้อนหลัง 7 วัน",
        total_source_events=2,
    )


def _fake_candidates():
    return [
        {
            "event_id": "ev-1", "canonical_title": "ข่าวเศรษฐกิจสำคัญ",
            "links": ["https://example.com/news"], "publisher": "Reuters",
            "published_at": "2026-07-23", "verification_status": "verified",
        },
        {
            "event_id": "ev-2", "canonical_title": "ข่าวยืนยันจากอีกสำนัก",
            "links": ["https://second.example.com/news"], "publisher": "Bloomberg",
            "published_at": "2026-07-23", "verification_status": "verified",
        },
    ]


def test_default_run_fn_youtube_pitch_flow_pauses_for_approval(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.content.youtube_pitcher.fetch_news_for_pitching", lambda **kwargs: (_fake_candidates(), "macro_str", False))
    monkeypatch.setattr("tools.content.youtube_pitcher.generate_youtube_pitches", lambda **kwargs: _fake_pitches_batch())
    try:
        import agents.youtube_pitch_flow as flow_mod
        monkeypatch.setattr(flow_mod, "fetch_news_for_pitching", lambda **kwargs: (_fake_candidates(), "macro_str", False))
        monkeypatch.setattr(flow_mod, "generate_youtube_pitches", lambda **kwargs: _fake_pitches_batch())
    except ImportError:
        pass
    monkeypatch.setenv("CHECKPOINT_DB_PATH", str(tmp_path / "checkpoints.sqlite"))
    monkeypatch.setenv("WEBUI_STATE_DB_PATH", str(tmp_path / "webui_state.sqlite"))
    monkeypatch.setattr("agents.research_swarm_flow.generate_manager_summary", lambda instruction, deliverables: None)

    conn = state_db.get_connection(str(tmp_path / "webui_state.sqlite"))
    state_db.create_job(conn, "job-pitch-1", "thread-pitch-1", "card-1", "key-1", "หาไอเดียทำคลิป YouTube", status="running", flow="youtube_pitch")
    conn.close()

    default_run_fn(job_id="job-pitch-1", thread_id="thread-pitch-1", instruction="หาไอเดียทำคลิป YouTube", flow="youtube_pitch", resume_value=None)

    conn = state_db.get_connection(str(tmp_path / "webui_state.sqlite"))
    job = state_db.get_job(conn, "job-pitch-1")
    conn.close()

    assert job["status"] == "awaiting_approval"
    payload = json.loads(job["interrupt_payload"])
    assert payload["type"] == "youtube_pitch_approval"
    assert payload["pitches"][0]["pitch_id"] == "pitch-abc"
    assert payload["pitches"][0]["working_titles"][0] == "หัวข้อคำถาม?"


def test_default_run_fn_youtube_pitch_flow_resumes_and_completes(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.content.youtube_pitcher.fetch_news_for_pitching", lambda **kwargs: (_fake_candidates(), "macro_str", False))
    class DummyResult:
        content = "# Briefing Book เนื้อหาเต็ม"
        quality_report = None
        override_audit = None
    dummy = DummyResult()
    monkeypatch.setattr("tools.content.youtube_pitcher.synthesize_notebooklm_source", lambda *args, **kwargs: dummy)
    mock_art = type('MockArt', (), {'path': 'C:/vault/saved_pitch.md'})()
    monkeypatch.setattr("tools.content.briefing_artifacts.save_briefing_artifact", lambda *args, **kwargs: mock_art)
    try:
        import agents.youtube_pitch_flow as flow_mod
        monkeypatch.setattr(flow_mod, "fetch_news_for_pitching", lambda **kwargs: (_fake_candidates(), "macro_str", False))
        monkeypatch.setattr(flow_mod, "generate_youtube_pitches", lambda **kwargs: _fake_pitches_batch())
        monkeypatch.setattr(flow_mod, "synthesize_notebooklm_source", lambda *args, **kwargs: dummy)
        monkeypatch.setattr(flow_mod, "save_briefing_artifact", lambda *args, **kwargs: mock_art)
    except ImportError:
        pass

    monkeypatch.setenv("CHECKPOINT_DB_PATH", str(tmp_path / "checkpoints.sqlite"))
    monkeypatch.setenv("WEBUI_STATE_DB_PATH", str(tmp_path / "webui_state.sqlite"))
    monkeypatch.setattr("agents.research_swarm_flow.generate_manager_summary", lambda instruction, deliverables: None)

    conn = state_db.get_connection(str(tmp_path / "webui_state.sqlite"))
    state_db.create_job(conn, "job-pitch-2", "thread-pitch-2", "card-2", "key-2", "หาไอเดียทำคลิป YouTube", status="running", flow="youtube_pitch")
    conn.close()

    # รอบ 1: วิ่งจนถึง gate แล้ว pause
    default_run_fn(job_id="job-pitch-2", thread_id="thread-pitch-2", instruction="หาไอเดียทำคลิป YouTube", flow="youtube_pitch", resume_value=None)

    # รอบ 2: Resume โดยส่ง approved_pitch_ids เข้าไป
    resume_value = {"approved_pitch_ids": ["pitch-abc"]}
    default_run_fn(job_id="job-pitch-2", thread_id="thread-pitch-2", instruction="หาไอเดียทำคลิป YouTube", flow="youtube_pitch", resume_value=resume_value)

    conn = state_db.get_connection(str(tmp_path / "webui_state.sqlite"))
    logs = state_db.get_job_reply_logs(conn, "job-pitch-2")
    conn.close()

    synth_logs = [row for row in logs if row["node_name"] == "synthesize_notebooklm"]
    assert len(synth_logs) > 0
    assert "บันทึก Briefing Book สำเร็จ: C:/vault/saved_pitch.md" in synth_logs[0]["content"]


def test_job_queue_resume_youtube_pitch_flow(tmp_path, monkeypatch):
    """ทดสอบ JobQueue._run_job ของ flow youtube_pitch ว่าเปลี่ยนสถานะเป็น done เมื่อ run เสร็จจริง"""
    import asyncio
    from api.jobs import JobQueue

    monkeypatch.setattr("tools.content.youtube_pitcher.fetch_news_for_pitching", lambda **kwargs: (_fake_candidates(), "macro_str", False))
    class DummyResult:
        content = "# Briefing Book เนื้อหาเต็ม"
        quality_report = None
        override_audit = None
    dummy = DummyResult()
    monkeypatch.setattr("tools.content.youtube_pitcher.synthesize_notebooklm_source", lambda *args, **kwargs: dummy)
    mock_art2 = type('MockArt', (), {'path': 'C:/vault/saved_pitch.md'})()
    monkeypatch.setattr("tools.content.briefing_artifacts.save_briefing_artifact", lambda *args, **kwargs: mock_art2)
    try:
        import agents.youtube_pitch_flow as flow_mod
        monkeypatch.setattr(flow_mod, "fetch_news_for_pitching", lambda **kwargs: (_fake_candidates(), "macro_str", False))
        monkeypatch.setattr(flow_mod, "generate_youtube_pitches", lambda **kwargs: _fake_pitches_batch())
        monkeypatch.setattr(flow_mod, "synthesize_notebooklm_source", lambda *args, **kwargs: dummy)
        monkeypatch.setattr(flow_mod, "save_briefing_artifact", lambda *args, **kwargs: mock_art2)
    except ImportError:
        pass

    monkeypatch.setenv("CHECKPOINT_DB_PATH", str(tmp_path / "checkpoints.sqlite"))
    monkeypatch.setenv("WEBUI_STATE_DB_PATH", str(tmp_path / "webui_state.sqlite"))
    monkeypatch.setattr("agents.research_swarm_flow.generate_manager_summary", lambda instruction, deliverables: None)

    db_path = str(tmp_path / "webui_state.sqlite")
    queue = JobQueue(run_fn=default_run_fn, db_path=db_path)
    job_id = queue.dispatch("หาไอเดียทำคลิป YouTube", card_id="card-2", flow="youtube_pitch")

    async def _drive():
        await queue._run_job(job_id)

    asyncio.run(_drive())

    conn = state_db.get_connection(db_path)
    job = state_db.get_job(conn, job_id)
    conn.close()
    assert job["status"] == "awaiting_approval"

    # สั่ง resume
    with closing(state_db.get_connection(db_path)) as conn:
        import json
        state_db.claim_job_resume(conn, job_id=job_id, resume_value_json=json.dumps({"approved_pitch_ids": ["pitch-abc"]}))
    queue.enqueue(job_id)

    conn = state_db.get_connection(db_path)
    job = state_db.get_job(conn, job_id)
    conn.close()
    assert job["status"] == "queued"

    asyncio.run(_drive())

    conn = state_db.get_connection(db_path)
    job = state_db.get_job(conn, job_id)
    conn.close()
    assert job["status"] == "done"
