"""Unit tests สำหรับ tools/content/youtube_pitcher.py"""
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from schemas.youtube_pitch_schemas import YouTubeContentPitchBatch, YouTubeContentPitchItem
from tools.content.youtube_pitcher import (
    fetch_news_for_pitching,
    generate_youtube_pitches,
    parse_date_filters_from_instruction,
    save_notebooklm_source,
    synthesize_notebooklm_source,
)


def test_parse_date_filters_from_instruction():
    # 1. แท็กครบ from_date และ to_date
    res1 = parse_date_filters_from_instruction("[from_date=2026-07-01, to_date=2026-07-18, lookback_days=17]")
    assert res1["from_date"] == "2026-07-01"
    assert res1["to_date"] == "2026-07-18"
    assert res1["lookback_days"] == 17

    # 2. คำสั่งข้อความภาษาไทย "ย้อนหลัง 14 วัน"
    res2 = parse_date_filters_from_instruction("หาไอเดียทำคลิป YouTube เจาะลึกข่าว ย้อนหลัง 14 วัน กรุณาจัดเต็ม")
    assert res2["lookback_days"] == 14

    # 3. คำสั่งว่าง
    res3 = parse_date_filters_from_instruction("")
    assert res3["lookback_days"] == 7
    assert res3["from_date"] is None


@patch("tools.content.youtube_pitcher.load_store")
@patch("tools.content.youtube_pitcher.get_macro_baselines")
def test_fetch_news_for_pitching_layer1_and_fallback(mock_baselines, mock_load, tmp_path, monkeypatch):
    monkeypatch.setattr("tools.content.youtube_pitcher.VAULT_PATH", tmp_path)
    mock_baselines.invoke.return_value = '{"macro": "ok"}'

    now_iso = datetime.now().isoformat()
    mock_load.return_value = {
        "pending_events": [
            {
                "event_id": "ev-1",
                "canonical_title": "ข่าวธนาคารกลางปรับดอกเบี้ย",
                "comprehensive_summary": "สรุปข่าวชั้น 1",
                "links": ["http://test.com/1"],
                "ingested_at": now_iso,
            }
        ]
    }

    # สร้างไฟล์ใน Layer 2 (30_Knowledge_Base/News/*.md)
    layer2_dir = tmp_path / "30_Knowledge_Base" / "News"
    layer2_dir.mkdir(parents=True)
    md_file = layer2_dir / "2026-07-01_ข่าวเก่า.md"
    md_file.write_text(
        "---\n"
        "title: ข่าวเก่าใน Layer 2 ที่ Store ตัดทิ้งไปแล้ว\n"
        f"date: {(datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')}\n"
        "---\n"
        "# สรุปเนื้อหาข่าวเก่าในอดีต\n"
        "รายละเอียดเนื้อหาเต็ม... https://test.com/layer2\n",
        encoding="utf-8",
    )

    # ทดสอบดึงข่าวโดยกำหนด lookback_days=14 (ย้อนหลังเกิน 7 วัน -> Trigger Layer 2 Fallback)
    candidates, macro_str, is_fallback = fetch_news_for_pitching(lookback_days=14)
    assert is_fallback is True
    assert len(candidates) >= 1
    assert any(c.get("event_id") == "ev-1" for c in candidates)
    assert any("ข่าวเก่า" in c.get("canonical_title", "") for c in candidates)
    assert macro_str == '{"macro": "ok"}'


@patch("tools.content.youtube_pitcher.invoke_structured_llm")
def test_generate_youtube_pitches_success(mock_invoke):
    mock_item = YouTubeContentPitchItem(
        pitch_id="uuid-test",
        working_titles=["คำถามเจาะลึก?", "วิเคราะห์สมมติฐาน", "เตือนภัยตลาด"],
        target_audience="นักลงทุนไทย",
        core_hook="Hook 30 วินาที",
        key_questions_to_answer=["ข้อ 1", "ข้อ 2", "ข้อ 3"],
        research_hypotheses=["สมมติฐาน 1", "สมมติฐาน 2"],
        source_event_ids=["ev-1"],
        source_links=["http://test.com/1"],
        source_titles=["ข่าวธนาคารกลาง"],
        recommended_format="Deep Dive 15m",
        estimated_impact="Impact",
    )
    mock_invoke.return_value = YouTubeContentPitchBatch(
        pitches=[mock_item],
        date_range_summary="7 วัน",
        total_source_events=1,
    )

    batch = generate_youtube_pitches(
        candidates=[{"event_id": "ev-1", "canonical_title": "ข่าวธนาคารกลาง", "links": ["http://test.com/1"]}],
        max_pitches=3,
    )
    assert len(batch.pitches) == 1
    assert batch.pitches[0].working_titles[0] == "คำถามเจาะลึก?"


@patch("tools.content.youtube_pitcher.invoke_structured_llm")
def test_generate_youtube_pitches_retry_and_lenient_fallback(mock_invoke):
    # จำลอง LLM ยกเว้น error (เช่น validation fail ซ้ำ) เพื่อยืนยัน heuristic fallback
    mock_invoke.side_effect = Exception("Mock LLM validation error")

    batch = generate_youtube_pitches(
        candidates=[{"event_id": "ev-1", "canonical_title": "ข่าวตลาดหุ้นผันผวน", "links": ["http://test.com/1"]}],
        max_pitches=3,
    )
    assert len(batch.pitches) == 1
    assert "เจาะลึก:" in batch.pitches[0].working_titles[0]
    assert len(batch.pitches[0].working_titles) == 3


@patch("tools.content.youtube_pitcher.get_llm")
def test_synthesize_notebooklm_source(mock_get_llm):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = "# 📑 สรุปผู้บริหารและแหล่งอ้างอิง\nเนื้อหา Briefing Book ครบ 7 Sections..."
    mock_get_llm.return_value = mock_llm

    pitch = YouTubeContentPitchItem(
        pitch_id="uuid-test",
        working_titles=["1", "2", "3"],
        target_audience="คนดู",
        core_hook="Hook",
        key_questions_to_answer=["Q1", "Q2", "Q3"],
        research_hypotheses=["H1", "H2"],
        source_event_ids=["ev-1"],
        source_links=["http://test.com/1"],
        source_titles=["ข่าว 1"],
        recommended_format="15m",
        estimated_impact="Impact",
    )

    res = synthesize_notebooklm_source(pitch, source_events=[{"event_id": "ev-1", "title": "ข่าว 1"}])
    assert "📑 สรุปผู้บริหารและแหล่งอ้างอิง" in res
    mock_get_llm.assert_called_once_with(
        provider="google",
        model_name="gemini-3.1-flash-lite-preview",
        max_output_tokens=16384,
    )


def test_save_notebooklm_source_with_thai_filename(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.content.youtube_pitcher.VAULT_PATH", tmp_path)
    content = "# Briefing Book เนื้อหาเต็ม"
    title = "วิเคราะห์หุ้นเทคไทยและโลก ปี 2026"

    saved_path_str = save_notebooklm_source(content, title, date_str="2026-07-18")
    saved_path = Path(saved_path_str)

    assert saved_path.exists()
    assert "วิเคราะห์หุ้นเทคไทยและโลก" in saved_path.name
    assert "2026-07-18_" in saved_path.name
    assert saved_path.read_text(encoding="utf-8") == content

    # ทดสอบ collision (_2)
    saved_path_2 = Path(save_notebooklm_source(content, title, date_str="2026-07-18"))
    assert saved_path_2.exists()
    assert "_2.md" in saved_path_2.name


@patch("tools.content.youtube_pitcher.load_store")
@patch("tools.content.youtube_pitcher.get_macro_baselines")
def test_fetch_news_for_pitching_with_youtube_summaries_always_include(mock_baselines, mock_load, tmp_path, monkeypatch):
    monkeypatch.setattr("tools.content.youtube_pitcher.VAULT_PATH", tmp_path)
    mock_baselines.invoke.return_value = '{"macro": "ok"}'
    mock_load.return_value = {"pending_events": []}

    yt_dir = tmp_path / "30_Knowledge_Base" / "YouTube_Summaries"
    yt_dir.mkdir(parents=True)
    md_file = yt_dir / "2026-07-15_youtube_summary.md"
    md_file.write_text(
        "---\n"
        "title: YouTube Insight abc12345678 2026-07-15\n"
        "entity_type: youtube_insight\n"
        "channel: Pi Securities\n"
        "date: 2026-07-15\n"
        "source_url: https://www.youtube.com/watch?v=abc12345678\n"
        "---\n"
        "# Title\n"
        "## ใจความสำคัญ\n"
        "- ตลาดหุ้นไทยฟื้นตัวอย่างแข็งแกร่งด้วยแรงหนุนจากหุ้น Defensive\n"
        "## แนวคิดการลงทุน\n"
        "- เน้นสะสมหุ้นปันผลและกลุ่มโรงพยาบาลเพื่อรับมือความผันผวน\n",
        encoding="utf-8",
    )

    candidates, macro_str, is_fallback = fetch_news_for_pitching(lookback_days=30)
    assert len(candidates) == 1
    c = candidates[0]
    assert c["source_layer"] == "layer2_youtube"
    assert "[YouTube Guru View - Pi Securities] ตลาดหุ้นไทยฟื้นตัวอย่างแข็งแกร่ง" in c["canonical_title"]
    assert "[ใจความสำคัญ]" in c["comprehensive_summary"]
    assert "[แนวคิดการลงทุน]" in c["comprehensive_summary"]
    assert c["links"] == ["https://www.youtube.com/watch?v=abc12345678"]


@patch("tools.content.youtube_pitcher.invoke_structured_llm")
def test_generate_pitches_internal_quota_and_truncation(mock_invoke):
    from tools.content.youtube_pitcher import _generate_pitches_internal

    mock_invoke.return_value = YouTubeContentPitchBatch(pitches=[], date_range_summary="30 วัน", total_source_events=20)

    # สร้าง candidates 15 news items + 10 yt items (โดยให้ long summary ที่ยาว 600 chars เพื่อเช็ค truncation)
    long_summary = "ก" * 600
    candidates = []
    for i in range(15):
        candidates.append({
            "event_id": f"news-{i}",
            "canonical_title": f"News {i}",
            "comprehensive_summary": long_summary,
            "source_layer": "layer1_store",
            "ingested_at": f"2026-07-15T10:{i:02d}:00",
        })
    for j in range(10):
        candidates.append({
            "event_id": f"yt-{j}",
            "canonical_title": f"YT {j}",
            "comprehensive_summary": long_summary,
            "source_layer": "layer2_youtube",
            "ingested_at": f"2026-07-16T10:{j:02d}:00",
        })

    _generate_pitches_internal(candidates, max_pitches=3, instruction="", date_summary="30 วัน")

    mock_invoke.assert_called_once()
    prompt_lines = mock_invoke.call_args[1]["prompt_lines"] if "prompt_lines" in mock_invoke.call_args[1] else mock_invoke.call_args[0][2]
    prompt_text = "\n".join(prompt_lines)

    # ต้องมี news 12 รายการ และ yt 8 รายการ (รวม 20 รายการ)
    assert "จำนวนข้อมูลและบทวิเคราะห์ที่คัดกรองมาทั้งหมด: 20 รายการ (จากทั้งหมด 25 รายการในคลัง)" in prompt_text
    # News truncation: ข่าวล่าสุด (news-14 ถึง news-3) ต้องอยู่ครบ และโดนตัดที่ 250 chars
    assert "news-14" in prompt_text
    assert "news-0" not in prompt_text
    # YT truncation: yt ล่าสุด (yt-9 ถึง yt-2) ต้องอยู่ครบ และโดนตัดที่ 550 chars
    assert "yt-9" in prompt_text
    assert "yt-0" not in prompt_text
    # เช็คความยาวสรุปใน prompt (โดย split ตาม block ของ candidate [news- หรือ [yt-)
    for block in prompt_text.split("\n["):
        if block.startswith("news-") and "   สรุป: " in block:
            summary_part = block.split("   สรุป: ")[1].split("\n")[0]
            assert len(summary_part) <= 250
        elif block.startswith("yt-") and "   สรุป: " in block:
            summary_part = block.split("   สรุป: ")[1].split("\n")[0]
            assert len(summary_part) == 550

