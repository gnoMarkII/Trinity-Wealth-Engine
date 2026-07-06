import os
import pytest
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

from tools.knowledge.youtube_monitor import load_recent_youtube_insights
from agents.macro_economist_agent import create_macro_economist


@pytest.fixture
def temp_youtube_vault(tmp_path, monkeypatch):
    vault_dir = tmp_path / "memories"
    summaries_dir = vault_dir / "30_Knowledge_Base" / "YouTube_Summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(vault_dir))
    return summaries_dir


def test_load_insights_within_lookback(temp_youtube_vault):
    ref_date = date(2026, 7, 6)
    file_path = temp_youtube_vault / "YouTube Insight 1uGxMU9K4Wk 2026-06-30.md"
    content = """---
title: YouTube Insight 1uGxMU9K4Wk 2026-06-30
entity_type: youtube_insight
channel: FINNOMENA
video_id: 1uGxMU9K4Wk
date: 2026-06-30
---
## ใจความสำคัญ
- มุมมองการลงทุนใน EM มีความท้าทาย
- ปรับลดน้ำหนักการลงทุนในตลาดจีน
"""
    file_path.write_text(content, encoding="utf-8")

    res = load_recent_youtube_insights(lookback_days=14, reference_date=ref_date)
    assert "[FINNOMENA: 1uGxMU9K4Wk | 2026-06-30]" in res
    assert "- มุมมองการลงทุนใน EM มีความท้าทาย" in res


def test_skip_insights_outside_lookback(temp_youtube_vault):
    ref_date = date(2026, 7, 6)
    file_path = temp_youtube_vault / "YouTube Insight old 2026-05-01.md"
    content = """---
title: YouTube Insight old 2026-05-01
entity_type: youtube_insight
channel: OLD_CHAN
video_id: old123
date: 2026-05-01
---
## ใจความสำคัญ
- เก่าเกินไป
"""
    file_path.write_text(content, encoding="utf-8")

    res = load_recent_youtube_insights(lookback_days=14, reference_date=ref_date)
    assert res == ""


def test_pre_aggregation_condensation_bullets(temp_youtube_vault):
    ref_date = date(2026, 7, 6)
    file_path = temp_youtube_vault / "YouTube Insight bullets 2026-06-30.md"
    content = """---
title: YouTube Insight bullets
entity_type: youtube_insight
channel: WEALTHION
video_id: wth001
date: 2026-06-30
---
## ใจความสำคัญ
- ข้อที่ 1
- ข้อที่ 2
- ข้อที่ 3
- ข้อที่ 4 ต้องถูกตัดทิ้งไม่เกิน 3 bullets
## แนวคิดการลงทุน
- ไม่ควรถูกดึงมา
"""
    file_path.write_text(content, encoding="utf-8")

    res = load_recent_youtube_insights(lookback_days=14, max_bullets_per_clip=3, reference_date=ref_date)
    assert "[WEALTHION: wth001 | 2026-06-30]" in res
    assert "- ข้อที่ 1" in res
    assert "- ข้อที่ 2" in res
    assert "- ข้อที่ 3" in res
    assert "- ข้อที่ 4" not in res
    assert "แนวคิดการลงทุน" not in res


def test_per_file_error_resilience(temp_youtube_vault):
    ref_date = date(2026, 7, 6)
    valid_file = temp_youtube_vault / "valid.md"
    valid_file.write_text("""---
entity_type: youtube_insight
channel: VALID
video_id: val123
date: 2026-06-30
---
## ใจความสำคัญ
- ข้อความปกติ
""", encoding="utf-8")

    corrupt_file = temp_youtube_vault / "corrupt.md"
    # Write invalid content or let's make it cause an error or invalid encoding
    corrupt_file.write_bytes(b"\x80\x81\x82 invalid utf8")

    res = load_recent_youtube_insights(lookback_days=14, reference_date=ref_date)
    assert "[VALID: val123 | 2026-06-30]" in res
    assert "- ข้อความปกติ" in res


def test_max_chars_truncation_at_file_boundary(temp_youtube_vault):
    ref_date = date(2026, 7, 6)
    # Clip 1 (Newer)
    f1 = temp_youtube_vault / "clip1.md"
    f1.write_text("""---
entity_type: youtube_insight
channel: CHAN1
video_id: vid1
date: 2026-07-02
---
## ใจความสำคัญ
- ข้อความคลิปที่ 1 ยาวประมาณเจ็ดสิบตัวอักษรเพื่อใช้ทดสอบการตัดทอนที่ขอบไฟล์
""", encoding="utf-8")

    # Clip 2 (Older)
    f2 = temp_youtube_vault / "clip2.md"
    f2.write_text("""---
entity_type: youtube_insight
channel: CHAN2
video_id: vid2
date: 2026-07-01
---
## ใจความสำคัญ
- ข้อความคลิปที่ 2 จะต้องถูกตัดทิ้งทั้งหมดเมื่อรวมแล้วความยาวเกินค่าโควตา
""", encoding="utf-8")

    # Set max_chars small enough to allow Clip 1 but exclude Clip 2
    res = load_recent_youtube_insights(lookback_days=14, max_chars=130, reference_date=ref_date)
    assert "[CHAN1: vid1 | 2026-07-02]" in res
    assert "[CHAN2: vid2 | 2026-07-01]" not in res
    assert "... [ตัดทอน — แสดงเฉพาะคลิปที่ใหม่ที่สุด ไม่เกิน 130 ตัวอักษร]" in res


def test_empty_vault_graceful(tmp_path, monkeypatch):
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(tmp_path / "non_existent"))
    res = load_recent_youtube_insights()
    assert res == ""


def test_missing_channel_frontmatter_fallback(temp_youtube_vault):
    ref_date = date(2026, 7, 6)
    file_path = temp_youtube_vault / "YouTube Insight -lJFTc__caM 2026-06-30.md"
    content = """---
title: YouTube Insight -lJFTc__caM 2026-06-30
entity_type: youtube_insight
channel: None
video_id: -lJFTc__caM
date: 2026-06-30
---
## ใจความสำคัญ
- ไฟล์จริงที่ channel เป็น None ต้องแสดง Unknown Channel
"""
    file_path.write_text(content, encoding="utf-8")

    res = load_recent_youtube_insights(lookback_days=14, reference_date=ref_date)
    assert "[Unknown Channel: -lJFTc__caM | 2026-06-30]" in res
    assert "None:" not in res


def test_economist_skill_prompt_contains_guardrail_rules():
    skill_path = Path(__file__).resolve().parents[3] / "prompts" / "skills" / "economist" / "SKILL.md"
    content = skill_path.read_text(encoding="utf-8")
    assert "Primary Signal" in content
    assert "Supporting & Tail Risk Evidence" in content
    assert "ห้ามใช้ความเห็นนักวิเคราะห์หักล้างสถิติจริง" in content
    assert "sources_summary" in content
    assert "Finnomena" in content


@patch("agents.macro_economist_agent.get_macro_baselines")
@patch("agents.macro_economist_agent.generate_news_radar_daily")
@patch("tools.knowledge.youtube_monitor.load_recent_youtube_insights")
def test_economist_context_injection_success(mock_yt, mock_news, mock_base):
    mock_base.invoke.return_value = "Baseline Data"
    mock_news.invoke.return_value = "News Data"
    mock_yt.return_value = "[FINNOMENA: vid1 | 2026-06-30]\n- YouTube signal"

    mock_model = MagicMock()
    mock_structured = MagicMock()
    mock_model.with_structured_output.return_value = mock_structured

    from schemas.macro_schemas import NarrativeContext
    mock_res = MagicMock()
    mock_res.model_dump_json.return_value = '{"market_sentiment": "neutral"}'
    mock_structured.invoke.return_value = mock_res

    runner = create_macro_economist(mock_model)
    runner.invoke({})

    call_args = mock_structured.invoke.call_args[0][0]
    user_msg = call_args[1]["content"]
    assert "=== Baseline ===\nBaseline Data" in user_msg
    assert "=== News ===\nNews Data" in user_msg
    assert "=== YouTube Analyst Insights ===\n[FINNOMENA: vid1 | 2026-06-30]\n- YouTube signal" in user_msg


@patch("agents.macro_economist_agent.get_macro_baselines")
@patch("agents.macro_economist_agent.generate_news_radar_daily")
@patch("tools.knowledge.youtube_monitor.load_recent_youtube_insights")
def test_economist_context_injection_error(mock_yt, mock_news, mock_base):
    mock_base.invoke.return_value = "Baseline Data"
    mock_news.invoke.return_value = "News Data"
    mock_yt.side_effect = Exception("Vault unreadable")

    mock_model = MagicMock()
    mock_structured = MagicMock()
    mock_model.with_structured_output.return_value = mock_structured

    mock_res = MagicMock()
    mock_res.model_dump_json.return_value = '{"market_sentiment": "neutral"}'
    mock_structured.invoke.return_value = mock_res

    runner = create_macro_economist(mock_model)
    runner.invoke({})

    call_args = mock_structured.invoke.call_args[0][0]
    user_msg = call_args[1]["content"]
    assert "=== YouTube Analyst Insights ===\nError fetching YouTube insights: Vault unreadable" in user_msg
