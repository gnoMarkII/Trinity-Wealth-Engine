"""Unit tests สำหรับ search_youtube_insights tool"""
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import pytest

from tools.knowledge.search_youtube_insights import (
    extract_first_bullet_of_key_takeaways,
    extract_sections,
    get_channel_with_fallback,
    get_display_title,
    search_youtube_insights,
)


def test_get_channel_with_fallback():
    # 1. มีใน frontmatter
    fm_content = "---\ntitle: test\nchannel: Pi Securities\n---\nbody"
    assert get_channel_with_fallback(fm_content) == "Pi Securities"

    # 2. ไม่มีใน frontmatter แต่มีในบรรทัด > แหล่งที่มา:
    alt_content = "---\ntitle: test\n---\n# Title\n> แหล่งที่มา: https://yt.com | ช่อง: Finnomena | วิธีสกัด: LLM"
    assert get_channel_with_fallback(alt_content) == "Finnomena"

    # 3. ไม่มีใน frontmatter แต่มีบรรทัด ช่อง:
    alt_content2 = "---\ntitle: test\n---\n# Title\nช่อง: Bualuang Securities\nเนื้อหา"
    assert get_channel_with_fallback(alt_content2) == "Bualuang Securities"

    # 4. ไม่มีเลย -> ไม่ระบุช่อง
    empty_content = "---\ntitle: test\n---\n# Title\nไม่มีข้อมูลช่อง"
    assert get_channel_with_fallback(empty_content) == "ไม่ระบุช่อง"


def test_extract_first_bullet_of_key_takeaways():
    content = """---
title: test
---
## ใจความสำคัญ
- **ตลาดหุ้นสหรัฐฯ** เผชิญแรงเทขายทำกำไรในกลุ่มเซมิคอนดักเตอร์และเทคโนโลยี หลังจากปรับตัวขึ้นมาสูงมาก (Year-to-date บางตัวบวกกว่า 200%)
- นักลงทุนเริ่มกังวลเรื่องเงินเฟ้อ
## แนวคิดการลงทุน
- AI Bubble
"""
    bullet = extract_first_bullet_of_key_takeaways(content, max_chars=50)
    assert "ตลาดหุ้นสหรัฐฯ" in bullet
    assert len(bullet) <= 53  # max_chars + len(...)


def test_extract_sections():
    # มีครบและไม่ครบ
    content = """---
title: test
---
## ใจความสำคัญ
- สรุป 1
## แนวคิดการลงทุน
- แนวคิด 1
- แนวคิด 2
"""
    sections = extract_sections(content, ["ใจความสำคัญ", "แนวคิดการลงทุน", "หุ้นและสินทรัพย์"])
    assert sections["ใจความสำคัญ"] == "- สรุป 1"
    assert sections["แนวคิดการลงทุน"] == "- แนวคิด 1\n- แนวคิด 2"
    assert sections["หุ้นและสินทรัพย์"] == ""  # ไม่มี section นี้ต้องคืน string ว่าง ไม่ error


def test_get_display_title():
    content = """---
title: YouTube Insight 1HVkvG8WWFs 2026-07-09
---
## ใจความสำคัญ
- ตลาดหุ้นไทยยังคงยืนได้ด้วยหุ้นกลุ่ม Defensive
"""
    title = get_display_title(content, "Pi Securities", fallback_title="Fallback Title")
    assert title.startswith("[Pi Securities] ตลาดหุ้นไทยยังคงยืนได้ด้วยหุ้นกลุ่ม Defensive")


def test_search_youtube_insights_tool(tmp_path, monkeypatch):
    # Mock VAULT_PATH ไปที่ tmp_path
    import importlib
    mod = importlib.import_module("tools.knowledge.search_youtube_insights")
    monkeypatch.setattr(mod, "VAULT_PATH", tmp_path)
    summaries_dir = tmp_path / "30_Knowledge_Base" / "YouTube_Summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    sample_md = summaries_dir / f"YouTube Insight abc12345678 {today_str}.md"
    sample_md.write_text(
        f"""---
title: YouTube Insight abc12345678 {today_str}
entity_type: youtube_insight
channel: Pi Securities
video_id: abc12345678
source_url: https://www.youtube.com/watch?v=abc12345678
date: {today_str}
tags: [youtube, ai, dividend]
---
# Title
## ใจความสำคัญ
- หุ้นปันผลไทยน่าสนใจมากในไตรมาสนี้
## แนวคิดการลงทุน
- กลยุทธ์หุ้นปันผลรับมือตลาดผันผวน
## หุ้นและสินทรัพย์
- [[BDMS]]
- [[KTC]]
""",
        encoding="utf-8",
    )

    # 1. ค้นหาทั่วไป (ปล่อยว่าง query)
    res = search_youtube_insights.invoke({"query": "", "channel": "", "lookback_days": 30})
    assert "Pi Securities" in res
    assert "หุ้นปันผลไทยน่าสนใจมากในไตรมาสนี้" in res
    assert "BDMS" in res

    # 2. ค้นหา query ตรง
    res_query = search_youtube_insights.invoke({"query": "ปันผล", "channel": "", "lookback_days": 30})
    assert "Pi Securities" in res_query

    # 3. ค้นหา query ไม่ตรง
    res_empty = search_youtube_insights.invoke({"query": "คริปโตที่ไม่มีจริง", "channel": "", "lookback_days": 30})
    assert "ไม่พบคลิป YouTube" in res_empty
