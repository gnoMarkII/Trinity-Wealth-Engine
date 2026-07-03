import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import json

from tools.archivist.writer import save_memory, write_raw_markdown
from tools.archivist.core import VAULT_PATH
import tools.archivist.writer as writer_module

@pytest.fixture
def test_vault(tmp_path, monkeypatch):
    vault_dir = tmp_path / "memories"
    vault_dir.mkdir(exist_ok=True)
    import tools.archivist.core as core_m
    import tools.archivist.writer as writer_m
    import tools.archivist.indexer as indexer_m
    import tools.archivist.search as search_m
    import tools.archivist.linter as linter_m
    import tools.archivist.parser as parser_m
    for mod in [core_m, writer_m, indexer_m, search_m, linter_m, parser_m]:
        monkeypatch.setattr(mod, "VAULT_PATH", vault_dir)
    return vault_dir

def test_save_memory_append(test_vault):
    # Setup an existing file
    target_dir = test_vault / "30_Knowledge_Base" / "Stocks"
    target_dir.mkdir(parents=True)
    existing_file = target_dir / "Test Memory.md"
    existing_file.write_text("---\ntitle: Test Memory\ntags:\n  - old_tag\naliases:\n  - old_alias\n---\nOld Content", encoding="utf-8")
    
    res = save_memory.func(
        title="Test Memory",
        content="New Content",
        folder_path="30_Knowledge_Base/Stocks",
        tags=["new_tag"],
        entity_type="Concept",
        aliases=["new_alias"],
        linked_files=["Linked_Doc"]
    )
    
    assert "เพิ่มข้อมูลสำเร็จ (append)" in res
    content = existing_file.read_text(encoding="utf-8")
    assert "old_tag" in content
    assert "new_tag" in content
    assert "old_alias" in content
    assert "new_alias" in content
    assert "Old Content" in content
    assert "New Content" in content
    assert "[[Linked_Doc]]" in content

def test_write_raw_markdown_stocks_and_date(test_vault):
    raw_md = "---\ntitle: Test AAPL\nentity_type: test\nticker: AAPL\ndate: 2026-06-20\n---\n# Content"
    res = write_raw_markdown.func(
        content=raw_md,
        folder_path="30_Knowledge_Base/Stocks",
        filename="Test AAPL"
    )
    assert "บันทึกสำเร็จ" in res
    
    # It should inject ticker subfolder
    saved_file = test_vault / "30_Knowledge_Base" / "Stocks" / "AAPL" / "Test AAPL.md"
    assert saved_file.exists()
    
    # It should create stock stub
    stub_file = test_vault / "30_Knowledge_Base" / "Stocks" / "AAPL" / "AAPL.md"
    assert stub_file.exists()
    stub_content = stub_file.read_text(encoding="utf-8")
    assert "entity_type: stock_entity" in stub_content
    assert "ticker: AAPL" in stub_content

    # Call it again to trigger early return in stub creation
    write_raw_markdown.func(
        content=raw_md,
        folder_path="30_Knowledge_Base/Stocks",
        filename="Test AAPL"
    )

def test_write_raw_markdown_daily_snapshots(test_vault):
    raw_md = "---\ntitle: Snapshot\ndate: 2026-06-20\n---\n# Content"
    res = write_raw_markdown.func(
        content=raw_md,
        folder_path="30_Knowledge_Base/Daily_Snapshots",
        filename="Snapshot"
    )
    assert "บันทึกสำเร็จ" in res
    
    # It should NOT inject date subfolder
    saved_file = test_vault / "30_Knowledge_Base" / "Daily_Snapshots" / "Snapshot.md"
    assert saved_file.exists()

def test_write_raw_markdown_news_and_youtube_flattened(test_vault):
    # News shouldn't create publisher folder
    raw_news = "---\npublisher: Reuters\n---\n# Content"
    write_raw_markdown.func(content=raw_news, folder_path="30_Knowledge_Base/News", filename="News1")
    assert (test_vault / "30_Knowledge_Base" / "News" / "News1.md").exists()
    
    # YouTube shouldn't create channel folder
    raw_yt = "---\nchannel: FINNOMENA\n---\n# Content"
    write_raw_markdown.func(content=raw_yt, folder_path="30_Knowledge_Base/YouTube_Summaries", filename="YT1")
    assert (test_vault / "30_Knowledge_Base" / "YouTube_Summaries" / "YT1.md").exists()

import pytest

@pytest.mark.skip(reason="Temporarily disabled per user request")
def test_write_raw_markdown_youtube_canvas(test_vault):
    raw_md = """---
title: YT Insights
entity_type: youtube_insight
video_id: dQw4w9WgXcQ
source_url: https://youtube.com/watch?v=dQw4w9WgXcQ
---
# Main
## ใจความสำคัญ
- ข้อ 1
- ข้อ 2
## แนวคิดการลงทุน
- แนวคิด 1
## ตัวเลขสำคัญทางเศรษฐกิจ
- GDP 5%
## เศรษฐกิจมหภาค
### ทั่วไป
- ข้อมูลรวม
### USA
- US data
## หุ้นและสินทรัพย์
- [[AAPL]] (Apple Inc) น่าสนใจ
- ไม่เจอ ticker นี้ใน vault
## ความเสี่ยง
- ความเสี่ยงสูงมาก
"""
    # Create the AAPL stub so it triggers existing file path
    aapl_dir = test_vault / "30_Knowledge_Base" / "Stocks" / "AAPL"
    aapl_dir.mkdir(parents=True)
    (aapl_dir / "AAPL.md").write_text("stub", encoding="utf-8")

    res = write_raw_markdown.func(
        content=raw_md,
        folder_path="30_Knowledge_Base/Videos",
        filename="YT_Insight"
    )
    assert "บันทึกสำเร็จ" in res
    
    canvas_file = test_vault / "30_Knowledge_Base" / "Videos" / "YT_Insight.canvas"
    assert canvas_file.exists()
    
    canvas_data = json.loads(canvas_file.read_text(encoding="utf-8"))
    nodes = canvas_data["nodes"]
    assert len(nodes) > 0
    
    # Check if nodes contain "ใจความสำคัญ", "USA", "AAPL", "ความเสี่ยง"
    texts = [n.get("text", "") for n in nodes]
    joined_texts = " ".join(texts)
    assert "ใจความสำคัญ" in joined_texts
    assert "USA" in joined_texts
    assert "ความเสี่ยง" in joined_texts

def test_create_youtube_canvas_exception(test_vault, monkeypatch):
    raw_md = "---\ntitle: YT\nentity_type: youtube_insight\n---\nContent"
    # force exception
    monkeypatch.setattr("tools.archivist.writer._parse_h2_sections", MagicMock(side_effect=Exception("parse error")))
    res = write_raw_markdown.func(
        content=raw_md,
        folder_path="30_Knowledge_Base/Videos",
        filename="YT_Insight_Fail"
    )
    assert "บันทึกสำเร็จ" in res
    # Canvas won't exist because it failed, but write_raw_markdown shouldn't crash
    canvas_file = test_vault / "30_Knowledge_Base" / "Videos" / "YT_Insight_Fail.canvas"
    assert not canvas_file.exists()
