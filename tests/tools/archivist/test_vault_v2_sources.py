"""Tests for Vault V2 News & YouTube sources integration (T06).

Verifies that News articles and YouTube insights:
1. Are routed to YYYY/MM subdirectories under layout_version >= 2.
2. Maintain deduplication checks across both V1 flat and V2 nested folders.
3. Are discoverable by all readers (pitcher, monitor, search, references).
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tools.archivist.writer import write_raw_markdown
from tools.knowledge.article import _is_url_already_processed
from tools.knowledge.youtube import _find_existing_insight
from tools.knowledge.youtube_monitor import load_recent_youtube_insights
from tools.knowledge.search_youtube_insights import search_youtube_insights
from tools.macro.content_references import recent_youtube_references
from tools.macro.news_radar import _is_url_fetched


def _setup_vault(tmp_path: Path, layout_version: int) -> Path:
    vault = tmp_path / f"vault_v{layout_version}"
    sys_dir = vault / ".system"
    sys_dir.mkdir(parents=True, exist_ok=True)
    (sys_dir / "vault_config.json").write_text(
        json.dumps({"layout_version": layout_version}), encoding="utf-8"
    )
    (vault / "30_Knowledge_Base" / "News").mkdir(parents=True, exist_ok=True)
    (vault / "30_Knowledge_Base" / "YouTube_Summaries").mkdir(parents=True, exist_ok=True)
    return vault


def test_news_v2_routing_and_deduplication(tmp_path: Path, monkeypatch) -> None:
    vault = _setup_vault(tmp_path, layout_version=2)
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(vault))

    url = "https://example.com/fed-rate-decision-aug-2026"
    content = f"""---
title: Fed Holds Rates Steady in August
entity_type: article_note
published_at: 2026-08-15
date: 2026-08-15
source_url: {url}
tags: [macro, fed, interest_rates]
---

# Fed Holds Rates Steady in August

The Federal Reserve decided to maintain current interest rates.
Source: {url}
"""
    result = write_raw_markdown.invoke({
        "content": content,
        "folder_path": "30_Knowledge_Base/News",
        "filename": "Fed_Holds_Rates_Steady",
    })

    # Verify file is written to 30_Knowledge_Base/News/2026/08/
    expected_dir = vault / "30_Knowledge_Base" / "News" / "2026" / "08"
    assert expected_dir.exists(), f"Expected directory {expected_dir} to exist"
    written_files = list(expected_dir.glob("*.md"))
    assert len(written_files) == 1
    assert "Fed_Holds_Rates_Steady" in written_files[0].name

    # Verify deduplication functions find the nested file
    assert _is_url_already_processed(url) is True
    assert _is_url_already_processed("https://example.com/unseen-article") is False
    assert _is_url_fetched(url) is True
    assert _is_url_fetched("https://example.com/unseen-article") is False


def test_youtube_v2_routing_and_deduplication(tmp_path: Path, monkeypatch) -> None:
    vault = _setup_vault(tmp_path, layout_version=2)
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(vault))

    video_id = "abc123XYZ09"
    source_url = f"https://www.youtube.com/watch?v={video_id}"
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    year, month = today_str[:4], today_str[5:7]

    content = f"""---
title: AI Semiconductor Industry Analysis
entity_type: youtube_insight
video_id: {video_id}
source_url: {source_url}
channel: Tech Insights
date: {today_str}
published_at: {today_str}
tags: [youtube, ai, semiconductors]
---

# AI Semiconductor Industry Analysis

> แหล่งที่มา: {source_url} | ช่อง: Tech Insights

## ใจความสำคัญ
- สรุปภาพรวมความต้องการชิป AI ยังเติบโตสูง
- [[NVDA]] และ [[TSM]] เป็นผู้นำในห่วงโซ่อุปทาน
"""
    result = write_raw_markdown.invoke({
        "content": content,
        "folder_path": "30_Knowledge_Base/YouTube_Summaries",
        "filename": f"YouTube_Insight_{video_id}",
    })

    # Verify file is written to 30_Knowledge_Base/YouTube_Summaries/YYYY/MM/
    expected_dir = vault / "30_Knowledge_Base" / "YouTube_Summaries" / year / month
    assert expected_dir.exists()
    written_files = list(expected_dir.glob("*.md"))
    assert len(written_files) == 1

    # Verify deduplication
    existing = _find_existing_insight(video_id)
    assert existing is not None
    assert existing.resolve() == written_files[0].resolve()
    assert _find_existing_insight("nonexistent99") is None

    # Verify search reader
    search_res = search_youtube_insights.invoke({"query": "Semiconductor", "lookback_days": 30})
    assert video_id in search_res or "AI Semiconductor" in search_res

    # Verify monitor reader
    monitor_res = load_recent_youtube_insights(lookback_days=30)
    assert video_id in monitor_res or "AI Semiconductor" in monitor_res

    # Verify content references reader
    refs = recent_youtube_references(lookback_days=30)
    assert len(refs) >= 1
    assert any(r.get("reference_id") == f"youtube_{video_id}" for r in refs)


def test_v1_legacy_writer_fails_closed(tmp_path: Path, monkeypatch) -> None:
    vault = _setup_vault(tmp_path, layout_version=1)
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(vault))

    url = "https://example.com/v1-legacy-news"
    content = f"""---
title: Legacy News
entity_type: article_note
date: 2026-07-10
source_url: {url}
---
Legacy content
"""
    with pytest.raises(RuntimeError, match="Legacy V1 Markdown writes are disabled"):
        write_raw_markdown.invoke({
            "content": content,
            "folder_path": "30_Knowledge_Base/News",
            "filename": "Legacy_News",
        })

    # V1 remains readable/migratable, but production writers must not create
    # a second legacy write path or mutate the V1 tree.
    v1_file = vault / "30_Knowledge_Base" / "News" / "2026-07-10 Legacy_News.md"
    assert not v1_file.exists()
