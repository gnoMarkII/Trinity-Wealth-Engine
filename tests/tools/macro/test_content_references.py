from datetime import datetime, timezone

from tools.macro.content_references import news_references_from_radar, recent_youtube_references


def test_news_references_are_extracted_from_radar_rows():
    radar = """| เลือก | หัวข้อข่าว | ความใหม่ | แหล่งข่าว |
|:---:|---|---|---|
| [ ] | [Central bank outlook](https://example.com/news) | Age: 4h | Example News (Sources: 2) |
"""

    references = news_references_from_radar(radar)

    assert references[0]["kind"] == "news"
    assert references[0]["url"] == "https://example.com/news"
    assert references[0]["age_hours"] == 4


def test_youtube_references_are_loaded_from_local_insights(tmp_path, monkeypatch):
    today = datetime.now(timezone.utc).date().isoformat()
    summaries_dir = tmp_path / "30_Knowledge_Base" / "YouTube_Summaries"
    summaries_dir.mkdir(parents=True)
    (summaries_dir / "YouTube_Insight_abcdefghijk_today.md").write_text(
        "\n".join(
            [
                "---",
                "entity_type: youtube_insight",
                "channel: Macro Channel",
                "video_id: abcdefghijk",
                "source_url: https://www.youtube.com/watch?v=abcdefghijk",
                "image: https://i.ytimg.com/vi/abcdefghijk/hqdefault.jpg",
                f"date: {today}",
                "---",
                "- Yield outlook remains restrictive.",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(tmp_path))

    references = recent_youtube_references()

    assert references[0]["reference_id"] == "youtube_abcdefghijk"
    assert references[0]["publisher"] == "Macro Channel"
    assert "Yield outlook" in references[0]["summary"]
