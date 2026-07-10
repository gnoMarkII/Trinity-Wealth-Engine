import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from tools.knowledge.youtube import (
    _extract_video_id,
    _find_existing_insight,
    _entries_to_text,
    _get_raw_transcript,
    ingest_youtube_transcript,
    TranscriptUnavailable,
)

def test_extract_video_id():
    assert _extract_video_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert _extract_video_id("https://youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert _extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert _extract_video_id("https://youtube.com/embed/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert _extract_video_id("https://youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert _extract_video_id("https://youtube.com/live/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert _extract_video_id("invalid_url_string") is None

def test_find_existing_insight(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.knowledge.youtube._YOUTUBE_SUMMARIES_PATH", tmp_path)
    
    assert _find_existing_insight("12345678901") is None
    
    # Create file
    f = tmp_path / "YouTube_Insight_12345678901_2024-01-01.md"
    f.touch()
    
    assert _find_existing_insight("12345678901") == f

def test_entries_to_text():
    entries = [{"text": "Hello "}, {"text": "World"}]
    assert _entries_to_text(entries) == "Hello  World"
    
    class DummySnippet:
        def __init__(self, t):
            self.text = t
    
    entries2 = [DummySnippet("Test "), DummySnippet("Snippet")]
    assert _entries_to_text(entries2) == "Test  Snippet"

@patch("tools.knowledge.youtube._ytt_api")
def test_get_raw_transcript(mock_api):
    # Success via fetch
    mock_api.fetch.return_value = [{"text": "Hello"}]
    assert _get_raw_transcript("vid") == "Hello"
    
    # NoTranscriptFound -> fallback to list
    from youtube_transcript_api import NoTranscriptFound
    mock_api.fetch.side_effect = NoTranscriptFound("not found", "not found", "not found")
    
    mock_transcript = MagicMock()
    mock_transcript.fetch.return_value = [{"text": "Fallback"}]
    mock_api.list.return_value = [mock_transcript]
    
    assert _get_raw_transcript("vid") == "Fallback"
    
    # List fails
    mock_transcript.fetch.side_effect = Exception("error")
    with pytest.raises(TranscriptUnavailable):
        _get_raw_transcript("vid")

@patch("tools.knowledge.youtube._extract_via_gemini_url_direct")
@patch("tools.knowledge.youtube._get_ytdlp_subtitles")
@patch("tools.knowledge.youtube._call_extractor_llm")
@patch("tools.knowledge.youtube._get_raw_transcript")
@patch("tools.knowledge.youtube._find_existing_insight")
def test_ingest_youtube_transcript_tiers(mock_find, mock_get_raw, mock_call_llm, mock_ytdlp, mock_gemini):
    # 1. Invalid ID
    assert "ERROR" in ingest_youtube_transcript.func("invalid")

    # 2. Duplicate
    mock_find.return_value = Path("test.md")
    assert "DUPLICATE" in ingest_youtube_transcript.func("dQw4w9WgXcQ")

    # 3. Video Unavailable (Live event)
    from youtube_transcript_api import VideoUnavailable
    mock_find.return_value = None
    mock_get_raw.side_effect = VideoUnavailable("live event")
    assert "ข้าม: วิดีโอนี้ยังไม่ถึงเวลา Live" in ingest_youtube_transcript.func("dQw4w9WgXcQ")

    # 4. Tier 1 Success
    mock_get_raw.side_effect = None
    mock_get_raw.return_value = "Hello from Tier 1"
    mock_call_llm.return_value = "Insight Tier 1"
    res1 = ingest_youtube_transcript.func("dQw4w9WgXcQ")
    assert "Insight Tier 1" in res1
    assert "Tier 1: YouTube Transcript API" in res1

    # 5. Tier 1 Fails -> Tier 2 Success
    mock_get_raw.side_effect = Exception("IP Blocked")
    mock_ytdlp.return_value = "Subtitle from yt-dlp"
    mock_call_llm.return_value = "Insight Tier 2"
    res2 = ingest_youtube_transcript.func("dQw4w9WgXcQ")
    assert "Insight Tier 2" in res2
    assert "Tier 2: yt-dlp Subtitle Scraper" in res2

    # 6. Tier 1 & Tier 2 Fail -> Tier 3 Success (Zero-Download Gemini Direct URL)
    mock_get_raw.side_effect = Exception("No transcript")
    mock_ytdlp.return_value = None
    mock_gemini.return_value = "Insight Tier 3 via Gemini"
    res3 = ingest_youtube_transcript.func("dQw4w9WgXcQ")
    assert "Insight Tier 3 via Gemini" in res3
    assert "Tier 3: Gemini Direct YouTube Multimodal (Zero-Download)" in res3

    # 7. All 3 Tiers Fail
    mock_gemini.side_effect = Exception("Gemini Error")
    res4 = ingest_youtube_transcript.func("dQw4w9WgXcQ")
    assert "ERROR: ดึงข้อมูลและสรุปเนื้อหาจาก YouTube ล้มเหลวทั้ง 3 ขั้นตอน" in res4

