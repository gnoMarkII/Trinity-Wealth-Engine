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

@patch("tools.knowledge.youtube._call_extractor_llm")
@patch("tools.knowledge.youtube._get_raw_transcript")
@patch("tools.knowledge.youtube._find_existing_insight")
def test_ingest_youtube_transcript(mock_find, mock_get_raw, mock_call_llm):
    # Invalid ID
    assert "ERROR" in ingest_youtube_transcript.func("invalid")
    
    # Duplicate
    mock_find.return_value = Path("test.md")
    assert "DUPLICATE" in ingest_youtube_transcript.func("dQw4w9WgXcQ")
    
    # Transcript Disabled
    from youtube_transcript_api import TranscriptsDisabled, VideoUnavailable
    mock_find.return_value = None
    mock_get_raw.side_effect = TranscriptsDisabled("disabled")
    assert "ERROR:" in ingest_youtube_transcript.func("dQw4w9WgXcQ")
    
    # Video Unavailable
    mock_get_raw.side_effect = VideoUnavailable("video ID")
    assert "ไม่พบวิดีโอ" in ingest_youtube_transcript.func("dQw4w9WgXcQ")
    
    # Generic exception in get_raw
    mock_get_raw.side_effect = Exception("generic")
    assert "ดึง Transcript ล้มเหลว" in ingest_youtube_transcript.func("dQw4w9WgXcQ")
    
    # Empty transcript
    mock_get_raw.side_effect = None
    mock_get_raw.return_value = "   "
    assert "ว่างเปล่า" in ingest_youtube_transcript.func("dQw4w9WgXcQ")
    
    # Success with cutoff
    mock_get_raw.return_value = "a" * 21000
    mock_call_llm.return_value = "Insight result"
    res = ingest_youtube_transcript.func("dQw4w9WgXcQ")
    assert "Insight result" in res
    assert "youtube_insight" in res
    assert mock_call_llm.call_args[0][0].endswith("[ตัดทอน — Transcript เกิน 20,000 ตัวอักษร]")
    
    # LLM ValueError
    mock_call_llm.side_effect = ValueError("LLM Error")
    assert "ERROR: LLM Error" in ingest_youtube_transcript.func("dQw4w9WgXcQ")
    
    # LLM Generic Exception
    mock_call_llm.side_effect = Exception("Other Error")
    assert "LLM Extraction ล้มเหลว" in ingest_youtube_transcript.func("dQw4w9WgXcQ")
