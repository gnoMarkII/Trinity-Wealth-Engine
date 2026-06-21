import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from tools.knowledge.article import _find_existing_article, ingest_article_url

def test_find_existing_article(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.knowledge.article._ARTICLES_PATH", tmp_path)
    
    assert _find_existing_article("nonexist") is None
    
    f = tmp_path / "Insight_target123_test.md"
    f.touch()
    
    assert _find_existing_article("target123") == f

@patch("tools.knowledge.article._call_extractor_llm")
@patch("tools.knowledge.article._build_article_md")
def test_ingest_article_url(mock_build, mock_call, monkeypatch):
    # Missing trafilatura
    with patch.dict("sys.modules", {"trafilatura": None}):
        assert "ไม่พบ library 'trafilatura'" in ingest_article_url.func("url")

    # Mock successful import
    with patch("trafilatura.fetch_url") as mock_fetch, \
         patch("trafilatura.extract_metadata") as mock_extract_meta, \
         patch("trafilatura.extract") as mock_extract:
        # Download failed (returns None)
        mock_fetch.return_value = None
        assert "ไม่สามารถดึงเนื้อหาจาก URL" in ingest_article_url.func("url")
        
        # Download exception
        mock_fetch.side_effect = Exception("NetErr")
        assert "ดึงเนื้อหาล้มเหลว" in ingest_article_url.func("url")
        
        # Extract empty text
        mock_fetch.side_effect = None
        mock_fetch.return_value = "<html></html>"
        mock_extract_meta.return_value = MagicMock(title="Title", image="img.png")
        mock_extract.return_value = "   "
        assert "ไม่สามารถสกัดเนื้อหาจาก URL" in ingest_article_url.func("url")
        
        # Success path
        mock_extract.return_value = "Good Text"
        mock_call.return_value = "Extracted"
        mock_build.return_value = "Markdown Output"
        
        res = ingest_article_url.func("url")
        assert res == "Markdown Output"
        # Check LLM ValueError
        mock_call.side_effect = ValueError("LLM Error")
        assert "ERROR: LLM Error" in ingest_article_url.func("url")
        
        # Check Generic Exception
        mock_call.side_effect = Exception("General Error")
        assert "LLM Extraction ล้มเหลว" in ingest_article_url.func("url")
