import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from tools.knowledge.article import _is_url_already_processed, ingest_article_url

def test_is_url_already_processed(tmp_path, monkeypatch):
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(tmp_path))
    
    assert _is_url_already_processed("https://example.com/1") is False
    
    news_dir = tmp_path / "30_Knowledge_Base/News"
    news_dir.mkdir(parents=True, exist_ok=True)
    f = news_dir / "article1.md"
    f.write_text("Source: https://example.com/1", encoding="utf-8")
    
    assert _is_url_already_processed("https://example.com/1") is True
    assert _is_url_already_processed("https://example.com/2") is False

@patch("tools.knowledge.article._call_extractor_llm")
@patch("tools.knowledge.article._build_article_md")
@patch("tools.knowledge.article.with_retry")
def test_ingest_article_url(mock_retry, mock_build, mock_call, monkeypatch, tmp_path):
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(tmp_path))
    
    # Missing trafilatura
    with patch.dict("sys.modules", {"trafilatura": None}):
        assert "ไม่พบ library 'trafilatura'" in ingest_article_url.func("url")

    # Mock successful import
    with patch("trafilatura.fetch_url") as mock_fetch, \
         patch("trafilatura.extract_metadata") as mock_extract_meta:
        
        mock_extract_meta.return_value = MagicMock(title="Mock Title", image="Mock Image")
        
        # Download failed / Text too short -> triggers length check error
        mock_retry.return_value = (None, None)
        assert "เนื้อหาที่ดึงได้สั้นเกินไป" in ingest_article_url.func("url")
        
        # Success path (Length > 800 or > 150 words)
        mock_retry.return_value = ("Good Text " * 200, "Title")
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
