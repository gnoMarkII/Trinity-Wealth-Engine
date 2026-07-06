import pytest
from unittest.mock import patch, MagicMock

from tools.knowledge.core import _call_extractor_llm, _build_article_md, _get_extractor_llm

def test_build_article_md():
    res = _build_article_md("extracted text", "http://example.com", "My Title", "2024-01-01", "2024-01-01 10:00:00", image="img.png")
    assert "title: My Title" in res
    assert "image: img.png" in res
    assert "extracted text" in res
    assert "http://example.com" in res

@patch("tools.knowledge.core.os.environ.get")
@patch("tools.knowledge.core._get_extractor_llm")
def test_call_extractor_llm(mock_get_llm, mock_env):
    mock_env.return_value = "key"
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "   result data   "
    mock_llm.invoke.return_value = mock_response
    mock_get_llm.return_value = mock_llm
    
    # Normal text
    res = _call_extractor_llm("short text", "src")
    assert res == "result data"
    
    # Too long text
    long_text = "a" * 25000
    _call_extractor_llm(long_text, "src")
    
    call_args = mock_llm.invoke.call_args[0][0]
    content = call_args[1]["content"]
    assert "[ตัดทอน — เนื้อหาเกิน 20,000 ตัวอักษร]" in content

@patch("tools.knowledge.core.get_llm")
def test_get_extractor_llm(mock_get_llm):
    mock_llm = MagicMock()
    mock_get_llm.return_value = mock_llm
    
    res = _get_extractor_llm()
    assert mock_llm.with_retry.call_count == 1
