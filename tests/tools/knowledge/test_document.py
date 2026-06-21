import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from tools.knowledge.document import ingest_pdf

@patch("tools.knowledge.document._call_extractor_llm")
@patch("tools.knowledge.document._build_article_md")
def test_ingest_pdf(mock_build, mock_call, tmp_path, monkeypatch):
    # Test missing file
    assert "ไม่พบไฟล์ PDF" in ingest_pdf.func("not_exist.pdf")
    
    # Test not PDF extension
    f_txt = tmp_path / "test.txt"
    f_txt.touch()
    assert "ไฟล์ไม่ใช่ PDF" in ingest_pdf.func(str(f_txt))
    
    # Create valid dummy PDF structure mock
    f_pdf = tmp_path / "test.pdf"
    f_pdf.touch()
    
    # Mock pypdf failure
    with patch("pypdf.PdfReader", side_effect=Exception("PDF Corrupted")):
        assert "อ่าน PDF ล้มเหลว" in ingest_pdf.func(str(f_pdf))
        
    # Mock pypdf success but empty text
    mock_reader = MagicMock()
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "   "
    mock_reader.pages = [mock_page]
    with patch("pypdf.PdfReader", return_value=mock_reader):
        assert "ไม่มีข้อความที่สกัดได้" in ingest_pdf.func(str(f_pdf))
        
    # Mock pypdf success with text
    mock_page.extract_text.return_value = "Financial Data"
    mock_call.return_value = "Extracted PDF Data"
    mock_build.return_value = "Final Markdown"
    
    with patch("pypdf.PdfReader", return_value=mock_reader):
        res = ingest_pdf.func(str(f_pdf))
        assert res == "Final Markdown"
        mock_call.assert_called_once_with("Financial Data", "PDF: test.pdf")
        
    # LLM Value Error
    mock_call.side_effect = ValueError("LLM Err")
    with patch("pypdf.PdfReader", return_value=mock_reader):
        assert "ERROR: LLM Err" in ingest_pdf.func(str(f_pdf))
        
    # LLM Generic Exception
    mock_call.side_effect = Exception("General Err")
    with patch("pypdf.PdfReader", return_value=mock_reader):
        assert "LLM Extraction ล้มเหลว" in ingest_pdf.func(str(f_pdf))
        
    # Simulate ImportError
    with patch.dict("sys.modules", {"pypdf": None}):
        assert "ไม่พบ library 'pypdf'" in ingest_pdf.func(str(f_pdf))
