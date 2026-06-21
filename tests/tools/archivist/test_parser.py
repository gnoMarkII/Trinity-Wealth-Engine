import pytest
from tools.archivist.parser import (
    _strip_frontmatter,
    _parse_h2_sections,
    _parse_h3_subsections,
    _split_bullets,
    _extract_asset_tickers,
    _chunk_file
)
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

def test_strip_frontmatter():
    text1 = "---\ntitle: abc\n---\nBody content"
    assert _strip_frontmatter(text1) == "Body content"
    
    text2 = "No frontmatter here"
    assert _strip_frontmatter(text2) == "No frontmatter here"
    
    text3 = "---\ninvalid"
    assert _strip_frontmatter(text3) == "---\ninvalid"

def test_parse_h2_sections():
    text = "## Section 1\nContent 1\n## Section 2\nContent 2\n## Section 3\n"
    res = _parse_h2_sections(text)
    assert res["Section 1"] == "Content 1"
    assert res["Section 2"] == "Content 2"
    assert "Section 3" in res
    
def test_parse_h3_subsections():
    text = "### Sub 1\nC1\n### Sub 2\nC2\n"
    res = _parse_h3_subsections(text)
    assert res["Sub 1"] == "C1"
    assert res["Sub 2"] == "C2"
    
    text_none = "Just some text"
    res2 = _parse_h3_subsections(text_none)
    assert res2["ทั่วไป"] == "Just some text"
    
    text_empty = ""
    res3 = _parse_h3_subsections(text_empty)
    assert res3 == {}

def test_split_bullets():
    text = "- a\n- b\n- c\n- d\n- e"
    res = _split_bullets(text, max_per_node=2)
    assert len(res) == 3
    assert res[0] == "- a\n- b"
    
    assert _split_bullets("") == []

def test_extract_asset_tickers():
    text = "- [[AAPL]] Apple is good\n* [[TSLA]] Tesla is volatile\n- Nothing here"
    res = _extract_asset_tickers(text)
    assert len(res) == 2
    assert res[0][0] == "AAPL"
    assert res[0][1] == "AAPL Apple is good"
    assert res[1][0] == "TSLA"

def test_chunk_file(tmp_path, monkeypatch):
    import tools.archivist.parser as parser_module
    vault_dir = tmp_path.resolve()
    monkeypatch.setattr(parser_module, "VAULT_PATH", vault_dir)
    f = vault_dir / "test.md"
    f.write_text("Hello World", encoding="utf-8")
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=5, chunk_overlap=0)
    texts, metas, ids = _chunk_file(f, splitter)
    assert len(texts) > 0
    assert metas[0]["source"] == "test.md"
    assert ids[0] == "test.md::0"
