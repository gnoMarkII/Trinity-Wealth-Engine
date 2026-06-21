import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from tools.archivist.search import search_all_memories, search_graph_context
import tools.archivist.search as search_module

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
    monkeypatch.setattr(search_m, "CHROMA_PATH", vault_dir / ".chroma_index")
    monkeypatch.setattr(search_m, "_CHROMA_MTIME_FILE", vault_dir / ".chroma_mtime")
    # Reset cache
    search_m._vs_cache.clear()
    return vault_dir

@patch("tools.archivist.search.Chroma")
@patch("tools.archivist.search.get_embeddings")
def test_search_all_memories_basic(mock_embeddings, mock_chroma_class, test_vault):
    mock_embeddings.return_value = MagicMock()
    
    mock_vs = MagicMock()
    # mock similarity_search returns 1 doc
    mock_doc = MagicMock()
    mock_doc.page_content = "Test content"
    mock_doc.metadata = {"source": "test.md"}
    mock_vs.similarity_search.return_value = [mock_doc]
    mock_chroma_class.return_value = mock_vs
    
    # Empty vault
    res = search_all_memories.func("test")
    assert "ยังไม่มีไฟล์ความจำใด" in res
    
    # Add a file
    (test_vault / "test.md").write_text("Hello World", encoding="utf-8")
    
    res = search_all_memories.func("test keyword")
    assert "ผลการค้นหาเชิงความหมาย" in res
    assert "Test content" in res
    
    # Test Chroma cache hit
    res2 = search_all_memories.func("test keyword")
    assert "ผลการค้นหาเชิงความหมาย" in res2

    # Test no results
    mock_vs.similarity_search.return_value = []
    res_empty = search_all_memories.func("test keyword")
    assert "ไม่พบความจำที่เกี่ยวข้องกับ" in res_empty
    
    # Test exception in Chroma init
    search_module._vs_cache.clear()
    mock_chroma_class.side_effect = [Exception("Init fail"), Exception("Init fail fallback")]
    res_fail = search_all_memories.func("test")
    assert "เกิดข้อผิดพลาดในการเปิด vectorstore" in res_fail

@patch("tools.archivist.search.Chroma")
@patch("tools.archivist.search.get_embeddings")
def test_search_all_memories_updates(mock_embeddings, mock_chroma_class, test_vault):
    # Test modified and removed
    mock_vs = MagicMock()
    mock_chroma_class.return_value = mock_vs
    
    f1 = test_vault / "f1.md"
    f2 = test_vault / "f2.md"
    f1.write_text("Old content", encoding="utf-8")
    f2.write_text("F2 content", encoding="utf-8")
    
    # First index
    search_all_memories.func("test")
    
    # Remove f2, update f1
    f2.unlink()
    import time
    time.sleep(0.01)
    f1.write_text("New content", encoding="utf-8")
    
    # Search again
    search_all_memories.func("test")
    assert mock_vs.delete.called
    assert mock_vs.add_texts.called

    # Test exception during similarity_search
    mock_vs.similarity_search.side_effect = Exception("search error")
    res_err = search_all_memories.func("test")
    assert "เกิดข้อผิดพลาดในการค้นหา:" in res_err

def test_search_graph_context(test_vault):
    res = search_graph_context.func("target")
    assert "ยังไม่มีไฟล์ความจำใด" in res
    
    f1 = test_vault / "target.md"
    f1.write_text("Target file links to [[linked|alias]]", encoding="utf-8")
    
    res2 = search_graph_context.func("target")
    assert "ไม่พบไฟล์ใน Vault" in res2
    assert "Target file links" in res2
    
    f2 = test_vault / "linked.md"
    # test long content
    f2.write_text("a" * 2000, encoding="utf-8")
    
    res3 = search_graph_context.func("target")
    assert "ตัดทอน" in res3

    res4 = search_graph_context.func("not_found")
    assert "ไม่พบไฟล์สำหรับ entity" in res4
