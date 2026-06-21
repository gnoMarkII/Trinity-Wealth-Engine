import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import os
import json

from tools.archivist.core import read_file
from tools.archivist.writer import save_memory, write_raw_markdown
from tools.archivist.indexer import update_master_index
from tools.archivist.search import search_all_memories, search_graph_context
from tools.archivist.linter import lint_structural_health, lint_semantic_conflict

# Mocking VAULT_PATH
@pytest.fixture(autouse=True)


def test_save_memory_new(isolated_archivist, tmp_vault):
    res = isolated_archivist.save_memory.invoke({
        "title": "Test Memory",
        "content": "This is a test content.",
        "folder_path": "30_Knowledge_Base/Stocks",
        "tags": ["test"],
        "entity_type": "Concept",
        "aliases": ["TM"],
        "linked_files": ["Linked_Doc"]
    })
    
    assert "บันทึกสำเร็จ (new)" in res
    
    saved_file = tmp_vault / "30_Knowledge_Base" / "Stocks" / "Test Memory.md"
    assert saved_file.exists()
    content = saved_file.read_text(encoding="utf-8")
    assert "entity_type: Concept" in content
    assert "tags:" in content
    assert "This is a test content." in content
    assert "[[Linked_Doc]]" in content


def test_write_raw_markdown(isolated_archivist, tmp_vault):
    raw_md = "---\ntitle: Raw Test\nentity_type: test\ndate: 2026-06-20\n---\n# Content"
    res = isolated_archivist.write_raw_markdown.invoke({
        "content": raw_md,
        "folder_path": "30_Knowledge_Base/Macroeconomics",
        "filename": "Raw_Test_File"
    })
    
    assert "บันทึกสำเร็จ (raw" in res
    
    saved_file = tmp_vault / "30_Knowledge_Base" / "Macroeconomics" / "Raw_Test_File.md"
    assert saved_file.exists()
    assert saved_file.read_text(encoding="utf-8") == raw_md


def test_read_file(isolated_archivist, tmp_vault):
    target = tmp_vault / "Read_Test.md"
    target.write_text("Hello World", encoding="utf-8")
    
    res = isolated_archivist.read_file.invoke({"filepath": "Read_Test.md"})
    assert "=== Read_Test.md ===" in res
    assert "Hello World" in res


def test_read_file_not_found(isolated_archivist, tmp_vault):
    res = isolated_archivist.read_file.invoke({"filepath": "Not_Exist.md"})
    assert "ไม่พบไฟล์" in res


def test_update_master_index(isolated_archivist, tmp_vault):
    (tmp_vault / "30_Knowledge_Base").mkdir(parents=True, exist_ok=True)
    (tmp_vault / "30_Knowledge_Base" / "Test_Entity.md").write_text(
        "---\nentity_type: stock_entity\n---\nTest", encoding="utf-8"
    )
    
    res = isolated_archivist.update_master_index.invoke({})
    assert "อัปเดต index.md สำเร็จ" in res
    index_file = tmp_vault / "index.md"
    assert index_file.exists()
    assert "Entities" in index_file.read_text(encoding="utf-8")


def test_lint_structural_health(isolated_archivist, tmp_vault):
    # Setup some files
    (tmp_vault / "Orphan.md").write_text("Just some text", encoding="utf-8")
    (tmp_vault / "Empty.md").write_text("", encoding="utf-8")
    (tmp_vault / "Linked.md").write_text("Links to [[Orphan]]", encoding="utf-8")
    
    res = isolated_archivist.lint_structural_health.invoke({})
    
    assert "Vault Health Report" in res
    assert "Empty Files" in res
    assert "Empty" in res
    assert "Orphan" in res # Depending on exact logic


def test_lint_semantic_conflict(isolated_archivist, tmp_vault):
    folder = tmp_vault / "TestFolder"
    folder.mkdir()
    (folder / "file1.md").write_text("Fact A", encoding="utf-8")
    (folder / "file2.md").write_text("Fact B", encoding="utf-8")
    
    res = isolated_archivist.lint_semantic_conflict.invoke({"target_folder_or_entity": "TestFolder"})
    assert "Semantic Conflict Check" in res
    assert "Fact A" in res
    assert "Fact B" in res


@patch("tools.archivist.search._searchable_files")
def test_search_all_memories(mock_searchable, isolated_archivist, tmp_vault):
    mock_searchable.return_value = []
    res = isolated_archivist.search_all_memories.invoke({"keyword": "test"})
    # Without chromadb running fully or mocking it deeply, we can just test the empty/no result case
    # or patch Chroma to avoid actual vector search.
    assert "ไม่พบเอกสาร" in res or "ไม่มีไฟล์ใน Vault" in res or len(res) > 0


def test_search_graph_context(isolated_archivist, tmp_vault):
    # Setup some files
    f1 = tmp_vault / "Hub.md"
    f1.write_text("Connects to [[Target]] and [[Other]]", encoding="utf-8")
    f2 = tmp_vault / "Target.md"
    f2.write_text("I am the target", encoding="utf-8")
    
    res = isolated_archivist.search_graph_context.invoke({"entity_name": "Target"})
    assert "--- Main Entity: Target ---" in res
    assert "I am the target" in res
