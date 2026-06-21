import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import os

from tools.archivist.indexer import update_master_index, _index_upsert, _is_indexable, _entity_category
from tools.archivist.linter import lint_structural_health, lint_semantic_conflict
from tools.archivist.core import init_vault_structure, read_file
import tools.archivist.indexer as indexer_module
import tools.archivist.linter as linter_module
import tools.archivist.core as core_module

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
    return vault_dir

def test_init_vault_structure(test_vault, monkeypatch):
    monkeypatch.setenv("VAULT_EXTRA_FOLDERS", "50_Crypto,60_Research/Drafts")
    init_vault_structure()
    
    assert (test_vault / "50_Crypto").exists()
    assert (test_vault / "60_Research/Drafts").exists()
    assert (test_vault / "30_Knowledge_Base").exists()

def test_read_file_limit(test_vault, monkeypatch):
    monkeypatch.setattr(core_module, "_READ_FILE_LIMIT", 10)
    
    target = test_vault / "Read_Test.md"
    target.write_text("Hello World Very Long", encoding="utf-8")
    
    res = read_file.func("Read_Test.md")
    assert "ตัดทอน" in res
    assert len(res) < 100

def test_linter_structural_no_files(test_vault):
    res = lint_structural_health.func()
    assert "ไม่มีไฟล์ใดใน Vault" in res

def test_lint_semantic_conflict_target(test_vault):
    # Test folder
    (test_vault / "Folder").mkdir()
    (test_vault / "Folder" / "f1.md").write_text("Long " * 1000, encoding="utf-8")
    res1 = lint_semantic_conflict.func("Folder")
    assert "ตัดทอน" in res1
    
    # Test not found
    res2 = lint_semantic_conflict.func("Not_Found")
    assert "ไม่พบไฟล์ที่ตรงกับ" in res2

def test_indexer_is_indexable(test_vault):
    # Test 00_Inbox
    sys_file = test_vault / "00_Inbox" / "test.md"
    assert not _is_indexable(sys_file)
    
    # Test normal
    normal_file = test_vault / "normal.md"
    assert _is_indexable(normal_file)

def test_indexer_upsert(test_vault):
    indexer_module._index_cache_built = False
    indexer_module._index_dirty = False
    indexer_module._index_cache.clear()
    
    # Upsert a new file
    f = test_vault / "30_Knowledge_Base" / "Stocks" / "AAPL.md"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("---\nentity_type: stock_entity\n---", encoding="utf-8")
    
    _index_upsert(f)
    assert indexer_module._index_dirty
    assert "30_Knowledge_Base\\Stocks" in indexer_module._index_cache or "30_Knowledge_Base/Stocks" in indexer_module._index_cache
    
    # Upsert existing file to trigger the replacement logic
    f.write_text("---\nentity_type: stock_entity\n---", encoding="utf-8")
    _index_upsert(f)

def test_indexer_oserror(test_vault, monkeypatch):
    def mock_read(*args, **kwargs):
        raise OSError("Permission denied")
    
    f = test_vault / "bad.md"
    f.write_text("test", encoding="utf-8")
    monkeypatch.setattr(Path, "read_text", mock_read)
    
    res = indexer_module._read_entity_type(f)
    assert res == "—"

def test_entity_category():
    assert _entity_category("30_Knowledge_Base/Stocks/AAPL") == "Stocks"
    assert _entity_category("Other_Folder") == "Other"

def test_update_master_index_no_files(test_vault):
    # Clear cache
    indexer_module._index_cache.clear()
    res = indexer_module._write_index_from_cache()
    assert "ไม่มีไฟล์ที่จะ index" in res
