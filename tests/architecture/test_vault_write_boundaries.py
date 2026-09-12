from __future__ import annotations

import importlib.util
from pathlib import Path


def _scanner_module():
    path = Path("scripts/scan_vault_writers_r8.py").resolve()
    spec = importlib.util.spec_from_file_location("scan_vault_writers_r8", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_writer_inventory_has_no_unresolved_production_intent() -> None:
    report = _scanner_module().scan(Path(".").resolve())
    assert report["counts"].get("parse_error", 0) == 0
    assert report["counts"].get("unresolved", 0) == 0
    assert report["counts"].get("allowlisted", 0) > 0


def test_application_knowledge_models_do_not_import_writer_infrastructure() -> None:
    for path in Path("application/knowledge").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "tools.archivist.artifact_writer" not in text
        assert "tools.archivist.write_broker" not in text
