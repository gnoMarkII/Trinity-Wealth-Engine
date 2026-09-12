from __future__ import annotations

import json
from pathlib import Path

from scripts.apply_live_file_repairs_r3 import _main
from tools.archivist.catalog_adapter import SqliteNoteCatalogAdapter
from tools.archivist.metadata import parse_note


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_live_file_repair_is_guarded_and_preserves_unknown_links(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    output = tmp_path / "evidence"
    _write(vault / ".system" / "vault_config.json", '{"layout_version": 2}\n')
    _write(
        vault / "index.md",
        "---\nschema_version: 1\nnote_id: root_index\ntitle: Home\n---\n\n"
        "# Home\n\n[[30_Knowledge_Base/Stocks|Stocks]]\n",
    )
    _write(
        vault / "30_Knowledge_Base" / "Stocks" / "AAPL" / "AAPL.md",
        "---\nschema_version: 1\nnote_id: aapl_hub\nentity_type: stock_hub\n"
        "title: AAPL\ndate: 2026-09-01\nticker: AAPL\n---\n\n# AAPL\n",
    )
    _write(
        vault / "30_Knowledge_Base" / "Concepts" / "AAPL.md",
        "---\nschema_version: 1\nnote_id: aapl_concept\nentity_type: concept\n"
        "title: Apple concept\ndate: 2026-09-01\n---\n\n# Apple concept\n",
    )
    catalogued = vault / "30_Knowledge_Base" / "News" / "2026" / "09" / "catalogued.md"
    _write(
        catalogued,
        "---\ntitle: Catalogued\nentity_type: company_news\ncreated: 2026-09-02\n---\n\n"
        "# Catalogued\n\n[[AAPL]] [[NVDA]] [[Unknown_Target]]\n",
    )
    catalog = SqliteNoteCatalogAdapter(vault_root=vault)
    catalog.sync_from_vault(force=True)
    uncatalogued = vault / "30_Knowledge_Base" / "News" / "2026" / "09" / "uncatalogued.md"
    _write(
        uncatalogued,
        "---\ntitle: Uncatalogued\nentity_type: company_news\ncreated: 2026-09-03\n---\n\n"
        "# Uncatalogued\n\n[[AAPL|Apple]]\n",
    )

    assert _main(["--vault", str(vault), "--output-root", str(output), "--apply"]) == 0

    meta, body, issues = parse_note(catalogued.read_text(encoding="utf-8"))
    assert not issues
    assert meta["schema_version"] == 2
    assert meta["note_id"].startswith("legacy_unresolved_")
    assert str(meta["date"]) == "2026-09-02"
    assert "[[30_Knowledge_Base/Stocks/AAPL/AAPL]]" in body
    assert "[[30_Knowledge_Base/Stocks/NVDA/NVDA]]" in body
    assert "[[Unknown_Target]]" in body

    uncatalogued_meta, uncatalogued_body, _ = parse_note(
        uncatalogued.read_text(encoding="utf-8")
    )
    assert uncatalogued_meta["note_id"].startswith("note_")
    assert "[[30_Knowledge_Base/Stocks/AAPL/AAPL|Apple]]" in uncatalogued_body
    assert (vault / "30_Knowledge_Base" / "Stocks" / "NVDA" / "NVDA.md").is_file()
    assert not (vault / "30_Knowledge_Base" / "Stocks" / "AMZN" / "AMZN.md").exists()
    assert (vault / "00_Index" / "Stocks_Hub.md").is_file()

    latest = json.loads((output / "latest-live-repair.json").read_text(encoding="utf-8"))
    assert latest["status"] == "PASS"
    assert latest["snapshot"]["restore_proof"] == "PASS"
    assert latest["unintended_active_changes"] == []
