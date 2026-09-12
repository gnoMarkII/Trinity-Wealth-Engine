from __future__ import annotations

from pathlib import Path

from tools.archivist.portable_links import resolve_vault_target_detailed
from tools.archivist.vault_audit import scan_vault


def test_source_relative_url_decoded_markdown_link_is_resolved(tmp_path: Path) -> None:
    vault = tmp_path / "memories"
    source = vault / "30_Knowledge_Base" / "News" / "2026" / "Source Note.md"
    target = vault / "30_Knowledge_Base" / "Concepts" / "Target (A).md"
    source.parent.mkdir(parents=True)
    target.parent.mkdir(parents=True)
    source.write_text(
        "[target](../../Concepts/Target%20%28A%29.md#Overview)\n",
        encoding="utf-8",
    )
    target.write_text("# Target\n", encoding="utf-8")

    resolved = resolve_vault_target_detailed(
        vault,
        "../../Concepts/Target%20%28A%29.md#Overview",
        source="30_Knowledge_Base/News/2026/Source Note.md",
    )

    assert resolved.status == "resolved"
    assert resolved.target == target.resolve()
    audit = scan_vault(vault)
    assert audit.stats["broken_links"] == 0
    assert audit.stats["ambiguous_links"] == 0


def test_graph_context_normalizes_relative_module_vault(tmp_path: Path, monkeypatch) -> None:
    vault = tmp_path / "memories"
    main = vault / "FTNT.md"
    source = vault / "Source.md"
    vault.mkdir(parents=True)
    main.write_text("# FTNT\n", encoding="utf-8")
    source.write_text("See [FTNT](FTNT.md).\n", encoding="utf-8")

    import tools.archivist.search as search

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(search, "VAULT_PATH", Path("memories"))
    output = search.search_graph_context.func("FTNT")

    assert "incoming" in output
    assert "Source.md" in output
    assert "Traceback" not in output
