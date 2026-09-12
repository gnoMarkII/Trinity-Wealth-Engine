from pathlib import Path

from tools.archivist.portable_links import (
    render_relative_markdown_link,
    render_resolved_markdown_link,
    resolve_vault_target,
)


def test_relative_link_encodes_unicode_spaces_and_parentheses(tmp_path: Path) -> None:
    source = tmp_path / "30_Knowledge_Base" / "Source Note.md"
    target = tmp_path / "30_Knowledge_Base" / "หุ้นไทย (A).md"
    source.parent.mkdir(parents=True)
    target.write_text("# target\n", encoding="utf-8")

    rendered = render_relative_markdown_link(tmp_path, source, target, label="A")

    assert rendered == "[A](%E0%B8%AB%E0%B8%B8%E0%B9%89%E0%B8%99%E0%B9%84%E0%B8%97%E0%B8%A2%20%28A%29.md)"
    assert resolve_vault_target(tmp_path, "หุ้นไทย (A)") == target


def test_unresolved_or_ambiguous_target_fails_closed(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "Same.md").write_text("# a\n", encoding="utf-8")
    (tmp_path / "b" / "Same.md").write_text("# b\n", encoding="utf-8")
    source = tmp_path / "source.md"

    assert resolve_vault_target(tmp_path, "Same") is None
    assert render_resolved_markdown_link(tmp_path, source, "missing", label="Missing") == "Missing"
