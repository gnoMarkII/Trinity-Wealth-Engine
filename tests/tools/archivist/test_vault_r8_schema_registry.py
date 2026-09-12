from __future__ import annotations

from tools.archivist.schema_registry import load_default_registry


def test_registry_alias_is_read_compatible_but_publish_requires_canonical_type() -> None:
    registry = load_default_registry()
    assert registry.canonical_entity_type("article") == "company_news"
    valid, issues = registry.validate_metadata(
        {"schema_version": 2, "entity_type": "article", "title": "Legacy"},
        allow_identity_allocation=True,
    )
    assert valid is False
    assert any(issue["code"] == "legacy_alias" for issue in issues)
    assert registry.digest()
    assert registry.policy_digest()
