"""Unit tests for Obsidian Vault V2 Knowledge Metadata, Identity, and Parsing Contracts."""
from __future__ import annotations

import pytest

from application.knowledge.identity import NoteIdentity, RevisionRef, build_document_key
from schemas.knowledge_metadata import (
    CURRENT_SCHEMA_VERSION,
    CommonNoteMetadata,
    EarningsCallMetadata,
    StockHubMetadata,
)
from tools.archivist.metadata import (
    dump_note,
    normalize_legacy_metadata,
    parse_note,
    validate_note,
)


def test_build_document_key() -> None:
    """Document key must be deterministic, versioned, and independent of filenames."""
    key1 = build_document_key(kind="stock_hub", source_identity="FTNT", role="hub")
    assert key1 == "v1:stock_hub:FTNT:hub"

    key2 = build_document_key(kind="earnings_call", source_identity="FTNT:2026-Q2", role="primary")
    assert key2 == "v1:earnings_call:FTNT:2026-Q2:primary"


def test_parse_note_clean_and_malformed() -> None:
    """parse_note must extract frontmatter safely and return issues on malformed YAML."""
    # 1. Clean markdown
    clean_md = (
        "---\n"
        "schema_version: 2\n"
        "note_id: note_001\n"
        "entity_type: stock_hub\n"
        "title: Apple Inc.\n"
        "ticker: 'AAPL'\n"
        "---\n\n"
        "# Apple Inc.\n"
        "Content goes here.\n"
    )
    meta, body, issues = parse_note(clean_md)
    assert len(issues) == 0
    assert meta["ticker"] == "AAPL"
    assert meta["note_id"] == "note_001"
    assert body == "# Apple Inc.\nContent goes here."

    # 2. Malformed YAML (syntax error)
    bad_md = (
        "---\n"
        "title: Broken YAML\n"
        "tags: [unclosed list\n"
        "---\n\n"
        "# Broken Body\n"
    )
    meta, body, issues = parse_note(bad_md)
    assert len(issues) > 0
    assert issues[0]["code"] == "yaml_syntax_error"
    assert meta == {}  # Must be empty on error to prevent accidental overwrite

    # 3. Duplicate keys in frontmatter
    dup_key_md = (
        "---\n"
        "title: Note 1\n"
        "title: Note 2\n"
        "---\n\n"
        "Body\n"
    )
    meta, body, issues = parse_note(dup_key_md)
    assert len(issues) > 0
    assert "duplicate key" in issues[0]["reason"].lower()
    assert meta == {}


def test_normalize_legacy_metadata_types_and_fields() -> None:
    """Normalizes legacy types, author, published_at, and verified gates."""
    legacy_meta = {
        "entity_type": "Company_News",
        "ticker": "005930",  # Samsung KRX numerical ticker
        "author": "John Doe",
        "published_at": "2026-09-01",
        "verified": True,
        "custom_alpha_metric": 123.45,
    }

    normalized, issues = normalize_legacy_metadata(legacy_meta)

    # 1. Canonical entity_type
    assert normalized["entity_type"] == "company_news"

    # 2. Numerical ticker preserved as string
    assert normalized["ticker"] == "005930"
    assert isinstance(normalized["ticker"], str)

    # 3. Author to authors list
    assert normalized["authors"] == ["John Doe"]

    # 4. published_at to published_date and date
    assert normalized["published_date"] == "2026-09-01"
    assert normalized["date"] == "2026-09-01"

    # 5. Verified gate separation: source verified, NOT content reviewed!
    assert normalized["source_verification_status"] == "verified"
    assert normalized["content_verification_status"] == "not_reviewed"

    # 6. Custom fields preserved
    assert normalized["custom_alpha_metric"] == 123.45


def test_earnings_call_pasted_and_annual_variations() -> None:
    """Earnings call supports pasted transcripts without URL and annual FY periods."""
    # 1. Pasted transcript without URL
    pasted_meta = {
        "schema_version": 2,
        "note_id": "ec_001",
        "document_key": "v1:earnings_call:FTNT:2026-Q2",
        "entity_type": "earnings_call",
        "title": "2026-Q2 FTNT Earnings Call",
        "ticker": "FTNT",
        "period": "2026-Q2",
        "fiscal_quarter": 2,
        "fiscal_year": 2026,
        "input_kind": "pasted_transcript",
        "transcript_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "source_verification_status": "not_reviewed",
    }
    model, issues = validate_note(pasted_meta, mode="strict")
    assert model is not None
    assert isinstance(model, EarningsCallMetadata)
    assert model.source_url is None
    assert model.input_kind == "pasted_transcript"

    # 2. Annual FY call without fiscal_quarter
    annual_meta = {
        "schema_version": 2,
        "note_id": "ec_002",
        "document_key": "v1:earnings_call:FTNT:2025-FY",
        "entity_type": "earnings_call",
        "title": "2025-FY FTNT Annual Earnings Call",
        "ticker": "FTNT",
        "period": "2025-FY",
        "period_type": "annual",
        "fiscal_year": 2025,
        "source_url": "https://example.com/annual",
    }
    model, issues = validate_note(annual_meta, mode="strict")
    assert model is not None
    assert isinstance(model, EarningsCallMetadata)
    assert model.fiscal_quarter is None
    assert model.period_type == "annual"


def test_schema_version_3_unsupported_not_downgraded() -> None:
    """Notes with schema_version > CURRENT_SCHEMA_VERSION must be flagged unsupported and never downgraded."""
    v3_meta = {
        "schema_version": 3,
        "note_id": "future_note_99",
        "entity_type": "advanced_quantum_model",
        "title": "Future Note",
        "custom_matrix": [[1, 2], [3, 4]],
    }
    model, issues = validate_note(v3_meta, mode="strict")
    assert model is not None
    assert any(i["code"] == "unsupported_schema" for i in issues)
    assert model.schema_version == 3  # Must NOT be downgraded to 2!
    assert getattr(model, "custom_matrix", None) == [[1, 2], [3, 4]]


def test_dump_note_roundtrip_preserves_custom_fields_and_body() -> None:
    """dump_note must preserve body text, custom fields, and lists accurately."""
    meta = {
        "schema_version": 2,
        "note_id": "note_roundtrip",
        "entity_type": "stock_hub",
        "title": "Roundtrip Note",
        "ticker": "005930",
        "custom_flag": "important",
        "nested_scores": {"growth": 85, "safety": 90},
    }
    body = "# Note Title\n\nParagraph 1.\n\n- Bullet A\n- Bullet B\n"

    serialized = dump_note(meta, body)

    # Re-parse serialized note
    re_meta, re_body, issues = parse_note(serialized)
    assert len(issues) == 0
    assert re_meta["ticker"] == "005930"
    assert re_meta["custom_flag"] == "important"
    assert re_meta["nested_scores"] == {"growth": 85, "safety": 90}
    assert re_body == body.strip()
