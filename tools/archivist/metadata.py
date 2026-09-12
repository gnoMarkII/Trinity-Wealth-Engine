"""Metadata Parser, Validator, and Normalizer for Obsidian Vault V2.

Implements safe frontmatter extraction, schema validation, legacy metadata normalization,
and round-trip serialization without data loss.
"""
from __future__ import annotations

import io
import re
from datetime import date, datetime
from typing import Any, Optional

import frontmatter as fm
import yaml

from schemas.knowledge_metadata import (
    CURRENT_SCHEMA_VERSION,
    BriefingBookMetadata,
    BookNoteMetadata,
    CommonNoteMetadata,
    CompanyNewsMetadata,
    EarningsCallMetadata,
    EquityAnalysisMetadata,
    GenericKnowledgeMetadata,
    MacroSnapshotMetadata,
    MacroStrategyMetadata,
    QuantSnapshotMetadata,
    StockHubMetadata,
    YouTubeSummaryMetadata,
)

# Registry mapping canonical entity_type strings to Pydantic schema classes
_TYPE_TO_MODEL = {
    "stock_hub": StockHubMetadata,
    "equity_analysis": EquityAnalysisMetadata,
    "quant_snapshot": QuantSnapshotMetadata,
    "earnings_call": EarningsCallMetadata,
    "company_news": CompanyNewsMetadata,
    "youtube_summary": YouTubeSummaryMetadata,
    "macro_strategy": MacroStrategyMetadata,
    "macro_snapshot": MacroSnapshotMetadata,
    "briefing_book": BriefingBookMetadata,
    "book_note": BookNoteMetadata,
    "concept": GenericKnowledgeMetadata,
    "capture": GenericKnowledgeMetadata,
    "portfolio_state": GenericKnowledgeMetadata,
    "holding": GenericKnowledgeMetadata,
    "watchlist_item": GenericKnowledgeMetadata,
    "goal": GenericKnowledgeMetadata,
}

# Legacy type alias normalization mapping
_LEGACY_TYPE_MAP = {
    "company_news": "company_news",
    "article": "company_news",
    "article_note": "company_news",
    "news_radar": "company_news",
    "equity analysis": "equity_analysis",
    "equity_analysis": "equity_analysis",
    "financial_trends": "equity_analysis",
    "financial_health": "equity_analysis",
    "stock_momentum": "equity_analysis",
    "analyst_consensus": "equity_analysis",
    "stock_entity": "stock_hub",
    "stock": "stock_hub",
    "company_entity": "stock_hub",
    "stock_hub": "stock_hub",
    "youtube_insight": "youtube_summary",
    "youtube_summary": "youtube_summary",
    "macro_strategy": "macro_strategy",
    "macro_daily": "macro_snapshot",
    "macro_global": "macro_snapshot",
    "regional_macro": "macro_snapshot",
    "macro_regional": "macro_snapshot",
    "macro_country": "macro_snapshot",
    "us_sectors_pulse": "macro_snapshot",
    "economic_fundamentals": "macro_snapshot",
    "equity_quant_snapshot": "quant_snapshot",
    "quant_snapshot": "quant_snapshot",
    "macro_event": "concept",
    "concept_stub": "concept",
    "company": "stock_hub",
    "briefing_book": "briefing_book",
    "book": "book_note",
}


def parse_note(text: str) -> tuple[dict[str, Any], str, list[dict[str, str]]]:
    """Safely extracts YAML frontmatter and body from markdown text.

    Returns:
        (metadata_dict, body_text, issues_list)
        If YAML is malformed, issues contains a parse error, and metadata remains empty
        to prevent accidental overwrite of broken notes.
    """
    issues: list[dict[str, str]] = []
    text = text or ""

    if not text.startswith("---"):
        return {}, text.strip(), issues

    parts = text.split("---", 2)
    if len(parts) < 3:
        issues.append({
            "code": "malformed_frontmatter",
            "field": "frontmatter",
            "reason": "Unclosed frontmatter block (missing terminating ---)",
        })
        return {}, text.strip(), issues

    yaml_block = parts[1]
    body = parts[2].strip()

    try:
        # Load using SafeLoader with duplicate key checking
        class UniqueKeyLoader(yaml.SafeLoader):
            pass

        def construct_mapping(loader, node, deep=False):
            mapping = {}
            for key_node, value_node in node.value:
                key = loader.construct_object(key_node, deep=deep)
                if key in mapping:
                    raise yaml.constructor.ConstructorError(
                        "while constructing a mapping",
                        node.start_mark,
                        f"found duplicate key '{key}'",
                        key_node.start_mark,
                    )
                mapping[key] = loader.construct_object(value_node, deep=deep)
            return mapping

        UniqueKeyLoader.add_constructor(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
            construct_mapping,
        )

        data = yaml.load(yaml_block, Loader=UniqueKeyLoader)
        if not isinstance(data, dict):
            issues.append({
                "code": "invalid_frontmatter_type",
                "field": "frontmatter",
                "reason": f"Expected dict in frontmatter, got {type(data).__name__}",
            })
            return {}, body, issues

        return data, body, issues

    except Exception as e:
        issues.append({
            "code": "yaml_syntax_error",
            "field": "frontmatter",
            "reason": str(e),
        })
        return {}, body, issues


def normalize_legacy_metadata(
    meta: dict[str, Any],
    producer: Optional[str] = None,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Normalizes legacy field names, casing, and types while preserving custom properties."""
    normalized = dict(meta)
    issues: list[dict[str, str]] = []

    # 1. Normalize entity_type
    raw_type = normalized.get("entity_type") or normalized.get("type")
    if raw_type:
        raw_type_str = str(raw_type).strip()
        canon_type = _LEGACY_TYPE_MAP.get(raw_type_str.lower(), raw_type_str.lower())
        normalized["entity_type"] = canon_type

    # 2. Normalize ticker / tickers
    # Ensure ticker strings (like "005930") remain strings and not converted to numbers
    if "ticker" in normalized and normalized["ticker"] is not None:
        normalized["ticker"] = str(normalized["ticker"]).strip().upper()

    if "tickers" in normalized and normalized["tickers"] is not None:
        t_val = normalized["tickers"]
        if isinstance(t_val, list):
            normalized["tickers"] = [str(t).strip().upper() for t in t_val if t]
        elif isinstance(t_val, str):
            normalized["tickers"] = [str(t_val).strip().upper()]

    # 3. Normalize dates
    # published_at date-only -> published_date
    if "published_at" in normalized:
        p_at = str(normalized["published_at"]).strip()
        if re.match(r"^\d{4}-\d{2}-\d{2}$", p_at):
            normalized.setdefault("published_date", p_at)
            normalized.setdefault("date", p_at)

    # 4. Normalize author -> authors
    if "author" in normalized and "authors" not in normalized:
        author_val = normalized["author"]
        if isinstance(author_val, str) and author_val.strip():
            normalized["authors"] = [author_val.strip()]
        elif isinstance(author_val, list):
            normalized["authors"] = [str(a).strip() for a in author_val if a]

    # 5. Normalize verification statuses
    # Legacy 'verified: true' means source verified, NOT content fact-checked
    if "verified" in normalized:
        is_verified = bool(normalized["verified"])
        if is_verified:
            normalized.setdefault("source_verification_status", "verified")
            normalized.setdefault("content_verification_status", "not_reviewed")
        else:
            normalized.setdefault("source_verification_status", "unverified")
            normalized.setdefault("content_verification_status", "not_reviewed")

    # 6. Ensure schema_version
    if "schema_version" not in normalized:
        normalized["schema_version"] = CURRENT_SCHEMA_VERSION

    # PyYAML's safe loader materializes ISO dates as ``date``/``datetime``
    # objects.  The cross-app contract stores them as portable ISO strings so
    # specialized schemas such as BookNoteMetadata.date_read do not reject a
    # legacy note during repair.
    for key, value in list(normalized.items()):
        if isinstance(value, (date, datetime)):
            normalized[key] = value.isoformat()

    return normalized, issues


def validate_note(
    meta: dict[str, Any],
    mode: str = "strict",
) -> tuple[Optional[CommonNoteMetadata], list[dict[str, str]]]:
    """Validates note metadata against Pydantic models.

    Args:
        meta: Parsed frontmatter dictionary
        mode: 'strict' for new writes (fails on missing required fields),
              'legacy' for reading existing notes (returns model or dict with issues)
    """
    # ``lenient`` was the public reader spelling used by the catalog/search
    # adapters before R7 named the modes explicitly.  Keep it as a
    # read-compatible alias so convergence does not break existing readers.
    if mode == "lenient":
        mode = "legacy"
    if mode not in {"strict", "legacy", "read_only"}:
        raise ValueError(f"Unsupported metadata validation mode: {mode!r}")
    issues: list[dict[str, str]] = []
    validation_meta = dict(meta)
    raw_type = validation_meta.get("entity_type", "concept")
    canonical_type = _LEGACY_TYPE_MAP.get(str(raw_type).strip().lower(), str(raw_type).strip().lower())
    validation_meta["entity_type"] = canonical_type
    schema_ver = validation_meta.get("schema_version", 1)

    # Check for newer unsupported schema version (e.g. schema_version=3)
    if isinstance(schema_ver, int) and schema_ver > CURRENT_SCHEMA_VERSION:
        issues.append({
            "code": "unsupported_schema",
            "field": "schema_version",
            "reason": f"Schema version {schema_ver} is newer than supported {CURRENT_SCHEMA_VERSION}. Treated as read-only.",
        })
        try:
            future_meta = dict(validation_meta)
            # Future-schema notes are read-only.  Preserve the ability to
            # inspect an older/future note even when this runtime predates the
            # document_key field, without silently persisting a fabricated key.
            future_meta.setdefault("document_key", f"legacy:{future_meta.get('note_id') or future_meta.get('title') or 'unknown'}")
            model = CommonNoteMetadata.model_validate(future_meta)
            return model, issues
        except Exception:
            return None, issues

    if mode == "strict":
        missing: list[str] = []
        for field in ("schema_version", "note_id", "document_key", "entity_type", "title"):
            value = validation_meta.get(field)
            if value is None or (isinstance(value, str) and not value.strip()):
                missing.append(field)
        if missing:
            issues.append({
                "code": "missing_required",
                "field": ",".join(missing),
                "reason": f"Published V2 notes require: {', '.join(missing)}",
            })
            return None, issues
        if schema_ver != CURRENT_SCHEMA_VERSION:
            issues.append({
                "code": "unsupported_schema",
                "field": "schema_version",
                "reason": f"Writes require schema_version {CURRENT_SCHEMA_VERSION}, got {schema_ver!r}.",
            })
            return None, issues
    elif not validation_meta.get("document_key"):
        # Lenient readers may inspect legacy notes.  Use an in-memory marker so
        # the model remains useful, but never write it back automatically.
        issues.append({
            "code": "needs_review",
            "field": "document_key",
            "reason": "Legacy note has no document_key; it is read-only until repaired.",
        })
        validation_meta["document_key"] = f"legacy:{validation_meta.get('note_id') or validation_meta.get('title') or 'unknown'}"

    # Identify entity_type model
    model_cls = _TYPE_TO_MODEL.get(canonical_type)

    if not model_cls:
        issues.append({
            "code": "unknown_type",
            "field": "entity_type",
            "reason": f"Unknown entity_type '{raw_type}'. Falling back to generic note.",
        })
        if mode == "strict":
            return None, issues
        model_cls = GenericKnowledgeMetadata

    try:
        validated = model_cls.model_validate(validation_meta)
        return validated, issues
    except Exception as e:
        if mode == "strict":
            issues.append({
                "code": "validation_error",
                "field": "metadata",
                "reason": str(e),
            })
            return None, issues
        else:
            # In legacy mode, attempt fallback to CommonNoteMetadata to allow reading with warnings
            issues.append({
                "code": "needs_review",
                "field": "metadata",
                "reason": f"Legacy metadata incomplete: {e}",
            })
            try:
                fallback = CommonNoteMetadata.model_validate(validation_meta)
                return fallback, issues
            except Exception:
                return None, issues


def validate_publish_request(
    meta: dict[str, Any],
) -> tuple[Optional[CommonNoteMetadata], list[dict[str, str]]]:
    """Validate metadata that is about to become a published V2 note."""
    return validate_note(meta, mode="strict")


def validate_capture_note(meta: dict[str, Any]) -> tuple[bool, list[dict[str, str]]]:
    """Validate the intentionally identity-free capture profile."""
    issues: list[dict[str, str]] = []
    required = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "entity_type": "capture",
        "capture_status": "pending_normalization",
        "search_scope": "excluded",
    }
    for field, expected in required.items():
        if meta.get(field) != expected:
            issues.append({
                "code": "capture_profile_error",
                "field": field,
                "reason": f"Capture requires {field}={expected!r}.",
            })
    for field in ("title", "captured_at", "capture_source"):
        if not str(meta.get(field) or "").strip():
            issues.append({
                "code": "missing_required",
                "field": field,
                "reason": f"Capture requires {field}.",
            })
    for field in ("note_id", "document_key"):
        if field in meta and meta.get(field) not in (None, ""):
            issues.append({
                "code": "forbidden_identity",
                "field": field,
                "reason": "Capture notes must remain identity-free until normalization.",
            })
    return not issues, issues


def validate_template_output(meta: dict[str, Any]) -> tuple[bool, list[dict[str, str]]]:
    """Ensure a template source has no real or placeholder identity."""
    issues: list[dict[str, str]] = []
    for field in ("note_id", "document_key"):
        value = str(meta.get(field) or "").strip()
        if value:
            issues.append({
                "code": "forbidden_identity",
                "field": field,
                "reason": "Templates must not carry an allocated identity.",
            })
    placeholder_re = re.compile(r"(?:TODO|TBD|\{\{|\}\}|<[^>]+>)", re.IGNORECASE)
    for field in ("title", "entity_type"):
        if placeholder_re.search(str(meta.get(field) or "")):
            issues.append({
                "code": "placeholder_identity",
                "field": field,
                "reason": "Template identity placeholders must be declared by the template adapter.",
            })
    return not issues, issues


def dump_note(meta: dict[str, Any] | CommonNoteMetadata, body: str) -> str:
    """Serializes metadata and body back to clean Markdown with YAML frontmatter.

    Guarantees that custom fields, lists, and unicode text are preserved.
    """
    if isinstance(meta, CommonNoteMetadata):
        data = meta.model_dump(mode="python", exclude_none=True)
    else:
        data = dict(meta)

    # Custom YAML dumper to preserve clean multiline output and UTF-8 strings
    stream = io.StringIO()
    yaml.dump(
        data,
        stream,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    )
    fm_text = stream.getvalue().strip()

    clean_body = body.strip()
    return f"---\n{fm_text}\n---\n\n{clean_body}\n"
