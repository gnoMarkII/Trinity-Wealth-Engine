"""Trust- and citation-gated answer contract for AI consumers of the vault."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

from tools.archivist.catalog_adapter import SqliteNoteCatalogAdapter
from tools.archivist.metadata import parse_note
from tools.archivist.schema_registry import load_default_registry


class AnswerContractError(RuntimeError):
    """Raised when an answer cannot be grounded in current vault evidence."""


def _content_sha256(path: Path) -> str:
    # The catalog hashes the decoded UTF-8 note content.  Using the same
    # logical-content hash here keeps the contract stable across Windows
    # CRLF and Unix LF line endings while still detecting note changes.
    content = path.read_text(encoding="utf-8")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def collect_evidence(
    vault_root: str | Path,
    catalog: SqliteNoteCatalogAdapter,
    documents: Iterable[Any],
    *,
    retrieval_namespace: str = "primary",
) -> list[dict[str, Any]]:
    """Join retrieved documents to active catalog rows and current file hashes."""
    root = Path(vault_root).resolve()
    evidence: list[dict[str, Any]] = []
    seen: set[str] = set()
    registry = load_default_registry()
    namespace = str(retrieval_namespace or "primary").strip().lower()
    for document in documents:
        metadata = getattr(document, "metadata", {}) or {}
        rel = str(metadata.get("relative_path") or metadata.get("source") or "").replace("\\", "/").strip("/")
        if not rel or rel in seen:
            continue
        seen.add(rel)
        entry = catalog.get_by_path(rel)
        path = root / rel
        if entry is None or not path.is_file():
            continue
        current_hash = _content_sha256(path)
        if current_hash != str(entry.content_sha256):
            continue
        note_meta, _, issues = parse_note(path.read_text(encoding="utf-8"))
        if issues:
            continue
        if not registry.is_index_eligible(note_meta, vector=False):
            continue
        sensitivity = str(note_meta.get("sensitivity") or "internal").lower()
        if namespace == "primary" and sensitivity not in {"public", "internal"}:
            continue
        if namespace not in {"primary", "all"} and sensitivity != namespace:
            continue
        lifecycle = str(note_meta.get("content_status") or "published").lower()
        if lifecycle not in {"reviewed", "published"}:
            continue
        verification = str(
            note_meta.get("verification_status")
            or note_meta.get("content_verification_status")
            or ""
        ).lower()
        if verification == "disputed":
            continue
        evidence.append(
            {
                "relative_path": rel,
                "note_id": entry.note_id,
                "document_key": entry.document_key,
                "title": str(note_meta.get("title") or entry.title),
                "content_sha256": current_hash,
                "revision_id": str(
                    note_meta.get("current_revision_id")
                    or note_meta.get("revision_id")
                    or getattr(entry, "current_revision_id", None)
                    or ""
                ),
                "retrieval_namespace": namespace,
                "sensitivity": sensitivity,
                "search_scope": str(note_meta.get("search_scope") or "included"),
                "content_status": str(note_meta.get("content_status") or "published"),
                "trust_tier": str(note_meta.get("trust_tier") or "T3"),
                "production_eligible": bool(note_meta.get("production_eligible", False)),
                "source_verification_status": str(note_meta.get("source_verification_status") or "not_reviewed"),
                "content_verification_status": str(note_meta.get("content_verification_status") or "not_reviewed"),
                "evidence_refs": list(note_meta.get("evidence_refs") or []),
                "retrieval_method": str(metadata.get("retrieval_method") or ""),
                "content_kind": str(metadata.get("content_kind") or ""),
                "content_truncated": bool(metadata.get("content_truncated", False)),
                "content_total_chars": int(metadata.get("content_total_chars") or 0),
                "matched_chunk_index": metadata.get("matched_chunk_index"),
                "matched_content_sha256": hashlib.sha256(
                    str(getattr(document, "page_content", "") or "").encode("utf-8")
                ).hexdigest(),
            }
        )
    return evidence


def validate_answer(
    answer: str,
    evidence: list[dict[str, Any]],
    *,
    cited_paths: Iterable[str],
    production_mode: bool = False,
    retrieval_namespace: str = "primary",
) -> dict[str, Any]:
    """Validate a proposed answer before it is shown as grounded output."""
    normalized_paths = [str(path).replace("\\", "/").strip("/") for path in cited_paths]
    evidence_paths = {str(item.get("relative_path") or "") for item in evidence}
    missing = sorted(set(normalized_paths) - evidence_paths)
    ungrounded = not str(answer or "").strip()
    eligible = [
        item
        for item in evidence
        if item.get("production_eligible")
        and item.get("trust_tier") in {"T1", "T2"}
        and item.get("source_verification_status") == "verified"
        and item.get("content_verification_status") == "verified"
    ]
    blocked_reasons: list[str] = []
    if ungrounded:
        blocked_reasons.append("empty_answer")
    if not evidence:
        blocked_reasons.append("no_current_catalog_evidence")
    if missing:
        blocked_reasons.append("citation_not_joined_to_evidence")
    expected_namespace = str(retrieval_namespace or "primary").lower()
    namespace_mismatch = sorted(
        {
            str(item.get("relative_path") or "")
            for item in evidence
            if expected_namespace not in {"all", str(item.get("retrieval_namespace") or "primary").lower()}
        }
    )
    if namespace_mismatch:
        blocked_reasons.append("retrieval_namespace_mismatch")
    incomplete_citations = sorted(
        {
            str(item.get("relative_path") or "")
            for item in evidence
            if not item.get("note_id") or not item.get("revision_id") or not item.get("content_sha256")
        }
    )
    if incomplete_citations:
        blocked_reasons.append("citation_identity_or_hash_incomplete")
    if production_mode and not eligible:
        blocked_reasons.append("production_trust_gate_not_met")
    status = "BLOCKED" if blocked_reasons else "PASS"
    return {
        "status": status,
        "production_mode": production_mode,
        "answer": answer if status == "PASS" else None,
        "citations": [item for item in evidence if item.get("relative_path") in normalized_paths],
        "evidence_count": len(evidence),
        "eligible_evidence_count": len(eligible),
        "missing_citations": missing,
        "namespace_mismatch": namespace_mismatch,
        "incomplete_citations": incomplete_citations,
        "blocked_reasons": blocked_reasons,
        "limitations": [] if status == "PASS" else [
            "This vault contains unreviewed or unavailable provenance; do not convert this result into a production decision."
        ],
    }


def format_answer(
    answer: str,
    evidence: list[dict[str, Any]],
    *,
    cited_paths: Iterable[str],
    production_mode: bool = False,
    retrieval_namespace: str = "primary",
) -> str:
    """Return a stable JSON response envelope suitable for an AI tool call."""
    import json

    return json.dumps(
        validate_answer(
            answer,
            evidence,
            cited_paths=cited_paths,
            production_mode=production_mode,
            retrieval_namespace=retrieval_namespace,
        ),
        ensure_ascii=False,
        indent=2,
    )
