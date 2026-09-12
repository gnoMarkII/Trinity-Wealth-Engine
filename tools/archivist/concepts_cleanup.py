"""Deterministic inventory and disposition rules for the Concepts cleanup.

This module is intentionally read-only.  It produces a snapshot that can be
reviewed, hashed, and passed to a separate maintenance executor; importing it
must never create a note, catalog row, tombstone, or directory.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from tools.archivist.metadata import parse_note
from tools.archivist.portable_links import (
    iter_internal_markdown_links,
    resolve_vault_target_detailed,
)
from tools.archivist.schema_registry import load_default_registry
from tools.archivist.vault_policy import is_searchable_note


AUTO_STUB_MARKERS = (
    "concept stub created automatically",
    "automatically created concept stub",
)
_DATE_RE = re.compile(r"(?<!\d)(20\d{2})[-_](\d{2})[-_](\d{2})(?!\d)")
_WIKILINK_RE = re.compile(r"\[\[(.*?)\]\]")
_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.-]{1,9}(?:\.[A-Z]{1,4})?$")


def _json_safe(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return value


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _body_without_noise(body: str) -> str:
    value = re.sub(r"<!--.*?-->", " ", body or "", flags=re.DOTALL)
    for marker in AUTO_STUB_MARKERS:
        value = re.sub(re.escape(marker), " ", value, flags=re.IGNORECASE)
    value = re.sub(r"^\s{0,3}#{1,6}\s+[^\n]*$", " ", value, flags=re.MULTILINE)
    return re.sub(r"\s+", " ", value).strip()


def _date_hint(relative_path: str, metadata: Mapping[str, Any]) -> Optional[str]:
    match = _DATE_RE.search(Path(relative_path).name)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    for key in ("date", "published_date", "analysis_date", "as_of", "authored_date"):
        raw = str(metadata.get(key) or "").strip()
        match = _DATE_RE.search(raw)
        if match:
            return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
        if re.match(r"^20\d{2}-\d{2}-\d{2}", raw):
            return raw[:10]
    return None


def _is_generated_source(relative_path: str, metadata: Mapping[str, Any]) -> bool:
    parts = set(Path(relative_path).parts)
    role = str(metadata.get("document_role") or "").lower()
    status = str(metadata.get("content_status") or "").lower()
    return bool(
        "00_Index" in parts
        or role in {"navigation", "projection", "derived_artifact"}
        or status == "generated"
        or Path(relative_path).name.casefold() in {"index.md", "home.md"}
    )


def _inbound_bucket(relative_path: str, metadata: Mapping[str, Any]) -> str:
    parts = set(Path(relative_path).parts)
    if "40_Archive" in parts or str(metadata.get("lifecycle_status") or "").lower() in {"retired", "superseded"}:
        return "archive"
    if _is_generated_source(relative_path, metadata):
        return "generated"
    return "real"


@dataclass(frozen=True)
class _Note:
    relative_path: str
    text: str
    body: str
    metadata: dict[str, Any]
    parse_issues: list[dict[str, Any]]
    content_sha256: str
    body_sha256: str


def _load_notes(root: Path) -> dict[str, _Note]:
    notes: dict[str, _Note] = {}
    for path in sorted(root.rglob("*.md"), key=lambda item: item.relative_to(root).as_posix()):
        if ".system" in path.relative_to(root).parts:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        metadata, body, issues = parse_note(text)
        relative = path.relative_to(root).as_posix()
        notes[relative] = _Note(
            relative_path=relative,
            text=text,
            body=body,
            metadata=_json_safe(metadata),
            parse_issues=_json_safe(issues),
            content_sha256=_sha256_text(text),
            body_sha256=_sha256_text(body),
        )
    return notes


def _edges(root: Path, notes: Mapping[str, _Note]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    edges: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for source, note in notes.items():
        raw_links: list[tuple[str, str, Optional[int]]] = [
            (link.destination, "markdown", link.line)
            for link in iter_internal_markdown_links(note.text)
        ]
        raw_links.extend(
            (raw.split("|", 1)[0].split("#", 1)[0].strip(), "wikilink", None)
            for raw in _WIKILINK_RE.findall(note.text)
        )
        seen: set[tuple[str, str]] = set()
        for raw_target, link_type, line in raw_links:
            key = (raw_target, link_type)
            if not raw_target or key in seen:
                continue
            seen.add(key)
            result = resolve_vault_target_detailed(root, raw_target, source=source)
            record = {
                "source": source,
                "raw_target": raw_target,
                "link_type": link_type,
                "line": line,
                "status": result.status,
            }
            if result.status == "resolved" and result.target is not None:
                record["target"] = result.target.relative_to(root).as_posix()
                edges.append(record)
            elif result.status in {"missing", "ambiguous"}:
                if result.candidates:
                    record["candidates"] = [candidate.relative_to(root).as_posix() for candidate in result.candidates]
                unresolved.append(record)
    return edges, unresolved


def _mention_stats(
    relative_path: str,
    title: str,
    aliases: Iterable[Any],
    eligible_notes: Mapping[str, _Note],
) -> dict[str, Any]:
    needles: list[tuple[str, str]] = []
    for label, value in [("title", title), ("stem", Path(relative_path).stem), *[("alias", item) for item in aliases]]:
        clean = str(value or "").strip().casefold()
        if clean and len(clean) >= 3 and (clean, label) not in needles:
            needles.append((clean, label))
    counts = {label: 0 for _, label in needles}
    sources: set[str] = set()
    for source, note in eligible_notes.items():
        if source == relative_path:
            continue
        body = note.body.casefold()
        hit = False
        for needle, label in needles:
            count = body.count(needle)
            counts[label] = counts.get(label, 0) + count
            hit = hit or count > 0
        if hit:
            sources.add(source)
    return {
        "title_exact": counts.get("title", 0),
        "stem_exact": counts.get("stem", 0),
        "alias_exact": counts.get("alias", 0),
        "source_count": len(sources),
        "sources": sorted(sources)[:50],
    }


def _canonical_security_target(root: Path, title: str, metadata: Mapping[str, Any]) -> Optional[str]:
    ticker = str(metadata.get("ticker") or title or "").strip().upper()
    if not _TICKER_RE.fullmatch(ticker):
        return None
    candidates = [
        root / "30_Knowledge_Base" / "Stocks" / ticker / f"{ticker}.md",
        root / "30_Knowledge_Base" / "Crypto" / ticker / f"{ticker}.md",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.relative_to(root).as_posix()
    return None


def scan_concepts(vault_root: str | Path) -> dict[str, Any]:
    """Return a complete, deterministic Concepts inventory and link snapshot."""
    root = Path(vault_root).resolve()
    notes = _load_notes(root)
    edges, unresolved = _edges(root, notes)
    registry = load_default_registry()
    concepts_prefix = Path("30_Knowledge_Base") / "Concepts"
    concept_paths = sorted(
        path for path in notes
        if Path(path).parent == concepts_prefix
    )
    eligible: dict[str, _Note] = {}
    for path, note in notes.items():
        if not is_searchable_note(root / path, vault_root=root):
            continue
        if registry.is_index_eligible(note.metadata, vector=False):
            eligible[path] = note

    inbound: dict[str, list[dict[str, Any]]] = {path: [] for path in notes}
    outbound: dict[str, list[str]] = {path: [] for path in notes}
    for edge in edges:
        source = str(edge["source"])
        target = str(edge["target"])
        if target in inbound:
            bucket = _inbound_bucket(source, notes[source].metadata)
            inbound[target].append({"source": source, "bucket": bucket, "link_type": edge["link_type"]})
        outbound.setdefault(source, []).append(target)

    records: list[dict[str, Any]] = []
    for path in concept_paths:
        note = notes[path]
        metadata = note.metadata
        title = str(metadata.get("title") or Path(path).stem)
        body_clean = _body_without_noise(note.body)
        marker_lower = note.body.casefold()
        auto_stub = any(marker in marker_lower for marker in AUTO_STUB_MARKERS)
        incoming = inbound.get(path, [])
        inbound_real = sorted({item["source"] for item in incoming if item["bucket"] == "real"})
        inbound_generated = sorted({item["source"] for item in incoming if item["bucket"] == "generated"})
        inbound_archive = sorted({item["source"] for item in incoming if item["bucket"] == "archive"})
        mentions = _mention_stats(path, title, metadata.get("aliases") or [], eligible)
        identity_values = [str(metadata.get(key) or "").strip() for key in ("note_id", "document_key")]
        identity_sources = sorted(
            source for source, other in notes.items()
            if source != path
            # Historical snapshots can retain the old identity as evidence,
            # but they are not active manifest/receipt dependencies for the
            # canonical Vault.  Do not let archive-only references block a
            # safe retirement decision.
            and _inbound_bucket(source, other.metadata) != "archive"
            and any(value and value in other.text for value in identity_values)
        )
        records.append(
            {
                "path": path,
                "title": title,
                "note_id": str(metadata.get("note_id") or ""),
                "document_key": str(metadata.get("document_key") or ""),
                "content_sha256": note.content_sha256,
                "body_sha256": note.body_sha256,
                "body_chars": len(note.body),
                "substantive_chars": len(body_clean),
                "has_substantive_content": len(body_clean) >= 80,
                "parse_issues": note.parse_issues,
                "entity_type": str(metadata.get("entity_type") or ""),
                "document_role": str(metadata.get("document_role") or ""),
                "lifecycle_status": str(metadata.get("lifecycle_status") or ""),
                "content_status": str(metadata.get("content_status") or ""),
                "search_scope": str(metadata.get("search_scope") or ""),
                "retention_class": str(metadata.get("retention_class") or ""),
                "trust_tier": str(metadata.get("trust_tier") or ""),
                "production_eligible": bool(metadata.get("production_eligible", False)),
                "verification_status": str(
                    metadata.get("verification_status")
                    or metadata.get("content_verification_status")
                    or ""
                ),
                "review_state": str(metadata.get("review_state") or ""),
                "review_owner": str(metadata.get("review_owner") or ""),
                "review_reason": str(metadata.get("review_reason") or ""),
                "ticker": str(metadata.get("ticker") or ""),
                "date_hint": _date_hint(path, metadata),
                "auto_stub": auto_stub,
                "inbound": incoming,
                "inbound_real": inbound_real,
                "inbound_generated": inbound_generated,
                "inbound_archive": inbound_archive,
                "outbound": sorted(set(outbound.get(path, []))),
                "mention_stats": mentions,
                "identity_reference_sources": identity_sources,
                "canonical_security_target": _canonical_security_target(root, title, metadata),
            }
        )

    fingerprint_rows = [
        {
            "path": path,
            "content_sha256": note.content_sha256,
            "note_id": str(note.metadata.get("note_id") or ""),
            "document_key": str(note.metadata.get("document_key") or ""),
        }
        for path, note in sorted(notes.items())
    ]
    snapshot_fingerprint = _sha256_text(json.dumps(fingerprint_rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    active_unresolved = [
        item
        for item in unresolved
        if _inbound_bucket(str(item.get("source") or ""), notes[str(item["source"])].metadata) != "archive"
    ]
    return {
        "schema_version": 1,
        "vault_relative_scope": "30_Knowledge_Base/Concepts",
        "snapshot_fingerprint": snapshot_fingerprint,
        "total_markdown_files": len(notes),
        "concept_file_count": len(records),
        "eligible_note_count": len(eligible),
        "edge_count": len(edges),
        # Historical revision snapshots are retained evidence, not part of
        # the active canonical graph.  Keep them in the raw report but do not
        # let their old relative links block the active-vault gate.
        "unresolved_edge_count": len(active_unresolved),
        "historical_unresolved_edge_count": len(unresolved) - len(active_unresolved),
        "edges": edges,
        "unresolved_edges": unresolved,
        "concepts": records,
    }


def _route_for(record: Mapping[str, Any]) -> tuple[str, Optional[str], str, float]:
    path = str(record["path"])
    name = Path(path).name
    lower = name.casefold()
    auto_stub = bool(record.get("auto_stub"))
    substantive = bool(record.get("has_substantive_content"))
    real_inbound = list(record.get("inbound_real") or [])
    mentions = record.get("mention_stats") or {}
    mention_sources = int(mentions.get("source_count") or 0)
    identity_sources = list(record.get("identity_reference_sources") or [])
    target = record.get("canonical_security_target")

    if name.casefold() == "test memory.md":
        return "RETIRE", None, "explicit test artifact", 1.0
    if lower.startswith("macro_baseline_"):
        date_hint = record.get("date_hint") or "_undated"
        year_month = str(date_hint)[:7] if date_hint != "_undated" else "_undated"
        return "RELOCATE", f"30_Knowledge_Base/Macroeconomics/Daily_Snapshots/{year_month}/{name}", "macro baseline is stored under Concepts", 0.98
    if lower.startswith("tsla_") and "2026-05-22" in lower:
        return "RELOCATE", f"30_Knowledge_Base/Stocks/TSLA/Analysis/{name}", "TSLA analysis is stored under Concepts", 0.98
    if _DATE_RE.search(name) or "news_youtube_" in lower:
        date_hint = record.get("date_hint") or "_undated"
        year_month = str(date_hint)[:7] if date_hint != "_undated" else "_undated"
        return "RELOCATE", f"30_Knowledge_Base/NotebookLM_Sources/{year_month}/{name}", "dated briefing/source content is stored under Concepts", 0.94
    if auto_stub and substantive:
        return "REVIEW", None, "auto-stub contains substantive content; route cannot be inferred safely", 0.55
    if auto_stub and real_inbound:
        if target:
            return "REDIRECT", target, "referenced security stub has a canonical hub", 0.97
        return "RETIRE", None, "empty generated stub; active Markdown references will be converted to plain text", 0.93
    if auto_stub and (mention_sources >= 2 or identity_sources):
        if identity_sources:
            return "REVIEW", None, "stub has an active identity dependency; data-owner review required", 0.75
        return "RETIRE", None, "empty generated stub; active mentions remain as plain text", 0.90
    if auto_stub:
        return "RETIRE", None, "empty auto-generated stub with no real inbound or identity dependency", 0.99
    if not substantive:
        return "REVIEW", None, "non-generated note has no substantive body", 0.90
    return "KEEP", None, "substantive concept retained pending normal profile review", 0.90


def build_cleanup_plan(snapshot: Mapping[str, Any], *, policy_version: str = "r10-concepts-v1") -> dict[str, Any]:
    """Attach one deterministic disposition to every Concepts inventory row."""
    concepts = []
    counts: dict[str, int] = {}
    for raw in snapshot.get("concepts") or []:
        record = dict(raw)
        disposition, target, reason, confidence = _route_for(record)
        approval_state = "approved" if disposition in {"RETIRE", "RELOCATE"} else "pending"
        record.update(
            {
                "disposition": disposition,
                "target_path": target,
                "reason": reason,
                "confidence": confidence,
                "approval_state": approval_state,
                "apply_eligible": disposition in {"RETIRE", "RELOCATE"},
            }
        )
        concepts.append(record)
        counts[disposition] = counts.get(disposition, 0) + 1
    policy_payload = {
        "policy_version": policy_version,
        "rules": "concepts_cleanup_r10",
        "snapshot_fingerprint": snapshot.get("snapshot_fingerprint"),
    }
    policy_digest = _sha256_text(json.dumps(policy_payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")))
    apply_items = [record for record in concepts if record["apply_eligible"] and record["approval_state"] == "approved"]
    return {
        "schema_version": 1,
        "policy_version": policy_version,
        "policy_digest": policy_digest,
        "snapshot_fingerprint": snapshot.get("snapshot_fingerprint"),
        "concept_file_count": len(concepts),
        "disposition_counts": dict(sorted(counts.items())),
        "apply_item_count": len(apply_items),
        "concepts": concepts,
    }


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(json.dumps(_json_safe(dict(row)), ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


__all__ = ["build_cleanup_plan", "scan_concepts", "write_jsonl"]
