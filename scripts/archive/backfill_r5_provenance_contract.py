"""Add the R5 provenance/trust contract without rewriting note bodies."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.archivist.catalog_adapter import SqliteNoteCatalogAdapter  # noqa: E402
from tools.archivist.catalog_runtime import resolve_catalog_path  # noqa: E402
from tools.archivist.maintenance_guard import assert_write_allowed  # noqa: E402
from tools.archivist.metadata import normalize_legacy_metadata, parse_note  # noqa: E402
from tools.archivist.core import _atomic_write_text  # noqa: E402


OWNER = "codex-vault-r5"
TRUST_TIERS = {"T1", "T2", "T3", "TX"}
SOURCE_REF_KEYS = ("source_url", "url", "source_path", "source_file", "source_key", "source_id")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def _yaml_value(key: str, value: Any) -> str:
    return yaml.safe_dump({key: value}, allow_unicode=True, sort_keys=False, default_flow_style=False).strip()


def _add_frontmatter_fields(text: str, fields: dict[str, Any]) -> str:
    if not fields:
        return text
    if not text.startswith("---"):
        raise ValueError("cannot add provenance fields to a note without frontmatter")
    close = text.find("\n---", 3)
    if close < 0:
        raise ValueError("cannot add provenance fields to malformed frontmatter")
    additions = "\n".join(_yaml_value(key, value) for key, value in fields.items())
    return text[:close] + "\n" + additions + text[close:]


def _desired_fields(metadata: dict[str, Any]) -> dict[str, Any]:
    normalized, _ = normalize_legacy_metadata(metadata)
    unavailable = str(normalized.get("source_unavailable_reason") or "").strip()
    source_status = str(normalized.get("source_verification_status") or "not_reviewed")
    content_status = str(normalized.get("content_verification_status") or "not_reviewed")
    fields: dict[str, Any] = {}

    if "verification_method" not in normalized:
        fields["verification_method"] = "r5-legacy-contract-backfill"
    if "verified_at" not in normalized:
        fields["verified_at"] = None
    if "evidence_refs" not in normalized:
        refs: list[str] = []
        for key in SOURCE_REF_KEYS:
            value = normalized.get(key)
            if isinstance(value, str) and value.strip():
                refs.append(value.strip())
            elif isinstance(value, list):
                refs.extend(str(item).strip() for item in value if str(item).strip())
        fields["evidence_refs"] = list(dict.fromkeys(refs))

    existing_tier = str(normalized.get("trust_tier") or "").strip()
    if existing_tier not in TRUST_TIERS:
        # Do not infer trust from the presence of a URL alone.  Existing R4
        # records are explicitly not_reviewed, so they remain T3 until a
        # human/source verification workflow promotes them.
        fields["trust_tier"] = "TX" if unavailable else "T3"

    production_eligible = bool(normalized.get("production_eligible", False))
    if "production_eligible" not in normalized or (production_eligible and (source_status != "verified" or content_status != "verified")):
        fields["production_eligible"] = False
    if unavailable and fields.get("trust_tier") != "TX" and existing_tier not in TRUST_TIERS:
        fields["trust_tier"] = "TX"
    return fields


def run(vault: Path, run_dir: Path, *, owner: str = OWNER, dry_run: bool = False) -> dict[str, Any]:
    vault = vault.resolve()
    run_dir = run_dir.resolve()
    assert_write_allowed(vault, owner=owner)
    catalog_path = resolve_catalog_path(vault, require_exists=True)
    catalog = SqliteNoteCatalogAdapter(db_path=catalog_path, vault_root=vault, read_only=True)
    records: list[dict[str, Any]] = []
    changed = 0
    skipped = 0
    for entry in catalog.iter_notes(page_size=500, include_non_searchable=True):
        path = vault / entry.relative_path
        if not path.is_file() or path.suffix.lower() != ".md":
            skipped += 1
            continue
        before_text = path.read_text(encoding="utf-8")
        metadata, _, issues = parse_note(before_text)
        if issues or not metadata:
            records.append({"relative_path": entry.relative_path, "status": "blocked", "issues": issues})
            continue
        fields = _desired_fields(metadata)
        record = {
            "relative_path": entry.relative_path,
            "note_id": str(metadata.get("note_id") or entry.note_id),
            "before_sha256": hashlib.sha256(before_text.encode("utf-8")).hexdigest(),
            "fields": fields,
            "status": "unchanged" if not fields else ("planned" if dry_run else "updated"),
        }
        if fields and not dry_run:
            after_text = _add_frontmatter_fields(before_text, fields)
            _atomic_write_text(path, after_text)
            after_meta, _, after_issues = parse_note(after_text)
            if after_issues or not after_meta:
                raise RuntimeError(f"provenance write did not parse: {entry.relative_path}: {after_issues}")
            record["after_sha256"] = hashlib.sha256(after_text.encode("utf-8")).hexdigest()
            changed += 1
        elif fields:
            changed += 1
        records.append(record)

    status = "PASS" if not any(record.get("status") == "blocked" for record in records) else "FAIL"
    result = {
        "status": status,
        "phase": "F06",
        "vault_root": str(vault),
        "catalog_path": str(catalog_path),
        "dry_run": dry_run,
        "scanned": len(records),
        "changed_or_planned": changed,
        "skipped": skipped,
        "blocked": sum(record.get("status") == "blocked" for record in records),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "records": records,
    }
    _write_json(run_dir / ("provenance-backfill-plan.json" if dry_run else "provenance-backfill.json"), result)
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, ensure_ascii=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", type=Path, default=Path("memories"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--owner", default=OWNER)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = run(args.vault, args.run_dir, owner=args.owner, dry_run=args.dry_run)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
