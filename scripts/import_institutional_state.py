"""Dry-run-first importer for legacy institutional JSON state."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api.config import get_state_db_path


_REQUIRED_FIELDS = {
    "routine": ("execution_id",),
    "catalyst": ("catalyst_id", "entity_id"),
    "estimate": ("estimate_id", "ticker", "period", "metric_name", "estimate_type", "value"),
    "source": ("source_id", "captured_at_utc", "source_locator"),
}


def _records_from_json(value: Any, family: str) -> list[dict[str, Any]]:
    if isinstance(value, list):
        records = value
    elif isinstance(value, dict):
        candidates = {
            "routine": ("records", "execution_history", "routine_history", "events"),
            "catalyst": ("records", "catalysts"),
            "estimate": ("records", "estimates", "ledger"),
            "source": ("records", "sources", "captures"),
        }[family]
        records = next((value[key] for key in candidates if isinstance(value.get(key), list)), None)
        if records is None:
            records = [value]
    else:
        raise ValueError("legacy JSON must be an object or list of objects")

    if not all(isinstance(record, dict) for record in records):
        raise ValueError("every legacy record must be a JSON object")
    missing = []
    for index, record in enumerate(records):
        absent = [field for field in _REQUIRED_FIELDS[family] if record.get(field) in (None, "")]
        if family == "source" and not (record.get("content") or record.get("content_snippet")):
            absent.append("content/content_snippet")
        if absent:
            missing.append({"record_index": index, "missing_fields": absent})
    if missing:
        raise ValueError(f"legacy records are missing required fields: {json.dumps(missing)}")
    return [dict(record) for record in records]


def inspect_legacy_file(path: Path, family: str) -> tuple[str, list[dict[str, Any]]]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not parse UTF-8 JSON from {path}") from exc
    digest = hashlib.sha256(raw).hexdigest()
    return digest, _records_from_json(value, family)


def import_file(path: Path, family: str, *, database: Path | None = None, artifact_root: Path | None = None) -> dict[str, Any]:
    digest, records = inspect_legacy_file(path, family)
    from api.db.adapters import SqliteInstitutionalLedgerRepositoryAdapter

    kwargs: dict[str, Any] = {}
    if database is not None:
        kwargs["db_path"] = str(database)
    if artifact_root is not None:
        kwargs["artifact_root"] = artifact_root
    counts = SqliteInstitutionalLedgerRepositoryAdapter(**kwargs).import_legacy_records(
        digest, family, records
    )
    return {
        "source": str(path.resolve()),
        "source_sha256": digest,
        "family": family,
        "record_count": len(records),
        **counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=sorted(_REQUIRED_FIELDS))
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--database", type=Path, default=Path(get_state_db_path()))
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--apply", action="store_true", help="write validated records; omitted means read-only dry run")
    args = parser.parse_args()
    source = args.source.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"legacy source file does not exist: {source}")

    digest, records = inspect_legacy_file(source, args.family)
    result: dict[str, Any] = {
        "source": str(source),
        "source_sha256": digest,
        "family": args.family,
        "record_count": len(records),
        "applied": False,
    }
    if args.apply:
        result.update(import_file(
            source,
            args.family,
            database=args.database.resolve(),
            artifact_root=args.artifact_root.resolve() if args.artifact_root else None,
        ))
        result["applied"] = True
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
