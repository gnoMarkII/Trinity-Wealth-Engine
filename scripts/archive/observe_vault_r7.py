"""Read-only post-release observation for the R7 Vault contract.

Unlike the older stability observer, this runner does not load the embedding
model or issue search queries. It samples the R7 preflight and validates the
active identity/profile surface for the requested wall-clock duration. A
valid normal workflow may change the tree; only contract violations fail the
observation.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.archive.run_vault_r7_preflight import _iter_markdown, collect  # noqa: E402
from tools.archivist.metadata import parse_note, validate_capture_note, validate_note  # noqa: E402


ARCHIVE_PARTS = {".backups", "Revisions", "40_Archive"}
TEMPLATE_PARTS = {"99_Templates"}


def _relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _active_violations(root: Path) -> list[dict[str, Any]]:
    note_ids: dict[str, list[str]] = {}
    document_keys: dict[str, list[str]] = {}
    violations: list[dict[str, Any]] = []

    for path in _iter_markdown(root):
        relative = _relative(root, path)
        parts = set(path.relative_to(root).parts)
        if parts & ARCHIVE_PARTS or parts & TEMPLATE_PARTS:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            violations.append({"path": relative, "code": "unreadable", "detail": str(exc)})
            continue

        metadata, _body, parse_issues = parse_note(text)
        if parse_issues:
            violations.append({"path": relative, "code": "parse_error", "detail": parse_issues})

        capture_like = relative == "00_Inbox/index.md" or relative.startswith("00_Inbox/")
        raw_entity = str(metadata.get("entity_type") or "").strip().lower()
        if capture_like or raw_entity == "capture":
            ok, issues = validate_capture_note(metadata)
            if not ok:
                violations.append({"path": relative, "code": "capture_profile", "detail": issues})
            continue
        if raw_entity == "navigation" or metadata.get("document_role") == "navigation":
            continue

        ok, issues = validate_note(metadata, mode="strict")
        if not ok:
            violations.append({"path": relative, "code": "metadata_contract", "detail": issues})
        if "[[" in text or "![[" in text or "obsidian://" in text:
            violations.append({"path": relative, "code": "non_portable_link", "detail": "Obsidian-only link syntax"})

        if metadata.get("note_id"):
            note_ids.setdefault(str(metadata["note_id"]), []).append(relative)
        if metadata.get("document_key"):
            document_keys.setdefault(str(metadata["document_key"]), []).append(relative)

    for value, paths in sorted(note_ids.items()):
        if len(paths) > 1:
            violations.append({"code": "active_duplicate_note_id", "value": value, "paths": paths})
    for value, paths in sorted(document_keys.items()):
        if len(paths) > 1:
            violations.append({"code": "active_duplicate_document_key", "value": value, "paths": paths})
    return violations


def _sample(root: Path) -> dict[str, Any]:
    report = collect(root)
    active_violations = _active_violations(root)
    violations: list[dict[str, Any]] = []
    if report.get("missing"):
        violations.append({"code": "preflight_missing", "detail": report["missing"]})
    if report.get("issues"):
        violations.append({"code": "preflight_issues", "detail": report["issues"]})
    if report.get("wikilink_count"):
        violations.append({"code": "preflight_wikilinks", "detail": report["wikilink_count"]})
    violations.extend(active_violations)
    return {
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "tree_fingerprint": report.get("tree_fingerprint"),
        "markdown_count": report.get("markdown_count"),
        "profiles": report.get("profiles", {}),
        "missing": report.get("missing", {}),
        "issues": report.get("issues", {}),
        "wikilink_count": report.get("wikilink_count", 0),
        "active_violation_count": len(active_violations),
        "violations": violations,
    }


def observe(vault: Path, output: Path, *, duration_seconds: int, interval_seconds: int) -> dict[str, Any]:
    if duration_seconds <= 0 or interval_seconds <= 0 or interval_seconds > 600:
        raise ValueError("duration_seconds must be positive and interval_seconds must be between 1 and 600")
    root = vault.resolve()
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    first = _sample(root)
    samples = [first]
    events: list[dict[str, Any]] = []

    def write_progress(status: str) -> None:
        progress = {
            "status": status,
            "started_at": started.isoformat(),
            "elapsed_seconds": (datetime.now(timezone.utc) - started).total_seconds(),
            "samples": len(samples),
            "event_count": len(events),
            "latest": samples[-1] if samples else None,
        }
        output.with_suffix(output.suffix + ".progress.json").write_text(
            json.dumps(progress, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    write_progress("RUNNING")
    deadline = time.monotonic() + duration_seconds
    previous = first
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(interval_seconds, remaining))
        current = _sample(root)
        current["tree_changed_since_previous"] = current["tree_fingerprint"] != previous["tree_fingerprint"]
        samples.append(current)
        if current["violations"]:
            events.append({"observed_at": current["observed_at"], "violations": current["violations"]})
        previous = current
        write_progress("RUNNING")

    ended = datetime.now(timezone.utc)
    result = {
        "schema": "vault-r7-observation-v1",
        "status": "PASS" if not first["violations"] and not events else "FAIL",
        "reason": None if not first["violations"] and not events else "active R7 contract violation observed",
        "started_at": started.isoformat(),
        "ended_at": ended.isoformat(),
        "duration_seconds": (ended - started).total_seconds(),
        "requested_duration_seconds": duration_seconds,
        "interval_seconds": interval_seconds,
        "sample_count": len(samples),
        "event_count": len(events),
        "events": events,
        "baseline_tree_fingerprint": first["tree_fingerprint"],
        "final_tree_fingerprint": samples[-1]["tree_fingerprint"],
        "tree_changed_sample_count": sum(1 for item in samples[1:] if item.get("tree_changed_since_previous")),
        "initial": first,
        "final": samples[-1],
        "samples": samples,
        "read_only": True,
        "model_loading": False,
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_progress(result["status"])
    print(json.dumps({key: result[key] for key in ("status", "duration_seconds", "sample_count", "event_count", "tree_changed_sample_count")}, ensure_ascii=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", type=Path, default=Path("memories"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-seconds", type=int, default=3600)
    parser.add_argument("--interval-seconds", type=int, default=60)
    args = parser.parse_args()
    result = observe(args.vault, args.output, duration_seconds=args.duration_seconds, interval_seconds=args.interval_seconds)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
