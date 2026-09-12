"""Apply the frozen R6 path/link migration with an active maintenance lease."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.archivist.maintenance_guard import (  # noqa: E402
    MaintenanceLeaseError,
    assert_write_allowed,
    load_maintenance_lease,
)
from tools.archivist.portable_links import render_relative_markdown_link  # noqa: E402

from scripts.build_multi_app_migration_r6 import (  # noqa: E402
    FENCE_RE,
    WIKILINK_RE,
    _split_link,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _assert_baseline_unchanged(vault: Path, run_dir: Path) -> None:
    """Refuse to mutate a live tree that drifted after R6 preflight."""
    inventory_path = run_dir / "baseline-inventory.jsonl"
    expected_rows = _load_jsonl(inventory_path)
    if not expected_rows:
        raise RuntimeError("R6 baseline inventory is missing or empty")
    ignored = {".system/maintenance.json"}
    expected = {
        str(row["relative_path"]): str(row["sha256"])
        for row in expected_rows
        if str(row.get("relative_path")) not in ignored
    }
    actual: dict[str, str] = {}
    for path in vault.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(vault).as_posix()
        if rel in ignored:
            continue
        actual[rel] = _sha256(path)
    missing = sorted(set(expected) - set(actual))
    added = sorted(set(actual) - set(expected))
    changed = sorted(rel for rel in set(expected) & set(actual) if expected[rel] != actual[rel])
    if missing or added or changed:
        preview = {
            "missing": missing[:20],
            "added": added[:20],
            "changed": changed[:20],
            "missing_count": len(missing),
            "added_count": len(added),
            "changed_count": len(changed),
        }
        raise RuntimeError(f"R6 baseline drift detected before mutation: {json.dumps(preview, ensure_ascii=False)}")


def _assert_lease(vault: Path, owner: str) -> None:
    lease = load_maintenance_lease(vault)
    if lease is None or not lease.is_active() or lease.owner != owner:
        raise MaintenanceLeaseError(f"R6 apply requires active lease owned by {owner!r}")
    assert_write_allowed(vault / ".system" / "storage_contract.json", owner=owner)


def _split_frontmatter(text: str) -> tuple[str, str]:
    if not text.startswith("---"):
        return "", text
    end = text.find("\n---", 3)
    if end < 0:
        return "", text
    end += len("\n---")
    return text[:end], text[end:]


def _rewrite_body(body: str, replacements: dict[str, str]) -> tuple[str, int]:
    output: list[str] = []
    changed = 0
    in_fence = False
    fence_char = ""
    for line in body.splitlines(keepends=True):
        fence = FENCE_RE.match(line)
        if fence:
            marker = fence.group(1)
            if not in_fence:
                in_fence = True
                fence_char = marker[0]
            elif marker[0] == fence_char:
                in_fence = False
            output.append(line)
            continue
        if in_fence:
            output.append(line)
            continue

        parts = re.split(r"(`+)", line)
        for index in range(0, len(parts), 2):
            before = parts[index]
            def replace(match: re.Match[str]) -> str:
                nonlocal changed
                raw = match.group(0)
                replacement = replacements.get(raw)
                if replacement is None:
                    return raw
                changed += 1
                return replacement
            parts[index] = WIKILINK_RE.sub(replace, before)
        output.append("".join(parts))
    return "".join(output), changed


def _replacement(row: dict[str, Any]) -> str:
    target = row.get("final_target")
    if not target:
        raise ValueError(f"missing final target for {row}")
    alias = row.get("alias") or Path(str(row.get("target") or target)).stem
    return render_relative_markdown_link(
        Path(row["vault_root"]),
        row["source_final_path"],
        target,
        label=str(alias),
        fragment=row.get("fragment"),
    )


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".r6tmp", dir=str(path.parent))
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def apply(vault: Path, run_dir: Path, *, owner: str) -> dict[str, Any]:
    vault = vault.resolve()
    run_dir = run_dir.resolve()
    _assert_lease(vault, owner)
    _assert_baseline_unchanged(vault, run_dir)
    plan = json.loads((run_dir / "migration-plan.json").read_text(encoding="utf-8"))
    if plan.get("status") != "PASS":
        raise RuntimeError("refusing to apply a blocked R6 migration plan")
    path_rows = _load_jsonl(run_dir / "path-map.jsonl")
    link_rows = _load_jsonl(run_dir / "link-rewrite-plan.jsonl")
    if any(row.get("final_target") is None for row in link_rows):
        raise RuntimeError("refusing to apply link plan with unresolved targets")

    by_source: dict[str, list[dict[str, Any]]] = {}
    for row in link_rows:
        row = dict(row)
        row["vault_root"] = str(vault)
        by_source.setdefault(str(row["source_path"]), []).append(row)

    staged: dict[str, str] = {}
    for source_old, rows in by_source.items():
        source = vault / source_old
        if not source.is_file():
            raise FileNotFoundError(source)
        replacements: dict[str, str] = {}
        for row in rows:
            replacement = _replacement(row)
            raw = str(row["raw"])
            previous = replacements.setdefault(raw, replacement)
            if previous != replacement:
                raise RuntimeError(f"same raw link resolves differently in {source_old}: {raw}")
        text = source.read_text(encoding="utf-8")
        frontmatter, body = _split_frontmatter(text)
        rewritten_body, changed = _rewrite_body(body, replacements)
        expected = len(rows)
        if changed != expected:
            raise RuntimeError(f"rewrite count mismatch for {source_old}: expected {expected}, got {changed}")
        staged[source_old] = frontmatter + rewritten_body

    journal: list[dict[str, Any]] = []
    moved: list[tuple[Path, Path]] = []
    changed_files = sorted(set(staged) | {row["old_path"] for row in path_rows})
    try:
        _assert_lease(vault, owner)
        for row in path_rows:
            old = vault / row["old_path"]
            new = vault / row["new_path"]
            if not old.is_file():
                raise FileNotFoundError(old)
            if _sha256(old) != row["sha256"]:
                raise RuntimeError(f"before hash mismatch: {row['old_path']}")
            if new.exists():
                raise FileExistsError(f"migration target already exists: {new}")
        for row in path_rows:
            old = vault / row["old_path"]
            new = vault / row["new_path"]
            new.parent.mkdir(parents=True, exist_ok=True)
            os.replace(old, new)
            moved.append((old, new))
            journal.append({
                "operation": "rename",
                "old_path": row["old_path"],
                "new_path": row["new_path"],
                "before_sha256": row["sha256"],
            })
        for source_old, text in staged.items():
            final_path = vault / next(
                (row["new_path"] for row in path_rows if row["old_path"] == source_old),
                source_old,
            )
            before_sha = None
            if final_path.is_file():
                before_sha = _sha256(final_path)
            _atomic_write(final_path, text)
            after_sha = _sha256(final_path)
            journal.append({
                "operation": "rewrite",
                "old_path": source_old,
                "new_path": final_path.relative_to(vault).as_posix(),
                "before_sha256": before_sha,
                "after_sha256": after_sha,
                "link_count": len(by_source[source_old]),
            })
    except Exception:
        for old, new in reversed(moved):
            if new.exists() and not old.exists():
                old.parent.mkdir(parents=True, exist_ok=True)
                os.replace(new, old)
        raise

    _write_jsonl(run_dir / "mutation-journal.jsonl", journal)
    result = {
        "status": "PASS",
        "run_id": run_dir.name,
        "path_rename_count": len(path_rows),
        "link_rewrite_file_count": len(staged),
        "link_rewrite_count": len(link_rows),
        "changed_file_count": len(changed_files),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "owner": owner,
    }
    _write_json(run_dir / "path-migration-result.json", result)
    _write_json(run_dir / "link-rewrite-result.json", result)
    print(json.dumps(result, ensure_ascii=False))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", type=Path, default=Path("memories"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--owner", default="codex-vault-r6")
    args = parser.parse_args()
    apply(args.vault, args.run_dir, owner=args.owner)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
