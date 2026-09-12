"""Active R5 observation: read-only queries plus vault/runtime stability."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.archivist.catalog_runtime import catalog_runtime_root  # noqa: E402
from tools.archivist.vector_generation import vector_runtime_path  # noqa: E402


QUERIES = ("อัตราดอกเบี้ย", "interest rate", "FTNT", "inflation")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory(roots: dict[str, Path]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for prefix, root in roots.items():
        if not root.exists():
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            if prefix == "vector_runtime" and "query_cache" in path.relative_to(root).parts:
                continue
            rel = f"{prefix}/{path.relative_to(root).as_posix()}"
            stat = path.stat()
            result[rel] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": None}
    return result


def _hydrate(roots: dict[str, Path], inventory: dict[str, dict[str, Any]], selected: set[str] | None = None) -> None:
    wanted = selected if selected is not None else set(inventory)
    for rel in sorted(wanted):
        item = inventory.get(rel)
        if item is None:
            continue
        prefix, child = rel.split("/", 1)
        path = roots[prefix] / Path(child)
        if path.is_file():
            item["sha256"] = _sha256(path)


def _fingerprint(inventory: dict[str, dict[str, Any]], *, stat: bool = False) -> str:
    digest = hashlib.sha256()
    for rel in sorted(inventory):
        item = inventory[rel]
        value = item["mtime_ns"] if stat else item["sha256"]
        digest.update(f"{rel}\0{item['size']}\0{value}\n".encode("utf-8"))
    return digest.hexdigest()


def _changes(roots: dict[str, Path], before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    changed = {
        rel for rel in set(before) | set(after)
        if rel not in before
        or rel not in after
        or before[rel]["size"] != after[rel]["size"]
        or before[rel]["mtime_ns"] != after[rel]["mtime_ns"]
    }
    _hydrate(roots, after, changed)
    return [{"relative_path": rel, "before": before.get(rel), "after": after.get(rel)} for rel in sorted(changed)]


def _classify_changes(changes: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Separate material mutations from external read-access mtime churn."""
    material: list[dict[str, Any]] = []
    stat_only: list[dict[str, Any]] = []
    for change in changes:
        rel = str(change.get("relative_path") or "")
        before = change.get("before") or {}
        after = change.get("after") or {}
        same_content = (
            before
            and after
            and before.get("size") == after.get("size")
            and before.get("sha256")
            and before.get("sha256") == after.get("sha256")
        )
        if rel.startswith("vault/") or not same_content:
            material.append(change)
        else:
            stat_only.append(change)
    return material, stat_only


def _query(query: str, vault: Path) -> dict[str, Any]:
    try:
        os.environ["OBSIDIAN_VAULT_PATH"] = str(vault)
        os.environ.setdefault("VAULT_EMBEDDINGS_BACKEND", "huggingface")
        from tools.archivist import search as search_module

        search_module.VAULT_PATH = vault
        search_module.CHROMA_PATH = vector_runtime_path(vault)
        result = str(search_module.search_all_memories.func(query))
        return {"query": query, "status": "PASS", "result_sha256": hashlib.sha256(result.encode("utf-8")).hexdigest(), "result_chars": len(result)}
    except Exception as exc:  # pragma: no cover - exercised by live observation
        return {"query": query, "status": "FAIL", "error": str(exc)}


def observe(vault: Path, output: Path, *, duration_seconds: int, interval_seconds: int) -> dict[str, Any]:
    vault = vault.resolve()
    output = output.resolve()
    if duration_seconds <= 0 or interval_seconds <= 0 or interval_seconds > 600:
        raise ValueError("duration_seconds must be positive and interval_seconds must be between 1 and 600")
    roots = {
        "vault": vault,
        "catalog_runtime": catalog_runtime_root(vault),
        "vector_runtime": vector_runtime_path(vault),
    }
    started = datetime.now(timezone.utc)
    baseline = _inventory(roots)
    _hydrate(roots, baseline)
    baseline_content = _fingerprint(baseline)
    baseline_stat = _fingerprint(baseline, stat=True)
    events: list[dict[str, Any]] = []
    stat_churn: list[dict[str, Any]] = []
    query_samples: list[dict[str, Any]] = []
    samples = 0

    def write_progress(status: str, *, events: list[dict[str, Any]], stat_churn: list[dict[str, Any]], query_samples: list[dict[str, Any]]) -> None:
        progress = {
            "status": status,
            "started_at": started.isoformat(),
            "elapsed_seconds": (datetime.now(timezone.utc) - started).total_seconds(),
            "samples": samples,
            "query_sample_count": len(query_samples),
            "event_count": len(events),
            "stat_churn_count": len(stat_churn),
            "recent_events": events[-1:] if events else [],
            "query_failures": [
                item for sample in query_samples for item in sample["queries"] if item.get("status") != "PASS"
            ],
        }
        output.with_suffix(output.suffix + ".progress.json").write_text(
            json.dumps(progress, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    # Exercise the read path before the first timed sample.
    query_samples.append({"observed_at": datetime.now(timezone.utc).isoformat(), "queries": [_query(query, vault) for query in QUERIES]})
    write_progress("RUNNING", events=events, stat_churn=stat_churn, query_samples=query_samples)
    deadline = time.monotonic() + duration_seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(interval_seconds, remaining))
        current = _inventory(roots)
        changes = _changes(roots, baseline, current)
        material_changes, stat_only_changes = _classify_changes(changes)
        stat_churn.extend(stat_only_changes)
        query_result = {"observed_at": datetime.now(timezone.utc).isoformat(), "queries": [_query(query, vault) for query in QUERIES]}
        query_samples.append(query_result)
        if material_changes:
            events.append({"observed_at": query_result["observed_at"], "changes": material_changes})
        baseline = current
        samples += 1
        write_progress("RUNNING", events=events, stat_churn=stat_churn, query_samples=query_samples)

    ended = datetime.now(timezone.utc)
    _hydrate(roots, baseline)
    query_failures = [item for sample in query_samples for item in sample["queries"] if item.get("status") != "PASS"]
    result = {
        "status": "PASS" if not events and not query_failures else "FAIL",
        "reason": None if not events and not query_failures else "runtime mutation or query failure observed",
        "started_at": started.isoformat(),
        "ended_at": ended.isoformat(),
        "duration_seconds": (ended - started).total_seconds(),
        "requested_duration_seconds": duration_seconds,
        "interval_seconds": interval_seconds,
        "samples": samples,
        "query_sample_count": len(query_samples),
        "queries": list(QUERIES),
        "query_failures": query_failures,
        "roots": {key: str(value) for key, value in roots.items()},
        "baseline_fingerprint": baseline_content,
        "final_fingerprint": _fingerprint(baseline),
        "baseline_stat_fingerprint": baseline_stat,
        "final_stat_fingerprint": _fingerprint(baseline, stat=True),
        "events": events,
        "event_count": len(events),
        "stat_churn_count": len(stat_churn),
        "stat_churn": stat_churn[:100],
        "read_only": True,
        "query_samples": query_samples,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_progress(result["status"], events=events, stat_churn=stat_churn, query_samples=query_samples)
    print(json.dumps(result, ensure_ascii=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", type=Path, default=Path("memories"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-seconds", type=int, default=3600)
    parser.add_argument("--interval-seconds", type=int, default=300)
    args = parser.parse_args()
    result = observe(args.vault, args.output, duration_seconds=args.duration_seconds, interval_seconds=args.interval_seconds)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
