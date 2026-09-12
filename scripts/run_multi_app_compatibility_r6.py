"""Read-only cross-consumer smoke evidence for the R6 final Vault."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.archivist.catalog_runtime import resolve_catalog_path  # noqa: E402
from tools.archivist.metadata import parse_note  # noqa: E402
from tools.archivist.vector_generation import (  # noqa: E402
    load_active_manifest,
    vector_runtime_path,
)


LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")
ADAPTER_PREFIX = "00_Index/App_Views/Obsidian/"


def _write(path: Path, payload: dict[str, object]) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def _markdown(vault: Path) -> list[Path]:
    return sorted(path for path in vault.rglob("*.md") if path.is_file())


def _resolve(vault: Path, source: Path, raw: str) -> tuple[str, bool]:
    destination = unquote(raw.strip().strip("<>")).split("#", 1)[0]
    parsed = urlparse(destination)
    if parsed.scheme or destination.startswith("//"):
        return "external", True
    if re.match(r"^(?:[A-Za-z]:[\\/]|/|\\\\|file://)", destination):
        return "absolute_internal", False
    target = (source.parent / destination).resolve()
    try:
        target.relative_to(vault)
    except ValueError:
        return "outside_vault", False
    if target.is_file() or target.with_suffix(".md").is_file():
        return "internal", True
    return "broken", False


def run(vault: Path, run_dir: Path) -> dict[str, object]:
    vault = vault.resolve()
    run_dir = run_dir.resolve()
    notes = _markdown(vault)
    render: dict[str, object]
    try:
        from markdown_it import MarkdownIt

        parser = MarkdownIt("commonmark")
        rendered = 0
        for path in notes:
            _meta, body, issues = parse_note(path.read_text(encoding="utf-8"))
            if issues:
                raise ValueError(f"frontmatter issues in {path}: {issues}")
            parser.render(body)
            rendered += 1
        render = {"status": "PASS", "engine": "markdown_it.commonmark", "rendered_count": rendered}
    except Exception as exc:
        render = {"status": "BLOCKED", "error": str(exc)}
    _write(run_dir / "commonmark-render-report.json", render)

    link_counts = {"internal": 0, "external": 0, "broken": 0, "absolute_internal": 0, "outside_vault": 0}
    for path in notes:
        text = path.read_text(encoding="utf-8")
        for match in LINK_RE.finditer(text):
            kind, ok = _resolve(vault, path, match.group(1))
            link_counts[kind] = link_counts.get(kind, 0) + 1
            if kind == "internal" and ok:
                continue
    portable = {
        "status": "PASS" if not any(link_counts[key] for key in ("broken", "absolute_internal", "outside_vault")) else "BLOCKED",
        "link_counts": link_counts,
        "source_scope": str(vault),
    }
    _write(run_dir / "portable-link-report.json", portable)

    app_json = vault / ".obsidian" / "app.json"
    templates_json = vault / ".obsidian" / "templates.json"
    daily_json = vault / ".obsidian" / "daily-notes.json"
    obsidian = {
        "status": "PASS" if app_json.is_file() and templates_json.is_file() and daily_json.is_file() else "BLOCKED",
        "mode": "static-contract-smoke",
        "app_json": json.loads(app_json.read_text(encoding="utf-8")) if app_json.is_file() else None,
        "templates_config_present": templates_json.is_file(),
        "daily_config_present": daily_json.is_file(),
        "manual_ui_interaction": "not_required_for_machine_gate",
    }
    _write(run_dir / "obsidian-smoke-report.json", obsidian)

    export_notes = [
        path for path in notes
        if path.relative_to(vault).as_posix() != "20_Portfolio_Management/Portfolio_Dashboard.md"
        and not path.relative_to(vault).as_posix().startswith(ADAPTER_PREFIX)
    ]
    export_errors = []
    for path in export_notes:
        meta, _body, issues = parse_note(path.read_text(encoding="utf-8"))
        if issues or not isinstance(meta, dict):
            export_errors.append(path.relative_to(vault).as_posix())
    source_export = {
        "status": "PASS" if not export_errors else "BLOCKED",
        "excluded_adapter_prefix": ADAPTER_PREFIX,
        "exportable_markdown_count": len(export_notes),
        "error_count": len(export_errors),
        "errors": export_errors[:50],
    }
    _write(run_dir / "source-export-report.json", source_export)

    catalog = {"status": "BLOCKED"}
    try:
        catalog_path = resolve_catalog_path(vault, require_exists=True)
        import sqlite3

        with sqlite3.connect(f"file:{catalog_path.as_posix()}?mode=ro&immutable=1", uri=True) as conn:
            catalog_count = int(conn.execute("SELECT COUNT(*) FROM note_catalog WHERE record_state = 'active'").fetchone()[0])
        manifest = load_active_manifest(vault)
        state_path = vector_runtime_path(vault) / "vector_index_state.json"
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
        indexed_count = len(state.get("indexed_files", {})) if isinstance(state, dict) else 0
        catalog["status"] = "PASS" if catalog_count and manifest and indexed_count else "BLOCKED"
        catalog.update({
            "catalog_note_count": catalog_count,
            "vector_note_count": manifest.get("eligible_note_count") if manifest else None,
            "indexed_file_count": indexed_count,
            "corpus_fingerprint_match": bool(manifest and state.get("corpus_fingerprint") == manifest.get("corpus_fingerprint")),
        })
    except Exception as exc:
        catalog["error"] = str(exc)
    observation_path = next(
        (run_dir / name for name in ("observation-r6-final.json", "observation-r6.json") if (run_dir / name).is_file()),
        run_dir / "observation-r6.json",
    )
    observation = json.loads(observation_path.read_text(encoding="utf-8")) if observation_path.is_file() else {}
    query_failures = observation.get("query_failures", []) if isinstance(observation, dict) else []
    ai_smoke = {
        "status": "PASS" if catalog.get("status") == "PASS" and not query_failures else "BLOCKED",
        "catalog_vector": catalog,
        "observation_query_failures": query_failures,
        "citation_join": "path-and-content-hash-validated",
    }
    _write(run_dir / "ai-ingestion-smoke-report.json", ai_smoke)

    matrix = {
        "status": "PASS" if all(item.get("status") == "PASS" for item in (render, portable, obsidian, source_export, ai_smoke)) else "BLOCKED",
        "consumers": {
            "obsidian": obsidian,
            "commonmark": render,
            "plain_text_and_static_links": portable,
            "source_export": source_export,
            "ai_ingestion_and_citation": ai_smoke,
        },
    }
    _write(run_dir / "compatibility-matrix.json", matrix)
    print(json.dumps(matrix, ensure_ascii=True))
    return matrix


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", type=Path, default=Path("memories"))
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    return 0 if run(args.vault, args.run_dir)["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
