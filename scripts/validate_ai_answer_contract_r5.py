"""Smoke-test the R5 citation/trust gate against current vault evidence."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.archivist.ai_answer_contract import collect_evidence, validate_answer  # noqa: E402
from tools.archivist.catalog_adapter import SqliteNoteCatalogAdapter  # noqa: E402
from tools.archivist.catalog_runtime import resolve_catalog_path  # noqa: E402


def run(vault: Path, output: Path) -> dict:
    vault = vault.resolve()
    catalog_path = resolve_catalog_path(vault, require_exists=True)
    catalog = SqliteNoteCatalogAdapter(vault_root=vault, read_only=True)
    entry = next(catalog.iter_notes(page_size=1))
    document = Document(page_content="grounded smoke test", metadata={"relative_path": entry.relative_path})
    evidence = collect_evidence(vault, catalog, [document])
    research = validate_answer(
        "This is a research-mode answer with an explicit citation.",
        evidence,
        cited_paths=[entry.relative_path],
        production_mode=False,
    )
    production = validate_answer(
        "This must not be presented as a production decision.",
        evidence,
        cited_paths=[entry.relative_path],
        production_mode=True,
    )
    result = {
        "status": "PASS" if research["status"] == "PASS" and production["status"] == "BLOCKED" else "FAIL",
        "phase": "F08",
        "catalog_path": str(catalog_path),
        "research_mode": research,
        "production_mode": production,
    }
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", type=Path, default=Path("memories"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return 0 if run(args.vault, args.output)["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
