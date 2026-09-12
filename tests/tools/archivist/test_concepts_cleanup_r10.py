from __future__ import annotations

from pathlib import Path

from tools.archivist.concepts_cleanup import build_cleanup_plan, scan_concepts


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_concepts_planner_is_deterministic_and_routes_safe_candidates(tmp_path: Path) -> None:
    vault = tmp_path / "memories"
    _write(
        vault / "30_Knowledge_Base" / "Concepts" / "Unused.md",
        "---\ntitle: Unused\nentity_type: concept\nsearch_scope: excluded\nlifecycle_status: stub\n---\n\n"
        "<!-- Concept Stub created automatically -->\n",
    )
    _write(
        vault / "30_Knowledge_Base" / "Concepts" / "2026-07-20 Briefing.md",
        "---\ntitle: Briefing\nentity_type: concept\ndate: 2026-07-20\n---\n\nReal briefing content.\n",
    )
    _write(
        vault / "30_Knowledge_Base" / "News" / "Source.md",
        "---\ntitle: Source\nentity_type: company_news\nsearch_scope: included\n---\n\n"
        "[unused](../Concepts/Unused.md)\n",
    )

    snapshot = scan_concepts(vault)
    plan = build_cleanup_plan(snapshot)
    plan_again = build_cleanup_plan(snapshot)
    by_path = {row["path"]: row for row in plan["concepts"]}

    assert plan["snapshot_fingerprint"] == plan_again["snapshot_fingerprint"]
    assert plan["policy_digest"] == plan_again["policy_digest"]
    assert by_path["30_Knowledge_Base/Concepts/Unused.md"]["disposition"] == "RETIRE"
    assert by_path["30_Knowledge_Base/Concepts/2026-07-20 Briefing.md"]["disposition"] == "RELOCATE"
    assert all(row["disposition"] == "REVIEW" or row["apply_eligible"] for row in plan["concepts"])
