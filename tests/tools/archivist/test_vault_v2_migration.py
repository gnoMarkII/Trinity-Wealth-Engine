"""Unit tests for Vault V2 Migration Engine (Plan, Apply, Verify, Rollback)."""
import json
from pathlib import Path
import pytest

from tools.archivist.vault_migration import (
    create_migration_plan,
    apply_migration_plan,
    verify_migration,
    rollback_migration,
    _compute_sha256,
)


@pytest.fixture
def legacy_vault(tmp_path):
    """Creates a synthetic legacy vault with notes in V1 locations."""
    vault = tmp_path / "test_legacy_vault"
    vault.mkdir()

    # 1. Equity note in legacy Equities/ folder
    eq_dir = vault / "30_Knowledge_Base" / "Equities"
    eq_dir.mkdir(parents=True)
    eq_note = eq_dir / "FTNT.md"
    eq_note.write_text("""---
schema_version: 1
entity_type: equity_analysis
title: FTNT Equity Analysis
ticker: FTNT
date: 2026-09-01
---
# FTNT Analysis V1
""", encoding="utf-8")

    # Sidecar JSON
    eq_sidecar = eq_dir / "FTNT.json"
    eq_sidecar.write_text(json.dumps({"ticker": "FTNT", "score": 90}), encoding="utf-8")

    # 2. News note in legacy News/ folder
    news_dir = vault / "30_Knowledge_Base" / "News"
    news_dir.mkdir(parents=True)
    news_note = news_dir / "2026-09-02_Tech_Rally.md"
    news_note.write_text("""---
schema_version: 1
entity_type: company_news
title: Tech Rally News
date: 2026-09-02
---
# Tech Rally
""", encoding="utf-8")

    # 3. Macro note in legacy Strategies/ folder
    macro_dir = vault / "30_Knowledge_Base" / "Strategies"
    macro_dir.mkdir(parents=True)
    macro_note = macro_dir / "2026-09-03_Macro_Outlook.md"
    macro_note.write_text("""---
schema_version: 1
entity_type: macro_strategy
title: Macro Outlook
date: 2026-09-03
---
# Macro Outlook
""", encoding="utf-8")

    return vault


def test_migration_lifecycle(legacy_vault, tmp_path):
    # Capture baseline hashes
    files_before = {p.relative_to(legacy_vault).as_posix(): _compute_sha256(p) for p in legacy_vault.rglob("*") if p.is_file()}

    # 1. CREATE PLAN
    plan_file = tmp_path / "test_plan.json"
    plan = create_migration_plan(legacy_vault, output_file=plan_file)

    assert plan_file.exists()
    assert plan.total_files == 3  # FTNT.md, 2026-09-02_Tech_Rally.md, 2026-09-03_Macro_Outlook.md
    assert plan.summary["relocate"] == 3

    # Check that FTNT companion sidecar was included
    ftnt_act = next(a for a in plan.actions if "FTNT.md" in a.source_rel)
    assert len(ftnt_act.companion_sidecars) == 1
    assert "FTNT.json" in ftnt_act.companion_sidecars[0]["source_rel"]

    # 2. APPLY PLAN
    apply_res = apply_migration_plan(plan_file, vault_root=legacy_vault)
    assert apply_res["applied_count"] == 4  # 3 markdown notes + 1 sidecar

    # Verify config updated to V2
    config_p = legacy_vault / ".system" / "vault_config.json"
    assert config_p.exists()
    cfg = json.loads(config_p.read_text(encoding="utf-8"))
    assert cfg["layout_version"] == 2

    # Verify new paths exist
    v2_ftnt = legacy_vault / "30_Knowledge_Base" / "Stocks" / "FTNT" / "Analysis" / "FTNT.md"
    v2_sidecar = legacy_vault / "30_Knowledge_Base" / "Stocks" / "FTNT" / "Analysis" / "FTNT.json"
    v2_news = legacy_vault / "30_Knowledge_Base" / "News" / "2026" / "09" / "2026-09-02_Tech_Rally.md"
    v2_macro = legacy_vault / "30_Knowledge_Base" / "Macroeconomics" / "Strategies" / "2026" / "09" / "2026-09-03_Macro_Outlook.md"

    assert v2_ftnt.exists()
    assert v2_sidecar.exists()
    assert v2_news.exists()
    assert v2_macro.exists()

    # 3. VERIFY PLAN
    verify_res = verify_migration(plan_file, vault_root=legacy_vault)
    assert verify_res["success"] is True
    assert verify_res["verified_count"] == 3
    assert len(verify_res["missing_targets"]) == 0
    assert len(verify_res["hash_mismatches"]) == 0

    # 4. ROLLBACK PLAN
    journal_p = Path(apply_res["journal_file"])
    assert journal_p.exists()

    rollback_res = rollback_migration(journal_p, vault_root=legacy_vault)
    assert rollback_res["rolled_back_count"] == 4

    # Verify original paths restored
    assert (legacy_vault / "30_Knowledge_Base" / "Equities" / "FTNT.md").exists()
    assert (legacy_vault / "30_Knowledge_Base" / "Equities" / "FTNT.json").exists()
    assert (legacy_vault / "30_Knowledge_Base" / "News" / "2026-09-02_Tech_Rally.md").exists()
    assert (legacy_vault / "30_Knowledge_Base" / "Strategies" / "2026-09-03_Macro_Outlook.md").exists()

    # Verify config reverted
    cfg_reverted = json.loads(config_p.read_text(encoding="utf-8"))
    assert cfg_reverted["layout_version"] == 1

    # Verify 100% hash preservation
    files_after = {p.relative_to(legacy_vault).as_posix(): _compute_sha256(p) for p in legacy_vault.rglob("*") if p.is_file() and not p.name.startswith("vault_config") and not p.name.startswith("vault_catalog")}
    for rel_p, orig_hash in files_before.items():
        assert rel_p in files_after
        assert files_after[rel_p] == orig_hash


def test_safety_gate_protects_live_memories(tmp_path):
    live_vault = tmp_path / "memories"
    live_vault.mkdir()

    plan_file = tmp_path / "plan.json"
    plan = create_migration_plan(live_vault, output_file=plan_file)

    # Calling apply without allow_live must raise PermissionError
    with pytest.raises(PermissionError):
        apply_migration_plan(plan_file, vault_root=live_vault, allow_live=False)
