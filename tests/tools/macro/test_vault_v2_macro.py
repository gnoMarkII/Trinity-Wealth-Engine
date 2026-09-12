"""Tests for Vault V2 Macro Single-Source Strategy & Indicator Series (T07).

Verifies that:
1. Macro Strategy (.md & .json) is stored only in Macroeconomics/Strategies/YYYY/MM in V2.
2. Dual-path archiving to Daily_Snapshots or Strategies/ is eliminated in V2.
3. Indicator series are stored in Macroeconomics/Indicator_Series in V2.
4. StrategyVaultAdapter and IndicatorSeriesAdapter read both V2 and V1 structures.
"""
import json
from pathlib import Path

import pytest

from schemas.macro_schemas import (
    MacroStrategyDirection,
    AssetAllocationView,
    AssetStance,
    EconomicState,
)
from tools.archivist.writer import write_raw_markdown
from tools.macro.adapters.strategy_vault_adapter import IndicatorSeriesAdapter, StrategyVaultAdapter
from tools.macro.dashboard import load_indicator_series
from tools.macro.evaluation import evaluate_macro_matrix, load_latest_macro_observables
from tools.macro.report_formatter import write_strategy_json_sidecar


def _create_mock_direction(date_str: str = "2026-08-10") -> MacroStrategyDirection:
    return MacroStrategyDirection(
        evaluated_at=f"{date_str}T12:00:00Z",
        overall_regime=EconomicState.GOLDILOCKS,
        asset_allocation=[
            AssetAllocationView(
                asset_class="หุ้น",
                asset_bucket="equities",
                stance=AssetStance.OVERWEIGHT,
                rationale="เศรษฐกิจเติบโต",
            ),
            AssetAllocationView(
                asset_class="พันธบัตร",
                asset_bucket="fixed_income",
                stance=AssetStance.NEUTRAL,
                rationale="Yield curve ปกติ",
            ),
        ],
        focus_themes=["เทคโนโลยี", "Growth"],
        conviction_level="high",
        conviction_rationale="ตัวเลขสนับสนุนชัดเจน",
        quant_narrative_alignment="aligned",
        divergence_note="",
    )


def _setup_vault(tmp_path: Path, layout_version: int) -> Path:
    vault = tmp_path / f"vault_macro_v{layout_version}"
    sys_dir = vault / ".system"
    sys_dir.mkdir(parents=True, exist_ok=True)
    (sys_dir / "vault_config.json").write_text(
        json.dumps({"layout_version": layout_version}), encoding="utf-8"
    )
    (vault / "30_Knowledge_Base" / "Macroeconomics" / "Strategies").mkdir(parents=True, exist_ok=True)
    (vault / "30_Knowledge_Base" / "Macroeconomics" / "Daily_Snapshots").mkdir(parents=True, exist_ok=True)
    (vault / "30_Knowledge_Base" / "Strategies").mkdir(parents=True, exist_ok=True)
    return vault


def test_macro_strategy_v2_single_source(tmp_path: Path, monkeypatch) -> None:
    """In V2, macro strategy is placed in Macroeconomics/Strategies/YYYY/MM with NO dual-path duplicate."""
    vault = _setup_vault(tmp_path, layout_version=2)
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(vault))

    date_str = "2026-08-10"
    direction = _create_mock_direction(date_str)

    # 1. Write JSON sidecar
    json_path = write_strategy_json_sidecar(direction, date_str)
    expected_v2_dir = vault / "30_Knowledge_Base" / "Macroeconomics" / "Strategies" / "2026" / "08"
    assert json_path.parent.resolve() == expected_v2_dir.resolve()
    assert json_path.exists()

    # 2. Write Markdown report
    md_content = f"""---
title: Macro Strategy Direction {date_str}
entity_type: macro_strategy
date: {date_str}
published_at: {date_str}
tags: [macro, strategy]
---

# Macro Strategy Direction {date_str}

Dominant Regime: Goldilocks
"""
    write_raw_markdown.invoke({
        "content": md_content,
        "folder_path": "30_Knowledge_Base/Macroeconomics/Strategies",
        "filename": f"Macro_Strategy_Direction_{date_str}",
    })

    # Assert MD report is in the exact same V2 folder
    md_path = expected_v2_dir / f"Macro_Strategy_Direction_{date_str}.md"
    assert md_path.exists()

    # Zero Dual-Path: Ensure NO file was created in legacy 30_Knowledge_Base/Strategies or Daily_Snapshots
    legacy_strat_dir = vault / "30_Knowledge_Base" / "Strategies"
    assert len(list(legacy_strat_dir.glob("Macro_Strategy_Direction_*"))) == 0

    legacy_daily_dir = vault / "30_Knowledge_Base" / "Macroeconomics" / "Daily_Snapshots"
    assert len(list(legacy_daily_dir.glob("Macro_Strategy_Direction_*"))) == 0

    # 3. StrategyVaultAdapter reads latest from V2
    adapter = StrategyVaultAdapter(vault)
    latest = adapter.latest()
    assert latest["evaluated_at"].startswith(date_str)
    assert latest["overall_regime"] == "Goldilocks"


def test_indicator_series_v2_placement_and_reading(tmp_path: Path, monkeypatch) -> None:
    """In V2, indicator series are stored in Macroeconomics/Indicator_Series."""
    vault = _setup_vault(tmp_path, layout_version=2)
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(vault))

    date_str = "2026-08-10"
    mock_indicator = {
        "series_key": "cpi_yoy",
        "value": 2.9,
        "observed_at": date_str,
        "label": "US CPI YoY",
        "unit": "%",
    }
    from tools.macro.dashboard import persist_indicator_series
    persist_indicator_series(vault, [mock_indicator])

    # Indicator_Series directory in Macroeconomics
    v2_indicator_dir = vault / "30_Knowledge_Base" / "Macroeconomics" / "Indicator_Series"
    assert v2_indicator_dir.exists()
    assert (v2_indicator_dir / "cpi_yoy.json").exists()

    adapter = StrategyVaultAdapter(vault)
    indicator_adapter = IndicatorSeriesAdapter(adapter)
    series = indicator_adapter.load("cpi_yoy", "1m")
    assert len(series) == 1
    assert series[0]["value"] == 2.9


def test_macro_v1_legacy_compatibility(tmp_path: Path, monkeypatch) -> None:
    """In V1, strategy sidecar is written to 30_Knowledge_Base/Strategies and dual-path works."""
    vault = _setup_vault(tmp_path, layout_version=1)
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(vault))

    date_str = "2026-07-06"
    direction = _create_mock_direction(date_str)

    json_path = write_strategy_json_sidecar(direction, date_str)
    expected_v1_dir = vault / "30_Knowledge_Base" / "Strategies"
    assert json_path.parent.resolve() == expected_v1_dir.resolve()
    assert json_path.exists()

    # StrategyVaultAdapter still finds it in V1
    adapter = StrategyVaultAdapter(vault)
    latest = adapter.latest()
    assert latest["evaluated_at"].startswith(date_str)
