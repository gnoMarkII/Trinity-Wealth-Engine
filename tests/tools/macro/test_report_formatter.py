import json
import pytest
from datetime import datetime
from schemas.macro_schemas import MacroStrategyDirection, AssetAllocationView, AssetStance, EconomicState
from tools.macro.report_formatter import _translate_warning, format_macro_strategy_report, write_strategy_json_sidecar
import tools.macro.report_formatter as report_formatter_module
from schemas.warning_registry import (
    WarningMessage,
    GRACEFUL_DROP_PAIR_TRADES,
    PT_GRACEFUL_DOWNGRADE,
    SINGLE_SOURCE_PENALTY,
    STALE_DATA_DEGRADATION,
    PORTFOLIO_DEFENSIVE_LOW,
    ACTIVE_ALLOC_GUARDRAIL,
    DEFENSIVE_LOW_SUPPORTING_DATA,
    PT_RISK_BUDGET_MED_TO_SMALL,
    COVERAGE_BACKFILL_EXPANDED,
)

def test_format_macro_strategy_report():
    direction = MacroStrategyDirection(
        evaluated_at=datetime.now().isoformat(),
        overall_regime=EconomicState.GOLDILOCKS,
        asset_allocation=[
            AssetAllocationView(asset_class="หุ้น", asset_bucket="equities", stance=AssetStance.OVERWEIGHT, rationale="เศรษฐกิจเติบโต"),
            AssetAllocationView(asset_class="พันธบัตร", asset_bucket="fixed_income", stance=AssetStance.NEUTRAL, rationale="Yield curve ปกติ")
        ],
        focus_themes=["เทคโนโลยี", "Growth"],
        conviction_level="high",
        conviction_rationale="ตัวเลขสนับสนุนชัดเจน",
        quant_narrative_alignment="aligned",
        divergence_note=""
    )

    report = format_macro_strategy_report(direction)

    assert "entity_type: macro_strategy" in report
    assert "Goldilocks" in report
    assert "Neutral" in report
    assert "เทคโนโลยี" in report
    assert "ความมั่นใจรวมอยู่ในระดับต่ำ" in report

def test_format_macro_strategy_report_divergent():
    direction = MacroStrategyDirection(
        evaluated_at=datetime.now().isoformat(),
        overall_regime=EconomicState.RECESSION,
        asset_allocation=[],
        focus_themes=[],
        conviction_level="low",
        conviction_rationale="ตัวเลขแย่มาก",
        quant_narrative_alignment="divergent",
        divergence_note="ข่าวดีแต่ตัวเลขแย่"
    )

    report = format_macro_strategy_report(direction)

    assert "[!WARNING] Quant-Narrative Divergence" in report
    assert "ข่าวดีแต่ตัวเลขแย่" in report


def test_write_strategy_json_sidecar(tmp_path, monkeypatch):
    monkeypatch.setattr(report_formatter_module, "VAULT_PATH", tmp_path)

    direction = MacroStrategyDirection(
        evaluated_at=datetime.now().isoformat(),
        overall_regime=EconomicState.GOLDILOCKS,
        asset_allocation=[
            AssetAllocationView(asset_class="หุ้น", asset_bucket="equities", stance=AssetStance.OVERWEIGHT, rationale="เศรษฐกิจเติบโต"),
        ],
        focus_themes=["เทคโนโลยี"],
        conviction_level="high",
        conviction_rationale="ตัวเลขสนับสนุนชัดเจน",
        quant_narrative_alignment="aligned",
        divergence_note=""
    )

    json_path = write_strategy_json_sidecar(direction, "2026-07-06")

    assert json_path == tmp_path / "30_Knowledge_Base" / "Strategies" / "Macro_Strategy_Direction_2026-07-06.json"
    assert json_path.exists()
    # ต้องไม่เหลือ temp file ค้างจากการเขียนแบบ atomic
    assert list(json_path.parent.glob("*.tmp")) == []

    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved["overall_regime"] == "Goldilocks"
    assert saved["asset_allocation"][0]["asset_class"] == "หุ้น"


def test_translate_warning_dynamic_graceful_drop_to_thai():
    translated = _translate_warning(str(WarningMessage(GRACEFUL_DROP_PAIR_TRADES, {"count": "2"})))
    assert "ตัด Pair Trade ออกอย่างปลอดภัยจำนวน 2 รายการ" in translated
    assert "omitted" not in translated

    downgrade = _translate_warning(str(WarningMessage(PT_GRACEFUL_DOWNGRADE)))
    assert "ลดความมั่นใจของ Pair Trade" in downgrade
    assert "single-point" not in downgrade


def test_translate_warning_single_source_and_stale_to_thai():
    single_source = _translate_warning(str(WarningMessage(SINGLE_SOURCE_PENALTY)))
    assert "ลดความมั่นใจเป็น MEDIUM" in single_source
    assert "Downgraded" not in single_source

    stale = _translate_warning(str(WarningMessage(STALE_DATA_DEGRADATION)))
    assert "ลดระดับ conviction รวมจาก HIGH เป็น MEDIUM" in stale
    assert "downgraded" not in stale.lower()


def test_translate_warning_structured_ids_to_thai():
    # Test simple ID
    single_source = _translate_warning("[SINGLE_SOURCE_PENALTY]")
    assert "ลดความมั่นใจเป็น MEDIUM" in single_source

    # Test ID with JSON payload
    drop_pt = _translate_warning('[GRACEFUL_DROP_PAIR_TRADES] {"count": "3"}')
    assert "ตัด Pair Trade ออกอย่างปลอดภัยจำนวน 3 รายการ" in drop_pt


def test_formatter_sanitizes_stale_low_and_weak_why_not_high():
    direction = MacroStrategyDirection(
        evaluated_at="2026-07-03T19:40:01",
        overall_regime=EconomicState.REFLATION,
        asset_allocation=[
            AssetAllocationView(
                asset_class="Precious Metals (Gold)",
                asset_bucket="commodities",
                stance=AssetStance.OVERWEIGHT,
                rationale="Gold hedge",
                confidence="medium",
                supporting_data=["Gold Futures = 4,186.00 USD/oz"],
                source_refs=["Global_Macro_Snapshot_2026-07-03.md"],
                why_not_high="ไม่มีเหตุผลที่ความมั่นใจไม่ถึงระดับสูง",
                validation_warnings=[
                    str(WarningMessage(SINGLE_SOURCE_PENALTY))
                ],
            )
            ,
            AssetAllocationView(
                asset_class="US Equities",
                asset_bucket="equities",
                stance=AssetStance.OVERWEIGHT,
                rationale="Growth",
                confidence="medium",
                supporting_data=["S&P 500 = 7483.24"],
                source_refs=["Global_Macro_Snapshot_2026-07-03.md"],
            ),
            AssetAllocationView(
                asset_class="Fixed Income",
                asset_bucket="fixed_income",
                stance=AssetStance.NEUTRAL,
                rationale="Rates balanced",
                confidence="medium",
                supporting_data=["10-Year Treasury Yield = 4.4850%"],
                source_refs=["Global_Macro_Snapshot_2026-07-03.md"],
            ),
            AssetAllocationView(
                asset_class="Currencies",
                asset_bucket="fx",
                stance=AssetStance.UNDERWEIGHT,
                rationale="USD/THB pressure",
                confidence="medium",
                supporting_data=["USD/THB = 33.13"],
                source_refs=["Country_Macro_Snapshot_2026-07-03.md"],
            ),
            AssetAllocationView(
                asset_class="Cash",
                asset_bucket="cash",
                stance=AssetStance.NEUTRAL,
                rationale="Cash yield",
                confidence="medium",
                supporting_data=["13-Week T-Bill Yield = 3.6680%"],
                source_refs=["Global_Macro_Snapshot_2026-07-03.md"],
            ),
        ],
        focus_themes=["Gold"],
        conviction_level="low",
        conviction_rationale="Stale data only",
        quant_narrative_alignment="aligned",
        source_files=[
            "Global_Macro_Snapshot_2026-07-03.md",
            "Country_Macro_Snapshot_2026-07-03.md",
            "Regional_Macro_Snapshot_2026-07-03.md",
        ],
        stale_data_warnings=["Real GDP - ข้อมูลล่าช้าเกิน 180 วัน"],
        validation_warnings=[
            str(WarningMessage(STALE_DATA_DEGRADATION))
        ],
    )

    report = format_macro_strategy_report(direction)
    conviction_line = next(line for line in report.splitlines() if "Conviction):" in line)

    assert "LOW" not in conviction_line
    assert "MEDIUM" in conviction_line
    assert "Downgraded confidence" not in report
    assert "Portfolio conviction downgraded to LOW" not in report
    assert "ไม่มีเหตุผลที่ความมั่นใจไม่ถึงระดับสูง" not in report
    assert "อ้างอิงแหล่งข้อมูลโดยตรงเพียงแหล่งเดียว" in report
    assert "Macro_Baseline_" in report





def test_formatter_sanitizes_cash_fallback_and_english_labels():
    direction = MacroStrategyDirection(
        evaluated_at="2026-07-04T10:05:12",
        overall_regime=EconomicState.REFLATION,
        asset_allocation=[
            AssetAllocationView(
                asset_class="Cash / T-Bills",
                asset_bucket="cash",
                stance=AssetStance.NEUTRAL,
                rationale="Backfilled core asset class (Cash / T-Bills) due to input data constraints",
                confidence="medium",
                data_confidence="medium",
                signal_confidence="medium",
                implementation_confidence="medium",
                supporting_data=["Fed Funds Rate = 3.63%", "Core PCE = 3.41%"],
            )
        ],
        focus_themes=["Cash"],
        conviction_level="medium",
        conviction_rationale="Cash observables are sufficient",
        quant_narrative_alignment="aligned",
        validation_warnings=[
            str(WarningMessage(COVERAGE_BACKFILL_EXPANDED, {"old": "4", "new": "5"}))
        ],
    )

    report = format_macro_strategy_report(direction)

    assert "Cash / T-Bills" in report
    assert "3-6 Months" in report
    assert "Backfilled core asset class (Cash / T-Bills)" in report


def test_formatter_normalizes_delta_horizon_and_gold_real_yield_rationale():
    direction = MacroStrategyDirection(
        evaluated_at="2026-07-04T14:29:03",
        overall_regime=EconomicState.RECESSION,
        asset_allocation=[
            AssetAllocationView(
                asset_class="Precious Metals (Gold)",
                asset_bucket="commodities",
                stance=AssetStance.OVERWEIGHT,
                allocation_delta="Overweight",
                time_horizon="6-12 Months",
                rationale="ทองคำและสินทรัพย์ปลอดภัย",
                confidence="medium",
                supporting_data=["Gold Futures = 4,187.30 USD/oz", "10-Year Real Yield (TIPS) = 2.25%"],
                source_refs=["Global_Macro_Snapshot_2026-07-04.md", "Country_Macro_Snapshot_2026-07-04.md"],
            )
        ],
        focus_themes=["Gold"],
        conviction_level="medium",
        conviction_rationale="Gold risk hedge",
        quant_narrative_alignment="aligned",
    )

    report = format_macro_strategy_report(direction)

    assert "+3% ถึง +5% vs benchmark" in report
    assert "6-12 Months" in report

