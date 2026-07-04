import pytest
from datetime import datetime
from schemas.macro_schemas import MacroStrategyDirection, AssetAllocationView, AssetStance, EconomicState
from tools.macro.report_formatter import _translate_warning, format_macro_strategy_report

def test_format_macro_strategy_report():
    direction = MacroStrategyDirection(
        evaluated_at=datetime.now().isoformat(),
        overall_regime=EconomicState.GOLDILOCKS,
        asset_allocation=[
            AssetAllocationView(asset_class="หุ้น", stance=AssetStance.OVERWEIGHT, rationale="เศรษฐกิจเติบโต"),
            AssetAllocationView(asset_class="พันธบัตร", stance=AssetStance.NEUTRAL, rationale="Yield curve ปกติ")
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


def test_translate_warning_dynamic_graceful_drop_to_thai():
    translated = _translate_warning(
        "Graceful Drop: 2 pair trade(s) omitted due to insufficient numeric market data or missing executable controls."
    )
    assert "ตัด Pair Trade ออกอย่างปลอดภัยจำนวน 2 รายการ" in translated
    assert "omitted" not in translated

    downgrade = _translate_warning(
        "Pair Trade Graceful Downgrade: Used single-point relative spread/ratio evidence instead of historical beta/correlation time series."
    )
    assert "ลดความมั่นใจของ Pair Trade" in downgrade
    assert "single-point" not in downgrade


def test_translate_warning_single_source_and_stale_to_thai():
    single_source = _translate_warning(
        "Single-Source Penalty: Downgraded confidence to MEDIUM because view relies on only 1 source reference."
    )
    assert "ลดความมั่นใจเป็น MEDIUM" in single_source
    assert "Downgraded" not in single_source

    stale = _translate_warning(
        "Stale Data Degradation: Portfolio conviction downgraded to LOW due to presence of stale data warnings."
    )
    assert "ลดระดับ conviction รวมลงเป็น MEDIUM" in stale
    assert "downgraded" not in stale.lower()


def test_formatter_sanitizes_stale_low_and_weak_why_not_high():
    direction = MacroStrategyDirection(
        evaluated_at="2026-07-03T19:40:01",
        overall_regime=EconomicState.REFLATION,
        asset_allocation=[
            AssetAllocationView(
                asset_class="Precious Metals (Gold)",
                stance=AssetStance.OVERWEIGHT,
                rationale="Gold hedge",
                confidence="medium",
                supporting_data=["Gold Futures = 4,186.00 USD/oz"],
                source_refs=["Global_Macro_Snapshot_2026-07-03.md"],
                why_not_high="ไม่มีเหตุผลที่ความมั่นใจไม่ถึงระดับสูง",
                validation_warnings=[
                    "Single-Source Penalty: Downgraded confidence to MEDIUM because view relies on only 1 source reference."
                ],
            )
            ,
            AssetAllocationView(
                asset_class="US Equities",
                stance=AssetStance.OVERWEIGHT,
                rationale="Growth",
                confidence="medium",
                supporting_data=["S&P 500 = 7483.24"],
                source_refs=["Global_Macro_Snapshot_2026-07-03.md"],
            ),
            AssetAllocationView(
                asset_class="Fixed Income",
                stance=AssetStance.NEUTRAL,
                rationale="Rates balanced",
                confidence="medium",
                supporting_data=["10-Year Treasury Yield = 4.4850%"],
                source_refs=["Global_Macro_Snapshot_2026-07-03.md"],
            ),
            AssetAllocationView(
                asset_class="Currencies",
                stance=AssetStance.UNDERWEIGHT,
                rationale="USD/THB pressure",
                confidence="medium",
                supporting_data=["USD/THB = 33.13"],
                source_refs=["Country_Macro_Snapshot_2026-07-03.md"],
            ),
            AssetAllocationView(
                asset_class="Cash",
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
            "Stale Data Degradation: Portfolio conviction downgraded to LOW due to presence of stale data warnings."
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


def test_catch_all_sanitization_negative_assertions():
    from tools.macro.report_formatter import _sanitize_english_prefixes_catch_all
    sample_text = (
        "> - ⚠️ **คำเตือน (Warning):** Defensive Degradation: Portfolio conviction downgraded to LOW because >=50% of asset views lack numeric hard data.\n"
        "> - ⚠️ **คำเตือน (Warning):** Single-Source Penalty: Downgraded confidence to MEDIUM.\n"
        "> - ⚠️ **คำเตือน (Warning):** Stale Data Degradation: Downgraded overall conviction level from HIGH to MEDIUM.\n"
        "> - ⚠️ **คำเตือน (Warning):** Active Allocation Guardrail: Changed stance from Overweight to Neutral.\n"
        "> - ⚠️ **คำเตือน (Warning):** Unknown Future Guardrail: Some fallback message.\n"
    )
    sanitized = _sanitize_english_prefixes_catch_all(sample_text)
    assert "Defensive Degradation:" not in sanitized
    assert "Single-Source Penalty:" not in sanitized
    assert "Stale Data Degradation:" not in sanitized
    assert "Active Allocation Guardrail:" not in sanitized
    assert "Unknown Future Guardrail:" not in sanitized
    assert "ลดระดับความมั่นใจ (Defensive):" in sanitized
    assert "ลดความมั่นใจ (Single-Source):" in sanitized
    assert "ลดระดับความมั่นใจ (Stale Data):" in sanitized
    assert "กรอบควบคุมมุมมองเชิงรุก:" in sanitized
    assert "คำเตือนระบบ (Unknown Future Guardrail):" in sanitized
