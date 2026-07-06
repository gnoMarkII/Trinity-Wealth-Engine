import re
from datetime import datetime

from schemas.macro_schemas import AssetStance, MacroStrategyDirection
from schemas.report_labels import (
    ALLOCATION_DELTA_DEFAULTS,
    CONVICTION_LOW_DISPLAY,
    WHY_NOT_HIGH_MESSAGES,
)
from schemas.warning_registry import (
    PORTFOLIO_DEFENSIVE_LOW,
    SINGLE_SOURCE_PENALTY,
    SOURCE_REF_PENALTY,
    STALE_DATA_DEGRADATION,
    translate_warning,
)
from core.text_utils import repair_mojibake, _MOJIBAKE_MARKERS, _repair_mojibake_chunk


def _join_or_dash(items: list[str]) -> str:
    return ", ".join(items) if items else "-"


_translate_warning = translate_warning


def _translate_warnings(messages: list[str]) -> list[str]:
    return [translate_warning(repair_mojibake(message)) for message in messages]


def _translate_report_text(value: str) -> str:
    return repair_mojibake(str(value)) if value is not None else ""





def _display_conviction_level(direction: MacroStrategyDirection) -> str:
    level = str(getattr(direction, "conviction_level", "medium")).lower()
    warnings = [str(w) for w in getattr(direction, "validation_warnings", [])]
    has_defensive_low = any(f"[{PORTFOLIO_DEFENSIVE_LOW}]" in w for w in warnings)
    has_stale_low = any(f"[{STALE_DATA_DEGRADATION}]" in w for w in warnings)
    if level == "low" and getattr(direction, "stale_data_warnings", []) and has_stale_low and not has_defensive_low:
        return "medium"
    return level


def _display_why_not_high(asset) -> str:
    text = _translate_report_text(getattr(asset, "why_not_high", "") or "")
    lowered = text.lower().strip()
    weak = (
        not lowered
        or lowered in {"-", "none", "n/a"}
        or "ไม่มีเหตุผล" in lowered
        or "no reason" in lowered
        or lowered == WHY_NOT_HIGH_MESSAGES["default"].lower()
        or lowered == WHY_NOT_HIGH_MESSAGES["low_confidence"].lower()
    )
    if getattr(asset, "confidence", "medium") == "high" or not weak:
        return text or "-"
    warnings = [str(w) for w in getattr(asset, "validation_warnings", [])]
    if any(f"[{SINGLE_SOURCE_PENALTY}]" in w for w in warnings):
        return WHY_NOT_HIGH_MESSAGES["single_source"]
    if any(f"[{SOURCE_REF_PENALTY}]" in w or "SOURCE_REF_PENALTY" in w or "source_refs ถูกอนุมาน" in w for w in warnings):
        return WHY_NOT_HIGH_MESSAGES["source_ref_inferred"]
    if "gold" in str(getattr(asset, "asset_class", "")).lower():
        return WHY_NOT_HIGH_MESSAGES["gold_real_yield"]
    if getattr(asset, "confidence", "medium") == "low":
        return WHY_NOT_HIGH_MESSAGES["low_confidence"]
    return WHY_NOT_HIGH_MESSAGES["default"]


def _display_source_files(direction: MacroStrategyDirection, today: str) -> list[str]:
    source_files = list(getattr(direction, "source_files", []) or [])
    baseline = f"Macro_Baseline_{today}.md"
    if baseline not in source_files:
        source_files.append(baseline)
    return source_files


def _display_allocation_delta(asset) -> str:
    raw = str(getattr(asset, "allocation_delta", "") or "").strip()
    stance = getattr(getattr(asset, "stance", ""), "value", str(getattr(asset, "stance", ""))).lower()
    if not raw:
        return "-"
    if raw.lower() in {"overweight", "underweight", "neutral"} or raw.lower() == stance:
        return ALLOCATION_DELTA_DEFAULTS.get(stance, "0% vs benchmark")
    return raw


def _display_time_horizon(value: str) -> str:
    return str(value or "3-6 Months")


def format_macro_strategy_report(direction: MacroStrategyDirection) -> str:
    """Build a clean markdown report without source-level mojibake literals."""
    today = datetime.now().strftime("%Y-%m-%d")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    display_conviction = _display_conviction_level(direction)

    lines = [
        "---",
        f"title: Macro Strategy Direction {today}",
        "entity_type: macro_strategy",
        f"date: {today}",
        f"last_updated: {now}",
        f"generated_by: {getattr(direction, 'generated_by', 'strategic_allocator')}",
        "tags: [macro, strategy, allocation, institutional]",
        "---\n",
        f"# 🧭 1. Executive View — ทิศทางกลยุทธ์การลงทุนเชิงมหภาค ({today})\n",
        f"> **สภาวะเศรษฐกิจ (Overall Regime):** {direction.overall_regime.value}",
        f"> **กรอบระยะเวลาลงทุน (Time Horizon):** {getattr(direction, 'time_horizon', '3-6 Months')}",
        f"> **ระดับความมั่นใจ (Conviction):** {CONVICTION_LOW_DISPLAY if display_conviction == 'low' else display_conviction.upper()}",
        f"> **ความสอดคล้อง Quant-Narrative:** {direction.quant_narrative_alignment}",
        f"> **ประเมินเมื่อ (Evaluated At):** {direction.evaluated_at}\n",
    ]

    reg_probs = getattr(direction, "regime_probabilities", {})
    if reg_probs:
        lines.extend(["### 🎲 ความน่าจะเป็นของสภาวะเศรษฐกิจ (Regime Probabilities)\n", "| สภาวะเศรษฐกิจ (Regime) | ความน่าจะเป็น (Probability) |", "|------------------------|---------------------------|"])
        for reg, prob in reg_probs.items():
            lines.append(f"| **{reg}** | {prob} |")
        lines.append("")

    assumptions = getattr(direction, "key_assumptions", [])
    if assumptions:
        lines.append("### 📌 สมมติฐานหลัก (Key Assumptions)\n")
        for asm in assumptions:
            lines.append(f"- {asm}")
        lines.append("")

    reg_evidences = getattr(direction, "regime_evidence", [])
    if reg_evidences:
        lines.extend([
            "## 📊 2. Evidence Dashboard (ตารางหลักฐานสภาวะเศรษฐกิจ 5 มิติ)\n",
            "| มิติ (Dimension) | ทิศทางสัญญาณ (Signal) | ตัวเลข Hard Data รองรับ | ข้อขัดแย้ง (Conflict) | ความมั่นใจ |",
            "|------------------|-----------------------|-------------------------|-----------------------|------------|",
        ])
        for ev in reg_evidences:
            conflict_str = ev.conflict if ev.conflict else "-"
            lines.append(f"| **{ev.dimension}** | {ev.signal} | {ev.evidence} | {conflict_str} | {ev.confidence.upper()} |")
        lines.append("")

    lines.extend([
        "## 📈 3. Cross-Asset Allocation Summary (ตารางสรุปมุมมองจัดสรรสินทรัพย์)\n",
        "| สินทรัพย์ (Asset Class) | มุมมอง (Stance) | Delta vs Benchmark | Benchmark Ref | Time Horizon | ความมั่นใจรวม | Why Not HIGH | ข้อมูลตลาดรองรับ (Key Observables) |",
        "|------------------------|-----------------|--------------------|---------------|--------------|----------------|--------------|-----------------------------------|",
    ])
    for a in direction.asset_allocation:
        stance_fmt = f"**{a.stance.value}**" if a.stance != AssetStance.NEUTRAL else a.stance.value
        data_str = ", ".join(_translate_report_text(item) for item in getattr(a, "supporting_data", [])) if getattr(a, "supporting_data", []) else "ไม่มี hard data"
        lines.append(
            f"| **{a.asset_class}** | {stance_fmt} | {_display_allocation_delta(a)} | "
            f"{getattr(a, 'benchmark_ref', '') or '-'} | {_display_time_horizon(getattr(a, 'time_horizon', 'Macro (3-6 Months)'))} | "
            f"{getattr(a, 'confidence', 'medium').upper()} | {_display_why_not_high(a)} | {data_str} |"
        )

    lines.append("\n### 🔎 เจาะลึกเหตุผลการจัดสรรรายสินทรัพย์ (Detailed Rationale)\n")
    for a in direction.asset_allocation:
        warnings = _translate_warnings(getattr(a, "validation_warnings", []))
        data_str = ", ".join(_translate_report_text(item) for item in getattr(a, "supporting_data", [])) if getattr(a, "supporting_data", []) else "ไม่มี hard data"
        lines.append(
            f"> [!note]- **{a.asset_class}** — {a.stance.value.upper()} "
            f"(Overall Conf: {getattr(a, 'confidence', 'medium').upper()} | "
            f"Data: {getattr(a, 'data_confidence', 'medium').upper()} | "
            f"Signal: {getattr(a, 'signal_confidence', 'medium').upper()} | "
            f"Implementation: {getattr(a, 'implementation_confidence', 'medium').upper()})"
        )
        lines.append(f"> - **เหตุผลรองรับ (Rationale):** {_translate_report_text(a.rationale)}")
        why_not_high = _display_why_not_high(a)
        if why_not_high and why_not_high != "-":
            lines.append(f"> - **เหตุผลที่ไม่เป็น HIGH (Why Not HIGH):** {why_not_high}")
        lines.append(f"> - **ข้อมูลตลาดรองรับ (Market Observables):** {data_str}")
        if warnings:
            lines.append(f"> - ⚠️ **คำเตือน (Warning):** {' '.join(warnings)}")
        stale_warning = getattr(a, "stale_data_warning", "")
        if stale_warning:
            lines.append(f"> - ⚠️ **คำเตือนข้อมูลล่าช้า (Stale Data Warning):** {stale_warning}")
        lines.append(f"> - **แหล่งข้อมูลอ้างอิง (Source Refs):** {_join_or_dash(getattr(a, 'source_refs', []))}")
        invals = getattr(a, "invalidation_conditions", [])
        if invals:
            lines.append(f"> - **เงื่อนไขยกเลิกมุมมอง (Invalidation Conditions):** {', '.join(invals)}")
        lines.append("")

    has_contradictions = (
        direction.quant_narrative_alignment == "divergent"
        or direction.divergence_note
        or any("Contradiction" in w or "Divergent" in w for w in getattr(direction, "validation_warnings", []))
    )
    lines.append("## ⚡ 4. Key Contradictions & Quant-Narrative Divergence (จุดขัดแย้งเชิงตรรกะ)\n")
    if has_contradictions:
        if direction.quant_narrative_alignment == "divergent":
            lines.append(f"> [!WARNING] Quant-Narrative Divergence — ความไม่สอดคล้องระหว่าง Quant และ Narrative\n> {direction.divergence_note}\n")
        elif direction.divergence_note:
            lines.append(f"> [!NOTE] หมายเหตุความแตกต่าง (Divergence Note)\n> {direction.divergence_note}\n")
        for w in _translate_warnings(getattr(direction, "validation_warnings", [])):
            if "Contradiction" in w or "Divergent" in w:
                lines.append(f"> [!WARNING] ข้อความเตือนความขัดแย้งจากระบบ (Auto-Detected Guardrail Contradiction)\n> {w}\n")
    else:
        lines.append("> [!NOTE] ไม่พบจุดขัดแย้งสำคัญระหว่างข้อมูลเชิงปริมาณและสภาวะตลาด (Aligned)\n")

    pair_trades = getattr(direction, "pair_trades", [])
    lines.append("## ⚖️ 5. Relative Value & Pair Trades (Trade Ideas) — กลยุทธ์จับคู่เทรดเชิงมูลค่าสัมพัทธ์\n")
    if pair_trades:
        for pt in pair_trades:
            sizing = pt.sizing_guidance.upper().replace("_", " ")
            lines.append(f"> [!tip]- **Pair Trade:** Long {pt.long_leg} / Short {pt.short_leg} (Confidence: {pt.confidence.upper()} | Risk Budget: {sizing})")
            for label, value in [
                ("Instrument Proxy / เครื่องมือจริง", getattr(pt, "instrument_proxy", "")),
                ("Hedge Ratio / สัดส่วน", getattr(pt, "hedge_ratio", "")),
                ("FX Handling / การบริหารค่าเงิน", getattr(pt, "fx_handling", "")),
                ("Entry Trigger / จุดเข้าเทรด", getattr(pt, "entry_trigger", "")),
                ("Implementation Idea", getattr(pt, "implementation_idea", "")),
            ]:
                if value:
                    lines.append(f"> - **{label}:** {value}")
            lines.append(f"> - **แนวคิดหลัก (Thesis):** {pt.thesis}")
            lines.append(f"> - **ปัจจัยกระตุ้น (Catalyst):** {pt.catalyst}")
            lines.append(f"> - **ความเสี่ยง (Risk):** {pt.risk}")
            for label, value in [
                ("Stop Loss Trigger / จุดตัดขาดทุน", getattr(pt, "stop_loss_trigger", "")),
                ("Target Gain / เป้าหมายทำกำไร", getattr(pt, "target_gain_or_rebalance", "")),
                ("Max Drawdown Limit / ขีดจำกัดผลขาดทุน", getattr(pt, "max_drawdown_limit", "")),
                ("Review Frequency / ความถี่ทบทวน", getattr(pt, "review_frequency", "")),
            ]:
                if value:
                    lines.append(f"> - **{label}:** {value}")
            lines.append(f"> - **กรอบระยะเวลา (Time Horizon):** {pt.time_horizon}")
            lines.append(f"> - **ข้อมูลตัวเลขรองรับ (Supporting Data):** {', '.join(_translate_report_text(item) for item in pt.supporting_data) if pt.supporting_data else 'ไม่มี hard data'}")
            for w in _translate_warnings(pt.validation_warnings):
                lines.append(f"> - ⚠️ **คำเตือน (Warning):** {w}")
            lines.append("")
    else:
        lines.append("> [!NOTE] ไม่พบโอกาสจับคู่เทรดที่เข้าเกณฑ์ความมั่นใจเชิงสถิติในรอบการประเมินนี้\n")

    risk_scenarios = getattr(direction, "risk_scenarios", [])
    lines.append("## 🛡️ 6. Portfolio Risk Mitigation & Hedging (Hedging Plan) — แผนการบริหารความเสี่ยงและป้องกันพอร์ต\n")
    if risk_scenarios:
        for rs in risk_scenarios:
            purpose = getattr(rs, "hedge_purpose", "portfolio_hedge")
            lines.append(
                f"> [!warning]- **Tail Risk:** {rs.tail_risk} "
                f"(Probability: {rs.probability.upper()} | Impact: {rs.impact.upper()} | "
                f"Confidence: {rs.confidence.upper()} | Purpose: {purpose})"
            )
            for label, value in [
                ("Hedge Size / ขนาดป้องกันความเสี่ยง", getattr(rs, "hedge_size", "")),
                ("Warning Indicators / สัญญาณเตือนล่วงหน้า", _join_or_dash(rs.early_warning_indicators) if rs.early_warning_indicators else ""),
                ("Hedge Instruments / เครื่องมือป้องกันความเสี่ยง", _join_or_dash(rs.hedge_instruments) if rs.hedge_instruments else ""),
                ("Trigger Type / ประเภทจุดตัด", getattr(rs, "trigger_type", "")),
                ("Trigger to Activate / จุดตัดทำงาน", rs.trigger_to_activate),
                ("Volume Threshold / ปริมาณซื้อขายยืนยัน", getattr(rs, "volume_threshold", "")),
                ("Unwind / Cover Condition / เงื่อนไขยกเลิก", getattr(rs, "unwind_or_cover_condition", "")),
                ("Trade-off / Cost / ต้นทุนหรือส่วนเสีย", rs.cost_or_tradeoff),
            ]:
                if value:
                    lines.append(f"> - **{label}:** {value}")
            lines.append(f"> - **ข้อมูลตัวเลขรองรับ (Supporting Data):** {', '.join(_translate_report_text(item) for item in rs.supporting_data) if rs.supporting_data else 'ไม่มี hard data'}")
            for w in _translate_warnings(rs.validation_warnings):
                lines.append(f"> - ⚠️ **คำเตือน (Warning):** {w}")
            lines.append("")
    else:
        lines.append("> [!NOTE] ไม่พบแผนป้องกันความเสี่ยงที่เข้าเกณฑ์เงื่อนไขเชิงปริมาณในรอบการประเมินนี้\n")

    lines.append("## 📋 7. หมายเหตุด้านคุณภาพข้อมูลและระบบ\n")
    source_files = _display_source_files(direction, today)
    if source_files:
        lines.append(f"- **ไฟล์ต้นทางที่ประเมิน (Source Files Evaluated):** {', '.join(source_files)}")
    if getattr(direction, "data_timestamp_notes", []):
        lines.append(f"- **หมายเหตุเวลาของข้อมูล (Data Timestamp Notes):** {', '.join(direction.data_timestamp_notes)}")
    for sw in getattr(direction, "stale_data_warnings", []):
        lines.append(f"- ⚠️ **คำเตือนข้อมูลล่าช้า (Stale Data Warning):** {sw}")

    val_warnings = getattr(direction, "validation_warnings", [])
    if val_warnings:
        lines.append("\n> [!WARNING] ประกาศด้านคุณภาพและความครอบคลุมตามมาตรฐานสถาบัน")
        for w in _translate_warnings(val_warnings):
            lines.append(f"> - {w}")
        lines.append("")

    lines.append("\n### 🎯 ธีมหลักที่ควรจับตา (Focus Themes)\n")
    for theme in direction.focus_themes:
        lines.append(f"- {theme}")

    lines.append(f"\n### 💡 เหตุผลรองรับระดับความมั่นใจรวม (Conviction Rationale)\n{direction.conviction_rationale}\n")
    lines.append(
        "> [!CAUTION] ข้อสงวนสิทธิ์และคำชี้แจงการใช้งาน\n"
        "> รายงานฉบับนี้จัดทำขึ้นเพื่อใช้เป็นกรอบกลยุทธ์การลงทุนเชิงมหภาคและสนับสนุนการตัดสินใจเท่านั้น "
        "ไม่ถือเป็นคำแนะนำการลงทุนรายบุคคล คำสั่งซื้อขาย หรือการชี้ชวนให้ซื้อขายหลักทรัพย์ใดๆ ผู้ใช้งานควรประเมินข้อจำกัดและความเสี่ยงของพอร์ตการลงทุนก่อนดำเนินการเสมอ"
    )

    raw_markdown = "\n".join(lines)
    return repair_mojibake(raw_markdown)
