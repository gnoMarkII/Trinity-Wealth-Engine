import re
from datetime import datetime

from schemas.macro_schemas import AssetStance, MacroStrategyDirection


_MOJIBAKE_MARKERS = ("à¸", "à¹", "ðŸ", "â€", "â€”", "âš", "Ã", "Â")


def _repair_mojibake_chunk(text: str) -> str:
    repaired = text
    for _ in range(3):
        if not any(marker in repaired for marker in _MOJIBAKE_MARKERS):
            break
        changed = False
        for encoding in ("cp1252", "latin1"):
            try:
                candidate = repaired.encode(encoding).decode("utf-8")
            except UnicodeError:
                continue
            if candidate != repaired:
                repaired = candidate
                changed = True
                break
        if not changed:
            break
    return repaired


def repair_mojibake(text: str) -> str:
    """Repair common UTF-8 decoded-as-Windows mojibake without touching real Thai text."""
    if not any(marker in text for marker in _MOJIBAKE_MARKERS):
        return text
    # If model output already contains valid Thai, split on Thai spans and only
    # repair the non-Thai spans that can still contain formatter/prompt mojibake.
    parts = re.split(r"([\u0e00-\u0e7f]+)", text)
    return "".join(
        part if re.fullmatch(r"[\u0e00-\u0e7f]+", part) else _repair_mojibake_chunk(part)
        for part in parts
    )


def _join_or_dash(items: list[str]) -> str:
    return ", ".join(items) if items else "-"


def _translate_warning(message: str) -> str:
    translations = {
        "Defensive Degradation: Downgraded confidence to LOW due to lack of numeric hard data in supporting_data.": "ลดระดับความมั่นใจเป็น LOW เพราะไม่มีตัวเลข hard data ใน supporting_data",
        "Source Reference Backfill: source_refs populated from MacroStrategyDirection.source_files because the asset view omitted explicit refs.": "เติมแหล่งอ้างอิงจาก source_files ของรายงาน เพราะมุมมองสินทรัพย์ไม่ได้ระบุ source_refs โดยตรง",
        "Source Reference Warning: source_refs is empty, so data confidence cannot be treated as high.": "source_refs ว่าง จึงไม่ควรถือว่าความมั่นใจด้านข้อมูลเป็นระดับสูง",
        "Source Reference Warning: asset view omitted source_refs.": "มุมมองสินทรัพย์ไม่ได้ระบุ source_refs",
        "Coverage Backfill: Added low-confidence neutral view because Strategic Allocator omitted this core asset class.": "เติมมุมมอง Neutral ระดับความมั่นใจต่ำ เพราะ Strategic Allocator ไม่ได้วิเคราะห์สินทรัพย์หลักกลุ่มนี้",
        "Regime Probability Coverage Warning: fewer than 4 regime probabilities were provided; distribution is incomplete for institutional review.": "การกระจายความน่าจะเป็นของ regime ยังไม่ครบอย่างน้อย 4 สภาวะ จึงยังไม่พอสำหรับการทบทวนระดับสถาบัน",
        "Contradiction Warning: Portfolio recommends Overweight on both Equities Growth and Long Treasuries/Bonds without explicit reconciliation (e.g., Barbell strategy or Duration hedge).": "พบความขัดแย้ง: พอร์ตแนะนำ Overweight ทั้งหุ้นเติบโตและพันธบัตร/Duration โดยยังไม่อธิบายให้ชัดว่าเป็น Barbell strategy หรือ Duration hedge",
        "Stale Data Degradation: Downgraded overall conviction level from HIGH to MEDIUM due to presence of stale data warnings.": "ลดระดับ conviction รวมจาก HIGH เป็น MEDIUM เพราะมีคำเตือนเรื่องข้อมูลล่าช้า",
    }
    if message in translations:
        return translations[message]
    if message.startswith("Single-Source Penalty:"):
        return "ลดความมั่นใจเป็น MEDIUM เพราะมุมมองนี้อ้างอิงแหล่งข้อมูลโดยตรงเพียงแหล่งเดียว"
    if message.startswith("Stale Data Degradation:"):
        return "ลดระดับ conviction รวมลงเป็น MEDIUM เพราะมีคำเตือนเรื่องข้อมูลล่าช้า แต่ยังไม่ลดเป็น LOW หากไม่ได้มีหลักฐานเสียหายระดับวิกฤต"
    if message.startswith("Source Reference Penalty:"):
        return "ลดระดับความมั่นใจจาก HIGH เป็น MEDIUM เพราะ source_refs ถูกอนุมานจาก source_files ระดับรายงาน"
    if message.startswith("Gold Rationale Warning:"):
        return "ลดความมั่นใจของมุมมอง Overweight ทองคำเป็น MEDIUM เพราะเหตุผลยังพึ่งพาความเสี่ยงภูมิรัฐศาสตร์มากเกินไป และยังไม่ผูกกับ real yields เงินเฟ้อ หรือนโยบายการเงินอย่างชัดเจน"
    if message.startswith("Contradiction Degradation: Gold Overweight rationale lacks yield/inflation macro anchoring."):
        return "ลดความมั่นใจของทองคำเพราะเหตุผล Overweight ยังไม่มีหลักยึดจาก yield เงินเฟ้อ หรือกรอบนโยบายการเงินเพียงพอ"
    if message.startswith("Coverage Warning: Missing required core asset classes"):
        return "เติมสินทรัพย์หลักที่ขาดเพื่อให้ครบกรอบ Equities, Bonds/Duration, Commodities, FX และ Cash"
    if message.startswith("Coverage Warning: Asset allocation covers"):
        return "ความครอบคลุมสินทรัพย์ยังไม่ครบกรอบหลัก 5 กลุ่ม จึงเติมรายการที่ขาดเป็นมุมมอง Neutral ตามข้อจำกัดของข้อมูล"
    if message.startswith("Portfolio Conviction Cap:"):
        return "ลดระดับความมั่นใจของสินทรัพย์จาก HIGH เป็น MEDIUM เพราะ conviction รวมของพอร์ตเป็น LOW"
    if message.startswith("Implementation Confidence Cap:"):
        return "ลดระดับความมั่นใจด้านการนำไปปฏิบัติจาก HIGH เป็น MEDIUM เพราะ source refs หรือ conviction รวมยังไม่แข็งพอ"
    if message.startswith("Pair Trade Execution Guardrail:"):
        return "ลดความมั่นใจของ pair trade เป็น LOW เพราะข้อมูล hedge ratio, จุดตัดขาดทุน, เป้าหมาย, drawdown หรือเครื่องมือที่ใช้จริงยังไม่ครบ"
    if message.startswith("Regime Probability Backfill:"):
        return "เติม scenario สำรองให้ regime probabilities ครบอย่างน้อย 4 สภาวะตามเกณฑ์ institutional"
    if message == "Defensive Degradation: Downgraded confidence to LOW due to incomplete indicators/hedges or missing numeric hard data.":
        return "ลดระดับความมั่นใจเป็น LOW เพราะตัวชี้วัดล่วงหน้า เครื่องมือ hedge หรือ hard data เชิงตัวเลขยังไม่ครบ"
    if message.startswith("Active Allocation Guardrail: Changed stance from "):
        match = re.search(r"from (.+?) to Neutral because (.+?)\.", message)
        if match:
            return f"ปรับมุมมองจาก {match.group(1)} เป็น Neutral เพราะ {_translate_report_text(match.group(2))}"
    if message.startswith("Coverage Backfill: Asset allocation expanded from "):
        match = re.search(r"expanded from (\d+) to (\d+)", message)
        if match:
            return f"เติมสินทรัพย์หลักให้ครบจาก {match.group(1)} เป็น {match.group(2)} กลุ่ม โดยใช้มุมมอง Neutral ความมั่นใจต่ำในส่วนที่ข้อมูลไม่พอ"
    if message.startswith("Defensive Degradation: Portfolio conviction downgraded to LOW because"):
        return "ความมั่นใจรวมอยู่ในระดับต่ำ: ลด conviction รวมเป็น LOW เพราะมุมมองสินทรัพย์ตั้งแต่ครึ่งหนึ่งขึ้นไปขาดตัวเลข hard data หรือมีความมั่นใจ LOW"
    match = re.match(r"Graceful Drop: (\d+) pair trade\(s\) omitted due to insufficient numeric market data or missing executable controls\.", message)
    if match:
        return f"ตัด Pair Trade ออกอย่างปลอดภัยจำนวน {match.group(1)} รายการ เนื่องจากหลักฐานตัวเลขราคาตลาดหรือรายละเอียดการปฏิบัติการยังไม่ครบถ้วน"
    match = re.match(r"Graceful Drop: (\d+) pair trade\(s\) omitted due to insufficient numeric market data\.", message)
    if match:
        return f"ตัด Pair Trade ออกอย่างปลอดภัยจำนวน {match.group(1)} รายการ เนื่องจากหลักฐานตัวเลขราคาตลาดหรือสถิติเชิงสัมพัทธ์ไม่เพียงพอ"
    match = re.match(r"Graceful Drop: (\d+) risk scenario\(s\) omitted due to insufficient numeric market data\.", message)
    if match:
        return f"ตัดแผนป้องกันความเสี่ยงออกอย่างปลอดภัยจำนวน {match.group(1)} รายการ เนื่องจากตัวเลข Hard Data ไม่เพียงพอ"
    if message.startswith("Pair Trade Graceful Downgrade:"):
        return "ลดความมั่นใจของ Pair Trade เป็น MEDIUM เนื่องจากใช้หลักฐานเชิงสัมพัทธ์จากจุดเดียวแทนการคำนวณ beta/correlation จากอนุกรมเวลาย้อนหลัง"
    if message.startswith("Graceful Drop:"):
        return message.replace("Graceful Drop:", "ตัดรายการที่ข้อมูลไม่พอออกอย่างปลอดภัย:")
    if message.startswith("Regime Probability Sum Warning:"):
        return message.replace("Regime Probability Sum Warning:", "คำเตือนผลรวมความน่าจะเป็น regime:")
    if message.startswith("Regime Consistency Adjustment:"):
        return message.replace("Regime Consistency Adjustment:", "ปรับ regime ให้สอดคล้อง:")
    return message


def _translate_warnings(messages: list[str]) -> list[str]:
    return [_translate_warning(repair_mojibake(message)) for message in messages]


def _translate_report_text(value: str) -> str:
    text = repair_mojibake(str(value)) if value is not None else ""
    replacements = {
        "No hard data": "ไม่มี hard data",
        "Confidence is LOW due to macro uncertainty or data constraints.": "ความมั่นใจเป็น LOW เพราะยังมีความไม่แน่นอนเชิงมหภาคหรือข้อจำกัดด้านข้อมูล",
        "Backfilled low-confidence neutral view due to insufficient source evidence.": "เติมมุมมอง Neutral ระดับความมั่นใจต่ำ เพราะหลักฐานจากแหล่งข้อมูลยังไม่เพียงพอ",
        "Data Insufficient: input evidence is not strong enough to form an active allocation view for this core asset class.": "ข้อมูลไม่เพียงพอที่จะสร้างมุมมองเชิงรุกสำหรับสินทรัพย์หลักกลุ่มนี้",
        "asset view lacks sufficient numeric hard data for an active allocation": "มุมมองสินทรัพย์ยังขาดตัวเลข hard data ที่เพียงพอสำหรับการจัดสรรเชิงรุก",
        "confidence is LOW or numeric hard data is insufficient": "ความมั่นใจเป็น LOW หรือข้อมูลตัวเลข hard data ยังไม่เพียงพอ",
        "overall portfolio conviction is LOW and the asset view is not HIGH confidence": "conviction รวมของพอร์ตเป็น LOW และมุมมองสินทรัพย์ไม่ได้มีความมั่นใจระดับ HIGH",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _display_conviction_level(direction: MacroStrategyDirection) -> str:
    level = str(getattr(direction, "conviction_level", "medium")).lower()
    warnings = [str(w) for w in getattr(direction, "validation_warnings", [])]
    has_defensive_low = any("Defensive Degradation: Portfolio conviction downgraded to LOW" in w for w in warnings)
    has_legacy_stale_low = any("Stale Data Degradation:" in w and "LOW" in w for w in warnings)
    if level == "low" and getattr(direction, "stale_data_warnings", []) and has_legacy_stale_low and not has_defensive_low:
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
    )
    if getattr(asset, "confidence", "medium") == "high" or not weak:
        return text or "-"
    warnings = [str(w) for w in getattr(asset, "validation_warnings", [])]
    if any("Single-Source Penalty" in w for w in warnings):
        return "ความมั่นใจถูกจำกัดไว้ที่ MEDIUM เพราะมุมมองนี้อ้างอิงแหล่งข้อมูลโดยตรงเพียงแหล่งเดียว"
    if "gold" in str(getattr(asset, "asset_class", "")).lower():
        return "ความมั่นใจถูกจำกัดไว้ที่ MEDIUM เพราะมุมมองทองคำยังต้องอ้างอิง real yields เงินเฟ้อ หรือนโยบายการเงินให้ชัดกว่านี้"
    if getattr(asset, "confidence", "medium") == "low":
        return "ข้อมูลตัวเลขและหลักฐานอ้างอิงยังไม่เพียงพอสำหรับความมั่นใจระดับ HIGH"
    return "ยังมีข้อจำกัดด้านข้อมูลหรือการนำไปปฏิบัติ จึงยังไม่เหมาะกับความมั่นใจระดับ HIGH"


def _display_source_files(direction: MacroStrategyDirection, today: str) -> list[str]:
    source_files = list(getattr(direction, "source_files", []) or [])
    baseline = f"Macro_Baseline_{today}.md"
    if baseline not in source_files:
        source_files.append(baseline)
    return source_files


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
        f"> **ระดับความมั่นใจ (Conviction):** {'ความมั่นใจรวมอยู่ในระดับต่ำ (LOW)' if display_conviction == 'low' else display_conviction.upper()}",
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
            f"| **{a.asset_class}** | {stance_fmt} | {getattr(a, 'allocation_delta', '') or '-'} | "
            f"{getattr(a, 'benchmark_ref', '') or '-'} | {getattr(a, 'time_horizon', 'Macro (3-6 Months)')} | "
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
    if has_contradictions:
        lines.append("## ⚡ 4. Key Contradictions & Quant-Narrative Divergence (จุดขัดแย้งเชิงตรรกะ)\n")
        if direction.quant_narrative_alignment == "divergent":
            lines.append(f"> [!WARNING] Quant-Narrative Divergence — ความไม่สอดคล้องระหว่าง Quant และ Narrative\n> {direction.divergence_note}\n")
        elif direction.divergence_note:
            lines.append(f"> [!NOTE] หมายเหตุความแตกต่าง (Divergence Note)\n> {direction.divergence_note}\n")
        for w in _translate_warnings(getattr(direction, "validation_warnings", [])):
            if "Contradiction" in w or "Divergent" in w:
                lines.append(f"> [!WARNING] ข้อความเตือนความขัดแย้งจากระบบ (Auto-Detected Guardrail Contradiction)\n> {w}\n")

    pair_trades = getattr(direction, "pair_trades", [])
    if pair_trades:
        lines.append("## ⚖️ Relative Value & Pair Trades (Trade Ideas) — กลยุทธ์จับคู่เทรดเชิงมูลค่าสัมพัทธ์\n")
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

    risk_scenarios = getattr(direction, "risk_scenarios", [])
    if risk_scenarios:
        lines.append("## 🛡️ Portfolio Risk Mitigation & Hedging (Hedging Plan) — แผนการบริหารความเสี่ยงและป้องกันพอร์ต\n")
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

    lines.append("## 📋 7. Data Quality Notes & Institutional Metadata (หมายเหตุด้านคุณภาพข้อมูลและระบบ)\n")
    source_files = _display_source_files(direction, today)
    if source_files:
        lines.append(f"- **ไฟล์ต้นทางที่ประเมิน (Source Files Evaluated):** {', '.join(source_files)}")
    if getattr(direction, "data_timestamp_notes", []):
        lines.append(f"- **หมายเหตุเวลาของข้อมูล (Data Timestamp Notes):** {', '.join(direction.data_timestamp_notes)}")
    for sw in getattr(direction, "stale_data_warnings", []):
        lines.append(f"- ⚠️ **คำเตือนข้อมูลล่าช้า (Stale Data Warning):** {sw}")

    val_warnings = getattr(direction, "validation_warnings", [])
    if val_warnings:
        lines.append("\n> [!WARNING] ประกาศด้านคุณภาพและความครอบคลุมตามมาตรฐานสถาบัน (Institutional Quality & Coverage Notices)")
        for w in _translate_warnings(val_warnings):
            lines.append(f"> - {w}")
        lines.append("")

    lines.append("\n### 🎯 ธีมหลักที่ควรจับตา (Focus Themes)\n")
    for theme in direction.focus_themes:
        lines.append(f"- {theme}")

    lines.append(f"\n### 💡 เหตุผลรองรับระดับความมั่นใจรวม (Conviction Rationale)\n{direction.conviction_rationale}\n")
    lines.append(
        "> [!CAUTION] Institutional Compliance Disclaimer\n"
        "> รายงานฉบับนี้จัดทำขึ้นเพื่อใช้เป็นกรอบกลยุทธ์การลงทุนเชิงมหภาคและสนับสนุนการตัดสินใจเท่านั้น "
        "ไม่ถือเป็นคำแนะนำการลงทุนรายบุคคล คำสั่งซื้อขาย หรือการชี้ชวนให้ซื้อขายหลักทรัพย์ใดๆ ผู้ใช้งานควรประเมินข้อจำกัดและความเสี่ยงของพอร์ตการลงทุนก่อนดำเนินการเสมอ"
    )

    return repair_mojibake("\n".join(lines))
