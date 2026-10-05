import json
import os
import re
import hashlib
from datetime import datetime
from pathlib import Path
from filelock import FileLock

from tools.archivist.core import VAULT_PATH, _atomic_write_text
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


def format_macro_strategy_report(
    direction: MacroStrategyDirection,
    *,
    sector_analysis: dict | None = None,
    strategy_report_id: str | None = None,
) -> str:
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

    if strategy_report_id:
        lines.insert(3, f"strategy_report_id: {strategy_report_id}")

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

    thai_stance = getattr(direction, "thailand_market_stance", None)
    if thai_stance and any(thai_stance.values()):
        flow_mb = thai_stance.get("investor_flow", {}).get("foreign_net_mb")
        ad_r = thai_stance.get("market_breadth", {}).get("advance_decline_ratio")
        ad_sent = thai_stance.get("market_breadth", {}).get("sentiment", "neutral")
        pe_v = thai_stance.get("valuation", {}).get("pe_ratio")
        gold_v = thai_stance.get("physical_gold", {}).get("bar_sell_thb")
        spread_v = thai_stance.get("policy_spread_bps")

        lines.append("### 🇹🇭 Thailand Market Stance & Microstructure (สรุปสภาวะตลาดทุนไทย)\n")
        lines.append("> [!note]- **สรุปทัศนะตลาดหุ้นและสภาพคล่องไทย (Microstructure Stance):**")
        thai_rationale = thai_stance.get("rationale")
        if thai_rationale:
            lines.append(f"> - **บทสรุปทัศนะ AI (Macro Stance Narrative):** {thai_rationale}")
        if flow_mb is not None:
            flow_label = "ต่างชาติซื้อสุทธิ" if flow_mb > 0 else "ต่างชาติขายสุทธิ"
            lines.append(f"> - **SET Foreign Net Flow:** {flow_mb:,.2f} ล้านบาท ({flow_label})")
        if ad_r is not None:
            lines.append(f"> - **Market Breadth (Advance/Decline Ratio):** {ad_r:.2f}x (Sentiment: {ad_sent})")
        if pe_v is not None:
            lines.append(f"> - **SET Valuation (P/E Ratio):** {pe_v:.2f}x")
        if gold_v is not None:
            lines.append(f"> - **ราคาทองคำแท่งในประเทศ (GTA Bar Sell):** {gold_v:,.0f} บาท/บาททองคำ")
        if spread_v is not None:
            sign = "+" if spread_v > 0 else ""
            lines.append(f"> - **ส่วนต่างอัตราดอกเบี้ยนโยบาย (Fed - BOT Spread):** {sign}{spread_v:.1f} bps")
        lines.append("> - **หมายเหตุสภาวะเศรษฐกิจมหภาค (Macro Regime):** สถานะเศรษฐกิจไทยยังคงเป็น 'Unknown' ตามนโยบาย Fail-Closed เนื่องจากอยู่ระหว่างรอเชื่อมต่อ API ทางการจาก สศช. (NESDC) และ สนค. (MOC)\n")

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

    if sector_analysis:
        lines.extend(["## Sector Rotation Context", ""])
        lines.append(f"Analysis status: `{sector_analysis.get('analysis_status') or 'unavailable'}`")
        if sector_analysis.get("unavailable_reason"):
            lines.append(f"Unavailable reason: `{sector_analysis['unavailable_reason']}`")
        lines.append(f"Snapshot: `{sector_analysis.get('snapshot_id') or 'unavailable'}` · as of `{sector_analysis.get('as_of_date') or 'unavailable'}`")
        if sector_analysis.get("summary_th"):
            lines.extend(["", str(sector_analysis["summary_th"])])
        for metric in sector_analysis.get("resolved_metrics", []) or []:
            if not isinstance(metric, dict):
                continue
            value = metric.get("numeric_value")
            rendered = f"{float(value):+.2f} {metric.get('unit', '')}" if isinstance(value, (int, float)) else str(metric.get("categorical_value") or "unavailable")
            lines.append(f"- {metric.get('ticker', '')} `{metric.get('metric_ref', '')}`: {rendered}; horizon {metric.get('horizon', '')}; as of {metric.get('metric_as_of', '')}.")
        if sector_analysis.get("validation_warnings"):
            lines.append("- Some LLM sector claims were rejected by deterministic reference/value validation.")
        lines.append("")

    if strategy_report_id:
        lines.append(f"\n[Open canonical report](/api/macro/reports/{strategy_report_id})")
    raw_markdown = "\n".join(lines)
    return repair_mojibake(raw_markdown)


_STRATEGY_SUBDIR = "30_Knowledge_Base/Strategies"


def _strategy_report_id(run_text: str) -> str:
    safe_run_id = re.sub(r"[^A-Za-z0-9_-]", "_", run_text)[:80] or "unknown"
    run_hash = hashlib.sha256(run_text.encode("utf-8")).hexdigest()[:10]
    return f"macro_report_{safe_run_id}_{run_hash}"


def _validate_sector_report_links(payload: dict) -> None:
    """Keep report-level, analysis-level, and resolved metric snapshot refs aligned."""
    sector_analysis = payload.get("sector_analysis")
    snapshot_ref = payload.get("sector_snapshot_id")
    if not isinstance(sector_analysis, dict):
        if snapshot_ref:
            raise RuntimeError("sector_report_analysis_missing_for_snapshot")
        return

    analysis_ref = sector_analysis.get("snapshot_id")
    status = sector_analysis.get("analysis_status")
    if snapshot_ref and snapshot_ref != analysis_ref:
        raise RuntimeError("sector_report_snapshot_link_mismatch")
    if status in {"available", "limited"} and (not snapshot_ref or analysis_ref != snapshot_ref):
        raise RuntimeError("sector_report_snapshot_link_missing")

    resolved = sector_analysis.get("resolved_metrics") or []
    if status == "unavailable" and (resolved or sector_analysis.get("fact_claims") or sector_analysis.get("watch_conditions")):
        raise RuntimeError("unavailable_sector_report_contains_claims")
    for claim in resolved:
        if not isinstance(claim, dict):
            raise RuntimeError("sector_report_resolved_metric_shape_invalid")
        input_refs = claim.get("input_refs") or []
        if (not analysis_ref or claim.get("snapshot_id") != analysis_ref
                or analysis_ref not in input_refs):
            raise RuntimeError("sector_report_metric_snapshot_ref_mismatch")


def _load_committed_strategy_report(report_id: str, vault_base: Path) -> tuple[dict, dict] | None:
    from tools.archivist.artifact_store import ArtifactError, DurableArtifactStore
    from tools.archivist.composition import build_knowledge_write_port
    from tools.archivist.vault_paths import VaultPaths

    paths = VaultPaths(vault_base)
    port = build_knowledge_write_port(vault_paths=paths)
    idempotency_key = f"macro-strategy-report:{report_id}"
    receipt = port.get_receipt(idempotency_key=idempotency_key)
    if receipt is None:
        return None
    if not receipt.is_success or not receipt.note_id or not receipt.revision_id:
        return None
    try:
        artifact = DurableArtifactStore(paths).get_revision_artifact(receipt.note_id, receipt.revision_id)
    except ArtifactError as exc:
        raise RuntimeError("macro_strategy_report_evidence_integrity_failure") from exc
    payload = json.loads(artifact.body)
    if payload.get("strategy_report_id") != report_id:
        raise RuntimeError("macro_strategy_report_artifact_identity_mismatch")
    evidence = {
        "note_id": receipt.note_id,
        "revision_id": receipt.revision_id,
        "relative_path": receipt.relative_path,
        "content_hash": receipt.content_hash,
        "artifact_set_hash": receipt.artifact_set_hash,
        "idempotency_key": receipt.idempotency_key,
    }
    return payload, evidence


def _stage_strategy_report(payload: dict, vault_base: Path) -> tuple[Path, dict]:
    from tools.archivist.runtime_layout import runtime_root_for

    report_id = str(payload["strategy_report_id"])
    pending_path = runtime_root_for(vault_base, create=True) / "macro_strategy_reports" / f"{report_id}.json"
    pending_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = pending_path.with_suffix(".lock")
    with FileLock(str(lock_path), timeout=30):
        if pending_path.exists():
            staged = json.loads(pending_path.read_text(encoding="utf-8"))
            if staged.get("strategy_report_id") != report_id:
                raise RuntimeError("pending_macro_strategy_report_identity_mismatch")
            _validate_sector_report_links(staged)
            return pending_path, staged
        _validate_sector_report_links(payload)
        _atomic_write_text(pending_path, json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
        return pending_path, payload


def write_strategy_json_sidecar(
    direction: MacroStrategyDirection,
    evaluated_date: str,
    *,
    observable_registry: dict | None = None,
    report_references: list[dict] | None = None,
    regional_assessments: dict | None = None,
    evaluated_sources: list[str] | None = None,
    run_id: str | None = None,
    job_id: str | None = None,
    run_started_at: str | None = None,
    sector_snapshot_id: str | None = None,
    sector_analysis: dict | None = None,
    return_canonical_payload: bool = False,
) -> Path | tuple[Path, dict]:
    """เขียน direction เป็น JSON sidecar คู่กับรายงาน .md ที่ Archivist จะบันทึกทีหลัง

    เขียนตรงจาก Python (atomic, ไม่ผ่าน Archivist LLM tool-call) เพราะเนื้อหา JSON
    มีขนาดใหญ่และการฝังไปในข้อความที่ไหลผ่าน LLM context เสี่ยง mangle/truncate —
    ไฟล์นี้คือ source of truth สำหรับ Web API, ไม่ใช่สำหรับแสดงใน Obsidian
    """
    from tools.macro.dashboard import build_dashboard_indicators, persist_indicator_series
    vault_base = Path(os.getenv("OBSIDIAN_VAULT_PATH", str(VAULT_PATH))).resolve()

    dashboard_indicators = build_dashboard_indicators(direction, observable_registry, vault_base)
    # Predict chart availability from the current point plus the existing series,
    # without changing any projection before the canonical report is committed.
    for ind in dashboard_indicators:
        s_key = ind.get("series_key", "")
        if ind.get("value") is not None:
            try:
                from tools.macro.dashboard import _series_path, _SERIES_SUBDIR
                path = _series_path(vault_base, s_key)
                if not path.exists():
                    legacy = vault_base / _SERIES_SUBDIR / f"{s_key}.json"
                    path = legacy if legacy.exists() else None
                if path and path.exists():
                    series_data = json.loads(path.read_text(encoding="utf-8"))
                    pts = series_data.get("points", [])
                    distinct_dates = {p.get("observed_at") for p in pts if isinstance(p, dict) and "observed_at" in p}
                    if ind.get("observed_at"):
                        distinct_dates.add(ind["observed_at"])
                    ind["chart_available"] = len(distinct_dates) >= 2
            except Exception:
                ind["chart_available"] = False

    payload = direction.model_dump(mode="json")
    payload["dashboard_indicators"] = dashboard_indicators
    payload["report_references"] = report_references or []
    payload["regional_assessments"] = regional_assessments or {}
    payload["evaluated_sources"] = evaluated_sources or []
    payload["run_id"] = run_id or f"run_{evaluated_date}_{int(datetime.now().timestamp())}"
    payload["job_id"] = job_id or payload["run_id"]
    payload["snapshot_id"] = f"Macro_Strategy_Direction_{evaluated_date}"
    payload["sector_snapshot_id"] = sector_snapshot_id
    if hasattr(sector_analysis, "model_dump"):
        sector_analysis = sector_analysis.model_dump(mode="json")
    elif sector_analysis is not None and not isinstance(sector_analysis, dict):
        raise TypeError("sector_analysis_must_be_a_mapping_or_pydantic_model")
    payload["sector_analysis"] = sector_analysis
    payload["run_started_at"] = run_started_at or datetime.now().astimezone().isoformat()
    run_text = str(payload["run_id"])
    report_id = _strategy_report_id(run_text)
    payload["strategy_report_id"] = report_id
    if observable_registry:
        payload["observable_registry"] = {
            k: (v.model_dump(mode="json") if hasattr(v, "model_dump") else v)
            for k, v in observable_registry.items()
        }
    from tools.archivist.vault_paths import VaultPaths
    vp = VaultPaths(vault_base)
    preflight_dir = (vault_base / "30_Knowledge_Base" / "Macroeconomics" / "Strategies" / evaluated_date[:4] / evaluated_date[5:7]
                     if vp.layout_version >= 2 and len(evaluated_date) >= 7 else vault_base / _STRATEGY_SUBDIR)
    preflight_json = preflight_dir / f"Macro_Strategy_Direction_{evaluated_date}.json"
    preflight_latest = (vault_base / "30_Knowledge_Base" / "Macroeconomics" / "Strategies"
                        if vp.layout_version >= 2 else vault_base / _STRATEGY_SUBDIR) / "Macro_Strategy_Latest.json"
    from tools.archivist.maintenance_guard import assert_write_allowed
    assert_write_allowed(preflight_json)
    assert_write_allowed(preflight_latest)
    existing = _load_committed_strategy_report(report_id, vault_base)
    pending_path = None
    if existing:
        payload, archive_ref = existing
        dashboard_indicators = payload.get("dashboard_indicators", [])
    else:
        pending_path, payload = _stage_strategy_report(payload, vault_base)
        dashboard_indicators = payload.get("dashboard_indicators", [])
        archive_ref = _archive_strategy_report(payload, vault_base)
        pending_path.unlink(missing_ok=True)
    projection_date = str(payload.get("snapshot_id", "")).removeprefix("Macro_Strategy_Direction_") or evaluated_date
    if vp.layout_version >= 2 and len(projection_date) >= 7:
        target_dir = vault_base / "30_Knowledge_Base" / "Macroeconomics" / "Strategies" / projection_date[:4] / projection_date[5:7]
    else:
        target_dir = vault_base / _STRATEGY_SUBDIR
    json_path = target_dir / f"Macro_Strategy_Direction_{projection_date}.json"
    latest_dir = (vault_base / "30_Knowledge_Base" / "Macroeconomics" / "Strategies"
                  if vp.layout_version >= 2 else vault_base / _STRATEGY_SUBDIR)
    latest_manifest = latest_dir / "Macro_Strategy_Latest.json"
    assert_write_allowed(json_path)
    assert_write_allowed(latest_manifest)
    payload["strategy_report_evidence"] = archive_ref
    try:
        persist_indicator_series(vault_base, dashboard_indicators)
        target_dir.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(json_path.with_suffix(json_path.suffix + ".lock")), timeout=15)
        with lock:
            publish_daily_projection = True
            if json_path.exists():
                try:
                    current = json.loads(json_path.read_text(encoding="utf-8"))
                    old_key = (str(current.get("evaluated_at", ""))[:10], str(current.get("run_started_at", "")))
                    new_key = (str(payload.get("evaluated_at", ""))[:10], str(payload.get("run_started_at", "")))
                    publish_daily_projection = new_key >= old_key
                except (OSError, json.JSONDecodeError, AttributeError):
                    publish_daily_projection = True
            if publish_daily_projection:
                _atomic_write_text(json_path, json.dumps(payload, ensure_ascii=False, indent=2))

        # The durable latest pointer is published only after the canonical report commit.
        latest_lock = FileLock(str(latest_manifest.with_suffix(latest_manifest.suffix + ".lock")), timeout=15)
        with latest_lock:
            publish_manifest = True
            if latest_manifest.exists():
                try:
                    current = json.loads(latest_manifest.read_text(encoding="utf-8"))
                    old_key = (str(current.get("evaluated_at", ""))[:10], str(current.get("run_started_at", "")))
                    new_key = (str(payload.get("evaluated_at", ""))[:10], str(payload.get("run_started_at", "")))
                    publish_manifest = new_key >= old_key
                except (OSError, json.JSONDecodeError, AttributeError):
                    publish_manifest = True
            if publish_manifest:
                manifest = {
                    "strategy_report_id": payload["strategy_report_id"],
                    "evaluated_at": payload.get("evaluated_at"),
                    "run_started_at": payload.get("run_started_at"),
                }
                _atomic_write_text(latest_manifest, json.dumps(manifest, ensure_ascii=False, indent=2))
    except Exception as exc:
        # The committed report remains available by ID; projections can be rebuilt on retry.
        import logging
        logging.getLogger(__name__).warning(
            "Committed Macro report projection failed (%s): %s", payload["strategy_report_id"], exc,
        )
    return (json_path, payload) if return_canonical_payload else json_path


def _archive_strategy_report(payload: dict, vault_base: Path) -> dict:
    """Commit an immutable report through the KnowledgeWritePort before projection."""
    from application.knowledge.write_models import KnowledgeWriteCommand
    from tools.archivist.artifact_store import DurableArtifactStore
    from tools.archivist.composition import build_knowledge_write_port
    from tools.archivist.vault_paths import VaultPaths

    report_id = str(payload["strategy_report_id"])
    document_key = f"macro:strategy_report:{report_id}"
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(body.encode("utf-8")) > 4_500_000:
        raise RuntimeError("macro_strategy_report_exceeds_safe_payload_limit")
    command = KnowledgeWriteCommand(
        operation="upsert_note",
        idempotency_key=f"macro-strategy-report:{report_id}",
        document_key=document_key,
        entity_type="macro_strategy",
        producer="macro-strategic-allocator",
        producer_version="sector-rotation-v2",
        actor="macro-strategy-report",
        payload={
            "metadata": {
                "schema_version": 2,
                "document_key": document_key,
                "entity_type": "macro_strategy",
                "title": f"Macro Strategy Report {payload.get('evaluated_at', '')}",
                "as_of_date": str(payload.get("evaluated_at", ""))[:10],
                "strategy_report_id": report_id,
                "sector_snapshot_id": payload.get("sector_snapshot_id"),
            },
            "body": body,
            "filename": f"Macro_Strategy_Report_{report_id}.md",
            "profile_id": "published",
        },
    )
    paths = VaultPaths(vault_base)
    receipt = build_knowledge_write_port(vault_paths=paths).submit(command)
    if not receipt.is_success or not receipt.note_id or not receipt.revision_id:
        raise RuntimeError(f"macro_strategy_report_not_committed:{receipt.status}:{receipt.error_code or 'unknown'}")
    artifact = DurableArtifactStore(paths).get_revision_artifact(receipt.note_id, receipt.revision_id)
    archived = json.loads(artifact.body)
    if archived.get("strategy_report_id") != report_id or archived != json.loads(body):
        raise RuntimeError("macro_strategy_report_artifact_identity_mismatch")
    return {
        "note_id": receipt.note_id,
        "revision_id": receipt.revision_id,
        "relative_path": receipt.relative_path,
        "content_hash": receipt.content_hash,
        "artifact_set_hash": receipt.artifact_set_hash,
        "idempotency_key": receipt.idempotency_key,
    }
