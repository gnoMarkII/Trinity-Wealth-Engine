"""Portfolio Advisory Synthesizer Agent.

Architecture Layer: agents/
Single-pass LLM synthesizer generating strategic advisory playbooks with
post-LLM consistency validation (Guardrail I6) and deterministic fallback recovery.
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
import json
import logging
from typing import Any, Optional

from langchain_core.language_models.chat_models import BaseChatModel

from core.portfolio_advisory.models import (
    AdvisoryActionCode,
    PortfolioAdvisoryQuantReport,
)
from core.portfolio_advisory.narrative_validator import validate_advisory_narrative_consistency
from core.prompt_harness import get_harness
from schemas.portfolio_advisory_schemas import PortfolioAdvisoryNarrativeDTO

log = logging.getLogger(__name__)


def build_deterministic_fallback_narrative(
    quant_report: PortfolioAdvisoryQuantReport,
    notice: Optional[str] = None,
) -> PortfolioAdvisoryNarrativeDTO:
    """Generates an accurate, deterministic playbook when LLM synthesis is unavailable or fails."""
    cash_plan = quant_report.cash_plan
    radar = quant_report.risk_radar

    # 1. Executive Summary
    macro_summary = quant_report.macro_regime_summary
    regime = macro_summary.get("overall_regime", "Balanced Growth") if macro_summary else "Unassigned / Neutral"
    summary_text = (
        f"พอร์ตโฟลิโอปัจจุบันมีมูลค่าสินทรัพย์สุทธิ {quant_report.total_nav_thb:,.2f} บาท "
        f"สภาวะการวิเคราะห์ระดับ {quant_report.data_mode.value} ภายใต้สภาวะมหภาค '{regime}'. "
        f"มีสภาพคล่องเงินสดพร้อมใช้งาน {cash_plan.cash_thb_available:,.2f} THB และ ${cash_plan.cash_usd_available:,.2f} USD."
    )
    if notice:
        summary_text += f" [ระบบสังเคราะห์กลยุทธ์สำรองแบบ Deterministic: {notice}]"

    # 2. Immediate Playbook
    immediate_items = []
    for h in quant_report.holdings_advisory:
        if h.is_immediate_execution:
            lot_str = ""
            if h.lot_sizing:
                if h.lot_sizing.board_lot_shares is not None:
                    lot_str = f"จำนวน {h.lot_sizing.board_lot_shares:,} หุ้น (Board Lot) + {h.lot_sizing.odd_lot_shares} หุ้นเศษ"
                elif h.lot_sizing.whole_shares is not None:
                    lot_str = f"จำนวน {h.lot_sizing.whole_shares} หุ้น (หรือ {h.lot_sizing.fractional_shares} หุ้น Fractional)"
            bullet = f"[{h.action_code.value}] {h.symbol} ({h.asset_name}) — {h.action_rationale}. {lot_str}".strip()
            immediate_items.append(bullet)

    if cash_plan.fx_conversion_needed_thb > 0:
        immediate_items.append(
            f"[FX_CONVERSION] สั่งซื้อหุ้น US เกินดอลลาร์ในพอร์ต — แนะนำแลกเงินบาทเพิ่ม {cash_plan.fx_conversion_needed_thb:,.2f} THB"
        )

    if not immediate_items:
        immediate_items.append("ไม่มีคำสั่งที่ต้องดำเนินการทันทีวันนี้ — สัดส่วนสินทรัพย์ปัจจุบันอยู่ในกรอบสมดุลความเสี่ยง")

    # 3. Conditional Dip Plan
    dip_items = []
    for h in quant_report.holdings_advisory:
        if h.action_code == AdvisoryActionCode.ACCUMULATE_ON_DIP:
            zone_str = f"โซนตั้งรับ {h.conditional_buy_zone_min} - {h.conditional_buy_zone_max}" if h.conditional_buy_zone_min else "โซนราคาต่ำกว่าราคาปัจจุบัน"
            dip_items.append(
                f"[ACCUMULATE_ON_DIP] {h.symbol} — {zone_str} ตามกรอบแนวรับทางเทคนิค (Conviction Score: {h.core_conviction_score}/10)"
            )

    if not dip_items:
        dip_items.append("ไม่มีหุ้นที่อยู่ในสถานะรอรับเมื่อย่อตัว (Accumulate on Dip)")

    # 4. Risk Mitigation & Hedging
    risk_items = []
    if radar.sector_crowding:
        crowded_str = ", ".join(f"{k} ({v:.1f}%)" for k, v in radar.sector_crowding.items())
        risk_items.append(f"ความเสี่ยงกระจุกตัวในหมวดอุตสาหกรรมเกิน 30%: {crowded_str}")

    if radar.unassigned_drift_guard_active:
        risk_items.append(
            f"สินทรัพย์ที่ยังไม่ได้จัดเข้า Strategy Bucket คิดเป็น {radar.unassigned_holdings_nav_pct:.1f}% ของ NAV — ควรจัดกลุ่มก่อนเพิ่มน้ำหนัก"
        )

    for q in quant_report.quarantined_holdings:
        risk_items.append(f"[QUARANTINED] {q.symbol} ถูกระงับคำสั่งซื้อขาย: {q.reason}")

    if not risk_items:
        risk_items.append("ความเสี่ยงภาพรวมอยู่ในเกณฑ์ปกติ การกระจายตัวของความเสี่ยงและอัตราแลกเปลี่ยนมีความสมดุล")

    # 5. What's Working (Positive Tailwinds & Defensible Assets)
    working_items = []
    for h in quant_report.holdings_advisory:
        if h.action_code in (AdvisoryActionCode.LET_WINNER_RUN, AdvisoryActionCode.ACCUMULATE_NOW) or (
            h.core_conviction_score and h.core_conviction_score >= 8.0
        ):
            working_items.append(
                f"{h.symbol} ({h.asset_name}) — ปัจจัยพื้นฐานและระดับ Conviction โดดเด่น ({h.core_conviction_score}/10) สอดรับกับโมเมนตัมปัจจุบัน"
            )
    if cash_plan.cash_thb_available > 0 or cash_plan.cash_usd_available > 0:
        working_items.append(
            f"กระสุนเงินสดรวมพร้อมใช้งาน ({cash_plan.cash_thb_available:,.0f} THB / ${cash_plan.cash_usd_available:,.0f} USD) เพิ่มขีดความสามารถและความยืดหยุ่น (Optionality) ในการเลือกลงทุนยามตลาดผันผวน"
        )
    if not working_items:
        working_items.append("พอร์ตมีสัดส่วนสินทรัพย์กระจายตัวตามเกณฑ์มาตรฐานความเสี่ยง")

    # 6. What Concerns Me (Headwinds & Unhedged Gaps)
    concerns_items = []
    for h in quant_report.holdings_advisory:
        if h.action_code in (AdvisoryActionCode.URGENT_REDUCE, AdvisoryActionCode.TRIM_OVERWEIGHT):
            concerns_items.append(f"{h.symbol} — {h.action_rationale} (น้ำหนักปัจจุบัน {h.current_weight_pct:.1f}%)")
    if radar.effective_holdings_count and radar.effective_holdings_count < Decimal("4.0"):
        concerns_items.append(
            f"ความกระจุกตัวสูงระดับพอร์ต: มีพฤติกรรมเสมือนถือหุ้นเพียง {radar.effective_holdings_count:.1f} ตัว (Effective Holdings / HHI {radar.hhi:.4f}) โดย Top 3 ครองสัดส่วนถึง {radar.top_3_weight_pct:.1f}% ของ NAV"
        )
    if radar.sector_crowding:
        for sec, pct in radar.sector_crowding.items():
            concerns_items.append(f"การกระจุกตัวในกลุ่ม {sec} สูงถึง {pct:.1f}% ของ NAV ซึ่งขาดการป้องกันหากเกิด Sector Rotation")
    if not concerns_items:
        concerns_items.append("ไม่พบความเสี่ยงกระจุกตัวหรือสัญญาณอันตรายระดับวิกฤตในขณะนี้")

    # 7. Strategic Discussion Points
    discussion_items = []
    if radar.concentration_risk_level == "HIGH" and radar.hhi:
        discussion_items.append(
            f"พิจารณากระจายความเสี่ยงเพื่อลดดัชนีกระจุกตัว HHI ({radar.hhi:.4f}) โดยทยอยลดน้ำหนักหุ้นที่ครองเกิน 15–20% ของพอร์ต"
        )
    usd_fx_pct = radar.fx_exposure_pct.get("USD", 0)
    if usd_fx_pct > 40:
        discussion_items.append(
            f"สัดส่วนความเสี่ยงอัตราแลกเปลี่ยน USD อยู่ที่ {usd_fx_pct:.1f}% — ควรพิจารณาแนวทางบริหาร FX Exposure หรือจุดล็อกกำไรค่าเงิน"
        )
    if radar.unassigned_drift_guard_active:
        discussion_items.append("พิจารณากำหนดกรอบกลยุทธ์ (Strategy Bucket) ให้กับสินทรัพย์ที่ยังไม่ได้จัดกลุ่มเพื่อความชัดเจนในการติดตามผล")
    discussion_items.append("ทบทวนสมมติฐานสัดส่วนสินทรัพย์เติบโตเทียบกับสินทรัพย์ป้องกันความเสี่ยง (Real Assets / Hedges) ภายใต้สภาวะมหภาคปัจจุบัน")

    return PortfolioAdvisoryNarrativeDTO(
        synthesized_at=datetime.now(timezone.utc).isoformat(),
        portfolio_state_hash=quant_report.portfolio_state_hash,
        macro_executive_summary=summary_text,
        immediate_action_playbook=immediate_items,
        conditional_dip_plan=dip_items,
        risk_mitigation_and_hedging=risk_items,
        whats_working=working_items,
        what_concerns_me=concerns_items,
        strategic_discussion_points=discussion_items,
        consistency_verified=True,
        generated_by="deterministic_fallback_engine",
    )


def invoke_portfolio_synthesizer(
    model: BaseChatModel,
    payload_json: str,
    quant_report: PortfolioAdvisoryQuantReport,
    max_retries: int = 1,
) -> PortfolioAdvisoryNarrativeDTO:
    """Invokes LLM Portfolio Synthesizer with Guardrail I6 post-validation and auto-fallback."""
    from validators.structured_output_retry import invoke_with_retry

    try:
        harness = get_harness("portfolio_synthesizer")
        system_content = harness.get_system_prompt()
        human_content = harness.get_skill_text(
            "HUMAN.md",
            advisory_payload_json=payload_json,
            feedback_prompt="",
        )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "human", "content": human_content},
        ]

        narrative, _ = invoke_with_retry(
            model=model,
            messages=messages,
            output_schema=PortfolioAdvisoryNarrativeDTO,
            observable_registry={},
            max_retries=max_retries,
            agent_name="portfolio_synthesizer",
        )

        # Guardrail I6: Validate consistency against deterministic quant report
        is_valid, validation_errors = validate_advisory_narrative_consistency(narrative, quant_report)
        if not is_valid:
            log.warning("Narrative consistency validation failed: %s. Triggering fallback.", validation_errors)
            # If invalid after retry, fall back to safe deterministic narrative
            return build_deterministic_fallback_narrative(
                quant_report, notice=f"ความขัดแย้งเชิงกลยุทธ์: {'; '.join(validation_errors[:2])}"
            )

        narrative.portfolio_state_hash = quant_report.portfolio_state_hash
        if not getattr(narrative, "synthesized_at", None):
            narrative.synthesized_at = datetime.now(timezone.utc).isoformat()
        narrative.consistency_verified = True
        return narrative

    except Exception as e:
        log.warning("invoke_portfolio_synthesizer failed (%s). Reverting to deterministic narrative.", e)
        return build_deterministic_fallback_narrative(quant_report, notice=f"LLM Unavailable ({str(e)})")
