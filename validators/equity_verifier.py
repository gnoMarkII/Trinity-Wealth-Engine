"""Equity Narrative Verifier & Consistency Checker.

Architecture Layer: validators/
Dependencies: Standard Library + Pydantic. Complies with AST dependency rules.

Validates LLM-generated narrative against deterministic quant data:
1. Minimum content check (analysis and summary non-empty and sufficient).
2. Flag consistency: If DCF is non-actionable or unavailable in quant_signals,
   narrative must not claim a definitive DCF target price as valid.
3. Stance consistency: If scorecard action stance is WAIT_DO_NOT_ENTER or TRIM_OVERWEIGHT,
   narrative must not give an unhedged strong buy recommendation.
"""
from dataclasses import dataclass, field
import json
import logging
import re
from typing import Any, Dict, List, Optional

from schemas.micro_quant_schemas import EquityNarrativeOutput

logger = logging.getLogger(__name__)


@dataclass
class EquityVerificationResult:
    """Assessment of EquityNarrativeOutput quality and consistency."""
    issues: List[str] = field(default_factory=list)
    should_retry: bool = False
    retry_feedback: str = ""

    @property
    def critical_ids(self) -> List[str]:
        return self.issues

    @property
    def retryable_ids(self) -> List[str]:
        return self.issues

    @property
    def soft_ids(self) -> List[str]:
        return []

    @classmethod
    def from_narrative(
        cls,
        narrative: EquityNarrativeOutput,
        quant_data: Optional[Dict[str, Any]] = None,
    ) -> "EquityVerificationResult":
        issues: List[str] = []
        if not quant_data:
            quant_data = {}

        quant_signals = quant_data.get("quant_signals", {}) or {}
        scorecard = quant_data.get("deterministic_scorecard", {}) or {}
        data_quality_flags = set(quant_signals.get("data_quality_flags", []) or [])

        narrative_text = (getattr(narrative, "narrative_analysis", "") or "").strip()
        summary_text = (getattr(narrative, "base_case_summary", "") or "").strip()
        combined_text = f"{narrative_text}\n{summary_text}"

        # 1. Minimum content check
        if not narrative_text or len(narrative_text) < 20:
            issues.append("เนื้อหาการวิเคราะห์ (narrative_analysis) สั้นเกินไปหรือว่างเปล่า กรุณาวิเคราะห์ปัจจัยพื้นฐานให้ครอบคลุม")
        if not summary_text or len(summary_text) < 10:
            issues.append("บทสรุปกรณีฐาน (base_case_summary) สั้นเกินไปหรือว่างเปล่า")

        # 2. Flag consistency: DCF Unavailability
        dcf_unavailable_flags = {
            "negative_fcf_dcf_unavailable:dcf",
            "beta_unavailable_dcf_unavailable:dcf",
            "non_actionable_dcf:valuation",
        }
        has_dcf_gap = bool(data_quality_flags.intersection(dcf_unavailable_flags))
        if has_dcf_gap:
            # Check if narrative contradicts by asserting a definitive DCF target price
            if re.search(r'(?i)(เป้าหมายราคาตาม\s*dcf|dcf\s*target\s*price|ประเมินมูลค่าด้วย\s*dcf\s*ได้|มูลค่าพื้นฐานจาก\s*dcf)', combined_text):
                issues.append(
                    "พบความขัดแย้งเชิงตรรกะ: ข้อมูลงบการเงินไม่สามารถคำนวณ DCF ได้ (FCF ติดลบ หรือขาดค่า Beta) "
                    "แต่บทวิเคราะห์กลับระบุการประเมินมูลค่าด้วย DCF กรุณาอ้างอิงวิธีประเมินมูลค่าอื่น เช่น P/E, EV/EBITDA หรือระบุข้อจำกัดของ DCF ให้ชัดเจน"
                )

        # 3. Action stance consistency
        action_stance = scorecard.get("action_stance") or quant_data.get("position_action_hint")
        if action_stance in ("WAIT_DO_NOT_ENTER", "TRIM_OVERWEIGHT"):
            if re.search(r'(?i)(แนะนำซื้ออย่างยิ่ง|strong buy|เข้าสะสมเต็มที่|all-in|ซื้อเต็มพอร์ต)', combined_text):
                issues.append(
                    f"พบความขัดแย้งกับ Scorecard: ท่าทีการลงทุนจากโมเดลเชิงปริมาณระบุเป็น '{action_stance}' "
                    "แต่บทวิเคราะห์สรุปเชิญชวนเข้าซื้อเชิงรุก กรุณาปรับเนื้อหาให้สอดคล้องกับความเสี่ยงและกรอบการปฏิบัติการ"
                )

        should_retry = len(issues) > 0
        retry_feedback = ""
        if should_retry:
            lines = [
                "[EQUITY VERIFICATION PROBE - RETRY REQUIRED]",
                "ระบบตรวจพบข้อบกพร่องในบทวิเคราะห์หุ้น กรุณาแก้ไขเฉพาะจุดต่อไปนี้ก่อนส่งคืน EquityNarrativeOutput:",
            ]
            for idx, issue in enumerate(issues, 1):
                lines.append(f"  {idx}. {issue}")
            retry_feedback = "\n".join(lines)

        return cls(
            issues=issues,
            should_retry=should_retry,
            retry_feedback=retry_feedback,
        )


def invoke_equity_with_verifier(
    model: Any,
    messages: list[dict],
    quant_data: Optional[Dict[str, Any]] = None,
    max_retries: int = 1,
    agent_name: str = "equity_synthesizer",
) -> tuple[EquityNarrativeOutput, EquityVerificationResult]:
    """Invoke LLM for EquityNarrativeOutput with verification probe and best-of retry."""
    structured = model.with_structured_output(EquityNarrativeOutput)
    current_messages = list(messages)

    narrative = structured.invoke(current_messages)
    quality = EquityVerificationResult.from_narrative(narrative, quant_data)
    initial_issues = list(quality.issues)
    retries_left = max_retries

    while quality.should_retry and retries_left > 0:
        logger.warning(
            f"Equity verification probe triggered retry ({retries_left} retries left). Issues: {quality.issues}"
        )
        retry_msg = {
            "role": "human",
            "content": f"{quality.retry_feedback}\n\nกรุณาสร้างและส่งคืนอ็อบเจกต์ EquityNarrativeOutput ฉบับแก้ไขให้ถูกต้องตามข้อจำกัดข้างต้น",
        }
        current_messages.append(retry_msg)
        try:
            new_narrative = structured.invoke(current_messages)
            new_quality = EquityVerificationResult.from_narrative(new_narrative, quant_data)

            # Best-of selection: pick candidate with fewer issues
            if len(new_quality.issues) < len(quality.issues):
                narrative = new_narrative
                quality = new_quality
            else:
                logger.info("Equity best-of selection: retry result had equal or more issues, keeping previous candidate.")
                if not quality.should_retry:
                    break
        except Exception as e:
            logger.error(f"Error during equity narrative retry invocation: {e}")
            break
        finally:
            retries_left -= 1

    try:
        from tools.telemetry.failure_ledger import record_probe_event
        if initial_issues:
            event_type = "probe_resolved" if not quality.should_retry else "probe_failed"
            resolved = not quality.should_retry
            issues = initial_issues
        else:
            event_type = "probe_clean_first_pass"
            resolved = True
            issues = []

        record_probe_event(
            domain="equity",
            event_type=event_type,
            issues=issues,
            resolved=resolved,
            retries_count=max_retries - retries_left,
            context_meta={"agent": agent_name, "ticker": quant_data.get("ticker") if quant_data else None},
        )
    except Exception as exc:
        logger.debug(f"Telemetry recording skipped: {exc}")

    return narrative, quality
