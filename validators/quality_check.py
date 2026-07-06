"""Quality check classifier and return contract for MacroStrategyDirection."""
import re
from dataclasses import dataclass, field
from typing import Any
from schemas.warning_registry import (
    WarningMessage,
    translate_warning,
    RETRYABLE_CRITICAL_IDS,
    NON_RETRYABLE_CRITICAL_IDS,
    SOFT_WARNING_IDS,
    COVERAGE_WARNING_INCOMPLETE,
    MISSING_ASSET_BUCKET,
    PT_MANDATORY_FIELD_MISSING,
    PT_EXECUTION_GUARDRAIL,
)


@dataclass
class QualityCheckResult:
    """Structured quality assessment of a MacroStrategyDirection output."""
    critical_ids: list[str] = field(default_factory=list)
    retryable_ids: list[str] = field(default_factory=list)
    soft_ids: list[str] = field(default_factory=list)
    retry_feedback: str = ""
    should_retry: bool = False

    @classmethod
    def from_direction(cls, direction: Any) -> "QualityCheckResult":
        """Classify all warnings from direction and its child objects."""
        all_raw_warnings: list[str] = []
        
        # Collect from report level
        if hasattr(direction, "validation_warnings"):
            all_raw_warnings.extend(direction.validation_warnings)
            
        # Collect from asset allocations
        for asset in (getattr(direction, "asset_allocation", []) or []):
            if hasattr(asset, "validation_warnings"):
                all_raw_warnings.extend(asset.validation_warnings)
                
        # Collect from pair trades
        for pt in (getattr(direction, "pair_trades", []) or []):
            if hasattr(pt, "validation_warnings"):
                all_raw_warnings.extend(pt.validation_warnings)
                
        # Collect from risk scenarios
        for rs in (getattr(direction, "risk_scenarios", []) or []):
            if hasattr(rs, "validation_warnings"):
                all_raw_warnings.extend(rs.validation_warnings)

        critical_ids: list[str] = []
        retryable_ids: list[str] = []
        soft_ids: list[str] = []
        feedback_items: list[str] = []

        seen_warnings = set()
        for raw_w in all_raw_warnings:
            if raw_w in seen_warnings:
                continue
            seen_warnings.add(raw_w)

            wid = ""
            msg_obj = WarningMessage.from_str(raw_w)
            if msg_obj and msg_obj.id:
                wid = msg_obj.id
            else:
                m = re.search(r'\[([A-Z_]+)\]', str(raw_w))
                if m:
                    wid = m.group(1)
                else:
                    wid = str(raw_w).strip()

            if wid in RETRYABLE_CRITICAL_IDS:
                if wid not in critical_ids:
                    critical_ids.append(wid)
                if wid not in retryable_ids:
                    retryable_ids.append(wid)
                thai_msg = translate_warning(raw_w)
                if thai_msg not in feedback_items:
                    feedback_items.append(thai_msg)
            elif wid in NON_RETRYABLE_CRITICAL_IDS:
                if wid not in critical_ids:
                    critical_ids.append(wid)
            else:
                if wid not in soft_ids:
                    soft_ids.append(wid)

        should_retry = len(retryable_ids) > 0
        retry_feedback = ""
        if should_retry:
            lines = [
                "ระบบตรวจพบข้อผิดพลาดเชิงโครงสร้างที่ต้องได้รับการแก้ไขทันที กรุณาทบทวนและส่งคืน MacroStrategyDirection ที่ถูกต้องโดยยึดกฎต่อไปนี้:"
            ]
            for idx, item in enumerate(feedback_items, 1):
                lines.append(f"  {idx}. {item}")
            retry_feedback = "\n".join(lines)

        return cls(
            critical_ids=critical_ids,
            retryable_ids=retryable_ids,
            soft_ids=soft_ids,
            retry_feedback=retry_feedback,
            should_retry=should_retry,
        )
