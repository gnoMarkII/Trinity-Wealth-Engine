"""Structured output retry layer with explicit fallback placeholders."""
import logging
from typing import Any
from langchain_core.language_models.chat_models import BaseChatModel
from schemas.macro_schemas import MacroStrategyDirection, AssetAllocationView, AssetStance
from schemas.warning_registry import COVERAGE_WARNING_INCOMPLETE, SYSTEM_PLACEHOLDER, WarningMessage
from validators.quality_check import QualityCheckResult

logger = logging.getLogger(__name__)

CORE_BUCKETS = {"equities", "fixed_income", "commodities", "fx", "cash"}

BUCKET_DISPLAY_NAMES = {
    "equities": "Global Equities",
    "fixed_income": "Global Fixed Income / Duration",
    "commodities": "Commodities / Precious Metals",
    "fx": "Currencies / FX",
    "cash": "Cash / Short-term Liquidity",
}


def explicit_fallback_placeholders(
    direction: MacroStrategyDirection,
    observable_registry: dict[str, Any]
) -> MacroStrategyDirection:
    """Inject explicit system placeholders for missing core asset buckets, then revalidate."""
    if not hasattr(direction, "asset_allocation") or direction.asset_allocation is None:
        direction.asset_allocation = []

    present_buckets = {
        str(a.asset_bucket).lower().strip()
        for a in direction.asset_allocation
        if hasattr(a, "asset_bucket") and a.asset_bucket is not None
    }

    missing_buckets = CORE_BUCKETS - present_buckets
    if missing_buckets:
        logger.info(f"Explicit coverage fallback triggered for missing buckets: {missing_buckets}")
        for bucket in sorted(missing_buckets):
            display_name = BUCKET_DISPLAY_NAMES.get(bucket, f"Core Asset ({bucket.upper()})")
            placeholder_view = AssetAllocationView(
                asset_class=f"{display_name} [System Placeholder]",
                asset_bucket=bucket,  # type: ignore
                stance=AssetStance.NEUTRAL,
                rationale="[SYSTEM_PLACEHOLDER] ระบบสร้างมุมมองนี้โดยอัตโนมัติ เนื่องจาก LLM ไม่ได้วิเคราะห์สินทรัพย์กลุ่มนี้",
                confidence="low",
                supporting_data=[],
                validation_warnings=[
                    str(WarningMessage(SYSTEM_PLACEHOLDER)),
                ],
                why_not_high="ข้อมูลไม่เพียงพอเนื่องจากเป็น System Placeholder",
                allocation_delta="0% vs benchmark",
            )
            direction.asset_allocation.append(placeholder_view)

        if hasattr(direction, "validation_warnings") and direction.validation_warnings is not None:
            direction.validation_warnings = [
                w for w in direction.validation_warnings
                if "COVERAGE_WARNING_INCOMPLETE" not in str(w) and "Coverage Warning" not in str(w)
            ]

        # Crucial requirement (Finding 3): revalidate after injecting placeholders
        if hasattr(direction, "revalidate_with_registry"):
            direction = direction.revalidate_with_registry(observable_registry)

    return direction


def invoke_with_retry(
    model: BaseChatModel,
    messages: list[dict],
    output_schema: type[MacroStrategyDirection],
    observable_registry: dict[str, Any],
    max_retries: int = 1,
    agent_name: str = "allocator",
) -> tuple[MacroStrategyDirection, QualityCheckResult]:
    """Invoke LLM with structured output, inspect quality, and retry if retryable critical issues exist."""
    structured = model.with_structured_output(output_schema)
    current_messages = list(messages)
    
    direction = structured.invoke(current_messages)
    if hasattr(direction, "revalidate_with_registry"):
        direction = direction.revalidate_with_registry(observable_registry)
    
    quality = QualityCheckResult.from_direction(direction)
    retries_left = max_retries

    while quality.should_retry and retries_left > 0:
        logger.warning(f"Quality check triggered retry ({retries_left} retries left). Retryable IDs: {quality.retryable_ids} (All Criticals: {quality.critical_ids})")
        from core.prompt_harness import get_harness
        feedback_content = get_harness(agent_name).get_few_shots_feedback(quality.retry_feedback, strict=False)
        schema_name = getattr(output_schema, "__name__", "Structured Object")
        retry_msg = {
            "role": "human",
            "content": (
                f"[SYSTEM AUTOMATED FEEDBACK - RETRY REQUIRED]\n{feedback_content}\n\n"
                f"กรุณาสร้างและส่งคืนอ็อบเจกต์ {schema_name} ฉบับแก้ไขให้สมบูรณ์และถูกต้องตามข้อจำกัดข้างต้นทั้งหมด"
            )
        }
        current_messages.append(retry_msg)
        try:
            new_direction = structured.invoke(current_messages)
            if hasattr(new_direction, "revalidate_with_registry"):
                new_direction = new_direction.revalidate_with_registry(observable_registry)
            direction = new_direction
            quality = QualityCheckResult.from_direction(direction)
        except Exception as e:
            logger.error(f"Error during structured output retry invocation: {e}")
            break
        finally:
            retries_left -= 1

    # After all retries exhausted, check if explicit fallback placeholders are needed for coverage
    if COVERAGE_WARNING_INCOMPLETE in quality.critical_ids or COVERAGE_WARNING_INCOMPLETE in quality.soft_ids:
        direction = explicit_fallback_placeholders(direction, observable_registry)
        quality = QualityCheckResult.from_direction(direction)

    return direction, quality
