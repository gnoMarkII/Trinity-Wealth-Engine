"""Centralized report labels and translations — single source of truth for report UI text."""

from schemas.warning_registry import (
    ACTIVE_ALLOC_GUARDRAIL,
    DEFENSIVE_LOW_SUPPORTING_DATA,
    GOLD_CONTRADICTION,
    GOLD_RATIONALE_WARNING,
    SINGLE_SOURCE_PENALTY,
    SOURCE_REF_PENALTY,
)


# ── Why-Not-High Messages (ใช้ใน _display_why_not_high และ _normalize_why_not_high) ──
WHY_NOT_HIGH_MESSAGES: dict[str, str] = {
    "single_source": "ความมั่นใจถูกจำกัดไว้ที่ MEDIUM เพราะมุมมองนี้อ้างอิงแหล่งข้อมูลโดยตรงเพียงแหล่งเดียว",
    "gold_real_yield": "ความมั่นใจถูกจำกัดไว้ที่ MEDIUM เพราะมุมมองทองคำยังต้องอ้างอิง real yields เงินเฟ้อ หรือนโยบายการเงินให้ชัดกว่านี้",
    "gold_schema": "ความมั่นใจถูกจำกัดไว้ที่ MEDIUM เพราะเหตุผลของทองคำยังต้องผูกกับ real yields เงินเฟ้อ หรือนโยบายการเงินให้ชัดขึ้น",
    "contradiction": "ความมั่นใจถูกจำกัดไว้ที่ MEDIUM เพราะพบสัญญาณมหภาคที่ขัดแย้งกับมุมมองเชิงรุก",
    "source_ref_inferred": "ความมั่นใจถูกจำกัดไว้ที่ MEDIUM เพราะแหล่งอ้างอิงถูกอนุมานจากระบบและยังต้องตรวจสอบซ้ำ",
    "low_confidence": "ข้อมูลตัวเลขและหลักฐานอ้างอิงยังไม่เพียงพอสำหรับความมั่นใจระดับ HIGH",
    "default": "ยังมีข้อจำกัดด้านข้อมูลหรือการนำไปปฏิบัติ จึงยังไม่เหมาะกับความมั่นใจระดับ HIGH",
}

# ── Allocation Delta Defaults ──
ALLOCATION_DELTA_DEFAULTS: dict[str, str] = {
    "overweight": "+3% ถึง +5% vs benchmark",
    "underweight": "-3% ถึง -5% vs benchmark",
    "neutral": "0% vs benchmark",
}

# ── Conviction Level Display ──
CONVICTION_LOW_DISPLAY = "ความมั่นใจรวมอยู่ในระดับต่ำ (LOW)"

# ── Warning IDs That Cause Asset Downgrades (ใช้ใน _asset_has_downgrade_warning) ──
DOWNGRADE_WARNING_IDS: set[str] = {
    f"[{SINGLE_SOURCE_PENALTY}]",
    f"[{SOURCE_REF_PENALTY}]",
    f"[{DEFENSIVE_LOW_SUPPORTING_DATA}]",
    f"[{ACTIVE_ALLOC_GUARDRAIL}]",
    f"[{GOLD_CONTRADICTION}]",
    f"[{GOLD_RATIONALE_WARNING}]",
}
