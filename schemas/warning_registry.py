import json
import re
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass(frozen=True)
class WarningMessage:
    """Structured warning message with ID and optional JSON payload for translation lookup."""
    id: str
    params: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """Serialize to string format: [ID] {"key": "value"}"""
        if self.params:
            payload = json.dumps(self.params, ensure_ascii=False)
            return f"[{self.id}] {payload}"
        return f"[{self.id}]"

    @classmethod
    def from_str(cls, text: str) -> Optional['WarningMessage']:
        """Parse from serialized string format [ID] or [ID] {"key": "val"}."""
        if not text:
            return None
        m = re.match(r'^\[([A-Z_]+)\]\s*(.*)?$', text.strip())
        if not m:
            return None
        wid = m.group(1)
        raw_payload = (m.group(2) or "").strip()
        params = {}
        if raw_payload:
            try:
                params = json.loads(raw_payload)
            except (json.JSONDecodeError, TypeError):
                # Fallback for key=val or plain text
                if "=" in raw_payload:
                    for pair in raw_payload.split("|"):
                        if "=" in pair:
                            k, v = pair.split("=", 1)
                            params[k.strip()] = v.strip()
                else:
                    params["detail"] = raw_payload
        return cls(id=wid, params=params)


# ── Warning ID Constants ──────────────────────────────────────────
# Asset-level
DEFENSIVE_LOW_SUPPORTING_DATA = "DEFENSIVE_LOW_SUPPORTING_DATA"
SINGLE_SOURCE_PENALTY = "SINGLE_SOURCE_PENALTY"
SOURCE_REF_PENALTY = "SOURCE_REF_PENALTY"
SOURCE_REF_BACKFILL = "SOURCE_REF_BACKFILL"
SOURCE_REF_WARNING_EMPTY = "SOURCE_REF_WARNING_EMPTY"
SOURCE_REF_WARNING_OMITTED = "SOURCE_REF_WARNING_OMITTED"
ACTIVE_ALLOC_GUARDRAIL = "ACTIVE_ALLOC_GUARDRAIL"
COVERAGE_BACKFILL_VIEW = "COVERAGE_BACKFILL_VIEW"

# Pair Trade
PT_MANDATORY_FIELD_MISSING = "PT_MANDATORY_FIELD_MISSING"
PT_DEFENSIVE_LOW = "PT_DEFENSIVE_LOW"
PT_EXECUTION_GUARDRAIL = "PT_EXECUTION_GUARDRAIL"
PT_RISK_BUDGET_HIGH_TO_MED = "PT_RISK_BUDGET_HIGH_TO_MED"
PT_RISK_BUDGET_HIGH_TO_SMALL = "PT_RISK_BUDGET_HIGH_TO_SMALL"
PT_RISK_BUDGET_MED_TO_SMALL = "PT_RISK_BUDGET_MED_TO_SMALL"
PT_GRACEFUL_DOWNGRADE = "PT_GRACEFUL_DOWNGRADE"

# Risk Scenario
RS_DEFENSIVE_LOW = "RS_DEFENSIVE_LOW"

# Portfolio-level
PORTFOLIO_DEFENSIVE_LOW = "PORTFOLIO_DEFENSIVE_LOW"
STALE_DATA_DEGRADATION = "STALE_DATA_DEGRADATION"
GRACEFUL_DROP_PAIR_TRADES = "GRACEFUL_DROP_PAIR_TRADES"
GRACEFUL_DROP_RISK_SCENARIOS = "GRACEFUL_DROP_RISK_SCENARIOS"
COVERAGE_BACKFILL_EXPANDED = "COVERAGE_BACKFILL_EXPANDED"
COVERAGE_WARNING_INCOMPLETE = "COVERAGE_WARNING_INCOMPLETE"
GOLD_RATIONALE_WARNING = "GOLD_RATIONALE_WARNING"
GOLD_CONTRADICTION = "GOLD_CONTRADICTION"
US_EQUITY_CONTRADICTION = "US_EQUITY_CONTRADICTION"
REGIME_CONTRADICTION = "REGIME_CONTRADICTION"
REGIME_PROB_COVERAGE = "REGIME_PROB_COVERAGE"
REGIME_PROB_BACKFILL = "REGIME_PROB_BACKFILL"
REGIME_PROB_SUM = "REGIME_PROB_SUM"
REGIME_CONSISTENCY_ADJ = "REGIME_CONSISTENCY_ADJ"
CONVICTION_CONTRADICTION = "CONVICTION_CONTRADICTION"
IMPLEMENTATION_CAP = "IMPLEMENTATION_CAP"
PORTFOLIO_CONVICTION_CAP = "PORTFOLIO_CONVICTION_CAP"

# New IDs for LLM-Agnostic Refactoring
MISSING_ASSET_BUCKET = "MISSING_ASSET_BUCKET"
STATISTICAL_OVERCLAIM = "STATISTICAL_OVERCLAIM"
ALLOCATION_DELTA_INVALID = "ALLOCATION_DELTA_INVALID"
FX_STANCE_MISMATCH = "FX_STANCE_MISMATCH"
SYSTEM_PLACEHOLDER = "SYSTEM_PLACEHOLDER"

# Lean MVP Upgrade Warnings
VALUATION_RICH_WARNING = "VALUATION_RICH_WARNING"
CREDIT_SPREAD_WARNING = "CREDIT_SPREAD_WARNING"
CORRELATION_BREAKDOWN_WARNING = "CORRELATION_BREAKDOWN_WARNING"

# ── Warning Severity Sets for Retry Layer ─────────────────────────
# 🟢 RETRYABLE CRITICAL — ข้อผิดพลาดเชิงโครงสร้างที่ LLM ลืมหรือใส่ไม่ครบ สามารถสั่งให้แก้ไขใหม่ได้
RETRYABLE_CRITICAL_IDS: set[str] = {
    COVERAGE_WARNING_INCOMPLETE,
    MISSING_ASSET_BUCKET,
    PT_MANDATORY_FIELD_MISSING,
    PT_EXECUTION_GUARDRAIL,
    GRACEFUL_DROP_PAIR_TRADES,
}

# 🟡 NON-RETRYABLE CRITICAL — ความขัดแย้งเชิงตรรกะที่เกิดจากสภาวะตลาดจริง หรือ Guardrail ตั้งใจ Downgrade
NON_RETRYABLE_CRITICAL_IDS: set[str] = {
    DEFENSIVE_LOW_SUPPORTING_DATA,
    PT_DEFENSIVE_LOW,
    RS_DEFENSIVE_LOW,
    GOLD_CONTRADICTION,
    US_EQUITY_CONTRADICTION,
    REGIME_CONTRADICTION,
    CONVICTION_CONTRADICTION,
}

# ⚪ SOFT WARNINGS — ข้อจำกัดคุณภาพข้อมูลทั่วไป (Downgrade confidence only)
SOFT_WARNING_IDS: set[str] = {
    SINGLE_SOURCE_PENALTY,
    SOURCE_REF_PENALTY,
    STALE_DATA_DEGRADATION,
    PT_GRACEFUL_DOWNGRADE,
    PT_RISK_BUDGET_HIGH_TO_MED,
    PT_RISK_BUDGET_HIGH_TO_SMALL,
    PT_RISK_BUDGET_MED_TO_SMALL,
    STATISTICAL_OVERCLAIM,
    ALLOCATION_DELTA_INVALID,
    FX_STANCE_MISMATCH,
    SYSTEM_PLACEHOLDER,
}


# ── Thai Translation Templates ────────────────────────────────────
class _SafeDict(dict):
    def __missing__(self, key):
        return f"{{{key}}}"


THAI_TEMPLATES: Dict[str, str] = {
    # Asset-level
    DEFENSIVE_LOW_SUPPORTING_DATA:
        "ลดระดับความมั่นใจเป็น LOW เพราะไม่มีตัวเลข hard data ใน supporting_data",
    SINGLE_SOURCE_PENALTY:
        "ลดความมั่นใจเป็น MEDIUM เพราะมุมมองนี้อ้างอิงแหล่งข้อมูลโดยตรงเพียงแหล่งเดียว",
    SOURCE_REF_PENALTY:
        "ลดระดับความมั่นใจจาก HIGH เป็น MEDIUM เพราะ source_refs ถูกอนุมานจาก source_files ระดับรายงาน",
    SOURCE_REF_BACKFILL:
        "เติมแหล่งอ้างอิงจาก source_files ของรายงาน เพราะมุมมองสินทรัพย์ไม่ได้ระบุ source_refs โดยตรง",
    SOURCE_REF_WARNING_EMPTY:
        "source_refs ว่าง จึงไม่ควรถือว่าความมั่นใจด้านข้อมูลเป็นระดับสูง",
    SOURCE_REF_WARNING_OMITTED:
        "มุมมองสินทรัพย์ไม่ได้ระบุ source_refs",
    ACTIVE_ALLOC_GUARDRAIL:
        "ปรับมุมมองเป็น Neutral เพราะความมั่นใจเป็น LOW หรือขาดตัวเลข hard data",
    COVERAGE_BACKFILL_VIEW:
        "เติมมุมมอง Neutral ระดับความมั่นใจต่ำ เพราะ Strategic Allocator ไม่ได้วิเคราะห์สินทรัพย์หลักกลุ่มนี้",

    # Pair Trade
    PT_MANDATORY_FIELD_MISSING:
        "ข้อมูลสำคัญขาดหาย: ฟิลด์ '{field}' ว่างเปล่า",
    PT_DEFENSIVE_LOW:
        "ลดระดับความมั่นใจเป็น LOW เพราะข้อมูลตัวเลข hard data ยังไม่ครบถ้วน",
    PT_EXECUTION_GUARDRAIL:
        "ลดความมั่นใจเป็น LOW เพราะรายละเอียดการปฏิบัติการของ Pair Trade ยังไม่ครบถ้วน",
    PT_RISK_BUDGET_HIGH_TO_MED:
        "ปรับขนาดความเสี่ยงจากระดับสูงลงเป็น medium_risk_budget เพราะความมั่นใจอยู่ที่ MEDIUM",
    PT_RISK_BUDGET_HIGH_TO_SMALL:
        "ปรับขนาดความเสี่ยงจากระดับสูงลงเป็น small_risk_budget เพราะความมั่นใจอยู่ที่ LOW",
    PT_RISK_BUDGET_MED_TO_SMALL:
        "ปรับขนาดความเสี่ยงจาก medium_risk_budget ลงเป็น small_risk_budget เพราะความมั่นใจอยู่ที่ LOW",
    PT_GRACEFUL_DOWNGRADE:
        "ลดความมั่นใจของ Pair Trade เป็น MEDIUM เนื่องจากใช้หลักฐานเชิงสัมพัทธ์จากจุดเดียวแทนการคำนวณ beta/correlation จากอนุกรมเวลาย้อนหลัง",

    # Risk Scenario
    RS_DEFENSIVE_LOW:
        "ลดระดับความมั่นใจเป็น LOW เพราะตัวชี้วัดล่วงหน้า เครื่องมือ hedge หรือ hard data เชิงตัวเลขยังไม่ครบ",

    # Portfolio-level
    PORTFOLIO_DEFENSIVE_LOW:
        "ความมั่นใจรวมอยู่ในระดับต่ำ: ลด conviction รวมเป็น LOW เพราะมุมมองสินทรัพย์ {ratio} ขาดตัวเลข hard data หรือมีความมั่นใจ LOW",
    STALE_DATA_DEGRADATION:
        "ลดระดับ conviction รวมจาก HIGH เป็น MEDIUM เพราะมีคำเตือนเรื่องข้อมูลล่าช้า",
    GRACEFUL_DROP_PAIR_TRADES:
        "ตัด Pair Trade ออกอย่างปลอดภัยจำนวน {count} รายการ เนื่องจากหลักฐานตัวเลขราคาตลาดหรือรายละเอียดการปฏิบัติการยังไม่ครบถ้วน",
    GRACEFUL_DROP_RISK_SCENARIOS:
        "ตัดแผนป้องกันความเสี่ยงออกอย่างปลอดภัยจำนวน {count} รายการ เนื่องจากตัวเลข Hard Data ไม่เพียงพอ",
    COVERAGE_BACKFILL_EXPANDED:
        "เติมสินทรัพย์หลักให้ครบจาก {old} เป็น {new} กลุ่ม โดยใช้มุมมอง Neutral ความมั่นใจต่ำในส่วนที่ข้อมูลไม่พอ",
    COVERAGE_WARNING_INCOMPLETE:
        "ความครอบคลุมสินทรัพย์ยังไม่ครบกรอบหลัก 5 กลุ่ม จึงเติมรายการที่ขาดเป็นมุมมอง Neutral ตามข้อจำกัดของข้อมูล",
    GOLD_RATIONALE_WARNING:
        "ลดความมั่นใจของมุมมอง Overweight ทองคำเป็น MEDIUM เพราะเหตุผลยังพึ่งพาความเสี่ยงภูมิรัฐศาสตร์มากเกินไป และยังไม่ผูกกับ real yields เงินเฟ้อ หรือนโยบายการเงินอย่างชัดเจน",
    GOLD_CONTRADICTION:
        "ลดความมั่นใจของทองคำเพราะเหตุผล Overweight ยังไม่มีหลักยึดจาก yield เงินเฟ้อ หรือกรอบนโยบายการเงินเพียงพอ",
    US_EQUITY_CONTRADICTION:
        "พบสัญญาณขัดแย้งใน US Equities: Overweight ขัดกับ rising yields, weak housing, low sentiment",
    REGIME_CONTRADICTION:
        "สภาวะ Reflation/Goldilocks ขัดแย้งกับสัญญาณเงินเฟ้อสูง (CPI > 3%)",
    REGIME_PROB_COVERAGE:
        "การกระจายความน่าจะเป็นของ regime ยังไม่ครบอย่างน้อย 4 สภาวะ จึงยังไม่พอสำหรับการทบทวนระดับสถาบัน",
    REGIME_PROB_BACKFILL:
        "เติม scenario สำรองให้ regime probabilities ครบอย่างน้อย 4 สภาวะตามเกณฑ์ institutional",
    REGIME_PROB_SUM:
        "คำเตือนผลรวมความน่าจะเป็น regime: {detail}",
    REGIME_CONSISTENCY_ADJ:
        "ปรับ regime ให้สอดคล้อง: {detail}",
    CONVICTION_CONTRADICTION:
        "พบความขัดแย้ง: พอร์ตแนะนำ Overweight ทั้งหุ้นเติบโตและพันธบัตร/Duration โดยยังไม่อธิบายให้ชัดว่าเป็น Barbell strategy หรือ Duration hedge",
    IMPLEMENTATION_CAP:
        "ลดระดับความมั่นใจด้านการนำไปปฏิบัติจาก HIGH เป็น MEDIUM เพราะ source refs หรือ conviction รวมยังไม่แข็งพอ",
    PORTFOLIO_CONVICTION_CAP:
        "ลดระดับความมั่นใจของสินทรัพย์จาก HIGH เป็น MEDIUM เพราะ conviction รวมของพอร์ตเป็น LOW",

    # New Templates for LLM-Agnostic Refactoring
    MISSING_ASSET_BUCKET:
        "สินทรัพย์ไม่ได้ระบุหมวดหมู่ (asset_bucket) กรุณาระบุให้ชัดเจน",
    STATISTICAL_OVERCLAIM:
        "ลดความมั่นใจเป็น MEDIUM เนื่องจากใช้คำศัพท์สถิติเชิงความผันผวนโดยไม่มีข้อมูลสถิติรองรับในตาราง",
    ALLOCATION_DELTA_INVALID:
        "การปรับน้ำหนัก (allocation_delta) ไม่ได้ระบุเป็นตัวเลขเปรียบเทียบกับเกณฑ์อ้างอิง",
    FX_STANCE_MISMATCH:
        "มุมมองค่าเงินขัดแย้งกับสถานะการจัดสรร (เช่น ดอลลาร์แข็ง/บาทอ่อน ควรเป็น Overweight USD/THB)",
    SYSTEM_PLACEHOLDER:
        "[SYSTEM_PLACEHOLDER] ระบบสร้างมุมมองนี้โดยอัตโนมัติ เนื่องจาก LLM ไม่ได้วิเคราะห์สินทรัพย์กลุ่มนี้",
    VALUATION_RICH_WARNING:
        "ลดระดับความมั่นใจของ US Equities จาก HIGH เป็น MEDIUM เนื่องจาก Equity Risk Premium (ERP) ต่ำกว่าเกณฑ์ 1.5% โดยไม่มีอรรถาธิบายการป้องกันความเสี่ยงหรืออ้างอิง Earnings Revision ที่ชัดเจน",
    CREDIT_SPREAD_WARNING:
        "แจ้งเตือนระดับความเสี่ยงด้านสินเชื่อ: High Yield Bond Spread กว้างกว่าเกณฑ์ 5.0% หรือปรับตัวกว้างขึ้นอย่างรวดเร็วในช่วง 3 เดือนที่ผ่านมา",
    CORRELATION_BREAKDOWN_WARNING:
        "แจ้งเตือนความสัมพันธ์ระหว่างหุ้นกับพันธบัตร: Stock-Bond Correlation ในรอบ 60 วันมีค่าเป็นบวก (> 0.30) ซึ่งอาจทำให้พันธบัตรไม่สามารถป้องกันความเสี่ยง (Hedge) ให้กับตลาดหุ้นได้อย่างมีประสิทธิภาพ ควรพิจารณาใช้ Cash, Gold หรือ Options แทน",
}


def translate_warning(warning: Any) -> str:
    """Translate warning (WarningMessage object or serialized string [ID]) to Thai."""
    if isinstance(warning, WarningMessage):
        template = THAI_TEMPLATES.get(warning.id)
        if template:
            return template.format_map(_SafeDict(warning.params))
        return str(warning)

    text = str(warning or "").strip()
    parsed = WarningMessage.from_str(text)
    if parsed and parsed.id in THAI_TEMPLATES:
        return THAI_TEMPLATES[parsed.id].format_map(_SafeDict(parsed.params))

    return text
