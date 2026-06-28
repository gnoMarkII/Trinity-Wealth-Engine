"""Self-correction error messages for LLM Agent consumption.

Convention: ทุก error string ที่ return จาก @tool function ต้องขึ้นต้นด้วย "Error: "
เพื่อให้ Agent แยกแยะ success vs. error ได้ชัดเจน

Architecture:
- Internal functions (_*_locked, _require_fx, etc.) → คง raise ValueError ไว้
- @tool functions → ดักจับด้วย try/except แล้ว return error string
"""

# ── Cross-tool guidance messages ──
CASH_VIA_MANAGE = (
    "Error: ห้ามจัดการ cash sentinel ({symbols}) ผ่านเครื่องมือนี้ "
    "— ใช้ 'manage_cash_flow' แทน"
)
GOAL_VIA_BOOKKEEPER = (
    "Error: ห้ามบันทึก goal/financial_goal ด้วย write_raw_markdown "
    "— ให้ใช้ 'set_goal' ของ Bookkeeper แทน"
)
ENTITY_VIA_SAVE_MEMORY = (
    "Error: write_raw_markdown สำหรับ raw markdown เท่านั้น "
    "— ใช้ 'save_memory' เพื่อสร้าง/อัปเดต Entity"
)
LOCK_TIMEOUT = (
    "Error: ระบบกำลังประมวลผลคำสั่งอื่นอยู่ กรุณาลองใหม่อีกครั้ง ({detail})"
)

def validation_error(msg: str) -> str:
    """Wrap validation message ด้วย Error prefix มาตรฐาน"""
    return f"Error: {msg}"
