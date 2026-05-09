from langchain_core.language_models import BaseChatModel
from langgraph.prebuilt import create_react_agent

from core.llm_factory import get_llm
from tools.macro_tools import ingest_daily_macro, ingest_regional_pulse

RESEARCHER_SYSTEM_PROMPT = """คุณคือ The Researcher หน่วยดึงข้อมูลจากภายนอก

หน้าที่ของคุณ: เรียก tool แล้วส่ง output ดิบกลับมาเท่านั้น ห้ามสรุป ห้ามวิเคราะห์ ห้ามเพิ่มความคิดเห็น

กฎการเลือก Tool:
- ข้อมูลมหภาค (Yield, VIX, DXY, ทองคำ, น้ำมัน) → เรียก ingest_daily_macro
- ภาพรวมรายภูมิภาค (จีน/ยุโรป/EM/ญี่ปุ่น/เอเชียแปซิฟิก) → เรียก ingest_regional_pulse
- หากดึงข้อมูลไม่ได้ → รายงาน error สั้นๆ

กฎสำคัญ:
- ส่งเฉพาะ output ดิบที่ได้จาก tool call เท่านั้น — ไม่ต้องแต่งประโยคเพิ่ม
- ห้ามบันทึกหรือเขียนไฟล์ใดๆ"""

_researcher_tools = [ingest_daily_macro, ingest_regional_pulse]


def create_researcher(model: BaseChatModel | None = None):
    """สร้าง Researcher ReAct agent พร้อม External Data tools"""
    if model is None:
        model = get_llm(provider="google", model_name="gemini-3-flash-preview", use_fallback=True)
    return create_react_agent(model, _researcher_tools, prompt=RESEARCHER_SYSTEM_PROMPT)
