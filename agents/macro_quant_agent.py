from langchain_core.language_models.chat_models import BaseChatModel
from langchain.agents import create_agent
from tools.macro.evaluation import evaluate_macro_matrix

MACRO_QUANT_SYSTEM_PROMPT = """คุณคือ Macro Quant — ผู้ประเมินสภาวะเศรษฐกิจผ่านตัวเลข สถิติ และความน่าจะเป็น

หน้าที่:
- เรียก evaluate_macro_matrix เพื่อคำนวณ QuantScore จากข้อมูลดิบใน Vault
- ส่ง output ดิบที่ได้จากเครื่องมือกลับมาโดยตรง 100%
- ห้ามคิดเลขเอง ห้ามวิเคราะห์เชิงคุณภาพ ห้ามเพิ่มคำชวนคุย

กฎสำคัญ:
- ส่งผลลัพธ์ตามที่ได้จาก evaluate_macro_matrix เท่านั้น"""

_macro_quant_tools = [evaluate_macro_matrix]

def create_macro_quant(model: BaseChatModel):
    return create_agent(
        model=model,
        tools=_macro_quant_tools,
        system_prompt=MACRO_QUANT_SYSTEM_PROMPT
    )
