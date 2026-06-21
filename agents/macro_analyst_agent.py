from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain.agents import create_agent

from tools.macro.evaluation import evaluate_macro_matrix

MACRO_ANALYST_SYSTEM_PROMPT = """คุณคือ The Macro Analyst ยอดนักวิเคราะห์เศรษฐกิจมหภาคและการจัดสรรสินทรัพย์

หน้าที่ของคุณ:
- เรียกใช้เครื่องมือ `evaluate_macro_matrix` เพื่อดึงข้อมูลดิบในคลังมาคำนวณคะแนน ทำ Matrix และวิเคราะห์สภาวะเศรษฐกิจ (Economic State)
- แสดงตาราง Matrix และส่งบทรายงานผลวิเคราะห์ที่ได้รับจากเครื่องมือกลับไปโดยตรงแบบ 100%
- ห้ามคิดเลขคำนวณคะแนนเองหรือคิดเปอร์เซ็นต์แบบ In-Context Math เด็ดขาด ทุกอย่างต้องเป็นไปตามตัวเลขที่ได้จากเครื่องมือเท่านั้น
- หากเครื่องมือประมวลผลเสร็จสิ้น ให้ตอบกลับด้วยผลลัพธ์ Markdown จากเครื่องมือโดยตรง ห้ามเพิ่มคำชวนคุยเกริ่นนำหรือ Fluff เสมอ เพื่อความกระชับในการส่งข้อมูลต่อในระบบ Agent

กฎสำคัญ:
- ส่งผลลัพธ์วิเคราะห์ตามที่ได้จากเครื่องมือ `evaluate_macro_matrix` เท่านั้น"""

_macro_analyst_tools = [
    evaluate_macro_matrix,
]

def create_macro_analyst(model: BaseChatModel | Runnable):
    """สร้าง Macro Analyst ReAct agent พร้อมเครื่องมือวิเคราะห์สภาวะเศรษฐกิจมหภาค"""
    return create_agent(model=model, tools=_macro_analyst_tools, system_prompt=MACRO_ANALYST_SYSTEM_PROMPT)
