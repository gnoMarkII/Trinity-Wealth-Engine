from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain.agents import create_agent

from tools.macro.ingest import (
    ingest_global_macro,
    ingest_regional_macro,
    ingest_country_macro,
    ingest_us_sectors,
)
from tools.market.fundamentals import ingest_stock_fundamentals, ingest_financial_health
from tools.market.technical import ingest_stock_momentum
from tools.market.news import ingest_stock_news
from tools.market.consensus import ingest_stock_consensus
from tools.market.financials import ingest_financial_trends
from tools.knowledge.article import ingest_article_url
from tools.knowledge.document import ingest_pdf
from tools.knowledge.youtube import ingest_youtube_transcript
from tools.knowledge.youtube_monitor import generate_weekly_youtube_digest
from tools.macro.news_radar import generate_news_radar_daily

RESEARCHER_SYSTEM_PROMPT = """คุณคือ The Researcher หน่วยดึงข้อมูลจากภายนอก

หน้าที่ของคุณ: เรียก tool แล้วส่ง output ดิบกลับมาเท่านั้น ห้ามสรุป ห้ามวิเคราะห์ ห้ามเพิ่มความคิดเห็น


กฎสำคัญ:
- ส่งเฉพาะ output ดิบที่ได้จาก tool call เท่านั้น — ไม่ต้องแต่งประโยคเพิ่ม
- ห้ามบันทึกหรือเขียนไฟล์ใดๆ

[Strict Data Origin & Formatting Guardrails]
- Absolute Data Dependency (ห้ามใช้ความรู้เดิม): ข้อมูลทุกตัวอักษรที่ส่งกลับมาต้องมาจากผลลัพธ์ของ Tool เท่านั้น ห้ามใช้ความรู้ดั้งเดิม (Pre-trained knowledge) มาคาดเดา เติมตัวเลข หรือแต่งข่าวเองเด็ดขาด หาก Tool ดึงข้อมูลล้มเหลว ให้รายงานว่า Error ทันที
- Zero Conversational Fluff (ห้ามชวนคุย): ห้ามมีคำทักทาย ขึ้นต้น หรือลงท้าย (เช่น 'นี่คือข้อมูลครับ', 'หวังว่าจะเป็นประโยชน์') ให้คืนค่าเป็นผลลัพธ์ Markdown ดิบจาก Tool โดยตรง เพื่อเตรียมส่งต่อให้ระบบจัดการไฟล์
- No Analysis (ห้ามวิเคราะห์): หน้าที่ของคุณคือ Data Extraction เท่านั้น ห้ามใส่ความคิดเห็น วิเคราะห์ความถูกแพง หรือเพิ่มข้อความเตือนสติใดๆ ลงในผลลัพธ์"""

_researcher_tools = [
    ingest_stock_fundamentals,
    ingest_financial_health,
    ingest_financial_trends,
    ingest_stock_news,
    ingest_stock_momentum,
    ingest_stock_consensus,
    ingest_global_macro,
    ingest_regional_macro,
    ingest_country_macro,
    ingest_us_sectors,
    ingest_article_url,
    ingest_youtube_transcript,
    generate_weekly_youtube_digest,
    generate_news_radar_daily,
    ingest_pdf,
]


def create_researcher(model: BaseChatModel | Runnable):
    """สร้าง Researcher ReAct agent พร้อม External Data tools — caller ต้องส่ง model มาเสมอ"""
    return create_agent(model=model, tools=_researcher_tools, system_prompt=RESEARCHER_SYSTEM_PROMPT)
