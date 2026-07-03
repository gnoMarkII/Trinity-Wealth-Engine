from langchain_core.language_models.chat_models import BaseChatModel
from langchain.agents import create_agent
from tools.macro.news_radar import generate_news_radar_daily
from tools.knowledge.youtube_monitor import generate_weekly_youtube_digest
from tools.knowledge.article import ingest_article_url
from tools.knowledge.youtube import ingest_youtube_transcript
from tools.macro.baselines import get_macro_baselines

MACRO_ECONOMIST_SYSTEM_PROMPT = """คุณคือ Macro Economist — ผู้จับกระแสทิศทางลมจากข่าว นโยบาย และเทรนด์

หน้าที่:
- ดึงข้อมูลจาก News Radar, YouTube Digest, และ Article URL
- สกัดธีมการลงทุนหลักและ Tail Risks
- สรุป Market Sentiment และ Policy Signals สำคัญ

กฎสำคัญ:
- ขั้นที่ 1: เรียก `get_macro_baselines` เพื่อดูข้อมูลเก่า
- ขั้นที่ 2: เรียก `generate_news_radar_daily` เพื่อดึงข่าวของวันนี้
- ขั้นที่ 3: **สังเคราะห์** ข้อมูลทั้งหมด แล้วเขียนเป็น JSON ที่สมบูรณ์
- ให้คัดลอกตัวเลข sources_count และ age_hours จากข้อความต้นฉบับ
- หากมีการเปลี่ยนแปลงทิศทางจากอดีต (Pivot) ให้ระบุให้ชัดเจนโดยอิงจากข้อมูล `get_macro_baselines`
- **คำเตือนขั้นเด็ดขาด**: ห้ามคัดลอกผลลัพธ์ของ Tool มาตอบตรงๆ หน้าที่ของคุณคือ **สังเคราะห์เป็น JSON** ตาม Schema ด้านล่างนี้เท่านั้น
- ส่งคืนข้อความที่เป็น JSON ล้วนๆ (ไม่ต้องมี \`\`\`json คร่อม) ห้ามมีข้อความอื่นปนเด็ดขาด

[NarrativeContext JSON Schema]
{
  "evaluated_at": "ISO format string",
  "dominant_themes": [
    {
      "category": "policy|growth|inflation|liquidity|geopolitics|earnings|risk_sentiment",
      "theme_title": "string",
      "deduplicated_summary": "string",
      "age_hours": 0,
      "sources_count": 0,
      "asset_impacts": {"equity": "bullish|bearish|neutral", "bond": "bullish|bearish|neutral"},
      "market_impact_score": 0.0,
      "event_confidence": 1.0,
      "pivot_strength": "none|weak|moderate|strong",
      "changed_from": "string or null",
      "baseline_date": "string or null",
      "pivot_evidence": "string or null"
    }
  ],
  "market_sentiment": "bullish|neutral|bearish",
  "tail_risks": ["string"],
  "policy_signals": ["string"],
  "key_narratives_by_region": {"RegionName": "string"},
  "sources_summary": "string"
}"""

_macro_economist_tools = [
    generate_news_radar_daily,
    generate_weekly_youtube_digest,
    ingest_article_url,
    ingest_youtube_transcript,
    get_macro_baselines,
]

def create_macro_economist(model: BaseChatModel):
    from langchain_core.runnables import RunnableLambda
    from langchain_core.messages import AIMessage
    from schemas.macro_schemas import NarrativeContext
    
    def _run_economist(input_dict):
        # 1. Fetch data directly without relying on ReAct loop
        try:
            baseline_text = get_macro_baselines.invoke({})
        except Exception as e:
            baseline_text = f"Error fetching baselines: {e}"
            
        try:
            news_text = generate_news_radar_daily.invoke({})
        except Exception as e:
            news_text = f"Error fetching news: {e}"
            
        context = f"=== Baseline ===\n{baseline_text}\n\n=== News ===\n{news_text}"
        
        # 2. Use structured output to force correct schema
        structured = model.with_structured_output(NarrativeContext)
        
        res = structured.invoke([
            {"role": "system", "content": MACRO_ECONOMIST_SYSTEM_PROMPT},
            {"role": "user", "content": f"Please synthesize the NarrativeContext JSON based on the following data:\n\n{context}"}
        ])
        
        return {"messages": [AIMessage(content=res.model_dump_json(), name="macro_economist")]}

    return RunnableLambda(_run_economist)
