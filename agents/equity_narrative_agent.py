from langchain_core.language_models.chat_models import BaseChatModel
from core.prompt_harness import get_harness

# EQUITY_NARRATIVE_SYSTEM_PROMPT ถูกย้ายไปที่ prompts/skills/equity_narrative/SKILL.md ผ่านระบบ PromptHarness


def create_equity_narrative(model: BaseChatModel):
    """RunnableLambda มิเรอร์ agents/macro_economist_agent.py — เรียก tools ที่มีอยู่แล้วตรงๆ ในโค้ด
    (ไม่ให้ LLM เลือกเอง เพราะไม่มีอะไรให้ 'เลือก' — ต้องดึงทั้ง Vault sentiment และข่าวล่าสุดเสมอ)
    แล้วบังคับ schema ผ่าน with_structured_output ครั้งเดียวตอนท้าย
    """
    from langchain_core.runnables import RunnableLambda
    from langchain_core.messages import AIMessage
    from schemas.micro_quant_schemas import EquitySentimentContext
    from tools.archivist.search import search_memories_with_evidence
    from tools.market.news import ingest_stock_news

    def _run_equity_narrative(input_dict):
        ticker = input_dict.get("ticker") or ""
        market = input_dict.get("market", "US")
        company_name = input_dict.get("company_name")

        # ค้นด้วยชื่อบริษัทเต็มถ้ามี (ไม่ใช่แค่ ticker เปล่าๆ) ผ่าน evidence-bound semantic/vector retrieval
        # search ค้นด้วยชื่อบริษัทภาษาธรรมชาติมักได้ผลลัพธ์ตรงกว่าค้นด้วย ticker symbol สั้นๆ
        search_keyword = f"{ticker} {company_name}".strip() if company_name else ticker

        try:
            vault_text = search_memories_with_evidence.invoke(
                {
                    "keyword": search_keyword,
                    "retrieval_namespace": "primary",
                    "production_mode": False,
                }
            )
        except Exception as e:
            vault_text = f"Error searching vault: {e}"

        try:
            news_text = ingest_stock_news.invoke({"ticker": ticker, "market": market})
        except Exception as e:
            news_text = f"Error fetching news: {e}"

        context = f"=== Vault Sentiment History ===\n{vault_text}\n\n=== Latest News ===\n{news_text}"

        structured = model.with_structured_output(EquitySentimentContext)
        harness = get_harness("equity_narrative")
        res = structured.invoke([
            {"role": "system", "content": harness.get_system_prompt()},
            {"role": "user", "content": harness.get_skill_text("HUMAN.md", ticker=ticker, context=context)},
        ])

        return {
            "messages": [AIMessage(content=res.model_dump_json(), name="equity_narrative")],
            "equity_news_raw": news_text,
        }

    return RunnableLambda(_run_equity_narrative)
