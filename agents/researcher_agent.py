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
from tools.knowledge.document import ingest_pdf
from core.prompt_harness import get_harness

# RESEARCHER_SYSTEM_PROMPT ถูกย้ายไปที่ prompts/skills/researcher/SKILL.md ผ่านระบบ PromptHarness

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
    ingest_pdf,
]


def create_researcher(model: BaseChatModel | Runnable):
    """สร้าง Researcher ReAct agent พร้อม External Data tools — caller ต้องส่ง model มาเสมอ"""
    return create_agent(
        model=model,
        tools=_researcher_tools,
        system_prompt=get_harness("researcher").get_system_prompt()
    )
