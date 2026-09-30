import os
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.agents import create_agent
from tools.market.equity_quant_tool import compute_equity_quant_signals
from tools.macro.terminal_v2_tools import (
    fetch_cboe_options_max_pain,
    fetch_equity_news_discovery,
    fetch_finra_short_volume,
    fetch_nasdaq_equity_consensus,
    fetch_sec_financial_facts,
    fetch_sec_insider_trades,
)
from core.prompt_harness import get_harness

# EQUITY_QUANT_SYSTEM_PROMPT ถูกย้ายไปที่ prompts/skills/equity_quant/SKILL.md ผ่านระบบ PromptHarness

_ENABLE_TERMINAL_V2_TOOLS = os.getenv("ENABLE_TERMINAL_V2_TOOLS", "true").strip().lower() in ("true", "1", "yes")

_base_equity_quant_tools = [compute_equity_quant_signals]
_institutional_equity_tools = [
    fetch_finra_short_volume,
    fetch_cboe_options_max_pain,
    fetch_nasdaq_equity_consensus,
    fetch_sec_financial_facts,
    fetch_sec_insider_trades,
    fetch_equity_news_discovery,
]


def get_equity_quant_tools() -> list:
    """Return active equity quant tools based on feature flag."""
    if _ENABLE_TERMINAL_V2_TOOLS:
        return _base_equity_quant_tools + _institutional_equity_tools
    return _base_equity_quant_tools


def create_equity_quant(model: BaseChatModel):
    return create_agent(
        model=model,
        tools=get_equity_quant_tools(),
        system_prompt=get_harness("equity_quant").get_system_prompt(),
    )

