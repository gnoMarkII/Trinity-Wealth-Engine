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
from core.tool_harness import (
    ProgressiveToolRegistry,
    ToolGroupSpec,
    wrap_tools_with_harness,
)

# EQUITY_QUANT_SYSTEM_PROMPT ถูกย้ายไปที่ prompts/skills/equity_quant/SKILL.md ผ่านระบบ PromptHarness

_ENABLE_TERMINAL_V2_TOOLS = os.getenv("ENABLE_TERMINAL_V2_TOOLS", "true").strip().lower() in ("true", "1", "yes")

_base_equity_quant_tools = [compute_equity_quant_signals]

EQUITY_QUANT_TOOL_REGISTRY = ProgressiveToolRegistry(
    agent_name="equity_quant",
    core_tools=[compute_equity_quant_signals],
    groups=[
        ToolGroupSpec(
            name="options_short_flow",
            tools=(fetch_finra_short_volume, fetch_cboe_options_max_pain),
            keywords=("options", "short", "finra", "max_pain"),
            description="Short interest volume and CBOE options Max Pain metrics",
        ),
        ToolGroupSpec(
            name="sec_fundamentals_insider",
            tools=(fetch_sec_financial_facts, fetch_sec_insider_trades, fetch_nasdaq_equity_consensus),
            keywords=("sec", "insider", "consensus", "fundamentals", "facts"),
            description="SEC 10-K/10-Q XBRL facts, insider Form 4 trades, and analyst consensus",
        ),
        ToolGroupSpec(
            name="news_discovery",
            tools=(fetch_equity_news_discovery,),
            keywords=("news", "discovery", "ข่าว"),
            description="Automated equity news discovery and sentiment headlines",
        ),
    ],
)


def get_equity_quant_tools(instruction: str | None = None) -> list:
    """Return active equity quant tools based on feature flag and progressive instruction."""
    if _ENABLE_TERMINAL_V2_TOOLS:
        raw_tools = EQUITY_QUANT_TOOL_REGISTRY.select_tools(instruction)
    else:
        raw_tools = list(EQUITY_QUANT_TOOL_REGISTRY.core_tools)
    return wrap_tools_with_harness(
        raw_tools,
        exempt_tool_names={"compute_equity_quant_signals"},
    )


def create_equity_quant(model: BaseChatModel, instruction: str | None = None):
    return create_agent(
        model=model,
        tools=get_equity_quant_tools(instruction),
        system_prompt=get_harness("equity_quant").get_system_prompt(),
    )

