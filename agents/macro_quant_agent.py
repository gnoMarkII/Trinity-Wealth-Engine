import os
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.agents import create_agent
from tools.macro.evaluation import evaluate_macro_matrix
from tools.macro.terminal_v2_tools import (
    fetch_cftc_metals_positioning,
    fetch_commodity_volatility,
    fetch_fred_macro_series,
    fetch_global_policy_rates,
    fetch_nyfed_reference_rates,
    fetch_ofr_financial_stress,
    fetch_polymarket_prediction_markets,
    fetch_spot_etf_flows,
    fetch_thai_fund_asset_allocation,
    fetch_thai_market_flow,
    fetch_thai_public_debt,
    fetch_thai_retail_gold,
    fetch_treasury_auction_demand,
    fetch_treasury_yield_curve,
)
from core.prompt_harness import get_harness
from core.tool_harness import (
    ProgressiveToolRegistry,
    ToolGroupSpec,
    wrap_tools_with_harness,
)

# MACRO_QUANT_SYSTEM_PROMPT ถูกย้ายไปที่ prompts/skills/macro_quant/SKILL.md ผ่านระบบ PromptHarness

_ENABLE_TERMINAL_V2_TOOLS = os.getenv("ENABLE_TERMINAL_V2_TOOLS", "true").strip().lower() in ("true", "1", "yes")

_base_macro_quant_tools = [evaluate_macro_matrix]

MACRO_QUANT_TOOL_REGISTRY = ProgressiveToolRegistry(
    agent_name="macro_quant",
    core_tools=[evaluate_macro_matrix],
    groups=[
        ToolGroupSpec(
            name="thai_macro",
            tools=(
                fetch_thai_market_flow,
                fetch_thai_retail_gold,
                fetch_thai_fund_asset_allocation,
                fetch_thai_public_debt,
            ),
            keywords=("thai", "ไทย", "บาท", "set", "set50", "gold_retail"),
            description="Thai local market liquidity, retail gold, public debt and mutual fund flows",
        ),
        ToolGroupSpec(
            name="us_rates_treasury",
            tools=(
                fetch_fred_macro_series,
                fetch_nyfed_reference_rates,
                fetch_treasury_yield_curve,
                fetch_treasury_auction_demand,
                fetch_global_policy_rates,
            ),
            keywords=("fred", "yield", "treasury", "rates", "ดอกเบี้ย", "พันธบัตร", "auction"),
            description="Global & US interest rates, yield curve, and central bank policy rates",
        ),
        ToolGroupSpec(
            name="flows_positioning_stress",
            tools=(
                fetch_polymarket_prediction_markets,
                fetch_spot_etf_flows,
                fetch_ofr_financial_stress,
                fetch_cftc_metals_positioning,
                fetch_commodity_volatility,
            ),
            keywords=("flow", "stress", "cftc", "positioning", "etf", "polymarket", "volatility"),
            description="Cross-asset positioning, ETF flows, financial stress indices and sentiment",
        ),
    ],
)


def get_macro_quant_tools(instruction: str | None = None) -> list:
    """Return active macro quant tools based on feature flag and progressive instruction."""
    if _ENABLE_TERMINAL_V2_TOOLS:
        raw_tools = MACRO_QUANT_TOOL_REGISTRY.select_tools(instruction)
    else:
        raw_tools = list(MACRO_QUANT_TOOL_REGISTRY.core_tools)
    return wrap_tools_with_harness(
        raw_tools,
        exempt_tool_names={"evaluate_macro_matrix"},
    )


def create_macro_quant(model: BaseChatModel, instruction: str | None = None):
    return create_agent(
        model=model,
        tools=get_macro_quant_tools(instruction),
        system_prompt=get_harness("macro_quant").get_system_prompt(),
    )

