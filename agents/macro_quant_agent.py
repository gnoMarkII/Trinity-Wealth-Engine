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

# MACRO_QUANT_SYSTEM_PROMPT ถูกย้ายไปที่ prompts/skills/macro_quant/SKILL.md ผ่านระบบ PromptHarness

_ENABLE_TERMINAL_V2_TOOLS = os.getenv("ENABLE_TERMINAL_V2_TOOLS", "true").strip().lower() in ("true", "1", "yes")

_base_macro_quant_tools = [evaluate_macro_matrix]
_terminal_v2_macro_tools = [
    fetch_thai_market_flow,
    fetch_thai_retail_gold,
    fetch_fred_macro_series,
    fetch_nyfed_reference_rates,
    fetch_treasury_yield_curve,
    fetch_thai_fund_asset_allocation,
    fetch_thai_public_debt,
    fetch_polymarket_prediction_markets,
    fetch_spot_etf_flows,
    fetch_ofr_financial_stress,
    fetch_cftc_metals_positioning,
    fetch_global_policy_rates,
    fetch_commodity_volatility,
    fetch_treasury_auction_demand,
]


def get_macro_quant_tools() -> list:
    """Return active macro quant tools based on feature flag."""
    if _ENABLE_TERMINAL_V2_TOOLS:
        return _base_macro_quant_tools + _terminal_v2_macro_tools
    return _base_macro_quant_tools


def create_macro_quant(model: BaseChatModel):
    return create_agent(
        model=model,
        tools=get_macro_quant_tools(),
        system_prompt=get_harness("macro_quant").get_system_prompt(),
    )

