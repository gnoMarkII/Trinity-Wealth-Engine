from langchain_core.language_models.chat_models import BaseChatModel
from langchain.agents import create_agent
from tools.macro.evaluation import evaluate_macro_matrix
from core.prompt_harness import get_harness

# MACRO_QUANT_SYSTEM_PROMPT ถูกย้ายไปที่ prompts/skills/macro_quant/SKILL.md ผ่านระบบ PromptHarness

_macro_quant_tools = [evaluate_macro_matrix]

def create_macro_quant(model: BaseChatModel):
    return create_agent(
        model=model,
        tools=_macro_quant_tools,
        system_prompt=get_harness("macro_quant").get_system_prompt()
    )
