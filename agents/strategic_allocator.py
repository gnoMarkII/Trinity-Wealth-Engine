from typing import Any, Optional
from langchain_core.language_models.chat_models import BaseChatModel

from core.prompt_harness import get_harness
from schemas.macro_schemas import MacroStrategyDirection


def invoke_strategic_allocator(
    model: BaseChatModel,
    quant_json: str,
    narrative_json: str,
    observable_registry: Optional[dict[str, Any]] = None,
) -> MacroStrategyDirection:
    """Invoke the model with structured output, retry layer, and clean, modular skill prompts."""
    from validators.structured_output_retry import invoke_with_retry
    if observable_registry is None:
        observable_registry = {}
        
    harness = get_harness("allocator")
    system_content = harness.get_system_prompt()
    human_content = harness.get_skill_text(
        "HUMAN.md",
        quant_json=quant_json,
        narrative_json=narrative_json,
    )
    
    messages = [
        {"role": "system", "content": system_content},
        {"role": "human", "content": human_content},
    ]
    direction, _ = invoke_with_retry(
        model=model,
        messages=messages,
        output_schema=MacroStrategyDirection,
        observable_registry=observable_registry,
        max_retries=1,
    )
    return direction
