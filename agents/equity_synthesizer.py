from langchain_core.language_models.chat_models import BaseChatModel

from core.prompt_harness import get_harness
from schemas.micro_quant_schemas import EquityNarrativeOutput


def invoke_equity_synthesizer(
    model: BaseChatModel,
    quant_json: str,
    narrative_json: str,
) -> EquityNarrativeOutput:
    """Invoke the model with structured output + retry layer — มิเรอร์ agents/strategic_allocator.py

    LLM เห็นเฉพาะ quant_json/narrative_json ที่ validate ผ่านมาแล้วเป็น read-only context
    เขียนได้แค่ narrative text ตาม EquityNarrativeOutput schema — ไม่มี field ตัวเลขให้แตะเลย
    """
    from validators.structured_output_retry import invoke_with_retry

    harness = get_harness("equity_synthesizer")
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
    import json
    try:
        quant_data = json.loads(quant_json) if quant_json else {}
    except Exception:
        quant_data = {}

    narrative, _ = invoke_with_retry(
        model=model,
        messages=messages,
        output_schema=EquityNarrativeOutput,
        observable_registry={"quant_data": quant_data},
        max_retries=1,
        agent_name="equity_synthesizer",
    )
    return narrative
