"""Automated Quality Gate for Tool Design & Prompt Harness Conventions.

Validates:
1. Tool Description Quality (>= 15 characters, no unfinished placeholders)
2. Tool Name Collision (No duplicate names in any Agent tool registry)
3. Schema Token Budget (Tool argument JSON Schema <= 4,000 chars)
4. Wrapper Transparency (wrap_tools_with_harness preserves 100% of tool contract)
"""
import json
import pytest

from agents.archivist_agent import ARCHIVIST_TOOL_REGISTRY
from agents.equity_quant_agent import EQUITY_QUANT_TOOL_REGISTRY
from agents.macro_quant_agent import MACRO_QUANT_TOOL_REGISTRY
from core.tool_harness import wrap_tools_with_harness

REGISTRIES = [
    ARCHIVIST_TOOL_REGISTRY,
    MACRO_QUANT_TOOL_REGISTRY,
    EQUITY_QUANT_TOOL_REGISTRY,
]


@pytest.mark.parametrize("registry", REGISTRIES, ids=lambda r: r.agent_name)
def test_rule_1_description_quality(registry):
    """Rule 1: Every tool must have a clear, descriptive description (>= 15 chars, no placeholders)."""
    placeholders = {"todo", "fixme", "tbd", "<placeholder>", "xxx"}

    for tool in registry.all_tools():
        desc = (tool.description or "").strip()
        assert len(desc) >= 15, (
            f"Tool '{tool.name}' in agent '{registry.agent_name}' has an overly short "
            f"description ({len(desc)} chars): '{desc}'"
        )
        for ph in placeholders:
            assert ph not in desc.lower(), (
                f"Tool '{tool.name}' in agent '{registry.agent_name}' contains placeholder '{ph}' "
                f"in description: '{desc}'"
            )


@pytest.mark.parametrize("registry", REGISTRIES, ids=lambda r: r.agent_name)
def test_rule_2_no_tool_name_collision(registry):
    """Rule 2: No duplicate tool names within the same Agent registry."""
    all_tools = registry.all_tools()
    names = [t.name for t in all_tools]
    unique_names = set(names)
    assert len(names) == len(unique_names), (
        f"Duplicate tool names detected in agent '{registry.agent_name}': "
        f"{[n for n in names if names.count(n) > 1]}"
    )


@pytest.mark.parametrize("registry", REGISTRIES, ids=lambda r: r.agent_name)
def test_rule_3_schema_token_budget(registry):
    """Rule 3: Tool JSON schema must not exceed 4,000 characters to prevent context bloat."""
    for tool in registry.all_tools():
        if tool.args_schema:
            schema_json = json.dumps(tool.args_schema.model_json_schema(), ensure_ascii=False)
            assert len(schema_json) <= 4000, (
                f"Tool '{tool.name}' in agent '{registry.agent_name}' has an excessively large "
                f"schema ({len(schema_json)} chars > 4000 limit)."
            )


@pytest.mark.parametrize("registry", REGISTRIES, ids=lambda r: r.agent_name)
def test_rule_4_wrapper_transparency(registry):
    """Rule 4: wrap_tools_with_harness preserves 100% of tool name, description, args_schema, and return_direct."""
    original_tools = registry.all_tools()
    wrapped_tools = wrap_tools_with_harness(original_tools)

    assert len(original_tools) == len(wrapped_tools)
    for orig, wrapped in zip(original_tools, wrapped_tools):
        assert orig.name == wrapped.name
        assert orig.description == wrapped.description
        assert orig.args_schema == wrapped.args_schema
        assert orig.return_direct == wrapped.return_direct
