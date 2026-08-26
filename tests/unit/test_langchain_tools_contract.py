"""Phase 0 Gate: Read-Only Contract Test for LangChain Agent Tools.

Asserts that LangChain tools maintain their names, descriptions, input schemas, and no extra tools are leaked.
"""
import json
from pathlib import Path
import pytest

from langchain_core.tools import BaseTool
import tools.portfolio.agent_tools as agent_tools

MANIFEST_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "manifest_langchain_tools.json"


@pytest.fixture(scope="module")
def golden_manifest():
    assert MANIFEST_PATH.exists(), f"Missing baseline manifest: {MANIFEST_PATH}"
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_langchain_tools_exist_and_match(golden_manifest):
    expected_tools = golden_manifest["tools"]

    # Check exported tools in agent_tools
    actual_tool_objs = {
        name: getattr(agent_tools, name)
        for name in dir(agent_tools)
        if isinstance(getattr(agent_tools, name), BaseTool)
    }

    # Ensure no missing tools
    for tool_name, expected_info in expected_tools.items():
        func_name = expected_info["func_name"]
        assert hasattr(agent_tools, func_name), f"Missing tool function '{func_name}' for tool '{tool_name}'"
        tool_obj = getattr(agent_tools, func_name)

        assert tool_obj.name == expected_info["name"], f"Tool name mismatch on {func_name}"
        assert tool_obj.description.strip() == expected_info["description"].strip(), f"Tool description mismatch on {tool_name}"
        assert tool_obj.return_direct == expected_info.get("return_direct", False), f"return_direct mismatch on {tool_name}"

        if tool_obj.args_schema:
            actual_schema = tool_obj.args_schema.model_json_schema()
            expected_schema = expected_info["args_schema"]
            # Verify properties and required fields
            assert actual_schema.get("properties") == expected_schema.get(
                "properties"
            ), f"Args schema properties drift on tool '{tool_name}'"
            assert actual_schema.get("required") == expected_schema.get(
                "required"
            ), f"Args schema required fields drift on tool '{tool_name}'"
