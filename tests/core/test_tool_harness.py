"""Unit tests for core/tool_harness.py: Truncation, Tripwires, and Progressive Tool Registry."""
import os
import pytest
from langchain_core.tools import tool

from core.tool_harness import (
    DEFAULT_MAX_TOOL_OUTPUT_CHARS,
    MAX_CONSECUTIVE_TOOL_ERRORS,
    MAX_IDENTICAL_TOOL_CALLS,
    MAX_REPEATED_SIGNATURE_CALLS,
    MAX_TOTAL_TOOL_CALLS,
    ProgressiveToolRegistry,
    RunawayTracker,
    ToolGroupSpec,
    get_current_runaway_tracker,
    runaway_guard_scope,
    truncate_tool_output,
    wrap_tools_with_harness,
)


# --- 1. Truncation Tests ---

def test_truncate_tool_output_short():
    short_text = "Analysis completed successfully."
    assert truncate_tool_output(short_text, max_chars=100) == short_text


def test_truncate_tool_output_preserves_valid_json_dict_and_numbers():
    import json
    large_payload = {
        "ticker": "NVDA",
        "pe_ratio": 24.5,
        "revenue": 150000000,
        "is_active": True,
        "news_headline": "Comprehensive long report " * 600,
        "items": [f"item_{i}" for i in range(100)],
    }
    res = truncate_tool_output(large_payload, max_chars=4000)
    assert len(res) <= 4000
    parsed = json.loads(res)
    assert parsed["ticker"] == "NVDA"
    assert parsed["pe_ratio"] == 24.5
    assert parsed["revenue"] == 150000000
    assert parsed["is_active"] is True
    assert "[TRUNCATED" in parsed["news_headline"]
    assert len(parsed["items"]) <= 50
    assert parsed["_truncation_metadata"]["truncated"] is True
    assert parsed["_truncation_metadata"]["partial_result"] is True
    assert parsed["_truncation_metadata"]["omitted_arrays"][0]["path"] == "$.items"
    assert parsed["_truncation_metadata"]["omitted_arrays"][0]["original_count"] == 100
    assert parsed["_truncation_metadata"]["omitted_arrays"][0]["returned_count"] == len(parsed["items"])
    assert parsed["_truncation_metadata"]["omitted_arrays"][0]["omitted_count"] == 100 - len(parsed["items"])


def test_truncate_tool_output_r11_omitted_list_and_wide_dict_completeness():
    import json

    # 1. R11 repro: payload with total_count=100 and 100 items at budget 1800
    payload = {
        "total_count": 100,
        "items": [{"symbol": f"SYM{i}", "weight": 1.0} for i in range(100)],
    }
    res = truncate_tool_output(payload, max_chars=1800)
    assert len(res) <= 1800
    parsed = json.loads(res)
    assert parsed["total_count"] == 100
    assert all(isinstance(x, dict) and "symbol" in x for x in parsed["items"])
    assert parsed["_truncation_metadata"]["truncated"] is True
    assert parsed["_truncation_metadata"]["omitted_arrays"] == [
        {
            "path": "$.items",
            "original_count": 100,
            "returned_count": len(parsed["items"]),
            "omitted_count": 100 - len(parsed["items"]),
        }
    ]
    assert "retrieval_hint" in parsed["_truncation_metadata"]

    # 2. JSON-string input with nested array
    nested_str = json.dumps({"portfolio": {"holdings": [{"ticker": f"T{i}"} for i in range(80)]}})
    res_nested = truncate_tool_output(nested_str, max_chars=900)
    assert len(res_nested) <= 900
    parsed_nested = json.loads(res_nested)
    assert parsed_nested["_truncation_metadata"]["truncated"] is True
    assert parsed_nested["_truncation_metadata"]["omitted_arrays"][0]["path"] == "$.portfolio.holdings"

    # 3. Root numeric-only array must not inject sentinel strings into numeric list
    root_nums = list(range(500))
    res_root = truncate_tool_output(root_nums, max_chars=400)
    assert len(res_root) <= 400
    parsed_root = json.loads(res_root)
    assert isinstance(parsed_root, dict)
    assert all(isinstance(x, int) for x in parsed_root["items"])
    assert parsed_root["_truncation_metadata"]["truncated"] is True
    assert parsed_root["_truncation_metadata"]["original_count"] == 500
    assert parsed_root["_truncation_metadata"]["omitted_count"] == 500 - len(parsed_root["items"])

    # 4. Wide dict with 1,000 short numeric keys at 1,000-char limit stays valid JSON <= 1,000
    wide = {f"k{i}": i for i in range(1000)}
    res_wide = truncate_tool_output(wide, max_chars=1000)
    assert len(res_wide) <= 1000
    parsed_wide = json.loads(res_wide)
    meta = parsed_wide["_truncation_metadata"]
    assert meta["truncated"] is True
    assert meta["partial_result"] is True
    assert meta["omitted_keys"][0]["original_key_count"] == 1000
    assert meta["omitted_keys"][0]["omitted_key_count"] > 0

    # 5. Very small positive budget retains bounded valid JSON envelope
    res_tiny = truncate_tool_output(wide, max_chars=40)
    assert len(res_tiny) <= 40
    parsed_tiny = json.loads(res_tiny)
    assert parsed_tiny.get("_truncated") is True or parsed_tiny.get("_truncation_metadata", {}).get("truncated") is True

    # 6. Unsupported / below-minimum budget settings raise explicitly
    with pytest.raises(ValueError):
        truncate_tool_output(wide, max_chars=0)
    with pytest.raises(ValueError):
        truncate_tool_output(wide, max_chars=-10)
    with pytest.raises(ValueError):
        truncate_tool_output(wide, max_chars=10)


@pytest.mark.asyncio
async def test_wrap_tools_with_harness_async_sync_parity():
    @tool
    async def async_echo_tool(msg: str) -> str:
        """Async echo tool for harness parity test."""
        return msg

    wrapped = wrap_tools_with_harness([async_echo_tool], max_output_chars=120)
    w_tool = wrapped[0]
    assert w_tool.coroutine is not None

    with runaway_guard_scope(RunawayTracker(max_identical_calls=3)) as tracker:
        out = await w_tool.ainvoke({"msg": '{"items":[1,2,3,4,5],"note":"ok"}'})
        assert "items" in out
        assert tracker.total_calls == 1
        await w_tool.ainvoke({"msg": "same"})
        await w_tool.ainvoke({"msg": "same"})
        with pytest.raises(RuntimeError, match="identical tool call loop detected"):
            await w_tool.ainvoke({"msg": "same"})


def test_truncate_tool_output_preserves_json_string_for_verification_loop():
    import json
    from pydantic import BaseModel
    from core.verification_loop import extract_and_validate_json_schema

    class SimpleSignalSchema(BaseModel):
        ticker: str
        score: float
        summary: str

    raw_json_str = json.dumps({
        "ticker": "AAPL",
        "score": 88.5,
        "summary": "Detailed narrative about company performance " * 400,
    }, indent=2)

    res = truncate_tool_output(raw_json_str, max_chars=2000)
    assert len(res) <= 2000
    outcome = extract_and_validate_json_schema(res, SimpleSignalSchema)
    assert outcome.ok is True
    assert outcome.parsed_model.ticker == "AAPL"
    assert outcome.parsed_model.score == 88.5
    assert "[TRUNCATED" in outcome.parsed_model.summary


def test_truncate_tool_output_long():
    long_text = "A" * 1000 + "B" * 1000
    res = truncate_tool_output(long_text, max_chars=100)
    assert len(res) < len(long_text)
    assert "[TRUNCATED:" in res
    assert res.startswith("A" * 75)
    assert res.endswith("B" * 25)


def test_truncate_tool_output_frontmatter_exempt():
    frontmatter_note = (
        "---\n"
        "title: Comprehensive Equity Thesis\n"
        "entity_type: research_report\n"
        "date: 2026-10-16\n"
        "---\n\n"
        + "Detailed body text..." * 500
    )
    # Even if max_chars is very small, frontmatter with entity_type must not be truncated
    res = truncate_tool_output(frontmatter_note, max_chars=100, preserve_frontmatter_notes=True)
    assert res == frontmatter_note


def test_truncate_tool_output_env_override(monkeypatch):
    monkeypatch.setenv("MAX_TOOL_OUTPUT_CHARS", "200")
    text = "X" * 300
    res = truncate_tool_output(text, max_chars=None)
    assert "[TRUNCATED:" in res
    assert res.startswith("X" * 150)
    assert res.endswith("X" * 50)


# --- 2. Runaway Tracker & Tripwires Tests ---

def test_runaway_tracker_identical_call_trip():
    tracker = RunawayTracker(max_identical_calls=3)
    tracker.check_before_call("search_notes", {"query": "NVDA"})
    tracker.check_before_call("search_notes", {"query": "NVDA"})
    with pytest.raises(RuntimeError, match="identical tool call loop detected"):
        tracker.check_before_call("search_notes", {"query": "NVDA"})


def test_runaway_tracker_resets_on_different_args():
    tracker = RunawayTracker(max_identical_calls=3)
    tracker.check_before_call("search_notes", {"query": "NVDA"})
    tracker.check_before_call("search_notes", {"query": "NVDA"})
    tracker.check_before_call("search_notes", {"query": "AAPL"})  # Different args
    tracker.check_before_call("search_notes", {"query": "AAPL"})
    assert tracker.identical_call_count == 2


def test_runaway_tracker_ping_pong_cyclic_loop_trip():
    # Ping-Pong alternating calls: A -> B -> A -> B -> A
    tracker = RunawayTracker(max_identical_calls=3, max_signature_calls=3)
    tracker.check_before_call("tool_a", {"id": 1})  # a: 1
    tracker.check_before_call("tool_b", {"id": 2})  # b: 1
    tracker.check_before_call("tool_a", {"id": 1})  # a: 2
    tracker.check_before_call("tool_b", {"id": 2})  # b: 2
    tracker.check_before_call("tool_a", {"id": 1})  # a: 3
    tracker.check_before_call("tool_b", {"id": 2})  # b: 3
    with pytest.raises(RuntimeError, match="cyclic/repeated tool call loop detected"):
        tracker.check_before_call("tool_a", {"id": 1})  # a: 4 > 3 -> trips!


def test_runaway_tracker_consecutive_errors_trip():
    tracker = RunawayTracker(max_consecutive_errors=4)
    tracker.record_output("Error: connection refused")
    tracker.record_output("Error: timeout")
    tracker.record_output({"error": "server error"})
    with pytest.raises(RuntimeError, match="consecutive tool errors encountered"):
        tracker.record_output("Error: fatal")


def test_runaway_tracker_recovers_consecutive_errors():
    tracker = RunawayTracker(max_consecutive_errors=4)
    tracker.record_output("Error: bad gateway")
    tracker.record_output("Error: bad gateway")
    tracker.record_output("OK: 200 Success")  # Success resets error count
    assert tracker.consecutive_error_count == 0
    tracker.record_output("Error: new error")
    assert tracker.consecutive_error_count == 1


def test_runaway_tracker_total_calls_cap():
    tracker = RunawayTracker(max_total_calls=5)
    for i in range(5):
        tracker.check_before_call(f"tool_{i}", {"i": i})
    with pytest.raises(RuntimeError, match="total tool calls .* exceeded cap"):
        tracker.check_before_call("tool_overflow", {})


def test_runaway_guard_scope_isolation():
    assert get_current_runaway_tracker() is None
    with runaway_guard_scope() as tracker1:
        assert get_current_runaway_tracker() is tracker1
        tracker1.check_before_call("t1", {})

    assert get_current_runaway_tracker() is None

    with runaway_guard_scope() as tracker2:
        assert get_current_runaway_tracker() is tracker2
        assert tracker2.total_calls == 0  # Clean state


# --- 3. Progressive Tool Registry Tests ---

@tool
def sample_core_tool(param: str) -> str:
    """Core tool description for test."""
    return f"core_{param}"

@tool
def sample_write_tool(text: str) -> str:
    """Write tool description for test."""
    return f"write_{text}"

@tool
def sample_search_tool(query: str) -> str:
    """Search tool description for test."""
    return f"search_{query}"


def test_progressive_tool_registry_selection():
    registry = ProgressiveToolRegistry(
        agent_name="tester",
        core_tools=[sample_core_tool],
        groups=[
            ToolGroupSpec(
                name="write_group",
                tools=(sample_write_tool,),
                keywords=("บันทึก", "save", "write"),
            ),
            ToolGroupSpec(
                name="search_group",
                tools=(sample_search_tool,),
                keywords=("ค้นหา", "search"),
            ),
        ],
    )

    # 1. all_tools contains all tools
    all_names = [t.name for t in registry.all_tools()]
    assert all_names == ["sample_core_tool", "sample_write_tool", "sample_search_tool"]

    # 2. None or empty instruction loads all tools
    assert len(registry.select_tools(None)) == 3
    assert len(registry.select_tools("")) == 3

    # 3. Keyword matching write group loads core + write
    selected_write = registry.select_tools("กรุณาบันทึกข้อมูลลงไฟล์")
    names_write = [t.name for t in selected_write]
    assert "sample_core_tool" in names_write
    assert "sample_write_tool" in names_write
    assert "sample_search_tool" not in names_write

    # 4. Unknown instruction falls back to all tools (Zero False-Negative)
    fallback = registry.select_tools("สวัสดีตอนเช้า สบายดีไหม")
    assert len(fallback) == 3


# --- 4. Tool Wrapping & Harness Integration Tests ---

def test_wrap_tools_with_harness_transparency_and_truncation():
    @tool
    def echo_tool(msg: str) -> str:
        """Echo tool description that is sufficiently long."""
        return msg

    wrapped = wrap_tools_with_harness([echo_tool], max_output_chars=50, exempt_tool_names=None)
    assert len(wrapped) == 1
    w_tool = wrapped[0]

    # Verify transparency
    assert w_tool.name == echo_tool.name
    assert w_tool.description == echo_tool.description
    assert w_tool.args_schema == echo_tool.args_schema
    assert w_tool.return_direct == echo_tool.return_direct

    # Verify execution & truncation
    long_msg = "Hello World! " * 20
    res = w_tool.invoke({"msg": long_msg})
    assert "[TRUNCATED:" in res

    # Verify exemption
    wrapped_exempt = wrap_tools_with_harness([echo_tool], max_output_chars=50, exempt_tool_names={"echo_tool"})
    res_exempt = wrapped_exempt[0].invoke({"msg": long_msg})
    assert res_exempt == long_msg


# --- 5. PreToolUse Guard Hook Tests ---

def test_guard_pre_tool_hook():
    import json
    import subprocess
    import sys
    from pathlib import Path

    hook_script = Path(__file__).resolve().parent.parent.parent / ".claude" / "hooks" / "guard_pre_tool.py"
    assert hook_script.exists()

    def run_hook(payload: dict) -> tuple[int, str]:
        proc = subprocess.Popen(
            [sys.executable, str(hook_script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        _stdout, stderr = proc.communicate(json.dumps(payload))
        return proc.returncode, stderr

    # 1. Safe command allows execution (code 0)
    code, _ = run_hook({"tool_name": "Bash", "tool_input": {"command": "git status"}})
    assert code == 0

    # 2. git push --force blocks (code 2)
    code, err = run_hook({"tool_name": "Bash", "tool_input": {"command": "git push origin main --force"}})
    assert code == 2
    assert "git push --force" in err

    # 3. Reading .env directly blocks (code 2)
    code, err = run_hook({"tool_name": "Bash", "tool_input": {"command": "cat .env"}})
    assert code == 2
    assert ".env" in err

    # 4. Overwriting manifest directly blocks (code 2)
    code, err = run_hook({"tool_name": "Bash", "tool_input": {"command": "echo test > tests/fixtures/manifest_openapi_schema.json"}})
    assert code == 2
    assert "manifest" in err

