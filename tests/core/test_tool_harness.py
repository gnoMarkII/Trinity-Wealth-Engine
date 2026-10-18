"""Unit tests for core/tool_harness.py: Truncation, Tripwires, and Progressive Tool Registry."""
import os
import pytest
from langchain_core.tools import tool

from core.tool_harness import (
    DEFAULT_MAX_TOOL_OUTPUT_CHARS,
    MAX_CONSECUTIVE_TOOL_ERRORS,
    MAX_IDENTICAL_TOOL_CALLS,
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

