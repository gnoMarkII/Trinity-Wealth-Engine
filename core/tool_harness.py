"""General Runtime Tool Harness, Progressive Disclosure, and Execution Guards.

Architecture Layer: core/
Dependencies: Standard Library + langchain_core.tools (No API/Adapter imports).
Complies with 16 AST Dependency Rules in tests/architecture/test_dependency_rules.py.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import json
import logging
import os
import re
from typing import Any, Callable, Coroutine, Optional, Sequence

from langchain_core.tools import BaseTool, StructuredTool

log = logging.getLogger(__name__)

# Constants & Defaults
DEFAULT_MAX_TOOL_OUTPUT_CHARS = 12000
MAX_IDENTICAL_TOOL_CALLS = 3
MAX_CONSECUTIVE_TOOL_ERRORS = 4
MAX_TOTAL_TOOL_CALLS = 20

_FRONTMATTER_ENTITY_RE = re.compile(r"^---\s*\n.*?entity_type:\s*\S+", re.DOTALL | re.MULTILINE)


def truncate_tool_output(
    output: Any,
    max_chars: Optional[int] = None,
    preserve_frontmatter_notes: bool = True,
) -> str:
    """Truncates large tool output preserving head (75%) and tail (25%) tokens.

    Preserves full content if preserve_frontmatter_notes is True and content starts
    with Obsidian YAML Frontmatter containing `entity_type:`.
    """
    if output is None:
        return ""
    if isinstance(output, (dict, list)):
        try:
            text = json.dumps(output, ensure_ascii=False)
        except Exception:
            text = str(output)
    else:
        text = str(output)

    if preserve_frontmatter_notes and _FRONTMATTER_ENTITY_RE.match(text.strip()):
        return text

    effective_max = max_chars
    if effective_max is None:
        env_limit = os.getenv("MAX_TOOL_OUTPUT_CHARS")
        if env_limit and env_limit.strip().isdigit():
            effective_max = int(env_limit.strip())
        else:
            effective_max = DEFAULT_MAX_TOOL_OUTPUT_CHARS

    if len(text) <= effective_max:
        return text

    head_chars = int(effective_max * 0.75)
    tail_chars = int(effective_max * 0.25)
    omitted = len(text) - (head_chars + tail_chars)
    notice = f"\n\n... [TRUNCATED: {omitted} characters omitted to preserve context budget] ...\n\n"
    return text[:head_chars] + notice + text[-tail_chars:]


def _canonical_args(args: Any) -> str:
    if isinstance(args, dict):
        try:
            return json.dumps(args, sort_keys=True, ensure_ascii=False)
        except Exception:
            return str(sorted(args.items()))
    return str(args)


class RunawayTracker:
    """Tracks tool invocation patterns within a worker turn to trip runaway loops."""

    def __init__(
        self,
        max_identical_calls: int = MAX_IDENTICAL_TOOL_CALLS,
        max_consecutive_errors: int = MAX_CONSECUTIVE_TOOL_ERRORS,
        max_total_calls: int = MAX_TOTAL_TOOL_CALLS,
    ) -> None:
        self.max_identical_calls = max_identical_calls
        self.max_consecutive_errors = max_consecutive_errors
        self.max_total_calls = max_total_calls

        self.last_call_key: Optional[str] = None
        self.identical_call_count: int = 0
        self.consecutive_error_count: int = 0
        self.total_calls: int = 0

    def check_before_call(self, tool_name: str, args: Any) -> None:
        self.total_calls += 1
        if self.total_calls > self.max_total_calls:
            raise RuntimeError(
                f"Runaway guard tripped: total tool calls ({self.total_calls}) "
                f"exceeded cap ({self.max_total_calls}) in a single worker turn."
            )

        call_key = f"{tool_name}:{_canonical_args(args)}"
        if call_key == self.last_call_key:
            self.identical_call_count += 1
            if self.identical_call_count >= self.max_identical_calls:
                raise RuntimeError(
                    f"Runaway guard tripped: identical tool call loop detected for "
                    f"'{tool_name}' ({self.identical_call_count} consecutive calls with identical arguments)."
                )
        else:
            self.last_call_key = call_key
            self.identical_call_count = 1

    def record_output(self, output: Any) -> None:
        is_error = False
        if isinstance(output, str):
            clean = output.strip()
            if clean.startswith("Error:") or clean.startswith("Error ") or "[ERROR]" in clean[:50]:
                is_error = True
        elif isinstance(output, dict):
            if output.get("error") or output.get("status") == "error":
                is_error = True

        if is_error:
            self.consecutive_error_count += 1
            if self.consecutive_error_count >= self.max_consecutive_errors:
                raise RuntimeError(
                    f"Runaway guard tripped: {self.consecutive_error_count} consecutive "
                    f"tool errors encountered without recovery."
                )
        else:
            self.consecutive_error_count = 0


_CURRENT_RUNAWAY_TRACKER: ContextVar[Optional[RunawayTracker]] = ContextVar(
    "current_runaway_tracker", default=None
)


def get_current_runaway_tracker() -> Optional[RunawayTracker]:
    """Returns the active RunawayTracker for the current context/thread."""
    return _CURRENT_RUNAWAY_TRACKER.get()


@contextmanager
def runaway_guard_scope(tracker: Optional[RunawayTracker] = None):
    """Context manager setting a dedicated RunawayTracker for a worker graph turn."""
    active_tracker = tracker or RunawayTracker()
    token = _CURRENT_RUNAWAY_TRACKER.set(active_tracker)
    try:
        yield active_tracker
    finally:
        _CURRENT_RUNAWAY_TRACKER.reset(token)


@dataclass(frozen=True)
class ToolGroupSpec:
    """Capability group metadata for progressive tool loading."""

    name: str
    tools: tuple[BaseTool, ...]
    keywords: tuple[str, ...] = ()
    description: str = ""


class ProgressiveToolRegistry:
    """General-purpose progressive tool loader for any ReAct agent."""

    def __init__(
        self,
        agent_name: str,
        core_tools: Sequence[BaseTool],
        groups: Sequence[ToolGroupSpec] = (),
    ) -> None:
        self.agent_name = agent_name
        self.core_tools = tuple(core_tools)
        self.groups = tuple(groups)

    def all_tools(self) -> list[BaseTool]:
        """Returns all registered tools in order without duplicates."""
        seen: set[str] = set()
        ordered: list[BaseTool] = []

        for tool in self.core_tools:
            if tool.name not in seen:
                seen.add(tool.name)
                ordered.append(tool)

        for group in self.groups:
            for tool in group.tools:
                if tool.name not in seen:
                    seen.add(tool.name)
                    ordered.append(tool)

        return ordered

    def select_tools(self, instruction: Optional[str] = None) -> list[BaseTool]:
        """Selects core tools + matched groups, or falls back to all_tools() if no match."""
        if not instruction or not instruction.strip():
            return self.all_tools()

        inst_lower = instruction.lower()
        matched_groups = [
            group for group in self.groups
            if any(keyword.lower() in inst_lower for keyword in group.keywords)
        ]

        # Zero False-Negative: If no group keywords matched, load all tools
        if not matched_groups:
            return self.all_tools()

        seen: set[str] = set()
        selected: list[BaseTool] = []

        for tool in self.core_tools:
            if tool.name not in seen:
                seen.add(tool.name)
                selected.append(tool)

        for group in matched_groups:
            for tool in group.tools:
                if tool.name not in seen:
                    seen.add(tool.name)
                    selected.append(tool)

        return selected


def wrap_tools_with_harness(
    tools: Sequence[BaseTool],
    *,
    max_output_chars: int = DEFAULT_MAX_TOOL_OUTPUT_CHARS,
    exempt_tool_names: Optional[set[str]] = None,
) -> list[BaseTool]:
    """Wraps tools with RunawayTracker checks and output truncation.

    Preserves 100% of tool name, description, args_schema, and return_direct.
    """
    exempt = exempt_tool_names or set()
    wrapped_list: list[BaseTool] = []

    for tool in tools:
        is_exempt = tool.name in exempt

        def _make_sync_wrapper(target_tool: BaseTool, exempt_flag: bool) -> Callable[..., Any]:
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                tracker = get_current_runaway_tracker()
                call_args = kwargs if kwargs else (args[0] if len(args) == 1 else args)
                if tracker:
                    tracker.check_before_call(target_tool.name, call_args)

                try:
                    result = target_tool.invoke(call_args)
                except Exception as exc:
                    if tracker:
                        tracker.record_output(f"Error: {exc}")
                    raise

                if tracker:
                    tracker.record_output(result)

                if not exempt_flag:
                    result = truncate_tool_output(result, max_chars=max_output_chars)
                return result

            return sync_wrapper

        def _make_async_wrapper(target_tool: BaseTool, exempt_flag: bool) -> Callable[..., Coroutine[Any, Any, Any]]:
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                tracker = get_current_runaway_tracker()
                call_args = kwargs if kwargs else (args[0] if len(args) == 1 else args)
                if tracker:
                    tracker.check_before_call(target_tool.name, call_args)

                try:
                    result = await target_tool.ainvoke(call_args)
                except Exception as exc:
                    if tracker:
                        tracker.record_output(f"Error: {exc}")
                    raise

                if tracker:
                    tracker.record_output(result)

                if not exempt_flag:
                    result = truncate_tool_output(result, max_chars=max_output_chars)
                return result

            return async_wrapper

        sync_fn = _make_sync_wrapper(tool, is_exempt)
        coroutine_fn = _make_async_wrapper(tool, is_exempt) if tool.coroutine else None

        wrapped_tool = StructuredTool.from_function(
            func=sync_fn,
            coroutine=coroutine_fn,
            name=tool.name,
            description=tool.description,
            args_schema=tool.args_schema,
            return_direct=tool.return_direct,
        )
        wrapped_list.append(wrapped_tool)

    return wrapped_list
