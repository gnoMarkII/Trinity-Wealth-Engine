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
MIN_STRUCTURED_JSON_CHARS = 24
MAX_IDENTICAL_TOOL_CALLS = 3
MAX_CONSECUTIVE_TOOL_ERRORS = 4
MAX_TOTAL_TOOL_CALLS = 20
MAX_REPEATED_SIGNATURE_CALLS = 4

_FRONTMATTER_ENTITY_RE = re.compile(r"^---\s*\n.*?entity_type:\s*\S+", re.DOTALL | re.MULTILINE)


def _shrink_json_value(
    value: Any,
    max_str_len: int,
    max_list_items: int,
    max_dict_keys: Optional[int] = None,
    path: str = "$",
    stats: Optional[dict[str, Any]] = None,
) -> Any:
    """Recursively shrinks strings, lists, and optionally wide dicts while tracking omitted items/keys."""
    if isinstance(value, str):
        if len(value) <= max_str_len:
            return value
        if stats is not None:
            stats["truncated_strings"] = stats.get("truncated_strings", 0) + 1
        head = int(max_str_len * 0.75)
        tail = max(0, max_str_len - head)
        omitted = len(value) - (head + tail)
        return f"{value[:head]} ...[TRUNCATED {omitted} chars]... {value[-tail:] if tail else ''}"
    if isinstance(value, dict):
        items = list(value.items())
        if max_dict_keys is not None and len(items) > max_dict_keys:
            kept_items = items[:max_dict_keys]
            if stats is not None:
                stats.setdefault("omitted_keys", []).append({
                    "path": path,
                    "original_key_count": len(items),
                    "returned_key_count": len(kept_items),
                    "omitted_key_count": len(items) - len(kept_items),
                })
        else:
            kept_items = items
        return {
            k: _shrink_json_value(
                v,
                max_str_len=max_str_len,
                max_list_items=max_list_items,
                max_dict_keys=max_dict_keys,
                path=f"{path}.{k}",
                stats=stats,
            )
            for k, v in kept_items
        }
    if isinstance(value, list):
        sliced = value[:max_list_items]
        if stats is not None and len(sliced) < len(value):
            stats.setdefault("omitted_arrays", []).append({
                "path": path,
                "original_count": len(value),
                "returned_count": len(sliced),
                "omitted_count": len(value) - len(sliced),
            })
        return [
            _shrink_json_value(
                item,
                max_str_len=max_str_len,
                max_list_items=max_list_items,
                max_dict_keys=max_dict_keys,
                path=f"{path}[{idx}]",
                stats=stats,
            )
            for idx, item in enumerate(sliced)
        ]
    # Numbers (int, float), booleans, and None are never modified
    return value


def _build_truncation_metadata(
    stats: dict[str, Any],
    *,
    max_entries: int = 8,
    include_hint: bool = True,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "truncated": True,
        "partial_result": True,
    }
    omitted_arrays = stats.get("omitted_arrays") or []
    omitted_keys = stats.get("omitted_keys") or []
    if omitted_arrays:
        if len(omitted_arrays) > max_entries:
            meta["omitted_arrays"] = omitted_arrays[:max_entries]
            meta["omitted_arrays_truncated"] = len(omitted_arrays) - max_entries
        else:
            meta["omitted_arrays"] = omitted_arrays
    if omitted_keys:
        if len(omitted_keys) > max_entries:
            meta["omitted_keys"] = omitted_keys[:max_entries]
            meta["omitted_keys_truncated"] = len(omitted_keys) - max_entries
        else:
            meta["omitted_keys"] = omitted_keys
    if stats.get("truncated_strings"):
        meta["truncated_strings"] = stats["truncated_strings"]
    if include_hint:
        meta["retrieval_hint"] = (
            "Output was truncated to fit context budget; narrow query, request specific keys, "
            "or paginate for omitted records."
        )
    return meta


def _wrap_structured_result(
    raw_obj: Any,
    shrunk: Any,
    stats: dict[str, Any],
    *,
    max_entries: int = 8,
    include_hint: bool = True,
) -> Any:
    has_structural_omission = bool(stats.get("omitted_arrays") or stats.get("omitted_keys"))
    has_any_truncation = has_structural_omission or bool(stats.get("truncated_strings"))
    if not has_any_truncation:
        return shrunk

    meta = _build_truncation_metadata(stats, max_entries=max_entries, include_hint=include_hint)
    if isinstance(shrunk, dict):
        # Preserve original semantics when only a string was shrunk unless arrays/keys were dropped,
        # OR attach completeness metadata when arrays/keys were omitted.
        if has_structural_omission:
            meta_key = "_truncation_metadata" if "_truncation_metadata" not in shrunk else "__tool_truncation_metadata__"
            out = dict(shrunk)
            out[meta_key] = meta
            return out
        return shrunk

    if isinstance(shrunk, list):
        envelope: dict[str, Any] = {
            "items": shrunk,
            "_truncation_metadata": {
                **meta,
                "original_count": len(raw_obj),
                "returned_count": len(shrunk),
                "omitted_count": len(raw_obj) - len(shrunk),
            },
        }
        return envelope

    return shrunk


def _try_structure_aware_json_truncate(raw_obj: Any, effective_max: int) -> Optional[str]:
    """Attempts to serialize and progressively compact a JSON-compatible object within effective_max chars."""
    try:
        compact = json.dumps(raw_obj, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return None

    if len(compact) <= effective_max:
        return compact

    schedule: tuple[tuple[int, int, Optional[int]], ...] = (
        (1500, 50, None),
        (500, 25, None),
        (200, 10, None),
        (80, 5, None),
        (32, 3, None),
        (120, 10, 50),
        (80, 5, 25),
        (48, 3, 12),
        (32, 2, 6),
        (24, 1, 3),
        (16, 1, 1),
    )

    compacted_candidates: list[tuple[Any, dict[str, Any]]] = []
    for max_str_len, max_list_items, max_dict_keys in schedule:
        stats: dict[str, Any] = {"omitted_arrays": [], "omitted_keys": [], "truncated_strings": 0}
        shrunk = _shrink_json_value(
            raw_obj,
            max_str_len=max_str_len,
            max_list_items=max_list_items,
            max_dict_keys=max_dict_keys,
            path="$",
            stats=stats,
        )
        compacted_candidates.append((shrunk, stats))
        for max_entries in (8, 3, 2, 1):
            wrapped = _wrap_structured_result(
                raw_obj, shrunk, stats, max_entries=max_entries, include_hint=True
            )
            try:
                candidate = json.dumps(wrapped, ensure_ascii=False, separators=(",", ":"))
            except Exception:
                continue
            if len(candidate) <= effective_max:
                return candidate

    # Tight-budget fallback: preserve completeness metadata while dropping retrieval_hint only if needed
    for shrunk, stats in compacted_candidates:
        for max_entries in (8, 3, 2, 1, 0):
            wrapped = _wrap_structured_result(
                raw_obj, shrunk, stats, max_entries=max_entries, include_hint=False
            )
            try:
                candidate = json.dumps(wrapped, ensure_ascii=False, separators=(",", ":"))
            except Exception:
                continue
            if len(candidate) <= effective_max:
                return candidate

    # Guaranteed bounded partial-result envelope fallback when data+metadata still exceeded budget
    orig_count = len(raw_obj) if isinstance(raw_obj, (dict, list)) else None
    fallbacks: list[dict[str, Any]] = [
        {
            "_truncation_metadata": {
                "truncated": True,
                "partial_result": True,
                "original_type": type(raw_obj).__name__,
                **({"original_count": orig_count, "returned_count": 0, "omitted_count": orig_count} if orig_count is not None else {}),
                "retrieval_hint": "Payload too large for budget; narrow query or paginate.",
            }
        },
        {
            "_truncation_metadata": {
                "truncated": True,
                "partial_result": True,
                **({"omitted_count": orig_count} if orig_count is not None else {}),
            }
        },
        {"_truncated": True, "_partial": True},
        {"_truncated": True},
    ]
    for fb in fallbacks:
        try:
            candidate = json.dumps(fb, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            continue
        if len(candidate) <= effective_max:
            return candidate

    return None


def truncate_tool_output(
    output: Any,
    max_chars: Optional[int] = None,
    preserve_frontmatter_notes: bool = True,
) -> str:
    """Truncates large tool output preserving head (75%) and tail (25%) tokens for text,
    or structure-aware JSON with completeness metadata for structured payloads.

    Preserves full content if preserve_frontmatter_notes is True and content starts
    with Obsidian YAML Frontmatter containing `entity_type:`.
    """
    if max_chars is not None:
        if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars <= 0:
            raise ValueError(f"max_chars must be a positive integer, got {max_chars!r}")
        effective_max = max_chars
    else:
        env_limit = os.getenv("MAX_TOOL_OUTPUT_CHARS")
        if env_limit is not None and env_limit.strip():
            raw_env = env_limit.strip()
            try:
                effective_max = int(raw_env)
            except ValueError as exc:
                raise ValueError(f"MAX_TOOL_OUTPUT_CHARS must be a positive integer, got {env_limit!r}") from exc
            if effective_max <= 0:
                raise ValueError(f"MAX_TOOL_OUTPUT_CHARS must be a positive integer, got {effective_max!r}")
        else:
            effective_max = DEFAULT_MAX_TOOL_OUTPUT_CHARS

    if output is None:
        return ""

    structured_obj: Any = None
    if isinstance(output, (dict, list)):
        structured_obj = output
        try:
            text = json.dumps(output, ensure_ascii=False)
        except Exception:
            text = str(output)
            structured_obj = None
    else:
        text = str(output)
        stripped = text.strip()
        if len(text) > effective_max and stripped and stripped[0] in ("{", "[") and stripped[-1] in ("}", "]"):
            try:
                parsed_json = json.loads(stripped)
                if isinstance(parsed_json, (dict, list)):
                    structured_obj = parsed_json
            except Exception:
                structured_obj = None

    if preserve_frontmatter_notes and _FRONTMATTER_ENTITY_RE.match(text.strip()):
        return text

    if len(text) <= effective_max:
        return text

    if structured_obj is not None:
        if effective_max < MIN_STRUCTURED_JSON_CHARS:
            raise ValueError(
                f"max_chars={effective_max} is below minimum structured JSON budget ({MIN_STRUCTURED_JSON_CHARS})"
            )
        smart_json = _try_structure_aware_json_truncate(structured_obj, effective_max)
        if smart_json is not None:
            return smart_json
        raise ValueError(
            f"Unable to represent truncated structured JSON within max_chars={effective_max}"
        )

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
        max_signature_calls: Optional[int] = None,
    ) -> None:
        self.max_identical_calls = max_identical_calls
        self.max_consecutive_errors = max_consecutive_errors
        self.max_total_calls = max_total_calls
        self.max_signature_calls = (
            max_signature_calls if max_signature_calls is not None else max_identical_calls + 1
        )

        self.last_call_key: Optional[str] = None
        self.identical_call_count: int = 0
        self.consecutive_error_count: int = 0
        self.total_calls: int = 0
        self.signature_counts: dict[str, int] = {}

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

        sig_count = self.signature_counts.get(call_key, 0) + 1
        self.signature_counts[call_key] = sig_count
        if sig_count > self.max_signature_calls:
            raise RuntimeError(
                f"Runaway guard tripped: cyclic/repeated tool call loop detected for "
                f"'{tool_name}' ({sig_count} total calls with identical arguments across turn)."
            )

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

        has_async = bool(getattr(tool, "coroutine", None)) or (type(tool)._arun is not BaseTool._arun)
        sync_fn = _make_sync_wrapper(tool, is_exempt)
        coroutine_fn = _make_async_wrapper(tool, is_exempt) if has_async else None

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
