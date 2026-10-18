"""General-purpose Deterministic Verification & Self-Correction Loop for Agent Workers.

Architecture Layer: core/
Dependencies: Standard Library + pydantic + langchain_core.messages (No API/Adapter imports).
Complies with 16 AST Dependency Rules in tests/architecture/test_dependency_rules.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import re
from typing import Any, Callable, Optional, Sequence, TypeVar

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, ValidationError

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


@dataclass
class VerificationOutcome:
    """Result of a deterministic verification check on an agent's output or trajectory."""

    ok: bool
    cleaned_text: str = ""
    parsed_model: Any = None
    errors: list[str] = field(default_factory=list)

    @property
    def feedback_message(self) -> str:
        bullet_list = "\n".join(f"  - {err}" for err in self.errors)
        return (
            "[SYSTEM DETERMINISTIC VERIFICATION FAILED - SELF-CORRECTION REQUIRED]\n"
            "ผลลัพธ์ของคุณไม่ผ่านการตรวจสอบความถูกต้องเชิงโครงสร้างดังต่อไปนี้:\n"
            f"{bullet_list}\n\n"
            "กรุณาแก้ไขเฉพาะส่วนที่ผิดพลาดและส่งผลลัพธ์ที่ถูกต้องกลับมาทันที "
            "(ไม่ต้องเรียกเครื่องมือดึงข้อมูลซ้ำหากมีข้อมูลในบริบทแล้ว)"
        )


def strip_json_envelope(raw_text: str) -> str:
    """Strips markdown code fences and surrounding prose around a JSON object."""
    cleaned = _CODE_FENCE_RE.sub("", (raw_text or "").strip()).strip()
    if not cleaned.startswith("{") and "{" in cleaned and "}" in cleaned:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if 0 <= start < end:
            cleaned = cleaned[start : end + 1]
    return cleaned


def extract_and_validate_json_schema(raw_text: Any, schema_cls: type[T]) -> VerificationOutcome:
    """Extracts JSON from raw LLM text or dict and validates it against a Pydantic schema."""
    if isinstance(raw_text, dict):
        try:
            validated = schema_cls.model_validate(raw_text)
            return VerificationOutcome(
                ok=True,
                cleaned_text=validated.model_dump_json(),
                parsed_model=validated,
            )
        except ValidationError as val_err:
            err_details = [
                f"ฟิลด์ '{'.'.join(str(loc) for loc in err['loc'])}': {err['msg']}"
                for err in val_err.errors()[:5]
            ]
            return VerificationOutcome(ok=False, cleaned_text=str(raw_text), errors=err_details)

    cleaned = strip_json_envelope(str(raw_text or ""))
    if not cleaned:
        return VerificationOutcome(
            ok=False,
            cleaned_text="",
            errors=[f"ผลลัพธ์ว่างเปล่า กรุณาตอบกลับเป็น JSON Object ตามโครงสร้าง {schema_cls.__name__}"],
        )

    try:
        data = json.loads(cleaned)
    except Exception as exc:
        return VerificationOutcome(
            ok=False,
            cleaned_text=cleaned,
            errors=[
                f"รูปแบบไม่ใช่ JSON ที่ถูกต้อง ({exc}) กรุณาตอบกลับเป็น JSON Object ล้วนๆ "
                f"ตามโครงสร้าง {schema_cls.__name__} โดยไม่ต้องใส่ Markdown ครอบ"
            ],
        )

    try:
        validated = schema_cls.model_validate(data)
        return VerificationOutcome(
            ok=True,
            cleaned_text=validated.model_dump_json(),
            parsed_model=validated,
        )
    except ValidationError as val_err:
        err_details = [
            f"ฟิลด์ '{'.'.join(str(loc) for loc in err['loc'])}': {err['msg']}"
            for err in val_err.errors()[:5]
        ]
        return VerificationOutcome(
            ok=False,
            cleaned_text=cleaned,
            errors=err_details,
        )


def verify_required_tool_called(
    messages: Sequence[Any],
    required_tool_names: set[str],
    *,
    min_calls: int = 1,
) -> VerificationOutcome:
    """Verifies deterministically that at least `min_calls` to `required_tool_names` occurred."""
    matching_calls = [
        m
        for m in messages
        if isinstance(m, ToolMessage) and getattr(m, "name", None) in required_tool_names
    ]
    if len(matching_calls) >= min_calls:
        return VerificationOutcome(ok=True)

    expected_str = ", ".join(sorted(required_tool_names))
    if min_calls > 1:
        err_msg = (
            f"คุณเรียกใช้เครื่องมือ ({expected_str}) เพียง {len(matching_calls)}/{min_calls} ครั้ง "
            f"กรุณาเรียกใช้เครื่องมือให้ครบตามคำสั่งก่อนจบการทำงาน"
        )
    else:
        err_msg = (
            f"คุณยังไม่ได้เรียกใช้เครื่องมือที่กำหนด ({expected_str}) "
            f"ห้ามตอบกลับด้วยข้อความเปล่าโดยไม่เรียกใช้เครื่องมือจริง"
        )
    return VerificationOutcome(ok=False, errors=[err_msg])


def has_human_turn_context(messages: Sequence[Any]) -> bool:
    """Returns True if `messages` contains at least one HumanMessage (real LangGraph trajectory).

    Used to distinguish real ReAct agent runs (`[HumanMessage, AIMessage, ...]`) from synthetic
    1-message unit test mocks (`MockGraph({"messages": [AIMessage(...)]})`).
    """
    return any(isinstance(m, HumanMessage) for m in messages)


def invoke_react_with_verification(
    agent_graph: Any,
    initial_messages: list[Any],
    config: Optional[dict[str, Any]],
    *,
    extract_reply_fn: Callable[[list[Any]], str],
    schema_cls: Optional[type[BaseModel]] = None,
    required_tools: Optional[set[str]] = None,
    min_tool_calls: int = 1,
    max_self_corrections: int = 1,
) -> tuple[dict[str, Any], str]:
    """Invokes a ReAct agent graph and runs up to `max_self_corrections` in-place repair turns."""
    result = agent_graph.invoke({"messages": initial_messages}, config=config)
    messages = list(result.get("messages") or [])
    reply = extract_reply_fn(messages) if messages else ""

    # Short-circuit if the worker returned an explicit system/tool error string
    if reply.strip().startswith("Error:") or reply.strip().startswith("Error "):
        return result, reply

    # 1. If schema_cls is provided, always try stripping markdown fences & validating first
    if schema_cls is not None:
        outcome = extract_and_validate_json_schema(reply, schema_cls)

        # Deterministic ToolMessage Fallback: if LLM echo failed validation, inspect ToolMessages
        if not outcome.ok:
            for m in reversed(messages):
                if isinstance(m, ToolMessage) and m.content:
                    fallback = extract_and_validate_json_schema(m.content, schema_cls)
                    if fallback.ok:
                        log.info(
                            "Recovered valid %s directly from ToolMessage output (%s).",
                            schema_cls.__name__,
                            getattr(m, "name", "tool"),
                        )
                        outcome = fallback
                        reply = outcome.cleaned_text
                        break

        if outcome.ok:
            reply = outcome.cleaned_text
            if not required_tools:
                return result, reply
        elif has_human_turn_context(messages) and max_self_corrections > 0:
            for _ in range(max_self_corrections):
                log.warning(
                    "Verification failed for %s (%s); triggering in-place self-correction.",
                    schema_cls.__name__,
                    "; ".join(outcome.errors),
                )
                repair_messages = messages + [
                    HumanMessage(content=outcome.feedback_message, name="manager")
                ]
                result = agent_graph.invoke({"messages": repair_messages}, config=config)
                messages = list(result.get("messages") or [])
                reply = extract_reply_fn(messages) if messages else ""
                outcome = extract_and_validate_json_schema(reply, schema_cls)
                if outcome.ok:
                    reply = outcome.cleaned_text
                    break
                for m in reversed(messages):
                    if isinstance(m, ToolMessage) and m.content:
                        fallback = extract_and_validate_json_schema(m.content, schema_cls)
                        if fallback.ok:
                            outcome = fallback
                            reply = outcome.cleaned_text
                            break
                if outcome.ok:
                    break
        return result, reply

    # 2. If required_tools is specified (e.g. Archivist must call write_raw_markdown)
    if required_tools and has_human_turn_context(messages) and max_self_corrections > 0:
        tool_outcome = verify_required_tool_called(
            messages, required_tools, min_calls=min_tool_calls
        )
        if not tool_outcome.ok:
            for _ in range(max_self_corrections):
                log.warning(
                    "Required tool verification failed (%s); triggering in-place self-correction.",
                    "; ".join(tool_outcome.errors),
                )
                repair_messages = messages + [
                    HumanMessage(content=tool_outcome.feedback_message, name="manager")
                ]
                result = agent_graph.invoke({"messages": repair_messages}, config=config)
                messages = list(result.get("messages") or [])
                reply = extract_reply_fn(messages) if messages else ""
                tool_outcome = verify_required_tool_called(
                    messages, required_tools, min_calls=min_tool_calls
                )
                if tool_outcome.ok:
                    break

    return result, reply
