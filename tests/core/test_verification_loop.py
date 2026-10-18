"""Unit tests for core/verification_loop.py.

Architecture Layer: tests/core/
Tests deterministic schema validation, markdown fence stripping, tool verification,
and in-place ReAct self-correction loops.
"""
from typing import Any
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field
import pytest

from core.verification_loop import (
    VerificationOutcome,
    extract_and_validate_json_schema,
    has_human_turn_context,
    invoke_react_with_verification,
    strip_json_envelope,
    verify_required_tool_called,
)


class SampleSchema(BaseModel):
    ticker: str
    score: float = Field(ge=0.0, le=100.0)
    recommendation: str


class TestStripJsonEnvelope:
    def test_clean_json(self):
        raw = '{"ticker": "NVDA", "score": 85.5}'
        assert strip_json_envelope(raw) == raw

    def test_markdown_code_fences(self):
        raw = '```json\n{"ticker": "NVDA", "score": 85.5}\n```'
        assert strip_json_envelope(raw) == '{"ticker": "NVDA", "score": 85.5}'

    def test_markdown_code_fences_without_json_tag(self):
        raw = '```\n{"ticker": "NVDA", "score": 85.5}\n```'
        assert strip_json_envelope(raw) == '{"ticker": "NVDA", "score": 85.5}'

    def test_conversational_envelope(self):
        raw = 'Here is the requested output:\n{"ticker": "NVDA", "score": 85.5}\nHope this helps!'
        assert strip_json_envelope(raw) == '{"ticker": "NVDA", "score": 85.5}'

    def test_empty_string(self):
        assert strip_json_envelope("") == ""
        assert strip_json_envelope("   ") == ""


class TestExtractAndValidateJsonSchema:
    def test_valid_json(self):
        raw = '{"ticker": "AAPL", "score": 90.0, "recommendation": "BUY"}'
        outcome = extract_and_validate_json_schema(raw, SampleSchema)
        assert outcome.ok is True
        assert isinstance(outcome.parsed_model, SampleSchema)
        assert outcome.parsed_model.ticker == "AAPL"
        assert outcome.parsed_model.score == 90.0

    def test_invalid_json_syntax(self):
        raw = '{"ticker": "AAPL", "score": '
        outcome = extract_and_validate_json_schema(raw, SampleSchema)
        assert outcome.ok is False
        assert len(outcome.errors) > 0
        assert "ไม่ใช่ JSON ที่ถูกต้อง" in outcome.errors[0]

    def test_schema_validation_failure(self):
        # score > 100 violates constraint
        raw = '{"ticker": "AAPL", "score": 150.0, "recommendation": "BUY"}'
        outcome = extract_and_validate_json_schema(raw, SampleSchema)
        assert outcome.ok is False
        assert any("score" in err for err in outcome.errors)

    def test_missing_required_field(self):
        raw = '{"ticker": "AAPL"}'
        outcome = extract_and_validate_json_schema(raw, SampleSchema)
        assert outcome.ok is False
        assert any("score" in err or "recommendation" in err for err in outcome.errors)

    def test_feedback_message_formatting(self):
        outcome = VerificationOutcome(ok=False, errors=["ฟิลด์ 'score': required"])
        msg = outcome.feedback_message
        assert "[SYSTEM DETERMINISTIC VERIFICATION FAILED - SELF-CORRECTION REQUIRED]" in msg
        assert "ฟิลด์ 'score': required" in msg


class TestVerifyRequiredToolCalled:
    def test_tool_called_successfully(self):
        messages = [
            HumanMessage(content="run analysis"),
            ToolMessage(content="ok", name="write_raw_markdown", tool_call_id="call_1"),
        ]
        outcome = verify_required_tool_called(messages, {"write_raw_markdown"})
        assert outcome.ok is True

    def test_tool_not_called(self):
        messages = [
            HumanMessage(content="run analysis"),
            AIMessage(content="I am done"),
        ]
        outcome = verify_required_tool_called(messages, {"write_raw_markdown"})
        assert outcome.ok is False
        assert "write_raw_markdown" in outcome.errors[0]

    def test_min_calls_enforcement(self):
        messages = [
            HumanMessage(content="run analysis"),
            ToolMessage(content="1", name="write_raw_markdown", tool_call_id="call_1"),
        ]
        outcome = verify_required_tool_called(messages, {"write_raw_markdown"}, min_calls=2)
        assert outcome.ok is False
        assert "1/2" in outcome.errors[0]


class TestHasHumanTurnContext:
    def test_with_human_message(self):
        assert has_human_turn_context([HumanMessage(content="hello"), AIMessage(content="hi")]) is True

    def test_without_human_message(self):
        assert has_human_turn_context([AIMessage(content="hi")]) is False
        assert has_human_turn_context([]) is False


class MockAgentGraph:
    def __init__(self, responses: list[list[Any]]):
        self.responses = responses
        self.call_count = 0
        self.invoked_messages: list[list[Any]] = []

    def invoke(self, state: dict[str, Any], config: Any = None) -> dict[str, Any]:
        self.call_count += 1
        msgs = state.get("messages", [])
        self.invoked_messages.append(list(msgs))
        response = self.responses[min(self.call_count - 1, len(self.responses) - 1)]
        return {"messages": list(msgs) + response}


def _dummy_extract_reply(messages: list[Any]) -> str:
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            return str(m.content)
    return ""


class TestInvokeReactWithVerification:
    def test_success_on_first_turn(self):
        graph = MockAgentGraph([
            [AIMessage(content='{"ticker": "NVDA", "score": 95.0, "recommendation": "BUY"}')]
        ])
        initial = [HumanMessage(content="evaluate NVDA")]
        result, reply = invoke_react_with_verification(
            graph,
            initial,
            config=None,
            extract_reply_fn=_dummy_extract_reply,
            schema_cls=SampleSchema,
        )
        assert graph.call_count == 1
        assert "NVDA" in reply

    def test_self_correction_on_broken_schema(self):
        # First attempt: invalid score 150. Second attempt (after feedback): valid score 80.
        graph = MockAgentGraph([
            [AIMessage(content='{"ticker": "NVDA", "score": 150.0, "recommendation": "BUY"}')],
            [AIMessage(content='{"ticker": "NVDA", "score": 80.0, "recommendation": "BUY"}')],
        ])
        initial = [HumanMessage(content="evaluate NVDA")]
        result, reply = invoke_react_with_verification(
            graph,
            initial,
            config=None,
            extract_reply_fn=_dummy_extract_reply,
            schema_cls=SampleSchema,
            max_self_corrections=1,
        )
        assert graph.call_count == 2
        assert "80.0" in reply
        # Check that repair message had verification feedback
        second_call_input = graph.invoked_messages[1]
        assert any("SELF-CORRECTION REQUIRED" in str(m.content) for m in second_call_input)

    def test_short_circuits_on_error_message(self):
        graph = MockAgentGraph([
            [AIMessage(content="Error: Tool execution failed.")]
        ])
        initial = [HumanMessage(content="run")]
        result, reply = invoke_react_with_verification(
            graph,
            initial,
            config=None,
            extract_reply_fn=_dummy_extract_reply,
            schema_cls=SampleSchema,
        )
        assert graph.call_count == 1
        assert reply == "Error: Tool execution failed."

    def test_synthetic_mock_without_human_message_does_not_loop(self):
        # When messages has no HumanMessage (e.g. 1-element AIMessage test fixture),
        # it should NOT trigger a retry loop.
        graph = MockAgentGraph([
            [AIMessage(content="not valid json")]
        ])
        initial = [AIMessage(content="mock previous")]
        result, reply = invoke_react_with_verification(
            graph,
            initial,
            config=None,
            extract_reply_fn=_dummy_extract_reply,
            schema_cls=SampleSchema,
        )
        assert graph.call_count == 1

    def test_recovers_schema_from_tool_message_when_ai_message_truncated(self):
        valid_json = '{"ticker": "NVDA", "score": 92.5, "recommendation": "BUY"}'
        tool_msg = ToolMessage(content=valid_json, tool_call_id="call_1", name="compute_signals")
        # LLM truncates or returns non-JSON commentary in AIMessage
        ai_msg = AIMessage(content="Here is the valuation report for NVDA: [TRUNCATED]...")
        graph = MockAgentGraph([
            [tool_msg, ai_msg]
        ])
        initial = [HumanMessage(content="analyze NVDA")]
        result, reply = invoke_react_with_verification(
            graph,
            initial,
            config=None,
            extract_reply_fn=_dummy_extract_reply,
            schema_cls=SampleSchema,
        )
        assert graph.call_count == 1
        assert "NVDA" in reply
        assert "92.5" in reply
