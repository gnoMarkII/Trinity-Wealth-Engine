"""Unit tests for LangGraph checkpoint resumption in agents/job_runner.py.

Architecture Layer: tests/unit/
Tests _resolve_stream_input and verifies that partially completed workflows
resume from checkpoint without duplicating previous stage executions.
"""
from typing import Any, Optional, TypedDict
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
import pytest

from agents.job_runner import _resolve_stream_input


class ResumeState(TypedDict):
    step1_runs: int
    step2_runs: int
    data: str


def test_resolve_stream_input_with_resume_value():
    """When resume_value is provided, it must wrap it in Command(resume=resume_value)."""
    resume_val = {"approved": True}
    resolved = _resolve_stream_input(
        graph=None,
        config={},
        fresh_inputs={"instruction": "test"},
        resume_value=resume_val,
    )
    assert isinstance(resolved, Command)
    assert resolved.resume == resume_val


def test_resolve_stream_input_fresh_when_no_checkpoint(tmp_path):
    """When checkpointer has no previous state for thread_id, returns fresh_inputs."""
    db_file = tmp_path / "checkpoints.sqlite"
    with SqliteSaver.from_conn_string(str(db_file)) as checkpointer:
        builder = StateGraph(ResumeState)
        builder.add_node("step1", lambda s: {"step1_runs": s.get("step1_runs", 0) + 1})
        builder.add_edge(START, "step1")
        builder.add_edge("step1", END)
        graph = builder.compile(checkpointer=checkpointer)

        config = {"configurable": {"thread_id": "thread-fresh-1"}}
        fresh = {"step1_runs": 0, "step2_runs": 0, "data": "initial"}
        resolved = _resolve_stream_input(graph, config, fresh, resume_value=None)
        assert resolved == fresh


def test_resolve_stream_input_resumes_from_checkpoint(tmp_path):
    """When checkpointer has pending next nodes, _resolve_stream_input returns None."""
    db_file = tmp_path / "checkpoints.sqlite"
    with SqliteSaver.from_conn_string(str(db_file)) as checkpointer:
        builder = StateGraph(ResumeState)

        step1_exec_count = 0
        step2_exec_count = 0

        def node_1(state: ResumeState) -> dict[str, Any]:
            nonlocal step1_exec_count
            step1_exec_count += 1
            return {"step1_runs": step1_exec_count, "data": "step1_done"}

        def node_2(state: ResumeState) -> dict[str, Any]:
            nonlocal step2_exec_count
            step2_exec_count += 1
            return {"step2_runs": step2_exec_count, "data": "all_done"}

        builder.add_node("step1", node_1)
        builder.add_node("step2", node_2)
        builder.add_edge(START, "step1")
        builder.add_edge("step1", "step2")
        builder.add_edge("step2", END)

        # Interrupt after step1 to create a partial checkpoint
        graph = builder.compile(checkpointer=checkpointer, interrupt_after=["step1"])

        config = {"configurable": {"thread_id": "thread-resume-1"}}
        initial_input = {"step1_runs": 0, "step2_runs": 0, "data": "fresh"}

        # Run first step until interrupt
        events = list(graph.stream(initial_input, config=config))
        assert step1_exec_count == 1
        assert step2_exec_count == 0

        # Check graph state has pending next
        state_snapshot = graph.get_state(config)
        assert "step2" in state_snapshot.next

        # _resolve_stream_input must now return None (LangGraph resume signal)
        active_input = _resolve_stream_input(graph, config, initial_input, resume_value=None)
        assert active_input is None

        # Re-compile graph without interruption to finish execution
        resumed_graph = builder.compile(checkpointer=checkpointer)

        # Resuming with None runs step2 without re-running step1
        events_resume = list(resumed_graph.stream(active_input, config=config))
        assert step1_exec_count == 1  # Step 1 was NOT re-executed
        assert step2_exec_count == 1  # Step 2 was executed cleanly

        final_state = resumed_graph.get_state(config)
        assert final_state.values["data"] == "all_done"
