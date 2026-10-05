"""LangGraph workflow execution driver for background jobs.

The driver depends on the application job port rather than SQLite or FastAPI.
The API composition root owns the concrete repository and transaction scope.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Optional

from application.jobs.ports import JobRepositoryPort


def _log_manager_messages(job_repo: JobRepositoryPort, job_id: str, event: dict) -> None:
    from langchain_core.messages import HumanMessage
    from core.utils import normalize_content

    for node_name, node_state in event.items():
        if not isinstance(node_state, dict) or "messages" not in node_state:
            continue
        messages = node_state.get("messages")
        if not messages:
            continue
        if not isinstance(messages, list):
            messages = [messages]
        for last in messages:
            content = normalize_content(getattr(last, "content", ""))
            if not content:
                continue
            if isinstance(last, HumanMessage):
                sender = getattr(last, "name", None) or "manager"
                role = "instruction"
                label = f"{sender} → {node_name}"
            else:
                role = "reply"
                label = node_name
            job_repo.append_job_log(job_id, node_name, content, role=role, label=label)


def _append_manager_summary(job_repo: JobRepositoryPort, job_id: str, instruction: str, flow: str) -> None:
    if flow != "manager":
        return
    from agents.manager_agent import generate_manager_summary

    reply_logs = job_repo.get_job_reply_logs(job_id)
    if any(row["node_name"] == "manager_summary" for row in reply_logs):
        return
    excluded_nodes = {"supervisor", "manager_summary"}
    deliverables = [
        (row["node_name"] or "Specialist", row["content"] or "")
        for row in reply_logs
        if row["node_name"] not in excluded_nodes
        and not (row["node_name"] or "").startswith(("post_", "prepare_"))
    ]
    if not deliverables:
        deliverables = [
            (row["node_name"] or "Manager", row["content"] or "")
            for row in reply_logs
            if row["node_name"] == "supervisor"
        ]
    summary = generate_manager_summary(instruction, deliverables)
    if summary:
        job_repo.append_job_log(job_id, "manager_summary", summary, role="reply", label="Manager Summary")


def run_job_workflow(
    *,
    state: JobRepositoryPort,
    job_id: str,
    thread_id: str,
    instruction: str,
    flow: str = "manager",
    scope: str = "both",
    resume_value: Optional[dict[str, Any]] = None,
) -> None:
    """Run one graph and persist logs/status through the injected port."""
    from langgraph.checkpoint.sqlite import SqliteSaver
    from core.retry import with_retry

    checkpoint_path = os.getenv("CHECKPOINT_DB_PATH", "data/checkpoints.sqlite")
    with SqliteSaver.from_conn_string(checkpoint_path) as checkpointer:
        terminal_status: Optional[str] = None
        terminal_error: Optional[str] = None

        if flow == "equity_refresh":
            from application.equity.refresh_workflow import EquityRefreshPayload, execute_equity_refresh_workflow
            try:
                if instruction.strip().startswith("{"):
                    payload_dict = json.loads(instruction)
                    refresh_payload = EquityRefreshPayload.model_validate(payload_dict)
                else:
                    ticker = instruction.strip().upper()
                    market = "TH" if ticker.endswith(".BK") else "US"
                    refresh_payload = EquityRefreshPayload(ticker=ticker, market=market)
            except Exception:
                ticker = instruction.strip().upper()
                market = "TH" if ticker.endswith(".BK") else "US"
                refresh_payload = EquityRefreshPayload(ticker=ticker, market=market)

            terminal_status, terminal_error = execute_equity_refresh_workflow(
                payload=refresh_payload,
                job_repo=state,
                job_id=job_id,
            )
            if terminal_status:
                state.update_job_status(job_id, terminal_status, error_message=terminal_error)
            return

        if flow == "news_youtube":
            from agents.news_youtube_flow import build_news_youtube_graph
            graph = build_news_youtube_graph(checkpointer=checkpointer)
            fresh_inputs: dict[str, Any] = {"scope": scope}
        elif flow == "news_funnel":
            from agents.news_funnel_flow import build_news_funnel_graph
            graph = build_news_funnel_graph(checkpointer=checkpointer)
            fresh_inputs = {}
        elif flow == "youtube_pitch":
            from agents.youtube_pitch_flow import build_youtube_pitch_graph
            graph = build_youtube_pitch_graph(checkpointer=checkpointer)
            fresh_inputs = {"instruction": instruction}
        else:
            from agents.manager_agent import build_graph
            graph = build_graph(checkpointer=checkpointer)
            fresh_inputs = {
                "messages": [("user", instruction)],
                "macro_run_id": job_id,
                "macro_run_started_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "macro_task_run_id": None,
                "macro_task_started_at": None,
                "macro_task_sequence": 0,
                "sector_rotation_snapshot_id": None,
                "sector_rotation_context": None,
                "sector_context_status": None,
                "quant_raw": None,
                "quant_score": None,
                "narrative_raw": None,
                "narrative_context": None,
                "task_queue": [],
                "replan_count": 0,
                "route_meta": {},
            }

        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 40,
            "tags": ["invest-agents", "web-session", flow],
            "metadata": {"run_type": "chain", "session_source": "web", "job_id": job_id},
        }
        if resume_value is not None:
            from langgraph.types import Command
            stream_input = Command(resume=resume_value)
        else:
            stream_input = fresh_inputs

        def _stream_and_log() -> None:
            nonlocal terminal_status, terminal_error
            for event in graph.stream(stream_input, config=config, stream_mode="updates"):
                if "__interrupt__" in event:
                    payload = event["__interrupt__"][0].value
                    state.set_job_awaiting_approval(job_id, json.dumps(payload, ensure_ascii=False))
                    return
                _log_manager_messages(state, job_id, event)
                if flow == "manager":
                    synth_event = event.get("equity_synthesizer")
                    if isinstance(synth_event, dict):
                        synth_out = synth_event.get("equity_output") or {}
                        if "LLM_UNAVAILABLE" in synth_out.get("narrative_analysis", "") or synth_out.get("narrative_status") == "unavailable":
                            terminal_status = "done_with_warnings"
                            terminal_error = "Numbers refreshed; narrative unavailable"
                if flow == "youtube_pitch":
                    for node_name in ("synthesize_notebooklm", "persist_parking_lot"):
                        node_update = event.get(node_name)
                        if not isinstance(node_update, dict):
                            continue
                        status = node_update.get("synthesis_status")
                        if status == "done_with_errors":
                            failures = node_update.get("synthesis_failures") or []
                            terminal_status = "done_with_errors"
                            terminal_error = "\n".join(str(item) for item in failures) or "Some approved pitches failed"
                        elif status == "done_with_warnings" and terminal_status != "done_with_errors":
                            warnings = node_update.get("synthesis_warnings") or []
                            terminal_status = "done_with_warnings"
                            terminal_error = "\n".join(str(item) for item in warnings) or "Completed with warnings (Unverified Drafts or Parking Lot partial notices)"
            _append_manager_summary(state, job_id, instruction, flow)

        with_retry(_stream_and_log)
        if terminal_status:
            state.update_job_status(job_id, terminal_status, error_message=terminal_error)


__all__ = ["run_job_workflow"]
