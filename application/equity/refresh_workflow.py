"""Dedicated deterministic Equity Refresh Workflow.

Bypasses the LangGraph manager/ReAct router entirely to ensure deterministic
quantitative calculation, immediate numeric persistence, and decoupled optional LLM narrative.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Literal, Optional, Tuple

from pydantic import BaseModel, Field

from application.jobs.ports import JobRepositoryPort
from core.llm_factory import check_llm_preflight, classify_llm_exception, get_chat_model
from schemas.micro_quant_schemas import (
    MicroQuantOutput,
    QuantSignals,
    QuantSignalsFailureResult,
    build_unavailable_sentiment_context,
)
from tools.market.equity_quant_tool import compute_equity_quant_signals
from tools.market.equity_sidecar import write_equity_sidecar

log = logging.getLogger(__name__)


class EquityRefreshPayload(BaseModel):
    ticker: str
    market: Literal["TH", "US"] = "US"
    save_to_vault: bool = True
    narrative_requested: bool = True


def _fetch_stock_news(ticker: str, market: str) -> str:
    try:
        from tools.market.news import ingest_stock_news
        return ingest_stock_news.invoke({"ticker": ticker, "market": market})
    except Exception as e:
        return f"Error fetching news: {e}"


def _search_vault_memories(keyword: str) -> str:
    try:
        from tools.archivist.search import search_all_memories
        return search_all_memories.invoke({"keyword": keyword})
    except Exception as e:
        return f"Error searching vault: {e}"


def execute_equity_refresh_workflow(
    payload: EquityRefreshPayload,
    job_repo: JobRepositoryPort,
    job_id: str,
) -> Tuple[str, Optional[str]]:
    """Execute the 5-phase deterministic equity refresh pipeline.

    Phases:
    1. equity_quant: Compute deterministic Python quant signals & snapshot.
    2. persist_numeric: Atomically write numeric revision sidecar with narrative_status="pending".
    3. equity_news: Ingest raw news evidence.
    4. equity_narrative: Synthesize LLM narrative if requested & available.
    5. persist_final: Update sidecar with final narrative status & error code.

    Returns:
        tuple (terminal_status, terminal_error) e.g. ("done", None) or ("done_with_warnings", "...")
    """
    clean_ticker = payload.ticker.strip().upper()
    market = payload.market

    # -------------------------------------------------------------
    # Phase 1: equity_quant
    # -------------------------------------------------------------
    job_repo.append_job_log(
        job_id, "equity_quant",
        f"Starting deterministic quantitative analysis for {clean_ticker} ({market})...",
        role="instruction", label="Equity Quant"
    )

    quant_raw_json = compute_equity_quant_signals(ticker=clean_ticker, market=market)
    try:
        quant_data = json.loads(quant_raw_json)
    except Exception as e:
        err_msg = f"Failed to parse quant output JSON: {e}"
        job_repo.append_job_log(job_id, "equity_quant", err_msg, role="reply", label="Quant Parse Error")
        return "error", err_msg

    if quant_data.get("run_status") == "error":
        err_code = quant_data.get("error_code") or "QUANT_EXECUTION_ERROR"
        err_msg = quant_data.get("error_message") or "Quant execution error"
        job_repo.append_job_log(job_id, "equity_quant", f"Quant failed [{err_code}]: {err_msg}", role="reply", label="Quant Error")
        return "error", f"{err_code}: {err_msg}"

    try:
        signals = QuantSignals.model_validate(quant_data)
    except Exception as e:
        err_msg = f"Schema validation failed for QuantSignals: {e}"
        job_repo.append_job_log(job_id, "equity_quant", err_msg, role="reply", label="Validation Error")
        return "error", err_msg

    analysis_price_str = f"${signals.atomic_market_snapshot.analysis_price}" if signals.atomic_market_snapshot else "N/A"
    job_repo.append_job_log(
        job_id, "equity_quant",
        f"Quant analysis complete for {clean_ticker}: Price={analysis_price_str}",
        role="reply", label="Equity Quant Complete"
    )

    # -------------------------------------------------------------
    # Phase 2: persist_numeric
    # -------------------------------------------------------------
    run_id = signals.evidence_snapshot.metadata.analysis_run_id if signals.evidence_snapshot else f"run_{clean_ticker}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    ev_hash = signals.evidence_snapshot.metadata.snapshot_sha256 if signals.evidence_snapshot else None
    as_of_date = (
        signals.atomic_market_snapshot.analysis_price_as_of
        if signals.atomic_market_snapshot
        else signals.evaluated_at[:10]
    )

    provisional_output = MicroQuantOutput(
        ticker=clean_ticker,
        market=market,
        analysis_date=as_of_date,
        quant_signals=signals,
        sentiment_context=build_unavailable_sentiment_context(evaluated_at=signals.evaluated_at),
        narrative_analysis="",
        base_case_summary="",
        narrative_status="pending",
        error_code=None,
        numeric_revision_id=run_id,
        evidence_snapshot_hash=ev_hash,
        piotroski_breakdown=signals.piotroski_breakdown,
        reverse_dcf_result=signals.reverse_dcf_result,
        tactical_setup=signals.tactical_setup,
        insider_conviction=signals.insider_conviction,
        earnings_guidance_context=signals.earnings_guidance_context,
        deterministic_scorecard=signals.deterministic_scorecard,
        thesis_falsifiers=signals.thesis_falsifiers,
        evidence_snapshot=signals.evidence_snapshot,
    )

    if payload.save_to_vault:
        try:
            write_equity_sidecar(provisional_output)
            job_repo.append_job_log(
                job_id, "persist_numeric",
                f"Numeric revision published successfully ({run_id}) | Narrative pending",
                role="reply", label="Persist Numeric"
            )
        except Exception as e:
            log.warning("Failed to persist provisional sidecar: %s", e)
            job_repo.append_job_log(
                job_id, "persist_numeric",
                f"Warning: Sidecar numeric persist error: {e}",
                role="reply", label="Persist Warning"
            )

    if not payload.narrative_requested:
        provisional_output.narrative_status = "unavailable"
        if payload.save_to_vault:
            write_equity_sidecar(provisional_output)
        return "done", None

    # -------------------------------------------------------------
    # Phase 3: equity_news (Fetch raw news)
    # -------------------------------------------------------------
    news_text = _fetch_stock_news(clean_ticker, market)

    # -------------------------------------------------------------
    # Phase 4 & 5: equity_narrative & persist_final
    # -------------------------------------------------------------
    is_llm_ready, preflight_msg = check_llm_preflight()
    if not is_llm_ready:
        log.info("LLM preflight failed for narrative generation: %s", preflight_msg)
        provisional_output.narrative_status = "unavailable"
        provisional_output.error_code = "LLM_UNAVAILABLE"
        if payload.save_to_vault:
            write_equity_sidecar(provisional_output)
        job_repo.append_job_log(
            job_id, "equity_narrative",
            f"LLM endpoint unavailable: {preflight_msg}. Numeric snapshot preserved.",
            role="reply", label="Equity Narrative Unavailable"
        )
        return "done_with_warnings", "Numbers refreshed; narrative unavailable"

    try:
        from agents.equity_synthesizer import invoke_equity_synthesizer

        company_name = signals.company_name or clean_ticker
        search_keyword = f"{clean_ticker} {company_name}".strip()
        vault_text = _search_vault_memories(search_keyword)

        synth_model = get_chat_model("EQUITY_SYNTHESIZER_MODEL", default="gemini-3.1-flash-lite-preview")
        quant_json = signals.model_dump_json()
        narrative_json = json.dumps({
            "ticker": clean_ticker,
            "vault_context": vault_text,
            "news_context": news_text,
        }, ensure_ascii=False)

        narrative_res = invoke_equity_synthesizer(synth_model, quant_json=quant_json, narrative_json=narrative_json)

        final_output = provisional_output.model_copy()
        final_output.narrative_analysis = narrative_res.narrative_analysis
        final_output.base_case_summary = narrative_res.base_case_summary
        final_output.narrative_status = "available"
        final_output.error_code = None

        if payload.save_to_vault:
            write_equity_sidecar(final_output)

        job_repo.append_job_log(
            job_id, "equity_narrative",
            "Equity narrative synthesis complete and merged with numeric revision.",
            role="reply", label="Equity Narrative Complete"
        )
        return "done", None
    except Exception as e:
        err_code, err_desc = classify_llm_exception(e)
        log.warning("Equity narrative synthesis failed [%s]: %s", err_code, e)

        failed_output = provisional_output.model_copy()
        failed_output.narrative_status = "unavailable"
        failed_output.error_code = err_code

        if payload.save_to_vault:
            write_equity_sidecar(failed_output)

        job_repo.append_job_log(
            job_id, "equity_narrative",
            f"Narrative synthesis failed [{err_code}]: {err_desc}. Numeric snapshot preserved.",
            role="reply", label="Equity Narrative Error"
        )
        return "done_with_warnings", f"Numbers refreshed; narrative unavailable ({err_code})"
