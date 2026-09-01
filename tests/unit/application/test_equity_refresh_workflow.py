import json
from unittest.mock import MagicMock, patch
import pytest

from application.equity.refresh_workflow import EquityRefreshPayload, execute_equity_refresh_workflow
from schemas.micro_quant_schemas import (
    AtomicMarketSnapshot,
    EquityNarrativeOutput,
    QuantSignals,
    QuantSignalsFailureResult,
)


def _make_mock_signals():
    return QuantSignals(
        ticker="FTNT",
        market="US",
        currency="USD",
        company_name="Fortinet, Inc.",
        as_of_date="2026-08-27",
        evaluated_at="2026-08-27T10:00:00Z",
        quality_score=76.0,
        atomic_market_snapshot=AtomicMarketSnapshot(
            analysis_price=172.78,
            analysis_price_as_of="2026-08-27",
            price_source="ohlcv_close",
            latest_ohlcv_close=172.78,
            latest_ohlcv_date="2026-08-27",
            shares_outstanding=733713653,
            market_cap=126771044069.69,
            retrieved_at="2026-08-27T10:00:00Z",
        ),
    )


def test_equity_refresh_workflow_llm_unavailable_yields_done_with_warnings():
    job_repo = MagicMock()
    mock_signals = _make_mock_signals()

    with patch("application.equity.refresh_workflow.compute_equity_quant_signals", return_value=mock_signals.model_dump_json()), \
         patch("application.equity.refresh_workflow.check_llm_preflight", return_value=(False, "Connection refused")), \
         patch("application.equity.refresh_workflow._fetch_stock_news", return_value="Mock news text"), \
         patch("application.equity.refresh_workflow.write_equity_sidecar") as mock_sidecar:

        payload = EquityRefreshPayload(ticker="FTNT", market="US", save_to_vault=True, narrative_requested=True)
        status, error_msg = execute_equity_refresh_workflow(payload, job_repo, job_id="job-123")

        assert status == "done_with_warnings"
        assert error_msg == "Numbers refreshed; narrative unavailable"
        assert mock_sidecar.call_count >= 1
        
        # Verify sidecar call received unavailable status
        last_call_output = mock_sidecar.call_args[0][0]
        assert last_call_output.narrative_status == "unavailable"
        assert last_call_output.error_code == "LLM_UNAVAILABLE"
        assert last_call_output.quant_signals.atomic_market_snapshot.analysis_price == 172.78


def test_equity_refresh_workflow_happy_path():
    job_repo = MagicMock()
    mock_signals = _make_mock_signals()
    mock_narrative = EquityNarrativeOutput(
        narrative_analysis="Fortinet demonstrates strong SASE expansion.",
        base_case_summary="Target price $180 based on cash flow visibility.",
    )

    with patch("application.equity.refresh_workflow.compute_equity_quant_signals", return_value=mock_signals.model_dump_json()), \
         patch("application.equity.refresh_workflow.check_llm_preflight", return_value=(True, "Ready")), \
         patch("application.equity.refresh_workflow.get_chat_model", return_value=MagicMock()), \
         patch("agents.equity_synthesizer.invoke_equity_synthesizer", return_value=mock_narrative), \
         patch("application.equity.refresh_workflow._search_vault_memories", return_value="Historical note"), \
         patch("application.equity.refresh_workflow._fetch_stock_news", return_value="Latest earnings beat"), \
         patch("application.equity.refresh_workflow.write_equity_sidecar") as mock_sidecar:

        payload = EquityRefreshPayload(ticker="FTNT", market="US", save_to_vault=True, narrative_requested=True)
        status, error_msg = execute_equity_refresh_workflow(payload, job_repo, job_id="job-123")

        assert status == "done"
        assert error_msg is None
        last_call_output = mock_sidecar.call_args[0][0]
        assert last_call_output.narrative_status == "available"
        assert last_call_output.narrative_analysis == "Fortinet demonstrates strong SASE expansion."


def test_equity_refresh_workflow_quant_failure_yields_error():
    job_repo = MagicMock()
    failure_dto = QuantSignalsFailureResult(
        run_status="error",
        ticker="BADTICKER",
        market="US",
        error_code="DATA_PROVIDER_UNAVAILABLE",
        error_message="Could not fetch quotes",
        evaluated_at="2026-08-27T10:00:00Z",
    )

    with patch("application.equity.refresh_workflow.compute_equity_quant_signals", return_value=failure_dto.model_dump_json()):
        payload = EquityRefreshPayload(ticker="BADTICKER", market="US", save_to_vault=True)
        status, error_msg = execute_equity_refresh_workflow(payload, job_repo, job_id="job-123")

        assert status == "error"
        assert "DATA_PROVIDER_UNAVAILABLE" in error_msg
