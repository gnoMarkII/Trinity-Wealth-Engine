"""Inbound HTTP adapter for Earnings Call Summarization & Kanban Dispatch."""
from fastapi import APIRouter, Depends, HTTPException, Response, status

from api.schemas.equity import (
    EarningsCallSummarizeRequest,
    EarningsCallSummarizeResponse,
    EarningsCallRunResponse,
    EarningsCallListResponse,
    EarningsCallNoteItem,
)
from api.dependencies import get_earnings_call_service
from application.earnings_call.dto import (
    EarningsCallSummarizeRequestDTO,
    EarningsCallRunDTO,
)
from application.earnings_call.errors import (
    EarningsCallValidationError,
    EarningsCallProviderUnavailableError,
    EarningsCallRunNotFoundError,
    EarningsCallTickerMismatchError,
)
from application.earnings_call.service import EarningsCallApplicationService
from application.earnings_call.workflow import EarningsCallRunStatus

router = APIRouter()


def _to_summarize_response(run: EarningsCallRunDTO) -> EarningsCallSummarizeResponse:
    return EarningsCallSummarizeResponse(
        run_id=run.run_id,
        ticker=run.ticker,
        period=run.period,
        status=run.status.value,
        kanban_status=run.kanban_status.value,
        highlights=run.highlights,
        vault_path=run.vault_path,
        kanban_card_id=run.kanban_card_id,
        reused_existing_run=run.reused_existing_run,
        is_idempotent_replay=run.is_idempotent_replay,
    )


def _to_run_response(run: EarningsCallRunDTO) -> EarningsCallRunResponse:
    return EarningsCallRunResponse(
        run_id=run.run_id,
        ticker=run.ticker,
        period=run.period,
        status=run.status.value,
        kanban_status=run.kanban_status.value,
        highlights=run.highlights,
        vault_path=run.vault_path,
        kanban_card_id=run.kanban_card_id,
        reused_existing_run=run.reused_existing_run,
        is_idempotent_replay=run.is_idempotent_replay,
        last_error_code=run.last_error_code,
        created_at=run.created_at,
        updated_at=run.updated_at,
    )


@router.post(
    "/{ticker}/earnings-call/summarize",
    response_model=EarningsCallSummarizeResponse,
    status_code=status.HTTP_200_OK,
    summary="Summarize Earnings Call Transcript",
)
def summarize_earnings_call(
    ticker: str,
    body: EarningsCallSummarizeRequest,
    response: Response,
    service: EarningsCallApplicationService = Depends(get_earnings_call_service),
) -> EarningsCallSummarizeResponse:
    """Summarizes an earnings call transcript, writes highlights & raw text to Obsidian, and dispatches to Kanban via Outbox."""
    request_dto = EarningsCallSummarizeRequestDTO(
        ticker=ticker,
        period=body.period,
        transcript=body.transcript,
    )
    try:
        run = service.summarize_and_store(request_dto)
    except EarningsCallValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except EarningsCallProviderUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM provider is currently unavailable or timed out. Please try again shortly.",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing the earnings call.",
        ) from exc

    if run.status != EarningsCallRunStatus.COMPLETED:
        response.status_code = status.HTTP_202_ACCEPTED

    return _to_summarize_response(run)


@router.get(
    "/{ticker}/earnings-call/runs/{run_id}",
    response_model=EarningsCallRunResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Earnings Call Run Status",
)
def get_earnings_call_run(
    ticker: str,
    run_id: str,
    service: EarningsCallApplicationService = Depends(get_earnings_call_service),
) -> EarningsCallRunResponse:
    """Fetches status and artifacts of a specific earnings call workflow run."""
    try:
        run = service.get_run_for_ticker(ticker=ticker, run_id=run_id)
    except (EarningsCallRunNotFoundError, EarningsCallTickerMismatchError) as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while fetching the run.",
        ) from exc

    return _to_run_response(run)


@router.post(
    "/{ticker}/earnings-call/runs/{run_id}/retry",
    response_model=EarningsCallRunResponse,
    status_code=status.HTTP_200_OK,
    summary="Retry Earnings Call Kanban Delivery",
)
def retry_earnings_call_run(
    ticker: str,
    run_id: str,
    response: Response,
    service: EarningsCallApplicationService = Depends(get_earnings_call_service),
) -> EarningsCallRunResponse:
    """Manually retries Kanban card dispatch for an earnings call run."""
    try:
        run = service.retry_run_for_ticker(ticker=ticker, run_id=run_id)
    except (EarningsCallRunNotFoundError, EarningsCallTickerMismatchError) as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while retrying the run.",
        ) from exc

    if run.status != EarningsCallRunStatus.COMPLETED:
        response.status_code = status.HTTP_202_ACCEPTED

    return _to_run_response(run)


@router.get(
    "/{ticker}/earnings-calls",
    response_model=EarningsCallListResponse,
    status_code=status.HTTP_200_OK,
    summary="List Existing Earnings Calls for Ticker",
)
def get_earnings_calls(
    ticker: str,
    service: EarningsCallApplicationService = Depends(get_earnings_call_service),
) -> EarningsCallListResponse:
    """Retrieves all existing earnings call notes and parsed AI highlights for a ticker."""
    try:
        notes = service.list_earnings_calls(ticker=ticker)
    except EarningsCallValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while listing earnings calls.",
        ) from exc

    items = [
        EarningsCallNoteItem(
            title=n.title,
            ticker=n.ticker,
            period=n.period,
            vault_path=n.vault_path,
            highlights=n.highlights,
            date=n.date,
            last_updated=n.last_updated,
            has_full_transcript=n.has_full_transcript,
        )
        for n in notes
    ]

    return EarningsCallListResponse(
        ticker=ticker.upper(),
        total_count=len(items),
        items=items,
    )


__all__ = [
    "router",
    "summarize_earnings_call",
    "get_earnings_call_run",
    "retry_earnings_call_run",
    "get_earnings_calls",
]
