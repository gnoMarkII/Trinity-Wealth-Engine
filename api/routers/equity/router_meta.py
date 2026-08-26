"""Inbound HTTP adapter for Equity meta and note-content queries."""
from typing import List

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_equity_query_service
from api.schemas import EquityNoteContentDTO, EquitySummaryDTO
from application.equity.query_service import EquityNoteAccessError, EquityResearchQueryService

router = APIRouter()


@router.get("/latest", response_model=List[EquitySummaryDTO])
def get_latest_equities(
    service: EquityResearchQueryService = Depends(get_equity_query_service),
) -> List[EquitySummaryDTO]:
    return [EquitySummaryDTO.model_validate(item) for item in service.list_latest()]


@router.get("/notes/content", response_model=EquityNoteContentDTO)
def get_equity_note_content(
    rel_path: str,
    service: EquityResearchQueryService = Depends(get_equity_query_service),
) -> EquityNoteContentDTO:
    try:
        return EquityNoteContentDTO.model_validate(service.read_note(rel_path))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Note file not found") from exc
    except EquityNoteAccessError as exc:
        status = 403 if str(exc).startswith("Access denied") else 400
        raise HTTPException(status_code=status, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read note content: {exc}") from exc
