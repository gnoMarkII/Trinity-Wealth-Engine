"""Inbound HTTP adapter for NotebookLM source and generation workflows."""
from fastapi import APIRouter, Depends, HTTPException

from api.auth import require_session
from api.dependencies import get_notebooklm_service
from api.schemas import NotebookLMAvailableSourceDTO, NotebookLMGenerateRequest, NotebookLMGenerateResponse, NotebookLMStatusDTO
from application.notebooklm.service import NotebookLMApplicationService, NotebookLMPreflightError

router = APIRouter(dependencies=[Depends(require_session)])


@router.get("/api/notebooklm/available-sources", response_model=list[NotebookLMAvailableSourceDTO])
def list_available_sources(
    service: NotebookLMApplicationService = Depends(get_notebooklm_service),
) -> list[NotebookLMAvailableSourceDTO]:
    return service.list_available_sources()


@router.post("/api/notebooklm/generate", response_model=NotebookLMGenerateResponse)
def generate_notebooklm_audio(
    payload: NotebookLMGenerateRequest,
    service: NotebookLMApplicationService = Depends(get_notebooklm_service),
) -> NotebookLMGenerateResponse:
    try:
        result = service.generate(payload.card_id, payload.briefing_file_path)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except NotebookLMPreflightError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return NotebookLMGenerateResponse(**result)


@router.get("/api/notebooklm/status/{job_id}", response_model=NotebookLMStatusDTO)
def get_notebooklm_status(
    job_id: str,
    service: NotebookLMApplicationService = Depends(get_notebooklm_service),
) -> NotebookLMStatusDTO:
    status = service.get_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"ไม่พบ job_id: {job_id}")
    return status


__all__ = ["router", "list_available_sources", "generate_notebooklm_audio", "get_notebooklm_status"]
