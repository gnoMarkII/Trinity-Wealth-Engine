"""Local/API transport for the application knowledge write port."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import get_knowledge_write_service
from api.auth import require_session
from api.schemas.knowledge_writes import KnowledgeWriteRequest, KnowledgeWriteResponse
from application.knowledge.errors import KnowledgeWriteError, WriteCommandValidationError
from application.knowledge.write_models import KnowledgeWriteCommand
from application.knowledge.write_service import KnowledgeWriteService


router = APIRouter(
    prefix="/api/knowledge/writes",
    tags=["Knowledge Writes"],
    dependencies=[Depends(require_session)],
)


def _response(receipt) -> KnowledgeWriteResponse:
    return KnowledgeWriteResponse.model_validate(receipt.to_dict())


@router.post("", response_model=KnowledgeWriteResponse, status_code=status.HTTP_200_OK)
def submit_knowledge_write(
    request: KnowledgeWriteRequest,
    service: KnowledgeWriteService = Depends(get_knowledge_write_service),
) -> KnowledgeWriteResponse:
    try:
        value = request.model_dump(exclude_none=True)
        command = KnowledgeWriteCommand.from_dict(value)
        return _response(service.submit(command))
    except (WriteCommandValidationError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except KnowledgeWriteError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/{command_id}", response_model=KnowledgeWriteResponse)
def get_knowledge_write(
    command_id: str,
    service: KnowledgeWriteService = Depends(get_knowledge_write_service),
) -> KnowledgeWriteResponse:
    receipt = service.get_receipt(command_id=command_id)
    if receipt is None:
        raise HTTPException(status_code=404, detail="write receipt not found")
    return _response(receipt)
