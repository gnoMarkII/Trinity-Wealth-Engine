"""REST & SSE Streaming Router for Trinity Guru Roundtable Studio."""
from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List
from fastapi import APIRouter, Body, Depends, HTTPException, status
from fastapi.responses import Response, StreamingResponse

from api.auth import require_session
from api.dependencies import get_roundtable_service
from application.roundtable.service import RoundtableApplicationService
from core.security import validate_ticker
from schemas.roundtable_schemas import (
    CreateRoundtableSessionRequest,
    InterjectRequest,
    ResetGuruProfilesRequest,
    RoundtableGuruConfigCatalogDTO,
    RoundtableSessionDetailDTO,
    RoundtableSessionSummaryDTO,
    RoundtableStreamRequest,
    SaveToVaultRequest,
    SaveToVaultResponseDTO,
    UpdateGuruProfilesRequest,
    UpdateThesisFromPlaybookRequest,
    VoicePreviewRequest,
    VoicePreviewResponseDTO,
)

router = APIRouter(
    prefix="/api/roundtable",
    tags=["roundtable"],
    dependencies=[Depends(require_session)],
)


@router.get("/guru-profiles", response_model=RoundtableGuruConfigCatalogDTO)
def get_roundtable_guru_profiles(
    service: RoundtableApplicationService = Depends(get_roundtable_service),
) -> RoundtableGuruConfigCatalogDTO:
    """Return effective Guru profiles paired with the 12-voice Gemini TTS catalog and 18 presets."""
    return service.get_guru_config_catalog()


@router.put("/guru-profiles", response_model=RoundtableGuruConfigCatalogDTO)
def update_roundtable_guru_profiles(
    req: UpdateGuruProfilesRequest = Body(...),
    service: RoundtableApplicationService = Depends(get_roundtable_service),
) -> RoundtableGuruConfigCatalogDTO:
    """Persist user-customized Guru profiles and voice selections globally."""
    try:
        return service.update_guru_profiles(req)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc


@router.post("/guru-profiles/reset", response_model=RoundtableGuruConfigCatalogDTO)
def reset_roundtable_guru_profiles(
    req: ResetGuruProfilesRequest = Body(default_factory=ResetGuruProfilesRequest),
    service: RoundtableApplicationService = Depends(get_roundtable_service),
) -> RoundtableGuruConfigCatalogDTO:
    """Reset one or all Guru profiles back to factory defaults."""
    return service.reset_guru_profiles(req)


@router.post("/voices/preview", response_model=VoicePreviewResponseDTO)
async def preview_roundtable_voice(
    req: VoicePreviewRequest = Body(...),
    service: RoundtableApplicationService = Depends(get_roundtable_service),
) -> VoicePreviewResponseDTO:
    """Synthesize or retrieve cached preview audio for a selected Gemini TTS voice."""
    try:
        return await service.preview_voice(req)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc


@router.get("/sessions", response_model=List[RoundtableSessionSummaryDTO])
def list_roundtable_sessions(
    limit: int = 30,
    service: RoundtableApplicationService = Depends(get_roundtable_service),
) -> List[RoundtableSessionSummaryDTO]:
    """List past Guru Roundtable sessions ordered by latest update."""
    return service.list_sessions(limit=limit)


@router.post(
    "/sessions",
    response_model=RoundtableSessionDetailDTO,
    status_code=status.HTTP_201_CREATED,
)
async def create_roundtable_session(
    req: CreateRoundtableSessionRequest = Body(...),
    service: RoundtableApplicationService = Depends(get_roundtable_service),
) -> RoundtableSessionDetailDTO:
    """Create a new Roundtable session, freeze its context packet, and check past settlements."""
    try:
        return await service.create_session(req)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc


@router.get("/sessions/{session_id}", response_model=RoundtableSessionDetailDTO)
def get_roundtable_session(
    session_id: str,
    check_drift: bool = True,
    service: RoundtableApplicationService = Depends(get_roundtable_service),
) -> RoundtableSessionDetailDTO:
    """Retrieve a single Roundtable session with turns, playbook, drift status & scorecard."""
    detail = service.get_session_detail(session_id, check_drift=check_drift)
    if detail is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Roundtable session not found")
    return detail


@router.post("/sessions/{session_id}/stream")
async def stream_roundtable_debate(
    session_id: str,
    req: RoundtableStreamRequest = Body(default_factory=RoundtableStreamRequest),
    service: RoundtableApplicationService = Depends(get_roundtable_service),
) -> StreamingResponse:
    """Stream turn-by-turn Guru Roundtable dialogue, TTS readiness, and Hermes playbook via SSE."""
    existing = service.get_session_detail(session_id)
    if existing is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Roundtable session not found")

    async def event_generator() -> AsyncIterator[str]:
        try:
            async for event_name, payload in service.stream_debate(
                session_id,
                voice_enabled=req.voice_enabled,
                refresh_context_on_drift=req.refresh_context_on_drift,
            ):
                serialized = json.dumps(payload, ensure_ascii=False)
                yield f"event: {event_name}\ndata: {serialized}\n\n"
        except Exception as exc:
            err = json.dumps({"error": str(exc)}, ensure_ascii=False)
            yield f"event: error\ndata: {err}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/sessions/{session_id}/interject")
async def interject_roundtable_session(
    session_id: str,
    req: InterjectRequest = Body(...),
    service: RoundtableApplicationService = Depends(get_roundtable_service),
) -> Dict[str, Any]:
    """Pause running stream, record user raise-hand question, run optional Sandbox Decimal math, and reply."""
    res = await service.handle_interjection(session_id, req)
    if res is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Roundtable session not found")
    return res


@router.get("/audio/{audio_cache_key}")
def get_roundtable_audio(
    audio_cache_key: str,
    service: RoundtableApplicationService = Depends(get_roundtable_service),
) -> Response:
    """Stream cached WAV audio bytes generated by Gemini TTS."""
    audio_bytes = service.get_audio_bytes(audio_cache_key)
    if audio_bytes is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Audio clip not found")
    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@router.get("/sessions/{session_id}/youtube-pack")
async def download_roundtable_youtube_pack(
    session_id: str,
    service: RoundtableApplicationService = Depends(get_roundtable_service),
) -> Response:
    """Download stitched WAV audio, SRT subtitles, and YouTube Production Kit as a ZIP archive."""
    pack = await service.export_youtube_pack(session_id)
    if pack is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Roundtable session not found")
    return Response(
        content=pack["zip_bytes"],
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{pack["filename"]}"',
            "Cache-Control": "no-store",
        },
    )


@router.post("/sessions/{session_id}/save-to-vault", response_model=SaveToVaultResponseDTO)
def save_roundtable_to_vault(
    session_id: str,
    req: SaveToVaultRequest = Body(default_factory=SaveToVaultRequest),
    service: RoundtableApplicationService = Depends(get_roundtable_service),
) -> SaveToVaultResponseDTO:
    """Save the distilled Roundtable Playbook & highlights to Obsidian Vault."""
    res = service.save_to_vault(session_id, req)
    if res is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Roundtable session not found")
    return res


@router.post("/sessions/{session_id}/update-thesis")
def update_thesis_from_roundtable(
    session_id: str,
    req: UpdateThesisFromPlaybookRequest = Body(default_factory=UpdateThesisFromPlaybookRequest),
    service: RoundtableApplicationService = Depends(get_roundtable_service),
) -> Dict[str, Any]:
    """Revise or create an Active Stock Thesis claim anchored on this Roundtable's verdict."""
    try:
        res = service.update_thesis(session_id, req)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    if res is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Roundtable session not found")
    return res


@router.post("/prerequisites/{ticker}/dispatch")
def dispatch_roundtable_prerequisite(
    ticker: str,
    service: RoundtableApplicationService = Depends(get_roundtable_service),
) -> Dict[str, Any]:
    """Dispatch background Deep Equity Analysis pipeline for an unanalyzed ticker."""
    try:
        clean = validate_ticker(ticker)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    return service.dispatch_missing_ticker(clean)

