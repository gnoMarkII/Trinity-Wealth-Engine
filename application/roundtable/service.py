"""Application Service for Trinity Guru Roundtable Studio.

Architecture Rule:
- Lives in ``application/roundtable/``
- Depends ONLY on ``core/``, ``schemas/``, and ``application/roundtable/ports``
- NEVER imports ``api.*``, ``sqlite3``, ``yfinance``, ``fastapi``, or ``tools.*``
"""
from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timezone
import json
import logging
import re
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence, Tuple
import uuid

from application.roundtable.context_assembler import RoundtableContextAssembler
from application.roundtable.ports import (
    AudioSynthesisResult,
    ContextDataProvidersPort,
    RoundtableDebateEnginePort,
    RoundtableSessionRepoPort,
    RoundtableTTSPort,
    RoundtableVerifierPort,
    SandboxCalcPort,
    ThesisUpdaterPort,
    VaultNoteWriterPort,
)
from core.roundtable.models import (
    GEMINI_VOICE_CATALOG,
    GURU_PERSONA_CATALOG,
    GURU_PRESET_CATALOG,
    PRESET_BY_SEAT_AND_ID,
    VOICE_BY_NAME,
    FrozenContextPacket,
    GuruPersonaSpec,
    RoundStage,
    RoundtableSession,
    RoundtableTurn,
    SpeakerId,
    analyze_debate_pillar_coverage,
    get_seat_ui_context,
    match_prioritized_speakers,
    merge_persona_overrides,
    normalize_debate_depth,
    normalize_debate_intensity,
    resolve_personas_from_packet,
    select_crossfire_duels,
    suggest_opening_order_for_agenda,
)
from core.roundtable.youtube_export import build_youtube_production_pack
from schemas.roundtable_schemas import (
    CreateRoundtableSessionRequest,
    DebatePillarCoverageDTO,
    DebatePillarStatusDTO,
    DistilledDecisionPlaybook,
    GuruPersonaDTO,
    GuruPresetItemDTO,
    InterjectRequest,
    PastScorecardDTO,
    ResetGuruProfilesRequest,
    RoundtableGuruConfigCatalogDTO,
    RoundtableSessionDetailDTO,
    RoundtableSessionSummaryDTO,
    RoundtableToolCallEventDTO,
    RoundtableTurnDTO,
    RoundtableTurnPayload,
    SaveToVaultRequest,
    SaveToVaultResponseDTO,
    UpdateGuruProfilesRequest,
    UpdateThesisFromPlaybookRequest,
    VoiceCatalogItemDTO,
    VoicePreviewRequest,
    VoicePreviewResponseDTO,
)

log = logging.getLogger(__name__)

_MATH_TRIGGER_RE = re.compile(
    r"(?:\d+(?:\.\d+)?\s*(?:%|หุ้น|บาท|usd|thb|ดอลลาร์|ไม้|percent)|ต้นทุนเฉลี่ย|ซื้อเพิ่ม|ขายออก|ลดลง|ปรับฐาน|สัดส่วน|calculate|คำนวณ|what.?if|dca)",
    re.IGNORECASE,
)

_MENTION_PATTERNS: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"@value\b|@buffett\b|@graham\b|@klarman\b|@damodaran\b|คุณ\s*value|คุณปู่|บัฟเฟตต์|เกรแฮม|คลาร์แมน", re.IGNORECASE), SpeakerId.VALUE_GURU.value),
    (re.compile(r"@quality\b|@munger\b|@akre\b|@smith\b|คุณ\s*quality|มังเจอร์|เอเคอร์", re.IGNORECASE), SpeakerId.QUALITY_GURU.value),
    (re.compile(r"@growth\b|@lynch\b|@fisher\b|คุณ\s*growth|ลินช์|ฟิชเชอร์", re.IGNORECASE), SpeakerId.GROWTH_GURU.value),
    (re.compile(r"@macro\b|@marks\b|@dalio\b|คุณ\s*macro|ดรัคเคน|ดาลิโอ|มหภาค", re.IGNORECASE), SpeakerId.MACRO_TACTICAL_GURU.value),
    (re.compile(r"@skeptic\b|@burry\b|@chanos\b|@taleb\b|คุณ\s*skeptic|เบอร์รี่|ทาเล็บ", re.IGNORECASE), SpeakerId.SKEPTIC_GURU.value),
    (re.compile(r"@hermes\b|คุณ\s*hermes|เฮอร์เมส", re.IGNORECASE), SpeakerId.HERMES_MODERATOR.value),
)


def _detect_target_speaker(
    message: str,
    explicit_target: Optional[str],
    active_personas: Sequence[GuruPersonaSpec] = (),
) -> str:
    if explicit_target:
        return explicit_target
    text = (message or "").strip()

    enabled_personas = [
        p for p in active_personas if p.enabled or p.speaker_id == SpeakerId.HERMES_MODERATOR
    ]
    enabled_ids = {p.speaker_id.value for p in enabled_personas} if active_personas else None

    for spec in sorted(enabled_personas, key=lambda s: len(s.short_name or ""), reverse=True):
        short_clean = (spec.short_name or "").strip()
        if not short_clean:
            continue
        custom_pat = re.compile(
            rf"(?:@{re.escape(short_clean)}(?=\s|$|[,\.!?:;])|คุณ\s*{re.escape(short_clean)}(?=\s|$|[,\.!?:;]))",
            re.IGNORECASE,
        )
        if custom_pat.search(text):
            return spec.speaker_id.value

    for pat, speaker_id in _MENTION_PATTERNS:
        if pat.search(text):
            if enabled_ids is None or speaker_id in enabled_ids:
                return speaker_id
    return SpeakerId.HERMES_MODERATOR.value


def _should_run_sandbox_math(message: str, explicit_flag: bool) -> bool:
    if explicit_flag:
        return True
    return bool(_MATH_TRIGGER_RE.search(message or ""))


def _turn_model_to_dto(
    turn: RoundtableTurn,
    audio_status: Optional[str] = None,
) -> RoundtableTurnDTO:
    status = audio_status or ("ready" if turn.audio_cache_key else "disabled")
    return RoundtableTurnDTO(
        turn_id=turn.turn_id,
        session_id=turn.session_id,
        seq=turn.seq,
        speaker_id=turn.speaker_id,
        round_stage=turn.round_stage,
        replying_to=turn.replying_to,
        spoken_dialogue=turn.spoken_dialogue,
        evidence_bullets=list(turn.evidence_bullets),
        stance_tag=turn.stance_tag,  # type: ignore[arg-type]
        sizing_posture=turn.sizing_posture,
        audio_cache_key=turn.audio_cache_key,
        audio_status=status,  # type: ignore[arg-type]
        created_at=turn.created_at,
    )


def _turn_model_to_payload(turn: RoundtableTurn) -> RoundtableTurnPayload:
    return RoundtableTurnPayload(
        speaker_id=turn.speaker_id,  # type: ignore[arg-type]
        round_stage=turn.round_stage,  # type: ignore[arg-type]
        replying_to_speaker_id=turn.replying_to,
        spoken_dialogue=turn.spoken_dialogue,
        evidence_bullets=list(turn.evidence_bullets),
        stance_tag=turn.stance_tag,  # type: ignore[arg-type]
        sizing_posture=turn.sizing_posture,
        audio_cache_key=turn.audio_cache_key,
        audio_status="ready" if turn.audio_cache_key else "disabled",
    )


def _session_to_summary_dto(session: RoundtableSession) -> RoundtableSessionSummaryDTO:
    return RoundtableSessionSummaryDTO(
        session_id=session.session_id,
        title=session.title,
        primary_ticker=session.primary_ticker,
        benchmark_ticker=session.benchmark_ticker,
        agenda_prompt=session.agenda_prompt,
        context_refs=list(session.context_refs),
        final_rating=session.final_rating,  # type: ignore[arg-type]
        settlement_status=session.settlement_status,
        settled_return_pct=session.settled_return_pct,
        settled_alpha_pct=session.settled_alpha_pct,
        status=session.status,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


def _build_persona_dtos(
    prioritized_speakers: Sequence[str],
    personas: Optional[Sequence[GuruPersonaSpec]] = None,
) -> List[GuruPersonaDTO]:
    prio_set = set(prioritized_speakers)
    catalog = list(personas if personas is not None else GURU_PERSONA_CATALOG)
    dtos: List[GuruPersonaDTO] = []
    for spec in sorted(catalog, key=lambda s: s.debate_order):
        ui_ctx = get_seat_ui_context(spec.speaker_id.value, spec.preset_id)
        preset_tmpl = PRESET_BY_SEAT_AND_ID.get((spec.speaker_id.value, spec.preset_id))
        dtos.append(
            GuruPersonaDTO(
                speaker_id=spec.speaker_id.value,
                display_name=spec.display_name,
                short_name=spec.short_name,
                archetype=spec.archetype,
                voice_name=spec.voice_name,
                accent_color=spec.accent_color,
                is_prioritized_role_model=spec.speaker_id.value in prio_set,
                emoji=spec.emoji,
                pitch=spec.pitch,
                rate=spec.rate,
                enabled=spec.enabled,
                debate_order=spec.debate_order,
                preset_id=spec.preset_id,
                preset_label=preset_tmpl.label_th if preset_tmpl else "",
                speaking_style=spec.speaking_style,
                custom_directive=spec.custom_directive,
                role_lens_label=ui_ctx["role_lens_label"],
                focus_summary=ui_ctx["focus_summary"],
                primary_evidence_labels=ui_ctx["primary_evidence_labels"],
                analytical_rubric=ui_ctx["analytical_rubric"],
            )
        )
    return dtos


def _build_pillar_coverage_dto(turns: Sequence[Any]) -> Optional[DebatePillarCoverageDTO]:
    if not turns:
        return None
    cov = analyze_debate_pillar_coverage(turns)
    raised_by: Dict[str, List[str]] = cov.get("raised_by") or {}
    challenged_by: Dict[str, List[str]] = cov.get("challenged_by") or {}
    unchallenged = list(cov.get("unchallenged_pillars") or [])
    pillars = [
        DebatePillarStatusDTO(
            pillar=pillar,
            raised_by=list(raised_by.get(pillar) or []),
            challenged_by=list(challenged_by.get(pillar) or []),
            is_challenged=bool(challenged_by.get(pillar)),
        )
        for pillar in raised_by
    ]
    return DebatePillarCoverageDTO(
        pillars=pillars,
        raised_count=sum(1 for p in pillars if p.raised_by),
        challenged_count=sum(1 for p in pillars if p.is_challenged),
        unchallenged_pillars=unchallenged,
    )


class RoundtableApplicationService:
    """Coordinates Guru Roundtable sessions, SSE streaming debates, TTS pipelining & Vault/Thesis actions."""

    def __init__(
        self,
        *,
        repo: RoundtableSessionRepoPort,
        assembler: RoundtableContextAssembler,
        engine: RoundtableDebateEnginePort,
        tts: RoundtableTTSPort,
        verifier: Optional[RoundtableVerifierPort] = None,
        sandbox_calc: Optional[SandboxCalcPort] = None,
        vault_writer: Optional[VaultNoteWriterPort] = None,
        thesis_updater: Optional[ThesisUpdaterPort] = None,
        providers: Optional[ContextDataProvidersPort] = None,
    ) -> None:
        self._repo = repo
        self._assembler = assembler
        self._engine = engine
        self._tts = tts
        self._verifier = verifier
        self._sandbox_calc = sandbox_calc
        self._vault_writer = vault_writer
        self._thesis_updater = thesis_updater
        self._providers = providers
        self._session_locks: Dict[str, asyncio.Lock] = {}
        self._stream_locks: Dict[str, asyncio.Lock] = {}
        self._cancel_events: Dict[str, asyncio.Event] = {}

    def _get_stream_lock(self, session_id: str) -> asyncio.Lock:
        lock = self._stream_locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._stream_locks[session_id] = lock
        return lock

    def _get_lock(self, session_id: str) -> asyncio.Lock:
        lock = self._session_locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._session_locks[session_id] = lock
        return lock

    def _ordered_guru_speakers(self, packet: Optional[FrozenContextPacket] = None) -> List[str]:
        personas = resolve_personas_from_packet(packet)
        return suggest_opening_order_for_agenda(
            packet.agenda_prompt if packet is not None else "",
            personas,
        )

    def _build_detail_dto(
        self,
        session: RoundtableSession,
        *,
        packet: FrozenContextPacket,
        turns: Optional[List[RoundtableTurn]] = None,
        drift_detected: bool = False,
    ) -> RoundtableSessionDetailDTO:
        session_turns = turns if turns is not None else self._repo.get_turns(session.session_id)
        playbook: Optional[DistilledDecisionPlaybook] = None
        if session.decision_playbook_json:
            try:
                playbook = DistilledDecisionPlaybook.model_validate_json(session.decision_playbook_json)
            except Exception as exc:
                log.warning("Failed parsing playbook JSON for %s: %s", session.session_id, exc)

        scorecard_dto: Optional[PastScorecardDTO] = None
        if packet.past_scorecard is not None:
            sc = packet.past_scorecard
            scorecard_dto = PastScorecardDTO(
                previous_session_id=sc.previous_session_id,
                previous_date=sc.previous_date,
                primary_ticker=sc.primary_ticker,
                benchmark_ticker=sc.benchmark_ticker,
                entry_price=sc.entry_price,
                settlement_price=sc.settlement_price,
                raw_return_pct=sc.raw_return_pct,
                benchmark_return_pct=sc.benchmark_return_pct,
                alpha_pct=sc.alpha_pct,
                previous_final_rating=sc.previous_final_rating,  # type: ignore[arg-type]
                outcome_label=sc.outcome_label,
                reflection_lesson=sc.reflection_lesson,
                settlement_status=sc.settlement_status,
            )

        active_personas = resolve_personas_from_packet(packet)
        return RoundtableSessionDetailDTO(
            session=_session_to_summary_dto(session),
            turns=[_turn_model_to_dto(t) for t in session_turns],
            decision_playbook=playbook,
            past_scorecard=scorecard_dto,
            missing_prerequisite_tickers=list(packet.missing_prerequisite_tickers),
            prioritized_speakers=list(packet.prioritized_speakers),
            user_role_models=list(packet.user_role_models),
            portfolio_attachment_state=packet.portfolio_guard.attachment_state.value,
            context_drift_detected=drift_detected,
            packet_fingerprint_sha256=packet.packet_fingerprint_sha256,
            debate_intensity=normalize_debate_intensity(packet.debate_intensity, default="balanced"),  # type: ignore[arg-type]
            debate_depth=normalize_debate_depth(packet.debate_depth, default="concise"),  # type: ignore[arg-type]
            personas=_build_persona_dtos(packet.prioritized_speakers, personas=active_personas),
            pillar_coverage=_build_pillar_coverage_dto(session_turns),
        )

    def _parse_packet(self, session: RoundtableSession) -> FrozenContextPacket:
        try:
            raw = json.loads(session.frozen_packet_json or "{}")
            return FrozenContextPacket.from_dict(raw)
        except Exception:
            return FrozenContextPacket(
                session_id=session.session_id,
                created_at_iso="",
                agenda_prompt=session.agenda_prompt,
                context_refs=tuple(session.context_refs),
                primary_ticker=session.primary_ticker,
                packet_fingerprint_sha256=session.packet_fingerprint_sha256,
            )

    async def create_session(self, req: CreateRoundtableSessionRequest) -> RoundtableSessionDetailDTO:
        saved_profiles = self._repo.list_guru_profiles()
        session_overrides = (
            [p.model_dump(exclude_none=True) for p in req.persona_overrides]
            if req.persona_overrides
            else None
        )
        effective_personas = merge_persona_overrides(
            GURU_PERSONA_CATALOG,
            saved_profiles,
            session_overrides,
        )

        session_id = f"rt_{uuid.uuid4().hex[:12]}"
        packet = self._assembler.assemble(
            session_id=session_id,
            agenda_prompt=req.agenda_prompt,
            context_refs=req.context_refs,
            portfolio_id=req.portfolio_id,
            debate_intensity=req.debate_intensity,
            debate_depth=req.debate_depth,
        )
        packet = replace(packet, active_personas=tuple(p.to_dict() for p in effective_personas))

        # Closed-Loop Reflection Lesson: settle & persist once when prior session completes evaluation
        if packet.past_scorecard is not None:
            sc = packet.past_scorecard
            prev_session = self._repo.get_session(sc.previous_session_id)
            if prev_session is not None and sc.settlement_status == "settled":
                lesson = sc.reflection_lesson
                if not lesson:
                    prev_verdict = prev_session.agenda_prompt
                    if prev_session.decision_playbook_json:
                        try:
                            prev_pb = json.loads(prev_session.decision_playbook_json)
                            prev_verdict = str(prev_pb.get("executive_verdict") or prev_verdict)
                        except Exception:
                            pass
                    lesson = await self._engine.generate_reflection_lesson(
                        previous_rating=sc.previous_final_rating,
                        raw_return_pct=sc.raw_return_pct,
                        alpha_pct=sc.alpha_pct,
                        previous_verdict=prev_verdict,
                        ticker=sc.primary_ticker,
                    )
                    self._repo.update_session_settlement(
                        session_id=prev_session.session_id,
                        settlement_status="settled",
                        settled_return_pct=sc.raw_return_pct,
                        settled_alpha_pct=sc.alpha_pct,
                        settled_at_date=datetime.now(timezone.utc).date().isoformat(),
                        reflection_lesson=lesson,
                    )
                    packet = replace(packet, past_scorecard=replace(sc, reflection_lesson=lesson))

        clean_title = (req.title or "").strip()
        if not clean_title:
            if packet.primary_ticker:
                clean_title = f"ประชุมกูรู ${packet.primary_ticker}"
            else:
                first_line = (req.agenda_prompt or "ประชุมกูรู").strip().splitlines()[0][:55]
                clean_title = first_line or "ประชุมกูรูรอบโต๊ะ"

        session = self._repo.create_session(
            session_id=session_id,
            title=clean_title,
            primary_ticker=packet.primary_ticker,
            benchmark_ticker=packet.instrument_anchor.benchmark_ticker if packet.instrument_anchor else None,
            entry_price_at_session=packet.instrument_anchor.current_price if packet.instrument_anchor else None,
            agenda_prompt=req.agenda_prompt,
            context_refs=packet.context_refs,
            frozen_packet_json=json.dumps(packet.to_dict(), ensure_ascii=False),
            packet_fingerprint_sha256=packet.packet_fingerprint_sha256,
            final_rating="REVIEW",
            status="active",
        )
        return self._build_detail_dto(session, packet=packet, turns=[], drift_detected=False)

    def list_sessions(self, limit: int = 30) -> List[RoundtableSessionSummaryDTO]:
        sessions = self._repo.list_sessions(limit=limit)
        return [_session_to_summary_dto(s) for s in sessions]

    def get_session_detail(
        self,
        session_id: str,
        *,
        check_drift: bool = True,
    ) -> Optional[RoundtableSessionDetailDTO]:
        session = self._repo.get_session(session_id)
        if session is None:
            return None
        packet = self._parse_packet(session)
        drift_detected = False
        if check_drift:
            try:
                fresh = self._assembler.assemble(
                    session_id=session.session_id,
                    agenda_prompt=session.agenda_prompt,
                    context_refs=session.context_refs,
                    portfolio_id=packet.portfolio_id or "default",
                    debate_intensity=packet.debate_intensity,
                    debate_depth=packet.debate_depth,
                )
                if fresh.packet_fingerprint_sha256 and session.packet_fingerprint_sha256:
                    drift_detected = fresh.packet_fingerprint_sha256 != session.packet_fingerprint_sha256
            except Exception as exc:
                log.debug("Context drift check skipped for %s: %s", session_id, exc)
        return self._build_detail_dto(session, packet=packet, drift_detected=drift_detected)

    async def stream_debate(
        self,
        session_id: str,
        *,
        voice_enabled: bool = True,
        refresh_context_on_drift: bool = False,
    ) -> AsyncIterator[Tuple[str, Dict[str, Any]]]:
        """Stream one debate per session at a time, yielding SSE event payloads."""
        stream_lock = self._get_stream_lock(session_id)
        if stream_lock.locked():
            yield ("error", {"error": "A debate stream is already active for this session"})
            return

        await stream_lock.acquire()
        cancel_ev = asyncio.Event()
        self._cancel_events[session_id] = cancel_ev
        try:
            async for event in self._stream_debate_locked(
                session_id,
                voice_enabled=voice_enabled,
                refresh_context_on_drift=refresh_context_on_drift,
                cancel_ev=cancel_ev,
            ):
                yield event
        finally:
            if self._cancel_events.get(session_id) is cancel_ev:
                self._cancel_events.pop(session_id, None)
            stream_lock.release()

    async def _stream_debate_locked(
        self,
        session_id: str,
        *,
        voice_enabled: bool,
        refresh_context_on_drift: bool,
        cancel_ev: asyncio.Event,
    ) -> AsyncIterator[Tuple[str, Dict[str, Any]]]:
        session = self._repo.get_session(session_id)
        if session is None:
            yield ("error", {"error": f"Roundtable session '{session_id}' not found"})
            return

        packet = self._parse_packet(session)

        if refresh_context_on_drift:
            preserved_personas = packet.active_personas
            refreshed = self._assembler.assemble(
                session_id=session.session_id,
                agenda_prompt=session.agenda_prompt,
                context_refs=session.context_refs,
                portfolio_id=packet.portfolio_id or "default",
                debate_intensity=packet.debate_intensity,
                debate_depth=packet.debate_depth,
            )
            packet = replace(refreshed, active_personas=preserved_personas)
            self._repo.update_session_context(
                session_id=session_id,
                frozen_packet_json=json.dumps(packet.to_dict(), ensure_ascii=False),
                packet_fingerprint_sha256=packet.packet_fingerprint_sha256,
            )

        active_personas = resolve_personas_from_packet(packet)
        persona_map = {p.speaker_id.value: p for p in active_personas}
        persona_dtos = _build_persona_dtos(packet.prioritized_speakers, personas=active_personas)

        yield (
            "session_meta",
            {
                "session_id": session_id,
                "primary_ticker": packet.primary_ticker,
                "portfolio_attachment_state": packet.portfolio_guard.attachment_state.value,
                "prioritized_speakers": list(packet.prioritized_speakers),
                "missing_prerequisite_tickers": list(packet.missing_prerequisite_tickers),
                "packet_fingerprint_sha256": packet.packet_fingerprint_sha256,
                "context_refreshed": refresh_context_on_drift,
                "debate_intensity": packet.debate_intensity,
                "debate_depth": packet.debate_depth,
                "personas": [dto.model_dump() for dto in persona_dtos],
            },
        )

        existing_turns = self._repo.get_turns(session_id)
        history_payloads: List[RoundtableTurnPayload] = [_turn_model_to_payload(t) for t in existing_turns]
        ordered_gurus = self._ordered_guru_speakers(packet)

        completed_intro = any(t.round_stage == RoundStage.STAGE_INTRO.value for t in existing_turns)
        completed_opening = {
            t.speaker_id for t in existing_turns if t.round_stage == RoundStage.OPENING.value
        }
        completed_rebuttal = {
            t.speaker_id for t in existing_turns if t.round_stage == RoundStage.REBUTTAL.value
        }
        completed_crossfire_count = sum(
            1 for t in existing_turns if t.round_stage == RoundStage.CROSSFIRE.value
        )
        has_synthesis = any(t.round_stage == RoundStage.SYNTHESIS.value for t in existing_turns)

        eff_depth = normalize_debate_depth(packet.debate_depth, default="concise")
        eff_intensity = normalize_debate_intensity(packet.debate_intensity, default="balanced")

        pending_audio_tasks: Dict[asyncio.Task[AudioSynthesisResult], Tuple[str, str]] = {}
        audio_done_signal: asyncio.Queue[None] = asyncio.Queue()
        _result_sentinel = object()

        def _notify_audio_done(_task: asyncio.Task[AudioSynthesisResult]) -> None:
            try:
                audio_done_signal.put_nowait(None)
            except Exception:
                pass

        def _schedule_tts(turn_id: str, speaker_id: str, text: str) -> None:
            if not voice_enabled:
                return
            voice_override = persona_map[speaker_id].voice_name if speaker_id in persona_map else None
            task = asyncio.create_task(
                self._tts.synthesize_turn_audio(
                    text=text,
                    speaker_id=speaker_id,
                    enabled=True,
                    voice_name=voice_override,
                )
            )
            pending_audio_tasks[task] = (turn_id, speaker_id)
            task.add_done_callback(_notify_audio_done)

        def _drain_finished_audio() -> List[Tuple[str, Dict[str, Any]]]:
            events: List[Tuple[str, Dict[str, Any]]] = []
            done_tasks = [t for t in list(pending_audio_tasks.keys()) if t.done()]
            for task in done_tasks:
                turn_id, speaker_id = pending_audio_tasks.pop(task)
                try:
                    res = task.result()
                except Exception as exc:
                    res = AudioSynthesisResult(
                        audio_cache_key=None,
                        audio_status="fallback_web_speech",
                        error_message=str(exc),
                    )
                if res.audio_cache_key:
                    self._repo.update_turn_audio_key(turn_id, res.audio_cache_key)
                events.append(
                    (
                        "audio_ready",
                        {
                            "turn_id": turn_id,
                            "speaker_id": speaker_id,
                            "audio_cache_key": res.audio_cache_key,
                            "audio_status": res.audio_status,
                            "error_message": res.error_message,
                        },
                    )
                )
            return events

        async def _await_llm_draining_audio(
            coro: Any,
        ) -> AsyncIterator[Tuple[str, Any]]:
            """Run an LLM turn coroutine while emitting finished TTS audio_ready events immediately."""
            llm_task = asyncio.create_task(coro)
            try:
                while True:
                    while not audio_done_signal.empty():
                        audio_done_signal.get_nowait()
                    for audio_ev in _drain_finished_audio():
                        yield audio_ev
                    if llm_task.done():
                        break
                    if not pending_audio_tasks:
                        await asyncio.wait({llm_task})
                        break
                    sig_task = asyncio.create_task(audio_done_signal.get())
                    try:
                        await asyncio.wait({llm_task, sig_task}, return_when=asyncio.FIRST_COMPLETED)
                    finally:
                        if not sig_task.done():
                            sig_task.cancel()
                            try:
                                await sig_task
                            except asyncio.CancelledError:
                                pass
                for audio_ev in _drain_finished_audio():
                    yield audio_ev
                yield ("_result", await llm_task)
            finally:
                if not llm_task.done():
                    llm_task.cancel()
                    try:
                        await llm_task
                    except (asyncio.CancelledError, Exception):
                        pass

        try:
            # Stage Intro (YouTube / Deep Debate Cold Open Hook by Hermes)
            if eff_depth == "deep_youtube" and not completed_intro and not has_synthesis:
                if cancel_ev.is_set():
                    yield ("paused", {"session_id": session_id, "reason": "user_interjection"})
                    return

                yield (
                    "speaker_started",
                    {
                        "speaker_id": SpeakerId.HERMES_MODERATOR.value,
                        "round_stage": RoundStage.STAGE_INTRO.value,
                    },
                )
                if hasattr(self._engine, "generate_hermes_stage_intro"):
                    intro_payload = await self._engine.generate_hermes_stage_intro(
                        packet=packet,
                        ordered_gurus=ordered_gurus,
                        verifier=self._verifier,
                    )
                else:
                    intro_payload = None
                if intro_payload is None:
                    intro_payload = await self._engine.generate_guru_turn(
                        speaker_id=SpeakerId.HERMES_MODERATOR.value,
                        round_stage=RoundStage.STAGE_INTRO.value,
                        packet=packet,
                        prior_turns=history_payloads,
                        unspoken_speakers=ordered_gurus,
                        verifier=self._verifier,
                    )

                if cancel_ev.is_set():
                    yield ("paused", {"session_id": session_id, "reason": "user_interjection"})
                    return

                async with self._get_lock(session_id):
                    seq = self._repo.get_next_seq(session_id)
                    turn_id = f"turn_{uuid.uuid4().hex[:12]}"
                    saved_intro = self._repo.append_turn(
                        turn_id=turn_id,
                        session_id=session_id,
                        seq=seq,
                        speaker_id=SpeakerId.HERMES_MODERATOR.value,
                        round_stage=RoundStage.STAGE_INTRO.value,
                        replying_to=None,
                        spoken_dialogue=intro_payload.spoken_dialogue,
                        evidence_bullets=intro_payload.evidence_bullets,
                        stance_tag=intro_payload.stance_tag,
                        audio_cache_key=None,
                        sizing_posture=intro_payload.sizing_posture,
                    )
                    history_payloads.append(intro_payload)

                intro_dto = _turn_model_to_dto(
                    saved_intro,
                    audio_status="disabled" if not voice_enabled else "fallback_web_speech",
                )
                yield ("turn_completed", intro_dto.model_dump())
                _schedule_tts(saved_intro.turn_id, saved_intro.speaker_id, saved_intro.spoken_dialogue)
                for audio_ev in _drain_finished_audio():
                    yield audio_ev

            # Round 1 (Opening Thesis) & Round 2 (Cross-Exam + Sizing Posture)
            stage_plans = [
                (RoundStage.OPENING.value, completed_opening),
                (RoundStage.REBUTTAL.value, completed_rebuttal),
            ]
            for round_stage, completed_set in stage_plans:
                remaining_speakers = [s for s in ordered_gurus if s not in completed_set]
                for idx, speaker_id in enumerate(remaining_speakers):
                    if cancel_ev.is_set():
                        yield ("paused", {"session_id": session_id, "reason": "user_interjection"})
                        return

                    unspoken = remaining_speakers[idx + 1 :]
                    yield ("speaker_started", {"speaker_id": speaker_id, "round_stage": round_stage})

                    payload = _result_sentinel
                    async for ev_kind, ev_val in _await_llm_draining_audio(
                        self._engine.generate_guru_turn(
                            speaker_id=speaker_id,
                            round_stage=round_stage,
                            packet=packet,
                            prior_turns=history_payloads,
                            unspoken_speakers=unspoken,
                            verifier=self._verifier,
                        )
                    ):
                        if ev_kind == "_result":
                            payload = ev_val
                        else:
                            yield (ev_kind, ev_val)
                    assert payload is not _result_sentinel

                    if cancel_ev.is_set():
                        yield ("paused", {"session_id": session_id, "reason": "user_interjection"})
                        return

                    async with self._get_lock(session_id):
                        seq = self._repo.get_next_seq(session_id)
                        turn_id = f"turn_{uuid.uuid4().hex[:12]}"
                        saved_turn = self._repo.append_turn(
                            turn_id=turn_id,
                            session_id=session_id,
                            seq=seq,
                            speaker_id=payload.speaker_id,
                            round_stage=round_stage,
                            replying_to=payload.replying_to_speaker_id,
                            spoken_dialogue=payload.spoken_dialogue,
                            evidence_bullets=payload.evidence_bullets,
                            stance_tag=payload.stance_tag,
                            audio_cache_key=None,
                            sizing_posture=payload.sizing_posture,
                        )
                        history_payloads.append(payload)

                    dto = _turn_model_to_dto(
                        saved_turn,
                        audio_status="disabled" if not voice_enabled else "fallback_web_speech",
                    )
                    yield ("turn_completed", dto.model_dump())
                    _schedule_tts(saved_turn.turn_id, saved_turn.speaker_id, saved_turn.spoken_dialogue)
                    for audio_ev in _drain_finished_audio():
                        yield audio_ev

            # Round 2.5: Crossfire Duel Stage (Head-to-Head Right-of-Reply Clash)
            if eff_intensity in ("heated", "crossfire") and not has_synthesis:
                max_duels = 1 if eff_intensity == "heated" else 2
                base_turns = [t for t in history_payloads if t.round_stage != RoundStage.CROSSFIRE.value]
                duel_plan = select_crossfire_duels(base_turns, ordered_gurus, max_duels=max_duels)
                remaining_duel_turns = duel_plan[completed_crossfire_count:]
                for speaker_id, target_id in remaining_duel_turns:
                    if cancel_ev.is_set():
                        yield ("paused", {"session_id": session_id, "reason": "user_interjection"})
                        return

                    yield (
                        "speaker_started",
                        {
                            "speaker_id": speaker_id,
                            "round_stage": RoundStage.CROSSFIRE.value,
                            "replying_to": target_id,
                        },
                    )
                    async def _gen_crossfire(sp_id: str = speaker_id, tgt_id: str = target_id) -> Any:
                        try:
                            return await self._engine.generate_guru_turn(
                                speaker_id=sp_id,
                                round_stage=RoundStage.CROSSFIRE.value,
                                packet=packet,
                                prior_turns=history_payloads,
                                unspoken_speakers=[],
                                verifier=self._verifier,
                                crossfire_target=tgt_id,
                            )
                        except TypeError:
                            return await self._engine.generate_guru_turn(
                                speaker_id=sp_id,
                                round_stage=RoundStage.CROSSFIRE.value,
                                packet=packet,
                                prior_turns=history_payloads,
                                unspoken_speakers=[],
                                verifier=self._verifier,
                            )

                    payload = _result_sentinel
                    async for ev_kind, ev_val in _await_llm_draining_audio(_gen_crossfire()):
                        if ev_kind == "_result":
                            payload = ev_val
                        else:
                            yield (ev_kind, ev_val)
                    assert payload is not _result_sentinel

                    if cancel_ev.is_set():
                        yield ("paused", {"session_id": session_id, "reason": "user_interjection"})
                        return

                    reply_target = payload.replying_to_speaker_id or target_id
                    payload_recorded = payload.model_copy(
                        update={
                            "round_stage": RoundStage.CROSSFIRE.value,
                            "replying_to_speaker_id": reply_target,
                        }
                    )
                    async with self._get_lock(session_id):
                        seq = self._repo.get_next_seq(session_id)
                        turn_id = f"turn_{uuid.uuid4().hex[:12]}"
                        saved_cf = self._repo.append_turn(
                            turn_id=turn_id,
                            session_id=session_id,
                            seq=seq,
                            speaker_id=payload_recorded.speaker_id,
                            round_stage=RoundStage.CROSSFIRE.value,
                            replying_to=reply_target,
                            spoken_dialogue=payload_recorded.spoken_dialogue,
                            evidence_bullets=payload_recorded.evidence_bullets,
                            stance_tag=payload_recorded.stance_tag,
                            audio_cache_key=None,
                            sizing_posture=payload_recorded.sizing_posture,
                        )
                        history_payloads.append(payload_recorded)

                    cf_dto = _turn_model_to_dto(
                        saved_cf,
                        audio_status="disabled" if not voice_enabled else "fallback_web_speech",
                    )
                    yield ("turn_completed", cf_dto.model_dump())
                    _schedule_tts(saved_cf.turn_id, saved_cf.speaker_id, saved_cf.spoken_dialogue)
                    for audio_ev in _drain_finished_audio():
                        yield audio_ev

            # Round 3: Hermes Distillation Synthesis
            if not has_synthesis:
                if cancel_ev.is_set():
                    yield ("paused", {"session_id": session_id, "reason": "user_interjection"})
                    return

                yield (
                    "speaker_started",
                    {"speaker_id": SpeakerId.HERMES_MODERATOR.value, "round_stage": RoundStage.SYNTHESIS.value},
                )
                synth_res: Any = _result_sentinel
                async for ev_kind, ev_val in _await_llm_draining_audio(
                    self._engine.generate_hermes_synthesis(
                        packet=packet,
                        all_turns=history_payloads,
                        verifier=self._verifier,
                    )
                ):
                    if ev_kind == "_result":
                        synth_res = ev_val
                    else:
                        yield (ev_kind, ev_val)
                assert synth_res is not _result_sentinel
                closing_payload, playbook = synth_res

                if cancel_ev.is_set():
                    yield ("paused", {"session_id": session_id, "reason": "user_interjection"})
                    return

                async with self._get_lock(session_id):
                    seq = self._repo.get_next_seq(session_id)
                    turn_id = f"turn_{uuid.uuid4().hex[:12]}"
                    saved_closing = self._repo.append_turn(
                        turn_id=turn_id,
                        session_id=session_id,
                        seq=seq,
                        speaker_id=SpeakerId.HERMES_MODERATOR.value,
                        round_stage=RoundStage.SYNTHESIS.value,
                        replying_to=closing_payload.replying_to_speaker_id,
                        spoken_dialogue=closing_payload.spoken_dialogue,
                        evidence_bullets=closing_payload.evidence_bullets,
                        stance_tag=playbook.final_rating,
                        audio_cache_key=None,
                        sizing_posture=closing_payload.sizing_posture,
                    )
                    self._repo.update_session_playbook(
                        session_id=session_id,
                        decision_playbook_json=playbook.model_dump_json(),
                        final_rating=playbook.final_rating,
                        status="completed",
                    )

                closing_dto = _turn_model_to_dto(
                    saved_closing,
                    audio_status="disabled" if not voice_enabled else "fallback_web_speech",
                )
                yield ("turn_completed", closing_dto.model_dump())
                yield ("playbook_updated", playbook.model_dump())
                cov_dto = _build_pillar_coverage_dto(history_payloads)
                if cov_dto is not None:
                    yield ("pillar_coverage", cov_dto.model_dump())
                _schedule_tts(
                    saved_closing.turn_id, saved_closing.speaker_id, saved_closing.spoken_dialogue
                )

            # Flush any remaining in-flight audio synthesis tasks progressively as each completes
            while pending_audio_tasks:
                await asyncio.wait(list(pending_audio_tasks.keys()), return_when=asyncio.FIRST_COMPLETED)
                for audio_ev in _drain_finished_audio():
                    yield audio_ev

            yield ("done", {"session_id": session_id, "status": "completed"})
        finally:
            for t in list(pending_audio_tasks.keys()):
                if not t.done():
                    t.cancel()
            # The public stream wrapper clears its cancel token and releases the stream lock.

    async def handle_interjection(self, session_id: str, req: InterjectRequest) -> Optional[Dict[str, Any]]:
        """Pause any running debate stream, persist user question, run optional Decimal sandbox math & reply."""
        session = self._repo.get_session(session_id)
        if session is None:
            return None

        # Cancel any currently streaming round before taking turn lock
        cancel_ev = self._cancel_events.get(session_id)
        if cancel_ev is not None:
            cancel_ev.set()

        packet = self._parse_packet(session)
        if req.refresh_context:
            preserved_personas = packet.active_personas
            refreshed = self._assembler.assemble(
                session_id=session.session_id,
                agenda_prompt=session.agenda_prompt,
                context_refs=session.context_refs,
                portfolio_id=packet.portfolio_id or "default",
                debate_intensity=packet.debate_intensity,
                debate_depth=packet.debate_depth,
            )
            packet = replace(refreshed, active_personas=preserved_personas)
            self._repo.update_session_context(
                session_id=session_id,
                frozen_packet_json=json.dumps(packet.to_dict(), ensure_ascii=False),
                packet_fingerprint_sha256=packet.packet_fingerprint_sha256,
            )

        active_personas = resolve_personas_from_packet(packet)
        persona_map = {p.speaker_id.value: p for p in active_personas}
        target_speaker = _detect_target_speaker(
            req.message,
            req.target_speaker_id,
            active_personas=active_personas,
        )

        # Optional deterministic Decimal calculation via AST-guarded Sandbox
        tool_events: List[RoundtableToolCallEventDTO] = []
        sandbox_calc_output: Optional[str] = None
        if self._sandbox_calc is not None and _should_run_sandbox_math(req.message, req.run_sandbox_math):
            calc_res = await asyncio.to_thread(
                self._sandbox_calc.run_what_if_calculation, req.message, packet
            )
            raw_out = str(calc_res.get("output") or "").strip()
            detail_json: Optional[Dict[str, Any]] = None
            try:
                parsed = json.loads(raw_out)
                if isinstance(parsed, dict):
                    detail_json = parsed
            except Exception:
                detail_json = {"raw_output": raw_out} if raw_out else None

            is_ok = bool(calc_res.get("is_success"))
            if is_ok and raw_out:
                sandbox_calc_output = raw_out
            tool_events.append(
                RoundtableToolCallEventDTO(
                    tool_name="python_sandbox_decimal_calc",
                    status="completed" if is_ok else "failed",
                    elapsed_ms=int(calc_res.get("duration_ms") or 0),
                    summary="คำนวณจำลองตัวเลขด้วย Python Decimal สำเร็จ"
                    if is_ok
                    else "การคำนวณจำลองไม่สำเร็จ ใช้บริบทเดิมตอบแทน",
                    detail_json=detail_json,
                )
            )

        async with self._get_lock(session_id):
            existing_turns = self._repo.get_turns(session_id)
            history_payloads = [_turn_model_to_payload(t) for t in existing_turns]

            user_seq = self._repo.get_next_seq(session_id)
            user_turn = self._repo.append_turn(
                turn_id=f"turn_{uuid.uuid4().hex[:12]}",
                session_id=session_id,
                seq=user_seq,
                speaker_id=SpeakerId.USER.value,
                round_stage=RoundStage.USER_INTERJECTION.value,
                replying_to=target_speaker,
                spoken_dialogue=req.message.strip(),
                evidence_bullets=[],
                stance_tag="REVIEW",
                audio_cache_key=None,
            )
            history_payloads.append(_turn_model_to_payload(user_turn))

            reply_payload = await self._engine.generate_interjection_reply(
                user_message=req.message,
                target_speaker_id=target_speaker,
                packet=packet,
                history_turns=history_payloads,
                sandbox_calc_output=sandbox_calc_output,
                verifier=self._verifier,
            )

            target_voice = persona_map[target_speaker].voice_name if target_speaker in persona_map else None
            audio_res = await self._tts.synthesize_turn_audio(
                text=reply_payload.spoken_dialogue,
                speaker_id=target_speaker,
                enabled=req.voice_enabled,
                voice_name=target_voice,
            )

            reply_seq = self._repo.get_next_seq(session_id)
            reply_turn = self._repo.append_turn(
                turn_id=f"turn_{uuid.uuid4().hex[:12]}",
                session_id=session_id,
                seq=reply_seq,
                speaker_id=target_speaker,
                round_stage=RoundStage.INTERJECTION_REPLY.value,
                replying_to=SpeakerId.USER.value,
                spoken_dialogue=reply_payload.spoken_dialogue,
                evidence_bullets=reply_payload.evidence_bullets,
                stance_tag=reply_payload.stance_tag,
                audio_cache_key=audio_res.audio_cache_key,
                sizing_posture=reply_payload.sizing_posture,
            )

        return {
            "session_id": session_id,
            "user_turn": _turn_model_to_dto(user_turn, audio_status="disabled").model_dump(),
            "reply_turn": _turn_model_to_dto(reply_turn, audio_status=audio_res.audio_status).model_dump(),
            "tool_calls": [te.model_dump() for te in tool_events],
            "context_refreshed": req.refresh_context,
        }

    def get_audio_bytes(self, audio_cache_key: str) -> Optional[bytes]:
        return self._tts.get_audio_bytes(audio_cache_key)

    async def export_youtube_pack(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Assemble full-episode WAV, SRT subtitles & YouTube Production Kit ZIP for a session."""
        session = self._repo.get_session(session_id)
        if session is None:
            return None

        packet = self._parse_packet(session)
        active_personas = resolve_personas_from_packet(packet)
        persona_map = {p.speaker_id.value: p for p in active_personas}
        turns = self._repo.get_turns(session_id)

        audio_bytes_by_key: Dict[str, bytes] = {}
        export_turns: List[RoundtableTurn] = []
        for t in turns:
            cache_key = t.audio_cache_key
            raw_bytes: Optional[bytes] = self._tts.get_audio_bytes(cache_key) if cache_key else None
            if (not raw_bytes) and t.speaker_id != SpeakerId.USER.value and (t.spoken_dialogue or "").strip():
                voice_override = (
                    persona_map[t.speaker_id].voice_name if t.speaker_id in persona_map else None
                )
                try:
                    synth = await self._tts.synthesize_turn_audio(
                        text=t.spoken_dialogue,
                        speaker_id=t.speaker_id,
                        enabled=True,
                        voice_name=voice_override,
                    )
                    if synth.audio_cache_key:
                        raw_bytes = self._tts.get_audio_bytes(synth.audio_cache_key)
                        if raw_bytes:
                            cache_key = synth.audio_cache_key
                            self._repo.update_turn_audio_key(t.turn_id, synth.audio_cache_key)
                except Exception as exc:
                    log.debug("YouTube pack TTS backfill failed for turn %s: %s", t.turn_id, exc)
            if cache_key and raw_bytes:
                audio_bytes_by_key[cache_key] = raw_bytes
            export_turns.append(replace(t, audio_cache_key=cache_key) if cache_key != t.audio_cache_key else t)

        playbook_dict: Optional[Dict[str, Any]] = None
        if session.decision_playbook_json:
            try:
                parsed = json.loads(session.decision_playbook_json)
                if isinstance(parsed, dict):
                    playbook_dict = parsed
            except Exception:
                playbook_dict = None

        pack = build_youtube_production_pack(
            session_id=session.session_id,
            title=session.title,
            primary_ticker=session.primary_ticker,
            agenda_prompt=session.agenda_prompt,
            final_rating=session.final_rating,
            turns=export_turns,
            audio_bytes_by_key=audio_bytes_by_key,
            packet=packet,
            playbook_dict=playbook_dict,
        )
        safe_slug = re.sub(
            r"[^A-Za-z0-9_-]+",
            "-",
            (session.primary_ticker or session.session_id).strip(),
        ).strip("-") or session.session_id
        pack["filename"] = f"roundtable_youtube_{safe_slug}_{session.session_id[:12]}.zip"
        return pack

    def save_to_vault(
        self, session_id: str, req: SaveToVaultRequest
    ) -> Optional[SaveToVaultResponseDTO]:
        if self._vault_writer is None:
            return SaveToVaultResponseDTO(ok=False, message="Vault writer port is not configured")
        session = self._repo.get_session(session_id)
        if session is None:
            return None

        turns = self._repo.get_turns(session_id)
        packet = self._parse_packet(session)
        active_personas = resolve_personas_from_packet(packet)
        persona_map = {p.speaker_id.value: p for p in active_personas}
        active_panel_names = ", ".join(
            f"{p.display_name} ({p.voice_name})"
            for p in sorted(active_personas, key=lambda s: s.debate_order)
            if p.enabled or p.speaker_id == SpeakerId.HERMES_MODERATOR
        )
        playbook: Optional[DistilledDecisionPlaybook] = None
        if session.decision_playbook_json:
            try:
                playbook = DistilledDecisionPlaybook.model_validate_json(session.decision_playbook_json)
            except Exception:
                playbook = None

        title = (req.note_title or session.title or f"Guru Roundtable {session.session_id}").strip()
        lines: List[str] = [
            f"# {title}",
            "",
            f"- **Session ID:** `{session.session_id}`",
            f"- **Primary Ticker:** `{session.primary_ticker or 'Multi-Asset / Macro'}`",
            f"- **Final Rating:** `{session.final_rating}`",
            f"- **Active Panel:** {active_panel_names}",
            f"- **Agenda:** {session.agenda_prompt}",
            "",
        ]
        if playbook is not None:
            lines.extend(
                [
                    "## Hermes Executive Verdict",
                    playbook.executive_verdict,
                    "",
                    "### Consensus Points",
                    *[f"- {p}" for p in playbook.consensus_points],
                    "",
                    "### Kill-Switches & Risk Guardrails",
                    *[f"- {k}" for k in playbook.kill_switches],
                    "",
                ]
            )
        lines.append("## Discussion Highlights")
        for t in turns:
            if t.speaker_id == SpeakerId.USER.value:
                speaker_label = "ผู้ใช้ (Raise Hand)"
            elif t.speaker_id in persona_map:
                speaker_label = persona_map[t.speaker_id].display_name
            else:
                speaker_label = t.speaker_id
            lines.append(f"- **[{t.round_stage}] {speaker_label} ({t.stance_tag}):** {t.spoken_dialogue}")

        markdown_content = "\n".join(lines)
        tags = ["roundtable", "guru_council"]
        if session.primary_ticker:
            tags.append(session.primary_ticker.lower())

        res = self._vault_writer.save_roundtable_note(
            title=title,
            markdown_content=markdown_content,
            ticker=session.primary_ticker,
            tags=tags,
        )

        thesis_claim_id: Optional[str] = None
        if req.also_update_thesis and session.primary_ticker and self._thesis_updater is not None:
            upd = self.update_thesis(session_id, UpdateThesisFromPlaybookRequest())
            if upd:
                thesis_claim_id = str(upd.get("claim_id") or "") or None

        return SaveToVaultResponseDTO(
            ok=bool(res.get("ok", True)),
            vault_path=str(res.get("vault_path") or ""),
            thesis_claim_id=thesis_claim_id,
            message="บันทึกลง Obsidian Vault สำเร็จ",
        )

    def update_thesis(
        self, session_id: str, req: UpdateThesisFromPlaybookRequest
    ) -> Optional[Dict[str, Any]]:
        if self._thesis_updater is None:
            raise ValueError("Thesis updater port is not configured")
        session = self._repo.get_session(session_id)
        if session is None:
            return None

        ticker = (req.ticker or session.primary_ticker or "").strip().upper()
        if not ticker:
            raise ValueError("Ticker is required to update Active Stock Thesis")

        playbook: Optional[DistilledDecisionPlaybook] = None
        if session.decision_playbook_json:
            try:
                playbook = DistilledDecisionPlaybook.model_validate_json(session.decision_playbook_json)
            except Exception:
                playbook = None

        claim_statement = (
            (req.claim_statement or "").strip()
            or (playbook.executive_verdict if playbook else "")
            or session.agenda_prompt
        )
        kill_switches = list(playbook.kill_switches) if playbook else []
        return self._thesis_updater.revise_thesis_from_roundtable(
            ticker=ticker,
            claim_statement=claim_statement,
            kill_switches=kill_switches,
        )

    def dispatch_missing_ticker(self, ticker: str) -> Dict[str, Any]:
        clean = ticker.strip().upper()
        if not clean:
            raise ValueError("Ticker must not be empty")
        if self._providers is None:
            return {"ok": False, "ticker": clean, "job_id": None, "message": "Providers not configured"}
        job_id = self._providers.dispatch_equity_research_if_needed(clean)
        return {
            "ok": bool(job_id),
            "ticker": clean,
            "job_id": job_id,
            "message": f"ส่งคำสั่งวิเคราะห์หุ้น {clean} เข้าคิวเรียบร้อย" if job_id else f"ไม่สามารถส่งงานวิเคราะห์ {clean} ได้",
        }

    def get_guru_config_catalog(self) -> RoundtableGuruConfigCatalogDTO:
        saved = self._repo.list_guru_profiles()
        try:
            merged = merge_persona_overrides(GURU_PERSONA_CATALOG, saved, None)
        except Exception:
            merged = GURU_PERSONA_CATALOG

        prioritized: Tuple[str, ...] = ()
        if self._providers is not None:
            try:
                axis = self._providers.get_confirmed_investment_axis("default") or {}
                complete = axis.get("complete_sections") if isinstance(axis.get("complete_sections"), dict) else axis
                rms = (complete or {}).get("role_models") or []
                if rms:
                    prioritized = match_prioritized_speakers(rms)
            except Exception:
                prioritized = ()

        return RoundtableGuruConfigCatalogDTO(
            personas=_build_persona_dtos(prioritized, personas=merged),
            voices=[VoiceCatalogItemDTO(**v.to_dict()) for v in GEMINI_VOICE_CATALOG],
            presets=[
                GuruPresetItemDTO(
                    **p.to_dict(),
                    **get_seat_ui_context(p.speaker_id.value, p.preset_id),
                )
                for p in GURU_PRESET_CATALOG
            ],
        )

    def update_guru_profiles(self, req: UpdateGuruProfilesRequest) -> RoundtableGuruConfigCatalogDTO:
        saved = self._repo.list_guru_profiles()
        updates = [p.model_dump(exclude_none=True) for p in req.profiles]
        merged = merge_persona_overrides(GURU_PERSONA_CATALOG, saved, updates)
        self._repo.upsert_guru_profiles([spec.to_dict() for spec in merged])
        return self.get_guru_config_catalog()

    def reset_guru_profiles(self, req: ResetGuruProfilesRequest) -> RoundtableGuruConfigCatalogDTO:
        self._repo.reset_guru_profiles(speaker_id=req.speaker_id)
        return self.get_guru_config_catalog()

    async def preview_voice(self, req: VoicePreviewRequest) -> VoicePreviewResponseDTO:
        voice_opt = VOICE_BY_NAME.get(req.voice_name)
        if voice_opt is None:
            raise ValueError(f"Unsupported voice_name '{req.voice_name}'")
        sample_text = (req.sample_text or "").strip() or voice_opt.default_sample_text
        res = await self._tts.synthesize_turn_audio(
            text=sample_text,
            speaker_id=req.speaker_id,
            enabled=True,
            voice_name=req.voice_name,
        )
        return VoicePreviewResponseDTO(
            audio_status=res.audio_status,  # type: ignore[arg-type]
            audio_cache_key=res.audio_cache_key,
            sample_text=sample_text,
            voice_name=req.voice_name,
            pitch=req.pitch,
            rate=req.rate,
            error_message=res.error_message,
        )

