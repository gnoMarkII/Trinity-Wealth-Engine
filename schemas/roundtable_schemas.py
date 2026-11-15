"""Pydantic v2 Schemas & DTOs for Trinity Guru Roundtable Studio."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator

from core.roundtable.models import (
    ALLOWED_VOICE_NAMES,
    normalize_5tier_stance,
    sanitize_custom_persona_text,
)


RoundtableStanceType = Literal["BUY", "OVERWEIGHT", "HOLD", "UNDERWEIGHT", "SELL", "REVIEW"]
RoundStageType = Literal[
    "stage_intro",
    "opening",
    "rebuttal",
    "crossfire",
    "interjection_reply",
    "synthesis",
    "user_interjection",
]
DebateIntensityType = Literal["balanced", "heated", "crossfire"]
DebateDepthType = Literal["concise", "deep_youtube"]
CoreGuruIdType = Literal[
    "hermes_moderator",
    "value_guru",
    "quality_guru",
    "growth_guru",
    "macro_tactical_guru",
    "skeptic_guru",
]
SpeakerIdType = Literal[
    "hermes_moderator",
    "value_guru",
    "quality_guru",
    "growth_guru",
    "macro_tactical_guru",
    "skeptic_guru",
    "user",
]
AudioStatusType = Literal["ready", "cached", "fallback_web_speech", "disabled"]


class RoundtableEvidenceBullet(BaseModel):
    tag: Literal["FACT", "MATH", "INFERRED"] = Field(default="FACT")
    text: str = Field(..., min_length=1)


class RoundtableTurnPayload(BaseModel):
    """Structured turn output produced by a Guru or Hermes Moderator."""

    speaker_id: SpeakerIdType
    round_stage: RoundStageType
    replying_to_speaker_id: Optional[str] = None
    spoken_dialogue: str = Field(..., min_length=1, description="2-4 natural spoken sentences cleanly prepared for TTS")
    evidence_bullets: List[str] = Field(
        default_factory=list,
        description="2-3 short evidence bullets prefixed with [FACT], [MATH], or [INFERRED]",
    )
    stance_tag: RoundtableStanceType = Field(default="REVIEW")
    sizing_posture: Optional[str] = Field(
        default=None,
        description="Round 2 risk/sizing posture e.g. Aggressive Accumulate / Conservative Staged Dip / Defensive Trim/Wait",
    )
    audio_cache_key: Optional[str] = None
    audio_status: AudioStatusType = Field(default="disabled")

    @field_validator("stance_tag", mode="before")
    @classmethod
    def _normalize_stance_tag(cls, v: Any) -> str:
        return normalize_5tier_stance(str(v or "REVIEW"))


class ConsensusTensionAxis(BaseModel):
    topic: str
    bull_side_speaker: str
    bull_argument: str
    bear_side_speaker: str
    bear_argument: str


class PersonalAxisCheckItem(BaseModel):
    dimension: str
    status: Literal["aligned", "caution", "conflict", "not_set"]
    detail: str


class ActionPlaybookOption(BaseModel):
    option_id: str
    label: str
    speaker_source: str
    conviction_level: int = Field(ge=0, le=3, default=2)
    action_steps: List[str] = Field(default_factory=list)
    entry_zone: Optional[str] = None
    sizing_note: Optional[str] = None


class ActionPlaybookSpec(BaseModel):
    primary_action: ActionPlaybookOption
    alternatives: List[ActionPlaybookOption] = Field(default_factory=list)


class DistilledDecisionPlaybook(BaseModel):
    """Right-hand Decision Distillation card synthesized by Hermes Moderator."""

    final_rating: RoundtableStanceType = Field(default="REVIEW")
    action_stance: str = Field(default="REVIEW")
    core_conviction_score: int = Field(ge=0, le=3, default=1)
    executive_verdict: str
    guru_stance_breakdown: Dict[str, RoundtableStanceType] = Field(default_factory=dict)
    consensus_points: List[str] = Field(default_factory=list)
    tension_axes: List[ConsensusTensionAxis] = Field(default_factory=list)
    personal_axis_alignment: List[PersonalAxisCheckItem] = Field(default_factory=list)
    portfolio_context_summary: str = ""
    action_playbook: ActionPlaybookSpec
    kill_switches: List[str] = Field(default_factory=list)
    reflection_summary: Optional[str] = None

    @field_validator("final_rating", mode="before")
    @classmethod
    def _normalize_final_rating(cls, v: Any) -> str:
        return normalize_5tier_stance(str(v or "REVIEW"))

    @field_validator("guru_stance_breakdown", mode="before")
    @classmethod
    def _normalize_breakdown(cls, v: Any) -> Dict[str, str]:
        if not isinstance(v, dict):
            return {}
        return {str(k): normalize_5tier_stance(str(val)) for k, val in v.items()}


class PastScorecardDTO(BaseModel):
    previous_session_id: str
    previous_date: str
    primary_ticker: str
    benchmark_ticker: str
    entry_price: Optional[str] = None
    settlement_price: Optional[str] = None
    raw_return_pct: Optional[str] = None
    benchmark_return_pct: Optional[str] = None
    alpha_pct: Optional[str] = None
    previous_final_rating: RoundtableStanceType = "REVIEW"
    outcome_label: str = "pending"
    reflection_lesson: Optional[str] = None
    settlement_status: str = "pending"


class UpdateGuruPersonaInput(BaseModel):
    speaker_id: CoreGuruIdType
    display_name: Optional[str] = Field(default=None, max_length=60)
    short_name: Optional[str] = Field(default=None, max_length=24)
    archetype: Optional[str] = Field(default=None, max_length=80)
    speaking_style: Optional[str] = Field(default=None, max_length=200)
    custom_directive: Optional[str] = Field(default=None, max_length=500)
    preset_id: Optional[str] = Field(default=None, max_length=48)
    voice_name: Optional[str] = None
    pitch: Optional[float] = Field(default=None, ge=0.75, le=1.25)
    rate: Optional[float] = Field(default=None, ge=0.80, le=1.25)
    accent_color: Optional[str] = Field(default=None, pattern=r"^#[0-9A-Fa-f]{6}$")
    emoji: Optional[str] = Field(default=None, max_length=8)
    enabled: Optional[bool] = None
    debate_order: Optional[int] = Field(default=None, ge=0, le=99)

    @field_validator("voice_name")
    @classmethod
    def _validate_voice_name(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        clean = v.strip()
        if clean not in ALLOWED_VOICE_NAMES:
            raise ValueError(f"Unsupported Gemini voice_name '{clean}'")
        return clean

    @field_validator("display_name")
    @classmethod
    def _sanitize_display_name(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        return sanitize_custom_persona_text(v, 60)

    @field_validator("short_name")
    @classmethod
    def _sanitize_short_name(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        return sanitize_custom_persona_text(v, 24)

    @field_validator("archetype")
    @classmethod
    def _sanitize_archetype(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        return sanitize_custom_persona_text(v, 80)

    @field_validator("speaking_style")
    @classmethod
    def _sanitize_speaking_style(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        return sanitize_custom_persona_text(v, 200)

    @field_validator("custom_directive")
    @classmethod
    def _sanitize_custom_directive(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        return sanitize_custom_persona_text(v, 500)

    @field_validator("preset_id")
    @classmethod
    def _sanitize_preset_id(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        return sanitize_custom_persona_text(v, 48)

    @field_validator("emoji")
    @classmethod
    def _sanitize_emoji(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        return sanitize_custom_persona_text(v, 8)


class CreateRoundtableSessionRequest(BaseModel):
    title: Optional[str] = None
    agenda_prompt: str = Field(..., min_length=1, description="Agenda or question for the Guru Roundtable")
    context_refs: List[str] = Field(default_factory=list, description="Attached context cards e.g. $NVDA, @Portfolio, @Macro, @Digest, @Essence")
    portfolio_id: str = Field(default="default")
    proceed_without_missing_tickers: bool = Field(
        default=False,
        description="If true, bypass missing ticker prerequisite block and run with available context only.",
    )
    persona_overrides: Optional[List[UpdateGuruPersonaInput]] = Field(
        default=None,
        description="Optional per-session guru persona & voice overrides",
    )
    debate_intensity: DebateIntensityType = Field(
        default="crossfire",
        description="Adversarial intensity: balanced (2 rounds) | heated (no soft agreement + 1 crossfire duel) | crossfire (2 duels full clash)",
    )
    debate_depth: DebateDepthType = Field(
        default="deep_youtube",
        description="Dialogue thoroughness: concise (2-4 sentences) | deep_youtube (Hermes opening hook + 4-7 sentences/turn YouTube deep dive)",
    )


class RoundtableStreamRequest(BaseModel):
    voice_enabled: bool = Field(default=True)
    refresh_context_on_drift: bool = Field(default=False)


class InterjectRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User interjection message (may mention @Guru)")
    target_speaker_id: Optional[SpeakerIdType] = Field(default=None, description="Optional targeted guru to respond first")
    voice_enabled: bool = Field(default=True)
    run_sandbox_math: bool = Field(
        default=False,
        description="Force Python Sandbox Decimal calculation if numerical scenario is requested.",
    )
    refresh_context: bool = Field(default=False, description="Refresh frozen context packet before answering if drift detected.")


class SaveToVaultRequest(BaseModel):
    note_title: Optional[str] = None
    also_update_thesis: bool = Field(default=False)


class UpdateThesisFromPlaybookRequest(BaseModel):
    ticker: Optional[str] = None
    claim_statement: Optional[str] = None


class GuruPersonaDTO(BaseModel):
    speaker_id: str
    display_name: str
    short_name: str
    archetype: str
    voice_name: str
    accent_color: str
    is_prioritized_role_model: bool = False
    emoji: str = "🎙️"
    pitch: float = 1.0
    rate: float = 1.0
    enabled: bool = True
    debate_order: int = 1
    preset_id: str = "default"
    preset_label: str = ""
    speaking_style: str = ""
    custom_directive: str = ""
    role_lens_label: str = ""
    focus_summary: str = ""
    primary_evidence_labels: List[str] = Field(default_factory=list)
    analytical_rubric: str = ""


class VoiceCatalogItemDTO(BaseModel):
    voice_name: str
    gender_tone: str
    label_th: str
    recommended_for: str
    default_pitch: float
    default_rate: float
    default_sample_text: str


class GuruPresetItemDTO(BaseModel):
    preset_id: str
    speaker_id: str
    label_th: str
    display_name: str
    short_name: str
    archetype: str
    speaking_style: str
    custom_directive: str = ""
    recommended_voice: str
    emoji: str
    role_lens_label: str = ""
    focus_summary: str = ""
    primary_evidence_labels: List[str] = Field(default_factory=list)
    analytical_rubric: str = ""


class RoundtableGuruConfigCatalogDTO(BaseModel):
    personas: List[GuruPersonaDTO]
    voices: List[VoiceCatalogItemDTO]
    presets: List[GuruPresetItemDTO]


class UpdateGuruProfilesRequest(BaseModel):
    profiles: List[UpdateGuruPersonaInput] = Field(..., min_length=1)


class ResetGuruProfilesRequest(BaseModel):
    speaker_id: Optional[CoreGuruIdType] = None


class VoicePreviewRequest(BaseModel):
    speaker_id: CoreGuruIdType
    voice_name: str
    sample_text: Optional[str] = Field(default=None, max_length=280)
    pitch: float = Field(default=1.0, ge=0.75, le=1.25)
    rate: float = Field(default=1.0, ge=0.80, le=1.25)

    @field_validator("voice_name")
    @classmethod
    def _validate_preview_voice(cls, v: str) -> str:
        clean = (v or "").strip()
        if clean not in ALLOWED_VOICE_NAMES:
            raise ValueError(f"Unsupported Gemini voice_name '{clean}'")
        return clean

    @field_validator("sample_text")
    @classmethod
    def _sanitize_sample_text(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        return sanitize_custom_persona_text(v, 280)


class VoicePreviewResponseDTO(BaseModel):
    audio_status: AudioStatusType
    audio_cache_key: Optional[str] = None
    sample_text: str
    voice_name: str
    pitch: float = 1.0
    rate: float = 1.0
    error_message: Optional[str] = None


class RoundtableTurnDTO(BaseModel):
    turn_id: str
    session_id: str
    seq: int
    speaker_id: str
    round_stage: str
    replying_to: Optional[str] = None
    spoken_dialogue: str
    evidence_bullets: List[str] = Field(default_factory=list)
    stance_tag: RoundtableStanceType = "REVIEW"
    sizing_posture: Optional[str] = None
    audio_cache_key: Optional[str] = None
    audio_status: AudioStatusType = "disabled"
    created_at: float


class RoundtableSessionSummaryDTO(BaseModel):
    session_id: str
    title: str
    primary_ticker: Optional[str] = None
    benchmark_ticker: Optional[str] = None
    agenda_prompt: str
    context_refs: List[str] = Field(default_factory=list)
    final_rating: RoundtableStanceType = "REVIEW"
    settlement_status: str = "pending"
    settled_return_pct: Optional[str] = None
    settled_alpha_pct: Optional[str] = None
    status: str = "active"
    created_at: float
    updated_at: float


class DebatePillarStatusDTO(BaseModel):
    pillar: str
    raised_by: List[str] = Field(default_factory=list)
    challenged_by: List[str] = Field(default_factory=list)
    is_challenged: bool = False


class DebatePillarCoverageDTO(BaseModel):
    pillars: List[DebatePillarStatusDTO] = Field(default_factory=list)
    raised_count: int = 0
    challenged_count: int = 0
    unchallenged_pillars: List[str] = Field(default_factory=list)


class RoundtableSessionDetailDTO(BaseModel):
    session: RoundtableSessionSummaryDTO
    turns: List[RoundtableTurnDTO] = Field(default_factory=list)
    decision_playbook: Optional[DistilledDecisionPlaybook] = None
    past_scorecard: Optional[PastScorecardDTO] = None
    missing_prerequisite_tickers: List[str] = Field(default_factory=list)
    prioritized_speakers: List[str] = Field(default_factory=list)
    user_role_models: List[str] = Field(default_factory=list)
    portfolio_attachment_state: str = "NOT_ATTACHED_DO_NOT_ASSUME"
    context_drift_detected: bool = False
    packet_fingerprint_sha256: str = ""
    debate_intensity: DebateIntensityType = "crossfire"
    debate_depth: DebateDepthType = "deep_youtube"
    personas: List[GuruPersonaDTO] = Field(default_factory=list)
    pillar_coverage: Optional[DebatePillarCoverageDTO] = None


class SaveToVaultResponseDTO(BaseModel):
    ok: bool
    vault_path: Optional[str] = None
    thesis_claim_id: Optional[str] = None
    message: str


class RoundtableToolCallEventDTO(BaseModel):
    tool_name: str
    status: Literal["running", "completed", "failed"]
    elapsed_ms: Optional[int] = None
    summary: str = ""
    detail_json: Optional[Dict[str, Any]] = None
