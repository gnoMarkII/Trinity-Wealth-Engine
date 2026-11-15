"""Hexagonal Ports for Trinity Guru Roundtable application layer."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

from core.roundtable.models import (
    FrozenContextPacket,
    RoundtableSession,
    RoundtableTurn,
)
from schemas.roundtable_schemas import (
    DistilledDecisionPlaybook,
    RoundtableTurnPayload,
)


@dataclass(frozen=True)
class AudioSynthesisResult:
    audio_cache_key: Optional[str]
    audio_status: str  # "ready" | "cached" | "fallback_web_speech" | "disabled"
    mime_type: str = "audio/wav"
    error_message: Optional[str] = None


@dataclass(frozen=True)
class RoundtableVerificationResult:
    is_valid: bool
    issues: Tuple[str, ...] = ()
    retry_feedback: str = ""


class RoundtableSessionRepoPort(Protocol):
    """Port for SQLite persistence of roundtable sessions and turns."""

    def create_session(
        self,
        *,
        session_id: str,
        title: str,
        primary_ticker: Optional[str],
        benchmark_ticker: Optional[str],
        entry_price_at_session: Optional[str],
        agenda_prompt: str,
        context_refs: Sequence[str],
        frozen_packet_json: str,
        packet_fingerprint_sha256: str,
        final_rating: str = "REVIEW",
        status: str = "active",
    ) -> RoundtableSession: ...

    def get_session(self, session_id: str) -> Optional[RoundtableSession]: ...

    def list_sessions(self, limit: int = 30) -> List[RoundtableSession]: ...

    def list_sessions_by_ticker(self, ticker: str, limit: int = 10) -> List[RoundtableSession]: ...

    def update_session_playbook(
        self,
        *,
        session_id: str,
        decision_playbook_json: str,
        final_rating: str,
        status: str = "completed",
    ) -> Optional[RoundtableSession]: ...

    def update_session_context(
        self,
        *,
        session_id: str,
        frozen_packet_json: str,
        packet_fingerprint_sha256: str,
        entry_price_at_session: Optional[str] = None,
    ) -> Optional[RoundtableSession]: ...

    def update_session_settlement(
        self,
        *,
        session_id: str,
        settlement_status: str,
        settled_return_pct: Optional[str],
        settled_alpha_pct: Optional[str],
        settled_at_date: Optional[str],
        reflection_lesson: Optional[str],
    ) -> Optional[RoundtableSession]: ...

    def append_turn(
        self,
        *,
        turn_id: str,
        session_id: str,
        seq: int,
        speaker_id: str,
        round_stage: str,
        replying_to: Optional[str],
        spoken_dialogue: str,
        evidence_bullets: Sequence[str],
        stance_tag: str,
        audio_cache_key: Optional[str],
        sizing_posture: Optional[str] = None,
    ) -> RoundtableTurn: ...

    def update_turn_audio_key(self, turn_id: str, audio_cache_key: str) -> None: ...

    def get_turns(self, session_id: str) -> List[RoundtableTurn]: ...

    def get_next_seq(self, session_id: str) -> int: ...

    def list_guru_profiles(self) -> List[Dict[str, Any]]: ...

    def upsert_guru_profiles(self, profiles: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]: ...

    def reset_guru_profiles(self, speaker_id: Optional[str] = None) -> None: ...


class RoundtableTTSPort(Protocol):
    """Port for synthesizing and caching spoken audio for Roundtable turns."""

    async def synthesize_turn_audio(
        self,
        *,
        text: str,
        speaker_id: str,
        enabled: bool = True,
        voice_name: Optional[str] = None,
    ) -> AudioSynthesisResult: ...

    def get_audio_bytes(self, audio_cache_key: str) -> Optional[bytes]: ...


class RoundtableDebateEnginePort(Protocol):
    """Port for generating multi-guru turns, user interjection replies, and Hermes synthesis."""

    async def generate_hermes_stage_intro(
        self,
        *,
        packet: FrozenContextPacket,
        ordered_gurus: Sequence[str],
        verifier: Optional["RoundtableVerifierPort"] = None,
    ) -> RoundtableTurnPayload:
        return await self.generate_guru_turn(
            speaker_id="hermes_moderator",
            round_stage="stage_intro",
            packet=packet,
            prior_turns=(),
            unspoken_speakers=ordered_gurus,
            verifier=verifier,
        )

    async def generate_guru_turn(
        self,
        *,
        speaker_id: str,
        round_stage: str,
        packet: FrozenContextPacket,
        prior_turns: Sequence[RoundtableTurnPayload],
        unspoken_speakers: Sequence[str],
        verifier: Optional["RoundtableVerifierPort"] = None,
    ) -> RoundtableTurnPayload: ...

    async def generate_hermes_synthesis(
        self,
        *,
        packet: FrozenContextPacket,
        all_turns: Sequence[RoundtableTurnPayload],
        verifier: Optional["RoundtableVerifierPort"] = None,
    ) -> Tuple[RoundtableTurnPayload, DistilledDecisionPlaybook]: ...

    async def generate_interjection_reply(
        self,
        *,
        user_message: str,
        target_speaker_id: str,
        packet: FrozenContextPacket,
        history_turns: Sequence[RoundtableTurnPayload],
        sandbox_calc_output: Optional[str] = None,
        verifier: Optional["RoundtableVerifierPort"] = None,
    ) -> RoundtableTurnPayload: ...

    async def generate_reflection_lesson(
        self,
        *,
        previous_rating: str,
        raw_return_pct: Optional[str],
        alpha_pct: Optional[str],
        previous_verdict: str,
        ticker: str,
    ) -> str: ...


class RoundtableVerifierPort(Protocol):
    """Port for checking guru dialogue and Hermes playbook consistency."""

    def verify_turn(
        self,
        turn: RoundtableTurnPayload,
        packet: FrozenContextPacket,
        unspoken_speakers: Sequence[str] = (),
        prior_turns: Sequence[RoundtableTurnPayload] = (),
    ) -> RoundtableVerificationResult: ...

    def verify_playbook(self, playbook: DistilledDecisionPlaybook, packet: FrozenContextPacket) -> RoundtableVerificationResult: ...


class ContextDataProvidersPort(Protocol):
    """Port reading existing Trinity domain data without importing infrastructure in application/."""

    def resolve_instrument(self, ticker: str) -> Dict[str, Any]: ...

    def get_equity_detail(self, ticker: str) -> Optional[Dict[str, Any]]: ...

    def get_equity_position_review(self, ticker: str, portfolio_id: str = "default") -> Optional[Dict[str, Any]]: ...

    def get_equity_news(self, ticker: str) -> Optional[Dict[str, Any]]: ...

    def list_equity_notes(self, ticker: str, days: int = 30) -> Optional[Dict[str, Any]]: ...

    def read_equity_note(self, relative_path: str) -> Optional[Dict[str, Any]]: ...

    def get_macro_dashboard(self) -> Optional[Dict[str, Any]]: ...

    def get_daily_digest(self) -> Optional[Dict[str, Any]]: ...

    def get_portfolio_state(self, portfolio_id: str = "default") -> Optional[Dict[str, Any]]: ...

    def get_confirmed_investment_axis(self, portfolio_id: str = "default") -> Optional[Dict[str, Any]]: ...

    def get_financial_context(self, portfolio_id: str = "default") -> Optional[Dict[str, Any]]: ...

    def get_bucket_policy(self, portfolio_id: str = "default") -> Optional[List[Dict[str, Any]]]: ...

    def get_active_thesis(self, ticker: str) -> Optional[Dict[str, Any]]: ...

    def get_thesis_pending_exceptions(self, ticker: str) -> Optional[List[Dict[str, Any]]]: ...

    def get_promise_ledger_scores(self, ticker: str) -> Optional[Dict[str, Any]]: ...

    def get_analyst_context(self, ticker: str) -> Optional[Dict[str, Any]]: ...

    def get_insider_db_summary(self, ticker: str) -> Optional[Dict[str, Any]]: ...

    def get_portfolio_advisory_summary(self, portfolio_id: str = "default") -> Optional[Dict[str, Any]]: ...

    def get_investor_essence(self, portfolio_id: str = "default") -> Optional[Dict[str, Any]]: ...

    def dispatch_equity_research_if_needed(self, ticker: str) -> Optional[str]: ...


class SettlementPricePort(Protocol):
    """Port for deterministic historical/latest closing prices for settlement math."""

    def get_close_on_or_after(self, ticker: str, target_date_iso: str) -> Optional[Tuple[str, Decimal]]: ...

    def get_latest_close(self, ticker: str) -> Optional[Tuple[str, Decimal]]: ...


class SandboxCalcPort(Protocol):
    """Port for running Decimal Python math calculations during interjections."""

    def run_what_if_calculation(self, question: str, packet: FrozenContextPacket) -> Dict[str, Any]: ...


class VaultNoteWriterPort(Protocol):
    """Port for writing distilled roundtable summary notes into Obsidian Vault."""

    def save_roundtable_note(
        self,
        *,
        title: str,
        markdown_content: str,
        ticker: Optional[str],
        tags: Sequence[str],
    ) -> Dict[str, Any]: ...


class ThesisUpdaterPort(Protocol):
    """Port for updating Active Stock Thesis from Roundtable Decision Playbook."""

    def revise_thesis_from_roundtable(
        self,
        *,
        ticker: str,
        claim_statement: str,
        kill_switches: Sequence[str],
    ) -> Dict[str, Any]: ...



class RoundtableResearchDispatchPort(Protocol):
    """Atomic outbound port for creating/reusing a prerequisite card and approval job."""

    def dispatch_approval_with_card(
        self,
        *,
        source_key: str,
        title: str,
        instruction: str,
        flow: str,
        scope: str,
        approval_metadata: Dict[str, Any],
    ) -> str: ...
