"""Unit tests for WP-3: Guru Prompt Integrity, Gemini TTS Adapter, Debate Engine, and Sandbox Calc."""
import io
import json
import wave

import pytest

from application.roundtable.ports import RoundtableVerificationResult
from application.roundtable.service import _detect_target_speaker
from core.prompt_harness import PromptHarness
from core.roundtable.models import (
    GURU_PERSONA_CATALOG,
    ContextAttachmentState,
    FrozenContextPacket,
    InstrumentIdentityAnchor,
    PastRoundtableScorecard,
    PortfolioHoldingGuardState,
    merge_persona_overrides,
)
from schemas.roundtable_schemas import (
    ActionPlaybookOption,
    ActionPlaybookSpec,
    DistilledDecisionPlaybook,
    RoundtableTurnPayload,
)
from tools.roundtable.adapters import RoundtableSandboxCalcAdapter
from tools.roundtable.debate_engine import (
    GeminiRoundtableEngine,
    _format_packet_prompt_context,
    get_ordered_guru_speakers,
    opponent_argument_or_opening,
)
from tools.roundtable.tts_adapter import (
    GeminiTTSAdapter,
    split_text_for_tts_recovery,
    validate_and_sanitize_tts_wav,
    wrap_pcm_to_wav,
)
from validators.roundtable_verifier import RoundtableConsistencyVerifier


@pytest.mark.asyncio
async def test_roundtable_engine_rejects_invalid_retry_and_uses_verified_fallbacks():
    packet = FrozenContextPacket(
        session_id="s_verifier_fallback",
        created_at_iso="2026-11-13T00:00:00Z",
        agenda_prompt="Review NVDA",
        primary_ticker="NVDA",
    )

    def _turn(speaker_id: str, stage: str, text: str) -> RoundtableTurnPayload:
        return RoundtableTurnPayload(
            speaker_id=speaker_id,  # type: ignore[arg-type]
            round_stage=stage,  # type: ignore[arg-type]
            spoken_dialogue=text,
            evidence_bullets=["[INFERRED] Review the attached packet"],
            stance_tag="HOLD",
        )

    class _Verifier:
        @staticmethod
        def _check(text: str) -> RoundtableVerificationResult:
            if "invalid" in text:
                return RoundtableVerificationResult(False, ("issue one", "issue two"), "retry")
            if "partial" in text:
                return RoundtableVerificationResult(False, ("issue one",), "retry")
            return RoundtableVerificationResult(True)

        def verify_turn(self, turn, *_args, **_kwargs):
            return self._check(turn.spoken_dialogue)

        def verify_playbook(self, playbook, _packet):
            return self._check(playbook.executive_verdict)

    class _QueuedLLM:
        def __init__(self, outputs):
            self.outputs = list(outputs)

        def with_structured_output(self, schema, include_raw=True):
            class _Runner:
                def invoke(_self, _messages):
                    item = self.outputs.pop(0)
                    return item(schema) if callable(item) else item

            return _Runner()

    def _engine(outputs):
        queued_llm = _QueuedLLM(outputs)
        return GeminiRoundtableEngine(llm_factory=lambda *_args: queued_llm)

    verifier = _Verifier()

    intro = await _engine([
        _turn("hermes_moderator", "stage_intro", "invalid intro claim"),
        _turn("hermes_moderator", "stage_intro", "partial intro claim"),
    ]).generate_hermes_stage_intro(packet=packet, ordered_gurus=["value_guru"], verifier=verifier)
    assert _Verifier._check(intro.spoken_dialogue).is_valid
    assert "partial" not in intro.spoken_dialogue

    guru = await _engine([
        _turn("value_guru", "opening", "invalid portfolio claim"),
        _turn("value_guru", "opening", "partial portfolio claim"),
    ]).generate_guru_turn(
        speaker_id="value_guru",
        round_stage="opening",
        packet=packet,
        prior_turns=[],
        unspoken_speakers=[],
        verifier=verifier,
    )
    assert _Verifier._check(guru.spoken_dialogue).is_valid
    assert "partial" not in guru.spoken_dialogue

    interjection = await _engine([
        _turn("value_guru", "interjection_reply", "invalid reply"),
        _turn("value_guru", "interjection_reply", "partial reply"),
    ]).generate_interjection_reply(
        user_message="What is the risk?",
        target_speaker_id="value_guru",
        packet=packet,
        history_turns=[],
        verifier=verifier,
    )
    assert _Verifier._check(interjection.spoken_dialogue).is_valid
    assert interjection.replying_to_speaker_id == "user"
    assert "partial" not in interjection.spoken_dialogue

    def _playbook(text: str) -> DistilledDecisionPlaybook:
        return DistilledDecisionPlaybook(
            final_rating="HOLD",
            action_stance="HOLD",
            core_conviction_score=1,
            executive_verdict=text,
            action_playbook=ActionPlaybookSpec(
                primary_action=ActionPlaybookOption(
                    option_id="p1",
                    label="Hold",
                    speaker_source="hermes_moderator",
                    action_steps=["Monitor the evidence"],
                )
            ),
        )

    def _combined(schema, marker: str):
        return schema(
            closing_turn=_turn("hermes_moderator", "synthesis", f"closing {marker}"),
            playbook=_playbook(marker),
        )

    closing, playbook = await _engine([
        lambda schema: _combined(schema, "invalid"),
        lambda schema: _combined(schema, "partial"),
    ]).generate_hermes_synthesis(packet=packet, all_turns=[], verifier=verifier)
    assert _Verifier._check(playbook.executive_verdict).is_valid
    assert "partial" not in playbook.executive_verdict
    assert "partial" not in closing.spoken_dialogue


@pytest.mark.asyncio
async def test_synthesis_retries_invalid_closing_even_when_playbook_is_valid():
    packet = FrozenContextPacket(
        session_id="s_synthesis_closing_guard",
        created_at_iso="2026-11-13T00:00:00Z",
        agenda_prompt="Review NVDA",
        primary_ticker="NVDA",
        instrument_anchor=InstrumentIdentityAnchor(
            ticker="NVDA",
            company_name="NVIDIA Corporation",
            market="US",
            sector="Technology",
            industry="Semiconductors",
            currency="USD",
            benchmark_ticker="SPY",
            current_price="135.00",
        ),
    )
    verifier = RoundtableConsistencyVerifier()

    def _playbook(verdict: str) -> DistilledDecisionPlaybook:
        return DistilledDecisionPlaybook(
            final_rating="HOLD",
            action_stance="HOLD",
            core_conviction_score=1,
            executive_verdict=verdict,
            portfolio_context_summary=packet.portfolio_guard.summary_text,
            action_playbook=ActionPlaybookSpec(
                primary_action=ActionPlaybookOption(
                    option_id="primary",
                    label="Monitor evidence",
                    speaker_source="hermes_moderator",
                    conviction_level=1,
                    action_steps=["Monitor the next verified filing"],
                )
            ),
        )

    def _closing(spoken: str, price: str) -> RoundtableTurnPayload:
        return RoundtableTurnPayload(
            speaker_id="hermes_moderator",
            round_stage="synthesis",
            spoken_dialogue=spoken,
            evidence_bullets=[f"[FACT] Current price is {price} USD"],
            stance_tag="HOLD",
        )

    invalid_closing = _closing(
        "\u0e04\u0e38\u0e13\u0e21\u0e35 100 \u0e2b\u0e38\u0e49\u0e19; we should continue holding.",
        "999",
    )
    assert not verifier.verify_turn(invalid_closing, packet, prior_turns=[]).is_valid
    assert verifier.verify_playbook(_playbook("Monitor the evidence and risk limits."), packet).is_valid

    invalid_output = (invalid_closing, _playbook("Monitor the evidence and risk limits."))
    valid_output = (
        _closing(
            "The panel will monitor verified evidence and downside risks before changing the stance. "
            "Keep the agreed risk limits in view.",
            "135.00",
        ),
        _playbook("Monitor verified evidence and keep risk limits in place."),
    )

    class _QueueLLM:
        def __init__(self):
            self.outputs = [invalid_output, valid_output]
            self.messages = []

        def with_structured_output(self, schema, include_raw=True):
            class _Runner:
                def invoke(_self, messages):
                    self.messages.append(messages)
                    closing, playbook = self.outputs.pop(0)
                    return schema(closing_turn=closing, playbook=playbook)

            return _Runner()

    llm = _QueueLLM()
    engine = GeminiRoundtableEngine(llm_factory=lambda *_args: llm)
    closing, playbook = await engine.generate_hermes_synthesis(
        packet=packet,
        all_turns=[],
        verifier=verifier,
    )

    assert len(llm.messages) == 2
    assert any("[ROUNDTABLE CONSISTENCY VERIFIER]" in str(message.content) for message in llm.messages[1])
    assert verifier.verify_turn(closing, packet, prior_turns=[]).is_valid
    assert verifier.verify_playbook(playbook, packet).is_valid



@pytest.mark.asyncio
async def test_synthesis_retries_valid_closing_with_factual_playbook_claims():
    packet = FrozenContextPacket(
        session_id="s_synthesis_bad_playbook",
        created_at_iso="2026-11-13T00:00:00Z",
        agenda_prompt="Review NVDA",
        primary_ticker="NVDA",
        instrument_anchor=InstrumentIdentityAnchor(
            ticker="NVDA",
            company_name="NVIDIA Corporation",
            market="US",
            sector="Technology",
            industry="Semiconductors",
            currency="USD",
            benchmark_ticker="SPY",
            current_price="135.00",
        ),
    )
    verifier = RoundtableConsistencyVerifier()
    closing = RoundtableTurnPayload(
        speaker_id="hermes_moderator",
        round_stage="synthesis",
        spoken_dialogue="The panel will monitor verified evidence and downside risks before changing the stance.",
        evidence_bullets=["[FACT] Current price is 135.00 USD"],
        stance_tag="HOLD",
    )

    def _playbook(verdict: str) -> DistilledDecisionPlaybook:
        return DistilledDecisionPlaybook(
            final_rating="HOLD",
            action_stance="HOLD",
            core_conviction_score=1,
            executive_verdict=verdict,
            portfolio_context_summary=packet.portfolio_guard.summary_text,
            action_playbook=ActionPlaybookSpec(
                primary_action=ActionPlaybookOption(
                    option_id="primary",
                    label="Monitor evidence",
                    speaker_source="hermes_moderator",
                    action_steps=["Monitor the next verified filing"],
                )
            ),
        )

    invalid = _playbook("Current price is 999 USD. Build a downside scenario.")
    valid = _playbook("Monitor verified evidence and keep risk limits in place.")
    outputs = [(closing, invalid), (closing, valid)]

    class _QueueLLM:
        def __init__(self):
            self.messages = []

        def with_structured_output(self, schema, include_raw=True):
            class _Runner:
                def invoke(_self, messages):
                    self.messages.append(messages)
                    turn, playbook = outputs.pop(0)
                    return schema(closing_turn=turn, playbook=playbook)

            return _Runner()

    llm = _QueueLLM()
    engine = GeminiRoundtableEngine(llm_factory=lambda *_args: llm)
    result_turn, result_playbook = await engine.generate_hermes_synthesis(
        packet=packet,
        all_turns=[],
        verifier=verifier,
    )

    assert len(llm.messages) == 2
    retry_messages = str([message.content for message in llm.messages[1]])
    assert "[HERMES PLAYBOOK VERIFIER]" in retry_messages
    assert verifier.verify_turn(result_turn, packet, prior_turns=[]).is_valid
    assert verifier.verify_playbook(result_playbook, packet).is_valid
    assert "999" not in result_playbook.executive_verdict

@pytest.mark.asyncio
async def test_synthesis_stops_when_deterministic_closing_fallback_is_rejected():
    packet = FrozenContextPacket(
        session_id="s_synthesis_fallback_rejected",
        created_at_iso="2026-11-13T00:00:00Z",
        agenda_prompt="Review NVDA",
        primary_ticker="NVDA",
        instrument_anchor=InstrumentIdentityAnchor(
            ticker="NVDA",
            company_name="NVIDIA Corporation",
            market="US",
            sector="Technology",
            industry="Semiconductors",
            currency="USD",
            benchmark_ticker="SPY",
            current_price="135.00",
        ),
    )
    real_verifier = RoundtableConsistencyVerifier()
    failure = RoundtableVerificationResult(False, ("forced fallback rejection",), "retry")

    class _VerifierRejectingFallback:
        def verify_turn(self, turn, checked_packet, *args, **kwargs):
            fallback_prefix = "\u0e02\u0e2d\u0e1a\u0e04\u0e38\u0e13\u0e01\u0e39\u0e23\u0e39\u0e17\u0e38\u0e01\u0e17\u0e48\u0e32\u0e19"
            if turn.spoken_dialogue.startswith(fallback_prefix):
                return failure
            return real_verifier.verify_turn(turn, checked_packet, *args, **kwargs)

        def verify_playbook(self, playbook, checked_packet):
            return real_verifier.verify_playbook(playbook, checked_packet)

    def _playbook() -> DistilledDecisionPlaybook:
        return DistilledDecisionPlaybook(
            final_rating="HOLD",
            action_stance="HOLD",
            core_conviction_score=1,
            executive_verdict="Monitor verified evidence and keep risk limits in place.",
            portfolio_context_summary=packet.portfolio_guard.summary_text,
            action_playbook=ActionPlaybookSpec(
                primary_action=ActionPlaybookOption(
                    option_id="primary",
                    label="Monitor evidence",
                    speaker_source="hermes_moderator",
                    conviction_level=1,
                    action_steps=["Monitor the next verified filing"],
                )
            ),
        )

    invalid_closing = RoundtableTurnPayload(
        speaker_id="hermes_moderator",
        round_stage="synthesis",
        spoken_dialogue="\u0e04\u0e38\u0e13\u0e21\u0e35 100 \u0e2b\u0e38\u0e49\u0e19; we should continue holding.",
        evidence_bullets=["[FACT] Current price is 999 USD"],
        stance_tag="HOLD",
    )
    outputs = [(invalid_closing, _playbook()), (invalid_closing, _playbook())]

    class _QueueLLM:
        def with_structured_output(self, schema, include_raw=True):
            class _Runner:
                def invoke(_self, _messages):
                    closing, playbook = outputs.pop(0)
                    return schema(closing_turn=closing, playbook=playbook)

            return _Runner()

    engine = GeminiRoundtableEngine(llm_factory=lambda *_args: _QueueLLM())
    with pytest.raises(ValueError, match="closing fallback: forced fallback rejection"):
        await engine.generate_hermes_synthesis(
            packet=packet,
            all_turns=[],
            verifier=_VerifierRejectingFallback(),
        )


def test_guru_roundtable_prompts_pass_integrity_audit():
    harness = PromptHarness("guru_roundtable")
    issues = harness.audit_skill_integrity()
    assert issues == [], f"Prompt integrity issues found: {issues}"
    # Verify every guru file loads cleanly
    for fname in (
        "SKILL.md",
        "VALUE_GURU.md",
        "QUALITY_GURU.md",
        "GROWTH_GURU.md",
        "MACRO_TACTICAL_GURU.md",
        "SKEPTIC_GURU.md",
        "HERMES_MODERATOR.md",
    ):
        assert len(harness.get_skill_text(fname, strict=False)) > 50


def test_wrap_pcm_to_wav_produces_valid_wave_header():
    raw_pcm = b"\x00\x01" * 240  # 10ms of 16-bit mono audio
    wav_bytes = wrap_pcm_to_wav(raw_pcm)
    assert wav_bytes[:4] == b"RIFF"
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 24000
        assert wf.readframes(240) == raw_pcm
    # Idempotency when already wrapped
    assert wrap_pcm_to_wav(wav_bytes) == wav_bytes


@pytest.mark.asyncio
async def test_gemini_tts_adapter_cache_lru_and_fallback(tmp_path):
    calls = []

    def _fake_fetcher(model_name: str, voice_name: str, text: str) -> bytes:
        calls.append((model_name, voice_name, text))
        return b"\x10\x00" * 480

    adapter = GeminiTTSAdapter(
        cache_dir=tmp_path,
        max_cache_files=2,
        model_name="gemini-2.5-flash-preview-tts",
        tts_fetcher=_fake_fetcher,
    )

    r1 = await adapter.synthesize_turn_audio(text="**สวัสดี** คุณ Value [FACT]", speaker_id="growth_guru")
    assert r1.audio_status == "ready"
    assert r1.audio_cache_key is not None
    assert calls[-1][1] == "Puck"
    assert "**" not in calls[-1][2] and "[FACT]" not in calls[-1][2]

    # Cache hit on same cleaned text & speaker
    r1_cached = await adapter.synthesize_turn_audio(text="สวัสดี คุณ Value", speaker_id="growth_guru")
    assert r1_cached.audio_status == "cached"
    assert r1_cached.audio_cache_key == r1.audio_cache_key
    assert len(calls) == 1

    # Add 2 more to trigger LRU eviction down to max_cache_files=2
    r2 = await adapter.synthesize_turn_audio(text="ประโยคที่สองครับ", speaker_id="value_guru")
    r3 = await adapter.synthesize_turn_audio(text="ประโยคที่สามครับ", speaker_id="skeptic_guru")
    assert len(list(tmp_path.glob("*.wav"))) == 2
    assert r3.audio_cache_key is not None and adapter.get_audio_bytes(r3.audio_cache_key) is not None

    # Path traversal blocked
    assert adapter.get_audio_bytes("../secret") is None

    # Graceful fallback on 429 error
    def _failing_fetcher(*_):
        raise RuntimeError("429 Resource Exhausted")

    fail_adapter = GeminiTTSAdapter(cache_dir=tmp_path, tts_fetcher=_failing_fetcher)
    r_fail = await fail_adapter.synthesize_turn_audio(text="ทดสอบ Fallback", speaker_id="hermes_moderator")
    assert r_fail.audio_status == "fallback_web_speech"
    assert r_fail.audio_cache_key is None


@pytest.mark.asyncio
async def test_debate_engine_structured_and_fallback_plus_hermes_synthesis():
    class _FakeLLM:
        def __init__(self, fail_structured: bool = False):
            self._fail_structured = fail_structured

        def with_structured_output(self, schema):
            outer = self

            class _Runner:
                def invoke(self, messages):
                    if outer._fail_structured:
                        raise RuntimeError("Structured output unsupported")
                    if schema.__name__ == "_HermesCombinedOutput":
                        pb = DistilledDecisionPlaybook(
                            final_rating="OVERWEIGHT",
                            action_stance="OVERWEIGHT",
                            core_conviction_score=2,
                            executive_verdict="พื้นฐานแกร่ง ทยอยสะสมแบบแบ่งไม้",
                            guru_stance_breakdown={"value_guru": "HOLD", "growth_guru": "BUY"},
                            consensus_points=["รายได้เติบโตต่อเนื่อง"],
                            action_playbook=ActionPlaybookSpec(
                                primary_action=ActionPlaybookOption(
                                    option_id="p1",
                                    label="Staged Accumulate",
                                    speaker_source="hermes_moderator",
                                    action_steps=["แบ่งซื้อ 2 ไม้"],
                                )
                            ),
                            kill_switches=["Gross margin < 65%"],
                        )
                        ct = RoundtableTurnPayload(
                            speaker_id="hermes_moderator",
                            round_stage="synthesis",
                            spoken_dialogue="สรุปผลการหารือเราแนะนำ Overweight แบบแบ่งไม้ครับ",
                            evidence_bullets=["[FACT] มติส่วนใหญ่ให้น้ำหนักเพิ่ม"],
                            stance_tag="OVERWEIGHT",
                        )
                        return schema(closing_turn=ct, playbook=pb)
                    return RoundtableTurnPayload(
                        speaker_id="value_guru",
                        round_stage="opening",
                        spoken_dialogue="**คุณ Growth** ราคาตอนนี้สูงกว่า MOS พอสมควรนะครับ",
                        evidence_bullets=["[FACT] Current price $135 vs MOS $115"],
                        stance_tag="Accumulate",  # tests normalization!
                    )

            return _Runner()

        def invoke(self, messages):
            class _Resp:
                content = json.dumps(
                    {
                        "speaker_id": "growth_guru",
                        "round_stage": "rebuttal",
                        "replying_to_speaker_id": "value_guru",
                        "spoken_dialogue": "คุณ Value ครับ ตลาดให้ค่ากับการเติบโตของ AI ครับ",
                        "evidence_bullets": ["[FACT] Revenue +80% YoY"],
                        "stance_tag": "BUY",
                        "sizing_posture": "Conservative Staged Dip",
                    },
                    ensure_ascii=False,
                )

            return _Resp()

    packet = FrozenContextPacket(
        session_id="s_test",
        created_at_iso="2026-11-10T00:00:00Z",
        agenda_prompt="วิเคราะห์ $NVDA",
        primary_ticker="NVDA",
        instrument_anchor=InstrumentIdentityAnchor(
            ticker="NVDA",
            company_name="NVIDIA Corporation",
            market="US",
            sector="Technology",
            industry="Semiconductors",
            currency="USD",
            benchmark_ticker="SPY",
            current_price="135.00",
        ),
        portfolio_guard=PortfolioHoldingGuardState(
            attachment_state=ContextAttachmentState.ATTACHED_HELD,
            portfolio_attached=True,
            is_held=True,
            units="10.0000",
            avg_cost="110.00",
            portfolio_total_nav_thb="1000000.00",
            summary_text="[PORTFOLIO_STATE: ATTACHED_HELD] ถืออยู่ 10 หุ้น",
        ),
    )

    guard_txt = opponent_argument_or_opening("value_guru", "opening", [], ["quality_guru", "growth_guru"])
    assert "ห้ามสมมติหรืออ้างคำพูดแทนกูรูคนอื่น" in guard_txt

    engine = GeminiRoundtableEngine(llm_factory=lambda *_: _FakeLLM(fail_structured=False))
    verifier = RoundtableConsistencyVerifier()

    turn1 = await engine.generate_guru_turn(
        speaker_id="value_guru",
        round_stage="opening",
        packet=packet,
        prior_turns=[],
        unspoken_speakers=["growth_guru"],
        verifier=verifier,
    )
    assert "**" not in turn1.spoken_dialogue
    assert turn1.stance_tag == "OVERWEIGHT"

    closing, playbook = await engine.generate_hermes_synthesis(
        packet=packet, all_turns=[turn1], verifier=verifier
    )
    assert closing.speaker_id == "hermes_moderator"
    assert playbook.final_rating == "OVERWEIGHT"

    # Fallback test when structured output fails
    fallback_engine = GeminiRoundtableEngine(llm_factory=lambda *_: _FakeLLM(fail_structured=True))
    turn2 = await fallback_engine.generate_guru_turn(
        speaker_id="growth_guru",
        round_stage="rebuttal",
        packet=packet,
        prior_turns=[turn1],
        unspoken_speakers=[],
    )
    assert turn2.speaker_id == "growth_guru"
    assert turn2.stance_tag == "BUY"

    # Interjection + sandbox math calculation test
    sandbox_adapter = RoundtableSandboxCalcAdapter()
    calc = sandbox_adapter.run_what_if_calculation("ถ้าซื้อเพิ่มอีก 5 หุ้น ต้นทุนเฉลี่ยใหม่จะเป็นเท่าไหร่?", packet)
    assert calc["is_success"] is True
    parsed_calc = json.loads(calc["output"])
    assert parsed_calc["blended_avg_cost"] == "118.33"


@pytest.mark.asyncio
async def test_tts_custom_voice_override_and_custom_persona_prompts(tmp_path):
    voice_calls = []

    def _voice_capturing_fetcher(model_name: str, voice_name: str, text: str) -> bytes:
        voice_calls.append((voice_name, text))
        return b"\x02\x00" * 240

    adapter = GeminiTTSAdapter(cache_dir=tmp_path, tts_fetcher=_voice_capturing_fetcher)
    res_zephyr = await adapter.synthesize_turn_audio(
        text="ทดสอบเสียงพิเศษ",
        speaker_id="value_guru",
        voice_name="Zephyr",
    )
    res_charon = await adapter.synthesize_turn_audio(
        text="ทดสอบเสียงพิเศษ",
        speaker_id="value_guru",
        voice_name="Charon",
    )
    assert res_zephyr.audio_status == "ready"
    assert res_charon.audio_status == "ready"
    assert res_zephyr.audio_cache_key != res_charon.audio_cache_key
    assert [c[0] for c in voice_calls] == ["Zephyr", "Charon"]

    # Custom persona override + disabled skeptic triggers Hermes Devil's Advocate fallback
    custom_personas = merge_persona_overrides(
        GURU_PERSONA_CATALOG,
        None,
        [
            {
                "speaker_id": "value_guru",
                "display_name": "Deep Value (Graham)",
                "short_name": "Graham",
                "archetype": "Benjamin Graham Deep Value",
                "speaking_style": "เข้มงวดเรื่อง Margin of Safety 30%",
                "custom_directive": "ตรวจ net-net balance sheet ก่อนเสมอ",
                "voice_name": "Iapetus",
                "preset_id": "value_graham_deep",
            },
            {
                "speaker_id": "macro_tactical_guru",
                "debate_order": 1,
            },
            {
                "speaker_id": "skeptic_guru",
                "enabled": False,
            },
        ],
    )
    packet = FrozenContextPacket(
        session_id="s_custom_engine",
        created_at_iso="2026-11-11T00:00:00Z",
        agenda_prompt="วิเคราะห์ $NVDA",
        primary_ticker="NVDA",
        active_personas=tuple(p.to_dict() for p in custom_personas),
    )

    ordered = get_ordered_guru_speakers(packet)
    assert ordered[0] == "macro_tactical_guru"
    assert "skeptic_guru" not in ordered

    guard_custom = opponent_argument_or_opening(
        "macro_tactical_guru",
        "opening",
        [],
        ["value_guru"],
        packet=packet,
    )
    assert "Graham (value_guru)" in guard_custom

    captured_prompts = []

    class _PromptCapturingLLM:
        def with_structured_output(self, schema):
            class _Runner:
                def invoke(self, messages):
                    captured_prompts.append((schema.__name__, messages[0].content, messages[1].content))
                    if schema.__name__ == "_HermesCombinedOutput":
                        return schema(
                            closing_turn=RoundtableTurnPayload(
                                speaker_id="hermes_moderator",
                                round_stage="synthesis",
                                spoken_dialogue="สรุปผลการหารือรอบโต๊ะครบถ้วนแล้วครับ",
                                evidence_bullets=["[FACT] Devil advocate check passed"],
                                stance_tag="HOLD",
                            ),
                            playbook=DistilledDecisionPlaybook(
                                final_rating="HOLD",
                                action_stance="HOLD",
                                core_conviction_score=1,
                                executive_verdict="ถือดูสถานการณ์ก่อน",
                                action_playbook=ActionPlaybookSpec(
                                    primary_action=ActionPlaybookOption(
                                        option_id="p1",
                                        label="Hold",
                                        speaker_source="hermes_moderator",
                                        action_steps=["เฝ้าระวังความเสี่ยง"],
                                    )
                                ),
                            ),
                        )
                    return RoundtableTurnPayload(
                        speaker_id="value_guru",
                        round_stage="opening",
                        spoken_dialogue="มุมมอง Graham เน้นส่วนเผื่อความปลอดภัยครับ",
                        evidence_bullets=["[FACT] Checking balance sheet"],
                        stance_tag="HOLD",
                    )

            return _Runner()

    engine = GeminiRoundtableEngine(llm_factory=lambda *_: _PromptCapturingLLM())
    turn = await engine.generate_guru_turn(
        speaker_id="value_guru",
        round_stage="opening",
        packet=packet,
        prior_turns=[],
        unspoken_speakers=[],
    )
    await engine.generate_hermes_synthesis(packet=packet, all_turns=[turn])

    turn_sys_prompt = captured_prompts[0][1]
    turn_human_prompt = captured_prompts[0][2]
    assert "[STRICT FACT-GROUNDING GUARDRAIL]" in turn_sys_prompt
    assert "[CUSTOM GURU PERSONA & SPEAKING STYLE OVERRIDE]" in turn_human_prompt
    assert "Graham" in turn_human_prompt
    assert "ตรวจ net-net balance sheet ก่อนเสมอ" in turn_human_prompt
    # Persona role block sits AFTER the shared Blackboard prefix so KV caching stays intact
    assert turn_human_prompt.index("<!-- SEC:") < turn_human_prompt.index("[CUSTOM GURU PERSONA & SPEAKING STYLE OVERRIDE]") if "<!-- SEC:" in turn_human_prompt else True
    assert turn_human_prompt.index("[แก่นการลงทุน") < turn_human_prompt.index("[CUSTOM GURU PERSONA & SPEAKING STYLE OVERRIDE]")

    hermes_human_prompt = captured_prompts[1][2]
    assert "[MANDATORY DEVIL'S ADVOCATE CHECK — SKEPTIC SEAT INACTIVE]" in hermes_human_prompt


@pytest.mark.asyncio
async def test_role_specific_context_slicing_legacy_fallback_and_hermes_4096_tokens(monkeypatch):
    curated_equity = (
        "=== ตารางเปรียบเทียบหุ้นที่แนบในวงเสวนา (COMPARATIVE SNAPSHOT TABLE) ===\n"
        "| Ticker | Price | Composite |\n|---|---|---|\n| $GOOGL | 182.40 USD | 84.5 |\n| $ORCL | 170.00 USD | 58.0 |\n\n"
        "=== หุ้น $GOOGL (Alphabet Inc.) ===\n\n"
        "<!-- SEC:HEADER -->\n- หุ้น: $GOOGL (Alphabet Inc.)\n\n"
        "<!-- SEC:VALUATION -->\n- Forward DCF: Base=210.0 | Reverse DCF: MoS Entry=165.0\n\n"
        "<!-- SEC:QUALITY -->\n- Capital Efficiency: ROIC=28.4% | Piotroski F-Score = 8/9\n\n"
        "<!-- SEC:GROWTH_GUIDANCE -->\n- Earnings Guidance: Cloud backlog hit record Q3 guidance\n\n"
        "<!-- SEC:MACRO_TACTICAL -->\n- Tactical Setup: STAGE_2_ADVANCING | R:R=2.4x\n\n"
        "<!-- SEC:RISKS_THESIS -->\n- Thesis Falsifiers: Cloud growth < 20%\n\n"
        "<!-- SEC:NEWS_NOTES -->\n- ข่าวล่าสุดของ GOOGL: Enterprise Gemini Rollout\n\n"
        "=== หุ้น $ORCL (Oracle Corporation) ===\n\n"
        "<!-- SEC:HEADER -->\n- หุ้น: $ORCL (Oracle Corporation)\n\n"
        "<!-- SEC:VALUATION -->\n- Forward DCF: None | Reverse DCF: Target=136.02\n\n"
        "<!-- SEC:RISKS_THESIS -->\n- Tail Risks: AI Capex cash burn & net debt 3.4x"
    )
    packet = FrozenContextPacket(
        session_id="s_slicing",
        created_at_iso="2026-11-11T00:00:00Z",
        agenda_prompt="เทียบ $GOOGL กับ $ORCL",
        context_refs=("$GOOGL", "$ORCL", "@Macro", "@Digest", "@Essence", "@Portfolio"),
        primary_ticker="GOOGL",
        instrument_anchor=InstrumentIdentityAnchor(
            ticker="GOOGL",
            company_name="Alphabet Inc.",
            market="US",
            sector="Communication Services",
            industry="Internet",
            currency="USD",
            benchmark_ticker="SPY",
            current_price="182.40",
        ),
        portfolio_guard=PortfolioHoldingGuardState(
            attachment_state=ContextAttachmentState.ATTACHED_HELD,
            portfolio_attached=True,
            is_held=True,
            units="40.0000",
            summary_text="[PORTFOLIO_STATE: ATTACHED_HELD] ถือ GOOGL 40 หุ้น",
        ),
        equity_report_text=curated_equity,
        macro_report_text="- Macro Regime: Late-Cycle Soft Landing\n- Thailand Stance: Defensive",
        digest_report_text="- Market Mood: Constructive\n- Transmission Chain: Lower 10Y supports quality growth",
        essence_report_text="1. นโยบายพื้นฐาน: คุณภาพ\n3. สิ่งที่ห้ามลงทุน (Exclude): Crypto Leverage\n9. บริบททางการเงินจริง: EmergencyReserve=12 เดือน",
        verified_numeric_facts=("current_price=182.40 USD", "roic_pct=28.4"),
    )

    # 1. Backward-compatible 1-arg call without speaker_id preserves full equity/macro/digest/essence
    default_ctx = _format_packet_prompt_context(packet)
    assert "COMPARATIVE SNAPSHOT TABLE" in default_ctx
    assert "Oracle Corporation" in default_ctx
    assert "EmergencyReserve=12 เดือน" in default_ctx

    # 2. Macro guru keeps byte-identical shared prefix and appends Macro-focused ROLE FOCUS LENS
    macro_ctx = _format_packet_prompt_context(packet, speaker_id="macro_tactical_guru")
    assert macro_ctx.startswith(default_ctx)
    assert "[ROLE FOCUS LENS" in macro_ctx
    assert "@Macro, @Digest" in macro_ctx.split("[ROLE FOCUS LENS", 1)[1]
    assert "Late-Cycle Soft Landing" in macro_ctx
    assert "STAGE_2_ADVANCING" in macro_ctx
    assert "Transmission Chain" in macro_ctx

    # 3. Growth guru keeps identical shared prefix and highlights Growth/Digest in Role Focus Lens
    growth_ctx = _format_packet_prompt_context(packet, speaker_id="growth_guru")
    assert growth_ctx.startswith(default_ctx)
    assert "Growth (" in growth_ctx.split("[ROLE FOCUS LENS", 1)[1]
    assert "Enterprise Gemini Rollout" in growth_ctx
    assert "Cloud backlog hit record Q3 guidance" in growth_ctx

    # 4. Value guru keeps identical shared prefix and focuses on Valuation/Quality/Risks in Role Focus Lens
    value_ctx = _format_packet_prompt_context(packet, speaker_id="value_guru")
    assert value_ctx.startswith(default_ctx)
    assert "ประเมินมูลค่า (DCF / Reverse DCF / Consensus / Relative)" in value_ctx.split("[ROLE FOCUS LENS", 1)[1]
    assert "Oracle Corporation" in value_ctx

    # 5. Legacy unsectioned packet (pre-upgrade SQLite rows) renders cleanly with no SEC tags
    legacy_pkt = FrozenContextPacket(
        session_id="s_legacy",
        created_at_iso="2026-10-01T00:00:00Z",
        agenda_prompt="วิเคราะห์ $NVDA",
        equity_report_text='{"ticker": "NVDA", "legacy": true}',
        macro_report_text="Legacy Macro text",
        digest_report_text="Legacy Digest text",
        essence_report_text="Legacy Essence text",
    )
    legacy_ctx = _format_packet_prompt_context(legacy_pkt, speaker_id="value_guru")
    assert '"legacy": true' in legacy_ctx
    assert "Legacy Macro text" in legacy_ctx

    # 6. Verify generate_guru_turn uses max_output_tokens=2048 while generate_hermes_synthesis uses 4096
    seen_token_budgets = []

    class _FakeProdLLM:
        def with_structured_output(self, schema):
            class _Runner:
                def invoke(self, messages):
                    if schema.__name__ == "_HermesCombinedOutput":
                        return schema(
                            closing_turn=RoundtableTurnPayload(
                                speaker_id="hermes_moderator",
                                round_stage="synthesis",
                                spoken_dialogue="สรุปเทียบ GOOGL และ ORCL เรียบร้อยครับ",
                                evidence_bullets=["[FACT] GOOGL ROIC 28.4% vs ORCL 13.8%"],
                                stance_tag="OVERWEIGHT",
                            ),
                            playbook=DistilledDecisionPlaybook(
                                final_rating="OVERWEIGHT",
                                action_stance="OVERWEIGHT",
                                core_conviction_score=2,
                                executive_verdict="เน้น GOOGL เหนือกว่า ORCL",
                                action_playbook=ActionPlaybookSpec(
                                    primary_action=ActionPlaybookOption(
                                        option_id="p1",
                                        label="Overweight GOOGL",
                                        speaker_source="hermes_moderator",
                                        action_steps=["แบ่งซื้อ GOOGL"],
                                    )
                                ),
                            ),
                        )
                    return RoundtableTurnPayload(
                        speaker_id="value_guru",
                        round_stage="opening",
                        spoken_dialogue="ผมเทียบ GOOGL กับ ORCL แล้วเห็นชัดว่าส่วนเผื่อความปลอดภัยต่างกันครับ",
                        evidence_bullets=["[FACT] GOOGL MoS 13.1% vs ORCL negative FCF"],
                        stance_tag="OVERWEIGHT",
                    )

            return _Runner()

    def _capturing_get_llm(**kwargs):
        seen_token_budgets.append(kwargs.get("max_output_tokens"))
        return _FakeProdLLM()

    import tools.roundtable.debate_engine as de_mod
    monkeypatch.setattr(de_mod, "detect_provider", lambda m: "gemini")
    monkeypatch.setattr(de_mod, "get_llm", _capturing_get_llm)

    prod_engine = GeminiRoundtableEngine(model_name="gemini-2.5-flash")
    t_out = await prod_engine.generate_guru_turn(
        speaker_id="value_guru",
        round_stage="opening",
        packet=packet,
        prior_turns=[],
        unspoken_speakers=[],
    )
    await prod_engine.generate_hermes_synthesis(packet=packet, all_turns=[t_out])
    assert seen_token_budgets == [2048, 4096]


def test_wp_a_c_e_three_ticker_fair_quota_expanded_budgets_and_preset_rubric_injection():
    def _stock_block(sym: str, name: str, pad_val: int = 1600, pad_sec: int = 1800) -> str:
        return (
            f"=== หุ้น ${sym} ({name}) ===\n\n"
            f"<!-- SEC:HEADER -->\n- หุ้น: ${sym} ({name})\n\n"
            f"<!-- SEC:VALUATION -->\n- {sym}_VAL_START: FairValue | " + ("v" * pad_val) + f" | {sym}_VAL_END\n\n"
            f"<!-- SEC:QUALITY -->\n- {sym}_QUAL_START: ROIC 25% | " + ("q" * 700) + f"\n\n"
            f"<!-- SEC:GROWTH_GUIDANCE -->\n- {sym}_GROW_START: Guidance strong | " + ("g" * pad_sec) + f"\n\n"
            f"<!-- SEC:MACRO_TACTICAL -->\n- {sym}_MAC_START: Stage 2\n\n"
            f"<!-- SEC:RISKS_THESIS -->\n- {sym}_RISK_START: Risk note\n\n"
            f"<!-- SEC:NEWS_NOTES -->\n- {sym}_NEWS_START: Earnings note"
        )

    curated_3 = "\n\n".join(
        [
            "=== ตารางเปรียบเทียบหุ้นที่แนบในวงเสวนา (COMPARATIVE SNAPSHOT TABLE) ===\n| Ticker |\n|---|\n| $GOOGL |\n| $ORCL |\n| $NFLX |",
            _stock_block("GOOGL", "Alphabet Inc.", pad_val=2200, pad_sec=1900),
            _stock_block("ORCL", "Oracle Corp.", pad_val=2200, pad_sec=1900),
            _stock_block("NFLX", "Netflix Inc.", pad_val=2200, pad_sec=1900),
        ]
    )
    long_macro = (
        "- Macro Regime: Expansion\n- Asset Allocation Stance: US Quality Overweight\n"
        + ("m" * 2400)
        + "\n@MACRO_END_3400_MARKER\n"
        + ("z" * 1600)
        + "\n@MACRO_PAST_3600_MARKER"
    )
    long_digest = (
        "- Market Mood: Risk-On\n- Transmission Chain: Cooling CPI\n"
        + ("d" * 1700)
        + "\n@DIGEST_YT_AT_1850\n"
        + ("y" * 900)
        + "\n@DIGEST_PAST_2400"
    )

    custom_personas = merge_persona_overrides(
        GURU_PERSONA_CATALOG,
        None,
        [
            {"speaker_id": "quality_guru", "preset_id": "quality_akre_compounder", "short_name": "Akre"},
            {"speaker_id": "skeptic_guru", "preset_id": "skeptic_taleb_tailrisk", "short_name": "TailRisk"},
        ],
    )
    packet = FrozenContextPacket(
        session_id="s_3tickers",
        created_at_iso="2026-11-12T00:00:00Z",
        agenda_prompt="เทียบ $GOOGL $ORCL $NFLX",
        context_refs=("$GOOGL", "$ORCL", "$NFLX", "@Macro", "@Digest"),
        primary_ticker="GOOGL",
        instrument_anchor=InstrumentIdentityAnchor(
            ticker="GOOGL",
            company_name="Alphabet Inc.",
            market="US",
            sector="Communication Services",
            industry="Internet",
            currency="USD",
            benchmark_ticker="SPY",
            current_price="182.40",
        ),
        secondary_anchors=(
            InstrumentIdentityAnchor(
                ticker="ORCL",
                company_name="Oracle Corp.",
                market="US",
                sector="Technology",
                industry="Software",
                currency="USD",
                benchmark_ticker="SPY",
                current_price="170.00",
            ),
            InstrumentIdentityAnchor(
                ticker="NFLX",
                company_name="Netflix Inc.",
                market="US",
                sector="Communication Services",
                industry="Entertainment",
                currency="USD",
                benchmark_ticker="SPY",
                current_price="920.00",
            ),
        ),
        secondary_scorecards=(
            PastRoundtableScorecard(
                previous_session_id="prev_orcl",
                previous_date="2026-09-10",
                primary_ticker="ORCL",
                benchmark_ticker="SPY",
                entry_price="150.00",
                settlement_price="170.00",
                raw_return_pct="13.33",
                benchmark_return_pct="4.00",
                alpha_pct="9.33",
                previous_final_rating="HOLD",
                outcome_label="outperformed",
            ),
        ),
        ticker_data_quality_flags={"ORCL": ("negative_fcf_dcf_unavailable:dcf",)},
        equity_report_text=curated_3,
        macro_report_text=long_macro,
        digest_report_text=long_digest,
        active_personas=tuple(p.to_dict() for p in custom_personas),
    )

    val_ctx = _format_packet_prompt_context(packet, speaker_id="value_guru")
    # WP-A multi-ticker shared context headers
    assert "[SECONDARY INSTRUMENT ANCHORS — VERIFIED]" in val_ctx
    assert "$ORCL (Oracle Corp.)" in val_ctx
    assert "$NFLX (Netflix Inc.)" in val_ctx
    assert "[DATA QUALITY FLAGS PER TICKER]: $ORCL: negative_fcf_dcf_unavailable:dcf" in val_ctx
    assert "[ย้อนรอยมติครั้งก่อนของหุ้นเปรียบเทียบ (Secondary Past Scorecards)]" in val_ctx
    # Fair water-filling quota preserves all 3 tickers across both primary & secondary sections!
    assert "GOOGL_VAL_START" in val_ctx
    assert "ORCL_VAL_START" in val_ctx
    assert "NFLX_VAL_START" in val_ctx
    assert "GOOGL_GROW_START" in val_ctx
    assert "ORCL_GROW_START" in val_ctx
    assert "NFLX_GROW_START" in val_ctx
    # Phase 3 Fixed Shared Blackboard gives every guru unclipped @Macro (8000) & @Digest (6000) identically
    assert "@MACRO_END_3400_MARKER" in val_ctx
    assert "@MACRO_PAST_3600_MARKER" in val_ctx
    assert "@DIGEST_YT_AT_1850" in val_ctx
    assert "@DIGEST_PAST_2400" in val_ctx

    skep_ctx = _format_packet_prompt_context(packet, speaker_id="skeptic_guru")
    assert "@MACRO_PAST_3600_MARKER" in skep_ctx
    assert "@DIGEST_PAST_2400" in skep_ctx
    # Byte-identical Shared Blackboard prefix prior to Role Focus Lens
    assert val_ctx.split("[ROLE FOCUS LENS", 1)[0] == skep_ctx.split("[ROLE FOCUS LENS", 1)[0]

    # WP-E Preset Analytical Rubric injection in system prompt
    engine = GeminiRoundtableEngine(llm_factory=lambda *_: None)
    akre_sys = engine._load_persona_system_prompt("quality_guru", packet)
    assert "[PRESET ANALYTICAL RUBRIC]" in akre_sys
    assert "Akre 3-Legged Stool" in akre_sys
    assert "capex_to_depreciation_ratio" in akre_sys

    taleb_sys = engine._load_persona_system_prompt("skeptic_guru", packet)
    assert "[PRESET ANALYTICAL RUBRIC]" in taleb_sys
    assert "Risk of Ruin" in taleb_sys
    assert "amihud_illiquidity_20d" in taleb_sys

    # Mention resolution for preset role models & custom short names
    assert _detect_target_speaker("@Akre ช่วยดู Moat หน่อย", None, custom_personas) == "quality_guru"
    assert _detect_target_speaker("@TailRisk ทดสอบวิกฤตหน่อย", None, custom_personas) == "skeptic_guru"
    assert _detect_target_speaker("@graham ขอ MoS", None, GURU_PERSONA_CATALOG) == "value_guru"
    assert _detect_target_speaker("@dalio มองเรจีมยังไง", None, GURU_PERSONA_CATALOG) == "macro_tactical_guru"
    assert _detect_target_speaker("@taleb มอง tail risk อะไร", None, GURU_PERSONA_CATALOG) == "skeptic_guru"


def test_wp_s2_s3_single_stock_zero_truncation_all_seats():
    curated_1 = (
        "=== หุ้น $GOOGL (Alphabet Inc.) ===\n\n"
        "<!-- SEC:HEADER -->\n- หุ้น: $GOOGL (Alphabet Inc.)\n\n"
        "<!-- SEC:VALUATION -->\n- Reverse DCF: Intrinsic=476.69\n" + ("v" * 1600) + "\n@END_VALUATION\n\n"
        "<!-- SEC:QUALITY -->\n- 3Y Trajectory:\n" + ("q" * 1800) + "\n@END_QUALITY_SEC\n\n"
        "<!-- SEC:GROWTH_GUIDANCE -->\n- Growth: Rev YoY=15.09%\n" + ("g" * 1000) + "\n@END_GROWTH_SEC\n\n"
        "<!-- SEC:MACRO_TACTICAL -->\n- Tactical Setup: Buy Zone=274.40-290.45\n" + ("t" * 1100) + "\n@END_TACTICAL_SEC\n\n"
        "<!-- SEC:RISKS_THESIS -->\n- Kill-Switch GOOGL_STOP_LOSS_BREAK\n" + ("r" * 1200) + "\n@END_RISKS_SEC\n\n"
        "<!-- SEC:NEWS_NOTES -->\n- News & Vault: Summary\n" + ("n" * 900) + "\n@END_NEWS_SEC"
    )
    single_macro = (
        "- Macro Regime: Reflation\n- Asset Allocation Stance:\n"
        + ("m" * 3400)
        + "\n- Regime Evidence:\n"
        + ("e" * 1200)
        + "\n- Key Macro Indicators: TIPS Real Yield=2.60% | CPI=2.61% | @END_MACRO_INDICATORS_5400"
    )
    single_digest = "- Market Mood: Neutral\n" + ("d" * 2400) + "\n@DIGEST_AT_2450"

    packet = FrozenContextPacket(
        session_id="s_single_zero_trunc",
        created_at_iso="2026-11-12T00:00:00Z",
        agenda_prompt="วิเคราะห์ $GOOGL แบบเจาะลึก",
        context_refs=("$GOOGL", "@Macro", "@Digest", "@Portfolio", "@Essence"),
        primary_ticker="GOOGL",
        instrument_anchor=InstrumentIdentityAnchor(
            ticker="GOOGL",
            company_name="Alphabet Inc.",
            market="US",
            sector="Communication Services",
            industry="Internet",
            currency="USD",
            benchmark_ticker="SPY",
            current_price="283.82",
        ),
        equity_report_text=curated_1,
        macro_report_text=single_macro,
        digest_report_text=single_digest,
    )

    # All 5 core seats + Hermes get full Key Macro Indicators without truncation
    for sid in ("value_guru", "quality_guru", "growth_guru", "macro_tactical_guru", "skeptic_guru"):
        ctx = _format_packet_prompt_context(packet, speaker_id=sid)
        assert "@END_MACRO_INDICATORS_5400" in ctx, f"{sid} truncated @Macro!"
        assert "@DIGEST_AT_2450" in ctx, f"{sid} truncated @Digest!"

    # growth_guru gets full QUALITY, RISKS_THESIS, AND MACRO_TACTICAL (~5,500 secondary chars <= 6,200)
    growth_ctx = _format_packet_prompt_context(packet, speaker_id="growth_guru")
    assert "@END_GROWTH_SEC" in growth_ctx
    assert "@END_QUALITY_SEC" in growth_ctx
    assert "@END_RISKS_SEC" in growth_ctx
    assert "@END_TACTICAL_SEC" in growth_ctx

    # macro_tactical_guru gets full MACRO_TACTICAL, VALUATION, QUALITY, GROWTH_GUIDANCE, AND RISKS_THESIS
    mac_ctx = _format_packet_prompt_context(packet, speaker_id="macro_tactical_guru")
    assert "@END_TACTICAL_SEC" in mac_ctx
    assert "@END_QUALITY_SEC" in mac_ctx
    assert "@END_GROWTH_SEC" in mac_ctx
    assert "@END_RISKS_SEC" in mac_ctx


def test_wp_b_single_stock_expanded_budgets_preserve_full_vault_ec_and_24_macro_indicators():
    # Stress-test heavy single-stock (~7,600 secondary chars incl. 2,300-char Vault EC + 24 Macro indicators ~6,400 chars + Digest ~3,400 chars)
    curated_heavy = (
        "=== หุ้น $FTNT (Fortinet Inc.) ===\n\n"
        "<!-- SEC:HEADER -->\n- หุ้น: $FTNT (Fortinet Inc.)\n\n"
        "<!-- SEC:VALUATION -->\n- Reverse DCF: Intrinsic=105.00\n" + ("v" * 2100) + "\n@END_VALUATION_HEAVY\n\n"
        "<!-- SEC:QUALITY -->\n- 3Y Trajectory + Sector KPIs:\n" + ("q" * 2300) + "\n@END_QUALITY_HEAVY\n\n"
        "<!-- SEC:GROWTH_GUIDANCE -->\n- Vault Earnings Guidance:\n" + ("g" * 850) + "\n@END_GROWTH_HEAVY\n\n"
        "<!-- SEC:MACRO_TACTICAL -->\n- Tactical Setup:\n" + ("t" * 1050) + "\n@END_TACTICAL_HEAVY\n\n"
        "<!-- SEC:RISKS_THESIS -->\n- 6 Falsifiers + 6 Triggers + 5 Gaps:\n" + ("r" * 1850) + "\n@END_RISKS_HEAVY\n\n"
        "<!-- SEC:NEWS_NOTES -->\n- Earnings Call Note (AI Highlights 2300c):\n" + ("n" * 2550) + "\n@END_NEWS_HEAVY"
    )
    heavy_macro = (
        "- Macro Regime: Disinflationary Expansion\n- Asset Allocation Stance:\n"
        + ("m" * 4400)
        + "\n- Key Macro Indicators: "
        + " | ".join(f"Ind_{i}={i}.1%" for i in range(24))
        + " | @END_24_MACRO_INDICATORS_6400"
    )
    heavy_digest = "- Market Mood: Bullish\n" + ("d" * 3300) + "\n@DIGEST_END_3350"

    packet = FrozenContextPacket(
        session_id="s_heavy_single",
        created_at_iso="2026-11-12T00:00:00Z",
        agenda_prompt="วิเคราะห์ $FTNT แบบละเอียดทุกมิติ",
        context_refs=("$FTNT", "@Macro", "@Digest", "@Portfolio", "@Essence"),
        primary_ticker="FTNT",
        instrument_anchor=InstrumentIdentityAnchor(
            ticker="FTNT",
            company_name="Fortinet Inc.",
            market="US",
            sector="Technology",
            industry="Cybersecurity",
            currency="USD",
            benchmark_ticker="SPY",
            current_price="84.50",
        ),
        equity_report_text=curated_heavy,
        macro_report_text=heavy_macro,
        digest_report_text=heavy_digest,
    )

    for sid in ("value_guru", "quality_guru", "growth_guru", "macro_tactical_guru", "skeptic_guru"):
        ctx = _format_packet_prompt_context(packet, speaker_id=sid)
        assert "@END_24_MACRO_INDICATORS_6400" in ctx, f"{sid} truncated 24th macro indicator!"
        assert "@DIGEST_END_3350" in ctx, f"{sid} truncated heavy digest!"

    # growth_guru secondary = QUALITY(2300) + RISKS_THESIS(1850) + NEWS_NOTES(2550) + MACRO_TACTICAL(1050) ≈ 7,900 <= 8,200
    growth_ctx = _format_packet_prompt_context(packet, speaker_id="growth_guru")
    for marker in ("@END_GROWTH_HEAVY", "@END_QUALITY_HEAVY", "@END_RISKS_HEAVY", "@END_NEWS_HEAVY", "@END_TACTICAL_HEAVY"):
        assert marker in growth_ctx, f"growth_guru missing {marker}"

    # macro_tactical_guru secondary = VALUATION(2100) + QUALITY(2300) + GROWTH(850) + RISKS(1850) ≈ 7,300 <= 8,200
    mac_ctx = _format_packet_prompt_context(packet, speaker_id="macro_tactical_guru")
    for marker in ("@END_TACTICAL_HEAVY", "@END_VALUATION_HEAVY", "@END_QUALITY_HEAVY", "@END_GROWTH_HEAVY", "@END_RISKS_HEAVY"):
        assert marker in mac_ctx, f"macro_tactical_guru missing {marker}"


def test_wp_s100_single_stock_untruncated_across_all_gurus_and_hermes():
    # Stress-test 100% single-stock ceilings (~17k equity, ~9.5k macro, ~3.8k digest, ~5.8k essence)
    curated_100 = (
        "=== หุ้น $FTNT (Fortinet Inc.) ===\n\n"
        "<!-- SEC:HEADER -->\n- หุ้น: $FTNT (Fortinet Inc.)\n@END_HEADER_100\n\n"
        "<!-- SEC:VALUATION -->\n- Forward DCF: Base=110.00 | " + ("v" * 2600) + "\n@END_VALUATION_100\n\n"
        "<!-- SEC:QUALITY -->\n- Narrative 2500c + 3Y + SubScores: " + ("q" * 3400) + "\n@END_QUALITY_100\n\n"
        "<!-- SEC:GROWTH_GUIDANCE -->\n- Guidance + Quotes: " + ("g" * 1600) + "\n@END_GROWTH_100\n\n"
        "<!-- SEC:MACRO_TACTICAL -->\n- Tactical Setup: " + ("t" * 1400) + "\n@END_TACTICAL_100\n\n"
        "<!-- SEC:RISKS_THESIS -->\n- Active Thesis + Falsifiers + ThenVsNow: " + ("r" * 2800) + "\n@END_RISKS_100\n\n"
        "<!-- SEC:NEWS_NOTES -->\n- 12 News + Full EC AI Highlights 3100c: " + ("n" * 4400) + "\n@END_NEWS_100"
    )
    macro_100 = (
        "- Macro Regime: Reflation\n- Asset Allocation Stance:\n"
        + ("m" * 4200)
        + "\n- Regional Macro States: 13 regions\n"
        + ("r" * 2600)
        + "\n- Key Macro Indicators: Chicago Fed National Financial Conditions Index=-0.48\n"
        + ("i" * 2000)
        + "\n- Macro News & Media References:\n@END_9500_MACRO_MARKER"
    )
    digest_100 = "- Market Mood: Constructive\n" + ("d" * 3700) + "\n@END_3800_DIGEST_MARKER"
    essence_100 = "1. นโยบายพื้นฐาน: " + ("e" * 5600) + "\n@END_5800_ESSENCE_MARKER"

    packet = FrozenContextPacket(
        session_id="s_single_100pct",
        created_at_iso="2026-11-12T00:00:00Z",
        agenda_prompt="วิเคราะห์ $FTNT แบบครบ 100%",
        context_refs=("$FTNT", "@Macro", "@Digest", "@Portfolio", "@Essence"),
        primary_ticker="FTNT",
        instrument_anchor=InstrumentIdentityAnchor(
            ticker="FTNT",
            company_name="Fortinet Inc.",
            market="US",
            sector="Technology",
            industry="Cybersecurity",
            currency="USD",
            benchmark_ticker="SPY",
            current_price="84.50",
        ),
        equity_report_text=curated_100,
        macro_report_text=macro_100,
        digest_report_text=digest_100,
        essence_report_text=essence_100,
    )

    all_sec_markers = (
        "@END_HEADER_100",
        "@END_VALUATION_100",
        "@END_QUALITY_100",
        "@END_GROWTH_100",
        "@END_TACTICAL_100",
        "@END_RISKS_100",
        "@END_NEWS_100",
    )
    for sid in ("value_guru", "quality_guru", "growth_guru", "macro_tactical_guru", "skeptic_guru", None):
        ctx = _format_packet_prompt_context(packet, speaker_id=sid)
        seat_label = sid or "hermes_moderator"
        assert "@END_9500_MACRO_MARKER" in ctx, f"{seat_label} truncated 9.5k @Macro!"
        assert "@END_3800_DIGEST_MARKER" in ctx, f"{seat_label} truncated 3.8k @Digest!"
        assert "@END_5800_ESSENCE_MARKER" in ctx, f"{seat_label} truncated 5.8k @Essence!"
        for mk in all_sec_markers:
            assert mk in ctx, f"{seat_label} truncated {mk}!"


def test_wp_s100_p2_default_preset_rubric_18_turn_memory_and_unclipped_packet():
    packet = FrozenContextPacket(
        session_id="s_p2_engine",
        created_at_iso="2026-11-12T00:00:00Z",
        agenda_prompt="วิเคราะห์ $FTNT ครบทุกมิติ",
        context_refs=("$FTNT", "@Macro", "@Digest", "@Portfolio", "@Essence"),
        primary_ticker="FTNT",
        instrument_anchor=InstrumentIdentityAnchor(
            ticker="FTNT",
            company_name="Fortinet Inc.",
            market="US",
            sector="Technology",
            industry="Cybersecurity",
            currency="USD",
            benchmark_ticker="SPY",
            current_price="173.46",
        ),
        equity_report_text=(
            "=== หุ้น $FTNT (Fortinet Inc.) ===\n\n"
            "<!-- SEC:HEADER -->\n- Sidecar Age: 47d\n- [STALE SIDECAR WARNING]\n\n"
            "<!-- SEC:VALUATION -->\n- Analyst Context Cache (as_of=2026-10-18): Target Mean=167.91\n@END_VAL\n\n"
            "<!-- SEC:QUALITY -->\n- ROIC=31.44%\n@END_QUAL\n\n"
            "<!-- SEC:GROWTH_GUIDANCE -->\n- Promise Ledger Credibility: Score=92.50/100\n@END_GROW\n\n"
            "<!-- SEC:MACRO_TACTICAL -->\n- [MACRO_ALIGNMENT_DRIFT_WARNING]: Drift note\n@END_MAC\n\n"
            "<!-- SEC:RISKS_THESIS -->\n- [WATCHDOG BREACHES (PENDING)]: FALSIFIER_BREACHED (x18)\n- Insider DB Ledger (180d)\n@END_RISK\n\n"
            "<!-- SEC:NEWS_NOTES -->\n- Earnings Call Note: Summary\n@END_NEWS"
        ),
        essence_report_text="0. แก่นแท้การลงทุนที่ยืนยันแล้ว (Confirmed Investor Essence): Focus on cash flow.\n" + ("e" * 8200) + "\n@END_ESSENCE_9500",
    )

    # 1. Default preset still injects [PRESET ANALYTICAL RUBRIC] for all 6 seats without customization
    engine = GeminiRoundtableEngine(llm_factory=lambda *_: None)
    for sid in ("value_guru", "quality_guru", "growth_guru", "macro_tactical_guru", "skeptic_guru", "hermes_moderator"):
        sys_p = engine._load_persona_system_prompt(sid, packet)
        assert "[PRESET ANALYTICAL RUBRIC]" in sys_p, f"{sid} default preset missing rubric!"

    # 2. Expanded 18-turn memory window, 3 evidence bullets, and [Sizing: ...]
    turns = [
        RoundtableTurnPayload(
            speaker_id="value_guru" if i % 2 == 0 else "growth_guru",
            round_stage="rebuttal" if i >= 5 else "opening",
            spoken_dialogue=f"ประเด็นลำดับที่ {i}",
            evidence_bullets=[f"B1_{i}", f"B2_{i}", f"B3_{i}", f"B4_{i}"],
            stance_tag="HOLD",
            sizing_posture=f"Staged {i}% tranche",
        )
        for i in range(1, 21)
    ]
    guard = opponent_argument_or_opening("skeptic_guru", "rebuttal", turns, [], packet=packet)
    assert "ประเด็นลำดับที่ 1 |" not in guard and "ประเด็นลำดับที่ 2 |" not in guard
    assert "ประเด็นลำดับที่ 3" in guard
    assert "ประเด็นลำดับที่ 20" in guard
    assert "B3_20" in guard
    assert "B4_20" not in guard
    assert "[Sizing: Staged 20% tranche]" in guard

    # 3. Single-stock expanded essence (10k) and all 7 SEC blocks fit for every seat without truncation
    for sid in ("value_guru", "quality_guru", "growth_guru", "macro_tactical_guru", "skeptic_guru", None):
        ctx = _format_packet_prompt_context(packet, speaker_id=sid)
        assert "@END_ESSENCE_9500" in ctx
        for mk in ("@END_VAL", "@END_QUAL", "@END_GROW", "@END_MAC", "@END_RISK", "@END_NEWS"):
            assert mk in ctx
        assert all(not ln.rstrip().endswith("…") for ln in ctx.splitlines())


@pytest.mark.asyncio
async def test_phase3_shared_prefix_byte_equality_fact_filter_and_turn_ledger():
    """Verify P3-B KV cache prefix byte-equality across all 6 seats, lossless fact filter, and 18-turn compact ledger."""
    from langchain_core.messages import HumanMessage
    from schemas.roundtable_schemas import (
        ActionPlaybookOption,
        ActionPlaybookSpec,
        DistilledDecisionPlaybook,
        RoundtableTurnPayload,
    )
    from tools.roundtable.debate_engine import (
        GeminiRoundtableEngine,
        _filter_unrepresented_numeric_facts,
        _format_shared_blackboard_context,
        opponent_argument_or_opening,
    )

    eq_text = (
        "=== หุ้น $GOOGL (Alphabet Inc.) ===\n\n"
        "<!-- SEC:HEADER -->\n- ราคาอ้างอิง: 293.45 USD | Market Cap=$1.73T\n\n"
        "<!-- SEC:VALUATION -->\n- Reverse DCF: Intrinsic Today=493.26 | Street Mean=400.37\n\n"
        "<!-- SEC:QUALITY -->\n- ROIC=35.73% | Piotroski F-Score = 6/9\n"
        + ("  * Deep Qualitative Moat & AI Infrastructure Detail.\n" * 120)
    )
    packet = FrozenContextPacket(
        session_id="s_prefix_cache",
        created_at_iso="2026-11-12T00:00:00Z",
        agenda_prompt="วิเคราะห์ $GOOGL ระยะ 3 ปี",
        context_refs=("$GOOGL", "@Macro", "@Digest", "@Essence"),
        primary_ticker="GOOGL",
        instrument_anchor=InstrumentIdentityAnchor(
            ticker="GOOGL",
            company_name="Alphabet Inc.",
            market="US",
            sector="Communication Services",
            industry="Internet",
            currency="USD",
            benchmark_ticker="SPY",
            current_price="293.45",
        ),
        portfolio_guard=PortfolioHoldingGuardState(
            attachment_state=ContextAttachmentState.NOT_ATTACHED_DO_NOT_ASSUME,
            portfolio_attached=False,
            summary_text="[PORTFOLIO_STATE: NOT_ATTACHED_DO_NOT_ASSUME]",
        ),
        equity_report_text=eq_text,
        macro_report_text="- Macro Regime: Reflation | 10Y Yield=4.15%",
        digest_report_text="- Market Mood: Constructive | Cloud Backlog Surge",
        essence_report_text="0. แก่นแท้การลงทุน: เน้น Quality Moat | mdd_max_annual=15.00%",
        verified_numeric_facts=(
            "current_price=293.45 USD",
            "roic_pct=35.73",
            "piotroski_f_score=6",
            "analyst_target_mean=400.3700000000000045",
            "pe_ratio=15.0",
            "unrendered_hidden_metric=777.88",
        ),
    )

    # 1. Represented facts drop automatically when unclipped; unrepresented/cross-card collisions are preserved!
    bb_txt, clipped = _format_shared_blackboard_context(packet)
    assert clipped is False
    assert "unrendered_hidden_metric=777.88" in bb_txt
    assert "pe_ratio=15.0" in bb_txt
    assert "current_price=293.45 USD" not in bb_txt
    assert "analyst_target_mean=400.37" not in bb_txt
    # When sections are clipped, ALL facts are retained as safety fallback
    assert len(_filter_unrepresented_numeric_facts(packet.verified_numeric_facts, bb_txt, has_clipped_sections=True)) == 6
    # Multi-ticker water-fill clipping on stock #1 (while stock #2 fits cleanly) still reports has_clipped_sections=True
    multi_clip_pkt = FrozenContextPacket(
        session_id="s_multi_clip",
        created_at_iso="2026-11-12T00:00:00Z",
        agenda_prompt="เทียบ $GOOGL $NVDA",
        context_refs=("$GOOGL", "$NVDA"),
        primary_ticker="GOOGL",
        equity_report_text=(
            "=== หุ้น $GOOGL (Alphabet) ===\n<!-- SEC:VALUATION -->\n" + ("v" * 24000)
            + "\n\n=== หุ้น $NVDA (NVIDIA) ===\n<!-- SEC:VALUATION -->\n- Short NVDA summary"
        ),
        macro_report_text="[ABSENT: @Macro]",
        digest_report_text="[ABSENT: @Digest]",
        essence_report_text="0. Basic",
        verified_numeric_facts=("pe_ratio=22.5",),
    )
    multi_bb, multi_clipped = _format_shared_blackboard_context(multi_clip_pkt)
    assert multi_clipped is True
    assert not multi_bb.endswith("…")
    assert "pe_ratio=22.5" in multi_bb

    # 2. Live engine calls across all 5 gurus + Hermes produce 100% byte-identical SystemMessage & HumanMessage Prefix!
    recorded_calls = []

    class _RecorderLLM:
        def with_structured_output(self, schema):
            class _Runner:
                def invoke(self, messages):
                    recorded_calls.append((schema.__name__, messages[0].content, messages[1].content))
                    if schema.__name__ == "_HermesCombinedOutput":
                        return schema(
                            closing_turn=RoundtableTurnPayload(
                                speaker_id="hermes_moderator",
                                round_stage="synthesis",
                                spoken_dialogue="สรุปมติเรียบร้อยครับ",
                                evidence_bullets=["[FACT] Checked"],
                                stance_tag="HOLD",
                            ),
                            playbook=DistilledDecisionPlaybook(
                                final_rating="HOLD",
                                action_stance="HOLD",
                                core_conviction_score=2,
                                executive_verdict="รอติดตาม",
                                action_playbook=ActionPlaybookSpec(
                                    primary_action=ActionPlaybookOption(
                                        option_id="p1",
                                        label="Hold",
                                        speaker_source="hermes_moderator",
                                        action_steps=["Hold"],
                                    )
                                ),
                            ),
                        )
                    return RoundtableTurnPayload(
                        speaker_id="value_guru",
                        round_stage="opening",
                        spoken_dialogue="ประเมินมูลค่าตาม Reverse DCF เรียบร้อยครับ",
                        evidence_bullets=["[FACT] Reverse DCF"],
                        stance_tag="OVERWEIGHT",
                    )

            return _Runner()

    engine = GeminiRoundtableEngine(llm_factory=lambda *_: _RecorderLLM())
    turns = []
    seats = ["value_guru", "quality_guru", "growth_guru", "macro_tactical_guru", "skeptic_guru"]
    for sid in seats:
        t = await engine.generate_guru_turn(
            speaker_id=sid,
            round_stage="opening",
            packet=packet,
            prior_turns=turns,
            unspoken_speakers=[s for s in seats if s != sid],
        )
        turns.append(t)
    await engine.generate_hermes_synthesis(packet=packet, all_turns=turns)

    assert len(recorded_calls) == 6
    sys_prompts = {c[1] for c in recorded_calls}
    assert len(sys_prompts) == 1, "All 6 seats must share byte-identical SystemMessage!"
    human_prefixes = {c[2].split("\n\n---\n\n", 1)[0] for c in recorded_calls}
    assert len(human_prefixes) == 1, "All 6 seats must share byte-identical HumanMessage Blackboard prefix!"
    assert next(iter(human_prefixes)) == bb_txt

    # 3. Structured Turn Ledger condenses turns older than latest 10 when 20 turns exist
    long_turns = [
        RoundtableTurnPayload(
            speaker_id="value_guru" if i % 2 == 0 else "growth_guru",
            round_stage="opening" if i <= 5 else "rebuttal",
            spoken_dialogue=f"Speech_{i} " + ("x" * 200),
            evidence_bullets=[f"B1_{i}", f"B2_{i}", f"B3_{i}"],
            stance_tag="HOLD",
        )
        for i in range(1, 21)
    ]
    ledger_guard = opponent_argument_or_opening("skeptic_guru", "rebuttal", long_turns, [], packet=packet)
    assert "Speech_1 " not in ledger_guard and "Speech_2 " not in ledger_guard
    # Turn 3 (index 1 in recent[-18:]) is condensed in ledger with max 2 bullets and clipped dialogue
    assert "Speech_3 " in ledger_guard
    assert "B2_3" in ledger_guard and "B3_3" not in ledger_guard
    # Turn 20 (latest 10) keeps full unclipped dialogue & all 3 bullets
    assert ("Speech_20 " + ("x" * 200)) in ledger_guard
    assert "B3_20" in ledger_guard


@pytest.mark.asyncio
async def test_engine_telemetry_include_raw_and_sandbox_fx_multinumber(caplog):
    import logging

    class _TelemetryRawLLM:
        def with_structured_output(self, schema, *, include_raw: bool = False):
            class _Runner:
                def invoke(self, messages):
                    payload = RoundtableTurnPayload(
                        speaker_id="value_guru",
                        round_stage="opening",
                        spoken_dialogue="ประเมินมูลค่าตามบริบทและ Telemetry ครบถ้วนครับ",
                        evidence_bullets=["[FACT] ROIC"],
                        stance_tag="HOLD",
                    )
                    if include_raw:
                        class _RawMsg:
                            usage_metadata = {
                                "input_tokens": 10000,
                                "output_tokens": 250,
                                "total_tokens": 10250,
                                "input_token_details": {"cache_read": 7500},
                            }
                            response_metadata = {}
                            content = ""

                        return {"raw": _RawMsg(), "parsed": payload, "parsing_error": None}
                    return payload

            return _Runner()

    pkt = FrozenContextPacket(
        session_id="s_tel_fx",
        created_at_iso="2026-11-12T00:00:00Z",
        agenda_prompt="วิเคราะห์ $NVDA",
        primary_ticker="NVDA",
        instrument_anchor=InstrumentIdentityAnchor(
            ticker="NVDA",
            company_name="NVIDIA Corporation",
            market="US",
            sector="Tech",
            industry="Semiconductors",
            currency="USD",
            benchmark_ticker="SPY",
            current_price="135.00",
        ),
        portfolio_guard=PortfolioHoldingGuardState(
            attachment_state=ContextAttachmentState.ATTACHED_HELD,
            portfolio_attached=True,
            is_held=True,
            units="10.0000",
            avg_cost="110.00",
            portfolio_total_nav_thb="1000000.00",
            summary_text="[PORTFOLIO_STATE: ATTACHED_HELD] NAV 1,000,000 THB | FX USD/THB: 35.00",
        ),
    )

    caplog.set_level(logging.INFO, logger="tools.roundtable.debate_engine")
    engine = GeminiRoundtableEngine(llm_factory=lambda *_: _TelemetryRawLLM())
    t = await engine.generate_guru_turn(
        speaker_id="value_guru",
        round_stage="opening",
        packet=pkt,
        prior_turns=[],
        unspoken_speakers=[],
    )
    assert t.stance_tag == "HOLD"
    assert any(
        "[Roundtable LLM Telemetry]" in rec.message
        and "input_tokens=10000" in rec.message
        and "cached_tokens=7500" in rec.message
        and "cache_hit=75.0%" in rec.message
        for rec in caplog.records
    )

    # Sandbox FX THB->USD conversion & multi-number target price + shares
    calc = RoundtableSandboxCalcAdapter()
    thb_res = calc.run_what_if_calculation("ถ้าซื้อเพิ่ม 70,000 บาท จะได้กี่หุ้น", pkt)
    assert thb_res["is_success"] is True
    thb_out = json.loads(thb_res["output"])
    assert thb_out["usd_thb_fx_rate"] == "35.00"
    assert thb_out["budget_converted_usd"] == "2000.00"
    # 2000 / 135 = 14.8148 shares -> blended ((10*110) + 2000) / 24.8148 = 124.93
    assert thb_out["estimated_shares_at_effective_price"] == "14.8148"
    assert thb_out["blended_avg_cost"] == "124.93"

    multi_res = calc.run_what_if_calculation("ถ้าราคาลงเหลือ 120 แล้วซื้อ 10 หุ้น ต้นทุนเฉลี่ยเท่าไหร่", pkt)
    assert multi_res["is_success"] is True
    multi_out = json.loads(multi_res["output"])
    assert multi_out["scenario_price"] == "120.00"
    assert multi_out["if_add_units"] == "10"
    assert multi_out["additional_cost_in_stock_ccy"] == "1200.00"
    assert multi_out["blended_avg_cost"] == "115.00"  # (1100 + 1200) / 20

    # Missing FX rate on THB budget vs USD stock emits explicit fx_conversion_missing warning instead of dividing THB/USD blindly
    no_fx_pkt = FrozenContextPacket(
        session_id="s_nofx",
        created_at_iso="2026-11-12T00:00:00Z",
        agenda_prompt="$NVDA",
        primary_ticker="NVDA",
        instrument_anchor=pkt.instrument_anchor,
        portfolio_guard=PortfolioHoldingGuardState(
            attachment_state=ContextAttachmentState.NOT_ATTACHED_DO_NOT_ASSUME,
            summary_text="[PORTFOLIO_STATE: NOT_ATTACHED_DO_NOT_ASSUME]",
        ),
    )
    no_fx_res = calc.run_what_if_calculation("ถ้าซื้อ 35,000 บาท", no_fx_pkt)
    assert no_fx_res["is_success"] is True
    no_fx_out = json.loads(no_fx_res["output"])
    assert "fx_conversion_missing" in no_fx_out
    assert "estimated_shares_at_effective_price" not in no_fx_out


def test_sandbox_what_if_keeps_allocation_price_drop_budget_and_price_distinct():
    packet = FrozenContextPacket(
        session_id="s_what_if_types",
        created_at_iso="2026-11-13T00:00:00Z",
        agenda_prompt="What if scenarios for NVDA",
        primary_ticker="NVDA",
        instrument_anchor=InstrumentIdentityAnchor(
            ticker="NVDA",
            company_name="NVIDIA Corporation",
            market="US",
            sector="Technology",
            industry="Semiconductors",
            currency="USD",
            benchmark_ticker="SPY",
            current_price="135.00",
        ),
        portfolio_guard=PortfolioHoldingGuardState(
            attachment_state=ContextAttachmentState.ATTACHED_HELD,
            portfolio_attached=True,
            is_held=True,
            units="10.0000",
            avg_cost="110.00",
            portfolio_total_nav_thb="1000000.00",
            summary_text="[PORTFOLIO_STATE: ATTACHED_HELD] NAV 1,000,000 THB",
        ),
        equity_data={"fx_rates": {"USD/THB": 35.0}},
    )
    adapter = RoundtableSandboxCalcAdapter()

    def _calculate(question: str):
        result = adapter.run_what_if_calculation(question, packet)
        assert result["is_success"] is True
        return json.loads(result["output"])

    allocation = _calculate("target allocation 10%")
    assert allocation["target_allocation_pct"] == "10"
    assert allocation["target_allocation_thb"] == "100000.00"
    assert "price_after_drop_pct" not in allocation
    assert "price_drop_percent" not in allocation

    price_drop = _calculate("price drops 10%")
    assert price_drop["price_drop_percent"] == "10"
    assert price_drop["price_after_drop_pct"] == "121.50"
    assert "target_allocation_pct" not in price_drop

    shares_at_price = _calculate("Buy 10 shares at $120")
    assert shares_at_price["scenario_price"] == "120.00"
    assert shares_at_price["if_add_units"] == "10"
    assert shares_at_price["additional_cost_in_stock_ccy"] == "1200.00"
    assert "budget_amount" not in shares_at_price
    assert "estimated_shares_at_effective_price" not in shares_at_price

    budget_and_allocation = _calculate("Buy 1000 USD with target allocation 10%")
    assert budget_and_allocation["budget_amount"] == "1000"
    assert budget_and_allocation["estimated_shares_at_effective_price"] == "7.4074"
    assert budget_and_allocation["target_allocation_pct"] == "10"
    assert "price_after_drop_pct" not in budget_and_allocation

    for question in (
        "What if revenue grows 10%?",
        "price rises 10%",
        "10%",
        "forecast over 10 years",
        "revenue multiple is 2.5x",
        "I currently hold 10 shares",
    ):
        unsupported = _calculate(question)
        assert unsupported["calculation_status"] == "needs_clarification"
        assert unsupported["scenario_supported"] is False
        assert "if_add_units" not in unsupported
        assert "estimated_shares_at_effective_price" not in unsupported
        assert "estimated_shares_at_current_price" not in unsupported
        assert "blended_avg_cost" not in unsupported

    decimal_price = _calculate("Buy at $120.50 with a $1000 budget")
    assert decimal_price["scenario_price"] == "120.50"
    assert decimal_price["budget_amount"] == "1000"
    assert decimal_price["estimated_shares_at_effective_price"] == "8.2988"
    assert decimal_price["additional_cost_in_stock_ccy"] == "1000.01"

    for invalid_question in (
        "Buy -$1000 of shares at $120",
        "Buy $-1000 of shares at $120",
        "Buy -1000 USD of shares at $120",
        "Buy -10 shares at $120",
        "Buy 10 shares at $0",
        "Buy $0 of shares at $120",
        "Buy 0 shares at $120",
        "Buy ?10 shares at $120",
        "Buy 10 shares at -$120",
        "Buy 1e3 USD of shares at $120",
    ):
        invalid = _calculate(invalid_question)
        assert invalid["calculation_status"] == "needs_clarification"
        assert invalid["scenario_supported"] is False
        assert not any(key in invalid for key in (
            "if_add_units",
            "estimated_shares_at_effective_price",
            "estimated_shares_at_current_price",
            "additional_cost_in_stock_ccy",
            "blended_avg_cost",
        ))

    usd_price_first = _calculate("At $120 buy $1000 of shares")
    usd_budget_first = _calculate("Buy $1000 of shares at $120")
    for result in (usd_price_first, usd_budget_first):
        assert result["scenario_price"] == "120.00"
        assert result["budget_amount"] == "1000"
        assert result["budget_currency"] == "USD"
        assert result["estimated_shares_at_effective_price"] == "8.3333"
        assert result["additional_cost_in_stock_ccy"] == "1000.00"

    thb_price_first = _calculate("At \u0e3f120 buy 1000 baht of shares")
    thb_budget_first = _calculate("Buy 1000 baht of shares at \u0e3f120")
    for result in (thb_price_first, thb_budget_first):
        assert result["scenario_price"] == "120.00"
        assert result["budget_amount"] == "1000"
        assert result["budget_currency"] == "THB"
        assert result["budget_converted_usd"] == "28.57"
        assert result["scenario_price_currency"] == "THB"
        assert result["scenario_price_in_stock_currency"] == "3.43"
        assert result["estimated_shares_at_effective_price"] == "8.3333"
    assert thb_price_first["additional_cost_in_stock_ccy"] == thb_budget_first["additional_cost_in_stock_ccy"]

    for question, expected_action in (
        ("Do not buy $1000 of shares at $120", "NEGATED"),
        ("I decided not to buy $1000 of shares at $120", "NEGATED"),
        ("\u0e44\u0e21\u0e48\u0e04\u0e27\u0e23\u0e0b\u0e37\u0e49\u0e2d 1000 baht of shares at \u0e3f120", "NEGATED"),
        ("\u0e2b\u0e49\u0e32\u0e21\u0e0b\u0e37\u0e49\u0e2d 1000 baht of shares at \u0e3f120", "NEGATED"),
        ("Sell $1000 of shares at $120", "SELL"),
        ("Sell 10 shares at $120", "SELL"),
        ("I currently hold 10 shares", "HELD"),
        ("\u0e02\u0e32\u0e22 1000 baht of shares at \u0e3f120", "SELL"),
        ("I hold $1000 worth of shares", "HELD"),
        ("\u0e44\u0e21\u0e48\u0e0b\u0e37\u0e49\u0e2d 1000 baht of shares at 120 baht", "NEGATED"),
    ):
        unsupported = _calculate(question)
        assert unsupported["calculation_status"] == "needs_clarification"
        assert unsupported["transaction_action"] == expected_action
        assert "if_add_units" not in unsupported
        assert "estimated_shares_at_effective_price" not in unsupported
        assert "blended_avg_cost" not in unsupported

    ambiguous_budget = _calculate("Buy $1000 or $2000 of shares at $120")
    assert ambiguous_budget["calculation_status"] == "needs_clarification"
    assert "multiple_budget_amounts" in ambiguous_budget["clarification_reasons"]
    assert "if_add_units" not in ambiguous_budget
    assert "estimated_shares_at_effective_price" not in ambiguous_budget

    from dataclasses import replace

    thb_stock_packet = replace(
        packet,
        instrument_anchor=replace(packet.instrument_anchor, currency="THB"),
    )
    thb_stock = json.loads(RoundtableSandboxCalcAdapter().run_what_if_calculation(
        "At $120 buy $1000 of shares", thb_stock_packet
    )["output"])
    assert thb_stock["scenario_price_currency"] == "USD"
    assert thb_stock["scenario_price_in_stock_currency"] == "4200.00"
    assert thb_stock["budget_converted_thb"] == "35000.00"
    assert thb_stock["estimated_shares_at_effective_price"] == "8.3333"
    assert thb_stock["additional_cost_in_stock_ccy"] == "34999.86"

    usd_stock_thb_price = _calculate("Buy $1000 of shares at 120 baht")
    assert usd_stock_thb_price["scenario_price_currency"] == "THB"
    assert usd_stock_thb_price["estimated_shares_at_effective_price"] == "291.6667"

    no_fx_packet = replace(packet, equity_data={})
    no_fx_price = json.loads(RoundtableSandboxCalcAdapter().run_what_if_calculation(
        "At \u0e3f120 buy $1000 of shares", no_fx_packet
    )["output"])
    assert no_fx_price["calculation_status"] == "needs_clarification"
    assert "fx_conversion_missing" in no_fx_price
    assert "if_add_units" not in no_fx_price
    assert "estimated_shares_at_effective_price" not in no_fx_price


@pytest.mark.asyncio
async def test_fierce_debate_engine_temperature_clash_ledger_stage_intro_and_kv_prefix():
    from tools.roundtable.debate_engine import (
        GeminiRoundtableEngine,
        _format_packet_prompt_context,
        opponent_argument_or_opening,
    )

    pkt = FrozenContextPacket(
        session_id="s_fierce_engine",
        created_at_iso="2026-11-12T00:00:00Z",
        agenda_prompt="วิเคราะห์ $NVDA ระยะยาว",
        primary_ticker="NVDA",
        debate_intensity="crossfire",
        debate_depth="deep_youtube",
        instrument_anchor=InstrumentIdentityAnchor(
            ticker="NVDA",
            company_name="NVIDIA Corporation",
            market="US",
            sector="Tech",
            industry="Semiconductors",
            currency="USD",
            benchmark_ticker="SPY",
            current_price="135.00",
        ),
    )

    # 1. Exact zero-prior legacy opening string preserved
    legacy_zero = opponent_argument_or_opening("value_guru", "opening", [], ["quality_guru"])
    assert "คุณเป็นผู้เปิดประเด็นคนแรก" in legacy_zero
    assert "ห้ามสมมติหรืออ้างคำพูดแทนกูรูคนอื่นที่ยังไม่ได้พูด" in legacy_zero

    # 2. Heated/crossfire ledger includes Clash Matrix, Unchallenged Pillars & Intra-Camp Friction
    prior = [
        RoundtableTurnPayload(
            speaker_id="value_guru",
            round_stage="opening",
            spoken_dialogue="ราคา P/E 40x ทำให้ Margin of Safety บางครับ",
            evidence_bullets=["[FACT] P/E 40x"],
            stance_tag="HOLD",
        ),
        RoundtableTurnPayload(
            speaker_id="quality_guru",
            round_stage="opening",
            spoken_dialogue="ROIC 50% และ Moat คุ้มค่าระยะยาว",
            evidence_bullets=["[FACT] ROIC 50%"],
            stance_tag="OVERWEIGHT",
        ),
        RoundtableTurnPayload(
            speaker_id="growth_guru",
            round_stage="opening",
            spoken_dialogue="รายได้ AI โต 90% YoY ต่อเนื่อง",
            evidence_bullets=["[FACT] Growth 90%"],
            stance_tag="BUY",
        ),
    ]
    rebuttal_ledger = opponent_argument_or_opening(
        "growth_guru",
        "rebuttal",
        prior,
        ["skeptic_guru"],
        packet=pkt,
    )
    assert "Active Clash & Unchallenged Pillars" in rebuttal_ledger
    assert "เสาหลักที่ยังไม่มีใครซักค้านอย่างจริงจัง" in rebuttal_ledger
    assert "No-Soft-Agreement" in rebuttal_ledger

    # Intra-Camp Friction triggers when all prior non-self speakers sit in same camp
    all_bull_prior = [
        RoundtableTurnPayload(
            speaker_id="quality_guru",
            round_stage="opening",
            spoken_dialogue="ROIC 50% และ Moat คุ้มค่าระยะยาว",
            evidence_bullets=["[FACT] ROIC 50%"],
            stance_tag="OVERWEIGHT",
        ),
        RoundtableTurnPayload(
            speaker_id="macro_tactical_guru",
            round_stage="opening",
            spoken_dialogue="สภาพคล่องหนุนตลาดขาขึ้นต่อเนื่อง",
            evidence_bullets=["[FACT] Reflation"],
            stance_tag="BUY",
        ),
    ]
    intra_ledger = opponent_argument_or_opening(
        "growth_guru",
        "rebuttal",
        all_bull_prior,
        ["skeptic_guru"],
        packet=pkt,
    )
    assert "Intra-Camp Friction" in intra_ledger

    cf_ledger = opponent_argument_or_opening(
        "skeptic_guru",
        "crossfire",
        prior,
        [],
        packet=pkt,
        crossfire_target="growth_guru",
    )
    assert "คำสั่งรอบ Crossfire ดวลตัวต่อตัว" in cf_ledger
    assert "growth_guru" in cf_ledger

    # 3. Dynamic temperature scaling + default constructor override preservation
    class _FakeLLM:
        def with_structured_output(self, schema):
            class _Runner:
                def invoke(self, messages):
                    return RoundtableTurnPayload(
                        speaker_id="hermes_moderator",
                        round_stage="stage_intro",
                        spoken_dialogue="ยินดีต้อนรับสู่เวทีโต้วาทีวิเคราะห์ $NVDA ครบทุกมุมมองครับ",
                        evidence_bullets=["[FACT] NVIDIA price 135.00"],
                        stance_tag="REVIEW",
                    )

            return _Runner()

    eng_default = GeminiRoundtableEngine(llm_factory=lambda *_: _FakeLLM())
    assert eng_default._effective_temperature(pkt, round_stage="opening") == 0.65
    assert eng_default._effective_temperature(pkt, round_stage="stage_intro") == 0.25
    assert eng_default._effective_temperature(pkt, round_stage="synthesis") == 0.25
    eng_custom = GeminiRoundtableEngine(llm_factory=lambda *_: _FakeLLM(), temperature=0.42)
    assert eng_custom._effective_temperature(pkt, round_stage="opening") == 0.42

    # 4. Stage Intro & KV cache prefix invariance check across roles
    intro = await eng_default.generate_hermes_stage_intro(
        packet=pkt,
        ordered_gurus=["value_guru", "growth_guru", "skeptic_guru"],
    )
    assert intro.speaker_id == "hermes_moderator"
    assert intro.round_stage == "stage_intro"
    shared_prefix = _format_packet_prompt_context(pkt)
    assert _format_packet_prompt_context(pkt, speaker_id="value_guru").startswith(shared_prefix)
    assert _format_packet_prompt_context(pkt, speaker_id="hermes_moderator").startswith(shared_prefix)








def test_role_section_plan_differentiation_anti_echo_opening_and_forensic_pair_prompt():
    from dataclasses import replace
    from core.roundtable.models import (
        GURU_PERSONA_CATALOG,
        FrozenContextPacket,
        InstrumentIdentityAnchor,
        SpeakerId,
        merge_persona_overrides,
    )
    from schemas.roundtable_schemas import RoundtableTurnPayload
    from tools.roundtable.debate_engine import (
        _ROLE_SECTION_PLAN,
        GeminiRoundtableEngine,
        _format_role_focus_lens,
        opponent_argument_or_opening,
    )

    # 1. Value vs Quality section plans are distinct (Valuation/Risks vs Quality/Growth)
    val_primary, val_secondary = _ROLE_SECTION_PLAN["value_guru"]
    qual_primary, qual_secondary = _ROLE_SECTION_PLAN["quality_guru"]
    assert "VALUATION" in val_primary and "RISKS_THESIS" in val_primary
    assert "QUALITY" in val_secondary
    assert "QUALITY" in qual_primary and "GROWTH_GUIDANCE" in qual_primary
    assert "VALUATION" in qual_secondary

    personas = merge_persona_overrides(GURU_PERSONA_CATALOG)
    forensic_personas = tuple(
        replace(p, preset_id="quality_forensic_auditor")
        if p.speaker_id == SpeakerId.QUALITY_GURU
        else (
            replace(p, preset_id="skeptic_forensic_sleuth")
            if p.speaker_id == SpeakerId.SKEPTIC_GURU
            else p
        )
        for p in personas
    )
    pkt = FrozenContextPacket(
        session_id="s_diff",
        created_at_iso="2026-11-13T00:00:00Z",
        agenda_prompt="ตรวจงบ $NVDA",
        context_refs=("$NVDA",),
        primary_ticker="NVDA",
        instrument_anchor=InstrumentIdentityAnchor(
            ticker="NVDA",
            company_name="NVIDIA Corporation",
            market="US",
            sector="Technology",
            industry="Semiconductors",
            currency="USD",
            benchmark_ticker="SPY",
            current_price="138.00",
        ),
        active_personas=tuple(p.to_dict() for p in forensic_personas),
    )

    q_lens = _format_role_focus_lens("quality_guru", pkt)
    assert "คุณภาพกำไร, ROIC" in q_lens.split("Cross-Check")[0]
    assert "ประเมินมูลค่า" in q_lens.split("Cross-Check")[1]

    # 2. Round 1 opening prompt emits Anti-Echo rule once an earlier guru has spoken
    val_open = RoundtableTurnPayload(
        speaker_id="value_guru",
        round_stage="opening",
        spoken_dialogue="ราคาปัจจุบันเผื่อ Margin of Safety แคบเมื่อเทียบ Free Cash Flow Yield 2.4%",
        evidence_bullets=["[MATH] FCF Yield 2.4%"],
        stance_tag="HOLD",
    )
    anti_echo_prompt = opponent_argument_or_opening(
        "quality_guru",
        "opening",
        prior_turns=[val_open],
        unspoken_speakers=["growth_guru", "macro_tactical_guru", "skeptic_guru"],
        packet=pkt,
    )
    assert "[กฎกันพูดซ้ำรอบเปิด (Anti-Echo Rule)]" in anti_echo_prompt
    assert "Margin of Safety" in anti_echo_prompt

    # 3. Forensic Pair Coordination block appears when quality_forensic_auditor + skeptic_forensic_sleuth co-exist
    engine = GeminiRoundtableEngine()
    q_role = engine._load_persona_role_block("quality_guru", pkt)
    s_role = engine._load_persona_role_block("skeptic_guru", pkt)
    assert "[FORENSIC PAIR COORDINATION — ห้ามวิเคราะห์ทับซ้อน]" in q_role
    assert "[FORENSIC PAIR COORDINATION — ห้ามวิเคราะห์ทับซ้อน]" in s_role
    assert "Quality Auditor" in q_role and "OCF/Net Income" in q_role
    assert "Skeptic Sleuth" in s_role and "Credit Rating, ICR" in s_role


@pytest.mark.asyncio
async def test_hermes_synthesis_prompts_missing_seat_coverage_when_non_skeptic_disabled():
    from dataclasses import replace
    from core.roundtable.models import (
        GURU_PERSONA_CATALOG,
        FrozenContextPacket,
        InstrumentIdentityAnchor,
        SpeakerId,
        merge_persona_overrides,
    )
    from schemas.roundtable_schemas import RoundtableTurnPayload
    from tools.roundtable.debate_engine import GeminiRoundtableEngine

    personas = merge_persona_overrides(GURU_PERSONA_CATALOG)
    partial_personas = tuple(
        replace(p, enabled=False)
        if p.speaker_id in (SpeakerId.VALUE_GURU, SpeakerId.MACRO_TACTICAL_GURU)
        else p
        for p in personas
    )
    pkt = FrozenContextPacket(
        session_id="s_missing_seats",
        created_at_iso="2026-11-13T00:00:00Z",
        agenda_prompt="วิเคราะห์ $NVDA",
        context_refs=("$NVDA",),
        primary_ticker="NVDA",
        instrument_anchor=InstrumentIdentityAnchor(
            ticker="NVDA",
            company_name="NVIDIA Corporation",
            market="US",
            sector="Technology",
            industry="Semiconductors",
            currency="USD",
            benchmark_ticker="SPY",
            current_price="138.00",
        ),
        active_personas=tuple(p.to_dict() for p in partial_personas),
    )

    captured_prompts: list[str] = []

    class _FakeStructured:
        def __init__(self, schema: type):
            self._schema = schema

        def invoke(self, messages):
            captured_prompts.append("\n".join(str(m.content) for m in messages))
            # Build minimal valid _HermesCombinedOutput instance
            closing = RoundtableTurnPayload(
                speaker_id="hermes_moderator",
                round_stage="synthesis",
                spoken_dialogue="สรุปผลการหารือครอบคลุมมิติมูลค่าและมหภาคที่ปิดไว้แล้วครับ",
                evidence_bullets=["[FACT] สรุปรอบด้านครบ 5 เสาหลัก"],
                stance_tag="OVERWEIGHT",
            )
            playbook = {
                "final_rating": "OVERWEIGHT",
                "action_stance": "OVERWEIGHT",
                "core_conviction_score": 2,
                "executive_verdict": "ทยอยสะสมอย่างระมัดระวัง",
                "guru_stance_breakdown": {"quality_guru": "OVERWEIGHT"},
                "consensus_points": ["คูเมืองแกร่ง"],
                "tension_axes": [],
                "personal_axis_alignment": [],
                "portfolio_context_summary": "ไม่ได้แนบพอร์ต",
                "action_playbook": {
                    "primary_action": {
                        "option_id": "opt_1",
                        "label": "Staged Entry",
                        "speaker_source": "hermes_moderator",
                        "conviction_level": 2,
                        "action_steps": ["แบ่งไม้เข้า"],
                    },
                    "alternatives": [],
                },
                "kill_switches": ["Gross Margin < 70%"],
            }
            parsed = self._schema(closing_turn=closing, decision_playbook=playbook)
            return {"parsed": parsed, "raw": None}

    class _FakeLLM:
        def with_structured_output(self, schema, include_raw=True):
            return _FakeStructured(schema)

    engine = GeminiRoundtableEngine(llm_factory=lambda _m, _t: _FakeLLM())
    sample_turn = RoundtableTurnPayload(
        speaker_id="quality_guru",
        round_stage="opening",
        spoken_dialogue="ROIC สูงและกระแสเงินสดแข็งแกร่งมาก",
        evidence_bullets=["[FACT] ROIC 68%"],
        stance_tag="OVERWEIGHT",
    )
    closing_turn, _ = await engine.generate_hermes_synthesis(
        packet=pkt,
        all_turns=[sample_turn],
    )
    assert closing_turn.speaker_id == "hermes_moderator"
    assert len(captured_prompts) == 1
    full_prompt = captured_prompts[0]
    assert "[MISSING SEAT COVERAGE CHECK — ปิดใช้งานบางสาย]" in full_prompt
    assert "value_guru" in full_prompt
    assert "macro_tactical_guru" in full_prompt
    # Skeptic was enabled so mandatory skeptic-inactive warning must NOT fire
    assert "[MANDATORY DEVIL'S ADVOCATE CHECK — SKEPTIC SEAT INACTIVE]" not in full_prompt


@pytest.mark.asyncio
async def test_tts_sanitization_rpm_pacing_and_429_retry(tmp_path):
    import asyncio
    import time
    from validators.roundtable_verifier import (
        prepare_dialogue_for_tts,
        sanitize_spoken_dialogue_for_tts,
    )

    # Mixed Cyrillic typos inside Thai loanwords must be repaired cleanly
    mixed = "การเข้าซื้อแบบ\u0e2d\u0433\u0440\u0435\u0441\u0e0b\u0e35\u0e1fใน $NVDA ที่ 67.1% และ P/E 25 @Value"
    sanitized = sanitize_spoken_dialogue_for_tts(mixed)
    assert "อเกรสซีฟ" in sanitized
    assert not any("\u0400" <= ch <= "\u04ff" for ch in sanitized)

    tts_ready = prepare_dialogue_for_tts(mixed)
    assert "67.1 เปอร์เซ็นต์" in tts_ready
    assert "พีอี" in tts_ready
    assert "$NVDA" not in tts_ready and "NVDA" in tts_ready
    assert "@Value" not in tts_ready and "Value" in tts_ready

    # RPM pacemaker spaces out bursts cleanly and 429 retry recovers instead of falling back
    attempts_by_text: dict[str, int] = {}
    call_times: list[float] = []

    def _flaky_paced_fetcher(_model: str, _voice: str, text: str) -> bytes:
        call_times.append(time.monotonic())
        attempts_by_text[text] = attempts_by_text.get(text, 0) + 1
        if "retry-me" in text and attempts_by_text[text] == 1:
            raise RuntimeError("429 RESOURCE_EXHAUSTED retryDelay: '1s'")
        return b"\x04\x00" * 240

    adapter = GeminiTTSAdapter(
        cache_dir=tmp_path / "paced",
        tts_fetcher=_flaky_paced_fetcher,
        rpm_limit=2,
        max_concurrency=2,
        max_429_retries=2,
        window_seconds=0.15,
        retry_delay_fn=lambda _exc, _attempt: 0.01,
    )

    res1, res2, res3 = await asyncio.gather(
        adapter.synthesize_turn_audio(text="ประโยคหนึ่งปกติ", speaker_id="value_guru"),
        adapter.synthesize_turn_audio(text="ประโยคสอง retry-me", speaker_id="skeptic_guru"),
        adapter.synthesize_turn_audio(text="ประโยคสามรอบถัดไป", speaker_id="growth_guru"),
    )
    assert all(r.audio_status == "ready" for r in (res1, res2, res3))
    assert attempts_by_text["ประโยคสอง retry-me"] == 2
    assert len(call_times) == 4
    assert (call_times[-1] - call_times[0]) >= 0.12


@pytest.mark.asyncio
async def test_stream_debate_emits_audio_ready_while_next_llm_turn_still_pending():
    import asyncio
    from dataclasses import replace
    import json
    from application.roundtable.ports import AudioSynthesisResult
    from application.roundtable.service import RoundtableApplicationService
    from core.roundtable.models import (
        GURU_PERSONA_CATALOG,
        FrozenContextPacket,
        InstrumentIdentityAnchor,
        RoundtableSession,
        RoundtableTurn,
        SpeakerId,
    )
    from schemas.roundtable_schemas import (
        ActionPlaybookOption,
        ActionPlaybookSpec,
        DistilledDecisionPlaybook,
        RoundtableTurnPayload,
    )

    personas = tuple(
        p
        if p.speaker_id
        in (SpeakerId.HERMES_MODERATOR, SpeakerId.VALUE_GURU, SpeakerId.GROWTH_GURU)
        else replace(p, enabled=False)
        for p in GURU_PERSONA_CATALOG
    )
    pkt = FrozenContextPacket(
        session_id="rt_rt_audio_flush",
        created_at_iso="2026-11-14T00:00:00Z",
        agenda_prompt="วิเคราะห์ $NVDA",
        context_refs=("$NVDA",),
        primary_ticker="NVDA",
        instrument_anchor=InstrumentIdentityAnchor(
            ticker="NVDA",
            company_name="NVIDIA Corporation",
            market="US",
            sector="Technology",
            industry="Semiconductors",
            currency="USD",
            benchmark_ticker="SPY",
            current_price="138.00",
        ),
        active_personas=tuple(p.to_dict() for p in personas),
    )
    session = RoundtableSession(
        session_id="rt_rt_audio_flush",
        title="NVDA Realtime Audio Flush",
        primary_ticker="NVDA",
        benchmark_ticker="SPY",
        entry_price_at_session="138.00",
        agenda_prompt="วิเคราะห์ $NVDA",
        context_refs=("$NVDA",),
        frozen_packet_json=json.dumps(pkt.to_dict(), ensure_ascii=False),
        packet_fingerprint_sha256=pkt.packet_fingerprint_sha256,
        decision_playbook_json=None,
        final_rating="REVIEW",
        settlement_status="pending",
        settled_return_pct=None,
        settled_alpha_pct=None,
        settled_at_date=None,
        reflection_lesson=None,
        status="active",
        created_at=1730000000.0,
        updated_at=1730000000.0,
    )

    class _MemRepo:
        def __init__(self, sess: RoundtableSession) -> None:
            self._session = sess
            self._turns: list[RoundtableTurn] = []

        def get_session(self, session_id: str):
            return self._session if session_id == self._session.session_id else None

        def get_turns(self, session_id: str):
            return list(self._turns) if session_id == self._session.session_id else []

        def get_next_seq(self, _session_id: str) -> int:
            return len(self._turns) + 1

        def append_turn(
            self,
            *,
            turn_id: str,
            session_id: str,
            seq: int,
            speaker_id: str,
            round_stage: str,
            replying_to,
            spoken_dialogue: str,
            evidence_bullets,
            stance_tag: str,
            audio_cache_key,
            sizing_posture=None,
        ) -> RoundtableTurn:
            t = RoundtableTurn(
                turn_id=turn_id,
                session_id=session_id,
                seq=seq,
                speaker_id=speaker_id,
                round_stage=round_stage,
                replying_to=replying_to,
                spoken_dialogue=spoken_dialogue,
                evidence_bullets=tuple(evidence_bullets),
                stance_tag=stance_tag,
                audio_cache_key=audio_cache_key,
                created_at=1730000000.0 + seq,
                sizing_posture=sizing_posture,
            )
            self._turns.append(t)
            return t

        def update_turn_audio_key(self, turn_id: str, audio_cache_key: str) -> None:
            self._turns = [
                replace(t, audio_cache_key=audio_cache_key) if t.turn_id == turn_id else t
                for t in self._turns
            ]

        def update_session_playbook(
            self, *, session_id: str, decision_playbook_json: str, final_rating: str, status: str = "completed"
        ):
            self._session = replace(
                self._session,
                decision_playbook_json=decision_playbook_json,
                final_rating=final_rating,
                status=status,
            )
            return self._session

    turn2_started = asyncio.Event()
    release_turn2 = asyncio.Event()

    class _GateEngine:
        def __init__(self) -> None:
            self.call_count = 0

        async def generate_guru_turn(self, *, speaker_id: str, round_stage: str, **_kwargs):
            self.call_count += 1
            if self.call_count == 2:
                turn2_started.set()
                await release_turn2.wait()
            return RoundtableTurnPayload(
                speaker_id=speaker_id,
                round_stage=round_stage,
                spoken_dialogue=f"บทพูดทดสอบ {speaker_id} รอบ {round_stage}",
                evidence_bullets=["[FACT] test"],
                stance_tag="HOLD",
            )

        async def generate_hermes_synthesis(self, **_kwargs):
            return (
                RoundtableTurnPayload(
                    speaker_id="hermes_moderator",
                    round_stage="synthesis",
                    spoken_dialogue="สรุปผลการประชุมสภากูรู",
                    evidence_bullets=["[FACT] synthesis"],
                    stance_tag="HOLD",
                ),
                DistilledDecisionPlaybook(
                    final_rating="HOLD",
                    action_stance="HOLD",
                    core_conviction_score=1,
                    executive_verdict="รอจังหวะ",
                    guru_stance_breakdown={"value_guru": "HOLD", "growth_guru": "HOLD"},
                    action_playbook=ActionPlaybookSpec(
                        primary_action=ActionPlaybookOption(
                            option_id="opt_1",
                            label="Hold",
                            speaker_source="value_guru",
                        )
                    ),
                ),
            )

    class _FastStubTTS:
        async def synthesize_turn_audio(self, *, text: str, speaker_id: str, **_kwargs):
            await asyncio.sleep(0.01)
            return AudioSynthesisResult(audio_cache_key=f"wav_{speaker_id}", audio_status="ready")

    class _UnusedAssembler:
        pass

    repo = _MemRepo(session)
    service = RoundtableApplicationService(
        repo=repo,  # type: ignore[arg-type]
        assembler=_UnusedAssembler(),  # type: ignore[arg-type]
        engine=_GateEngine(),  # type: ignore[arg-type]
        tts=_FastStubTTS(),  # type: ignore[arg-type]
    )

    events: list[tuple[str, dict]] = []
    turn1_id: str | None = None
    turn1_audio_flushed_while_turn2_blocked = False

    async def _run_consume():
        nonlocal turn1_id, turn1_audio_flushed_while_turn2_blocked
        async for ev_name, payload in service.stream_debate("rt_rt_audio_flush", voice_enabled=True):
            events.append((ev_name, payload))
            if ev_name == "turn_completed" and turn1_id is None:
                turn1_id = payload["turn_id"]
            if ev_name == "audio_ready" and payload.get("turn_id") == turn1_id:
                assert turn2_started.is_set()
                assert not release_turn2.is_set()
                turn1_audio_flushed_while_turn2_blocked = True
                release_turn2.set()

    await asyncio.wait_for(_run_consume(), timeout=3.0)
    assert turn1_audio_flushed_while_turn2_blocked is True
    assert any(ev == "done" for ev, _ in events)


def _synth_tone_pcm_bytes(duration_sec: float, *, amplitude: int = 5000, freq_hz: float = 220.0, sr: int = 24000) -> bytes:
    import math
    import struct

    n = int(round(duration_sec * sr))
    if amplitude <= 0:
        return b"\x00\x00" * n
    samples = [int(amplitude * math.sin(2.0 * math.pi * freq_hz * i / sr)) for i in range(n)]
    return struct.pack(f"<{n}h", *samples)


@pytest.mark.asyncio
async def test_tts_adapter_validates_trims_recovers_and_self_heals_poisoned_cache(tmp_path):
    long_text = " ".join(
        [
            "คุณคิวาลิตี้ครับ การที่คุณอ้างว่ากระแสเงินสดต่อกำไรสุทธิสูงนั้น",
            "มองข้ามค่าใช้จ่ายหุ้นพนักงานที่ไดลูทผู้ถือหุ้นอย่างต่อเนื่องครับ",
            "และมูลค่าเทอร์มินัลแวลูเกินเจ็ดสิบห้าเปอร์เซ็นต์ยังมีความเสี่ยงสูง",
            "ผมจึงขอยืนยันจุดยืนโฮลด์เพราะต้นทุนเงินทุนยังตึงตัวครับ",
        ]
        * 3
    )
    assert len(long_text) > 500
    assert len(split_text_for_tts_recovery(long_text)) >= 2

    # 1. Runaway non-EOS loop (~280s) on first attempt -> auto-splits into chunks & recovers cleanly
    call_texts: list[str] = []

    def _runaway_then_chunk_ok(_model: str, _voice: str, text: str) -> bytes:
        call_texts.append(text)
        if len(text) == len(long_text):
            speech = _synth_tone_pcm_bytes(16.0, amplitude=6000)
            noise_tail = _synth_tone_pcm_bytes(260.0, amplitude=160, freq_hz=40.0)
            return speech + noise_tail
        return _synth_tone_pcm_bytes(max(6.0, len(text) / 13.5), amplitude=6000)

    adapter = GeminiTTSAdapter(cache_dir=tmp_path / "rec", tts_fetcher=_runaway_then_chunk_ok)
    res_rec = await adapter.synthesize_turn_audio(text=long_text, speaker_id="skeptic_guru")
    assert res_rec.audio_status == "ready"
    assert res_rec.audio_cache_key is not None
    assert len(call_texts) == 3  # 1 failed full-text + 2 recovered sub-chunks
    recovered_wav = adapter.get_audio_bytes(res_rec.audio_cache_key)
    assert recovered_wav is not None
    with wave.open(io.BytesIO(recovered_wav), "rb") as wf:
        rec_dur = wf.getnframes() / float(wf.getframerate())
    assert 35.0 <= rec_dur <= 90.0

    # 2. Mild trailing non-speech noise (e.g. 6s tail) is trimmed on single call without re-calling API
    trim_calls = 0

    def _mild_tail_fetcher(_model: str, _voice: str, text: str) -> bytes:
        nonlocal trim_calls
        trim_calls += 1
        speech_sec = len(text) / 13.5
        return _synth_tone_pcm_bytes(speech_sec, amplitude=6000) + _synth_tone_pcm_bytes(6.0, amplitude=120, freq_hz=45.0)

    trim_adapter = GeminiTTSAdapter(cache_dir=tmp_path / "trim", tts_fetcher=_mild_tail_fetcher)
    res_trim = await trim_adapter.synthesize_turn_audio(text=long_text, speaker_id="value_guru")
    assert res_trim.audio_status == "ready"
    assert trim_calls == 1
    trimmed_wav = trim_adapter.get_audio_bytes(res_trim.audio_cache_key or "")
    assert trimmed_wav is not None
    with wave.open(io.BytesIO(trimmed_wav), "rb") as wf:
        trimmed_dur = wf.getnframes() / float(wf.getframerate())
    assert trimmed_dur < (len(long_text) / 13.5) + 1.5

    # 3. Unrecoverable mid-turn dropout (>3s) falls back to Web Speech and never writes poisoned cache
    def _always_dropout_fetcher(_model: str, _voice: str, _text: str) -> bytes:
        return (
            _synth_tone_pcm_bytes(10.0, amplitude=6000)
            + _synth_tone_pcm_bytes(4.2, amplitude=0)
            + _synth_tone_pcm_bytes(10.0, amplitude=6000)
        )

    drop_dir = tmp_path / "dropout"
    drop_adapter = GeminiTTSAdapter(cache_dir=drop_dir, tts_fetcher=_always_dropout_fetcher)
    res_drop = await drop_adapter.synthesize_turn_audio(text=long_text, speaker_id="quality_guru")
    assert res_drop.audio_status == "fallback_web_speech"
    assert res_drop.audio_cache_key is None
    assert list(drop_dir.glob("*.wav")) == []

    # 4. Self-healing: poisoned pre-existing WAV on disk is rejected, purged, and re-synthesized cleanly
    heal_dir = tmp_path / "heal"
    poison_adapter = GeminiTTSAdapter(cache_dir=heal_dir, tts_fetcher=_runaway_then_chunk_ok)
    key = poison_adapter._compute_cache_key("skeptic_guru", "Orus", long_text)
    heal_dir.mkdir(parents=True, exist_ok=True)
    poison_wav = wrap_pcm_to_wav(_synth_tone_pcm_bytes(10.0, amplitude=6000) + _synth_tone_pcm_bytes(250.0, amplitude=150))
    assert validate_and_sanitize_tts_wav(poison_wav, long_text).is_valid is False
    (heal_dir / f"{key}.wav").write_bytes(poison_wav)

    res_healed = await poison_adapter.synthesize_turn_audio(text=long_text, speaker_id="skeptic_guru")
    assert res_healed.audio_status == "ready"
    healed_bytes = poison_adapter.get_audio_bytes(key)
    assert healed_bytes is not None
    assert validate_and_sanitize_tts_wav(healed_bytes, long_text).is_valid is True



