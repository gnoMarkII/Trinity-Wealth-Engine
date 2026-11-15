"""Unit tests for WP-2: Context Assembler, Settlement Math, and Consistency Verifier."""
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
import time

import pytest

from application.roundtable.context_assembler import RoundtableContextAssembler, extract_context_tags
from core.roundtable.models import (
    ALLOWED_VOICE_NAMES,
    GEMINI_VOICE_CATALOG,
    GURU_PERSONA_CATALOG,
    GURU_PRESET_CATALOG,
    PRESET_ANALYTICAL_RUBRIC_MAP,
    ContextAttachmentState,
    FrozenContextPacket,
    InstrumentIdentityAnchor,
    PortfolioHoldingGuardState,
    RoundtableSession,
    SpeakerId,
    get_preset_analytical_rubric,
    get_seat_ui_context,
    merge_persona_overrides,
    resolve_personas_from_packet,
    sanitize_custom_persona_text,
)
from core.roundtable.settlement import compute_settlement_scorecard, default_benchmark_for_ticker
from schemas.roundtable_schemas import (
    ActionPlaybookOption,
    ActionPlaybookSpec,
    ConsensusTensionAxis,
    DistilledDecisionPlaybook,
    PersonalAxisCheckItem,
    RoundtableTurnPayload,
)
from validators.roundtable_verifier import (
    RoundtableConsistencyVerifier,
    normalize_5tier_stance,
    sanitize_spoken_dialogue_for_tts,
)


class _StubProviders:
    def __init__(
        self,
        *,
        equity_map=None,
        portfolio=None,
        axis=None,
        macro=None,
        digest=None,
        reviews=None,
        news_map=None,
        notes_map=None,
        note_contents=None,
        financial_context=None,
        bucket_policy=None,
        watchlist=None,
        theses=None,
        thesis_exceptions=None,
        promise_scores=None,
        analyst_context=None,
        insider_db=None,
        portfolio_advisory=None,
        investor_essence=None,
    ):
        self._equity_map = equity_map or {}
        self._portfolio = portfolio
        self._axis = axis or {}
        self._macro = macro
        self._digest = {"headline": "Tech earnings beat consensus"} if digest is None else digest
        self._reviews = reviews or {}
        self._news_map = news_map or {}
        self._notes_map = notes_map or {}
        self._note_contents = note_contents or {}
        self._financial_context = financial_context
        self._bucket_policy = bucket_policy
        self._watchlist = watchlist
        self._theses = theses or {}
        self._thesis_exceptions = thesis_exceptions or {}
        self._promise_scores = promise_scores or {}
        self._analyst_context = analyst_context or {}
        self._insider_db = insider_db or {}
        self._portfolio_advisory = portfolio_advisory
        self._investor_essence = investor_essence
        self.advisory_calls = 0

    def resolve_instrument(self, ticker: str):
        return {"ticker": ticker, "market": "TH" if ticker.endswith(".BK") else "US", "currency": "USD"}

    def get_equity_detail(self, ticker: str):
        return self._equity_map.get(ticker.upper())

    def get_equity_position_review(self, ticker: str, portfolio_id: str = "default"):
        return self._reviews.get(ticker.upper())

    def get_equity_news(self, ticker: str):
        return self._news_map.get(ticker.upper())

    def list_equity_notes(self, ticker: str, days: int = 30):
        return self._notes_map.get(ticker.upper())

    def read_equity_note(self, relative_path: str):
        return self._note_contents.get(relative_path)

    def get_macro_dashboard(self):
        return self._macro

    def get_daily_digest(self):
        return self._digest

    def get_portfolio_state(self, portfolio_id: str = "default"):
        return self._portfolio

    def get_confirmed_investment_axis(self, portfolio_id: str = "default"):
        return self._axis

    def get_financial_context(self, portfolio_id: str = "default"):
        return self._financial_context

    def get_bucket_policy(self, portfolio_id: str = "default"):
        return self._bucket_policy

    def get_watchlist(self, portfolio_id: str = "default"):
        return self._watchlist

    def get_active_thesis(self, ticker: str):
        return self._theses.get(ticker.upper())

    def get_thesis_pending_exceptions(self, ticker: str):
        return self._thesis_exceptions.get(ticker.upper())

    def get_promise_ledger_scores(self, ticker: str):
        return self._promise_scores.get(ticker.upper())

    def get_analyst_context(self, ticker: str):
        return self._analyst_context.get(ticker.upper())

    def get_insider_db_summary(self, ticker: str):
        return self._insider_db.get(ticker.upper())

    def get_portfolio_advisory_summary(self, portfolio_id: str = "default"):
        self.advisory_calls += 1
        return self._portfolio_advisory

    def get_investor_essence(self, portfolio_id: str = "default"):
        return self._investor_essence

    def dispatch_equity_research_if_needed(self, ticker: str):
        return None


class _StubRepo:
    def __init__(self, prior_sessions=None):
        self._prior = prior_sessions or []

    def list_sessions_by_ticker(self, ticker: str, limit: int = 10):
        return self._prior


class _StubPrices:
    def __init__(self, closes):
        self._closes = closes

    def get_close_on_or_after(self, ticker: str, target_date_iso: str):
        return self._closes.get(("start", ticker))

    def get_latest_close(self, ticker: str):
        return self._closes.get(("latest", ticker))


def test_settlement_decimal_math_and_lookahead_guard():
    assert default_benchmark_for_ticker("NVDA") == "SPY"
    assert default_benchmark_for_ticker("PTT.BK") == "^SET.BK"

    prev_ts = datetime(2026, 10, 1, 10, 0, tzinfo=timezone.utc).timestamp()
    # Look-ahead bias guard: settled_date == prev_date should skip settlement
    skipped = compute_settlement_scorecard(
        previous_session_id="s_old",
        previous_created_at=prev_ts,
        previous_final_rating="BUY",
        primary_ticker="NVDA",
        benchmark_ticker="SPY",
        entry_price="100.00",
        settled_price="120.00",
        settled_date_iso="2026-10-01",
    )
    assert skipped.settlement_status == "lookahead_skipped"
    assert skipped.raw_return_pct is None

    # Valid settlement: NVDA +20%, SPY +5% => Alpha +15.00%
    settled = compute_settlement_scorecard(
        previous_session_id="s_old",
        previous_created_at=prev_ts,
        previous_final_rating="OVERWEIGHT",
        primary_ticker="NVDA",
        benchmark_ticker="SPY",
        entry_price="100.00",
        settled_price="120.00",
        settled_date_iso="2026-11-01",
        benchmark_entry_price="500.00",
        benchmark_settled_price="525.00",
    )
    assert settled.settlement_status == "settled"
    assert settled.raw_return_pct == "20.00"
    assert settled.benchmark_return_pct == "5.00"
    assert settled.alpha_pct == "15.00"
    assert settled.outcome_label == "outperformed"


def test_context_assembler_prerequisite_guard_and_3state_portfolio():
    refs, tickers, cards = extract_context_tags(["@Macro"], "เปรียบเทียบ $NVDA กับ $TSLA ด้วย")
    assert "$NVDA" in refs and "$TSLA" in refs
    assert tickers == ["NVDA", "TSLA"]
    assert cards["macro"] is True and cards["portfolio"] is False

    # Case 1: TSLA missing -> missing_prerequisite_tickers; @Portfolio NOT attached
    providers = _StubProviders(
        equity_map={
            "NVDA": {
                "ticker": "NVDA",
                "company_name": "NVIDIA Corporation",
                "sector": "Semiconductors",
                "quant_signals": {"current_price": 130.0, "fcf_yield_pct": 1.8},
                "deterministic_scorecard": {"action_stance": "WAIT_DO_NOT_ENTER"},
            }
        },
        axis={
            "content_hash": "ax_1",
            "complete_sections": {"role_models": ["Charlie Munger", "Howard Marks"], "non_actions": ["No leverage"]},
        },
        macro={"regime": "Late Cycle"},
    )
    assembler = RoundtableContextAssembler(providers=providers)
    packet = assembler.assemble(
        session_id="s1",
        agenda_prompt="เทียบ $NVDA กับ $TSLA",
        context_refs=["@Macro"],
    )
    assert packet.missing_prerequisite_tickers == ("TSLA",)
    assert packet.portfolio_guard.attachment_state == ContextAttachmentState.NOT_ATTACHED_DO_NOT_ASSUME
    assert "quality_guru" in packet.prioritized_speakers
    assert "macro_tactical_guru" in packet.prioritized_speakers
    assert "[ABSENT:" in packet.digest_report_text

    # Case 2: @Portfolio attached AND NVDA held
    providers_held = _StubProviders(
        equity_map=providers._equity_map,
        portfolio={
            "total_value_thb": 1000000,
            "cash_pct": 20.0,
            "holdings": [{"symbol": "NVDA", "units": 50, "avg_cost_usd": 110.0, "market_value_thb": 220000}],
        },
    )
    packet_held = RoundtableContextAssembler(providers=providers_held).assemble(
        session_id="s2",
        agenda_prompt="วิเคราะห์ $NVDA",
        context_refs=["$NVDA", "@Portfolio"],
    )
    assert packet_held.portfolio_guard.attachment_state == ContextAttachmentState.ATTACHED_HELD
    assert packet_held.portfolio_guard.is_held is True
    assert packet_held.portfolio_guard.units == "50.0000"
    assert packet_held.portfolio_guard.current_weight_pct == "22.00"

    # Case 3: @Portfolio attached AND AAPL NOT held -> ATTACHED_FLAT_BOOK
    packet_flat = RoundtableContextAssembler(
        providers=_StubProviders(
            equity_map={"AAPL": {"ticker": "AAPL", "company_name": "Apple Inc."}},
            portfolio=providers_held._portfolio,
        )
    ).assemble(
        session_id="s3",
        agenda_prompt="ดู $AAPL",
        context_refs=["$AAPL", "@Portfolio"],
    )
    assert packet_flat.portfolio_guard.attachment_state == ContextAttachmentState.ATTACHED_FLAT_BOOK
    assert packet_flat.portfolio_guard.is_held is False


def test_context_assembler_computes_past_scorecard_when_prior_exists():
    old_ts = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc).timestamp()
    prior = RoundtableSession(
        session_id="old_nvda",
        title="NVDA Prior",
        primary_ticker="NVDA",
        benchmark_ticker="SPY",
        entry_price_at_session="100.00",
        agenda_prompt="ดู NVDA",
        context_refs=("$NVDA",),
        frozen_packet_json="{}",
        packet_fingerprint_sha256="f0",
        decision_playbook_json="{}",
        final_rating="BUY",
        settlement_status="pending",
        settled_return_pct=None,
        settled_alpha_pct=None,
        settled_at_date=None,
        reflection_lesson=None,
        status="completed",
        created_at=old_ts,
        updated_at=old_ts,
    )
    prices = _StubPrices(
        {
            ("latest", "NVDA"): ("2026-10-15", Decimal("115.00")),
            ("start", "SPY"): ("2026-09-01", Decimal("500.00")),
            ("latest", "SPY"): ("2026-10-15", Decimal("510.00")),
        }
    )
    assembler = RoundtableContextAssembler(
        providers=_StubProviders(equity_map={"NVDA": {"ticker": "NVDA", "company_name": "NVIDIA"}}),
        repo=_StubRepo([prior]),
        settlement_prices=prices,
    )
    packet = assembler.assemble(session_id="new_nvda", agenda_prompt="ดู $NVDA", context_refs=["$NVDA"])
    assert packet.past_scorecard is not None
    assert packet.past_scorecard.settlement_status == "settled"
    assert packet.past_scorecard.alpha_pct == "13.00"


def test_roundtable_consistency_verifier():
    assert normalize_5tier_stance("Strong Buy") == "BUY"
    assert normalize_5tier_stance("accumulate") == "OVERWEIGHT"
    assert normalize_5tier_stance("unknown text") == "REVIEW"
    assert sanitize_spoken_dialogue_for_tts("**ราคา** แพง [FACT] ครับ") == "ราคา แพง ครับ"

    verifier = RoundtableConsistencyVerifier()
    packet = FrozenContextPacket(
        session_id="s1",
        created_at_iso="2026-11-10T00:00:00Z",
        agenda_prompt="ดู $NVDA",
        primary_ticker="NVDA",
        data_quality_flags=("negative_fcf_dcf_unavailable:dcf",),
        action_stance_hint="WAIT_DO_NOT_ENTER",
    )

    bad_turn = RoundtableTurnPayload(
        speaker_id="value_guru",
        round_stage="opening",
        spoken_dialogue="**ผมมองว่า** คุณถืออยู่ 50 หุ้น แนะนำซื้อเต็มพอร์ต ราคาเป้าหมายตาม DCF คือ 200",
        evidence_bullets=[],
        stance_tag="BUY",
    )
    res = verifier.verify_turn(bad_turn, packet)
    assert not res.is_valid
    assert len(res.issues) >= 3

    good_turn = RoundtableTurnPayload(
        speaker_id="value_guru",
        round_stage="opening",
        spoken_dialogue="เนื่องจากกระแสเงินสดอิสระยังผันผวน เรายังไม่ควรเร่งซื้อในตอนนี้และควรรอส่วนเผื่อความปลอดภัยก่อนครับ",
        evidence_bullets=["[FACT] Wait stance active"],
        stance_tag="HOLD",
    )
    assert verifier.verify_turn(good_turn, packet).is_valid

    bad_playbook = DistilledDecisionPlaybook(
        final_rating="BUY",
        action_stance="WAIT_DO_NOT_ENTER",
        core_conviction_score=2,
        executive_verdict="ซื้อเลย",
        action_playbook=ActionPlaybookSpec(
            primary_action=ActionPlaybookOption(
                option_id="opt1",
                label="Buy",
                speaker_source="growth_guru",
                action_steps=["เข้าซื้อ"],
            )
        ),
    )
    assert not verifier.verify_playbook(bad_playbook, packet).is_valid


@pytest.mark.parametrize(
    "target_field",
    [
        "executive_verdict",
        "action_stance",
        "portfolio_context_summary",
        "consensus_points[0]",
        "tension_axes[0].topic",
        "tension_axes[0].bull_argument",
        "tension_axes[0].bear_argument",
        "personal_axis_alignment[0].dimension",
        "personal_axis_alignment[0].detail",
        "action_playbook.primary_action.label",
        "action_playbook.primary_action.action_steps[0]",
        "action_playbook.primary_action.entry_zone",
        "action_playbook.primary_action.sizing_note",
        "action_playbook.alternatives[0].label",
        "action_playbook.alternatives[0].action_steps[0]",
        "kill_switches[0]",
        "reflection_summary",
    ],
)
def test_playbook_factual_guards_cover_every_user_visible_text_field(target_field):
    packet = FrozenContextPacket(
        session_id="s_playbook_factual_fields",
        created_at_iso="2026-11-13T00:00:00Z",
        agenda_prompt="Review NVDA",
        primary_ticker="NVDA",
        data_quality_flags=("negative_fcf_dcf_unavailable:dcf",),
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
    claim = (
        "\u0e04\u0e38\u0e13\u0e21\u0e35 100 \u0e2b\u0e38\u0e49\u0e19; "
        "Current price is 999 USD; "
        "\u0e23\u0e32\u0e04\u0e32\u0e40\u0e1b\u0e49\u0e32\u0e2b\u0e21\u0e32\u0e22\u0e15\u0e32\u0e21 DCF \u0e04\u0e37\u0e2d 200"
    )
    playbook = DistilledDecisionPlaybook(
        final_rating="HOLD",
        action_stance="HOLD",
        core_conviction_score=1,
        executive_verdict="Monitor verified evidence.",
        portfolio_context_summary=packet.portfolio_guard.summary_text,
        action_playbook=ActionPlaybookSpec(
            primary_action=ActionPlaybookOption(
                option_id="primary",
                label="Monitor evidence",
                speaker_source="hermes_moderator",
                action_steps=["Monitor the next verified filing"],
            ),
            alternatives=[
                ActionPlaybookOption(
                    option_id="alternate",
                    label="Wait for evidence",
                    speaker_source="hermes_moderator",
                    action_steps=["Review the next filing"],
                )
            ],
        ),
    )
    if target_field == "executive_verdict":
        playbook = playbook.model_copy(update={"executive_verdict": claim})
    elif target_field == "action_stance":
        playbook = playbook.model_copy(update={"action_stance": claim})
    elif target_field == "portfolio_context_summary":
        playbook = playbook.model_copy(update={"portfolio_context_summary": claim})
    elif target_field == "consensus_points[0]":
        playbook = playbook.model_copy(update={"consensus_points": [claim]})
    elif target_field.startswith("tension_axes"):
        axis = ConsensusTensionAxis(
            topic=claim if target_field.endswith("topic") else "Valuation",
            bull_side_speaker="growth_guru",
            bull_argument=claim if target_field.endswith("bull_argument") else "Evidence supports growth",
            bear_side_speaker="value_guru",
            bear_argument=claim if target_field.endswith("bear_argument") else "Price risk remains",
        )
        playbook = playbook.model_copy(update={"tension_axes": [axis]})
    elif target_field.startswith("personal_axis_alignment"):
        item = PersonalAxisCheckItem(
            dimension=claim if target_field.endswith("dimension") else "Risk",
            status="caution",
            detail=claim if target_field.endswith("detail") else "Keep risk limits",
        )
        playbook = playbook.model_copy(update={"personal_axis_alignment": [item]})
    elif target_field.startswith("action_playbook.primary_action"):
        option = playbook.action_playbook.primary_action.model_copy(update={
            "label": claim if target_field.endswith("label") else "Monitor evidence",
            "action_steps": [claim] if target_field.endswith("action_steps[0]") else ["Monitor the next verified filing"],
            "entry_zone": claim if target_field.endswith("entry_zone") else None,
            "sizing_note": claim if target_field.endswith("sizing_note") else None,
        })
        playbook = playbook.model_copy(update={
            "action_playbook": playbook.action_playbook.model_copy(update={"primary_action": option})
        })
    elif target_field.startswith("action_playbook.alternatives"):
        option = playbook.action_playbook.alternatives[0].model_copy(update={
            "label": claim if target_field.endswith("label") else "Wait for evidence",
            "action_steps": [claim] if target_field.endswith("action_steps[0]") else ["Review the next filing"],
        })
        playbook = playbook.model_copy(update={
            "action_playbook": playbook.action_playbook.model_copy(update={"alternatives": [option]})
        })
    elif target_field == "kill_switches[0]":
        playbook = playbook.model_copy(update={"kill_switches": [claim]})
    elif target_field == "reflection_summary":
        playbook = playbook.model_copy(update={"reflection_summary": claim})

    result = RoundtableConsistencyVerifier().verify_playbook(playbook, packet)
    assert not result.is_valid
    assert any(target_field in issue for issue in result.issues)


def test_playbook_factual_guards_allow_explicit_conditional_scenarios():
    packet = FrozenContextPacket(
        session_id="s_playbook_conditional",
        created_at_iso="2026-11-13T00:00:00Z",
        agenda_prompt="Review NVDA",
        primary_ticker="NVDA",
        data_quality_flags=("negative_fcf_dcf_unavailable:dcf",),
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
    playbook = DistilledDecisionPlaybook(
        final_rating="HOLD",
        action_stance="HOLD",
        core_conviction_score=1,
        executive_verdict=(
            "If \u0e04\u0e38\u0e13\u0e21\u0e35 100 \u0e2b\u0e38\u0e49\u0e19 and current price is 999 USD, "
            "DCF fair value is 200 in this hypothetical scenario."
        ),
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
    assert RoundtableConsistencyVerifier().verify_playbook(playbook, packet).is_valid


def test_gemini_voice_and_preset_catalogs_plus_override_merging():
    import pytest

    # 12 curated Gemini voices + 18 presets (3 per core seat)
    assert len(GEMINI_VOICE_CATALOG) == 12
    assert len(ALLOWED_VOICE_NAMES) == 12
    for voice in GEMINI_VOICE_CATALOG:
        assert voice.label_th.strip()
        assert voice.recommended_for.strip()
        assert voice.default_sample_text.strip()

    assert len(GURU_PRESET_CATALOG) == 18
    for seat in (
        SpeakerId.HERMES_MODERATOR,
        SpeakerId.VALUE_GURU,
        SpeakerId.QUALITY_GURU,
        SpeakerId.GROWTH_GURU,
        SpeakerId.MACRO_TACTICAL_GURU,
        SpeakerId.SKEPTIC_GURU,
    ):
        seat_presets = [p for p in GURU_PRESET_CATALOG if p.speaker_id == seat]
        assert len(seat_presets) == 3
        for preset in seat_presets:
            assert preset.recommended_voice in ALLOWED_VOICE_NAMES
            ui_ctx = get_seat_ui_context(seat.value, preset.preset_id)
            assert ui_ctx["role_lens_label"].strip()
            assert ui_ctx["focus_summary"].strip()
            assert len(ui_ctx["primary_evidence_labels"]) >= 2
            assert ui_ctx["analytical_rubric"] == get_preset_analytical_rubric(seat.value, preset.preset_id)

    distinct_lens_labels = {
        get_seat_ui_context(seat.value, "default")["role_lens_label"]
        for seat in (
            SpeakerId.HERMES_MODERATOR,
            SpeakerId.VALUE_GURU,
            SpeakerId.QUALITY_GURU,
            SpeakerId.GROWTH_GURU,
            SpeakerId.MACRO_TACTICAL_GURU,
            SpeakerId.SKEPTIC_GURU,
        )
    }
    assert len(distinct_lens_labels) == 6

    # Prompt sanitization strips reserved control blocks & truncates cleanly
    dirty = "[NO_EXTERNAL_TOOLS DIRECTIVE] ทดสอบ\x00 [MANDATORY DEVIL] ตัวอักษรเสริม"
    cleaned = sanitize_custom_persona_text(dirty, 30)
    assert "NO_EXTERNAL_TOOLS" not in cleaned
    assert "MANDATORY DEVIL" not in cleaned
    assert "\x00" not in cleaned
    assert len(cleaned) <= 30

    # Hermes remains locked ON at debate_order=0; partial session override beats tie; values clamped
    merged = merge_persona_overrides(
        GURU_PERSONA_CATALOG,
        saved_profiles=[
            {
                "speaker_id": "hermes_moderator",
                "enabled": False,
                "debate_order": 5,
                "voice_name": "Kore",
            },
            {
                "speaker_id": "value_guru",
                "short_name": "Graham",
                "pitch": 0.20,  # clamped to 0.75
                "rate": 1.90,   # clamped to 1.25
            },
        ],
        session_overrides=[
            {"speaker_id": "skeptic_guru", "debate_order": 1, "voice_name": "Enceladus"},
            {"speaker_id": "growth_guru", "enabled": False},
        ],
    )
    assert merged[0].speaker_id == SpeakerId.HERMES_MODERATOR
    assert merged[0].enabled is True
    assert merged[0].debate_order == 0

    panel = [p for p in merged if p.speaker_id != SpeakerId.HERMES_MODERATOR]
    assert panel[0].speaker_id == SpeakerId.SKEPTIC_GURU
    assert panel[0].debate_order == 1
    assert panel[0].voice_name == "Enceladus"

    val_spec = next(p for p in panel if p.speaker_id == SpeakerId.VALUE_GURU)
    assert val_spec.short_name == "Graham"
    assert val_spec.pitch == 0.75
    assert val_spec.rate == 1.25
    assert next(p for p in panel if p.speaker_id == SpeakerId.GROWTH_GURU).enabled is False

    # FrozenContextPacket snapshot serialization & legacy fallback
    packet = FrozenContextPacket(
        session_id="s_custom",
        created_at_iso="2026-11-11T00:00:00Z",
        agenda_prompt="ดู $NVDA",
        active_personas=tuple(p.to_dict() for p in merged),
    )
    roundtrip = FrozenContextPacket.from_dict(packet.to_dict())
    resolved = resolve_personas_from_packet(roundtrip)
    assert resolved[1].speaker_id == SpeakerId.SKEPTIC_GURU
    legacy_resolved = resolve_personas_from_packet(
        FrozenContextPacket(session_id="old", created_at_iso="", agenda_prompt="old")
    )
    assert legacy_resolved == GURU_PERSONA_CATALOG

    # Minimum 2 enabled panel gurus invariant
    with pytest.raises(ValueError, match="อย่างน้อย 2 คน"):
        merge_persona_overrides(
            GURU_PERSONA_CATALOG,
            None,
            [
                {"speaker_id": "value_guru", "enabled": False},
                {"speaker_id": "quality_guru", "enabled": False},
                {"speaker_id": "growth_guru", "enabled": False},
                {"speaker_id": "macro_tactical_guru", "enabled": False},
            ],
        )

    # Unsupported voice raises ValueError
    with pytest.raises(ValueError, match="Unsupported voice_name"):
        merge_persona_overrides(
            GURU_PERSONA_CATALOG,
            None,
            [{"speaker_id": "value_guru", "voice_name": "RobotX"}],
        )


def _sample_nested_sidecars():
    fresh_eval_iso = (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%dT10:00:00Z")
    googl = {
        "ticker": "GOOGL",
        "company_name": "Alphabet Inc.",
        "sector": "Communication Services",
        "industry": "Interactive Media & Services",
        "evaluated_at": fresh_eval_iso,
        "composite_score": 84.5,
        "sensitivity_matrix": [{"wacc": i, "growth": i} for i in range(40)],
        "evidence_snapshot": {"huge_blob": "x" * 12000},
        "explicit_forecast_5y": [{"year": y} for y in range(5)],
        "portfolio_context": {"position_action_hint": "TRIM_OVERWEIGHT", "held_units": 999},
        "quant_signals": {
            "raw_analysis_price": 182.40,
            "atomic_market_snapshot": {"analysis_price": 182.40, "freshness_status": "fresh", "price_source": "yfinance"},
            "composite_score": 84.5,
            "value_score": 76.0,
            "quality_score": 91.0,
            "growth_score": 82.0,
            "roic_pct": 28.4,
            "fcf_yield_pct": 3.8,
            "ocf_to_net_income": 1.32,
            "net_debt_ebitda": -0.6,
            "revenue_growth_yoy_pct": 15.2,
            "net_income_growth_yoy_pct": 22.1,
            "eps_revision_net_30d": 6,
            "pe_vs_peer_avg_pct": -8.5,
            "deterministic_scorecard": {
                "action_stance": "OVERWEIGHT",
                "core_conviction_score": 8,
                "business_conviction_score": 9,
                "investment_conviction_score": 8,
                "execution_readiness_score": 8,
                "macro_regime": "Goldilocks Expansion",
                "macro_alignment_status": "Tailwind",
            },
            "dcf_result": {
                "is_actionable": True,
                "scenarios": {"base": {"target_price": 210.0, "margin_of_safety_pct": 13.1}},
                "wacc_pct": 8.9,
                "implied_exit_ebitda_multiple": 16.5,
            },
            "reverse_dcf_result": {
                "intrinsic_value_today": 195.0,
                "target_price_12m": 214.0,
                "upside_12m_pct": 17.3,
                "mos_entry_price_today": 165.0,
                "current_margin_of_safety_pct": 6.5,
                "market_implied_growth_pct": 11.2,
                "consensus_target_price": 218.0,
                "implied_exit_ebitda_multiple": 15.8,
                "sensitivity_summary": {
                    "min_target_price": 175.0,
                    "max_target_price": 245.0,
                    "macro_yield_shock_50bps_delta": -7.5,
                    "macro_yield_shock_100bps_delta": -14.2,
                    "primary_operating_driver": "Cloud operating leverage",
                },
            },
            "piotroski_breakdown": {"f_score": 8, "roa_positive": True, "cfo_positive": True},
            "earnings_guidance_context": {
                "latest_fiscal_period": "Q3 2026",
                "revenue_growth_guidance_pct": 14.5,
                "operating_margin_trajectory": "Expanding",
                "management_tone": "Confident",
                "guidance_summary": "Cloud AI demand accelerating faster than Search moderation.",
                "key_executive_quotes": ["Cloud backlog crossed record highs this quarter."],
            },
            "tactical_setup": {"price_stage": "STAGE_2_ADVANCING", "current_rr_ratio": 2.4},
            "thesis_falsifiers": [
                {"falsifier_id": "F1", "metric_name": "Cloud Growth", "condition": "<", "threshold_value": "20%"}
            ],
        },
    }
    orcl = {
        "ticker": "ORCL",
        "company_name": "Oracle Corporation",
        "sector": "Technology",
        "industry": "Software - Infrastructure",
        "evaluated_at": fresh_eval_iso,
        "composite_score": 58.0,
        "quant_signals": {
            "raw_analysis_price": 170.0,
            "data_quality_flags": ["negative_fcf_dcf_unavailable:dcf", "share_dilution_risk:capital_allocation"],
            "roic_pct": 13.8,
            "fcf_yield_pct": -0.9,
            "net_debt_ebitda": 3.4,
            "revenue_growth_yoy_pct": 8.5,
            "deterministic_scorecard": {
                "action_stance": "REDUCE",
                "core_conviction_score": 5,
            },
            "dcf_result": {"is_actionable": False, "actionability_reason": "Negative FCF from AI Capex"},
            "reverse_dcf_result": {
                "intrinsic_value_today": 122.35,
                "target_price_12m": 136.02,
                "mos_entry_price_today": 104.0,
                "market_implied_growth_pct": 18.0,
                "alternative_valuation_model": "justified_pbv",
                "alternative_target_price_12m": 140.0,
            },
            "piotroski_breakdown": {"f_score": 5},
            "tactical_setup": {"price_stage": "STAGE_4_MARKDOWN"},
        },
    }
    return googl, orcl


def test_nested_quant_signals_strict_3state_leak_guard_and_reduce_verifier():
    googl, orcl = _sample_nested_sidecars()
    providers = _StubProviders(
        equity_map={"GOOGL": googl, "ORCL": orcl},
        reviews={
            "GOOGL": {
                "holding_summary": {"units": 999, "avg_cost_usd": 120.0},
                "decompositions": [{"horizon_label": "1Y", "total_return_pct": 25.0, "eps_growth_pct": 19.0, "multiple_expansion_pct": 6.0}],
                "triggers": [{"action_note": "Trim 25 shares if weight > 20%"}],
            }
        },
    )
    assembler = RoundtableContextAssembler(providers=providers)

    # @Portfolio NOT attached -> ZERO holding/portfolio_context/trigger leak allowed!
    pkt_googl = assembler.assemble(
        session_id="s_nested_g",
        agenda_prompt="วิเคราะห์ $GOOGL",
        context_refs=["$GOOGL"],
    )
    assert pkt_googl.instrument_anchor is not None
    assert pkt_googl.instrument_anchor.current_price == "182.40"
    assert pkt_googl.action_stance_hint == "OVERWEIGHT"
    facts_joined = " | ".join(pkt_googl.verified_numeric_facts)
    assert "fair_value_base_dcf=210.0" in facts_joined
    assert "target_price_12m=214.0" in facts_joined
    assert "piotroski_f_score=8" in facts_joined
    assert "roic_pct=28.4" in facts_joined
    assert "portfolio_position_action_hint" not in facts_joined
    assert "TRIM_OVERWEIGHT" not in pkt_googl.equity_report_text
    assert "Trim 25 shares" not in pkt_googl.equity_report_text
    assert "holding_summary" not in json.dumps(pkt_googl.equity_data)
    assert "portfolio_context" not in pkt_googl.equity_data
    # Bloat matrices stripped from compact equity_data
    assert "sensitivity_matrix" not in pkt_googl.equity_data
    assert "evidence_snapshot" not in pkt_googl.equity_data
    assert "explicit_forecast_5y" not in pkt_googl.equity_data

    # Nested REDUCE stance + non-actionable DCF + alternative model check
    pkt_orcl = assembler.assemble(
        session_id="s_nested_o",
        agenda_prompt="วิเคราะห์ $ORCL",
        context_refs=["$ORCL"],
    )
    assert pkt_orcl.action_stance_hint == "REDUCE"
    assert "negative_fcf_dcf_unavailable:dcf" in pkt_orcl.data_quality_flags
    assert "non_actionable_dcf:valuation" in pkt_orcl.data_quality_flags
    orcl_facts = " | ".join(pkt_orcl.verified_numeric_facts)
    assert "alternative_valuation_model=justified_pbv" in orcl_facts
    assert "piotroski_f_score=5" in orcl_facts

    # Verifier blocks BUY / all-in on REDUCE action stance
    verifier = RoundtableConsistencyVerifier()
    bad_pb = DistilledDecisionPlaybook(
        final_rating="BUY",
        action_stance="REDUCE",
        core_conviction_score=2,
        executive_verdict="ซื้อเต็มตัว",
        action_playbook=ActionPlaybookSpec(
            primary_action=ActionPlaybookOption(
                option_id="p1",
                label="Buy",
                speaker_source="growth_guru",
                action_steps=["เข้าซื้อ"],
            )
        ),
    )
    assert not verifier.verify_playbook(bad_pb, pkt_orcl).is_valid
    assert not verifier.verify_turn(
        RoundtableTurnPayload(
            speaker_id="growth_guru",
            round_stage="opening",
            spoken_dialogue="แนะนำซื้อเต็มพอร์ตได้เลยตอนนี้ครับราคาดีมาก",
            evidence_bullets=["[FACT] ORCL F-Score 5"],
            stance_tag="BUY",
        ),
        pkt_orcl,
    ).is_valid


def test_multi_ticker_comparative_table_and_fingerprint_tracks_all_tickers():
    googl, orcl = _sample_nested_sidecars()
    providers = _StubProviders(equity_map={"GOOGL": googl, "ORCL": orcl})
    assembler = RoundtableContextAssembler(providers=providers)

    pkt_1 = assembler.assemble(
        session_id="s_multi_1",
        agenda_prompt="เทียบ $GOOGL กับ $ORCL ตัวไหนดีกว่า",
        context_refs=["$GOOGL", "$ORCL"],
    )
    assert "COMPARATIVE SNAPSHOT TABLE" in pkt_1.equity_report_text
    assert "=== หุ้น $GOOGL" in pkt_1.equity_report_text
    assert "=== หุ้น $ORCL" in pkt_1.equity_report_text
    assert "<!-- SEC:VALUATION -->" in pkt_1.equity_report_text
    assert "Oracle Corporation" in pkt_1.equity_report_text
    assert "Negative FCF from AI Capex" in pkt_1.equity_report_text

    # Changing 2nd ticker (ORCL) must change packet_fingerprint_sha256!
    orcl_v2 = dict(orcl)
    orcl_v2["evaluated_at"] = "2026-11-11T12:00:00Z"
    providers_v2 = _StubProviders(equity_map={"GOOGL": googl, "ORCL": orcl_v2})
    pkt_2 = RoundtableContextAssembler(providers=providers_v2).assemble(
        session_id="s_multi_2",
        agenda_prompt="เทียบ $GOOGL กับ $ORCL ตัวไหนดีกว่า",
        context_refs=["$GOOGL", "$ORCL"],
    )
    assert pkt_1.packet_fingerprint_sha256 != pkt_2.packet_fingerprint_sha256


def test_curated_macro_digest_full_essence_enriched_portfolio_and_wp3_notes():
    googl, _ = _sample_nested_sidecars()
    macro_payload = {
        "overall_regime": "Late-Cycle Soft Landing",
        "evaluated_at": "2026-11-10T08:00:00Z",
        "time_horizon": "3-6M",
        "conviction_level": "HIGH",
        "conviction_rationale": "US growth cooling gradually while AI capex stays resilient",
        "regime_probabilities": {"soft_landing": 65.0, "stagflation": 20.0},
        "observable_registry": {f"OBS_{i}": {"val": i, "raw": "noise" * 400} for i in range(60)},
        "thailand_market_stance": {
            "valuation": {"pe_ratio": 14.2, "pbv_ratio": 1.35, "dividend_yield": 3.4},
            "policy_spread_bps": -175,
            "yield_curve_spread_10y_2y_bps": 28,
            "investor_flow": {"institution_net_mb": 1200, "foreign_net_mb": -950, "retail_net_mb": -250},
            "market_breadth": {"sentiment": "Selective"},
            "rationale": "Thai defensive yield plays benefit from tourism recovery.",
        },
        "sector_analysis": {"as_of_date": "2026-11-10", "summary_th": "IT และ Healthcare นำตลาด"},
        "asset_allocation": [
            {"asset_class": "US Quality Growth", "region": "US", "stance": "Overweight", "rationale": "Strong earnings breadth"},
            {"asset_class": "Thai Equities", "region": "TH", "stance": "Neutral", "rationale": "Await fiscal stimulus"},
        ],
        "risk_scenarios": [{"tail_risk": "Bond yield spike >5%", "probability": "20%", "impact": "High", "trigger_to_activate": "US 10Y > 4.75%", "hedge_instruments": ["SHY"]}],
        "pair_trades": [{"long_leg": "QQQ", "short_leg": "IWM", "time_horizon": "3M", "thesis": "Balance sheet quality outperforms"}],
    }
    digest_payload = {
        "date": "2026-11-11",
        "period": "morning",
        "generated_at": "2026-11-11T07:00:00",
        "market_mood": "Cautiously Constructive",
        "market_mood_summary": "Tech mega-caps lifted S&P while Treasury yields held steady.",
        "transmission_chain": "US CPI steady -> Lower 10Y yield -> Support Quality Growth P/E",
        "top_portfolio_focus": "Watch GOOGL Cloud margin durability vs AI capex.",
        "items": [
            {"thai_headline": "น้ำมันดิบแกว่งตัวในกรอบ", "age_hours": 2, "freshness_label": "สดใหม่", "what_happened": "OPEC+ คงกำลังผลิต"},
            {"thai_headline": "GOOGL คว้าดีล Cloud รัฐบาลสหรัฐฯ", "age_hours": 8, "freshness_label": "สดใหม่", "what_happened": "Alphabet ขยายสัญญา Cloud"},
        ],
        "youtube_highlights": [
            {"thai_title": "เจาะงบ GOOGL AI Monetization", "channel": "InvestTalk", "mentioned_tickers": ["GOOGL"], "key_takeaways": ["Search ad growth steady"]}
        ],
    }
    axis_payload = {
        "content_hash": "axis_v8",
        "complete_sections": {
            "basic_policy": "เน้นถือธุรกิจคุณภาพที่มีความได้เปรียบยั่งยืน 5-10 ปี",
            "risk_limits": {"max_annual_drawdown_pct": {"value": 20, "unit": "%"}, "max_loss_per_trade_pct": {"value": 3, "unit": "%"}},
            "invest_targets": ["US Quality Tech", "Global Healthcare"],
            "exclude_targets": ["Cryptocurrency Leverage", "Penny Stocks"],
            "primary_methods": ["Quality Compounding", "Margin of Safety DCA"],
            "secondary_methods": ["Macro Tactical Rotation"],
            "investment_horizon": "5-10 ปี",
            "rebalance_frequency": "Quarterly",
            "allocation_basis": "bucket_target",
            "allocation_rows": [{"category_name": "Core Compounder", "target_percent": 60}, {"category_name": "Cash Reserve", "target_percent": 20}],
            "role_models": ["Charlie Munger", "Howard Marks"],
            "non_actions": ["ห้ามซื้อไล่ราคาเหนือ Fair Value > 25%"],
        },
    }
    fin_ctx = {
        "horizon_years": 8,
        "target_use_amount": 5000000,
        "target_use_timeline": "2034",
        "emergency_reserves_months": 12,
        "monthly_investable_contribution": 35000,
    }
    bucket_policy = [
        {"id": "core", "name": "Core Growth", "target_pct": 60.0},
        {"id": "income", "name": "Dividend Cushion", "target_pct": 20.0},
    ]
    portfolio_payload = {
        "total_value_thb": 1000000.0,
        "cash_thb": 150000.0,
        "cash_thb_only": 100000.0,
        "cash_usd_val_thb": 50000.0,
        "cash_pct": 15.0,
        "fx_usd_thb": 35.5,
        "total_unrealized_profit_thb": 85000.0,
        "holdings": [
            {"symbol": "GOOGL", "asset_type": "US Stock", "currency": "USD", "units": 40, "avg_cost": 150.0, "market_value_thb": 250000.0, "unrealized_pnl_percent": 21.6, "bucket_id": "core"},
            {"symbol": "NVDA", "asset_type": "US Stock", "currency": "USD", "units": 50, "avg_cost": 110.0, "market_value_thb": 350000.0, "unrealized_pnl_percent": 32.0, "bucket_id": "core"},
            {"symbol": "CASH_THB", "asset_type": "Cash", "currency": "THB", "units": 100000, "market_value_thb": 100000.0},
        ],
        "look_through_summary": {
            "top_overlapped_stocks": [{"ticker": "NVDA", "total_weight_pct": 38.5, "direct_weight_pct": 35.0, "indirect_weight_pct": 3.5}],
            "crowded_sectors": [{"sector_name": "Technology", "weight_pct": 62.0}],
            "hidden_concentration_warnings": ["Total look-through weight of NVDA exceeds 15% cap"],
        },
    }
    news_map = {
        "GOOGL": {
            "items": [{"title": "Google Gemini Enterprise Rollout", "age_hours": 14, "is_stale": False, "thesis_signal": "strengthens", "why_it_matters": "Upsells Workspace ARPU"}],
            "youtube_mentions": [{"thai_title": "AI Cloud Deep Dive", "channel": "TechLens", "key_takeaways": ["TPU efficiency"]}],
        }
    }
    notes_map = {
        "GOOGL": {
            "items": [
                {"title": "GOOGL Q3 2026 Call", "matched_by": "earnings_call", "relative_path": "30_Knowledge_Base/Earnings_Calls/GOOGL/q3.md", "modified_at": "2026-11-01T00:00:00Z", "snippet": "Cloud margin expanded 300bps"}
            ]
        }
    }
    # Force guidance empty on a shallow GOOGL to also verify WP-3 EC note excerpt loading
    googl_no_guidance = json.loads(json.dumps(googl))
    googl_no_guidance["quant_signals"].pop("earnings_guidance_context", None)

    providers = _StubProviders(
        equity_map={"GOOGL": googl_no_guidance},
        portfolio=portfolio_payload,
        axis=axis_payload,
        macro=macro_payload,
        digest=digest_payload,
        news_map=news_map,
        notes_map=notes_map,
        note_contents={
            "30_Knowledge_Base/Earnings_Calls/GOOGL/q3.md": {
                "title": "GOOGL Q3 2026 Call",
                "content": "---\nticker: GOOGL\n---\nManagement reiterated 2027 AI Cloud capex discipline and backlog strength.",
            }
        },
        financial_context=fin_ctx,
        bucket_policy=bucket_policy,
    )
    assembler = RoundtableContextAssembler(providers=providers)
    packet = assembler.assemble(
        session_id="s_curated",
        agenda_prompt="ประเมิน $GOOGL เทียบแผนลงทุนและพอร์ต",
        context_refs=["$GOOGL", "@Macro", "@Digest", "@Portfolio", "@Essence"],
    )

    # @Macro curated without 150KB observable_registry noise
    assert "observable_registry" not in packet.macro_report_text
    assert "Late-Cycle Soft Landing" in packet.macro_report_text
    assert "Thailand Stance:" in packet.macro_report_text
    assert "US Quality Growth" in packet.macro_report_text
    assert len(packet.macro_report_text) < 6000

    # @Digest includes exact transmission_chain & GOOGL headline
    assert "Transmission Chain:" in packet.digest_report_text
    assert "GOOGL คว้าดีล Cloud" in packet.digest_report_text
    assert "YouTube & Analyst Takeaways:" in packet.digest_report_text

    # Full 8-section @Essence + FinancialContext + BucketPolicy
    assert "Exclude" in packet.essence_report_text and "Cryptocurrency Leverage" in packet.essence_report_text
    assert "Risk Limits" in packet.essence_report_text and "max_annual_drawdown_pct=20%" in packet.essence_report_text
    assert "Financial Context" in packet.essence_report_text and "EmergencyReserve=12" in packet.essence_report_text
    assert "Bucket Policy" in packet.essence_report_text

    # Enriched @Portfolio Overview + Concentration + Bucket Room + Offline Look-Through
    pg_text = packet.portfolio_guard.summary_text
    assert "[PORTFOLIO OVERVIEW & ALLOCATION DETAIL]" in pg_text
    assert "Concentration Risk:" in pg_text and ">15%" in pg_text
    assert "เหลือ Room" in pg_text or "สมดุล" in pg_text or "เกินเป้า" in pg_text
    assert "Top Look-Through Stocks:" in pg_text

    # WP-3 News, Vault Notes, & Earnings Call Transcript excerpt
    assert "<!-- SEC:NEWS_NOTES -->" in packet.equity_report_text
    assert "Google Gemini Enterprise Rollout" in packet.equity_report_text
    assert "Management reiterated 2027 AI Cloud capex discipline" in packet.equity_report_text

    # @Portfolio-Only session (NO $Ticker) still gets full Portfolio Overview + Essence + Macro
    pkt_port_only = assembler.assemble(
        session_id="s_port_only",
        agenda_prompt="รีวิวพอร์ตและการกระจายความเสี่ยงตอนนี้",
        context_refs=["@Portfolio", "@Macro", "@Essence"],
    )
    assert "[ABSENT: การ์ด $Equity" in pkt_port_only.equity_report_text
    assert "[PORTFOLIO OVERVIEW & ALLOCATION DETAIL]" in pkt_port_only.portfolio_guard.summary_text
    assert "Core Compounder=60%" in pkt_port_only.essence_report_text


def test_wp_a_multi_ticker_parity_scorecard_and_ticker_scoped_dcf_verifier():
    googl, orcl = _sample_nested_sidecars()
    orcl["quant_signals"]["reverse_dcf_result"]["synthetic_credit_rating"] = "BBB"
    nflx = json.loads(json.dumps(googl))
    nflx["ticker"] = "NFLX"
    nflx["company_name"] = "Netflix Inc."
    nflx["quant_signals"]["raw_analysis_price"] = 920.0
    nflx["quant_signals"]["atomic_market_snapshot"]["analysis_price"] = 920.0

    old_ts = datetime(2026, 9, 10, 10, 0, tzinfo=timezone.utc).timestamp()

    class _TickerRepo:
        def list_sessions_by_ticker(self, ticker: str, limit: int = 10):
            if ticker.upper() == "ORCL":
                return [
                    RoundtableSession(
                        session_id="prior_orcl",
                        title="ORCL Prior",
                        primary_ticker="ORCL",
                        benchmark_ticker="SPY",
                        entry_price_at_session="150.00",
                        agenda_prompt="ดู $ORCL",
                        context_refs=("$ORCL",),
                        frozen_packet_json="{}",
                        packet_fingerprint_sha256="f_orcl",
                        decision_playbook_json="{}",
                        final_rating="HOLD",
                        settlement_status="pending",
                        settled_return_pct=None,
                        settled_alpha_pct=None,
                        settled_at_date=None,
                        reflection_lesson=None,
                        status="completed",
                        created_at=old_ts,
                        updated_at=old_ts,
                    )
                ]
            return []

    providers = _StubProviders(equity_map={"GOOGL": googl, "ORCL": orcl, "NFLX": nflx})
    prices = _StubPrices(
        {
            ("latest", "ORCL"): ("2026-11-01", Decimal("170.00")),
            ("start", "SPY"): ("2026-09-10", Decimal("500.00")),
            ("latest", "SPY"): ("2026-11-01", Decimal("520.00")),
        }
    )
    assembler = RoundtableContextAssembler(
        providers=providers,
        repo=_TickerRepo(),
        settlement_prices=prices,
    )
    pkt = assembler.assemble(
        session_id="s_wpa_multi",
        agenda_prompt="เทียบ $GOOGL vs $ORCL vs $NFLX",
        context_refs=["$GOOGL", "$ORCL", "$NFLX"],
    )

    assert pkt.primary_ticker == "GOOGL"
    assert [a.ticker for a in pkt.secondary_anchors] == ["ORCL", "NFLX"]
    assert pkt.secondary_anchors[0].current_price == "170.00"
    assert "ORCL" in pkt.ticker_data_quality_flags
    assert "negative_fcf_dcf_unavailable:dcf" in pkt.ticker_data_quality_flags["ORCL"]
    assert "share_dilution_risk:capital_allocation" in pkt.ticker_data_quality_flags["ORCL"]
    assert "GOOGL" not in pkt.ticker_data_quality_flags
    assert "ORCL" in pkt.equity_data.get("secondary_bundles", {})
    assert "NFLX" in pkt.equity_data.get("secondary_bundles", {})

    facts_str = " | ".join(pkt.verified_numeric_facts)
    assert "ORCL:current_price=170.00 USD" in facts_str
    assert "ORCL:synthetic_credit_rating=BBB" in facts_str
    assert "NFLX:current_price=920.00 USD" in facts_str

    assert len(pkt.secondary_scorecards) == 1
    assert pkt.secondary_scorecards[0].primary_ticker == "ORCL"
    assert pkt.secondary_scorecards[0].settlement_status == "settled"
    assert pkt.secondary_scorecards[0].alpha_pct == "9.33"

    # Backward-compatible roundtrip preserves WP-A secondary fields
    rt = FrozenContextPacket.from_dict(pkt.to_dict())
    assert [a.ticker for a in rt.secondary_anchors] == ["ORCL", "NFLX"]
    assert rt.ticker_data_quality_flags["ORCL"] == pkt.ticker_data_quality_flags["ORCL"]
    assert len(rt.secondary_scorecards) == 1

    # Ticker-scoped DCF Verifier: allows GOOGL DCF + ORCL Reverse DCF; blocks ORCL DCF!
    verifier = RoundtableConsistencyVerifier()
    mixed_ok = RoundtableTurnPayload(
        speaker_id="value_guru",
        round_stage="opening",
        spoken_dialogue="สำหรับ $GOOGL ราคาเป้าหมายตาม DCF คือ 210 ส่วน $ORCL ติด Negative FCF จึงใช้ Reverse DCF ที่ 136 แทนครับ",
        evidence_bullets=["[FACT] GOOGL Forward DCF Base=210; ORCL Reverse DCF 136"],
        stance_tag="OVERWEIGHT",
    )
    assert verifier.verify_turn(mixed_ok, pkt).is_valid

    bad_orcl_dcf = RoundtableTurnPayload(
        speaker_id="value_guru",
        round_stage="opening",
        spoken_dialogue="สำหรับ $GOOGL ราคาเป้าหมายตาม DCF คือ 210 ส่วนราคาเป้าหมายตาม DCF ของ $ORCL คือ 150 ครับ",
        evidence_bullets=["[FACT] ORCL DCF 150"],
        stance_tag="HOLD",
    )
    res_bad = verifier.verify_turn(bad_orcl_dcf, pkt)
    assert not res_bad.is_valid
    assert any("ORCL" in iss for iss in res_bad.issues)

    bad_orcl_clause = RoundtableTurnPayload(
        speaker_id="growth_guru",
        round_stage="opening",
        spoken_dialogue="เทียบแล้ว $ORCL, ราคาเป้าหมายตาม DCF คือ 150 เลยครับ",
        evidence_bullets=["[FACT] ORCL valuation"],
        stance_tag="HOLD",
    )
    assert not verifier.verify_turn(bad_orcl_clause, pkt).is_valid

    bad_orcl_stock_phrase = RoundtableTurnPayload(
        speaker_id="growth_guru",
        round_stage="opening",
        spoken_dialogue="ราคาเป้าหมายตาม DCF ของหุ้น $ORCL คือ 150 ครับ",
        evidence_bullets=["[FACT] ORCL valuation"],
        stance_tag="HOLD",
    )
    assert not verifier.verify_turn(bad_orcl_stock_phrase, pkt).is_valid


def test_wp_b_rich_sidecar_fields_and_historical_3y_bloat_stripping():
    googl, _ = _sample_nested_sidecars()
    rich = json.loads(json.dumps(googl))
    qs = rich["quant_signals"]
    qs["atomic_market_snapshot"].update(
        {
            "market_cap_str": "1671132203125.00",
            "shares_outstanding": 12400000000,
            "beta": 1.05,
            "volatility_pct": 28.5,
            "mdd_pct": -24.3,
            "adtv_local_currency": 4250000000,
            "amihud_illiquidity_20d": 0.0012,
        }
    )
    qs["gaap_operating_margin"] = {"value_pct": 31.2, "period_end": "2026-Q3"}
    qs["non_gaap_operating_margin"] = {"value_pct": 35.8, "period_end": "2026-Q3"}
    qs["historical_gaap_operating_margin"] = {"value_pct": 29.4}
    qs["provider_ebit_margin"] = {"value_pct": 32.0}
    qs["metric_basis"] = {"ebit": "GAAP LTM", "sbc": "Excluded in Non-GAAP"}
    qs["valuation_margin_source_used"] = "normalized_mid_cycle"
    qs["sector_kpis"] = {
        "sector_name": "Communication Services",
        "sector_health_score": 84,
        "kpis": [
            {"name": "Cloud Operating Margin", "value": 17.5, "unit": "%", "benchmark": "15% peer"}
        ],
    }
    qs["reverse_dcf_result"].update(
        {
            "company_archetype": "cyclical",
            "is_mid_cycle_normalized": True,
            "normalized_mid_cycle_ebit_margin_pct": 26.5,
            "current_ebit_margin_pct": 31.2,
            "archetype_rationale": "Ad & AI capex cyclicality requires mid-cycle smoothing",
            "synthetic_credit_rating": "AAA",
            "interest_coverage_ratio": 42.5,
            "credit_spread_bps": 45,
            "benchmark_cost_of_debt_pct": 4.75,
        }
    )
    qs["historical_financials_3y"] = [
        {"fiscal_year": 2024, "revenue": 307000000000, "gross_margin_pct": 56.9, "ebit_margin_pct": 27.4, "rd_pct_of_rev": 14.8, "fcf": 69500000000, "capex_to_depreciation_ratio": 2.1, "reinvestment_lifecycle_phase": "Heavy AI Capex"},
        {"fiscal_year": 2025, "revenue": 350000000000, "gross_margin_pct": 58.2, "ebit_margin_pct": 30.1, "rd_pct_of_rev": 14.2, "fcf": 74000000000, "capex_to_depreciation_ratio": 2.4, "reinvestment_lifecycle_phase": "Heavy AI Capex"},
        {"fiscal_year": 2026, "revenue": 403000000000, "gross_margin_pct": 59.0, "ebit_margin_pct": 31.2, "rd_pct_of_rev": 13.9, "fcf": 81500000000, "capex_to_depreciation_ratio": 2.0, "reinvestment_lifecycle_phase": "Operating Leverage"},
    ]
    qs["insider_conviction"] = {
        "status": "Net Selling",
        "signal_confidence": "Medium",
        "open_market_p_count_90d": 1,
        "open_market_p_value_usd_str": "$250,000",
        "c_suite_p_count": 1,
        "open_market_s_count_90d": 3,
        "open_market_s_value_usd_str": "$4,200,000",
        "rule_10b5_1_s_count_90d": 6,
        "rule_10b5_1_s_value_usd_str": "$12,000,000",
        "amendment_unresolved": False,
    }

    assembler = RoundtableContextAssembler(providers=_StubProviders(equity_map={"GOOGL": rich}))
    pkt = assembler.assemble(session_id="s_wpb", agenda_prompt="วิเคราะห์ $GOOGL เชิงลึก", context_refs=["$GOOGL"])
    eq_txt = pkt.equity_report_text
    facts_joined = " | ".join(pkt.verified_numeric_facts)

    # Curated fields rendered cleanly
    assert "Market Cap=$1.67T" in eq_txt
    assert "Shares=12.40B" in eq_txt
    assert "Beta=1.05" in eq_txt
    assert "Vol=28.50%" in eq_txt
    assert "MDD=-24.30%" in eq_txt
    assert "ADTV=$4.25B/day" in eq_txt
    assert "Amihud Illiq=0.0012" in eq_txt
    assert "Archetype=cyclical" in eq_txt
    assert "Mid-Cycle Normalized=True" in eq_txt
    assert "Normalized EBIT=26.50%" in eq_txt
    assert "Margin Source=normalized_mid_cycle" in eq_txt
    assert "GAAP Operating=31.20% (2026-Q3)" in eq_txt
    assert "Non-GAAP Operating=35.80% (2026-Q3)" in eq_txt
    assert "Synthetic Credit Profile (Damodaran): Rating=AAA" in eq_txt
    assert "ICR=42.50x" in eq_txt
    assert "Credit Spread=45.00bps" in eq_txt
    assert "Cost of Debt=4.75%" in eq_txt
    assert "Sector KPIs (Communication Services, Health Score=84.00/100)" in eq_txt
    assert "3Y Trajectory" in eq_txt and "FY2026|$403.00B|" in eq_txt and "|2.00x|Operating Leverage" in eq_txt
    assert "Open-Market Buy=1 ($250,000, C-Suite P=1)" in eq_txt
    assert "Open-Market Sell=3 ($4,200,000)" in eq_txt
    assert "Scheduled 10b5-1 Sell=6 ($12,000,000)" in eq_txt

    assert "market_cap=$1.67T" in facts_joined
    assert "synthetic_credit_rating=AAA" in facts_joined
    assert "normalized_mid_cycle_ebit_margin_pct=26.5" in facts_joined
    assert "company_archetype=cyclical" in facts_joined

    # Bloat invariant: raw historical_financials_3y stays stripped from DB snapshot equity_data!
    assert "historical_financials_3y" not in pkt.equity_data
    assert "historical_financials_3y" not in pkt.equity_data.get("quant_signals", {})


def test_wp_c_macro_brief_puts_strategy_before_evidence_within_3600_chars():
    googl, _ = _sample_nested_sidecars()
    large_macro = {
        "overall_regime": "Disinflationary Expansion",
        "evaluated_at": "2026-11-11T09:00:00Z",
        "time_horizon": "6M",
        "conviction_level": "HIGH",
        "conviction_rationale": "AI productivity & cooling core PCE sustain positive real earnings growth.",
        "regime_probabilities": {"expansion": 62.0, "soft_landing": 28.0, "stagflation": 10.0},
        "asset_allocation": [
            {"asset_class": "US Quality Growth", "region": "US", "stance": "Overweight", "rationale": "Strong margins"},
            {"asset_class": "Global Sovereigns", "region": "DM", "stance": "Neutral", "rationale": "Duration hedge"},
        ],
        "sector_analysis": {
            "as_of_date": "2026-11-11",
            "summary_th": "เน้น Software & Cloud Infrastructure ลดน้ำหนักพลังงาน",
            "watch_conditions": ["US 10Y yield < 4.5%"],
        },
        "risk_scenarios": [
            {"tail_risk": "Tariff Inflation Shock", "probability": "20%", "impact": "High", "trigger_to_activate": "Core PCE > 3.2%", "hedge_instruments": ["GLD", "TIPS"]},
        ],
        "thailand_market_stance": {
            "valuation": {"pe_ratio": 14.0, "pbv_ratio": 1.3, "dividend_yield": 3.5},
            "policy_spread_bps": -150,
            "yield_curve_spread_10y_2y_bps": 32,
            "investor_flow": {"institution_net_mb": 900, "foreign_net_mb": -600, "retail_net_mb": -300},
            "market_breadth": {"sentiment": "Cautious"},
            "rationale": "Selectively positive on Thai tourism & healthcare.",
        },
        "regional_assessments": {
            f"Region_{i}": {"economic_state": "Expanding", "growth_score": 0.6, "inflation_score": 0.3, "monetary_score": 0.4, "confidence": 0.8}
            for i in range(6)
        },
        "focus_themes": [f"Theme {i}: Long-duration AI capex cycle details " + ("x" * 60) for i in range(4)],
        "regime_evidence": [
            {"dimension": f"Dim_{i}", "signal": "Bullish", "evidence": "Detailed macro evidence block " + ("e" * 120)}
            for i in range(5)
        ],
        "dashboard_indicators": [
            {"label": f"Indicator_{i}", "display_value": f"{i}.25%", "stale_reason": None}
            for i in range(16)
        ],
        "observable_registry": {f"OBS_{i}": "noise" * 200 for i in range(30)},
    }
    pkt = RoundtableContextAssembler(
        providers=_StubProviders(equity_map={"GOOGL": googl}, macro=large_macro)
    ).assemble(
        session_id="s_wpc",
        agenda_prompt="ประเมิน $GOOGL กับ @Macro",
        context_refs=["$GOOGL", "@Macro"],
    )
    m_txt = pkt.macro_report_text
    assert len(m_txt) > 2500
    assert m_txt.index("Asset Allocation Stance:") < m_txt.index("Regional Macro States:")
    assert m_txt.index("Sector Rotation") < m_txt.index("Regime Evidence:")
    assert m_txt.index("Macro Risk Scenarios:") < m_txt.index("Regime Evidence:")
    assert m_txt.index("Thailand Stance:") < m_txt.index("Regime Evidence:")
    # All 4 high-signal strategic sections live comfortably inside the 3600-char fundamental macro ceiling
    for marker in ("Asset Allocation Stance:", "Sector Rotation", "Macro Risk Scenarios:", "Thailand Stance:"):
        assert m_txt.index(marker) < 3000


def test_wp_d_per_equity_watchlist_privacy_and_portfolio_goals():
    googl, orcl = _sample_nested_sidecars()
    wl_list = [
        {"symbol": "GOOGL", "asset_type": "US Stock", "target_price": 165.0, "target_currency": "USD", "target_price_basis": "MoS", "suggested_bucket_id": "core", "notes": "Add on Cloud dip"},
        {"symbol": "ORCL", "asset_type": "US Stock", "target_price": 140.0, "target_currency": "USD", "target_price_basis": "RDCF", "suggested_bucket_id": "core", "notes": "Wait for FCF turn"},
        {"symbol": "AAPL", "asset_type": "US Stock", "target_price": 180.0, "target_currency": "USD", "notes": "Unrelated watchlist name"},
    ]
    goals_list = [
        {"name": "Emergency Safety Net", "goal_type": "cash_target", "target_amount_thb": 500000, "current_amount_thb": 400000, "progress_pct": 80.0, "deadline": "2027-12-31", "deadline_days_left": 410, "priority_tier": "needs", "monthly_savings_thb": 15000, "required_return_pct": 4.2, "tension_level": "low"},
        {"name": "Retirement Compounding", "goal_type": "nav_target", "target_amount_thb": 10000000, "current_amount_thb": 1250000, "progress_pct": 12.5, "deadline": "2036-12-31", "deadline_days_left": 3700, "priority_tier": "wishes", "monthly_savings_thb": 25000, "required_return_pct": 14.8, "tension_level": "high"},
    ]
    port_payload = {
        "total_value_thb": 1000000.0,
        "cash_thb": 200000.0,
        "cash_pct": 20.0,
        "holdings": [{"symbol": "GOOGL", "units": 40, "avg_cost": 150.0, "market_value_thb": 250000.0, "bucket_id": "core"}],
        "watchlist_items": wl_list,
        "goals_progress": goals_list,
    }

    # 1. WITHOUT @Portfolio -> matched $GOOGL Watchlist shows per-equity under SEC:HEADER, while AAPL/Goals/Holdings stay strictly hidden!
    pkt_no_port = RoundtableContextAssembler(
        providers=_StubProviders(equity_map={"GOOGL": googl, "ORCL": orcl}, portfolio=port_payload, watchlist=wl_list)
    ).assemble(
        session_id="s_wpd_nop",
        agenda_prompt="วิเคราะห์ $GOOGL",
        context_refs=["$GOOGL"],
    )
    assert pkt_no_port.portfolio_guard.attachment_state == ContextAttachmentState.NOT_ATTACHED_DO_NOT_ASSUME
    assert "[User Watchlist]: $GOOGL (US Stock) | เป้าซื้อ=165.00 USD | Basis=MoS | Bucket=core | Notes=Add on Cloud dip" in pkt_no_port.equity_report_text
    assert "AAPL" not in pkt_no_port.equity_report_text
    assert "ORCL" not in pkt_no_port.equity_report_text
    assert "Emergency Safety Net" not in pkt_no_port.portfolio_guard.summary_text
    assert "Emergency Safety Net" not in pkt_no_port.equity_report_text
    assert "200000" not in pkt_no_port.portfolio_guard.summary_text
    assert "Portfolio Goals Progress" not in pkt_no_port.portfolio_guard.summary_text

    # Updating matched $GOOGL watchlist target updates fingerprint even without @Portfolio
    wl_updated = [dict(wl_list[0], target_price=172.0), wl_list[1], wl_list[2]]
    pkt_no_port_v2 = RoundtableContextAssembler(
        providers=_StubProviders(equity_map={"GOOGL": googl, "ORCL": orcl}, portfolio=port_payload, watchlist=wl_updated)
    ).assemble(
        session_id="s_wpd_nop2",
        agenda_prompt="วิเคราะห์ $GOOGL",
        context_refs=["$GOOGL"],
    )
    assert pkt_no_port.packet_fingerprint_sha256 != pkt_no_port_v2.packet_fingerprint_sha256

    # 2. WITH @Portfolio + $ORCL (flat book, in watchlist) -> Goals Progress (needs first) & Watchlist & ATTACHED_FLAT_BOOK watchlist note render!
    pkt_with_port = RoundtableContextAssembler(
        providers=_StubProviders(equity_map={"GOOGL": googl, "ORCL": orcl}, portfolio=port_payload, watchlist=wl_list)
    ).assemble(
        session_id="s_wpd_p",
        agenda_prompt="ดู $ORCL กับพอร์ต",
        context_refs=["$ORCL", "@Portfolio"],
    )
    pg_txt = pkt_with_port.portfolio_guard.summary_text
    assert pkt_with_port.portfolio_guard.attachment_state == ContextAttachmentState.ATTACHED_FLAT_BOOK
    assert "กำลังติดตามใน Watchlist (เป้าซื้อ=140.00 USD | Basis=RDCF" in pg_txt
    assert "Portfolio Goals Progress" in pg_txt
    assert "Emergency Safety Net [cash_target | needs]" in pg_txt
    assert pg_txt.index("Emergency Safety Net") < pg_txt.index("Retirement Compounding")
    assert "Req CAGR: 14.80% | Tension: high" in pg_txt
    assert "Portfolio Watchlist" in pg_txt
    assert "* ORCL (US Stock): เป้าซื้อ=140.00 USD" in pg_txt

    # Mutating Goal tension updates state_hash and fingerprint!
    port_mut = json.loads(json.dumps(port_payload))
    port_mut["goals_progress"][1]["tension_level"] = "critical"
    pkt_with_port_v2 = RoundtableContextAssembler(
        providers=_StubProviders(equity_map={"GOOGL": googl, "ORCL": orcl}, portfolio=port_mut, watchlist=wl_list)
    ).assemble(
        session_id="s_wpd_p2",
        agenda_prompt="ดู $ORCL กับพอร์ต",
        context_refs=["$ORCL", "@Portfolio"],
    )
    assert pkt_with_port.portfolio_guard.state_hash != pkt_with_port_v2.portfolio_guard.state_hash
    assert pkt_with_port.packet_fingerprint_sha256 != pkt_with_port_v2.packet_fingerprint_sha256


def test_wp_e_preset_catalog_directives_and_18_rubrics():
    assert len(GURU_PRESET_CATALOG) == 18
    assert len(PRESET_ANALYTICAL_RUBRIC_MAP) == 18
    for preset in GURU_PRESET_CATALOG:
        assert len(preset.custom_directive) <= 320, f"{preset.preset_id} UI directive exceeded 320 chars"
        rubric = get_preset_analytical_rubric(preset.speaker_id.value, preset.preset_id)
        assert rubric and "Priority Metrics:" in rubric


def test_wp_s_single_stock_sidecar_recovery_macro_formatting_and_clean_notes():
    googl, _ = _sample_nested_sidecars()
    rich = json.loads(json.dumps(googl))
    qs = rich["quant_signals"]
    qs.update(
        {
            "fcf_margin_pct": 4.19,
            "fcf_cagr_3y": -32.4,
            "current_ratio": 2.79,
            "de_ratio_pct": 20.95,
            "dividend_yield_pct": 0.30,
            "payout_ratio_pct": 7.6,
            "net_buyback_yield_pct": 1.01,
            "net_shareholder_yield_pct": 1.31,
            "price_percentile_5y": 80.48,
            "price_zscore_5y": 1.15,
        }
    )
    qs["deterministic_scorecard"].update(
        {
            "fundamental_quality_score": 70.6,
            "valuation_margin_score": 90.0,
            "guidance_expectation_score": 65.0,
            "setup_readiness_score": 90.0,
            "coverage_pct": 100.0,
            "verified_coverage_pct": 75.0,
        }
    )
    qs["tactical_setup"].update(
        {
            "key_support_level": 279.75,
            "key_resistance_level": 345.86,
            "breakout_trigger_price": 346.93,
            "breakout_target_price": 370.48,
            "breakout_stop_loss": 335.15,
            "breakout_planned_rr": 2.0,
            "breakout_volume_ratio": 1.16,
            "breakout_entry_status": "pre_trigger",
        }
    )
    # Real sidecars store historical_financials_3y & reported_ebit_margin_pct inside reverse_dcf_result!
    qs["reverse_dcf_result"].update(
        {
            "reported_ebit_margin_pct": 33.32,
            "ebit_margin_fiscal_period": "TTM (GAAP 10-Q)",
            "historical_financials_3y": [
                {"fiscal_year": "2023A", "revenue": 307394000000, "gross_margin_pct": 56.63, "ebit_margin_pct": 27.42, "rd_pct_of_rev": 14.78, "fcf": 69495000000, "capex_to_depreciation_ratio": 2.70, "reinvestment_lifecycle_phase": "aggressive_expansion"},
                {"fiscal_year": "2024A", "revenue": 350018000000, "gross_margin_pct": 58.20, "ebit_margin_pct": 32.11, "rd_pct_of_rev": 14.09, "fcf": 72764000000, "capex_to_depreciation_ratio": 3.43, "reinvestment_lifecycle_phase": "aggressive_expansion"},
                {"fiscal_year": "2025A", "revenue": 402836000000, "gross_margin_pct": 59.65, "ebit_margin_pct": 32.03, "rd_pct_of_rev": 15.17, "fcf": 73266000000, "capex_to_depreciation_ratio": 4.32, "reinvestment_lifecycle_phase": "aggressive_expansion"},
            ],
        }
    )
    long_narrative = ("Alphabet แสดงความได้เปรียบด้าน AI Search และ Cloud infrastructure อย่างแข็งแกร่ง " * 16).strip() + " FINAL_TRUNCATED_TAIL_CATALYST"
    rich["narrative_analysis"] = long_narrative

    # 13 holdings render with full individual detail (expanded up to Top-25 for single-stock/portfolio completeness)
    holdings = [
        {"symbol": "GOOGL", "asset_type": "US Stock", "currency": "USD", "units": 10, "avg_cost": 180.0, "market_value_thb": 150000.0, "bucket_id": "core"},
        *[
            {"symbol": f"STK{i}", "asset_type": "US Stock", "currency": "USD", "units": 5, "avg_cost": 100.0, "market_value_thb": 80000.0 - i * 3000, "bucket_id": "core"}
            for i in range(1, 13)
        ],
    ]
    total_nav = sum(h["market_value_thb"] for h in holdings)
    pos_review_payload = {
        "decompositions": [
            {
                "horizon_label": "1Y",
                "total_return_pct": float("nan"),
                "eps_growth_pct": 138.18,
                "pe_expansion_pct": -14.2,
                "primary_driver": "earnings_growth",
                "diagnosis_tier": "TWIN_DETERIORATION",
            }
        ],
        "then_vs_now": {
            "then_pe": 36.0,
            "now_pe": 15.3,
            "then_growth_pct": 15.0,
            "now_growth_pct": 24.7,
            "thesis_reflection_question": "Valuation has compressed alongside strong financial earnings delivery — is moat intact?",
        },
        "triggers": [
            {"trigger_id": "EVT_EARN", "title": "Upcoming Earnings", "condition_text": "Meets targets", "case_impact": "SETTLES_BUY_VS_SELL", "is_currently_triggered": False},
            {"trigger_id": "FLOOR_1", "title": "Price Floor", "condition_text": "price < 42.37", "case_impact": "BREAKS_BUY_CASE", "is_currently_triggered": False},
            {"trigger_id": "FLOOR_2", "title": "Rev Floor", "condition_text": "rev < 10.6", "case_impact": "BREAKS_BUY_CASE", "is_currently_triggered": False},
            {"trigger_id": "ANTI_AVERAGING_DOWN_RULE", "title": "Anti-Falling-Knife Rule", "condition_text": "Down -10.1% in Stage 4", "case_impact": "BLOCKS_DIP_BUYING", "is_currently_triggered": True},
        ],
    }
    macro_payload = {
        "overall_regime": "Reflation",
        "evaluated_at": "2026-11-12T08:00:00Z",
        "time_horizon": "3-6M",
        "conviction_level": "MEDIUM",
        "conviction_rationale": "Resilient labor market vs sticky CPI.",
        "regime_probabilities": {"Reflation": 55.0, "Stagflation": 30.0},
        "asset_allocation": [{"asset_class": "Equities", "region": "Global", "stance": "Neutral", "rationale": "Selective quality"}],
        "sector_analysis": {
            "as_of_date": "2026-11-06",
            "summary_th": "Defense & Value นำตลาด",
            "watch_conditions": [
                {"metric_ref": "XLK.relative_momentum", "operator": "crosses_above", "future_threshold": 100.0, "unit": "index", "reason": "Re-acceleration signal"}
            ],
        },
        "risk_scenarios": [{"tail_risk": "Energy Shock", "probability": "30%", "impact": "High", "trigger_to_activate": "WTI > 95", "hedge_instruments": ["XLE"]}],
        "thailand_market_stance": {
            "valuation": {"pe_ratio": 14.5, "pbv_ratio": 1.25, "dividend_yield": 3.8},
            "policy_spread_bps": 287.5,
            "yield_curve_spread_10y_2y_bps": 45.0,
            "investor_flow": {"institution_net_mb": 1200, "foreign_net_mb": -400, "retail_net_mb": -800},
            "market_breadth": {"sentiment": "bullish"},
            "physical_gold": {"bar_sell_thb": 64800.0, "unit": "THB/Baht-weight"},
        },
        "key_assumptions": ["US labor resilient", "Thai SET valuation recovery at 14.5x PE"],
        "regime_evidence": [{"dimension": "Inflation", "signal": "Hot", "evidence": "Core PCE sticky"}],
    }
    ea_path = "30_Knowledge_Base/Stocks/GOOGL/Analysis/GOOGL Equity Analysis 2026-10-30.md"
    notes_map = {
        "GOOGL": {
            "items": [
                {
                    "title": "GOOGL Equity Analysis 2026-10-30",
                    "folder": "30_Knowledge_Base/Stocks/GOOGL/Analysis",
                    "relative_path": ea_path,
                    "modified_at": "2026-10-30T00:00:00Z",
                    "matched_by": "equity_analysis",
                    "snippet": "--- schema_version: 2 title: 'GOOGL' entity_type: equity_analysis ticker: GOOGL --- Clean insight begins here.",
                }
            ]
        }
    }
    note_contents = {
        ea_path: {
            "title": "GOOGL Equity Analysis 2026-10-30",
            "content": (
                "---\nticker: GOOGL\n---\n# Title\n"
                "### 💼 Portfolio Position & Sizing Context\n- Holding Status: SECRET_HOLDING_99_UNITS\n\n"
                "## 📝 บทวิเคราะห์ (Narrative)\n" + long_narrative + "\n\n"
                "## 📰 Sentiment & Narrative Context\n"
                "**ธีมสำคัญ:** AI infrastructure demand accelerates across Enterprise.\n"
                "> ⚠️ EA_SENTIMENT_QUOTE_MARKER\n\n## Related\n- GOOGL\n"
            ),
        }
    }

    providers = _StubProviders(
        equity_map={"GOOGL": rich},
        reviews={"GOOGL": pos_review_payload},
        macro=macro_payload,
        notes_map=notes_map,
        note_contents=note_contents,
        portfolio={"total_value_thb": total_nav, "cash_thb": 0.0, "cash_pct": 0.0, "holdings": holdings},
    )

    # 1. With @Portfolio attached
    pkt = RoundtableContextAssembler(providers=providers).assemble(
        session_id="s_wps_full",
        agenda_prompt="วิเคราะห์ $GOOGL แบบเจาะลึก",
        context_refs=["$GOOGL", "@Portfolio", "@Macro"],
    )
    eq = pkt.equity_report_text
    assert "Pillars (0-100): FundQual=70.60 | ValMargin=90.00 | Guidance=65.00 | Setup=90.00 | Coverage=100.00% (Verified=75.00%)" in eq
    assert "Current=33.32% (TTM (GAAP 10-Q))" in eq
    assert "Support=279.75, Resistance=345.86" in eq
    assert "planned R:R=2.00x, VolRatio=1.16x" in eq
    assert "3Y Trajectory" in eq and "FY2025A|$402.84B|" in eq and "|4.32x|" in eq
    assert "Total Return=N/A (EPS Growth=138.18% vs Multiple Expansion=-14.20%) | Driver=earnings_growth | Tier=TWIN_DETERIORATION" in eq
    assert "Assessment: Valuation has compressed alongside strong financial earnings delivery" in eq
    assert "- Portfolio Decision Triggers: [ACTIVE] Anti-Falling-Knife Rule (BLOCKS_DIP_BUYING): Down -10.1% in Stage 4;" in eq
    assert "{'trigger_id':" not in eq
    assert "schema_version:" not in eq and "Clean insight begins here." in eq
    assert "FINAL_TRUNCATED_TAIL_CATALYST" in eq
    assert "สาระสำคัญเพิ่มเติมจาก Vault Analysis" in eq
    assert "AI infrastructure demand accelerates across Enterprise." in eq
    assert "EA_SENTIMENT_QUOTE_MARKER" in eq
    for idx in range(1, 13):
        assert f"  STK{idx}|" in pkt.portfolio_guard.summary_text

    # Macro formatting + top strategy offset invariant (<3000)
    m_txt = pkt.macro_report_text
    assert "XLK.relative_momentum crosses_above 100.0 (index): Re-acceleration signal" in m_txt
    assert "GoldBar=64800.00 THB/Baht-weight" in m_txt
    assert "- Key Macro Assumptions: US labor resilient | Thai SET valuation recovery at 14.5x PE" in m_txt
    for marker in ("Asset Allocation Stance:", "Sector Rotation", "Macro Risk Scenarios:", "Thailand Stance:"):
        assert m_txt.index(marker) < 3000

    # New verified_numeric_facts + bloat invariant
    facts_joined = " | ".join(pkt.verified_numeric_facts)
    for k in (
        "reported_ebit_margin_pct=33.32",
        "fcf_margin_pct=4.19",
        "fcf_cagr_3y=-32.4",
        "current_ratio=2.79",
        "de_ratio_pct=20.95",
        "dividend_yield_pct=0.3",
        "net_shareholder_yield_pct=1.31",
        "price_percentile_5y=80.48",
    ):
        assert k in facts_joined
    assert "historical_financials_3y" not in pkt.equity_data.get("quant_signals", {}).get("reverse_dcf_result", {})

    # 2. Strict 3-State Guard WITHOUT @Portfolio: secrets inside both triggers & Vault markdown never leak!
    pkt_nop = RoundtableContextAssembler(providers=providers).assemble(
        session_id="s_wps_noport",
        agenda_prompt="วิเคราะห์ $GOOGL แบบไม่มีพอร์ต",
        context_refs=["$GOOGL"],
    )
    assert "SECRET_HOLDING_99_UNITS" not in pkt_nop.equity_report_text
    assert "Anti-Falling-Knife Rule" not in pkt_nop.equity_report_text


def test_wp_b_full_single_stock_vault_180d_guidance_wacc_ohlcv_unclipping_macro_24_and_essence():
    googl, _ = _sample_nested_sidecars()
    ftnt = json.loads(json.dumps(googl))
    ftnt["ticker"] = "FTNT"
    ftnt["company_name"] = "Fortinet Inc."
    qs = ftnt["quant_signals"]
    qs.pop("earnings_guidance_context", None)
    qs["atomic_market_snapshot"].update(
        {
            "latest_ohlcv_date": "2026-10-29",
            "price_sync_status": "quote_ohlcv_mismatch",
        }
    )
    qs["reverse_dcf_result"].update(
        {
            "net_cash_debt": 75600000000,
            "terminal_growth_pct": 2.5,
            "implied_perpetual_growth_from_exit_pct": 4.8,
            "risk_free_rate_pct": 4.25,
            "erp_pct": 4.4,
            "cost_of_equity_pct": 8.65,
        }
    )
    qs["metric_basis"] = {
        "mb1": "v1",
        "mb2": "v2",
        "mb3": "v3",
        "mb4": "v4",
        "mb5_extra": "GAAP LTM Verified",
        "mb6_extra": "ROIC Ex-Goodwill",
    }
    qs["sector_kpis"] = {
        "sector_name": "Cybersecurity",
        "sector_health_score": 88,
        "kpis": [
            {"name": f"KPI_{i}", "value": 10 + i, "unit": "%", "benchmark": f"B_{i}"}
            for i in range(1, 6)
        ],
    }
    ftnt["sentiment_context"] = {
        "market_sentiment": "bullish",
        "key_themes": ["SASE", "OT Security"],
        "tail_risks": [f"TailRisk_{i}: Deep structural tail risk detail" for i in range(1, 6)],
    }
    qs["thesis_falsifiers"] = [
        {"falsifier_id": f"F{i}", "metric_name": f"Metric_{i}", "condition": "<", "threshold_value": i * 10, "narrative_explanation": f"Falsifier detail #{i}"}
        for i in range(1, 6)
    ]
    long_narrative = ("Fortinet สร้างความได้เปรียบจากชิป FortiASIC และ FortiOS แบบครบวงจร " * 17).strip()
    ftnt["narrative_analysis"] = long_narrative
    # Append ~560-char continuation beyond narrative_analysis inside Vault EA note so single-stock tail cap (850) captures all of it (vs old 500)
    ea_full_narr = long_narrative + " " + ("CONTINUATION_BLOCK " * 28) + "END_OF_850_TAIL_MARKER"

    pos_review = {
        "triggers": [
            {"trigger_id": f"TRIG_{i}", "title": f"Rule {i}", "condition_text": f"Condition {i}", "case_impact": "WATCH", "is_currently_triggered": (i == 5)}
            for i in range(1, 6)
        ],
        "honest_gaps": [f"Gap_{i}: Missing disclosure #{i}" for i in range(1, 5)],
    }

    ec_path = "30_Knowledge_Base/Stocks/FTNT/Earnings/Q2_2026_FTNT_Earnings_Call.md"
    ea_path = "30_Knowledge_Base/Stocks/FTNT/Analysis/FTNT Equity Analysis 2026-09-26.md"
    notes_map = {
        "FTNT": {
            "items": [
                *[
                    {"title": f"News Note {i}", "matched_by": "news", "relative_path": f"30_Knowledge_Base/Stocks/FTNT/News/n{i}.md", "modified_at": f"2026-11-0{7-i}T00:00:00Z", "snippet": f"News {i} content"}
                    for i in range(1, 7)
                ],
                {"title": "Q2_2026_FTNT_Earnings_Call", "matched_by": "earnings_call", "folder": "30_Knowledge_Base/Stocks/FTNT/Earnings", "relative_path": ec_path, "modified_at": "2026-08-26T00:00:00Z", "snippet": "Q2 beat"},
                {"title": "FTNT_latest", "matched_by": "equity_analysis", "folder": "30_Knowledge_Base/Stocks/FTNT/Analysis", "relative_path": "30_Knowledge_Base/Stocks/FTNT/Analysis/FTNT_latest.md", "modified_at": "2026-09-26T00:00:00Z", "snippet": "Alias note"},
                {"title": "FTNT Equity Analysis 2026-09-26", "matched_by": "equity_analysis", "folder": "30_Knowledge_Base/Stocks/FTNT/Analysis", "relative_path": ea_path, "modified_at": "2026-09-26T00:00:00Z", "snippet": "Main analysis"},
            ]
        }
    }
    ec_markdown = (
        "---\n"
        "date: '2026-08-26'\n"
        "entity_type: earnings_call\n"
        "period: Q2 2026\n"
        "guidance:\n"
        "  billings_guidance_range_usd: [9350000000, 9550000000]\n"
        "  revenue_guidance_range_usd: [8020000000, 8180000000]\n"
        "  revenue_growth_guidance_pct: 19.0\n"
        "  operating_margin_trajectory: expanding\n"
        "  management_tone: bullish\n"
        "  non_gaap_operating_margin_range_pct: [35.0, 37.0]\n"
        "  non_gaap_eps_range: [3.41, 3.47]\n"
        "  key_executive_quotes:\n"
        "    - 'Billings growth accelerated to 25% midpoint driven by SASE Firewall.'\n"
        "---\n\n"
        "## 🤖 AI Highlights\n"
        "### 1. 📊 Key Financial Highlights\n"
        "* **Billings:** 2.37B USD (+33% YoY)\n"
        "### 3. 🎙️ Analyst Q&A Key Takeaways\n"
        "* **OT Security:** เติบโตกว่า 55% ANALYST_QA_UNIQUE_MARKER\n\n"
        "## 📄 Full Transcript\n"
        "OPERATOR_TRANSCRIPT_BOILERPLATE_SHOULD_NOT_APPEAR\n"
    )
    ea_markdown = f"---\nticker: FTNT\n---\n## 📝 บทวิเคราะห์ (Narrative)\n{ea_full_narr}\n"

    macro_24 = {
        "overall_regime": "Disinflationary Expansion",
        "evaluated_at": "2026-11-12T08:00:00Z",
        "time_horizon": "6M",
        "conviction_level": "HIGH",
        "conviction_rationale": "AI productivity sustains real earnings.",
        "quant_narrative_alignment": "aligned",
        "divergence_note": "ข้อมูล Quant และ Narrative สอดคล้องกันในมิติของเงินเฟ้อ",
        "regime_probabilities": {"expansion": 65.0, "soft_landing": 25.0},
        "asset_allocation": [{"asset_class": "Equities", "region": "US", "stance": "Overweight", "rationale": "Strong margins"}],
        "sector_analysis": {"as_of_date": "2026-11-11", "summary_th": "เน้น Tech & Security"},
        "risk_scenarios": [{"tail_risk": "Inflation Re-acceleration", "probability": "20%", "impact": "High", "trigger_to_activate": "CPI > 3.5%", "hedge_instruments": ["TIPS"]}],
        "thailand_market_stance": {
            "valuation": {"pe_ratio": 14.0, "pbv_ratio": 1.3, "dividend_yield": 3.5},
            "policy_spread_bps": -150,
            "yield_curve_spread_10y_2y_bps": 32,
            "investor_flow": {"institution_net_mb": 900, "foreign_net_mb": -600, "retail_net_mb": -300},
            "market_breadth": {"sentiment": "Cautious"},
        },
        "dashboard_indicators": [
            {"label": f"MacroInd_{i}", "display_value": f"{i}.5%", "stale_reason": None}
            for i in range(24)
        ],
    }
    fin_ctx = {
        "horizon_years": 10,
        "target_use_amount": 5000000,
        "target_use_range": "4M-6M THB",
        "target_use_timeline": "2036",
        "emergency_reserves_months": 6,
        "reserves_outside_portfolio": 300000,
        "obligations_monthly": 10000,
        "obligations_description": "ไม่มีภาระผูกพันกระทบพอร์ตนี้",
        "essential_monthly_expense": 35000,
        "withdrawal_frequency": "ไม่มีการถอน (เน้นสะสมมูลค่าระยะยาว)",
        "withdrawal_amount": 0,
        "experience_description": "หุ้น กองทุนรวม ทองคำ",
        "monthly_investable_contribution": 25000,
    }

    recorded_days = []

    class _TrackingProviders(_StubProviders):
        def list_equity_notes(self, ticker: str, days: int = 30):
            recorded_days.append((ticker, days))
            return super().list_equity_notes(ticker, days)

    providers = _TrackingProviders(
        equity_map={"FTNT": ftnt},
        reviews={"FTNT": pos_review},
        notes_map=notes_map,
        note_contents={
            ec_path: {"title": "Q2_2026_FTNT_Earnings_Call", "content": ec_markdown},
            ea_path: {"title": "FTNT Equity Analysis 2026-09-26", "content": ea_markdown},
        },
        macro=macro_24,
        financial_context=fin_ctx,
        portfolio={"total_value_thb": 1000000.0, "cash_thb": 200000.0, "cash_pct": 20.0, "holdings": []},
    )

    pkt = RoundtableContextAssembler(providers=providers).assemble(
        session_id="s_wpb_full",
        agenda_prompt="วิเคราะห์ $FTNT กับ @Macro @Portfolio @Essence",
        context_refs=["$FTNT", "@Macro", "@Portfolio", "@Essence"],
    )

    # 1. Vault lookup window = 180 days & prioritized EC/EA notes despite 6 news notes ahead
    assert ("FTNT", 180) in recorded_days
    eq = pkt.equity_report_text
    assert "OHLCV: 2026-10-29 | Sync: quote_ohlcv_mismatch" in eq
    assert "Capital Structure & WACC Assumptions: Net Cash/Debt=$75.60B | Terminal Gr=2.50% | Implied Perpetual Gr=4.80% | WACC Inputs(Rf=4.25%, ERP=4.40%, Ke=8.65%)" in eq
    assert "mb5_extra=GAAP LTM Verified" in eq and "mb6_extra=ROIC Ex-Goodwill" in eq
    assert "KPI_4=14.00%" in eq and "KPI_5=15.00%" in eq
    assert "TailRisk_5:" in eq
    assert "[F5] Metric_5" in eq
    assert "TRIG_4" in eq or "Rule 4" in eq
    assert "Gap_4: Missing disclosure #4" in eq

    # 2. Vault Earnings Call YAML guidance fallback into SEC:GROWTH_GUIDANCE + cleaned AI Highlights (without Full Transcript)
    assert "Earnings Guidance (Vault Q2 2026, date=2026-08-26, tone=bullish): Rev Guidance=19.00% | Operating Margin Trajectory=expanding" in eq
    assert "Guidance Targets: Rev=$8.02B–$8.18B | Billings=$9.35B–$9.55B | Non-GAAP OpMargin=35.00%–37.00% | Non-GAAP EPS=$3.41–$3.47" in eq
    assert "Billings growth accelerated to 25% midpoint" in eq
    assert "Q2_2026_FTNT_Earnings_Call" in eq and "FTNT Equity Analysis 2026-09-26" in eq
    assert "FTNT_latest" not in eq
    assert "ANALYST_QA_UNIQUE_MARKER" in eq
    assert "OPERATOR_TRANSCRIPT_BOILERPLATE_SHOULD_NOT_APPEAR" not in eq
    assert "END_OF_850_TAIL_MARKER" in eq

    # 3. Verified numeric facts include Capital Structure & WACC
    facts_joined = " | ".join(pkt.verified_numeric_facts)
    for fk in (
        "net_cash_debt=$75.60B",
        "terminal_growth_pct=2.5",
        "implied_perpetual_growth_from_exit_pct=4.8",
        "cost_of_equity_pct=8.65",
        "risk_free_rate_pct=4.25",
        "erp_pct=4.4",
    ):
        assert fk in facts_joined

    # 4. @Macro renders all 24 indicators + Quant-Narrative Alignment while keeping strategy markers < 3000
    m_txt = pkt.macro_report_text
    assert "Quant-Narrative Alignment: Alignment=aligned | ข้อมูล Quant และ Narrative สอดคล้องกันในมิติของเงินเฟ้อ" in m_txt
    assert "MacroInd_0=0.5%" in m_txt and "MacroInd_23=23.5%" in m_txt
    for marker in ("Asset Allocation Stance:", "Sector Rotation", "Macro Risk Scenarios:", "Thailand Stance:"):
        assert m_txt.index(marker) < 3000

    # 5. @Essence renders full FinancialContext fields
    e_txt = pkt.essence_report_text
    for ef in (
        "range=4M-6M THB",
        "OutsideReserves=300000 THB",
        "ObligationsNote=ไม่มีภาระผูกพันกระทบพอร์ตนี้",
        "EssentialExpense/mo=35000 THB",
        "Withdrawal=ไม่มีการถอน (เน้นสะสมมูลค่าระยะยาว) (0 THB)",
        "Experience=หุ้น กองทุนรวม ทองคำ",
    ):
        assert ef in e_txt


def test_wp_s100_complete_single_stock_zero_clipping_all_fields_and_tvn_privacy():
    googl, orcl = _sample_nested_sidecars()
    rich = json.loads(json.dumps(googl))
    qs = rich["quant_signals"]
    qs["atomic_market_snapshot"].update(
        {
            "volume_confirmation": "confirmed_accumulation",
            "missing_trading_sessions": 2,
        }
    )
    qs["deterministic_scorecard"].pop("action_stance_reason", None)
    qs["deterministic_scorecard"].update(
        {
            "analyst_expectations_score": 78.5,
            "management_guidance_score": 84.0,
            "execution_score_breakdown": {"stage": 90.0, "setup": 82.0, "insider": 60.0, "setup_reason": "Healthy pullback"},
        }
    )
    qs["upside_pct"] = 19.5
    qs["downside_pct"] = -8.2
    qs["fcf_quality_score"] = 88.0
    qs["debt_quality_score"] = 95.0
    qs["solvency_score"] = 97.5
    qs["peer_relative_score"] = 76.0
    qs["dcf_result"]["scenarios"].setdefault("bear", {"target_price": 170.0})["margin_of_safety_pct"] = -4.5
    qs["dcf_result"]["scenarios"].setdefault("bull", {"target_price": 245.0})["margin_of_safety_pct"] = 32.0
    qs["dcf_result"]["valuation_sanity_audit"] = {
        "model_reliability_tier": "HIGH",
        "terminal_value_pct_of_ev": 62.5,
        "is_terminal_dependent": False,
        "implied_exit_fcf_multiple": 22.4,
        "benchmark_sector_multiple": 20.0,
        "sanity_warnings": ["None"],
    }
    long_wacc_adj = "Adjusted +35bps for sovereign spread"
    long_arch_rat = "Durable digital ad & Cloud infrastructure flywheel with multi-year TPU efficiency gains " * 4
    qs["reverse_dcf_result"].update(
        {
            "raw_wacc_pct": 8.40,
            "effective_wacc_pct": 8.75,
            "wacc_adjustment_reason": long_wacc_adj,
            "enterprise_value": 2150000000000,
            "equity_value": 2240000000000,
            "sum_pv_5y_fcf": 410000000000,
            "terminal_value_pv": 1740000000000,
            "base_revenue": 402800000000,
            "intrinsic_value_exit_multiple": 232.10,
            "company_archetype": "compounder",
            "archetype_rationale": long_arch_rat.strip(),
            "valuation_sanity_audit": {
                "model_reliability_tier": "HIGH",
                "reliability_score": 88,
                "terminal_value_pct_of_ev": 64.0,
                "is_terminal_dependent": False,
                "implied_exit_fcf_multiple": 21.8,
                "benchmark_sector_multiple": 20.0,
                "sanity_warnings": [],
            },
        }
    )
    qs["insider_conviction"] = {
        "status": "Mixed",
        "signal_confidence": "Medium",
        "open_market_p_count_90d": 0,
        "open_market_s_count_90d": 1,
        "open_market_s_value_usd_str": "$500,000",
        "rule_10b5_1_s_count_90d": 4,
        "rule_10b5_1_s_value_usd_str": "$8,000,000",
        "unflagged_s_count_90d": 2,
        "exercise_count_90d": 3,
        "tax_withholding_count_90d": 5,
        "amendment_unresolved": False,
    }
    long_2400_narr = ("Alphabet 2400-char narrative insight block. " * 55).strip() + " NARRATIVE_2400_UNCLIPPED_END"
    rich["narrative_analysis"] = long_2400_narr
    rich["thesis_delta"] = {
        "previous_analysis_date": "2026-10-01",
        "price_change_pct": 6.2,
        "conviction_score_delta": 1.0,
        "target_price_change_pct": 8.5,
        "previous_action_stance": "HOLD",
        "current_action_stance": "OVERWEIGHT",
        "f_score_delta": 1,
        "breached_falsifiers": [],
    }
    active_thesis = {
        "status": "ACTIVE",
        "evidence_tier": "HIGH_CONVICTION",
        "claim_statement": "Search + Cloud AI monetization sustains 15%+ compounding through 2028 " * 4,
        "falsifiers": [
            {"falsifier_id": "AF1", "metric_name": "Cloud Growth YoY", "operator": "<", "threshold_value": 18, "unit": "%", "invalidation_impact": "DOWNGRADE", "narrative_explanation": "Cloud deceleration below 18%"},
        ],
        "linked_catalysts": [
            {"catalyst_id": "CAT1", "event_name": "Q4 Gemini Enterprise Expansion", "expected_date_str": "2027-02-04", "status": "PENDING"},
        ],
    }
    long_stance_reason = "Strong ROIC and backlog visibility justify Overweight despite AI capex cycle " * 4
    pos_review = {
        "action_stance_reason": long_stance_reason.strip(),
        "then_vs_now": {
            "period_label": "Around Entry (2026-06-05)",
            "is_held": True,
            "then_date": "2026-06-05",
            "then_price": 157.89,
            "now_price": 182.4,
            "then_pe": 25.0,
            "now_pe": 22.0,
            "then_growth_pct": 14.0,
            "now_growth_pct": 18.0,
            "what_was_true_then": "Bought at compressed multiple during antitrust headlines.",
            "what_is_true_now": "Cloud backlog accelerated and operating margins expanded.",
            "thesis_reflection_question": "Is Alphabet worth holding at its current size given capex trajectory?",
        },
        "triggers": [{"trigger_id": "T1", "title": "Entry Trim", "condition_text": "Weight > 20%", "case_impact": "TRIM", "is_currently_triggered": False}],
    }

    ec_path = "30_Knowledge_Base/Stocks/GOOGL/Earnings/Q3_2026_GOOGL_Earnings_Call.md"
    ea_path = "30_Knowledge_Base/Stocks/GOOGL/Analysis/GOOGL Equity Analysis 2026-11-10.md"
    long_ec_body = "## 🤖 AI Highlights\n* Cloud backlog surged 35% YoY. " + ("Management emphasized disciplined capex returns. " * 65) + "EC_3200_END_MARKER"
    assert len(long_ec_body) > 2900
    long_ea_body = (
        "---\nticker: GOOGL\n---\n## 📝 บทวิเคราะห์ (Narrative)\n"
        + long_2400_narr
        + " Continuation beyond SQLite narrative with extra qualitative insight EA_OVER_3000_CONTINUATION_TAIL\n\n"
        + "## 📰 Sentiment & Narrative Context\n**ธีมสำคัญ:** Sovereign Cloud & TPU demand broaden moat.\n"
        + "> ⚠️ Single-source verification caveat: verify YoY capex guidance directly in 10-Q.\n"
    )
    notes_map = {
        "GOOGL": {
            "items": [
                {"title": "Q3_2026_GOOGL_Earnings_Call", "matched_by": "earnings_call", "relative_path": ec_path, "modified_at": "2026-11-01T00:00:00Z", "snippet": "Cloud up"},
                {"title": "GOOGL Equity Analysis 2026-11-10", "matched_by": "equity_analysis", "relative_path": ea_path, "modified_at": "2026-11-10T00:00:00Z", "snippet": "Core memo"},
            ]
        }
    }
    news_map = {
        "GOOGL": {
            "items": [{"title": f"GOOGL News Headline #{i}", "age_hours": i, "is_stale": False, "why_it_matters": f"Impact detail {i}"} for i in range(12)],
            "youtube_mentions": [{"thai_title": f"YT Video #{i}", "channel": "Ch", "key_takeaways": ["k1", "k2", "k3"]} for i in range(7)],
        }
    }
    holdings_28 = [
        {"symbol": "GOOGL", "asset_type": "US Stock", "currency": "USD", "units": 40, "avg_cost": 157.89, "market_value_thb": 250000.0, "unrealized_pnl_percent": 15.0, "bucket_id": "core"},
        *[
            {"symbol": f"HLD{i}", "asset_type": "US Stock", "currency": "USD", "units": 2, "avg_cost": 100.0, "market_value_thb": 40000.0 - i * 500, "unrealized_pnl_percent": 5.0, "bucket_id": "core"}
            for i in range(1, 28)
        ],
    ]
    port_full = {
        "total_value_thb": sum(h["market_value_thb"] for h in holdings_28),
        "total_cost_basis_thb": 980000.0,
        "cash_thb": 50000.0,
        "cash_pct": 4.0,
        "holdings": holdings_28,
        "look_through_summary": {
            "top_overlapped_stocks": [
                {"ticker": f"OV{i}", "total_weight_pct": 10.0 - i * 0.5, "direct_weight_pct": 5.0, "indirect_weight_pct": 5.0 - i * 0.5, "is_cap_breached": (i == 0), "is_hidden_concentration": (i == 0)}
                for i in range(10)
            ]
        },
        "goals_progress": [
            {"name": "Flex Goal", "goal_type": "nav_target", "priority_tier": "wants", "target_amount_thb": 3000000, "current_amount_thb": 1200000, "progress_pct": 40.0, "deadline": "2035-01-01", "flexibility": "adjustable", "tension_level": "medium"}
        ],
    }
    macro_13_reg = {
        "overall_regime": "Disinflationary Expansion",
        "evaluated_at": "2026-11-12T08:00:00Z",
        "time_horizon": "6M",
        "conviction_level": "HIGH",
        "conviction_rationale": "Resilient growth and disinflationary productivity gains.",
        "regime_probabilities": {"expansion": 70.0, "soft_landing": 20.0},
        "asset_allocation": [{"asset_class": "Equities", "region": "US", "stance": "Overweight", "confidence": "high", "allocation_delta": "+5%", "rationale": "Strong earnings", "invalidation_conditions": ["Core PCE > 3.5%"]}],
        "sector_analysis": {"as_of_date": "2026-11-11", "summary_th": "เน้น Software & Cloud"},
        "risk_scenarios": [{"tail_risk": "Rates Spike", "probability": "low", "impact": "moderate", "trigger_to_activate": "10Y > 5.0%", "hedge_instruments": ["SHY"], "mitigation_strategy": "Trim long duration", "early_warning_indicators": ["Term premium > 80bps"]}],
        "thailand_market_stance": {
            "valuation": {"pe_ratio": 14.0, "pbv_ratio": 1.3, "dividend_yield": 3.5},
            "policy_spread_bps": -150,
            "yield_curve_spread_10y_2y_bps": 32,
            "investor_flow": {"institution_net_mb": 900, "foreign_net_mb": -600, "retail_net_mb": -300},
            "market_breadth": {"sentiment": "Cautious"},
            "rationale": "ไทยได้แรงหนุนท่องเที่ยวและปันผลสูงแต่ยังถูกกดดันจาก Fund Flow ต่างชาติและความไม่แน่นอนด้านนโยบายการคลังระยะสั้น " * 4,
        },
        "regional_assessments": {
            rname: {"economic_state": "Expanding", "growth_score": 0.6, "inflation_score": 0.3, "monetary_score": 0.4, "confidence": 0.8}
            for rname in ("United States", "Thailand", "Euro Area", "China", "Japan", "India", "United Kingdom", "Taiwan", "South Korea", "ASEAN", "Brazil", "Canada", "Australia")
        },
        "dashboard_indicators": [
            {"label": "Chicago Fed National Financial Conditions Index", "display_value": "-0.48"},
            *[{"label": f"Ind_{i}", "display_value": f"{i}.0%"} for i in range(35)],
        ],
        "report_references": [
            {"kind": "news", "title": "Fed Waller signals patience on easing path", "publisher": "Reuters", "age_hours": 4, "summary": "Key takeaways on neutral rate."}
        ],
    }
    axis_rich = {
        "content_hash": "axis_100",
        "complete_sections": {
            "basic_policy": ("เน้นถือหุ้น Quality Compounder ที่มีความได้เปรียบเชิงแข่งขันยั่งยืนและกระแสเงินสดอิสระแข็งแกร่ง " * 7).strip(),
            "role_models": ["Terry Smith"],
        },
    }

    providers = _StubProviders(
        equity_map={"GOOGL": rich, "ORCL": orcl},
        reviews={"GOOGL": pos_review},
        theses={"GOOGL": active_thesis},
        portfolio=port_full,
        macro=macro_13_reg,
        axis=axis_rich,
        news_map=news_map,
        notes_map=notes_map,
        note_contents={
            ec_path: {"title": "Q3_2026_GOOGL_Earnings_Call", "content": long_ec_body},
            ea_path: {"title": "GOOGL Equity Analysis 2026-11-10", "content": long_ea_body},
        },
    )
    asm = RoundtableContextAssembler(providers=providers)

    # A. WITHOUT @Portfolio -> strict 3-State Privacy Guard strips Around Entry / worth holding / triggers / private entry_price
    pkt_nop = asm.assemble(
        session_id="s_100_nop",
        agenda_prompt="วิเคราะห์ $GOOGL แบบไม่มีพอร์ต",
        context_refs=["$GOOGL"],
    )
    assert pkt_nop.portfolio_guard.attachment_state == ContextAttachmentState.NOT_ATTACHED_DO_NOT_ASSUME
    ed_json = json.dumps(pkt_nop.equity_data, ensure_ascii=False)
    for forbidden in (
        "Around Entry",
        "worth holding",
        "at its current size",
        "is_held",
        "2026-06-05",
        "Entry Trim",
        "157.89",
        "Bought at compressed multiple",
        "Then vs Now",
    ):
        assert forbidden not in pkt_nop.equity_report_text, f"Privacy leak in report: {forbidden}"
        assert forbidden not in ed_json, f"Privacy leak in equity_data: {forbidden}"

    # B. WITH @Portfolio, @Macro, @Essence (Single Stock = 100% completeness!)
    pkt = asm.assemble(
        session_id="s_100_full",
        agenda_prompt="วิเคราะห์ $GOOGL แบบครบ 100%",
        context_refs=["$GOOGL", "@Portfolio", "@Macro", "@Essence"],
    )
    eq = pkt.equity_report_text
    for expected_eq in (
        "VolConfirm: confirmed_accumulation | MissingSessions: 2",
        "AnalystExp=78.50 | MgmtGuidance=84.00",
        "ExecBreakdown(Stage=90.00, Setup=82.00, Insider=60.00, reason=Healthy pullback)",
        long_stance_reason.strip(),
        "(Bear=-4.50%, Bull=32.00%)",
        "ExitFCFMult=22.40x (Bench=20.00x)",
        long_wacc_adj,
        "RDCF PV Breakdown: EV=$2.15T | EquityVal=$2.24T | PV 5Y FCF=$410.00B | PV Terminal=$1.74T | BaseRev=$402.80B | ExitMult Intrinsic=232.10",
        "Target Upside=19.50%, Downside=-8.20%",
        "RDCF Sanity: Tier=HIGH (Score=88)",
        long_arch_rat.strip(),
        "Quality & Solvency Sub-Scores (0-100): FCFQual=88.00 | DebtQual=95.00 | Solvency=97.50 | PeerRel=76.00",
        "NARRATIVE_2400_UNCLIPPED_END",
        "EA_OVER_3000_CONTINUATION_TAIL",
        "ธีมสำคัญ (Sentiment Context): Sovereign Cloud & TPU demand broaden moat.",
        "Single-source verification caveat:",
        "EC_3200_END_MARKER",
        "GOOGL News Headline #11",
        "YT Video #6",
        "Active Stock Thesis (Vault) [ACTIVE | HIGH_CONVICTION]:",
        "Active Falsifier [AF1] Cloud Growth YoY < 18 % [DOWNGRADE]:",
        "Linked Catalyst [CAT1]: Q4 Gemini Enterprise Expansion (Expected: 2027-02-04, Status: PENDING)",
        "TargetPriceDelta=8.50%, Stance=HOLD->OVERWEIGHT",
        "UnflaggedSell=2, Exercise=3, TaxWithhold=5",
        "Then vs Now (Around Entry (2026-06-05)): Price 157.89 -> 182.40",
        "Assessment: Is Alphabet worth holding at its current size given capex trajectory?",
    ):
        assert expected_eq in eq, f"Missing expected single-stock text: {expected_eq}"

    pg = pkt.portfolio_guard.summary_text
    assert "ต้นทุนรวม: 980000.00 THB" in pg
    assert "HLD24" in pg
    assert "Other Holdings (#26-28," in pg
    assert "OV9=" in pg and "[CAP_BREACH,HIDDEN_CONC]" in pg
    assert "Flex: adjustable" in pg

    m_txt = pkt.macro_report_text
    assert "Chicago Fed National Financial Conditions Index=-0.48" in m_txt
    assert "Australia|Expanding|" in m_txt
    assert "MacroInd_" not in m_txt and "Ind_34=34.0%" in m_txt
    assert "Macro News & Media References:" in m_txt
    assert "[NEWS] Fed Waller signals patience on easing path (Reuters, 4h ago)" in m_txt
    assert len(macro_13_reg["thailand_market_stance"]["rationale"]) > 220
    assert macro_13_reg["thailand_market_stance"]["rationale"].strip() in m_txt
    for marker in ("Asset Allocation Stance:", "Sector Rotation", "Macro Risk Scenarios:", "Thailand Stance:"):
        assert m_txt.index(marker) < 3000

    assert axis_rich["complete_sections"]["basic_policy"] in pkt.essence_report_text

    # C. Multi-ticker keeps compact @Macro (detailed=False -> 6 regions, no Macro News References)
    pkt_multi = asm.assemble(
        session_id="s_100_multi",
        agenda_prompt="เทียบ $GOOGL vs $ORCL",
        context_refs=["$GOOGL", "$ORCL", "@Macro"],
    )
    assert "Australia: Expanding" not in pkt_multi.macro_report_text
    assert "Macro News & Media References:" not in pkt_multi.macro_report_text


def test_wp_s100_p2_watchdog_promise_analyst_insider_advisory_essence_and_privacy():
    now_utc = datetime.now(timezone.utc)
    stale_30d_iso = (now_utc - timedelta(days=30)).strftime("%Y-%m-%dT09:00:00Z")
    past_due_date = (now_utc - timedelta(days=12)).strftime("%Y-%m-%d")
    stale_ac_date = (now_utc - timedelta(days=25)).strftime("%Y-%m-%dT05:00:00Z")

    googl, _ = _sample_nested_sidecars()
    ftnt = json.loads(json.dumps(googl))
    ftnt["ticker"] = "FTNT"
    ftnt["company_name"] = "Fortinet, Inc."
    ftnt["evaluated_at"] = stale_30d_iso
    ftnt["quant_signals"]["deterministic_scorecard"].update(
        {
            "macro_regime": "Late-Cycle Tightening",
            "macro_alignment_status": "Tailwind",
        }
    )
    ftnt["quant_signals"]["smart_money_flags"] = {
        "overall_smart_money_flag": "neutral",
        "insider_signal": "neutral",
        "insider_buy_count_90d": 0,
        "insider_sell_count_90d": 0,
        "insider_ownership_pct": 17.36,
        "institutional_ownership_pct": 71.52,
        "short_interest_pct": 2.26,
        "short_squeeze_risk": False,
    }

    ftnt_thesis = {
        "status": "ACTIVE",
        "evidence_tier": "MEDIUM_CONVICTION",
        "claim_statement": "Fortinet SASE & OT firewall transition compounds FCF > 25%.",
        "falsifiers": [],
        "linked_catalysts": [
            {
                "catalyst_id": "CAT_PAST",
                "event_name": "Q3 2026 Billings Inflection",
                "expected_date_str": past_due_date,
                "status": "PENDING",
            }
        ],
    }

    # 3 duplicate PENDING/REVIEW_REQUIRED rows + 1 RESOLVED row (should be filtered out)
    ftnt_exceptions = [
        {
            "exception_id": "EXC-1",
            "claim_id": "TH-FTNT-v1",
            "ticker": "FTNT",
            "detected_at_utc": "2026-11-01T08:00:00Z",
            "exception_type": "FALSIFIER_BREACHED",
            "metric_name": "current_price",
            "expected_value": "< 153.75 USD",
            "observed_value": "140.00 USD",
            "human_action": "REVIEW_REQUIRED",
            "status": "PENDING",
        },
        {
            "exception_id": "EXC-2",
            "claim_id": "TH-FTNT-v1",
            "ticker": "FTNT",
            "detected_at_utc": "2026-11-05T08:00:00Z",
            "exception_type": "FALSIFIER_BREACHED",
            "metric_name": "current_price",
            "expected_value": "< 153.75 USD",
            "observed_value": "130.04 USD",
            "human_action": "REVIEW_REQUIRED",
            "status": "PENDING",
        },
        {
            "exception_id": "EXC-3",
            "claim_id": "TH-FTNT-v1",
            "ticker": "FTNT",
            "detected_at_utc": "2026-11-03T08:00:00Z",
            "exception_type": "FALSIFIER_BREACHED",
            "metric_name": "current_price",
            "expected_value": "< 153.75 USD",
            "observed_value": "135.50 USD",
            "human_action": "REVIEW_REQUIRED",
            "status": "PENDING",
        },
        {
            "exception_id": "EXC-RES",
            "claim_id": "TH-FTNT-v0",
            "ticker": "FTNT",
            "detected_at_utc": "2026-10-01T08:00:00Z",
            "exception_type": "FALSIFIER_BREACHED",
            "metric_name": "gross_margin",
            "expected_value": "< 70%",
            "observed_value": "68%",
            "status": "RESOLVED",
        },
    ]

    ftnt_promise = {
        "credibility": 92.5,
        "grade": "A",
        "confidence": "High",
        "delivery": 94.0,
        "transparency": 90.0,
        "n_promises": 16,
        "n_resolved": 12,
        "counts": {"Delivered": 10, "Partial": 1, "Missed": 1},
        "bias": {
            "label": "Conservative sandbagger",
            "median_delta_pct": 3.4,
            "beat_rate": 85.0,
            "message": "Management consistently under-promises and beats guidance figures across multi-quarter horizons without slipping core milestones.",
        },
        "milestone_slip_quarters": 0.25,
        "trend": {"verdict": "improving"},
        "top_kept": ["P-FTNT-01", "P-FTNT-02"],
        "top_broken": ["P-FTNT-09"],
    }

    ftnt_analyst = {
        "ticker": "FTNT",
        "target_mean": 167.91,
        "target_low": 104.0,
        "target_high": 225.0,
        "num_analysts": 37,
        "next_earnings_date": past_due_date,
        "source_as_of": stale_ac_date,
        "earnings_history": [
            {"date_str": past_due_date, "eps_actual": None, "eps_estimate": 0.86},
            {"date_str": "2026-07-29", "eps_actual": 0.90, "eps_estimate": 0.75},
            {"date_str": "2026-05-06", "eps_actual": 0.82, "eps_estimate": 0.62},
        ],
    }

    ftnt_insider_db = {
        "ticker": "FTNT",
        "window_days": 180,
        "source_provenance": "YFINANCE_HEURISTIC",
        "reclassified_zero_dollar_buys": 12,
        "open_market_buy_count": 3,
        "open_market_buy_shares": 469108.0,
        "open_market_buy_usd": 11465472.66,
        "open_market_sale_count": 5,
        "paired_exercise_sale_count": 2,
        "total_sale_count": 7,
        "total_sale_shares": 475782.0,
        "total_sale_usd": 69575600.47,
        "exercise_count": 2,
        "award_grant_count": 12,
        "tax_withhold_count": 1,
        "recent_notable": [
            {
                "transaction_date": "2026-08-03",
                "owner": "XIE KENNETH QING",
                "officer_title": "Chief Executive Officer",
                "effective_code": "S",
                "shares": 161482.0,
                "price_per_share": 162.66,
                "usd_value": 26266662.12,
                "paired_exercise": True,
            }
        ],
    }

    ftnt_news = {
        "items": [
            {
                "title": "Fortinet wins major OT defense contract",
                "age_hours": 10,
                "is_stale": False,
                "thesis_signal": "strengthens",
                "impact_type": "IDIOSYNCRATIC",
                "macro_impact_score": 2.0,
                "asset_impact_score": 8.5,
                "why_it_matters": "Expands critical infrastructure backlog.",
            }
        ],
        "youtube_mentions": [
            {
                "thai_title": "เจาะงบไซเบอร์ FTNT",
                "channel": "CyberInvest",
                "alignment_with_news": "aligned",
                "alignment_note": "Corroborates strong OT backlog",
                "key_takeaways": ["OT +55% YoY"],
            }
        ],
    }

    rich_macro = {
        "overall_regime": "Disinflationary Expansion",
        "evaluated_at": "2026-11-12T08:00:00Z",
        "time_horizon": "6M",
        "conviction_level": "HIGH",
        "conviction_rationale": "Cooling PCE + resilient capex.",
        "regime_probabilities": {"expansion": 70.0},
        "asset_allocation": [{"asset_class": "Equities", "region": "US", "stance": "Overweight", "rationale": "Breadth improving"}],
        "sector_analysis": {
            "analysis_status": "available",
            "as_of_date": "2026-11-11",
            "summary_th": "IT & Cyber นำตลาด",
            "fact_claims": ["XLK Leading quadrant +4.2% RS"],
            "resolved_metrics": [{"metric_name": "XLK_RS_RATIO", "value": 104.2, "unit": ""}],
            "validation_warnings": ["momentum_direction_mismatch:XLF"],
        },
        "risk_scenarios": [{"tail_risk": "Rates Spike", "probability": "20%", "impact": "High", "trigger_to_activate": "10Y > 5%", "hedge_instruments": ["SHY"]}],
        "thailand_market_stance": {
            "valuation": {"pe_ratio": 14.0, "pbv_ratio": 1.3, "dividend_yield": 3.5},
            "policy_spread_bps": -150,
            "yield_curve_spread_10y_2y_bps": 32,
            "investor_flow": {"institution_net_mb": 900, "foreign_net_mb": -600, "retail_net_mb": -300},
            "market_breadth": {"sentiment": "Cautious"},
        },
        "strategy_report_evidence": [
            {"claim": "Tech capex upcycle intact", "source": "Strategy_Vault_Nov"},
        ],
        "data_timestamp_notes": ["FRED payrolls lagged 24h due to holiday"],
    }

    port_with_levers = {
        "total_value_thb": 1000000.0,
        "cash_thb": 150000.0,
        "cash_pct": 15.0,
        "holdings": [{"symbol": "NVDA", "units": 50, "avg_cost": 110.0, "market_value_thb": 850000.0, "bucket_id": "core"}],
        "goals_progress": [
            {
                "name": "Wealth Goal",
                "goal_type": "nav_target",
                "target_amount_thb": 10000000.0,
                "current_amount_thb": 1000000.0,
                "progress_pct": 10.0,
                "deadline": "2036-12-31",
                "deadline_days_left": 3700,
                "priority_tier": "wishes",
                "flexibility": "timeline_flexible",
                "monthly_savings_thb": 20000.0,
                "required_return_pct": 18.31,
                "tension_level": "CRITICAL_TENSION",
                "adjustment_levers": {
                    "save_more_monthly_thb": 29771.67,
                    "total_monthly_savings_needed_thb": 49771.67,
                    "extend_timeline_months": 87,
                    "total_months_needed": 209,
                    "downsized_target_thb": 4790041.88,
                },
            }
        ],
    }

    advisory_stub = {
        "evaluated_at": "2026-11-12T08:00:00Z",
        "data_mode": "OFFLINE_VAULT",
        "total_nav_thb": 1000000.0,
        "risk_radar": {
            "weighted_beta": 1.18,
            "total_portfolio_volatility_pct": 21.4,
            "hhi": 4200,
            "effective_holdings_count": 2.4,
            "concentration_risk_level": "HIGH",
            "bucket_active_share_pct": 34.5,
            "drawdown_pct": -6.2,
            "drawdown_posture_tier": "CAUTION",
            "unassigned_holdings_nav_pct": 0.0,
        },
        "cash_plan": {
            "cash_thb_available": 100000.0,
            "cash_usd_available": 1400.0,
            "usd_thb_fx_rate": 35.0,
            "immediate_deploy_thb": 25000.0,
            "immediate_deploy_usd": 500.0,
            "conditional_limit_order_thb": 30000.0,
            "conditional_limit_order_usd": 400.0,
            "fx_conversion_needed_thb": 0.0,
        },
        "holdings_advisory": [
            {
                "symbol": "FTNT",
                "evaluation_track": "TRACK_1_DIRECT_EQUITIES",
                "current_weight_pct": 0.0,
                "target_weight_pct": 4.5,
                "weight_drift_pct": -4.5,
                "action_code": "WAIT_PULLBACK",
                "is_immediate_execution": False,
                "live_upside_12m_pct": -12.0,
                "macro_alignment": "headwind",
                "sector_quadrant": "Weakening",
                "risk_contribution_pct": 0.0,
            }
        ],
        "anomaly_warnings": [
            {
                "guardrail_code": "TOP1_CONC",
                "ticker": "NVDA",
                "message": "Single holding exceeds 25% cap.",
                "severity": "HIGH",
            }
        ],
    }

    investor_essence_stub = {
        "statement": "เน้นลงทุนธุรกิจคุณภาพสูงที่มีกระแสเงินสดอิสระแกร่งและคูเมืองยั่งยืน",
        "confirmed_at_iso": "2026-10-15T10:00:00Z",
        "content_hash": "ess_hash_v1",
        "accepted_claims": [
            {"text": "หลีกเลี่ยงหุ้นที่คาดหวังการเติบโตเกินจริงโดยไร้ FCF รองรับ", "source_kind": "user", "fit_rating": "CORE"}
        ],
    }

    providers = _StubProviders(
        equity_map={"FTNT": ftnt},
        theses={"FTNT": ftnt_thesis},
        thesis_exceptions={"FTNT": ftnt_exceptions},
        promise_scores={"FTNT": ftnt_promise},
        analyst_context={"FTNT": ftnt_analyst},
        insider_db={"FTNT": ftnt_insider_db},
        news_map={"FTNT": ftnt_news},
        macro=rich_macro,
        portfolio=port_with_levers,
        portfolio_advisory=advisory_stub,
        investor_essence=investor_essence_stub,
    )
    asm = RoundtableContextAssembler(providers=providers)

    # 1. WITHOUT @Portfolio -> strict 3-State Guard NEVER calls get_portfolio_advisory_summary & hides advisory/levers
    pkt_nop = asm.assemble(
        session_id="s_p2_nop",
        agenda_prompt="วิเคราะห์ $FTNT กับ @Macro",
        context_refs=["$FTNT", "@Macro"],
    )
    assert providers.advisory_calls == 0
    assert pkt_nop.portfolio_guard.attachment_state == ContextAttachmentState.NOT_ATTACHED_DO_NOT_ASSUME
    combined_nop = pkt_nop.portfolio_guard.summary_text + "\n" + pkt_nop.equity_report_text
    for secret in ("Portfolio Risk Radar", "Dual-Currency Cash Plan", "Holdings Advisory Signals", "Goal Adjustment Levers", "Live Advisory MacroAlign"):
        assert secret not in combined_nop, f"Leak without @Portfolio: {secret}"

    # Verify @Essence auto-attached via investor_essence even when axis is empty and @Essence omitted
    assert "Confirmed Investor Essence" in pkt_nop.essence_report_text
    assert "หลีกเลี่ยงหุ้นที่คาดหวังการเติบโตเกินจริง" in pkt_nop.essence_report_text

    # 2. WITH @Portfolio, @Macro, @Essence -> full 100% completeness
    pkt = asm.assemble(
        session_id="s_p2_full",
        agenda_prompt="วิเคราะห์ $FTNT เต็มรูปแบบ",
        context_refs=["$FTNT", "@Macro", "@Portfolio", "@Essence"],
    )
    assert providers.advisory_calls == 1
    assert "sidecar_stale_30d:freshness" in pkt.data_quality_flags
    assert "active_thesis_breached:thesis_watchdog" in pkt.data_quality_flags

    eq = pkt.equity_report_text
    for expected_eq in (
        "Sidecar Age: 30d",
        "[STALE SIDECAR WARNING]",
        "[ACTIVE THESIS BREACH ALERT]: พบสัญญาณผิดเกณฑ์ Watchdog PENDING จำนวน 1 รายการ",
        "Analyst Context Cache",
        "[STALE 25d]",
        "[PAST-DUE EARNINGS DATE]",
        "Beat 2/2 reported",
        f"Upcoming Est: {past_due_date}: est=0.86",
        "Promise Ledger Credibility: Score=92.50/100 (Grade=A, Conf=High)",
        "without slipping core milestones.",
        "Standout Promises: Kept=P-FTNT-01, P-FTNT-02 | Broken/Dropped=P-FTNT-09",
        "Insider Own=17.36%",
        "[MACRO_ALIGNMENT_DRIFT_WARNING]: Sidecar อ้าง Regime='Late-Cycle Tightening' (Tailwind)",
        "Live Advisory MacroAlign='headwind'",
        "[WATCHDOG BREACHES (PENDING)]:",
        "Observed '130.04 USD' (Detected 2026-11-05, x3)",
        "Status=PENDING | Action=REVIEW_REQUIRED",
        "Insider DB Ledger (180d, Provenance=YFINANCE_HEURISTIC): Open-Market Paid Buy=3 ($11.47M, 469.11K sh) | Pure Open-Market Sell=5, Paired Exercise/Award Sell=2",
        "[DATA HYGIENE NOTE]: คัดแยกธุรกรรม yfinance ราคา $0 จำนวน 12 รายการ",
        "2026-08-03 | XIE KENNETH QING (Chief Executive Officer) | Code=S | 161.48K sh @ $162.66 ($26.27M) [Paired Exercise]",
        "Linked Catalyst [CAT_PAST]: Q3 2026 Billings Inflection",
        "[PAST-DUE CATALYST — ควรตรวจสอบผล]",
        "[Impact: IDIOSYNCRATIC | Macro:2.0 | Asset:8.5]",
        "[aligned (Corroborates strong OT backlog)]",
    ):
        assert expected_eq in eq, f"Missing expected eq text: {expected_eq}"
    assert "gross_margin" not in eq  # RESOLVED exception excluded

    # Macro new fields while keeping top strategic markers < 3000
    m_txt = pkt.macro_report_text
    for expected_m in (
        "Sector Analysis Status: available",
        "Sector Fact Claims: XLK Leading quadrant +4.2% RS",
        "Sector Resolved Metrics: XLK_RS_RATIO=104.2",
        "Sector Validation Warnings: momentum_direction_mismatch:XLF",
        "Strategy Vault Evidence: Tech capex upcycle intact [Strategy_Vault_Nov]",
        "Data Timestamp Notes: FRED payrolls lagged 24h due to holiday",
    ):
        assert expected_m in m_txt
    for marker in ("Asset Allocation Stance:", "Sector Rotation", "Macro Risk Scenarios:", "Thailand Stance:"):
        assert m_txt.index(marker) < 3000

    # Portfolio Risk Radar, Cash Plan, Holdings Advisory, Anomaly Warnings, Goal Adjustment Levers
    pg_txt = pkt.portfolio_guard.summary_text
    for expected_pg in (
        "Portfolio Risk Radar (Mode=OFFLINE_VAULT): Beta=1.18 | Vol=21.40%",
        "(HHI=4200, Risk=HIGH)",
        "Dual-Currency Cash Plan (FX USD/THB=35.00)",
        "Holdings Advisory Signals",
        "FTNT|WAIT_PULLBACK|TRACK_1_DIRECT_EQUITIES|",
        "Portfolio Advisory Anomaly Warnings:",
        "[HIGH] TOP1_CONC (NVDA): Single holding exceeds 25% cap.",
        "Goal Adjustment Levers: ออมเพิ่ม +29.77K THB/ด.",
    ):
        assert expected_pg in pg_txt, f"Missing expected pg text: {expected_pg}"

    # Verify fractional 0.0-1.0 HHI normalizes to 0-10,000 scale and >15 holdings advisory aren't truncated
    advisory_frac = {
        **advisory_stub,
        "risk_radar": {**advisory_stub["risk_radar"], "hhi": 0.1220, "effective_holdings_count": 8.2},
        "holdings_advisory": advisory_stub["holdings_advisory"]
        + [
            {
                "symbol": f"STK{idx:02d}",
                "evaluation_track": "TRACK_1_DIRECT_EQUITIES",
                "current_weight_pct": round(5.0 - idx * 0.1, 2),
                "target_weight_pct": 5.0,
                "weight_drift_pct": round(-idx * 0.1, 2),
                "action_code": "HOLD",
                "is_immediate_execution": False,
            }
            for idx in range(1, 17)
        ],
    }
    pkt_frac = RoundtableContextAssembler(
        providers=_StubProviders(
            equity_map={"FTNT": ftnt},
            portfolio=port_with_levers,
            portfolio_advisory=advisory_frac,
        )
    ).assemble(
        session_id="s_hhi_and_17",
        agenda_prompt="วิเคราะห์ $FTNT",
        context_refs=["$FTNT", "@Portfolio"],
    )
    assert "(HHI=1220, Risk=HIGH)" in pkt_frac.portfolio_guard.summary_text
    assert "STK16|HOLD|" in pkt_frac.portfolio_guard.summary_text

    # Verified numeric facts + fingerprint tracks investor_essence hash changes
    facts_joined = " | ".join(pkt.verified_numeric_facts)
    for fk in (
        "promise_credibility_score=92.5",
        "promise_grade=A",
        "analyst_cache_target_mean=167.91",
        "insider_db_180d_open_buy_count=3",
        "insider_db_180d_open_sale_count=5",
        "insider_db_180d_paired_sale_count=2",
    ):
        assert fk in facts_joined

    investor_essence_v2 = {**investor_essence_stub, "content_hash": "ess_hash_v2"}
    pkt_ess_v2 = RoundtableContextAssembler(
        providers=_StubProviders(
            equity_map={"FTNT": ftnt},
            theses={"FTNT": ftnt_thesis},
            thesis_exceptions={"FTNT": ftnt_exceptions},
            promise_scores={"FTNT": ftnt_promise},
            analyst_context={"FTNT": ftnt_analyst},
            insider_db={"FTNT": ftnt_insider_db},
            news_map={"FTNT": ftnt_news},
            macro=rich_macro,
            portfolio=port_with_levers,
            portfolio_advisory=advisory_stub,
            investor_essence=investor_essence_v2,
        )
    ).assemble(
        session_id="s_p2_full_2",
        agenda_prompt="วิเคราะห์ $FTNT เต็มรูปแบบ",
        context_refs=["$FTNT", "@Macro", "@Portfolio", "@Essence"],
    )
    assert pkt.packet_fingerprint_sha256 != pkt_ess_v2.packet_fingerprint_sha256


def test_single_stock_then_vs_now_public_baseline_and_formatting_hygiene():
    from types import SimpleNamespace
    from application.equity.position_review_service import PositionReviewApplicationService
    from application.roundtable.context_assembler import _format_falsifier_condition, _format_macro_brief

    # 1) _format_falsifier_condition deduplicates embedded numeric threshold
    assert _format_falsifier_condition("below 30.8%", 30.8) == "below 30.8%"
    assert _format_falsifier_condition("below 30%", 30.0) == "below 30%"
    assert _format_falsifier_condition("<", 15.0) == "< 15.0"
    assert _format_falsifier_condition("< 15%", 15.0) == "< 15%"

    # 2) _format_macro_brief removes leftover empty "()" from macro allocation rationale
    macro_with_parens = {
        "overall_regime": "Disinflation",
        "evaluated_at": "2026-11-12T00:00:00Z",
        "time_horizon": "6M",
        "conviction_level": "medium",
        "conviction_rationale": "Macro baseline",
        "asset_allocation": [
            {
                "asset_class": "USD vs THB",
                "region": "Thailand",
                "stance": "Neutral",
                "rationale": "ดอลลาร์ได้แรงหนุนจากส่วนต่างอัตราดอกเบี้ยนโยบาย () (ปัจจุบันยังไม่มีข้อมูลส่วนต่างฯ)",
            }
        ],
    }
    macro_rendered = _format_macro_brief(macro_with_parens, detailed=True)
    assert "()" not in macro_rendered
    assert "ส่วนต่างอัตราดอกเบี้ยนโยบาย (ปัจจุบันยังไม่มีข้อมูลส่วนต่างฯ)" in macro_rendered

    # 3) PositionReviewApplicationService honors portfolio_id="" by returning public 1Y baseline (Option B)
    base_dt = datetime(2025, 1, 1, tzinfo=timezone.utc)
    candles = []
    for i in range(300):
        dt_i = base_dt + timedelta(days=i)
        # idx_1y for 300 candles is len(300)-253 = 47 -> close = 140.0; latest idx 299 -> close = 290.0
        close_px = 140.0 if i == 47 else (290.0 if i == 299 else 150.0 + i * 0.2)
        candles.append(SimpleNamespace(timestamp=dt_i.timestamp() * 1000.0, close=close_px))

    class _FakeOhlcv:
        def get_ohlcv(self, ticker, range_str="5y", interval_str="1d"):
            return SimpleNamespace(candles=candles)

    class _FakeFundamentals:
        def get_stock_profile(self, ticker):
            return {"trailing_eps": 10.0, "trailing_pe": 29.0, "revenue_growth_pct": 16.0, "pe_5y_high": 32.0}

    def _fake_portfolio_reader(portfolio_id="default", refresh_prices=False):
        return SimpleNamespace(
            summary=SimpleNamespace(total_value_thb=100000.0),
            holdings=[
                SimpleNamespace(
                    symbol="GOOGL",
                    units=3.2297,
                    avg_cost_usd=323.25,
                    unrealized_pnl_percent=-9.2,
                    market_value_thb=30000.0,
                    bucket_id="core",
                )
            ],
        )

    pr_svc = PositionReviewApplicationService(
        ohlcv_service=_FakeOhlcv(),
        fundamentals_provider=_FakeFundamentals(),
        portfolio_reader=_fake_portfolio_reader,
    )
    rep_public = pr_svc.run_position_review("GOOGL", portfolio_id="")
    assert rep_public.holding_summary["is_held"] is False
    assert rep_public.then_vs_now.is_held is False
    assert rep_public.then_vs_now.period_label == "Baseline (1 Year Ago)"
    assert rep_public.then_vs_now.then_price == 140.0
    assert rep_public.then_vs_now.now_price == 290.0

    rep_port = pr_svc.run_position_review("GOOGL", portfolio_id="default")
    assert rep_port.holding_summary["is_held"] is True
    assert rep_port.then_vs_now.is_held is True
    assert rep_port.then_vs_now.then_price == 323.25

    # 4) RoundtableContextAssembler passes portfolio_id="" without @Portfolio and deduplicates falsifier threshold
    import dataclasses

    class _PortAwareStubProviders(_StubProviders):
        def get_equity_position_review(self, ticker: str, portfolio_id: str = "default"):
            rep = pr_svc.run_position_review(ticker, portfolio_id=portfolio_id)
            return dataclasses.asdict(rep)

    googl_detail = {
        "ticker": "GOOGL",
        "company_name": "Alphabet Inc.",
        "current_price": 290.0,
        "quant_signals": {
            "composite_score": 82.0,
            "quality_score": 90.0,
            "value_score": 70.0,
            "growth_score": 85.0,
            "smart_money_flags": {
                "overall_smart_money_flag": "neutral",
                "insider_signal": "neutral",
                "insider_buy_count_90d": 0,
                "insider_sell_count_90d": 0,
                "insider_ownership_pct": 1.6,
                "institutional_ownership_pct": 80.71,
                "short_interest_pct": 1.49,
                "short_squeeze_risk": False,
            },
            "insider_conviction": {
                "data_status": "available",
                "status": "selling_activity",
                "signal_confidence": "high",
                "open_market_p_count_90d": 0,
                "open_market_s_count_90d": 2,
                "open_market_s_value_usd": 51545.35,
                "rule_10b5_1_s_count_90d": 2,
                "rule_10b5_1_s_value_usd": 51545.35,
                "unflagged_s_count_90d": 0,
            },
        },
        "thesis_falsifiers": [
            {
                "falsifier_id": "F1",
                "metric_name": "Operating Margin",
                "condition": "below 30.8%",
                "threshold_value": 30.8,
                "narrative_explanation": "Margin erosion signal",
            }
        ],
        "earnings_guidance_context": {
            "latest_fiscal_period": "Q3 2026",
            "management_tone": "bullish",
            "key_executive_quotes": [
                '"Alphabet revenues grew 25% year over year." (ref: Q3_2026.md#char_1_40)'
            ],
        },
    }
    port_stub = {
        "total_value_thb": 100000.0,
        "total_cost_basis_thb": 35000.0,
        "cash_thb": 10000.0,
        "cash_pct": 10.0,
        "holdings": [{"symbol": "GOOGL", "asset_type": "US Stock", "currency": "USD", "units": 3.2297, "avg_cost": 323.25, "market_value_thb": 30000.0, "unrealized_pnl_percent": -9.2, "bucket_id": "core"}],
    }
    prov = _PortAwareStubProviders(equity_map={"GOOGL": googl_detail}, portfolio=port_stub, macro=macro_with_parens)
    asm = RoundtableContextAssembler(providers=prov)

    pkt_nop = asm.assemble(session_id="s_tvn_pub", agenda_prompt="วิเคราะห์ $GOOGL", context_refs=["$GOOGL", "@Macro"])
    assert "323.25" not in pkt_nop.equity_report_text
    assert "323.25" not in json.dumps(pkt_nop.equity_data)
    assert "Then vs Now (Baseline (1 Year Ago)): Price 140.00 -> 290.00" in pkt_nop.equity_report_text
    assert "what it cost in" not in pkt_nop.equity_report_text
    assert "worth holding" not in pkt_nop.equity_report_text
    assert "[F1] Operating Margin below 30.8%: Margin erosion signal" in pkt_nop.equity_report_text
    assert "30.8% 30.8" not in pkt_nop.equity_report_text
    assert 'Exec Quote: ""' not in pkt_nop.equity_report_text
    assert 'Exec Quote: "Alphabet revenues grew 25% year over year."' in pkt_nop.equity_report_text
    assert "()" not in pkt_nop.macro_report_text
    assert "Insider Signal=neutral (90d discretionary buys=0, sells=0; scheduled 10b5-1 sells=2 ($51.55K))" in pkt_nop.equity_report_text

    # Reconcile stale smart_money_flags when Form 4 has unflagged discretionary sells
    googl_unflagged = json.loads(json.dumps(googl_detail))
    googl_unflagged["quant_signals"]["insider_conviction"]["unflagged_s_count_90d"] = 3
    googl_unflagged["quant_signals"]["insider_conviction"]["rule_10b5_1_s_count_90d"] = 0
    prov_unf = _PortAwareStubProviders(equity_map={"GOOGL": googl_unflagged}, portfolio=port_stub, macro=macro_with_parens)
    pkt_unf = RoundtableContextAssembler(providers=prov_unf).assemble(
        session_id="s_tvn_unf", agenda_prompt="วิเคราะห์ $GOOGL", context_refs=["$GOOGL"]
    )
    assert "Overall=bearish_signal | Insider Signal=selling (90d discretionary buys=0, sells=3)" in pkt_unf.equity_report_text

    # Suppress noisy Automated fallback blockquote in Vault Analysis excerpt
    from application.roundtable.context_assembler import _extract_analysis_note_excerpt
    fallback_md = "## 📊 บทวิเคราะห์\nStrong cloud momentum.\n\n## Sentiment & Narrative\n**ธีมสำคัญ:** Sentiment analysis unavailable (LLM offline or pending)\n> Automated fallback: LLM narrative unavailable at run time.\n"
    assert _extract_analysis_note_excerpt(fallback_md, shown_narrative="Strong cloud momentum.", include_sentiment_context=True) == ""

    pkt_with_port = asm.assemble(session_id="s_tvn_port", agenda_prompt="วิเคราะห์ $GOOGL", context_refs=["$GOOGL", "@Portfolio"])
    assert "Price 323.25 -> 290.00" in pkt_with_port.equity_report_text
    assert "Around Entry" in pkt_with_port.equity_report_text


def test_phase1_phase2_lossless_compaction_and_toon_tables():
    """Verify Phase 1-2 lossless compaction: Forensics, Quote Alias, Narrative Condenser, Vault Dedupe, @Essence, TOON."""
    from application.roundtable.context_assembler import (
        RoundtableContextAssembler,
        _condense_repetitive_narrative,
        _format_exec_quotes_lines,
        _format_forensics_line,
    )

    # 1. Forensics compact key-value serialization retains 100% of Sloan + DuPont values without JSON punctuation bloat
    f_line = _format_forensics_line(
        {
            "sloan_accrual_ratio": -0.0623,
            "raw_sloan_accrual_ratio": -0.062251264,
            "earnings_quality_rating": "NEUTRAL",
            "assets_basis": "two_period_average",
            "interpretation": "Moderate accruals (-0.0623 [raw -0.062251]): Cash flow reasonably tracks reported net income.",
            "fiscal_period_end": "2025-12-31",
            "period_type": "annual",
            "prior_fiscal_period_end": "2024-12-31",
        },
        {
            "net_profit_margin_pct": 32.81,
            "asset_turnover": 0.6767,
            "equity_multiplier": 1.43,
            "roe_pct": 31.83,
            "closure_error": 0.0804,
            "internal_closure_error": 0.0,
            "published_roe_reconstructed_pct": 31.7496,
            "is_published_reconciled": False,
            "fiscal_period_end": "2025-12-31",
            "period_type": "annual",
        },
    )
    assert '"sloan_accrual_ratio"' not in f_line
    for tok in ("-0.0623", "-0.062251", "NEUTRAL", "32.81%", "0.6767", "31.83%", "31.7496%", "PublishedReconciled=False"):
        assert tok in f_line

    # 2. Exec Quotes clean outer double-quotes & hoist shared long Vault paths into [EC1]
    q_lines = _format_exec_quotes_lines(
        [
            '"Cloud backlog hit $561B." (ref: 30_Knowledge_Base/Stocks/GOOGL/Earnings/Q3_2026.md#char_10_40:abc)',
            '"CapEx range $195B-$205B." (ref: 30_Knowledge_Base/Stocks/GOOGL/Earnings/Q3_2026.md#char_50_90:def)',
        ],
        clip_limit=500,
    )
    assert q_lines[0] == "  * Quote Source [EC1]: 30_Knowledge_Base/Stocks/GOOGL/Earnings/Q3_2026.md"
    assert q_lines[1] == '  * Exec Quote: "Cloud backlog hit $561B." [EC1#char_10_40:abc]'
    assert q_lines[2] == '  * Exec Quote: "CapEx range $195B-$205B." [EC1#char_50_90:def]'
    assert '""' not in "\n".join(q_lines)
    # Vault paths with spaces also alias cleanly
    q_lines_sp = _format_exec_quotes_lines(
        [
            '"Quote A" (ref: 30_Knowledge_Base/Stocks/GOOGL/Earnings/Q3 2026 Note.md#char_10_20:aa)',
            '"Quote B" (ref: 30_Knowledge_Base/Stocks/GOOGL/Earnings/Q3 2026 Note.md#char_30_40:bb)',
        ],
        clip_limit=500,
    )
    assert q_lines_sp[0] == "  * Quote Source [EC1]: 30_Knowledge_Base/Stocks/GOOGL/Earnings/Q3 2026 Note.md"
    assert q_lines_sp[1] == '  * Exec Quote: "Quote A" [EC1#char_10_20:aa]'

    # 3. Narrative Condenser drops pure quant restatements while keeping unique qualitative Cloud/Anthropic/ROI insight
    sample_narr = (
        "Alphabet Inc. (GOOGL) แสดงสถานะทางการเงินที่แข็งแกร่งด้วย Piotroski F-Score ที่ 6/9 และ ROIC สูงถึง 35.73% เทียบ WACC 9.66% "
        "ในด้านการประเมินมูลค่า โมเดล Reverse DCF ชี้ว่ามี Margin of Safety 40.5% และไม่พึ่ง Terminal Value มากเกินไป (High Reliability Tier) "
        "ในเชิงเทคนิคและกระแสเงินสด หุ้น GOOGL อยู่ในสถานะ Stage 4 Markdown และอยู่ใน Buy Zone ที่มี Risk/Reward Ratio 2.86 เท่า "
        "ปัจจัยขับเคลื่อนหลัก (Catalysts) ได้แก่ การเติบโตของ Google Cloud และการลงทุนเชิงกลยุทธ์ใน Anthropic เพื่อเสริมระบบนิเวศ AI"
    )
    condensed = _condense_repetitive_narrative(sample_narr, max_chars=2000)
    assert "Anthropic" in condensed and "Google Cloud" in condensed
    assert "Piotroski F-Score" not in condensed and "Terminal Value" not in condensed and "Stage 4 Markdown" not in condensed

    # 4. End-to-end Vault snippet dedupe + @Essence Risk Limits synonym dedupe & uniform Accepted Claims + @Macro stub dedupe
    googl, _ = _sample_nested_sidecars()
    googl["quant_signals"]["historical_financials_3y"] = [
        {"fiscal_year": "2024A", "revenue": 350e9, "gross_margin_pct": 58.2, "ebit_margin_pct": 32.11, "rd_pct_of_rev": 14.09, "fcf": 72.76e9, "capex_to_depreciation_ratio": 3.43, "reinvestment_lifecycle_phase": "aggressive_expansion"}
    ]
    prov = _StubProviders(
        equity_map={"GOOGL": googl},
        macro={
            "overall_regime": "Reflation",
            "conviction_level": "HIGH",
            "regime_probabilities": {"Reflation": 65.0},
            "regional_assessments": {"United States": {"economic_state": "Reflation", "growth_score": 0.73, "inflation_score": -1.0, "monetary_score": 0.5, "confidence": 0.9}},
            "report_references": [
                {"kind": "news", "title": "Macro Ref 1", "publisher": "Reuters", "summary": "ข่าวเศรษฐกิจมหภาคที่ใช้ประกอบการประเมินรอบนี้"},
                {"kind": "news", "title": "Macro Ref 2", "publisher": "FT", "summary": "ข่าวเศรษฐกิจมหภาคที่ใช้ประกอบการประเมินรอบนี้"},
            ],
        },
        axis={
            "complete_sections": {
                "basic_policy": "เน้นหุ้นเทคโนโลยีที่มี Moat แข็งแกร่ง",
                "risk_limits": {
                    "mdd_max_annual": {"value": "15.00", "unit": "percent"},
                    "mdd_percent": "15.00percent",
                    "max_loss_per_trade": "2.00%",
                    "max_loss_per_trade_percent": "2.00percent",
                    "custom_divergent_limit": "5.00%",
                },
            }
        },
        investor_essence={
            "statement": "เน้น Quality Compounder",
            "accepted_claims": [
                {"text": "Claim A: หลีกเลี่ยงบริษัทหนี้ท่วม", "fit_rating": "exact", "source_kind": "user_stated"},
                {"text": "Claim B: เน้น ROIC สูงกว่า WACC", "fit_rating": "exact", "source_kind": "user_stated"},
                {"text": "Claim C: ไม่ไล่ราคาตอน Valuation ตึง", "fit_rating": "exact", "source_kind": "user_stated"},
            ],
        },
        news_map={
            "GOOGL": {
                "items": [{"title": "Google Commits $10B to Anthropic IPO", "age_hours": 5}],
            }
        },
        notes_map={
            "GOOGL": {
                "items": [
                    {"matched_by": "news", "title": "2026-11-12 GOOGL Latest News", "modified_at": "2026-11-12", "snippet": "ข่าวล่าสุด: 1. Google Commits $10B to Anthropic IPO"},
                    {"matched_by": "earnings_call", "title": "Q3_2026_GOOGL_Earnings_Call", "relative_path": "GOOGL/Earnings/Q3_2026.md", "modified_at": "2026-11-12", "snippet": "GOOGL Earnings Call intro duplicate..."},
                    {"matched_by": "equity_analysis", "title": "GOOGL Equity Analysis 2026-11-12", "relative_path": "GOOGL/Analysis/EA_2026_11_12.md", "modified_at": "2026-11-12", "snippet": "📊 บทวิเคราะห์เชิงปริมาณ: Alphabet Inc. Analysis Price: $293.45"},
                ]
            }
        },
        note_contents={
            "GOOGL/Earnings/Q3_2026.md": {"title": "Q3_2026_GOOGL_Earnings_Call", "content": "## AI Highlights\n- Cloud revenue +103% YoY"},
            "GOOGL/Analysis/EA_2026_11_12.md": {"title": "GOOGL Equity Analysis 2026-11-12", "content": "## 📝 บทวิเคราะห์ (Narrative)\nUnique vault continuation insight."},
        },
    )
    pkt = RoundtableContextAssembler(providers=prov).assemble(
        session_id="s_phase12_comp",
        agenda_prompt="วิเคราะห์ $GOOGL เทียบ @Macro และ @Essence",
        context_refs=["$GOOGL", "@Macro", "@Essence"],
    )
    # Vault snippet deduplication markers appear, preserving titles without re-dumping full summaries
    assert "[สรุปข่าวเดียวกับรายการด้านบน]" in pkt.equity_report_text
    assert "[สรุป AI Highlights ฉบับเต็มด้านล่าง]" in pkt.equity_report_text
    assert "[ตัวเลขหลักใน SEC:HEADER / สาระเสริมด้านล่าง]" in pkt.equity_report_text
    # @Macro generic stub dedupe & TOON Regional row
    assert "ข่าวเศรษฐกิจมหภาคที่ใช้ประกอบการประเมินรอบนี้" not in pkt.macro_report_text
    assert "United States|Reflation|0.73|-1.00|0.50|0.90" in pkt.macro_report_text
    # @Essence uniform Accepted Claims hoisting & Risk Limits synonym dedupe + percent -> %
    assert "Accepted Claims [exact/user_stated]):" in pkt.essence_report_text
    assert "mdd_max_annual=15.00%" in pkt.essence_report_text
    assert "mdd_percent=" not in pkt.essence_report_text
    assert "max_loss_per_trade=2.00%" in pkt.essence_report_text
    assert "max_loss_per_trade_percent=" not in pkt.essence_report_text
    assert "custom_divergent_limit=5.00%" in pkt.essence_report_text


def test_inline_pe_ratio_and_semantic_verifier_guards():
    # 1. PE= renders inline in SEC:VALUATION when qs.pe_ratio is present
    googl, _ = _sample_nested_sidecars()
    googl["quant_signals"]["pe_ratio"] = 24.35
    prov = _StubProviders(equity_map={"GOOGL": googl})
    pkt = RoundtableContextAssembler(providers=prov).assemble(
        session_id="s_inline_pe",
        agenda_prompt="ประเมินมูลค่า $GOOGL",
        context_refs=["$GOOGL"],
    )
    assert "PE=24.35x | PE vs Peer=" in pkt.equity_report_text

    # 2. Verifier guards: Anti-Strawman (disabled & strictly unspoken), Price contradiction, Expanded @Portfolio
    custom_personas = [
        {**p.to_dict(), "enabled": False} if p.speaker_id == SpeakerId.SKEPTIC_GURU else p.to_dict()
        for p in GURU_PERSONA_CATALOG
    ]
    guarded_pkt = FrozenContextPacket(
        session_id="s_ver_guards",
        created_at_iso="2026-11-12T00:00:00Z",
        agenda_prompt="วิเคราะห์ $GOOGL",
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
            attachment_state=ContextAttachmentState.ATTACHED_FLAT_BOOK,
            portfolio_attached=True,
            is_held=False,
            summary_text="[PORTFOLIO_STATE: ATTACHED_FLAT_BOOK]",
        ),
        active_personas=tuple(custom_personas),
    )
    verifier = RoundtableConsistencyVerifier()

    # Claiming disabled skeptic spoke OR citing strictly unspoken growth_guru -> flagged
    straw_turn = RoundtableTurnPayload(
        speaker_id="value_guru",
        round_stage="opening",
        replying_to_speaker_id="skeptic_guru",
        spoken_dialogue="ตามที่คุณ Skeptic เตือนเรื่องต้นทุน AI และที่คุณ Growth พูดไปเมื่อสักครู่ ราคาปัจจุบันอยู่ที่ 120 ดอลลาร์ครับ",
        evidence_bullets=["[FACT] Ref"],
        stance_tag="HOLD",
    )
    res_bad = verifier.verify_turn(
        straw_turn,
        guarded_pkt,
        unspoken_speakers=["growth_guru"],
        prior_turns=[],
    )
    joined_issues = " | ".join(res_bad.issues)
    assert "Anti-Strawman Guard" in joined_issues
    assert "Skeptic" in joined_issues and "Growth" in joined_issues
    assert "Instrument Identity Anchor" in joined_issues

    # ATTACHED_FLAT_BOOK rejects hallucinated existing holdings
    fake_hold_turn = RoundtableTurnPayload(
        speaker_id="value_guru",
        round_stage="opening",
        spoken_dialogue="ตอนนี้คุณมี 50 หุ้นของ GOOGL ต้นทุนเฉลี่ยของคุณอยู่ที่ 210 ดอลลาร์ จึงถือต่อได้ครับ",
        evidence_bullets=["[FACT] Holding"],
        stance_tag="HOLD",
    )
    res_port = verifier.verify_turn(fake_hold_turn, guarded_pkt)
    assert not res_port.is_valid
    assert any("ATTACHED_FLAT_BOOK" in iss for iss in res_port.issues)

    # Round 2 rebuttal mentioning a Round 1 speaker (even if in Round 2 remaining_speakers) is valid!
    r1_growth = RoundtableTurnPayload(
        speaker_id="growth_guru",
        round_stage="opening",
        spoken_dialogue="การเติบโตของ Cloud ยังเร่งตัวต่อเนื่องครับ",
        evidence_bullets=["[FACT] Cloud"],
        stance_tag="BUY",
    )
    valid_r2 = RoundtableTurnPayload(
        speaker_id="value_guru",
        round_stage="rebuttal",
        replying_to_speaker_id="growth_guru",
        spoken_dialogue="เห็นด้วยกับคุณ Growth เรื่องการเติบโต แต่ด้วยราคาปัจจุบันประมาณ 293 ดอลลาร์ ควรรอจังหวะครับ",
        evidence_bullets=["[FACT] Ref price 293.45"],
        stance_tag="HOLD",
    )
    res_ok = verifier.verify_turn(
        valid_r2,
        guarded_pkt,
        unspoken_speakers=["growth_guru"],
        prior_turns=[r1_growth],
    )
    assert res_ok.is_valid, f"Unexpected issues: {res_ok.issues}"


def test_crossfire_duel_selection_pillar_coverage_verifier_rule8_and_youtube_pack():
    import io
    import wave
    import zipfile
    from core.roundtable.models import (
        analyze_debate_pillar_coverage,
        select_crossfire_duels,
    )
    from core.roundtable.youtube_export import build_youtube_production_pack

    # 1. Pillar coverage tracks all 5 core axes
    turns_opening = [
        RoundtableTurnPayload(
            speaker_id="value_guru",
            round_stage="opening",
            spoken_dialogue="P/E 35x เทียบ intrinsic value ทำให้ margin of safety แคบเกินไปครับ",
            evidence_bullets=["[FACT] Reverse DCF price 100"],
            stance_tag="HOLD",
        ),
        RoundtableTurnPayload(
            speaker_id="quality_guru",
            round_stage="opening",
            spoken_dialogue="ROIC สูง 45% และ moat ของระบบนิเวศแข็งแกร่งมาก",
            evidence_bullets=["[FACT] ROIC 45%"],
            stance_tag="OVERWEIGHT",
        ),
        RoundtableTurnPayload(
            speaker_id="growth_guru",
            round_stage="opening",
            spoken_dialogue="Data Center revenue growth เร่งตัว +90% YoY",
            evidence_bullets=["[FACT] Revenue +90%"],
            stance_tag="BUY",
        ),
        RoundtableTurnPayload(
            speaker_id="macro_tactical_guru",
            round_stage="opening",
            spoken_dialogue="สภาพคล่อง Fed rate cut หนุนหุ้นกลุ่มเทคโนโลยี",
            evidence_bullets=["[FACT] Reflation"],
            stance_tag="OVERWEIGHT",
        ),
        RoundtableTurnPayload(
            speaker_id="skeptic_guru",
            round_stage="rebuttal",
            replying_to_speaker_id="growth_guru",
            spoken_dialogue="ถ้า AI Capex สะดุดจะเกิด downside risk และ inventory ล้นทันที",
            evidence_bullets=["[FACT] Capex risk"],
            stance_tag="UNDERWEIGHT",
            sizing_posture="Defensive Trim",
        ),
    ]
    cov = analyze_debate_pillar_coverage(turns_opening)
    raised_count = sum(1 for speakers in cov["raised_by"].values() if speakers)
    assert raised_count == 5
    assert "skeptic_guru" in cov["challenged_by"]["Downside, Credit & Falsifiers"]
    assert "Valuation & MOS" in cov["unchallenged_pillars"]

    # 2. Crossfire selection chooses polarized Duel #1 and disjoint Duel #2 (4 turns)
    ordered = ["value_guru", "quality_guru", "growth_guru", "macro_tactical_guru", "skeptic_guru"]
    duels = select_crossfire_duels(turns_opening, ordered, max_duels=2)
    assert len(duels) == 4
    duel1_pair = {duels[0][0], duels[1][0]}
    duel2_pair = {duels[2][0], duels[3][0]}
    assert duels[0] == (duels[1][1], duels[1][0])
    assert duels[2] == (duels[3][1], duels[3][0])
    assert duel1_pair.isdisjoint(duel2_pair)

    # 3. Verifier Guard #8 requires replying_to_speaker_id in crossfire & fierce rebuttal
    fierce_pkt = FrozenContextPacket(
        session_id="s_fierce",
        created_at_iso="2026-11-12T00:00:00Z",
        agenda_prompt="วิเคราะห์ $NVDA",
        primary_ticker="NVDA",
        debate_intensity="crossfire",
        debate_depth="deep_youtube",
    )
    verifier = RoundtableConsistencyVerifier()
    missing_target = RoundtableTurnPayload(
        speaker_id="value_guru",
        round_stage="crossfire",
        replying_to_speaker_id=None,
        spoken_dialogue="ตัวเลขปัจจุบันยังไม่คุ้มค่าต่อการไล่ราคาครับ",
        evidence_bullets=["[FACT] Price"],
        stance_tag="HOLD",
        sizing_posture="Wait for Dip",
    )
    res_missing = verifier.verify_turn(missing_target, fierce_pkt, prior_turns=turns_opening)
    assert not res_missing.is_valid
    assert any("replying_to_speaker_id" in iss and "crossfire" in iss for iss in res_missing.issues)

    self_reply = missing_target.model_copy(update={"replying_to_speaker_id": "value_guru"})
    res_self = verifier.verify_turn(self_reply, fierce_pkt, prior_turns=turns_opening)
    assert not res_self.is_valid
    assert any("replying_to_speaker_id" in iss for iss in res_self.issues)

    # 4. Pure YouTube Production Pack builder stitches WAV, SRT & Cue Sheet deterministically
    def _make_wav(samples: int = 2400) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(b"\x01\x00" * samples)
        return buf.getvalue()

    wav1 = _make_wav(4800)  # 0.2s
    export_turns = [
        {
            "turn_id": "t1",
            "speaker_id": "hermes_moderator",
            "round_stage": "stage_intro",
            "spoken_dialogue": "สวัสดีครับ วันนี้เจาะลึก NVDA กันแบบจัดเต็มครับ",
            "evidence_bullets": ["[FACT] Ref 135"],
            "stance_tag": "REVIEW",
            "audio_cache_key": "k1",
        },
        {
            "turn_id": "t2",
            "speaker_id": "growth_guru",
            "round_stage": "crossfire",
            "replying_to_speaker_id": "skeptic_guru",
            "spoken_dialogue": "ยอดขาย Data Center โตจริงไม่ใช่มโนครับ.",
            "evidence_bullets": ["[FACT] Rev +90%"],
            "stance_tag": "BUY",
            "sizing_posture": "Core Allocation",
            "audio_cache_key": None,  # exercises silence duration fallback
        },
    ]
    pack = build_youtube_production_pack(
        session_id="rt_yt_01",
        title="ประชุมกูรู $NVDA",
        primary_ticker="NVDA",
        agenda_prompt="วิเคราะห์ $NVDA",
        final_rating="OVERWEIGHT",
        turns=export_turns,
        audio_bytes_by_key={"k1": wav1},
        packet=fierce_pkt,
        playbook_dict={
            "executive_verdict": "ทยอยสะสม",
            "kill_switches": ["Margin < 65%"],
            "tension_axes": [],
        },
    )
    assert pack["missing_audio_turn_ids"] == ["t2"]
    assert pack["total_duration_sec"] > 3.5
    with zipfile.ZipFile(io.BytesIO(pack["zip_bytes"]), "r") as zf:
        names = set(zf.namelist())
        assert {"audio_full.wav", "subtitles.srt", "youtube_production_kit.md", "timeline_manifest.json"} <= names
        with wave.open(io.BytesIO(zf.read("audio_full.wav")), "rb") as out_wf:
            assert out_wf.getnchannels() == 1
            assert out_wf.getframerate() == 24000
            assert out_wf.getnframes() > 24000
        srt = zf.read("subtitles.srt").decode("utf-8")
        assert "[Hermes]" in srt and "[Growth]" in srt and "-->" in srt
        kit = zf.read("youtube_production_kit.md").decode("utf-8")
        assert "00:00 Opening Hook" in kit
        assert "Shot-by-Shot Visual & B-Roll Cue Sheet" in kit










def test_preset_ui_context_distinct_and_forensic_pair_split():
    from core.roundtable.models import (
        GURU_PRESET_CATALOG,
        PRESET_ANALYTICAL_RUBRIC_MAP,
        PRESET_ROLE_CONTEXT_MAP,
        get_seat_ui_context,
    )

    non_default_keys = [
        (p.speaker_id.value, p.preset_id)
        for p in GURU_PRESET_CATALOG
        if p.preset_id != "default"
    ]
    assert len(non_default_keys) == 12
    for key in non_default_keys:
        assert key in PRESET_ROLE_CONTEXT_MAP, f"Missing PRESET_ROLE_CONTEXT_MAP for {key}"

    # Verify every catalog preset resolves non-empty context and specialized presets diverge from default
    by_seat = {}
    for preset in GURU_PRESET_CATALOG:
        sid = preset.speaker_id.value
        ctx = get_seat_ui_context(sid, preset.preset_id)
        assert ctx["role_lens_label"]
        assert ctx["focus_summary"]
        assert len(ctx["primary_evidence_labels"]) >= 2
        assert ctx["analytical_rubric"]
        by_seat.setdefault(sid, []).append((preset.preset_id, ctx))

    for sid, entries in by_seat.items():
        default_ctx = next(c for pid, c in entries if pid == "default")
        for pid, ctx in entries:
            if pid == "default":
                continue
            assert (
                ctx["role_lens_label"] != default_ctx["role_lens_label"]
                or ctx["focus_summary"] != default_ctx["focus_summary"]
            ), f"{sid}/{pid} did not specialize UI context beyond default"

    # Forensic Quality Auditor vs Forensic Financial Sleuth must not overlap in core lane
    q_forensic = PRESET_ANALYTICAL_RUBRIC_MAP[("quality_guru", "quality_forensic_auditor")]
    s_forensic = PRESET_ANALYTICAL_RUBRIC_MAP[("skeptic_guru", "skeptic_forensic_sleuth")]
    assert "ocf_to_net_income" in q_forensic
    assert "sloan_accrual" in q_forensic
    assert "piotroski_f_score" in q_forensic
    assert "ปล่อยเรื่องเครดิต/ICR" in q_forensic
    assert "DATA QUALITY FLAGS" in s_forensic
    assert "interest_coverage_ratio" in s_forensic
    assert "insider_conviction" in s_forensic
    assert "ocf_to_net_income" not in s_forensic
    assert "sloan_accrual" not in s_forensic


def test_suggest_opening_order_for_agenda_default_vs_custom():
    from dataclasses import replace
    from core.roundtable.models import (
        GURU_PERSONA_CATALOG,
        SpeakerId,
        merge_persona_overrides,
        suggest_opening_order_for_agenda,
    )

    default_personas = merge_persona_overrides(GURU_PERSONA_CATALOG)

    # 1. Generic/balanced agenda keeps Value first (canonical order)
    generic_order = suggest_opening_order_for_agenda("วิเคราะห์หุ้น NVDA รอบด้าน", default_personas)
    assert generic_order == [
        "value_guru",
        "quality_guru",
        "growth_guru",
        "macro_tactical_guru",
        "skeptic_guru",
    ]

    # 2. Risk/credit/red-flag focused agenda promotes Skeptic first when order is stock default
    risk_order = suggest_opening_order_for_agenda(
        "จับผิดความเสี่ยง หนี้สิน credit dilution และ kill-switch ของ INTC",
        default_personas,
    )
    assert risk_order[0] == "skeptic_guru"
    assert set(risk_order) == set(generic_order)

    # 3. Growth/AI catalyst focused agenda promotes Growth first when order is stock default
    growth_order = suggest_opening_order_for_agenda(
        "ประเมิน growth guidance catalyst และ backlog AI รอบงบหน้า",
        default_personas,
    )
    assert growth_order[0] == "growth_guru"

    # 4. User-customized debate_order (Quality ahead of Value) is strictly respected even on risk agenda
    custom_personas = tuple(
        replace(p, debate_order=1 if p.speaker_id == SpeakerId.QUALITY_GURU else (2 if p.speaker_id == SpeakerId.VALUE_GURU else p.debate_order))
        for p in default_personas
    )
    custom_order = suggest_opening_order_for_agenda(
        "จับผิดความเสี่ยง หนี้สิน credit dilution และ kill-switch ของ INTC",
        custom_personas,
    )
    assert custom_order[:2] == ["quality_guru", "value_guru"]


def test_verifier_guards_9_10_11_anti_echo_no_soft_agreement_and_skeptic_kill_switch():
    from core.roundtable.models import (
        ContextAttachmentState,
        FrozenContextPacket,
        InstrumentIdentityAnchor,
        PortfolioHoldingGuardState,
    )
    from schemas.roundtable_schemas import RoundtableTurnPayload
    from validators.roundtable_verifier import RoundtableConsistencyVerifier

    def _pkt(intensity: str = "balanced") -> FrozenContextPacket:
        return FrozenContextPacket(
            session_id="rt_guard_test",
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
            portfolio_guard=PortfolioHoldingGuardState(
                attachment_state=ContextAttachmentState.NOT_ATTACHED_DO_NOT_ASSUME,
                portfolio_attached=False,
            ),
            debate_intensity=intensity,
        )

    verifier = RoundtableConsistencyVerifier()

    prior_val = RoundtableTurnPayload(
        speaker_id="value_guru",
        round_stage="opening",
        spoken_dialogue=(
            "หุ้น NVDA ราคาตลาด 138 ดอลลาร์มีส่วนต่างความปลอดภัยแคบเมื่อเทียบกับต้นทุนวัฏจักรชิปในระยะกลาง "
            "จึงควรรอจังหวะย่อตัวก่อนเพิ่มน้ำหนักลงทุน"
        ),
        evidence_bullets=["[MATH] current_price 138 USD", "[FACT] Margin of Safety จำกัด"],
        stance_tag="HOLD",
    )

    # Guard #9: Exact/near-exact Thai echo (>55 chars shared prefix) rejected on opening
    echoed_quality = RoundtableTurnPayload(
        speaker_id="quality_guru",
        round_stage="opening",
        spoken_dialogue=(
            "หุ้น NVDA ราคาตลาด 138 ดอลลาร์มีส่วนต่างความปลอดภัยแคบเมื่อเทียบกับต้นทุนวัฏจักรชิปในระยะกลาง "
            "ผมมองคล้ายกันครับ"
        ),
        evidence_bullets=["[FACT] ROIC สูง 68%"],
        stance_tag="HOLD",
    )
    res_echo = verifier.verify_turn(echoed_quality, _pkt(), prior_turns=[prior_val])
    assert not res_echo.is_valid
    assert any("Anti-Echo Guard" in iss for iss in res_echo.issues)

    # Distinctive Quality opening on ROIC/OCF passes Guard #9
    distinct_quality = RoundtableTurnPayload(
        speaker_id="quality_guru",
        round_stage="opening",
        spoken_dialogue=(
            "ในมุมคุณภาพกิจการ ROIC ที่ 68 เปอร์เซ็นต์และเงินสดจากการดำเนินงานสูงกว่ากำไรสุทธิชัดเจน "
            "สะท้อนคูเมืองซอฟต์แวร์ CUDA ที่คู่แข่งเลียนแบบได้ยากในระยะ 5 ปี"
        ),
        evidence_bullets=["[FACT] ROIC 68%", "[MATH] OCF/Net Income 1.15x"],
        stance_tag="OVERWEIGHT",
    )
    res_distinct = verifier.verify_turn(distinct_quality, _pkt(), prior_turns=[prior_val])
    assert res_distinct.is_valid, res_distinct.issues

    # Guard #10: Soft agreement opening in heated/crossfire rebuttal against opposing camp is rejected
    prior_growth_buy = RoundtableTurnPayload(
        speaker_id="growth_guru",
        round_stage="opening",
        spoken_dialogue="รายได้ AI Data Center ขยายตัวแรงมาก EPS revision เป็นบวกต่อเนื่อง แนะนำซื้อสะสมทันที",
        evidence_bullets=["[FACT] Revenue +94% YoY"],
        stance_tag="BUY",
    )
    soft_rebuttal_value = RoundtableTurnPayload(
        speaker_id="value_guru",
        round_stage="rebuttal",
        replying_to_speaker_id="growth_guru",
        spoken_dialogue=(
            "เห็นด้วยอย่างยิ่งกับ Growth ครับว่ารายได้ Data Center โตดี แต่ Valuation ปัจจุบันเผื่อความคาดหวังไว้สูงมากแล้ว"
        ),
        evidence_bullets=["[MATH] FCF Yield 2.4%"],
        stance_tag="UNDERWEIGHT",
    )
    res_soft_heated = verifier.verify_turn(
        soft_rebuttal_value, _pkt("heated"), prior_turns=[prior_growth_buy]
    )
    assert not res_soft_heated.is_valid
    assert any("No-Soft-Agreement Guard" in iss for iss in res_soft_heated.issues)

    # Same soft phrasing in balanced mode is allowed
    res_soft_balanced = verifier.verify_turn(
        soft_rebuttal_value, _pkt("balanced"), prior_turns=[prior_growth_buy]
    )
    assert not any("No-Soft-Agreement Guard" in iss for iss in res_soft_balanced.issues)

    # Guard #11: Long Skeptic turn (>=120 chars) with zero risk/kill-switch signal fails
    toothless_skeptic = RoundtableTurnPayload(
        speaker_id="skeptic_guru",
        round_stage="opening",
        spoken_dialogue=(
            "บริษัทมีรายได้จากกลุ่มศูนย์ข้อมูลเติบโตต่อเนื่องและมีฐานะเงินสดในงบดุลที่ดูเรียบร้อยดีในไตรมาสล่าสุด "
            "ภาพรวมธุรกิจยังคงดำเนินไปตามแผนกลยุทธ์ที่ผู้บริหารวางไว้"
        ),
        evidence_bullets=["[FACT] Cash 38B USD"],
        stance_tag="HOLD",
    )
    res_toothless = verifier.verify_turn(toothless_skeptic, _pkt())
    assert not res_toothless.is_valid
    assert any("Skeptic Kill-Switch Guard" in iss for iss in res_toothless.issues)

    sharp_skeptic = RoundtableTurnPayload(
        speaker_id="skeptic_guru",
        round_stage="opening",
        spoken_dialogue=(
            "จุดตายสำคัญคือหาก Hyperscaler ชะลอ CapEx จน Gross Margin หลุด 70 เปอร์เซ็นต์ "
            "ต้องตั้ง Kill-Switch ทันทีเพราะมูลค่าหุ้นไม่รองรับการหดตัวของกำไร"
        ),
        evidence_bullets=["[FACT] Kill-Switch: Gross Margin < 70% หรือ CapEx ชะลอ"],
        stance_tag="UNDERWEIGHT",
    )
    res_sharp = verifier.verify_turn(sharp_skeptic, _pkt())
    assert res_sharp.is_valid, res_sharp.issues


@pytest.mark.parametrize(
    "claim",
    [
        "Current price is 999 USD. Build a downside scenario.",
        "Build a downside scenario. Current price is 999 USD.",
        "Current price is 999 USD; build a downside scenario.",
        "Current price is 999 USD\nConsider a scenario.",
        "DCF fair value is 200 in this scenario.",
    ],
)
def test_conditional_words_in_nearby_sentences_do_not_exempt_factual_claims(claim):
    packet = FrozenContextPacket(
        session_id="s_claim_scope",
        created_at_iso="2026-11-14T00:00:00Z",
        agenda_prompt="Review NVDA",
        primary_ticker="NVDA",
        data_quality_flags=("negative_fcf_dcf_unavailable:dcf",),
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
    turn = RoundtableTurnPayload(
        speaker_id="hermes_moderator",
        round_stage="synthesis",
        spoken_dialogue=claim,
        evidence_bullets=[],
        stance_tag="HOLD",
    )
    result = RoundtableConsistencyVerifier().verify_turn(turn, packet)
    assert not result.is_valid
    assert any("Data Quality Flags" in issue or "current-price claim conflicts" in issue for issue in result.issues)


@pytest.mark.parametrize(
    "claim",
    [
        "If the market reprices lower, current price is 999 USD.",
        "If free cash flow turns positive, DCF fair value is 200 USD.",
        "If the portfolio contains 100 shares, that would change the sizing.",
        "DCF fair value is 200 in this hypothetical scenario.",
        "\u0e16\u0e49\u0e32\u0e04\u0e38\u0e13\u0e21\u0e35 100 \u0e2b\u0e38\u0e49\u0e19 \u0e23\u0e32\u0e04\u0e32\u0e40\u0e1b\u0e49\u0e32\u0e2b\u0e21\u0e32\u0e22\u0e15\u0e32\u0e21 DCF \u0e04\u0e37\u0e2d 200",
    ],
)
def test_directly_conditional_claims_remain_exempt_from_factual_guards(claim):
    packet = FrozenContextPacket(
        session_id="s_conditional_control",
        created_at_iso="2026-11-14T00:00:00Z",
        agenda_prompt="Review NVDA",
        primary_ticker="NVDA",
        data_quality_flags=("negative_fcf_dcf_unavailable:dcf",),
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
    playbook = DistilledDecisionPlaybook(
        final_rating="HOLD",
        action_stance="HOLD",
        core_conviction_score=1,
        executive_verdict=claim,
        action_playbook=ActionPlaybookSpec(
            primary_action=ActionPlaybookOption(
                option_id="monitor",
                label="Monitor evidence",
                speaker_source="hermes_moderator",
                action_steps=["Monitor the next filing"],
            )
        ),
    )
    assert RoundtableConsistencyVerifier().verify_playbook(playbook, packet).is_valid


def test_playbook_parent_ticker_scope_applies_to_child_claims_and_explicit_ticker_wins():
    packet = FrozenContextPacket(
        session_id="s_parent_ticker_scope",
        created_at_iso="2026-11-14T00:00:00Z",
        agenda_prompt="Compare GOOGL and ORCL",
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
        secondary_anchors=(InstrumentIdentityAnchor(
            ticker="ORCL",
            company_name="Oracle Corp.",
            market="US",
            sector="Technology",
            industry="Software",
            currency="USD",
            benchmark_ticker="SPY",
            current_price="170.00",
        ),),
        ticker_data_quality_flags={"ORCL": ("negative_fcf_dcf_unavailable:dcf",)},
    )

    def _playbook(topic, argument):
        return DistilledDecisionPlaybook(
            final_rating="HOLD",
            action_stance="HOLD",
            core_conviction_score=1,
            executive_verdict="Monitor verified evidence.",
            tension_axes=[ConsensusTensionAxis(
                topic=topic,
                bull_side_speaker="growth_guru",
                bull_argument=argument,
                bear_side_speaker="value_guru",
                bear_argument="Cash flow and risk limits remain relevant.",
            )],
            action_playbook=ActionPlaybookSpec(
                primary_action=ActionPlaybookOption(
                    option_id="monitor",
                    label="Monitor evidence",
                    speaker_source="hermes_moderator",
                    action_steps=["Monitor the next filing"],
                )
            ),
        )

    verifier = RoundtableConsistencyVerifier()
    blocked = verifier.verify_playbook(
        _playbook("$ORCL valuation", "DCF fair value is 150 USD; current price is 999 USD."),
        packet,
    )
    assert not blocked.is_valid
    assert any("DCF claims are blocked" in issue for issue in blocked.issues)
    assert any("current-price claim conflicts" in issue for issue in blocked.issues)

    allowed = verifier.verify_playbook(_playbook("$GOOGL valuation", "DCF fair value is 150 USD."), packet)
    assert allowed.is_valid

    explicit_blocked = verifier.verify_playbook(
        _playbook("$GOOGL valuation", "DCF fair value for ORCL is 150 USD."),
        packet,
    )
    assert not explicit_blocked.is_valid

    from dataclasses import replace

    reverse_packet = replace(
        packet,
        primary_ticker="ORCL",
        instrument_anchor=packet.secondary_anchors[0],
        secondary_anchors=(packet.instrument_anchor,),
    )
    reverse_blocked = verifier.verify_playbook(
        _playbook("$ORCL valuation", "DCF fair value is 150 USD."),
        reverse_packet,
    )
    assert not reverse_blocked.is_valid
    reverse_allowed = verifier.verify_playbook(
        _playbook("$GOOGL valuation", "DCF fair value is 150 USD."),
        reverse_packet,
    )
    assert reverse_allowed.is_valid


def test_multi_ticker_unscoped_dcf_and_price_claims_request_clarification():
    packet = FrozenContextPacket(
        session_id="s_ambiguous_ticker_scope",
        created_at_iso="2026-11-14T00:00:00Z",
        agenda_prompt="Compare GOOGL and ORCL",
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
        secondary_anchors=(InstrumentIdentityAnchor(
            ticker="ORCL",
            company_name="Oracle Corp.",
            market="US",
            sector="Technology",
            industry="Software",
            currency="USD",
            benchmark_ticker="SPY",
            current_price="170.00",
        ),),
        ticker_data_quality_flags={"ORCL": ("negative_fcf_dcf_unavailable:dcf",)},
    )
    playbook = DistilledDecisionPlaybook(
        final_rating="HOLD",
        action_stance="HOLD",
        core_conviction_score=1,
        executive_verdict="DCF fair value is 150 USD. Current price is 999 USD.",
        action_playbook=ActionPlaybookSpec(
            primary_action=ActionPlaybookOption(
                option_id="monitor",
                label="Monitor evidence",
                speaker_source="hermes_moderator",
                action_steps=["Monitor the next filing"],
            )
        ),
    )
    result = RoundtableConsistencyVerifier().verify_playbook(playbook, packet)
    assert not result.is_valid
    assert any("Ticker Scope: DCF claim is ambiguous" in issue for issue in result.issues)
    assert any("Ticker Scope: current-price claim is ambiguous" in issue for issue in result.issues)


def test_macro_and_equity_complete_numeric_propagation_to_gurus():
    macro_payload = {
        "overall_regime": "Late-Cycle Expansion",
        "evaluated_at": "2026-11-14T08:00:00Z",
        "time_horizon": "3-6 Months",
        "conviction_level": "medium",
        "conviction_rationale": "US growth resilient while rate spread remains wide.",
        "regime_probabilities": {"Expansion": 55.0, "Slowdown": 30.0, "Recession": 15.0},
        "asset_allocation": [
            {
                "asset_class": "US Equities (AI/Tech Growth)",
                "asset_bucket": "equities",
                "region": "US",
                "stance": "Overweight",
                "confidence": "medium",
                "allocation_delta": "+5% vs MSCI ACWI",
                "benchmark_ref": "S&P 500",
                "time_horizon": "3-6 Months",
                "rationale": "Strong earnings momentum and AI CapEx.",
                "supporting_data": ["US Quant Score = +0.85", "UST 10Y Yield = 4.35%"],
                "invalidation_conditions": ["UST 10Y > 4.85%"],
                "why_not_high": "Single-source corroboration on credit spreads",
            }
        ],
        "risk_scenarios": [
            {
                "tail_risk": "Yield Spike Above 5%",
                "probability": "medium",
                "impact": "severe",
                "confidence": "high",
                "trigger_to_activate": "VIX > 26.0 and UST10Y > 4.90%",
                "hedge_instruments": ["TLT Put Spread", "GLD"],
                "hedge_size": "8% of portfolio NAV",
                "cost_or_tradeoff": "-1.2% p.a. option premium",
                "volume_threshold": "Volume > 75,000 contracts",
                "unwind_or_cover_condition": "VIX < 17.5",
                "mitigation_strategy": "Tactical duration & volatility collar",
                "early_warning_indicators": ["MOVE Index > 125"],
                "supporting_data": ["MOVE = 112.4", "VIX = 16.8"],
            }
        ],
        "pair_trades": [
            {
                "long_leg": "US Semiconductors",
                "short_leg": "EU Autos",
                "time_horizon": "1-3 Months",
                "confidence": "high",
                "sizing_guidance": "medium_risk_budget",
                "instrument_proxy": "Long SOXX / Short EXV5",
                "hedge_ratio": "1.0 : 1.0 Dollar-Neutral",
                "fx_handling": "USD/EUR 100% FX Hedged",
                "entry_trigger": "SOXX/EXV5 ratio > 4.20x",
                "stop_loss_trigger": "-3.5% spread divergence",
                "target_gain_or_rebalance": "+8.0% spread convergence",
                "max_drawdown_limit": "-4.0% of risk budget",
                "review_frequency": "Weekly",
                "thesis": "Secular AI compute demand vs tariff headwind.",
                "catalyst": "Q4 hyperscaler CapEx guidance",
                "risk": "Export control tightening",
                "supporting_data": ["SOXX Rev YoY = +28.0%", "EU Auto Margin = 5.4%"],
            }
        ],
        "regime_evidence": [
            {
                "dimension": "Monetary",
                "signal": "Restrictive",
                "confidence": "high",
                "evidence": "Fed Funds 4.50% vs Core PCE 2.7%",
                "conflict": "Labor market cooling faster than services CPI",
            }
        ],
        "thailand_market_stance": {
            "policy_spread_bps": -225.0,
            "yield_curve_spread_10y_2y_bps": 34.0,
            "valuation": {"pe_ratio": 14.8, "pbv_ratio": 1.38, "dividend_yield": 3.45},
            "investor_flow": {"foreign_net_mb": -1850.5, "institution_net_mb": 920.0, "retail_net_mb": 930.5},
            "market_breadth": {"sentiment": "Neutral-Weak"},
            "physical_gold": {"bar_sell_thb": 43850, "unit": "THB"},
        },
    }
    equity_payload = {
        "ticker": "NVDA",
        "company_name": "NVIDIA Corporation",
        "market": "US",
        "currency": "USD",
        "analysis_date": "2026-11-14",
        "quant_signals": {
            "ticker": "NVDA",
            "market": "US",
            "raw_analysis_price": 138.5,
            "composite_score": 84.2,
            "value_score": 62.0,
            "quality_score": 96.0,
            "growth_score": 94.0,
            "momentum_score": 81.0,
            "dividend_score": 25.0,
            "composite_score_waterfall": {
                "effective_weights": {"value": 25.0, "quality": 25.0, "growth": 25.0, "momentum": 15.0, "dividend": 10.0},
                "missing_dimensions": [],
                "solvency_gate_passed": True,
            },
            "roic_pct": 68.4,
            "fcf_yield_pct": 2.35,
            "net_debt_ebitda": -0.45,
            "reverse_dcf_result": {
                "is_actionable": True,
                "valuation_verdict": "fairly_valued",
                "solver_status": "converged",
                "intrinsic_value_today": 132.0,
                "target_price_12m": 152.0,
                "upside_12m_pct": 9.75,
                "mos_entry_price_today": 118.8,
                "current_margin_of_safety_pct": -4.92,
                "synthetic_credit_rating": "AAA",
                "interest_coverage_ratio": 85.2,
                "credit_spread_bps": 45.0,
                "benchmark_cost_of_debt_pct": 4.80,
                "effective_tax_rate_pct": 13.5,
                "after_tax_cost_of_debt_pct": 4.15,
            },
            "historical_financials_3y": [
                {
                    "fiscal_year": 2024,
                    "revenue": 60922000000,
                    "gross_margin_pct": 72.7,
                    "ebit_margin_pct": 54.1,
                    "rd_pct_of_rev": 14.2,
                    "sga_pct_of_rev": 4.4,
                    "roic_pct": 68.4,
                    "fcf": 27021000000,
                    "capex_to_depreciation_ratio": 0.95,
                    "reinvestment_lifecycle_phase": "Capital-Light Scaling",
                }
            ],
        },
    }
    providers = _StubProviders(equity_map={"NVDA": equity_payload}, macro=macro_payload)
    packet = RoundtableContextAssembler(providers=providers).assemble(
        session_id="s_complete_nums",
        agenda_prompt="วิเคราะห์ $NVDA ร่วมกับ @Macro",
        context_refs=["$NVDA", "@Macro"],
    )
    # Macro quantitative details in @Macro report text
    assert "US Quant Score = +0.85" in packet.macro_report_text
    assert "Bench=S&P 500" in packet.macro_report_text
    assert "WhyNotHigh: Single-source corroboration" in packet.macro_report_text
    assert "Size=8% of portfolio NAV" in packet.macro_report_text
    assert "Cost=-1.2% p.a. option premium" in packet.macro_report_text
    assert "Unwind=VIX < 17.5" in packet.macro_report_text
    assert "MOVE = 112.4" in packet.macro_report_text
    assert "Proxy=Long SOXX / Short EXV5" in packet.macro_report_text
    assert "HedgeRatio=1.0 : 1.0 Dollar-Neutral" in packet.macro_report_text
    assert "Entry=SOXX/EXV5 ratio > 4.20x" in packet.macro_report_text
    assert "SL=-3.5% spread divergence" in packet.macro_report_text
    assert "Target=+8.0% spread convergence" in packet.macro_report_text
    assert "MaxDD=-4.0% of risk budget" in packet.macro_report_text
    assert "SOXX Rev YoY = +28.0%" in packet.macro_report_text
    assert "[Conf=high]" in packet.macro_report_text

    # Macro grounded numeric facts in packet.verified_numeric_facts
    assert "macro_regime=Late-Cycle Expansion" in packet.verified_numeric_facts
    assert "macro_th_pe_ratio=14.8" in packet.verified_numeric_facts
    assert "macro_th_policy_spread_bps=-225.0" in packet.verified_numeric_facts
    assert "macro_th_foreign_net_mb=-1850.5" in packet.verified_numeric_facts
    assert "macro_th_gold_bar_sell_thb=43850" in packet.verified_numeric_facts

    # Equity composite waterfall, credit tax/after-tax Kd, and 3Y SG&A%/ROIC% in $Equity report text
    assert "Composite Waterfall:" in packet.equity_report_text
    assert "SolvencyGatePassed=True" in packet.equity_report_text
    assert "After-Tax Kd=4.15%" in packet.equity_report_text
    assert "EffTax=13.50%" in packet.equity_report_text
    assert "Year|Rev|GrossM|EBIT|R&D%|SG&A%|ROIC%|FCF|CapEx/D&A|Phase" in packet.equity_report_text
    assert "|4.40%|68.40%|" in packet.equity_report_text

