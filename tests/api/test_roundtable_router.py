"""Integration tests for Trinity Guru Roundtable REST & SSE Router."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest
from fastapi.testclient import TestClient

from api.db.adapters import SqliteRoundtableRepositoryAdapter
from api.dependencies import get_roundtable_service
from api.main import app
from application.roundtable.context_assembler import RoundtableContextAssembler
from application.roundtable.ports import (
    AudioSynthesisResult,
    ContextDataProvidersPort,
    RoundtableDebateEnginePort,
    RoundtableVerifierPort,
    ThesisUpdaterPort,
    VaultNoteWriterPort,
)
from application.roundtable.service import RoundtableApplicationService
from core.roundtable.models import FrozenContextPacket
from schemas.roundtable_schemas import (
    ActionPlaybookOption,
    ActionPlaybookSpec,
    ConsensusTensionAxis,
    DistilledDecisionPlaybook,
    PersonalAxisCheckItem,
    RoundtableTurnPayload,
)
from tools.roundtable.adapters import RoundtableSandboxCalcAdapter
from tools.roundtable.tts_adapter import GeminiTTSAdapter
from validators.roundtable_verifier import RoundtableConsistencyVerifier


class _FakeProviders(ContextDataProvidersPort):
    def __init__(self, *, include_nvda: bool = True, updated_price: str = "135.00") -> None:
        self.include_nvda = include_nvda
        self.updated_price = updated_price
        self.dispatched_tickers: List[str] = []

    def resolve_instrument(self, ticker: str) -> Dict[str, Any]:
        return {
            "market": "US",
            "sector": "Technology",
            "industry": "Semiconductors",
            "currency": "USD",
            "company_name": "NVIDIA Corporation" if ticker == "NVDA" else ticker,
            "current_price": self.updated_price,
        }

    def get_equity_detail(self, ticker: str) -> Optional[Dict[str, Any]]:
        if ticker == "NVDA" and self.include_nvda:
            return {
                "ticker": "NVDA",
                "company_name": "NVIDIA Corporation",
                "composite_score": 82,
                "evaluated_at": f"2026-11-10@{self.updated_price}",
                "quant_signals": {
                    "current_price": float(self.updated_price),
                    "mos_entry_price_today": 115.0,
                    "roic_pct": 68.5,
                    "fcf_yield_pct": 2.4,
                },
            }
        return None

    def get_equity_position_review(self, ticker: str, portfolio_id: str = "default") -> Optional[Dict[str, Any]]:
        return None

    def get_equity_news(self, ticker: str) -> Optional[Dict[str, Any]]:
        return None

    def list_equity_notes(self, ticker: str, days: int = 30) -> Optional[Dict[str, Any]]:
        return None

    def read_equity_note(self, relative_path: str) -> Optional[Dict[str, Any]]:
        return None

    def get_macro_dashboard(self) -> Optional[Dict[str, Any]]:
        return {"overall_regime": "Goldilocks Expansion"}

    def get_daily_digest(self) -> Optional[Dict[str, Any]]:
        return {"headline": "AI Capex keeps compounding"}

    def get_portfolio_state(self, portfolio_id: str = "default") -> Optional[Dict[str, Any]]:
        return {
            "total_value_thb": 1000000.0,
            "cash_pct": 18.0,
            "holdings": [
                {
                    "symbol": "NVDA",
                    "units": 10,
                    "avg_cost": 110.0,
                    "market_value_thb": 47000.0,
                    "unrealized_pnl_percent": 22.7,
                    "bucket_id": "growth",
                }
            ],
        }

    def get_confirmed_investment_axis(self, portfolio_id: str = "default") -> Optional[Dict[str, Any]]:
        return {
            "content_hash": "axis_v1",
            "role_models": ["Warren Buffett", "Charlie Munger"],
            "investment_horizon": "5-10 years",
            "risk_limits": "Max single stock 15%",
            "rebalance_frequency": "Quarterly",
            "non_actions": ["No leverage"],
        }

    def get_financial_context(self, portfolio_id: str = "default") -> Optional[Dict[str, Any]]:
        return None

    def get_bucket_policy(self, portfolio_id: str = "default") -> Optional[List[Dict[str, Any]]]:
        return None

    def get_active_thesis(self, ticker: str) -> Optional[Dict[str, Any]]:
        return None

    def dispatch_equity_research_if_needed(self, ticker: str) -> Optional[str]:
        self.dispatched_tickers.append(ticker)
        return f"job_{ticker.lower()}_001"


class _FakeDebateEngine(RoundtableDebateEnginePort):
    def __init__(self, delay_sec: float = 0.0) -> None:
        self.delay_sec = delay_sec
        self.calls: List[Tuple[str, str]] = []

    async def generate_guru_turn(
        self,
        *,
        speaker_id: str,
        round_stage: str,
        packet: FrozenContextPacket,
        prior_turns: Sequence[RoundtableTurnPayload],
        unspoken_speakers: Sequence[str],
        verifier: Optional[RoundtableVerifierPort] = None,
    ) -> RoundtableTurnPayload:
        if self.delay_sec > 0:
            await asyncio.sleep(self.delay_sec)
        self.calls.append((speaker_id, round_stage))
        stance_map = {
            "value_guru": "HOLD",
            "quality_guru": "OVERWEIGHT",
            "growth_guru": "BUY",
            "macro_tactical_guru": "OVERWEIGHT",
            "skeptic_guru": "UNDERWEIGHT",
        }
        replying_to = prior_turns[-1].speaker_id if prior_turns and round_stage == "rebuttal" else None
        return RoundtableTurnPayload(
            speaker_id=speaker_id,  # type: ignore[arg-type]
            round_stage=round_stage,  # type: ignore[arg-type]
            replying_to_speaker_id=replying_to,
            spoken_dialogue=f"มุมมอง {round_stage} จาก {speaker_id} สำหรับหุ้น {packet.primary_ticker or 'Global'} ครับ",
            evidence_bullets=[f"[FACT] Ref price {packet.instrument_anchor.current_price if packet.instrument_anchor else 'N/A'}"],
            stance_tag=stance_map.get(speaker_id, "HOLD"),  # type: ignore[arg-type]
            sizing_posture="Conservative Staged Dip" if round_stage == "rebuttal" else None,
        )

    async def generate_hermes_synthesis(
        self,
        *,
        packet: FrozenContextPacket,
        all_turns: Sequence[RoundtableTurnPayload],
        verifier: Optional[RoundtableVerifierPort] = None,
    ) -> Tuple[RoundtableTurnPayload, DistilledDecisionPlaybook]:
        closing = RoundtableTurnPayload(
            speaker_id="hermes_moderator",
            round_stage="synthesis",
            spoken_dialogue="สรุปผลการหารือรอบโต๊ะ แนะนำ Overweight แบบแบ่งไม้เข้าซื้อครับ",
            evidence_bullets=["[FACT] ประมวลผลจาก 5 กูรูและกรอบความเสี่ยงส่วนตัว"],
            stance_tag="OVERWEIGHT",
        )
        playbook = DistilledDecisionPlaybook(
            final_rating="OVERWEIGHT",
            action_stance="OVERWEIGHT",
            core_conviction_score=2,
            executive_verdict="พื้นฐานธุรกิจเด่น ให้ทยอยสะสมเฉพาะจุดแนวรับ",
            guru_stance_breakdown={
                "value_guru": "HOLD",
                "quality_guru": "OVERWEIGHT",
                "growth_guru": "BUY",
                "macro_tactical_guru": "OVERWEIGHT",
                "skeptic_guru": "UNDERWEIGHT",
            },
            consensus_points=["คุณภาพงบแกร่งและ ROIC สูง"],
            tension_axes=[
                ConsensusTensionAxis(
                    topic="Valuation vs AI Runway",
                    bull_side_speaker="growth_guru",
                    bull_argument="AI capex cycle เพิ่งเริ่มต้น",
                    bear_side_speaker="value_guru",
                    bear_argument="ราคาสูงกว่า MOS",
                )
            ],
            personal_axis_alignment=[
                PersonalAxisCheckItem(
                    dimension="Risk Limit",
                    status="aligned",
                    detail="น้ำหนักปัจจุบันยังไม่เกินเพดาน",
                )
            ],
            portfolio_context_summary=packet.portfolio_guard.summary_text,
            action_playbook=ActionPlaybookSpec(
                primary_action=ActionPlaybookOption(
                    option_id="p1",
                    label="ทยอยสะสมแบบแบ่งไม้",
                    speaker_source="hermes_moderator",
                    conviction_level=2,
                    action_steps=["แบ่งเข้า 2 ไม้"],
                ),
                alternatives=[
                    ActionPlaybookOption(
                        option_id="alt1",
                        label="รอแนวรับ Value",
                        speaker_source="value_guru",
                        conviction_level=1,
                        action_steps=["ตั้ง Alert แถว MOS $115"],
                    )
                ],
            ),
            kill_switches=["Gross Margin < 65%"],
        )
        return closing, playbook

    async def generate_interjection_reply(
        self,
        *,
        user_message: str,
        target_speaker_id: str,
        packet: FrozenContextPacket,
        history_turns: Sequence[RoundtableTurnPayload],
        sandbox_calc_output: Optional[str] = None,
        verifier: Optional[RoundtableVerifierPort] = None,
    ) -> RoundtableTurnPayload:
        snippet = f" โดยผลคำนวณ Sandbox คือ {sandbox_calc_output[:80]}" if sandbox_calc_output else ""
        return RoundtableTurnPayload(
            speaker_id=target_speaker_id,  # type: ignore[arg-type]
            round_stage="interjection_reply",
            replying_to_speaker_id="user",
            spoken_dialogue=f"ตอบคำถามแทรกถึง {target_speaker_id} เรียบร้อยครับ{snippet}",
            evidence_bullets=[f"[MATH] Sandbox verified: {bool(sandbox_calc_output)}"],
            stance_tag="OVERWEIGHT",
        )

    async def generate_reflection_lesson(
        self,
        *,
        previous_rating: str,
        raw_return_pct: Optional[str],
        alpha_pct: Optional[str],
        previous_verdict: str,
        ticker: str,
    ) -> str:
        return f"บทเรียน {ticker}: มติเดิม {previous_rating} ทำ alpha {alpha_pct}%"


class _FakeVaultWriter(VaultNoteWriterPort):
    def __init__(self) -> None:
        self.saved_notes: List[Dict[str, Any]] = []

    def save_roundtable_note(
        self,
        *,
        title: str,
        markdown_content: str,
        ticker: Optional[str],
        tags: Sequence[str],
    ) -> Dict[str, Any]:
        self.saved_notes.append({"title": title, "ticker": ticker, "tags": list(tags), "content": markdown_content})
        return {"ok": True, "vault_path": f"20_Stocks/{ticker or 'Notes'}/{title}.md"}


class _FakeThesisUpdater(ThesisUpdaterPort):
    def __init__(self) -> None:
        self.updates: List[Dict[str, Any]] = []

    def revise_thesis_from_roundtable(
        self,
        *,
        ticker: str,
        claim_statement: str,
        kill_switches: Sequence[str],
    ) -> Dict[str, Any]:
        self.updates.append({"ticker": ticker, "claim_statement": claim_statement, "kill_switches": list(kill_switches)})
        return {"claim_id": f"claim_{ticker.lower()}_99", "ticker": ticker}


def _build_test_service(
    tmp_path: Path,
    *,
    include_nvda: bool = True,
    delay_sec: float = 0.0,
) -> Tuple[RoundtableApplicationService, _FakeProviders, _FakeVaultWriter, _FakeThesisUpdater]:
    repo = SqliteRoundtableRepositoryAdapter()
    providers = _FakeProviders(include_nvda=include_nvda)
    assembler = RoundtableContextAssembler(providers=providers, repo=repo)
    engine = _FakeDebateEngine(delay_sec=delay_sec)
    tts = GeminiTTSAdapter(
        cache_dir=tmp_path / "rt_audio",
        tts_fetcher=lambda model, voice, text: b"\x01\x00" * 240,
    )
    vault_writer = _FakeVaultWriter()
    thesis_updater = _FakeThesisUpdater()
    svc = RoundtableApplicationService(
        repo=repo,
        assembler=assembler,
        engine=engine,
        tts=tts,
        verifier=RoundtableConsistencyVerifier(),
        sandbox_calc=RoundtableSandboxCalcAdapter(),
        vault_writer=vault_writer,
        thesis_updater=thesis_updater,
        providers=providers,
    )
    return svc, providers, vault_writer, thesis_updater


def test_roundtable_requires_auth(client: TestClient):
    resp = client.get("/api/roundtable/sessions")
    assert resp.status_code == 401


def test_roundtable_full_lifecycle_sse_audio_interject_sandbox_vault_and_thesis(
    authed_client: TestClient,
    tmp_path: Path,
):
    svc, providers, vault_writer, thesis_updater = _build_test_service(tmp_path)
    app.dependency_overrides[get_roundtable_service] = lambda: svc
    try:
        # 1. Create session with $NVDA & @Portfolio (balanced/concise 11-turn flow)
        create_resp = authed_client.post(
            "/api/roundtable/sessions",
            json={
                "agenda_prompt": "ประเมินโอกาสซื้อเพิ่ม $NVDA เทียบมูลค่าปัจจุบัน",
                "context_refs": ["$NVDA", "@Portfolio"],
                "debate_intensity": "balanced",
                "debate_depth": "concise",
            },
        )
        assert create_resp.status_code == 201
        detail = create_resp.json()
        session_id = detail["session"]["session_id"]
        assert detail["session"]["primary_ticker"] == "NVDA"
        assert detail["portfolio_attachment_state"] == "ATTACHED_HELD"
        assert set(detail["prioritized_speakers"]) == {"value_guru", "quality_guru"}
        assert detail["missing_prerequisite_tickers"] == []
        assert detail["context_drift_detected"] is False

        # 2. List sessions
        list_resp = authed_client.get("/api/roundtable/sessions")
        assert list_resp.status_code == 200
        assert any(s["session_id"] == session_id for s in list_resp.json())

        # 3. Stream SSE 2-Stage Debate + Hermes Distillation + Pipelined TTS
        stream_resp = authed_client.post(
            f"/api/roundtable/sessions/{session_id}/stream",
            json={"voice_enabled": True},
        )
        assert stream_resp.status_code == 200
        assert "text/event-stream" in stream_resp.headers["content-type"]
        body = stream_resp.text
        assert "event: session_meta" in body
        assert "event: speaker_started" in body
        assert "event: turn_completed" in body
        assert "event: audio_ready" in body
        assert "event: playbook_updated" in body
        assert "event: done" in body

        # 4. Verify persisted turns (5 opening + 5 rebuttal + 1 synthesis = 11 turns) & audio retrieval
        detail_after = authed_client.get(f"/api/roundtable/sessions/{session_id}")
        assert detail_after.status_code == 200
        d_json = detail_after.json()
        assert d_json["session"]["status"] == "completed"
        assert d_json["session"]["final_rating"] == "OVERWEIGHT"
        assert len(d_json["turns"]) == 11
        assert d_json["decision_playbook"] is not None

        first_audio_key = next(t["audio_cache_key"] for t in d_json["turns"] if t["audio_cache_key"])
        wav_resp = authed_client.get(f"/api/roundtable/audio/{first_audio_key}")
        assert wav_resp.status_code == 200
        assert wav_resp.headers["content-type"].startswith("audio/wav")
        assert wav_resp.content[:4] == b"RIFF"

        # Path traversal / unknown audio key must 404
        assert authed_client.get("/api/roundtable/audio/nonexistent_key_999").status_code == 404

        # 5. Interject with @Value + Sandbox Python Decimal Calculation ("ซื้อเพิ่ม 5 หุ้น")
        interject_resp = authed_client.post(
            f"/api/roundtable/sessions/{session_id}/interject",
            json={
                "message": "@Value ถ้าซื้อเพิ่มอีก 5 หุ้น ต้นทุนเฉลี่ยใหม่จะเป็นเท่าไหร่?",
                "voice_enabled": True,
                "run_sandbox_math": True,
            },
        )
        assert interject_resp.status_code == 200
        ij = interject_resp.json()
        assert ij["user_turn"]["speaker_id"] == "user"
        assert ij["reply_turn"]["speaker_id"] == "value_guru"
        assert len(ij["tool_calls"]) == 1
        assert ij["tool_calls"][0]["tool_name"] == "python_sandbox_decimal_calc"
        assert ij["tool_calls"][0]["status"] == "completed"
        assert ij["tool_calls"][0]["detail_json"]["blended_avg_cost"] == "118.33"

        # 6. Fast history retrieval skips drift recomputation when check_drift=false,
        # while default check_drift=true detects price changes
        providers.updated_price = "142.50"
        fast_hist_resp = authed_client.get(f"/api/roundtable/sessions/{session_id}?check_drift=false")
        assert fast_hist_resp.status_code == 200
        fast_hist_json = fast_hist_resp.json()
        assert fast_hist_json["context_drift_detected"] is False
        assert len(fast_hist_json["turns"]) > 0

        drift_resp = authed_client.get(f"/api/roundtable/sessions/{session_id}")
        assert drift_resp.status_code == 200
        assert drift_resp.json()["context_drift_detected"] is True

        # 7. Save to Vault (+ update thesis)
        vault_resp = authed_client.post(
            f"/api/roundtable/sessions/{session_id}/save-to-vault",
            json={"also_update_thesis": True},
        )
        assert vault_resp.status_code == 200
        v_json = vault_resp.json()
        assert v_json["ok"] is True
        assert v_json["thesis_claim_id"] == "claim_nvda_99"
        assert len(vault_writer.saved_notes) == 1
        assert len(thesis_updater.updates) == 1

        # 8. Standalone Update Thesis endpoint
        thesis_resp = authed_client.post(
            f"/api/roundtable/sessions/{session_id}/update-thesis",
            json={"claim_statement": "Custom claim from Roundtable"},
        )
        assert thesis_resp.status_code == 200
        assert thesis_resp.json()["claim_id"] == "claim_nvda_99"
    finally:
        app.dependency_overrides.clear()


def test_roundtable_prerequisite_guard_and_dispatch(authed_client: TestClient, tmp_path: Path):
    svc, providers, _, _ = _build_test_service(tmp_path, include_nvda=False)
    app.dependency_overrides[get_roundtable_service] = lambda: svc
    try:
        create_resp = authed_client.post(
            "/api/roundtable/sessions",
            json={"agenda_prompt": "วิเคราะห์หุ้น $NVDA", "context_refs": ["$NVDA"]},
        )
        assert create_resp.status_code == 201
        assert create_resp.json()["missing_prerequisite_tickers"] == ["NVDA"]

        disp_resp = authed_client.post("/api/roundtable/prerequisites/NVDA/dispatch")
        assert disp_resp.status_code == 200
        assert disp_resp.json()["ok"] is True
        assert disp_resp.json()["job_id"] == "job_nvda_001"
        assert providers.dispatched_tickers == ["NVDA"]
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_roundtable_rejects_a_second_concurrent_stream(api_env, tmp_path: Path):
    svc, _, _, _ = _build_test_service(tmp_path, delay_sec=0.01)
    from schemas.roundtable_schemas import CreateRoundtableSessionRequest

    created = await svc.create_session(
        CreateRoundtableSessionRequest(
            agenda_prompt="Check one stream per session for $NVDA",
            context_refs=["$NVDA", "@Portfolio"],
        )
    )
    sid = created.session.session_id

    async def _consume():
        return [event async for event in svc.stream_debate(sid, voice_enabled=False)]

    first_events, second_events = await asyncio.gather(_consume(), _consume())
    all_events = first_events + second_events
    errors = [payload for name, payload in all_events if name == "error"]
    assert len(errors) == 1
    assert "already active" in errors[0]["error"]
    assert sum(name == "done" for name, _ in all_events) == 1

    engine = svc._engine
    opening_speakers = [speaker for speaker, stage in engine.calls if stage == "opening"]
    assert len(opening_speakers) == len(set(opening_speakers))
    detail = svc.get_session_detail(sid)
    assert detail is not None
    assert sum(turn.round_stage == "synthesis" for turn in detail.turns) == 1
    seqs = [turn.seq for turn in detail.turns]
    assert seqs == sorted(seqs)
    assert len(seqs) == len(set(seqs))


def test_roundtable_dependency_provider_serializes_cold_start_and_streams_once(api_env, tmp_path: Path, monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    import api.dependencies as deps
    import api.db.adapters as db_adapters
    from schemas.roundtable_schemas import CreateRoundtableSessionRequest

    monkeypatch.setattr(deps, "_ROUNDTABLE_SERVICE", None)
    monkeypatch.setattr(deps, "_ROUNDTABLE_CACHE_KEY", None)
    monkeypatch.setattr(deps, "_get_vault_path", lambda: tmp_path / "vault")

    original_repo_factory = db_adapters.SqliteRoundtableRepositoryAdapter
    first_repo_entered = threading.Event()
    release_first_repo = threading.Event()
    second_key_ready = threading.Event()
    second_provider_done = threading.Event()
    state_lock = threading.Lock()
    repo_calls = 0
    key_calls = 0

    def _tracked_vault_path():
        nonlocal key_calls
        with state_lock:
            key_calls += 1
            if key_calls == 2:
                second_key_ready.set()
        return tmp_path / "vault"

    def _slow_repo_factory(*args, **kwargs):
        nonlocal repo_calls
        with state_lock:
            repo_calls += 1
            call_number = repo_calls
        if call_number == 1:
            first_repo_entered.set()
            assert release_first_repo.wait(timeout=5)
        return original_repo_factory(*args, **kwargs)

    monkeypatch.setattr(deps, "_get_vault_path", _tracked_vault_path)
    monkeypatch.setattr(db_adapters, "SqliteRoundtableRepositoryAdapter", _slow_repo_factory)

    def _get_service(mark_done=False):
        service = deps.get_roundtable_service(request=None)
        if mark_done:
            second_provider_done.set()
        return service

    with ThreadPoolExecutor(max_workers=2) as pool:
        first_future = pool.submit(_get_service)
        assert first_repo_entered.wait(timeout=5)
        second_future = pool.submit(_get_service, True)
        assert second_key_ready.wait(timeout=5)
        assert not second_provider_done.wait(timeout=0.05)
        release_first_repo.set()
        first_service = first_future.result(timeout=10)
        second_service = second_future.result(timeout=10)

    assert repo_calls == 1
    assert first_service is second_service

    # The provider-returned service must keep one per-session stream lock and one debate creator.
    engine = _FakeDebateEngine(delay_sec=0.01)
    first_service._engine = engine
    created = asyncio.run(
        first_service.create_session(
            CreateRoundtableSessionRequest(agenda_prompt="Check cold-start stream coordination")
        )
    )
    session_id = created.session.session_id
    assert first_service._get_stream_lock(session_id) is second_service._get_stream_lock(session_id)

    async def _consume(service):
        return [event async for event in service.stream_debate(session_id, voice_enabled=False)]

    async def _run_streams():
        return await asyncio.gather(_consume(first_service), _consume(second_service))

    first_events, second_events = asyncio.run(_run_streams())
    all_events = first_events + second_events
    assert sum(name == "done" for name, _ in all_events) == 1
    errors = [payload for name, payload in all_events if name == "error"]
    assert len(errors) == 1
    assert "already active" in errors[0]["error"]
    assert len(engine.calls) == len(set(engine.calls))


@pytest.mark.asyncio
async def test_roundtable_context_refresh_preserves_session_entry_price(api_env, tmp_path: Path):
    svc, providers, _, _ = _build_test_service(tmp_path)
    from schemas.roundtable_schemas import CreateRoundtableSessionRequest, InterjectRequest

    created = await svc.create_session(
        CreateRoundtableSessionRequest(
            agenda_prompt="Record the entry price for $NVDA",
            context_refs=["$NVDA", "@Portfolio"],
        )
    )
    sid = created.session.session_id
    original = svc._repo.get_session(sid)
    assert original is not None
    assert original.entry_price_at_session == "135.00"

    providers.updated_price = "150.00"
    async for _event in svc.stream_debate(sid, voice_enabled=False, refresh_context_on_drift=True):
        pass
    after_stream_refresh = svc._repo.get_session(sid)
    assert after_stream_refresh is not None
    assert after_stream_refresh.entry_price_at_session == "135.00"
    assert svc._parse_packet(after_stream_refresh).instrument_anchor.current_price == "150.00"

    providers.updated_price = "165.00"
    await svc.handle_interjection(
        sid,
        InterjectRequest(
            message="Refresh the context and answer @Value",
            voice_enabled=False,
            refresh_context=True,
        ),
    )
    after_interjection_refresh = svc._repo.get_session(sid)
    assert after_interjection_refresh is not None
    assert after_interjection_refresh.entry_price_at_session == "135.00"
    assert svc._parse_packet(after_interjection_refresh).instrument_anchor.current_price == "165.00"


@pytest.mark.asyncio
async def test_roundtable_mid_stream_interjection_cancels_cleanly(api_env, tmp_path: Path):
    svc, _, _, _ = _build_test_service(tmp_path, delay_sec=0.05)
    from schemas.roundtable_schemas import CreateRoundtableSessionRequest, InterjectRequest

    created = await svc.create_session(
        CreateRoundtableSessionRequest(
            agenda_prompt="ทดสอบยกมือแทรกกลาง stream $NVDA",
            context_refs=["$NVDA", "@Portfolio"],
        )
    )
    sid = created.session.session_id
    emitted: List[str] = []

    async def _consume_stream():
        async for ev_name, _ in svc.stream_debate(sid, voice_enabled=False):
            emitted.append(ev_name)
            if ev_name == "turn_completed":
                # Trigger user raise-hand interjection mid-stream after turn 1 completes
                await svc.handle_interjection(
                    sid,
                    InterjectRequest(
                        message="@Skeptic ความเสี่ยงคู่แข่ง AMD น่ากังวลไหม",
                        voice_enabled=False,
                    ),
                )

    await _consume_stream()
    assert "paused" in emitted
    assert "done" not in emitted
    detail = svc.get_session_detail(sid)
    assert detail is not None
    stages = [t.round_stage for t in detail.turns]
    assert "user_interjection" in stages
    assert "interjection_reply" in stages
    # Sequential seq monotonicity preserved across stream + interjection
    seqs = [t.seq for t in detail.turns]
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == len(seqs)


def test_roundtable_guru_profiles_voice_preview_and_custom_panel_session(
    authed_client: TestClient,
    tmp_path: Path,
):
    repo = SqliteRoundtableRepositoryAdapter()
    providers = _FakeProviders(include_nvda=True)
    assembler = RoundtableContextAssembler(providers=providers, repo=repo)
    engine = _FakeDebateEngine()
    tts_voice_calls: List[Tuple[str, str]] = []

    def _recording_tts_fetcher(model: str, voice: str, text: str) -> bytes:
        tts_voice_calls.append((voice, text))
        return b"\x01\x00" * 240

    tts = GeminiTTSAdapter(cache_dir=tmp_path / "rt_custom_audio", tts_fetcher=_recording_tts_fetcher)
    vault_writer = _FakeVaultWriter()
    svc = RoundtableApplicationService(
        repo=repo,
        assembler=assembler,
        engine=engine,
        tts=tts,
        verifier=RoundtableConsistencyVerifier(),
        sandbox_calc=RoundtableSandboxCalcAdapter(),
        vault_writer=vault_writer,
        providers=providers,
    )
    app.dependency_overrides[get_roundtable_service] = lambda: svc
    try:
        # 1. GET /guru-profiles catalog returns 6 personas, 12 voices, 18 presets with UI context & rubrics
        cat_resp = authed_client.get("/api/roundtable/guru-profiles")
        assert cat_resp.status_code == 200
        cat_json = cat_resp.json()
        assert len(cat_json["personas"]) == 6
        assert len(cat_json["voices"]) == 12
        assert len(cat_json["presets"]) == 18
        for persona in cat_json["personas"]:
            assert persona["role_lens_label"]
            assert persona["focus_summary"]
            assert len(persona["primary_evidence_labels"]) >= 2
            assert persona["analytical_rubric"]
        for preset in cat_json["presets"]:
            assert preset["role_lens_label"]
            assert preset["focus_summary"]
            assert len(preset["primary_evidence_labels"]) >= 2
            assert preset["analytical_rubric"]

        # 2. PUT /guru-profiles rejects <2 enabled panel gurus or unknown voice with 422
        bad_quorum = authed_client.put(
            "/api/roundtable/guru-profiles",
            json={
                "profiles": [
                    {"speaker_id": "value_guru", "enabled": False},
                    {"speaker_id": "quality_guru", "enabled": False},
                    {"speaker_id": "growth_guru", "enabled": False},
                    {"speaker_id": "macro_tactical_guru", "enabled": False},
                ]
            },
        )
        assert bad_quorum.status_code == 422

        bad_voice = authed_client.put(
            "/api/roundtable/guru-profiles",
            json={"profiles": [{"speaker_id": "value_guru", "voice_name": "InvalidVoice"}]},
        )
        assert bad_voice.status_code == 422

        # 3. PUT /guru-profiles persists customized seat cleanly
        put_resp = authed_client.put(
            "/api/roundtable/guru-profiles",
            json={
                "profiles": [
                    {
                        "speaker_id": "value_guru",
                        "display_name": "Deep Value (Graham)",
                        "short_name": "Graham",
                        "voice_name": "Zephyr",
                        "pitch": 1.05,
                        "rate": 1.04,
                        "preset_id": "value_graham_deep",
                    }
                ]
            },
        )
        assert put_resp.status_code == 200
        graham_dto = next(p for p in put_resp.json()["personas"] if p["speaker_id"] == "value_guru")
        assert graham_dto["short_name"] == "Graham"
        assert graham_dto["voice_name"] == "Zephyr"
        assert "Current Ratio >= 1.5x" in graham_dto["analytical_rubric"]

        # 4. POST /voices/preview synthesizes audio with requested voice & serves WAV
        prev_resp = authed_client.post(
            "/api/roundtable/voices/preview",
            json={
                "speaker_id": "value_guru",
                "voice_name": "Zephyr",
                "sample_text": "สวัสดีจากเสียงตัวอย่างครับ",
                "pitch": 1.05,
                "rate": 1.04,
            },
        )
        assert prev_resp.status_code == 200
        prev_json = prev_resp.json()
        assert prev_json["audio_status"] == "ready"
        assert prev_json["audio_cache_key"]
        assert ("Zephyr", "สวัสดีจากเสียงตัวอย่างครับ") in tts_voice_calls
        assert authed_client.get(f"/api/roundtable/audio/{prev_json['audio_cache_key']}").status_code == 200

        # 5. Create session with per-session overrides (2 panel gurus + custom voice Iapetus)
        sess_resp = authed_client.post(
            "/api/roundtable/sessions",
            json={
                "agenda_prompt": "ทดสอบสภากูรูแบบเลือกเองสำหรับ $NVDA",
                "context_refs": ["$NVDA", "@Portfolio"],
                "debate_intensity": "balanced",
                "debate_depth": "concise",
                "persona_overrides": [
                    {"speaker_id": "value_guru", "voice_name": "Iapetus", "short_name": "Graham"},
                    {"speaker_id": "growth_guru", "enabled": False},
                    {"speaker_id": "macro_tactical_guru", "enabled": False},
                    {"speaker_id": "skeptic_guru", "enabled": False},
                ],
            },
        )
        assert sess_resp.status_code == 201
        sess_json = sess_resp.json()
        sid = sess_json["session"]["session_id"]

        # Stream should execute only 2 enabled panel gurus * 2 rounds + 1 synthesis = 5 turns
        stream_resp = authed_client.post(
            f"/api/roundtable/sessions/{sid}/stream",
            json={"voice_enabled": True},
        )
        assert stream_resp.status_code == 200
        assert '"personas"' in stream_resp.text
        detail_after = authed_client.get(f"/api/roundtable/sessions/{sid}").json()
        assert len(detail_after["turns"]) == 5
        assert {t["speaker_id"] for t in detail_after["turns"]} == {
            "value_guru",
            "quality_guru",
            "hermes_moderator",
        }

        # 6. Dynamic @Graham interjection routes to value_guru with session voice Iapetus
        tts_voice_calls.clear()
        ij_resp = authed_client.post(
            f"/api/roundtable/sessions/{sid}/interject",
            json={"message": "@Graham แนวรับ MoS อยู่ตรงไหน", "voice_enabled": True},
        )
        assert ij_resp.status_code == 200
        assert ij_resp.json()["reply_turn"]["speaker_id"] == "value_guru"
        assert any(voice == "Iapetus" for voice, _ in tts_voice_calls)

        # 7. Save to Vault lists Active Panel with custom display_name
        vault_resp = authed_client.post(f"/api/roundtable/sessions/{sid}/save-to-vault", json={})
        assert vault_resp.status_code == 200
        assert "Active Panel:" in vault_writer.saved_notes[0]["content"]
        assert "Deep Value (Graham)" in vault_writer.saved_notes[0]["content"]

        # 8. Reset single seat then all
        reset_resp = authed_client.post("/api/roundtable/guru-profiles/reset", json={"speaker_id": "value_guru"})
        assert reset_resp.status_code == 200
        reset_val = next(p for p in reset_resp.json()["personas"] if p["speaker_id"] == "value_guru")
        assert reset_val["short_name"] == "Value"
        assert reset_val["voice_name"] == "Charon"
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_roundtable_composition_portfolio_reader_fixes_silent_cash_bug(api_env, monkeypatch):
    """Verify composition root computes real cash_thb/cash_pct from Cash holdings (not missing summary.cash_thb)."""
    import api.dependencies as deps
    from schemas.roundtable_schemas import CreateRoundtableSessionRequest
    from tools.portfolio.domain.models import AllocationTarget, Holding, PortfolioState, Summary

    domain_state = PortfolioState(
        last_updated="2026-11-11T10:00:00",
        fx_rates={"USDTHB": 35.0},
        summary=Summary(
            total_value_thb=1000000.0,
            total_cost_basis_thb=800000.0,
            total_unrealized_profit=120000.0,
        ),
        allocation_targets=[
            AllocationTarget(bucket_id="growth", name="Growth Engine", target_percent=70.0),
            AllocationTarget(bucket_id="cash", name="Cash", target_percent=20.0),
        ],
        holdings=[
            Holding(symbol="CASH_THB", asset_type="Cash", units=150000.0, market_value_thb=150000.0),
            Holding(symbol="CASH_USD", asset_type="Cash", units=1000.0, market_value_thb=35000.0),
            Holding(
                symbol="NVDA",
                asset_type="US Stock",
                units=10.0,
                avg_cost_usd=110.0,
                market_value_thb=815000.0,
                unrealized_pnl_percent=18.5,
                bucket_id="growth",
            ),
        ],
    )

    class _StubPortSvc:
        def get_structured_portfolio_state(self, portfolio_id: str = "default", refresh_prices: bool = False):
            return domain_state

    class _StubEqSvc:
        def get_detail(self, ticker: str):
            if ticker == "NVDA":
                return {"ticker": "NVDA", "company_name": "NVIDIA Corporation", "quant_signals": {"raw_analysis_price": 138.0}}
            return None

        def get_news(self, ticker: str):
            return None

        def list_notes(self, ticker: str, days: int = 30):
            return None

    class _StubResolver:
        def resolve(self, ticker: str):
            class _Res:
                raw_symbol = ticker
                provider_symbol = ticker
                asset_class = "EQUITY"
                market = "US"
                confidence = 1.0
            return _Res()

    monkeypatch.setattr(deps, "get_portfolio_service", lambda: _StubPortSvc())
    monkeypatch.setattr(deps, "get_equity_query_service", lambda: _StubEqSvc())
    monkeypatch.setattr(deps, "get_asset_resolver", lambda: _StubResolver())

    svc = deps.get_roundtable_service(request=None)
    created = await svc.create_session(
        CreateRoundtableSessionRequest(
            agenda_prompt="ตรวจสัดส่วนเงินสดและ $NVDA ในพอร์ต",
            context_refs=["$NVDA", "@Portfolio"],
        )
    )
    sess = svc._repo.get_session(created.session.session_id)
    assert sess is not None
    packet_dict = json.loads(sess.frozen_packet_json)
    pg = packet_dict["portfolio_guard"]
    assert pg["portfolio_cash_pct"] == "18.50"
    assert pg["attachment_state"] == "ATTACHED_HELD"
    assert "[PORTFOLIO OVERVIEW & ALLOCATION DETAIL]" in pg["summary_text"]
    assert "THB 150000.00 + USD eq 35000.00 THB" in pg["summary_text"]
    assert "Growth Engine [growth]: จริง 81.50% vs เป้า 70.00%" in pg["summary_text"]


@pytest.mark.asyncio
async def test_roundtable_composition_wires_watchlist_per_equity_and_goals_under_guard(api_env, monkeypatch):
    """Verify api/dependencies wires get_structured_watchlist per $Equity (even without @Portfolio) and Goals strictly under @Portfolio."""
    import api.dependencies as deps
    from schemas.roundtable_schemas import CreateRoundtableSessionRequest
    from tools.portfolio.domain.models import (
        AllocationTarget,
        Holding,
        PortfolioState,
        Summary,
        WatchlistItem,
        WatchlistState,
    )

    domain_state = PortfolioState(
        last_updated="2026-11-12T08:00:00",
        fx_rates={"USDTHB": 35.0},
        summary=Summary(total_value_thb=1000000.0, total_cost_basis_thb=900000.0, total_unrealized_profit=100000.0),
        allocation_targets=[AllocationTarget(bucket_id="growth", name="Growth", target_percent=70.0)],
        holdings=[Holding(symbol="CASH_THB", asset_type="Cash", units=1000000.0, market_value_thb=1000000.0)],
    )
    watchlist_state = WatchlistState(
        last_updated="2026-11-12T08:00:00",
        items=[
            WatchlistItem(
                symbol="NVDA",
                asset_type="US Stock",
                target_price=125.0,
                target_currency="USD",
                target_price_basis="MoS",
                suggested_bucket_id="growth",
                notes="Buy AI dip",
                added_date="2026-11-01",
            ),
            WatchlistItem(
                symbol="AAPL",
                asset_type="US Stock",
                target_price=180.0,
                target_currency="USD",
                notes="Unrelated symbol",
                added_date="2026-11-01",
            ),
        ],
    )
    goals_list = [
        {
            "name": "Financial Independence",
            "goal_type": "nav_target",
            "target_amount_thb": 10000000.0,
            "current_amount_thb": 1000000.0,
            "progress_pct": 10.0,
            "deadline": "2036-12-31",
            "deadline_days_left": 3700,
            "priority_tier": "needs",
            "monthly_savings_thb": 30000.0,
            "required_return_pct": 15.2,
            "tension_level": "moderate",
        }
    ]

    class _StubPortSvc:
        def get_structured_portfolio_state(self, portfolio_id: str = "default", refresh_prices: bool = False):
            return domain_state

        def get_structured_watchlist(self, portfolio_id: str = "default"):
            return watchlist_state

        def get_structured_goals(self, portfolio_id: str = "default"):
            return goals_list

    class _StubEqSvc:
        def get_detail(self, ticker: str):
            if ticker == "NVDA":
                return {"ticker": "NVDA", "company_name": "NVIDIA Corporation", "quant_signals": {"raw_analysis_price": 138.0}}
            return None

        def get_news(self, ticker: str):
            return None

        def list_notes(self, ticker: str, days: int = 30):
            return None

    class _StubResolver:
        def resolve(self, ticker: str):
            class _Res:
                raw_symbol = ticker
                provider_symbol = ticker
                asset_class = "EQUITY"
                market = "US"
                confidence = 1.0
            return _Res()

    monkeypatch.setattr(deps, "get_portfolio_service", lambda: _StubPortSvc())
    monkeypatch.setattr(deps, "get_equity_query_service", lambda: _StubEqSvc())
    monkeypatch.setattr(deps, "get_asset_resolver", lambda: _StubResolver())

    svc = deps.get_roundtable_service(request=None)

    # 1. Without @Portfolio -> NVDA Watchlist shown under $Equity; Goals/Cash/AAPL strictly hidden
    s1 = await svc.create_session(
        CreateRoundtableSessionRequest(
            agenda_prompt="ดูพื้นฐาน $NVDA",
            context_refs=["$NVDA"],
        )
    )
    row1 = svc._repo.get_session(s1.session.session_id)
    assert row1 is not None
    p1 = json.loads(row1.frozen_packet_json)
    assert "[User Watchlist]: $NVDA (US Stock) | เป้าซื้อ=125.00 USD | Basis=MoS | Bucket=growth | Notes=Buy AI dip" in p1["equity_report_text"]
    assert "AAPL" not in p1["equity_report_text"]
    assert p1["portfolio_guard"]["attachment_state"] == "NOT_ATTACHED_DO_NOT_ASSUME"
    assert "Financial Independence" not in p1["portfolio_guard"]["summary_text"]
    assert "Financial Independence" not in p1["equity_report_text"]

    # 2. With @Portfolio -> Watchlist + Portfolio Goals Progress render in Portfolio Guard
    s2 = await svc.create_session(
        CreateRoundtableSessionRequest(
            agenda_prompt="ดู $NVDA เทียบพอร์ตและเป้าหมาย",
            context_refs=["$NVDA", "@Portfolio"],
        )
    )
    row2 = svc._repo.get_session(s2.session.session_id)
    assert row2 is not None
    p2 = json.loads(row2.frozen_packet_json)
    pg2 = p2["portfolio_guard"]
    assert pg2["attachment_state"] == "ATTACHED_FLAT_BOOK"
    assert "กำลังติดตามใน Watchlist (เป้าซื้อ=125.00 USD" in pg2["summary_text"]
    assert "Financial Independence [nav_target | needs]" in pg2["summary_text"]
    assert "Req CAGR: 15.20% | Tension: moderate" in pg2["summary_text"]
    assert "Portfolio Watchlist" in pg2["summary_text"]


@pytest.mark.asyncio
async def test_roundtable_composition_portfolio_reader_fixes_thai_fund_currency_labels(api_env, monkeypatch):
    """WP-S4: Thai mutual funds with avg_cost_thb label (Fund/THB) even when current_price_usd is populated."""
    import api.dependencies as deps
    from schemas.roundtable_schemas import CreateRoundtableSessionRequest
    from tools.portfolio.domain.models import AllocationTarget, Holding, PortfolioState, Summary

    domain_state = PortfolioState(
        last_updated="2026-11-12T08:00:00",
        fx_rates={"USDTHB": 35.0},
        summary=Summary(total_value_thb=1000000.0, total_cost_basis_thb=900000.0, total_unrealized_profit=100000.0),
        allocation_targets=[AllocationTarget(bucket_id="core", name="Core", target_percent=80.0)],
        holdings=[
            Holding(
                symbol="SCBS&P500E",
                asset_type="Fund",
                units=2500.0,
                avg_cost_thb=18.5,
                avg_cost_usd=None,
                current_price_usd=0.55,
                market_value_thb=50000.0,
                unrealized_pnl_percent=8.1,
                bucket_id="core",
            ),
            Holding(
                symbol="GOLD-99.99",
                asset_type="Gold",
                units=1.0,
                avg_cost_usd=2650.0,
                avg_cost_thb=None,
                current_price_usd=2700.0,
                market_value_thb=94500.0,
                unrealized_pnl_percent=1.9,
                bucket_id="core",
            ),
            Holding(symbol="CASH_THB", asset_type="Cash", units=855500.0, market_value_thb=855500.0),
        ],
    )

    class _StubPortSvc:
        def get_structured_portfolio_state(self, portfolio_id: str = "default", refresh_prices: bool = False):
            return domain_state

        def get_structured_watchlist(self, portfolio_id: str = "default"):
            return None

        def get_structured_goals(self, portfolio_id: str = "default"):
            return []

    class _StubEqSvc:
        def get_detail(self, ticker: str):
            return {"ticker": "NVDA", "company_name": "NVIDIA", "quant_signals": {"raw_analysis_price": 138.0}}

        def get_news(self, ticker: str):
            return None

        def list_notes(self, ticker: str, days: int = 30):
            return None

        def read_note(self, relative_path: str):
            return None

    class _StubResolver:
        def resolve(self, ticker: str):
            class _Res:
                raw_symbol = ticker
                provider_symbol = ticker
                asset_class = "EQUITY"
                market = "US"
                confidence = 1.0
            return _Res()

    monkeypatch.setattr(deps, "get_portfolio_service", lambda: _StubPortSvc())
    monkeypatch.setattr(deps, "get_equity_query_service", lambda: _StubEqSvc())
    monkeypatch.setattr(deps, "get_asset_resolver", lambda: _StubResolver())

    svc = deps.get_roundtable_service(request=None)
    created = await svc.create_session(
        CreateRoundtableSessionRequest(
            agenda_prompt="ตรวจสกุลเงินกองทุนไทยกับทองในพอร์ตเทียบ $NVDA",
            context_refs=["$NVDA", "@Portfolio"],
        )
    )
    row = svc._repo.get_session(created.session.session_id)
    assert row is not None
    pg = json.loads(row.frozen_packet_json)["portfolio_guard"]
    assert "SCBS&P500E|Fund/THB|" in pg["summary_text"]
    assert "SCBS&P500E|Fund/USD" not in pg["summary_text"]
    assert "GOLD-99.99|Gold/USD|" in pg["summary_text"]


@pytest.mark.asyncio
async def test_roundtable_composition_wires_advisory_insider_analyst_exceptions_and_investor_essence(
    api_env, monkeypatch
):
    """WP-S100%-P2 composition root: SQLite insider/analyst/exceptions/essence + PortfolioAdvisory + 3-state guard."""
    from contextlib import closing
    from datetime import datetime, timedelta, timezone

    import api.dependencies as deps
    from api.db.adapters import SqliteThesisExceptionRepositoryAdapter
    from api.db.connection import get_connection
    from api.db.repositories import cache_repository as cache_repo_dao
    from schemas.roundtable_schemas import CreateRoundtableSessionRequest
    from tools.portfolio.domain.models import AllocationTarget, Holding, PortfolioState, Summary

    vault_dir = api_env / "vault"
    vault_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("SECTOR_ROTATION_RUNTIME_DIR", str(api_env / "sector_rotation"))
    monkeypatch.setenv("FINNOMENA_CACHE_DIR", str(api_env / "finnomena_cache"))
    monkeypatch.setattr(deps, "_get_vault_path", lambda: vault_dir)
    monkeypatch.setattr(deps, "_ROUNDTABLE_SERVICE", None)

    recent_day = (datetime.now(timezone.utc) - timedelta(days=10)).strftime("%Y-%m-%d")
    stale_synced_ts = (datetime.now(timezone.utc) - timedelta(days=9)).timestamp()

    with closing(get_connection()) as conn:
        # 1. Analyst Context Cache (stale synced_at + synthetic entry filtered out + upcoming None eps)
        cache_repo_dao.upsert_analyst_context_cache(
            conn,
            "NVDA",
            {
                "provider_symbol": "NVDA",
                "market": "US",
                "currency": "USD",
                "exchange_tz": "America/New_York",
                "target_mean": 165.0,
                "target_high": 200.0,
                "target_low": 120.0,
                "num_analysts": 42,
                "next_earnings_date": "2026-11-25",
                "earnings_history": [
                    {"date_str": "2026-08-19", "eps_estimate": 0.70, "eps_actual": 0.75, "surprise_pct": 7.1},
                    {"date_str": "2026-11-25", "eps_estimate": 0.80, "eps_actual": None},
                    {"date_str": "2026-01-01", "note": "synthetic test row", "eps_estimate": 99.0, "eps_actual": 99.0},
                ],
                "source_as_of": "2026-11-01",
                "data_status": "ok",
                "synced_at": stale_synced_ts,
            },
        )

        # 2. Form4 + yfinance insider ledger rows (yf_ $0 P->A reclass + paired M/S)
        conn.execute(
            """
            INSERT INTO sec_form4_raw_ledger (
                accession_number, issuer_cik, ticker, filing_url, filed_at,
                reporting_owner_name, officer_title, raw_xml_payload, created_at
            ) VALUES
            ('yf_nvda_001', '0001045810', 'NVDA', 'https://yf.example/1', ?, 'Jane Doe', 'VP', NULL, ?),
            ('0001045810-26-000012', '0001045810', 'NVDA', 'https://sec.example/12', ?, 'Jensen Huang', 'CEO', '<ownershipDocument/>', ?)
            """,
            (recent_day, stale_synced_ts, recent_day, stale_synced_ts),
        )
        conn.execute(
            """
            INSERT INTO sec_insider_transactions (
                transaction_id, accession_number, ticker, transaction_date,
                transaction_code, shares, price_per_share, acquired_or_disposed, created_at
            ) VALUES
            ('tx_yf_zero', 'yf_nvda_001', 'NVDA', ?, 'P', 400.0, 0.0, 'A', ?),
            ('tx_sec_m', '0001045810-26-000012', 'NVDA', ?, 'M', 1000.0, 0.0, 'A', ?),
            ('tx_sec_s', '0001045810-26-000012', 'NVDA', ?, 'S', 1000.0, 140.0, 'D', ?)
            """,
            (recent_day, stale_synced_ts, recent_day, stale_synced_ts, recent_day, stale_synced_ts),
        )

        # 3. Confirmed Investor Essence snapshot in investor_confirmed_refs
        essence_payload = {
            "statement": "เน้นลงทุนธุรกิจที่มี Moat แข็งแกร่งและ ROIC ยั่งยืน 10 ปี",
            "accepted_claims": [
                {"effective_text": "หลีกเลี่ยงบริษัทที่ใช้หนี้เกินตัว", "source_kind": "user", "fit_rating": "HIGH"}
            ],
            "unresolved_topics": ["เพดานเงินสดสำรองช่วงดอกเบี้ยขาลง"],
            "confirmed_at_iso": "2026-11-10T08:00:00Z",
            "content_hash": "ess_hash_v1",
        }
        conn.execute(
            """
            INSERT INTO investor_confirmed_refs (
                pointer_key, scope, portfolio_id, ref_type, artifact_ref_json, snapshot_json, revision, confirmed_at
            ) VALUES (?, 'workspace', NULL, 'investor_essence', '{}', ?, 1, '2026-11-10T08:00:00Z')
            """,
            ("workspace:investor_essence", json.dumps(essence_payload, ensure_ascii=False)),
        )
        conn.commit()

    # 4. Duplicate PENDING Watchdog Exceptions (REVIEW_REQUIRED + status=PENDING)
    from core.thesis.models import ThesisExceptionRow

    ex_repo = SqliteThesisExceptionRepositoryAdapter()
    for idx in range(2):
        ex_repo.insert_exception(
            ThesisExceptionRow(
                exception_id=f"EX-NVDA-{idx}",
                claim_id="TH-NVDA-v1",
                ticker="NVDA",
                detected_at_utc="2026-11-11T09:00:00Z",
                exception_type="BREACH",
                metric_name="gross_margin_pct",
                expected_value=">= 70%",
                observed_value="68.5%",
                source_citation="Watchdog price/margin monitor",
                human_action="REVIEW_REQUIRED",
                status="PENDING",
            )
        )

    domain_state = PortfolioState(
        last_updated="2026-11-12T08:00:00",
        fx_rates={"USDTHB": 35.0},
        summary=Summary(total_value_thb=1000000.0, total_cost_basis_thb=900000.0, total_unrealized_profit=100000.0),
        allocation_targets=[AllocationTarget(bucket_id="growth", name="Growth", target_percent=70.0)],
        holdings=[
            Holding(symbol="CASH_THB", asset_type="Cash", units=300000.0, market_value_thb=300000.0),
            Holding(
                symbol="NVDA",
                asset_type="US Stock",
                units=142.85,
                avg_cost_usd=110.0,
                current_price_usd=140.0,
                market_value_thb=700000.0,
                unrealized_pnl_percent=27.27,
                bucket_id="growth",
            ),
        ],
    )
    goals_list = [
        {
            "name": "Retirement Core",
            "goal_type": "nav_target",
            "target_amount_thb": 12000000.0,
            "current_amount_thb": 1000000.0,
            "progress_pct": 8.3,
            "deadline": "2040-12-31",
            "deadline_days_left": 5160,
            "priority_tier": "needs",
            "monthly_savings_thb": 35000.0,
            "required_return_pct": 14.5,
            "tension_level": "moderate",
            "adjustment_levers": {
                "save_more_monthly_thb": 12000.0,
                "extend_timeline_months": 18,
            },
        }
    ]

    class _StubPortSvc:
        def get_structured_portfolio_state(self, portfolio_id: str = "default", refresh_prices: bool = False):
            return domain_state

        def get_structured_watchlist(self, portfolio_id: str = "default"):
            return None

        def get_structured_goals(self, portfolio_id: str = "default"):
            return goals_list

    class _StubEqSvc:
        def get_detail(self, ticker: str):
            return {
                "ticker": ticker,
                "company_name": "NVIDIA Corporation",
                "composite_score": 84,
                "evaluated_at": "2026-11-11",
                "quant_signals": {"current_price": 140.0, "roic_pct": 70.0},
            }

        def get_news(self, ticker: str):
            return None

        def list_notes(self, ticker: str, days: int = 30):
            return None

        def read_note(self, relative_path: str):
            return None

    class _StubResolver:
        def resolve(self, ticker: str):
            class _Res:
                raw_symbol = ticker
                provider_symbol = ticker
                asset_class = "EQUITY"
                market = "US"
                confidence = 1.0

            return _Res()

    class _StubMacroSvc:
        def dashboard(self):
            return {"overall_regime": "Reflation"}

    class _StubPromiseDto:
        def model_dump(self):
            return {
                "credibility": 92.0,
                "grade": "A",
                "confidence": "HIGH",
                "delivery": 95.0,
                "transparency": 90.0,
                "n_resolved": 4,
                "n_promises": 5,
                "counts": {"kept": 4, "pending": 1},
                "bias": {"label": "Conservative Beat", "median_delta_pct": 5.2, "beat_rate": 100.0, "message": "Guides prudently and beats consistently."},
                "trend": {"verdict": "improving"},
            }

    class _StubPromiseSvc:
        def get_ledger(self, ticker: str):
            return _StubPromiseDto()

    class _StubThesisSvc:
        def get_active_thesis(self, ticker: str):
            return None

    class _StubPosRevSvc:
        def run_position_review(self, ticker: str, portfolio_id: str = "default"):
            return None

    class _StubFinCtxSvc:
        def get_financial_context(self, portfolio_id: str = "default"):
            return None

    class _StubDigestSvc:
        def get_daily_digest(self):
            return None

    monkeypatch.setattr(deps, "get_portfolio_service", lambda: _StubPortSvc())
    monkeypatch.setattr(deps, "get_equity_query_service", lambda: _StubEqSvc())
    monkeypatch.setattr(deps, "get_asset_resolver", lambda: _StubResolver())
    monkeypatch.setattr(deps, "get_macro_service", lambda: _StubMacroSvc())
    monkeypatch.setattr(deps, "get_promise_service", lambda: _StubPromiseSvc())
    monkeypatch.setattr(deps, "get_thesis_service", lambda: _StubThesisSvc())
    monkeypatch.setattr(deps, "get_position_review_service", lambda: _StubPosRevSvc())
    monkeypatch.setattr(deps, "get_financial_context_service", lambda: _StubFinCtxSvc())
    monkeypatch.setattr(deps, "get_digest_service", lambda: _StubDigestSvc())

    svc = deps.get_roundtable_service(request=None)

    # A. Without @Portfolio -> Equity signals (SQLite Analyst/Insider/Watchdog/Promise) surface; Advisory & Essence hidden
    s_no_port = await svc.create_session(
        CreateRoundtableSessionRequest(
            agenda_prompt="วิเคราะห์ $NVDA เชิงลึกโดยไม่แนบพอร์ต",
            context_refs=["$NVDA"],
        )
    )
    row_no_port = svc._repo.get_session(s_no_port.session.session_id)
    assert row_no_port is not None
    p_no_port = json.loads(row_no_port.frozen_packet_json)
    eq_txt = p_no_port["equity_report_text"]
    assert "[WATCHDOG BREACHES (PENDING)]:" in eq_txt
    assert "(Detected 2026-11-11, x2)" in eq_txt
    assert "Analyst Context Cache" in eq_txt
    assert "Target Mean=165.00" in eq_txt
    assert "Beat 1/1 reported" in eq_txt
    assert "2026-08-19: 0.75 vs 0.70" in eq_txt
    assert "synthetic" not in eq_txt.lower()
    assert "Insider DB Ledger (180d, Provenance=MIXED_FORM4_AND_YFINANCE):" in eq_txt
    assert "Paired Exercise/Award Sell=1" in eq_txt
    assert "ราคา $0 จำนวน 1 รายการ" in eq_txt
    assert "Promise Ledger Credibility: Score=92.00/100" in eq_txt
    pg_no_port = p_no_port["portfolio_guard"]
    assert pg_no_port["attachment_state"] == "NOT_ATTACHED_DO_NOT_ASSUME"
    assert "[CONFIRMED INVESTOR ESSENCE]" not in pg_no_port["summary_text"]
    assert "Portfolio Risk Radar:" not in pg_no_port["summary_text"]
    assert "Retirement Core" not in pg_no_port["summary_text"]

    # Investor Essence philosophy is separate from Portfolio Holdings & appears in essence_report_text
    assert "Confirmed Investor Essence" in p_no_port["essence_report_text"]
    assert "เน้นลงทุนธุรกิจที่มี Moat แข็งแกร่ง" in p_no_port["essence_report_text"]
    assert "หลักยึดหลักที่ยอมรับ (Accepted Claims): หลีกเลี่ยงบริษัทที่ใช้หนี้เกินตัว [HIGH/user]" in p_no_port["essence_report_text"]

    # B. With @Portfolio -> Portfolio Advisory + Goal Adjustment Levers surface cleanly
    s_with_port = await svc.create_session(
        CreateRoundtableSessionRequest(
            agenda_prompt="วิเคราะห์ $NVDA เทียบพอร์ตและแก่นการลงทุน",
            context_refs=["$NVDA", "@Portfolio"],
        )
    )
    row_with_port = svc._repo.get_session(s_with_port.session.session_id)
    assert row_with_port is not None
    p_with_port = json.loads(row_with_port.frozen_packet_json)
    pg_with_port = p_with_port["portfolio_guard"]
    pg_txt = pg_with_port["summary_text"]
    assert pg_with_port["attachment_state"] == "ATTACHED_HELD"
    assert "Confirmed Investor Essence" in p_with_port["essence_report_text"]
    assert "Portfolio Risk Radar" in pg_txt
    assert "HHI=10000" in pg_txt
    assert "Dual-Currency Cash Plan" in pg_txt
    assert "Holdings Advisory Signals" in pg_txt
    assert "Retirement Core [nav_target | needs]" in pg_txt
    assert "Goal Adjustment Levers:" in pg_txt
    assert "ออมเพิ่ม +12.00K THB/ด." in pg_txt
    assert "ยืดเวลา +18 เดือน" in pg_txt


@pytest.mark.asyncio
async def test_roundtable_portfolio_id_propagation_and_sizing_posture_persistence(api_env, tmp_path: Path):
    """Verify non-default portfolio_id is preserved across get_detail/refresh/interjection & sizing_posture persists."""
    requested_pids: List[Tuple[str, str]] = []

    class _TrackingProviders(_FakeProviders):
        def get_portfolio_state(self, portfolio_id: str = "default") -> Optional[Dict[str, Any]]:
            requested_pids.append(("portfolio_state", portfolio_id))
            return super().get_portfolio_state(portfolio_id)

        def get_confirmed_investment_axis(self, portfolio_id: str = "default") -> Optional[Dict[str, Any]]:
            requested_pids.append(("axis", portfolio_id))
            return super().get_confirmed_investment_axis(portfolio_id)

    repo = SqliteRoundtableRepositoryAdapter()
    providers = _TrackingProviders(include_nvda=True)
    assembler = RoundtableContextAssembler(providers=providers, repo=repo)
    tts = GeminiTTSAdapter(cache_dir=tmp_path / "rt_pid_audio", tts_fetcher=lambda m, v, t: b"\x01\x00" * 120)
    svc = RoundtableApplicationService(
        repo=repo,
        assembler=assembler,
        engine=_FakeDebateEngine(),
        tts=tts,
        verifier=RoundtableConsistencyVerifier(),
        providers=providers,
    )

    from schemas.roundtable_schemas import CreateRoundtableSessionRequest, InterjectRequest

    created = await svc.create_session(
        CreateRoundtableSessionRequest(
            agenda_prompt="วิเคราะห์ $NVDA จากพอร์ตหลักที่สอง",
            context_refs=["$NVDA", "@Portfolio"],
            portfolio_id="aggressive_growth",
        )
    )
    sid = created.session.session_id
    requested_pids.clear()

    # 1. Drift check in get_session_detail uses packet.portfolio_id
    _ = svc.get_session_detail(sid)
    assert ("portfolio_state", "aggressive_growth") in requested_pids
    assert ("axis", "aggressive_growth") in requested_pids
    assert all(pid == "aggressive_growth" for _, pid in requested_pids)

    # 2. stream_debate with refresh_context_on_drift=True uses packet.portfolio_id & persists sizing_posture
    requested_pids.clear()
    async for _ev, _pl in svc.stream_debate(sid, voice_enabled=False, refresh_context_on_drift=True):
        pass
    assert ("portfolio_state", "aggressive_growth") in requested_pids
    assert all(pid == "aggressive_growth" for _, pid in requested_pids)

    detail = svc.get_session_detail(sid)
    assert detail is not None
    rebuttal_turns = [t for t in detail.turns if t.round_stage == "rebuttal"]
    assert len(rebuttal_turns) > 0
    assert all(t.sizing_posture == "Conservative Staged Dip" for t in rebuttal_turns)

    # 3. handle_interjection with refresh_context=True preserves packet.portfolio_id
    requested_pids.clear()
    await svc.handle_interjection(
        sid,
        InterjectRequest(message="อัปเดตบริบทแล้วตอบ @Value", voice_enabled=False, refresh_context=True),
    )
    assert ("portfolio_state", "aggressive_growth") in requested_pids
    assert all(pid == "aggressive_growth" for _, pid in requested_pids)


def test_roundtable_default_crossfire_deep_youtube_16_turns_and_youtube_pack_export(
    authed_client: TestClient,
    tmp_path: Path,
):
    import io
    import wave
    import zipfile

    svc, _, _, _ = _build_test_service(tmp_path)
    app.dependency_overrides[get_roundtable_service] = lambda: svc
    try:
        # 1. Default CreateRoundtableSessionRequest selects "โหมดยาวสุด" (crossfire + deep_youtube)
        create_resp = authed_client.post(
            "/api/roundtable/sessions",
            json={
                "agenda_prompt": "เจาะลึก $NVDA เพื่อทำรายการวิเคราะห์ YouTube",
                "context_refs": ["$NVDA", "@Portfolio"],
            },
        )
        assert create_resp.status_code == 201
        created = create_resp.json()
        session_id = created["session"]["session_id"]
        assert created["debate_intensity"] == "crossfire"
        assert created["debate_depth"] == "deep_youtube"

        # 2. Stream with voice_enabled=False to verify lazy TTS backfill inside /youtube-pack
        stream_resp = authed_client.post(
            f"/api/roundtable/sessions/{session_id}/stream",
            json={"voice_enabled": False},
        )
        assert stream_resp.status_code == 200
        assert "event: done" in stream_resp.text

        detail = authed_client.get(f"/api/roundtable/sessions/{session_id}").json()
        stages = [t["round_stage"] for t in detail["turns"]]
        # 1 stage_intro + 5 opening + 5 rebuttal + 4 crossfire (2 disjoint duels) + 1 synthesis = 16 turns
        assert len(detail["turns"]) == 16
        assert stages.count("stage_intro") == 1
        assert stages.count("opening") == 5
        assert stages.count("rebuttal") == 5
        assert stages.count("crossfire") == 4
        assert stages.count("synthesis") == 1
        assert stages[0] == "stage_intro"
        assert stages[-1] == "synthesis"
        assert all(t["audio_cache_key"] is None for t in detail["turns"])

        # 3. Export YouTube Production Pack (.zip) — backfills missing WAVs cleanly
        yt_resp = authed_client.get(f"/api/roundtable/sessions/{session_id}/youtube-pack")
        assert yt_resp.status_code == 200
        assert yt_resp.headers["content-type"].startswith("application/zip")
        assert "roundtable_youtube_NVDA_" in yt_resp.headers.get("content-disposition", "")

        with zipfile.ZipFile(io.BytesIO(yt_resp.content), "r") as zf:
            names = set(zf.namelist())
            assert {
                "audio_full.wav",
                "subtitles.srt",
                "youtube_production_kit.md",
                "timeline_manifest.json",
            } <= names
            with wave.open(io.BytesIO(zf.read("audio_full.wav")), "rb") as wf:
                assert wf.getnchannels() == 1
                assert wf.getframerate() == 24000
                assert wf.getnframes() > 24000
            srt = zf.read("subtitles.srt").decode("utf-8")
            assert "[Hermes]" in srt and "-->" in srt
            manifest = json.loads(zf.read("timeline_manifest.json").decode("utf-8"))
            assert len(manifest["cues"]) == 16
            assert manifest["missing_audio_turn_ids"] == []
            assert any(ch["round_stage"] == "crossfire" for ch in manifest["chapters"])

        # Unknown session returns 404
        assert authed_client.get("/api/roundtable/sessions/rt_does_not_exist/youtube-pack").status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_roundtable_preset_label_agenda_order_and_pillar_coverage_sse(
    authed_client: TestClient,
    tmp_path: Path,
):
    svc, _, _, _ = _build_test_service(tmp_path)
    app.dependency_overrides[get_roundtable_service] = lambda: svc
    try:
        # 1. Guru catalog returns specialized preset lens + non-default preset_label persistence
        catalog_resp = authed_client.get("/api/roundtable/guru-profiles")
        assert catalog_resp.status_code == 200
        catalog = catalog_resp.json()
        q_forensic = next(
            pr for pr in catalog["presets"]
            if pr["speaker_id"] == "quality_guru" and pr["preset_id"] == "quality_forensic_auditor"
        )
        assert q_forensic["role_lens_label"]
        assert "Sloan" in q_forensic["analytical_rubric"] or "ocf_to_net_income" in q_forensic["analytical_rubric"]

        save_resp = authed_client.put(
            "/api/roundtable/guru-profiles",
            json={
                "profiles": [
                    {"speaker_id": "quality_guru", "preset_id": "quality_forensic_auditor"}
                ]
            },
        )
        assert save_resp.status_code == 200
        saved_q = next(p for p in save_resp.json()["personas"] if p["speaker_id"] == "quality_guru")
        assert saved_q["preset_id"] == "quality_forensic_auditor"
        assert saved_q["preset_label"] == q_forensic["label_th"]

        # 2. Risk-focused agenda on stock debate_order promotes skeptic_guru first in opening + streams pillar_coverage
        risk_sess_resp = authed_client.post(
            "/api/roundtable/sessions",
            json={
                "agenda_prompt": "จับผิดความเสี่ยง หนี้สิน credit dilution และ kill-switch ของ $NVDA",
                "context_refs": ["$NVDA"],
                "debate_depth": "concise",
            },
        )
        assert risk_sess_resp.status_code == 201
        risk_created = risk_sess_resp.json()
        risk_sid = risk_created["session"]["session_id"]
        created_q = next(p for p in risk_created["personas"] if p["speaker_id"] == "quality_guru")
        assert created_q["preset_label"] == q_forensic["label_th"]

        stream_resp = authed_client.post(
            f"/api/roundtable/sessions/{risk_sid}/stream",
            json={"voice_enabled": False},
        )
        assert stream_resp.status_code == 200
        assert "event: pillar_coverage" in stream_resp.text

        detail = authed_client.get(f"/api/roundtable/sessions/{risk_sid}").json()
        opening_speakers = [t["speaker_id"] for t in detail["turns"] if t["round_stage"] == "opening"]
        assert opening_speakers[0] == "skeptic_guru"
        assert detail["pillar_coverage"] is not None
        assert len(detail["pillar_coverage"]["pillars"]) == 5

        # 3. Custom debate_order (growth_guru placed before value_guru) overrides agenda smart-order
        custom_sess_resp = authed_client.post(
            "/api/roundtable/sessions",
            json={
                "agenda_prompt": "จับผิดความเสี่ยง หนี้สิน credit dilution และ kill-switch ของ $NVDA",
                "context_refs": ["$NVDA"],
                "debate_depth": "concise",
                "persona_overrides": [
                    {"speaker_id": "growth_guru", "debate_order": 1},
                    {"speaker_id": "value_guru", "debate_order": 2},
                ],
            },
        )
        assert custom_sess_resp.status_code == 201
        custom_sid = custom_sess_resp.json()["session"]["session_id"]
        authed_client.post(f"/api/roundtable/sessions/{custom_sid}/stream", json={"voice_enabled": False})
        custom_detail = authed_client.get(f"/api/roundtable/sessions/{custom_sid}").json()
        custom_opening = [t["speaker_id"] for t in custom_detail["turns"] if t["round_stage"] == "opening"]
        assert custom_opening[0] == "growth_guru"
    finally:
        app.dependency_overrides.clear()









def test_roundtable_provider_delegates_to_atomic_dispatch_port(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import api.dependencies as dependencies

    class _Queue:
        def __init__(self):
            self.calls = []

        def dispatch_approval_with_card(self, **kwargs):
            self.calls.append(kwargs)
            return "job_nvda_001"

    registered_queue = _Queue()
    service = object()
    monkeypatch.setattr(dependencies, "_ROUNDTABLE_SERVICE", None)
    monkeypatch.setattr(dependencies, "_ROUNDTABLE_CACHE_KEY", None)
    monkeypatch.setattr(dependencies, "_GLOBAL_APP_JOB_QUEUE", None)
    monkeypatch.setattr(dependencies, "_get_vault_path", lambda: tmp_path)
    monkeypatch.setattr(dependencies, "_create_roundtable_service", lambda _key: service)
    monkeypatch.setattr(
        dependencies,
        "get_equity_query_service",
        lambda: SimpleNamespace(list_latest=lambda: None),
    )

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(job_queue=registered_queue)))
    assert dependencies.get_roundtable_service(request) is service
    assert dependencies._get_global_app_job_queue() is registered_queue
    assert dependencies._dispatch_roundtable_research("NVDA") == "job_nvda_001"
    assert len(registered_queue.calls) == 1
    call = registered_queue.calls[0]
    assert call["source_key"] == "roundtable:equity_intel:NVDA:portfolio:global:scope:both"
    assert call["flow"] == "manager"
    assert call["scope"] == "both"
    assert call["approval_metadata"]["ticker"] == "NVDA"
    assert call["approval_metadata"]["research_type"] == "equity_intel"

    startup_queue = _Queue()
    dependencies.set_global_app_job_queue(startup_queue)
    assert dependencies.get_roundtable_service() is service
    assert dependencies._dispatch_roundtable_research("NVDA") == "job_nvda_001"
    assert len(startup_queue.calls) == 1

    dependencies.set_global_app_job_queue(None)
    with pytest.raises(RuntimeError, match="running application job queue"):
        dependencies._dispatch_roundtable_research("NVDA")


def test_roundtable_dispatch_is_idempotent_across_repeated_concurrent_calls_and_terminal_rerun(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    import json

    from api.db.uow import DbUnitOfWork
    from api.jobs import JobQueue
    from application.roundtable.research_dispatch_service import RoundtableResearchDispatchService

    db_path = tmp_path / "roundtable_dispatch.sqlite3"
    queue = JobQueue(run_fn=lambda **_kwargs: None, db_path=str(db_path))
    dispatcher = RoundtableResearchDispatchService(queue)

    first_job_id = dispatcher.dispatch_prerequisite_research(" nvda ")
    assert dispatcher.dispatch_prerequisite_research("NVDA") == first_job_id

    call_barrier = threading.Barrier(8)

    def _concurrent_dispatch():
        call_barrier.wait(timeout=10)
        return dispatcher.dispatch_prerequisite_research("NVDA")

    with ThreadPoolExecutor(max_workers=8) as executor:
        concurrent_ids = list(executor.map(lambda _index: _concurrent_dispatch(), range(8)))
    assert set(concurrent_ids) == {first_job_id}

    with DbUnitOfWork(db_path=str(db_path)) as uow:
        cards = uow.kanban.list_kanban_cards()
        approval_jobs = uow.jobs.list_jobs_by_status(["awaiting_approval"], flows=["manager"])
        assert len(cards) == 1
        assert len(approval_jobs) == 1
        card = cards[0]
        job = approval_jobs[0]
        assert card["column_name"] == "approval"
        assert card["job_id"] == first_job_id
        assert job["job_id"] == first_job_id
        assert job["card_id"] == card["card_id"]
        approval = json.loads(job["interrupt_payload"])
        assert approval["ticker"] == "NVDA"
        assert approval["research_type"] == "equity_intel"
        assert approval["source_key"] == card["source_key"]

    with DbUnitOfWork(db_path=str(db_path)) as uow:
        uow.jobs.update_job_status(first_job_id, "done")
    rerun_job_id = dispatcher.dispatch_prerequisite_research("NVDA")
    assert rerun_job_id != first_job_id
    assert dispatcher.dispatch_prerequisite_research("NVDA") == rerun_job_id

    with DbUnitOfWork(db_path=str(db_path)) as uow:
        cards = uow.kanban.list_kanban_cards()
        jobs = uow.jobs.list_jobs_by_status(["done", "awaiting_approval"], flows=["manager"])
        assert len(cards) == 1
        assert len(jobs) == 2
        latest_approval = next(job for job in jobs if job["status"] == "awaiting_approval")
        assert latest_approval["job_id"] == rerun_job_id
        assert cards[0]["job_id"] == rerun_job_id
        approval = json.loads(latest_approval["interrupt_payload"])
        assert approval["type"] == "pre_run_approval"
        assert approval["ticker"] == "NVDA"

    separate_scope_job_id = dispatcher.dispatch_prerequisite_research(
        "NVDA", portfolio_id="family", scope="vault"
    )
    assert separate_scope_job_id not in {first_job_id, rerun_job_id}


def test_roundtable_dispatch_rolls_back_a_failed_request_before_next_request_succeeds(
    monkeypatch, tmp_path
):
    from concurrent.futures import ThreadPoolExecutor
    import threading

    from api.db.adapters import SqliteJobRepositoryAdapter
    from api.db.uow import DbUnitOfWork
    from api.jobs import JobQueue
    from application.roundtable.research_dispatch_service import RoundtableResearchDispatchService

    db_path = tmp_path / "roundtable_dispatch_race.sqlite3"
    queue = JobQueue(run_fn=lambda **_kwargs: None, db_path=str(db_path))
    dispatcher = RoundtableResearchDispatchService(queue)
    first_inside_transaction = threading.Event()
    second_started = threading.Event()
    release_first = threading.Event()
    thread_state = threading.local()
    original_set_approval = SqliteJobRepositoryAdapter.set_job_awaiting_approval

    def _fail_first_after_card_and_job_insert(self, job_id, payload):
        if getattr(thread_state, "fail_this_request", False):
            first_inside_transaction.set()
            if not release_first.wait(timeout=10):
                raise TimeoutError("test did not release the first dispatch")
            raise RuntimeError("injected failure before approval commit")
        return original_set_approval(self, job_id, payload)

    monkeypatch.setattr(
        SqliteJobRepositoryAdapter,
        "set_job_awaiting_approval",
        _fail_first_after_card_and_job_insert,
    )

    def _failing_request():
        thread_state.fail_this_request = True
        return dispatcher.dispatch_prerequisite_research("NVDA")

    def _successful_request():
        second_started.set()
        return dispatcher.dispatch_prerequisite_research("NVDA")

    with ThreadPoolExecutor(max_workers=2) as executor:
        failed = executor.submit(_failing_request)
        assert first_inside_transaction.wait(timeout=10)
        successful = executor.submit(_successful_request)
        assert second_started.wait(timeout=10)
        release_first.set()
        with pytest.raises(RuntimeError, match="injected failure"):
            failed.result(timeout=10)
        successful_job_id = successful.result(timeout=10)

    with DbUnitOfWork(db_path=str(db_path)) as uow:
        cards = uow.kanban.list_kanban_cards()
        approval_jobs = uow.jobs.list_jobs_by_status(["awaiting_approval"], flows=["manager"])
        assert len(cards) == 1
        assert len(approval_jobs) == 1
        assert approval_jobs[0]["job_id"] == successful_job_id
        assert approval_jobs[0]["card_id"] == cards[0]["card_id"]
        assert cards[0]["job_id"] == successful_job_id
        assert cards[0]["column_name"] == "approval"
