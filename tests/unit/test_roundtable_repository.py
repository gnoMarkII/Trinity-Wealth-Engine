"""Unit tests for WP-1: Roundtable SQLite DAO, Migrations, Adapter & UnitOfWork."""
import sqlite3
import pytest

from api.db.connection import get_connection, init_schema
from api.db.adapters import SqliteRoundtableRepositoryAdapter
from api.db.uow import DbUnitOfWork


@pytest.fixture
def temp_db(tmp_path):
    db_file = str(tmp_path / "roundtable_test.sqlite")
    conn = get_connection(db_file)
    conn.close()
    return db_file


def test_roundtable_tables_and_column_migrations_exist(tmp_path):
    db_file = str(tmp_path / "legacy.sqlite")
    conn = sqlite3.connect(db_file)
    conn.row_factory = sqlite3.Row
    # Simulate minimal old schema before running full init_schema
    conn.execute("CREATE TABLE roundtable_sessions (session_id TEXT PRIMARY KEY, title TEXT NOT NULL, agenda_prompt TEXT NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL)")
    conn.execute("CREATE TABLE roundtable_turns (turn_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, seq INTEGER NOT NULL, speaker_id TEXT NOT NULL, round_stage TEXT NOT NULL, spoken_dialogue TEXT NOT NULL, created_at REAL NOT NULL)")
    conn.commit()
    init_schema(conn)
    sess_cols = {r["name"] for r in conn.execute("PRAGMA table_info(roundtable_sessions)")}
    turn_cols = {r["name"] for r in conn.execute("PRAGMA table_info(roundtable_turns)")}
    profile_cols = {r["name"] for r in conn.execute("PRAGMA table_info(roundtable_guru_profiles)")}
    conn.close()

    assert {"packet_fingerprint_sha256", "decision_playbook_json", "settled_alpha_pct", "reflection_lesson"} <= sess_cols
    assert {"replying_to", "evidence_bullets_json", "stance_tag", "sizing_posture", "audio_cache_key"} <= turn_cols
    assert {
        "speaker_id",
        "display_name",
        "short_name",
        "archetype",
        "speaking_style",
        "custom_directive",
        "preset_id",
        "voice_name",
        "pitch",
        "rate",
        "accent_color",
        "emoji",
        "enabled",
        "debate_order",
        "updated_at",
    } <= profile_cols


def test_adapter_crud_turns_settlement_and_unique_seq(temp_db):
    repo = SqliteRoundtableRepositoryAdapter(db_path=temp_db)

    sess = repo.create_session(
        session_id="rt_001",
        title="ถกหุ้น NVDA",
        primary_ticker="NVDA",
        benchmark_ticker="SPY",
        entry_price_at_session="135.50",
        agenda_prompt="วิเคราะห์ NVDA ที่ราคานี้ควรทยอยเก็บหรือรอ",
        context_refs=["$NVDA", "@Macro", "@Portfolio"],
        frozen_packet_json='{"primary_ticker": "NVDA"}',
        packet_fingerprint_sha256="abc123fingerprint",
    )
    assert sess.session_id == "rt_001"
    assert sess.primary_ticker == "NVDA"
    assert sess.context_refs == ("$NVDA", "@Macro", "@Portfolio")
    assert sess.final_rating == "REVIEW"

    # Append turns sequentially
    seq1 = repo.get_next_seq("rt_001")
    assert seq1 == 1
    t1 = repo.append_turn(
        turn_id="turn_1",
        session_id="rt_001",
        seq=seq1,
        speaker_id="value_guru",
        round_stage="opening",
        replying_to=None,
        spoken_dialogue="ราคาปัจจุบันสูงกว่าส่วนเผื่อความปลอดภัยมากครับ",
        evidence_bullets=["[FACT] FCF Yield 1.8%"],
        stance_tag="HOLD",
        audio_cache_key="aud_1",
        sizing_posture="Conservative Staged Dip",
    )
    assert t1.seq == 1
    assert t1.evidence_bullets == ("[FACT] FCF Yield 1.8%",)
    assert t1.sizing_posture == "Conservative Staged Dip"

    # Unique (session_id, seq) guard
    with pytest.raises(sqlite3.IntegrityError):
        repo.append_turn(
            turn_id="turn_dup",
            session_id="rt_001",
            seq=1,
            speaker_id="growth_guru",
            round_stage="opening",
            replying_to=None,
            spoken_dialogue="ซ้ำลำดับ",
            evidence_bullets=[],
            stance_tag="BUY",
            audio_cache_key=None,
        )

    repo.update_turn_audio_key("turn_1", "aud_1_updated")
    turns = repo.get_turns("rt_001")
    assert len(turns) == 1
    assert turns[0].audio_cache_key == "aud_1_updated"
    assert turns[0].sizing_posture == "Conservative Staged Dip"

    # Update playbook & settlement
    updated = repo.update_session_playbook(
        session_id="rt_001",
        decision_playbook_json='{"final_rating": "OVERWEIGHT"}',
        final_rating="OVERWEIGHT",
        status="completed",
    )
    assert updated is not None
    assert updated.final_rating == "OVERWEIGHT"
    assert updated.status == "completed"

    settled = repo.update_session_settlement(
        session_id="rt_001",
        settlement_status="settled",
        settled_return_pct="12.40",
        settled_alpha_pct="5.10",
        settled_at_date="2026-11-10",
        reflection_lesson="โมเมนตัม AI ชนะความกังวล Valuation ระยะสั้น",
    )
    assert settled is not None
    assert settled.settlement_status == "settled"
    assert settled.settled_alpha_pct == "5.10"

    by_ticker = repo.list_sessions_by_ticker("nvda")
    assert len(by_ticker) == 1
    assert by_ticker[0].session_id == "rt_001"

    updated_ctx = repo.update_session_context(
        session_id="rt_001",
        frozen_packet_json='{"primary_ticker": "NVDA", "v": 2}',
        packet_fingerprint_sha256="new_hash",
        entry_price_at_session="140.00",
    )
    assert updated_ctx is not None
    assert updated_ctx.packet_fingerprint_sha256 == "new_hash"
    assert updated_ctx.entry_price_at_session == "140.00"


def test_uow_roundtable_rollback_and_commit(temp_db):
    with pytest.raises(RuntimeError):
        with DbUnitOfWork(db_path=temp_db) as uow:
            uow.roundtable.create_session(
                session_id="rt_rollback",
                title="Should rollback",
                primary_ticker="AAPL",
                benchmark_ticker="SPY",
                entry_price_at_session="220.00",
                agenda_prompt="Test",
                context_refs=["$AAPL"],
                frozen_packet_json="{}",
                packet_fingerprint_sha256="h1",
            )
            raise RuntimeError("boom")

    repo = SqliteRoundtableRepositoryAdapter(db_path=temp_db)
    assert repo.get_session("rt_rollback") is None

    with DbUnitOfWork(db_path=temp_db) as uow:
        uow.roundtable.create_session(
            session_id="rt_commit",
            title="Should commit",
            primary_ticker="AAPL",
            benchmark_ticker="SPY",
            entry_price_at_session="220.00",
            agenda_prompt="Test",
            context_refs=["$AAPL"],
            frozen_packet_json="{}",
            packet_fingerprint_sha256="h2",
        )
        uow.roundtable.append_turn(
            turn_id="t_commit",
            session_id="rt_commit",
            seq=1,
            speaker_id="hermes_moderator",
            round_stage="opening",
            replying_to=None,
            spoken_dialogue="เริ่มประชุม",
            evidence_bullets=[],
            stance_tag="REVIEW",
            audio_cache_key=None,
        )

    assert repo.get_session("rt_commit") is not None
    assert len(repo.get_turns("rt_commit")) == 1


def test_guru_profile_crud_and_reset_with_uow(temp_db):
    repo = SqliteRoundtableRepositoryAdapter(db_path=temp_db)
    assert repo.list_guru_profiles() == []

    # Rolled back transaction must not persist profiles
    with pytest.raises(RuntimeError):
        with DbUnitOfWork(db_path=temp_db) as uow:
            uow.roundtable.upsert_guru_profiles(
                [
                    {
                        "speaker_id": "value_guru",
                        "display_name": "Should Rollback",
                        "short_name": "Rollback",
                        "archetype": "Value",
                        "voice_name": "Iapetus",
                        "debate_order": 1,
                    }
                ]
            )
            raise RuntimeError("abort profile write")

    assert repo.list_guru_profiles() == []

    # Persist customized profiles cleanly via adapter & UoW
    repo.upsert_guru_profiles(
        [
            {
                "speaker_id": "value_guru",
                "display_name": "Deep Value (Graham)",
                "short_name": "Graham",
                "archetype": "Benjamin Graham Deep Value",
                "speaking_style": "เข้มงวดเรื่อง Margin of Safety",
                "custom_directive": "ตรวจ P/BV และหนี้สินทุกครั้ง",
                "preset_id": "value_graham_deep",
                "voice_name": "Iapetus",
                "pitch": 0.96,
                "rate": 1.0,
                "accent_color": "#10b981",
                "emoji": "📐",
                "enabled": True,
                "debate_order": 2,
            },
            {
                "speaker_id": "skeptic_guru",
                "display_name": "Tail-Risk Critic",
                "short_name": "TailRisk",
                "archetype": "Nassim Taleb Antifragile",
                "speaking_style": "ทดสอบความเสี่ยงหางอ้วน",
                "custom_directive": "ตรวจ fat tail",
                "preset_id": "skeptic_taleb_tailrisk",
                "voice_name": "Enceladus",
                "pitch": 0.92,
                "rate": 1.02,
                "accent_color": "#ef4444",
                "emoji": "⚠️",
                "enabled": False,
                "debate_order": 1,
            },
        ]
    )

    saved = repo.list_guru_profiles()
    assert len(saved) == 2
    assert saved[0]["speaker_id"] == "skeptic_guru"  # sorted by debate_order
    assert saved[0]["enabled"] is False
    assert saved[1]["speaker_id"] == "value_guru"
    assert saved[1]["short_name"] == "Graham"
    assert saved[1]["voice_name"] == "Iapetus"

    # Reset single seat keeps sibling seat customized
    repo.reset_guru_profiles("skeptic_guru")
    remaining = repo.list_guru_profiles()
    assert len(remaining) == 1
    assert remaining[0]["speaker_id"] == "value_guru"

    # Reset all clears remaining rows
    repo.reset_guru_profiles(None)
    assert repo.list_guru_profiles() == []

