"""Pure SQLite DAO repository for Trinity Guru Roundtable sessions and turns.

Follows AST DAO isolation rules:
- Accepts ``conn: sqlite3.Connection`` from caller / adapter / UnitOfWork.
- Never imports ``api.db.connection``.
- Never calls ``conn.commit()`` or ``conn.rollback()``.
"""
from __future__ import annotations

import json
import sqlite3
import time
from typing import Any, Dict, List, Optional, Sequence

from core.roundtable.models import RoundtableSession, RoundtableTurn


def _row_to_session(row: sqlite3.Row) -> RoundtableSession:
    try:
        refs = tuple(json.loads(row["context_refs_json"] or "[]"))
    except Exception:
        refs = ()
    return RoundtableSession(
        session_id=row["session_id"],
        title=row["title"],
        primary_ticker=row["primary_ticker"],
        benchmark_ticker=row["benchmark_ticker"],
        entry_price_at_session=row["entry_price_at_session"],
        agenda_prompt=row["agenda_prompt"],
        context_refs=refs,
        frozen_packet_json=row["frozen_packet_json"] or "{}",
        packet_fingerprint_sha256=row["packet_fingerprint_sha256"] or "",
        decision_playbook_json=row["decision_playbook_json"],
        final_rating=row["final_rating"] or "REVIEW",
        settlement_status=row["settlement_status"] or "pending",
        settled_return_pct=row["settled_return_pct"],
        settled_alpha_pct=row["settled_alpha_pct"],
        settled_at_date=row["settled_at_date"],
        reflection_lesson=row["reflection_lesson"],
        status=row["status"] or "active",
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
    )


def _row_to_turn(row: sqlite3.Row) -> RoundtableTurn:
    try:
        bullets = tuple(json.loads(row["evidence_bullets_json"] or "[]"))
    except Exception:
        bullets = ()
    row_keys = set(row.keys()) if hasattr(row, "keys") else set()
    return RoundtableTurn(
        turn_id=row["turn_id"],
        session_id=row["session_id"],
        seq=int(row["seq"]),
        speaker_id=row["speaker_id"],
        round_stage=row["round_stage"],
        replying_to=row["replying_to"],
        spoken_dialogue=row["spoken_dialogue"],
        evidence_bullets=bullets,
        stance_tag=row["stance_tag"] or "REVIEW",
        audio_cache_key=row["audio_cache_key"],
        created_at=float(row["created_at"]),
        sizing_posture=row["sizing_posture"] if "sizing_posture" in row_keys else None,
    )


def create_roundtable_session(
    conn: sqlite3.Connection,
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
) -> RoundtableSession:
    now = time.time()
    refs_json = json.dumps(list(context_refs), ensure_ascii=False)
    conn.execute(
        """
        INSERT INTO roundtable_sessions (
            session_id, title, primary_ticker, benchmark_ticker, entry_price_at_session,
            agenda_prompt, context_refs_json, frozen_packet_json, packet_fingerprint_sha256,
            decision_playbook_json, final_rating, settlement_status,
            settled_return_pct, settled_alpha_pct, settled_at_date, reflection_lesson,
            status, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, 'pending', NULL, NULL, NULL, NULL, ?, ?, ?)
        """,
        (
            session_id,
            title,
            primary_ticker,
            benchmark_ticker,
            entry_price_at_session,
            agenda_prompt,
            refs_json,
            frozen_packet_json,
            packet_fingerprint_sha256,
            final_rating,
            status,
            now,
            now,
        ),
    )
    row = conn.execute("SELECT * FROM roundtable_sessions WHERE session_id = ?", (session_id,)).fetchone()
    assert row is not None
    return _row_to_session(row)


def get_roundtable_session(conn: sqlite3.Connection, session_id: str) -> Optional[RoundtableSession]:
    row = conn.execute("SELECT * FROM roundtable_sessions WHERE session_id = ?", (session_id,)).fetchone()
    return _row_to_session(row) if row is not None else None


def list_roundtable_sessions(conn: sqlite3.Connection, limit: int = 30) -> List[RoundtableSession]:
    rows = conn.execute(
        "SELECT * FROM roundtable_sessions ORDER BY updated_at DESC LIMIT ?",
        (int(limit),),
    ).fetchall()
    return [_row_to_session(r) for r in rows]


def list_roundtable_sessions_by_ticker(
    conn: sqlite3.Connection, ticker: str, limit: int = 10
) -> List[RoundtableSession]:
    clean = ticker.strip().upper()
    rows = conn.execute(
        """
        SELECT * FROM roundtable_sessions
        WHERE UPPER(COALESCE(primary_ticker, '')) = ?
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (clean, int(limit)),
    ).fetchall()
    return [_row_to_session(r) for r in rows]


def update_roundtable_session_playbook(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    decision_playbook_json: str,
    final_rating: str,
    status: str = "completed",
) -> Optional[RoundtableSession]:
    now = time.time()
    conn.execute(
        """
        UPDATE roundtable_sessions
        SET decision_playbook_json = ?, final_rating = ?, status = ?, updated_at = ?
        WHERE session_id = ?
        """,
        (decision_playbook_json, final_rating, status, now, session_id),
    )
    return get_roundtable_session(conn, session_id)


def update_roundtable_session_context(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    frozen_packet_json: str,
    packet_fingerprint_sha256: str,
    entry_price_at_session: Optional[str] = None,
) -> Optional[RoundtableSession]:
    now = time.time()
    if entry_price_at_session is not None:
        conn.execute(
            """
            UPDATE roundtable_sessions
            SET frozen_packet_json = ?, packet_fingerprint_sha256 = ?, entry_price_at_session = ?, updated_at = ?
            WHERE session_id = ?
            """,
            (frozen_packet_json, packet_fingerprint_sha256, entry_price_at_session, now, session_id),
        )
    else:
        conn.execute(
            """
            UPDATE roundtable_sessions
            SET frozen_packet_json = ?, packet_fingerprint_sha256 = ?, updated_at = ?
            WHERE session_id = ?
            """,
            (frozen_packet_json, packet_fingerprint_sha256, now, session_id),
        )
    return get_roundtable_session(conn, session_id)


def update_roundtable_session_settlement(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    settlement_status: str,
    settled_return_pct: Optional[str],
    settled_alpha_pct: Optional[str],
    settled_at_date: Optional[str],
    reflection_lesson: Optional[str],
) -> Optional[RoundtableSession]:
    now = time.time()
    conn.execute(
        """
        UPDATE roundtable_sessions
        SET settlement_status = ?,
            settled_return_pct = ?,
            settled_alpha_pct = ?,
            settled_at_date = ?,
            reflection_lesson = ?,
            updated_at = ?
        WHERE session_id = ?
        """,
        (
            settlement_status,
            settled_return_pct,
            settled_alpha_pct,
            settled_at_date,
            reflection_lesson,
            now,
            session_id,
        ),
    )
    return get_roundtable_session(conn, session_id)


def get_next_roundtable_turn_seq(conn: sqlite3.Connection, session_id: str) -> int:
    row = conn.execute(
        "SELECT COALESCE(MAX(seq), 0) AS max_seq FROM roundtable_turns WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    return int(row["max_seq"]) + 1 if row is not None else 1


def append_roundtable_turn(
    conn: sqlite3.Connection,
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
) -> RoundtableTurn:
    now = time.time()
    bullets_json = json.dumps(list(evidence_bullets), ensure_ascii=False)
    conn.execute(
        """
        INSERT INTO roundtable_turns (
            turn_id, session_id, seq, speaker_id, round_stage,
            replying_to, spoken_dialogue, evidence_bullets_json, stance_tag,
            sizing_posture, audio_cache_key, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            turn_id,
            session_id,
            int(seq),
            speaker_id,
            round_stage,
            replying_to,
            spoken_dialogue,
            bullets_json,
            stance_tag,
            sizing_posture,
            audio_cache_key,
            now,
        ),
    )
    conn.execute(
        "UPDATE roundtable_sessions SET updated_at = ? WHERE session_id = ?",
        (now, session_id),
    )
    row = conn.execute("SELECT * FROM roundtable_turns WHERE turn_id = ?", (turn_id,)).fetchone()
    assert row is not None
    return _row_to_turn(row)


def update_roundtable_turn_audio_key(conn: sqlite3.Connection, turn_id: str, audio_cache_key: str) -> None:
    conn.execute(
        "UPDATE roundtable_turns SET audio_cache_key = ? WHERE turn_id = ?",
        (audio_cache_key, turn_id),
    )


def get_roundtable_turns(conn: sqlite3.Connection, session_id: str) -> List[RoundtableTurn]:
    rows = conn.execute(
        "SELECT * FROM roundtable_turns WHERE session_id = ? ORDER BY seq ASC",
        (session_id,),
    ).fetchall()
    return [_row_to_turn(r) for r in rows]


def _row_to_guru_profile(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "speaker_id": str(row["speaker_id"]),
        "display_name": str(row["display_name"]),
        "short_name": str(row["short_name"]),
        "archetype": str(row["archetype"]),
        "speaking_style": str(row["speaking_style"] or ""),
        "custom_directive": str(row["custom_directive"] or ""),
        "preset_id": str(row["preset_id"] or "default"),
        "voice_name": str(row["voice_name"]),
        "pitch": float(row["pitch"] if row["pitch"] is not None else 1.0),
        "rate": float(row["rate"] if row["rate"] is not None else 1.0),
        "accent_color": str(row["accent_color"]),
        "emoji": str(row["emoji"] or "🎙️"),
        "enabled": bool(row["enabled"]),
        "debate_order": int(row["debate_order"] if row["debate_order"] is not None else 1),
        "updated_at": float(row["updated_at"]),
    }


def list_roundtable_guru_profiles(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM roundtable_guru_profiles ORDER BY debate_order ASC, speaker_id ASC"
    ).fetchall()
    return [_row_to_guru_profile(r) for r in rows]


def upsert_roundtable_guru_profiles(
    conn: sqlite3.Connection,
    profiles: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    now = time.time()
    for p in profiles:
        speaker_id = str(p["speaker_id"])
        conn.execute(
            """
            INSERT INTO roundtable_guru_profiles (
                speaker_id, display_name, short_name, archetype,
                speaking_style, custom_directive, preset_id,
                voice_name, pitch, rate, accent_color, emoji,
                enabled, debate_order, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(speaker_id) DO UPDATE SET
                display_name = excluded.display_name,
                short_name = excluded.short_name,
                archetype = excluded.archetype,
                speaking_style = excluded.speaking_style,
                custom_directive = excluded.custom_directive,
                preset_id = excluded.preset_id,
                voice_name = excluded.voice_name,
                pitch = excluded.pitch,
                rate = excluded.rate,
                accent_color = excluded.accent_color,
                emoji = excluded.emoji,
                enabled = excluded.enabled,
                debate_order = excluded.debate_order,
                updated_at = excluded.updated_at
            """,
            (
                speaker_id,
                str(p.get("display_name") or speaker_id),
                str(p.get("short_name") or speaker_id),
                str(p.get("archetype") or ""),
                str(p.get("speaking_style") or ""),
                str(p.get("custom_directive") or ""),
                str(p.get("preset_id") or "default"),
                str(p.get("voice_name") or "Kore"),
                float(p.get("pitch") if p.get("pitch") is not None else 1.0),
                float(p.get("rate") if p.get("rate") is not None else 1.0),
                str(p.get("accent_color") or "#0ea5e9"),
                str(p.get("emoji") or "🎙️"),
                1 if bool(True if p.get("enabled") is None else p.get("enabled")) else 0,
                int(p.get("debate_order") if p.get("debate_order") is not None else 1),
                now,
            ),
        )
    return list_roundtable_guru_profiles(conn)


def reset_roundtable_guru_profiles(
    conn: sqlite3.Connection,
    speaker_id: Optional[str] = None,
) -> None:
    if speaker_id:
        conn.execute("DELETE FROM roundtable_guru_profiles WHERE speaker_id = ?", (speaker_id,))
    else:
        conn.execute("DELETE FROM roundtable_guru_profiles")


