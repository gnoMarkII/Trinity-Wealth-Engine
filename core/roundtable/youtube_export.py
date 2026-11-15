"""Pure stdlib YouTube Production Pack builder for Trinity Guru Roundtable sessions.

Architecture Layer: core/ (pure Python stdlib + core.roundtable.models only).
Assembles time-locked stitched WAV audio, chunked SRT subtitles, YouTube chapters,
and shot-by-shot visual cue sheets from persisted turns and WAV buffers.
"""
from __future__ import annotations

import io
import json
import re
import struct
from typing import Any, Dict, List, Optional, Sequence, Tuple
import wave
import zipfile

from core.roundtable.models import (
    FrozenContextPacket,
    RoundStage,
    resolve_personas_from_packet,
)

DEFAULT_WAV_CHANNELS = 1
DEFAULT_WAV_SAMPWIDTH = 2
DEFAULT_WAV_FRAMERATE = 24000
INTRA_STAGE_PAUSE_SEC = 0.40
INTER_STAGE_PAUSE_SEC = 0.65

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def _format_srt_timestamp(ms: int) -> str:
    safe_ms = max(0, int(ms))
    hours = safe_ms // 3_600_000
    rem = safe_ms % 3_600_000
    minutes = rem // 60_000
    rem %= 60_000
    seconds = rem // 1_000
    millis = rem % 1_000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def _format_yt_chapter_timestamp(ms: int) -> str:
    total_sec = max(0, int(ms) // 1000)
    hours = total_sec // 3600
    minutes = (total_sec % 3600) // 60
    seconds = total_sec % 60
    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _estimate_spoken_duration_sec(text: str) -> float:
    clean = re.sub(r"\s+", " ", (text or "")).strip()
    if not clean:
        return 3.0
    return round(max(3.0, min(60.0, len(clean) / 13.0)), 3)


def _decode_wav_pcm(
    wav_bytes: bytes,
    *,
    target_channels: int = DEFAULT_WAV_CHANNELS,
    target_sampwidth: int = DEFAULT_WAV_SAMPWIDTH,
    target_framerate: int = DEFAULT_WAV_FRAMERATE,
) -> Tuple[bytes, float]:
    """Decode WAV bytes into canonical mono PCM frames and duration in seconds."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        fr = wf.getframerate()
        nframes = wf.getnframes()
        frames = wf.readframes(nframes)
    if nframes <= 0 or not frames:
        return b"", 0.0

    # Handle trivial stereo->mono downmix for 16-bit PCM if ever needed
    if ch == 2 and sw == 2 and target_channels == 1:
        samples = struct.unpack(f"<{len(frames) // 2}h", frames)
        mono = [((samples[i] + samples[i + 1]) // 2) for i in range(0, len(samples) - 1, 2)]
        frames = struct.pack(f"<{len(mono)}h", *mono)
        ch = 1

    # Linear nearest-sample rate resample if framerate differs from canonical 24kHz
    if fr != target_framerate and sw == 2 and ch == 1 and fr > 0:
        src_samples = struct.unpack(f"<{len(frames) // 2}h", frames)
        src_len = len(src_samples)
        if src_len > 0:
            dst_len = max(1, int(round(src_len * (target_framerate / float(fr)))))
            dst_samples = [
                src_samples[min(src_len - 1, int(i * fr / float(target_framerate)))]
                for i in range(dst_len)
            ]
            frames = struct.pack(f"<{dst_len}h", *dst_samples)
            duration_sec = dst_len / float(target_framerate)
            return frames, duration_sec

    duration_sec = nframes / float(fr or target_framerate)
    if sw != target_sampwidth:
        # Fallback silence matching duration if sample width is non-16bit
        pad_len = int(round(duration_sec * target_framerate))
        frames = b"\x00\x00" * pad_len
    return frames, duration_sec


def _chunk_caption_lines(text: str, *, max_chars: int = 72) -> List[str]:
    clean = re.sub(r"\s+", " ", (text or "")).strip()
    if not clean:
        return []
    raw_parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(clean) if p.strip()]
    if not raw_parts:
        raw_parts = [clean]

    chunks: List[str] = []
    for part in raw_parts:
        if len(part) <= max_chars:
            chunks.append(part)
            continue
        words = part.split(" ")
        buf = ""
        for w in words:
            if not buf:
                if len(w) <= max_chars:
                    buf = w
                else:
                    for i in range(0, len(w), max_chars):
                        chunks.append(w[i : i + max_chars])
            elif len(buf) + 1 + len(w) <= max_chars:
                buf = f"{buf} {w}"
            else:
                chunks.append(buf)
                buf = w
        if buf:
            chunks.append(buf)
    return chunks or [clean]


def _stage_chapter_title(round_stage: str) -> str:
    mapping = {
        RoundStage.STAGE_INTRO.value: "Opening Hook · เปิดประเด็น",
        RoundStage.OPENING.value: "Round 1 · เปิดมุมมอง 5 กูรู",
        RoundStage.REBUTTAL.value: "Round 2 · ซักค้านจุดต่อจุด",
        RoundStage.CROSSFIRE.value: "Crossfire · ปะทะตัวต่อตัว",
        RoundStage.USER_INTERJECTION.value: "Q&A · คำถามแทรกจากพอร์ต",
        RoundStage.INTERJECTION_REPLY.value: "Q&A · คำตอบจากกูรู",
        RoundStage.SYNTHESIS.value: "Final Verdict · Hermes ชี้ขาด & Action Plan",
    }
    return mapping.get(round_stage, round_stage)


def _turn_attr(turn: Any, key: str, alt: str = "") -> Any:
    if isinstance(turn, dict):
        return turn.get(key) if key in turn else turn.get(alt)
    val = getattr(turn, key, None)
    if val is None and alt:
        val = getattr(turn, alt, None)
    return val


def build_youtube_production_pack(
    *,
    session_id: str,
    title: str,
    primary_ticker: Optional[str],
    agenda_prompt: str,
    final_rating: str,
    turns: Sequence[Any],
    audio_bytes_by_key: Dict[str, bytes],
    packet: Optional[FrozenContextPacket] = None,
    playbook_dict: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble full-episode WAV, time-synced SRT subtitles, Chapters & Cue Sheet ZIP."""
    persona_map = {p.speaker_id.value: p for p in resolve_personas_from_packet(packet)}

    pcm_segments: List[bytes] = []
    srt_entries: List[str] = []
    chapters: List[Dict[str, Any]] = []
    timeline_cues: List[Dict[str, Any]] = []
    missing_audio_turn_ids: List[str] = []

    cursor_ms = 0
    srt_idx = 1
    prev_stage: Optional[str] = None
    bytes_per_sample = DEFAULT_WAV_CHANNELS * DEFAULT_WAV_SAMPWIDTH

    def _append_silence(sec: float) -> None:
        nonlocal cursor_ms
        if sec <= 0:
            return
        n_samples = int(round(sec * DEFAULT_WAV_FRAMERATE))
        if n_samples <= 0:
            return
        pcm_segments.append(b"\x00" * (n_samples * bytes_per_sample))
        cursor_ms += int(round((n_samples / float(DEFAULT_WAV_FRAMERATE)) * 1000.0))

    crossfire_start_ms: Optional[int] = None
    crossfire_end_ms: Optional[int] = None

    for t_idx, turn in enumerate(turns or ()):
        speaker_id = str(_turn_attr(turn, "speaker_id") or "")
        round_stage = str(_turn_attr(turn, "round_stage") or "")
        turn_id = str(_turn_attr(turn, "turn_id") or f"turn_{t_idx + 1}")
        dialogue = str(_turn_attr(turn, "spoken_dialogue") or "").strip()
        stance = str(_turn_attr(turn, "stance_tag") or "")
        sizing = _turn_attr(turn, "sizing_posture")
        replying_to = _turn_attr(turn, "replying_to_speaker_id", "replying_to")
        bullets = list(_turn_attr(turn, "evidence_bullets") or ())
        cache_key = _turn_attr(turn, "audio_cache_key")

        # Skip silent user text cards without audio if non-voice, or pace cleanly when audio missing
        if prev_stage is not None:
            pause_sec = INTER_STAGE_PAUSE_SEC if round_stage != prev_stage else INTRA_STAGE_PAUSE_SEC
            _append_silence(pause_sec)

        if round_stage != prev_stage:
            chapters.append(
                {
                    "timestamp": _format_yt_chapter_timestamp(cursor_ms),
                    "start_ms": cursor_ms,
                    "round_stage": round_stage,
                    "title": _stage_chapter_title(round_stage),
                }
            )
            prev_stage = round_stage

        turn_start_ms = cursor_ms
        raw_wav = audio_bytes_by_key.get(str(cache_key)) if cache_key else None
        if raw_wav:
            try:
                pcm, dur_sec = _decode_wav_pcm(raw_wav)
            except Exception:
                pcm, dur_sec = b"", 0.0
        else:
            pcm, dur_sec = b"", 0.0

        if not pcm or dur_sec <= 0.0:
            dur_sec = _estimate_spoken_duration_sec(dialogue)
            n_samples = int(round(dur_sec * DEFAULT_WAV_FRAMERATE))
            pcm = b"\x00" * (n_samples * bytes_per_sample)
            if speaker_id != "user":
                missing_audio_turn_ids.append(turn_id)

        pcm_segments.append(pcm)
        dur_ms = max(500, int(round(dur_sec * 1000.0)))
        cursor_ms = turn_start_ms + dur_ms

        if round_stage == RoundStage.CROSSFIRE.value:
            if crossfire_start_ms is None:
                crossfire_start_ms = turn_start_ms
            crossfire_end_ms = cursor_ms

        spec = persona_map.get(speaker_id)
        speaker_label = spec.short_name if spec else ("คุณ (PM)" if speaker_id == "user" else speaker_id)

        # Split spoken dialogue into proportional timed SRT cues
        chunks = _chunk_caption_lines(dialogue, max_chars=72)
        total_chars = sum(len(c) for c in chunks) or 1
        sub_cursor = turn_start_ms
        for c_i, chunk in enumerate(chunks):
            if c_i == len(chunks) - 1:
                sub_end = cursor_ms
            else:
                slice_ms = max(600, int(round(dur_ms * (len(chunk) / float(total_chars)))))
                sub_end = min(cursor_ms, sub_cursor + slice_ms)
            srt_entries.append(
                f"{srt_idx}\n"
                f"{_format_srt_timestamp(sub_cursor)} --> {_format_srt_timestamp(sub_end)}\n"
                f"[{speaker_label}] {chunk}\n"
            )
            srt_idx += 1
            sub_cursor = sub_end

        timeline_cues.append(
            {
                "turn_id": turn_id,
                "start_ms": turn_start_ms,
                "end_ms": cursor_ms,
                "start_tc": _format_yt_chapter_timestamp(turn_start_ms),
                "round_stage": round_stage,
                "speaker_id": speaker_id,
                "speaker_label": speaker_label,
                "replying_to": replying_to,
                "stance_tag": stance,
                "sizing_posture": sizing,
                "spoken_dialogue": dialogue,
                "evidence_bullets": bullets,
            }
        )

    # Encode canonical stitched WAV
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as out_wf:
        out_wf.setnchannels(DEFAULT_WAV_CHANNELS)
        out_wf.setsampwidth(DEFAULT_WAV_SAMPWIDTH)
        out_wf.setframerate(DEFAULT_WAV_FRAMERATE)
        out_wf.writeframes(b"".join(pcm_segments))
    full_wav_bytes = wav_buf.getvalue()

    srt_text = "\n".join(srt_entries).strip() + ("\n" if srt_entries else "")
    total_duration_sec = round(cursor_ms / 1000.0, 2)

    ticker_lbl = f"${primary_ticker}" if primary_ticker else "Portfolio & Macro"
    pb = playbook_dict or {}
    verdict_text = str(pb.get("executive_verdict") or "")
    kill_switches = list(pb.get("kill_switches") or [])
    tension_axes = list(pb.get("tension_axes") or [])
    ref_price = (
        f"{packet.instrument_anchor.current_price} {packet.instrument_anchor.currency}"
        if packet and packet.instrument_anchor and packet.instrument_anchor.current_price
        else "N/A"
    )

    chapter_lines = "\n".join(f"{ch['timestamp']} {ch['title']}" for ch in chapters) or "00:00 บทนำ"
    cue_rows = "\n".join(
        f"| `{c['start_tc']}` | `{c['round_stage']}` | **{c['speaker_label']}**"
        + (f" (↩ @{c['replying_to']})" if c.get("replying_to") else "")
        + f" | `{c['stance_tag']}`"
        + (f" / `{c['sizing_posture']}`" if c.get("sizing_posture") else "")
        + f" | {' ; '.join(str(b) for b in c['evidence_bullets'][:2]) or '-'} |"
        for c in timeline_cues
    )

    shorts_note = (
        f"- **ช่วงแนะนำสำหรับ YouTube Shorts / Reels (Crossfire):** "
        f"`{_format_yt_chapter_timestamp(crossfire_start_ms)}` – `{_format_yt_chapter_timestamp(crossfire_end_ms or crossfire_start_ms)}`"
        if crossfire_start_ms is not None
        else "- **ช่วงแนะนำสำหรับ Shorts:** เลือกช่วง Rebuttal ที่ Stance ขัดกันแรงสุดใน Cue Sheet"
    )

    tension_bullets = "\n".join(
        f"- **{ax.get('topic', 'ประเด็นปะทะ')}:** Bull ({ax.get('bull_side_speaker', '')}) = {ax.get('bull_argument', '')} VS Bear ({ax.get('bear_side_speaker', '')}) = {ax.get('bear_argument', '')}"
        for ax in tension_axes
        if isinstance(ax, dict)
    ) or "- ดูรายละเอียดใน Timeline Cue Sheet"

    kill_bullets = "\n".join(f"- {ks}" for ks in kill_switches) or "- โปรดติดตามงบการเงินไตรมาสถัดไป"
    missing_warn = (
        f"\n> ⚠️ **หมายเหตุไฟล์เสียง:** เทิร์นต่อไปนี้ยังไม่มีไฟล์เสียงจาก TTS และเติมช่วงเงียบแทนเวลาพูดชั่วคราว: `{', '.join(missing_audio_turn_ids)}`\n"
        if missing_audio_turn_ids
        else ""
    )

    production_kit_md = f"""# YouTube Production Kit — {title}

- **Session ID:** `{session_id}`
- **หัวข้อ / หุ้นหลัก:** `{ticker_lbl}` (Reference Price: `{ref_price}`)
- **มติสุดท้าย (Hermes Verdict):** `{final_rating}`
- **ความยาวเสียงรวม:** `{_format_yt_chapter_timestamp(cursor_ms)}` (`{total_duration_sec:.1f}` วินาที)
{missing_warn}
---

## 1. ตัวเลือกชื่อคลิป YouTube & Thumbnail Hook
1. **ศึกชี้ชะตา {ticker_lbl}! 5 กูรูซัดกันเดือด ซื้อ-รอ-หรือหนี? (สรุปมติ {final_rating})**
2. **เจาะลึก {ticker_lbl} รอบด้าน: Value ปะทะ Growth กลางโต๊ะกลม (ตัวเลขจริงไม่อิงมโน)**
3. **เปิดจุดตายและโอกาสของ {ticker_lbl} ที่ราคา {ref_price} | Trinity Guru Crossfire**

- **ข้อความบนหน้าปก (Thumbnail):** `{ticker_lbl}: {final_rating}!` / `ซัดกันยับ 5 มุมมอง`
{shorts_note}

---

## 2. YouTube Chapters (ก๊อปปี้วางใน Description ได้เลย)
```text
{chapter_lines}
```

---

## 3. Description & Pinned Comment (พร้อม Disclaimer)
**วาระประชุม:** {agenda_prompt}

**สรุปผลชี้ขาด ({final_rating}):** {verdict_text or 'รอดูรายละเอียดการตัดสินใจในคลิป'}

**จุดปะทะหลักในวงเสวนา:**
{tension_bullets}

**Kill-Switches (จุดเตือนยอมแพ้):**
{kill_bullets}

> *Disclaimer: เนื้อหานี้จัดทำขึ้นเพื่อการศึกษาและวิเคราะห์ข้อมูลเชิงปริมาณประกอบการตัดสินใจเท่านั้น ไม่ใช่คำแนะนำการซื้อขายหลักทรัพย์ส่วนบุคคล ผู้ลงทุนควรศึกษาข้อมูลและประเมินความเสี่ยงตามพอร์ตของตนเอง*

---

## 4. Shot-by-Shot Visual & B-Roll Cue Sheet (สำหรับ CapCut / Premiere / DaVinci)
| Timecode | Round Stage | Speaker | Stance / Sizing | On-Screen Overlay (`[FACT]` / `[MATH]`) |
|---|---|---|---|---|
{cue_rows}
"""

    timeline_manifest = {
        "session_id": session_id,
        "title": title,
        "primary_ticker": primary_ticker,
        "final_rating": final_rating,
        "total_duration_ms": cursor_ms,
        "total_duration_sec": total_duration_sec,
        "sample_rate_hz": DEFAULT_WAV_FRAMERATE,
        "chapters": chapters,
        "missing_audio_turn_ids": missing_audio_turn_ids,
        "cues": timeline_cues,
    }

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("audio_full.wav", full_wav_bytes)
        zf.writestr("subtitles.srt", srt_text.encode("utf-8"))
        zf.writestr("youtube_production_kit.md", production_kit_md.encode("utf-8"))
        zf.writestr("timeline_manifest.json", json.dumps(timeline_manifest, ensure_ascii=False, indent=2).encode("utf-8"))

    return {
        "zip_bytes": zip_buf.getvalue(),
        "full_wav_bytes": full_wav_bytes,
        "srt_text": srt_text,
        "production_kit_md": production_kit_md,
        "chapters": chapters,
        "total_duration_sec": total_duration_sec,
        "missing_audio_turn_ids": missing_audio_turn_ids,
    }
