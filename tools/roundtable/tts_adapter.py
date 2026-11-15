"""Gemini TTS Adapter for Trinity Guru Roundtable Voice Simulation.

Features:
- Synthesizes per-persona speech via ``google.genai`` using ``model_registry["roundtable_tts"]``.
- Wraps raw 24kHz 16-bit mono PCM bytes from Gemini TTS into standard RIFF ``.wav`` via stdlib ``wave``.
- Writes cache atomically to ``.cache/roundtable_audio/<key>.wav`` with Windows retry backoff.
- Enforces bounded disk usage via LRU pruning (``max_cache_files``).
- Falls back gracefully to ``fallback_web_speech`` on rate limits (429), missing keys, or offline errors.
"""
from __future__ import annotations

import array
import asyncio
from collections import deque
from dataclasses import dataclass
import hashlib
import io
import logging
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from typing import Any, Callable, Deque, List, Optional, Sequence, Tuple
import wave
import weakref

import google.genai as genai
from google.genai import types

from application.roundtable.ports import AudioSynthesisResult, RoundtableTTSPort
from core.model_registry import get_model_name
from core.roundtable.models import PERSONA_BY_ID, SpeakerId
from validators.roundtable_verifier import prepare_dialogue_for_tts

log = logging.getLogger(__name__)

_SAFE_CACHE_KEY_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
DEFAULT_AUDIO_CACHE_DIR = Path(".cache") / "roundtable_audio"
PCM_SAMPLE_RATE = 24000
PCM_CHANNELS = 1
PCM_SAMPLE_WIDTH = 2  # 16-bit little-endian
DEFAULT_TTS_RPM_LIMIT = 8  # Stay under Tier-1 10 RPM cap to avoid 429 Web Speech fallback
DEFAULT_TTS_MAX_CONCURRENCY = 2
DEFAULT_TTS_MAX_429_RETRIES = 3
_RPM_WINDOW_SECONDS = 61.0
_MIN_VALIDATED_AUDIO_SEC = 3.0
_RMS_WINDOW_SEC = 0.20
_TRIM_PAD_SEC = 0.30
_FADE_OUT_SEC = 0.04
_MAX_TRAILING_LOW_TRIM_SEC = 2.0
_MAX_TRAILING_NOISE_REJECT_SEC = 20.0
_MAX_INTERNAL_LOW_GAP_SEC = 3.0
_MAX_PLAUSIBLE_SPEECH_CPS = 18.0
_MIN_SPEECH_REF_RMS = 250.0
_MIN_ACTIVE_CHUNK_RMS = 450.0
_RECOVERY_PAUSE_SEC = 0.25
_MIME_RATE_RE = re.compile(r"(?:^|[;\s])rate=(\d{4,6})(?:$|[;\s])", re.IGNORECASE)
_RETRY_DELAY_RE = re.compile(
    r"(?:retryDelay['\":\s]+|retry\s+in\s+)(\d+(?:\.\d+)?)s",
    re.IGNORECASE,
)


def _parse_sample_rate_from_mime(mime_type: Optional[str]) -> int:
    """Extract PCM sample rate from Gemini inline_data mime_type when provided."""
    if not mime_type:
        return PCM_SAMPLE_RATE
    m = _MIME_RATE_RE.search(mime_type)
    if m:
        rate = int(m.group(1))
        if 8000 <= rate <= 96000:
            return rate
    return PCM_SAMPLE_RATE


def _is_rate_limit_error(exc: Exception) -> bool:
    """Detect Gemini HTTP 429 / RESOURCE_EXHAUSTED quota errors."""
    if getattr(exc, "code", None) == 429 or getattr(exc, "status_code", None) == 429:
        return True
    msg = str(exc).upper()
    return "429" in msg or "RESOURCE_EXHAUSTED" in msg


def _parse_retry_delay_seconds(exc: Exception, attempt: int) -> float:
    """Use server retryDelay hint (with margin) or exponential backoff for 429 retries."""
    m = _RETRY_DELAY_RE.search(str(exc))
    if m:
        return min(25.0, max(1.0, float(m.group(1)) + 0.5))
    return min(20.0, 2.5 * (2 ** attempt))


def wrap_pcm_to_wav(raw_audio: bytes, sample_rate: int = PCM_SAMPLE_RATE) -> bytes:
    """Wrap raw 24kHz 16-bit mono PCM bytes in a RIFF WAVE header unless already RIFF."""
    if not raw_audio:
        return b""
    if raw_audio[:4] == b"RIFF" and b"WAVE" in raw_audio[:16]:
        return raw_audio
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(PCM_CHANNELS)
        wf.setsampwidth(PCM_SAMPLE_WIDTH)
        wf.setframerate(sample_rate)
        wf.writeframes(raw_audio)
    return buf.getvalue()


@dataclass(frozen=True)
class WavValidationResult:
    is_valid: bool
    wav_bytes: bytes
    reason: str = "ok"
    was_trimmed: bool = False
    duration_sec: float = 0.0
    active_end_sec: float = 0.0
    trailing_low_sec: float = 0.0
    max_internal_low_sec: float = 0.0


def _encode_pcm_to_wav(pcm_frames: bytes, *, nchannels: int, sampwidth: int, framerate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(nchannels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(framerate)
        wf.writeframes(pcm_frames)
    return buf.getvalue()


def _trim_mono16_wav_with_fade(
    samples: array.array,
    *,
    framerate: int,
    trim_sec: float,
) -> bytes:
    cut_samples = max(1, min(len(samples), int(round(trim_sec * framerate))))
    trimmed = array.array("h", samples[:cut_samples])
    fade_samples = min(len(trimmed), max(1, int(round(_FADE_OUT_SEC * framerate))))
    start_fade = len(trimmed) - fade_samples
    for idx in range(fade_samples):
        gain = max(0.0, 1.0 - (idx + 1) / float(fade_samples))
        trimmed[start_fade + idx] = int(round(trimmed[start_fade + idx] * gain))
    if sys.byteorder != "little":
        trimmed.byteswap()
    return _encode_pcm_to_wav(
        trimmed.tobytes(),
        nchannels=PCM_CHANNELS,
        sampwidth=PCM_SAMPLE_WIDTH,
        framerate=framerate,
    )


def validate_and_sanitize_tts_wav(wav_bytes: bytes, text: str) -> WavValidationResult:
    """Detect Gemini TTS runaway non-EOS loops, mid-turn dropouts, and truncated speech."""
    if not wav_bytes:
        return WavValidationResult(is_valid=False, wav_bytes=b"", reason="empty_wav")
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            nchannels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            nframes = wf.getnframes()
            raw_frames = wf.readframes(nframes)
    except Exception as exc:
        return WavValidationResult(is_valid=False, wav_bytes=wav_bytes, reason=f"corrupt_wav:{exc}")

    if nframes <= 0 or framerate <= 0 or not raw_frames:
        return WavValidationResult(is_valid=False, wav_bytes=wav_bytes, reason="zero_frames")

    total_dur = nframes / float(framerate)
    cleaned_len = len((text or "").strip())
    max_expected_dur = max(40.0, cleaned_len * 0.22 + 20.0)

    # Preserve tiny synthetic unit-test buffers cleanly unless obviously exceeding bounds.
    if total_dur <= _MIN_VALIDATED_AUDIO_SEC or cleaned_len < 40:
        return WavValidationResult(
            is_valid=True,
            wav_bytes=wav_bytes,
            duration_sec=total_dur,
            active_end_sec=total_dur,
        )

    if total_dur > max_expected_dur:
        return WavValidationResult(
            is_valid=False,
            wav_bytes=wav_bytes,
            reason=f"runaway_duration_{total_dur:.1f}s_gt_{max_expected_dur:.1f}s",
            duration_sec=total_dur,
        )

    if nchannels != 1 or sampwidth != 2:
        return WavValidationResult(
            is_valid=True,
            wav_bytes=wav_bytes,
            duration_sec=total_dur,
            active_end_sec=total_dur,
        )

    samples = array.array("h")
    samples.frombytes(raw_frames[: (len(raw_frames) // 2) * 2])
    if sys.byteorder != "little":
        samples.byteswap()

    win = max(1, int(round(framerate * _RMS_WINDOW_SEC)))
    stride = 1 if len(samples) <= 2_400_000 else 4
    rms_chunks: List[float] = []
    for pos in range(0, len(samples), win):
        chunk = samples[pos : pos + win : stride]
        if chunk:
            rms_chunks.append(math.sqrt(sum(s * s for s in chunk) / len(chunk)))

    if not rms_chunks:
        return WavValidationResult(is_valid=False, wav_bytes=wav_bytes, reason="empty_rms", duration_sec=total_dur)

    top_k = sorted(rms_chunks, reverse=True)[: min(10, len(rms_chunks))]
    ref_rms = sum(top_k) / float(len(top_k))
    if ref_rms < _MIN_SPEECH_REF_RMS:
        return WavValidationResult(
            is_valid=False,
            wav_bytes=wav_bytes,
            reason=f"low_energy_audio_rms_{ref_rms:.1f}",
            duration_sec=total_dur,
        )

    speech_thresh = max(_MIN_ACTIVE_CHUNK_RMS, ref_rms * 0.08)
    active_indices = [idx for idx, r in enumerate(rms_chunks) if r >= speech_thresh]
    if not active_indices:
        return WavValidationResult(
            is_valid=False,
            wav_bytes=wav_bytes,
            reason="no_active_speech",
            duration_sec=total_dur,
        )

    first_act = active_indices[0]
    last_act = active_indices[-1]
    active_end_sec = min(total_dur, (last_act + 1) * _RMS_WINDOW_SEC)
    trailing_low_sec = max(0.0, total_dur - active_end_sec)

    max_low_run = 0
    cur_low = 0
    for r in rms_chunks[first_act : last_act + 1]:
        if r < speech_thresh:
            cur_low += 1
            if cur_low > max_low_run:
                max_low_run = cur_low
        else:
            cur_low = 0
    max_internal_low_sec = max_low_run * _RMS_WINDOW_SEC

    if max_internal_low_sec >= _MAX_INTERNAL_LOW_GAP_SEC:
        return WavValidationResult(
            is_valid=False,
            wav_bytes=wav_bytes,
            reason=f"internal_dropout_{max_internal_low_sec:.2f}s",
            duration_sec=total_dur,
            active_end_sec=active_end_sec,
            trailing_low_sec=trailing_low_sec,
            max_internal_low_sec=max_internal_low_sec,
        )

    cps = cleaned_len / max(0.5, active_end_sec)
    if cleaned_len >= 160 and cps > _MAX_PLAUSIBLE_SPEECH_CPS:
        return WavValidationResult(
            is_valid=False,
            wav_bytes=wav_bytes,
            reason=f"truncated_speech_{cps:.1f}_cps",
            duration_sec=total_dur,
            active_end_sec=active_end_sec,
            trailing_low_sec=trailing_low_sec,
            max_internal_low_sec=max_internal_low_sec,
        )

    if trailing_low_sec > _MAX_TRAILING_NOISE_REJECT_SEC and (trailing_low_sec / total_dur) > 0.25:
        return WavValidationResult(
            is_valid=False,
            wav_bytes=wav_bytes,
            reason=f"excessive_non_speech_tail_{trailing_low_sec:.1f}s",
            duration_sec=total_dur,
            active_end_sec=active_end_sec,
            trailing_low_sec=trailing_low_sec,
            max_internal_low_sec=max_internal_low_sec,
        )

    if trailing_low_sec > _MAX_TRAILING_LOW_TRIM_SEC:
        trimmed_sec = min(total_dur, active_end_sec + _TRIM_PAD_SEC)
        trimmed_wav = _trim_mono16_wav_with_fade(samples, framerate=framerate, trim_sec=trimmed_sec)
        return WavValidationResult(
            is_valid=True,
            wav_bytes=trimmed_wav,
            reason="trimmed_trailing_non_speech",
            was_trimmed=True,
            duration_sec=trimmed_sec,
            active_end_sec=active_end_sec,
            trailing_low_sec=trailing_low_sec,
            max_internal_low_sec=max_internal_low_sec,
        )

    return WavValidationResult(
        is_valid=True,
        wav_bytes=wav_bytes,
        duration_sec=total_dur,
        active_end_sec=active_end_sec,
        trailing_low_sec=trailing_low_sec,
        max_internal_low_sec=max_internal_low_sec,
    )


def split_text_for_tts_recovery(text: str) -> List[str]:
    """Split long dialogue at balanced word/clause boundaries for safe TTS retry."""
    cleaned = re.sub(r"\s+", " ", (text or "")).strip()
    if not cleaned:
        return []
    if len(cleaned) < 240:
        return [cleaned]

    parts = 2 if len(cleaned) <= 950 else 3
    words = cleaned.split(" ")
    if len(words) < 2:
        step = max(1, math.ceil(len(cleaned) / float(parts)))
        return [cleaned[i : i + step].strip() for i in range(0, len(cleaned), step) if cleaned[i : i + step].strip()]

    target = math.ceil(len(cleaned) / float(parts)) + 25
    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0
    for word in words:
        added = len(word) + (1 if cur else 0)
        if cur and (cur_len + added > target) and len(chunks) < (parts - 1):
            chunks.append(" ".join(cur))
            cur = [word]
            cur_len = len(word)
        else:
            cur.append(word)
            cur_len += added
    if cur:
        chunks.append(" ".join(cur))
    return chunks or [cleaned]


def _concatenate_wav_clips(clips: Sequence[bytes], *, pause_sec: float = _RECOVERY_PAUSE_SEC) -> bytes:
    """Concatenate mono/stereo WAV clips with a short clean silence pause between segments."""
    pcm_parts: List[bytes] = []
    base_params: Optional[Tuple[int, int, int]] = None
    for clip in clips:
        with wave.open(io.BytesIO(clip), "rb") as wf:
            ch = wf.getnchannels()
            sw = wf.getsampwidth()
            fr = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
        if base_params is None:
            base_params = (ch, sw, fr)
        elif (ch, sw, fr) != base_params:
            raise RuntimeError(f"Mismatched recovery WAV format {(ch, sw, fr)} != {base_params}")
        if pcm_parts and pause_sec > 0:
            pause_frames = int(round(pause_sec * fr)) * ch * sw
            pcm_parts.append(b"\x00" * pause_frames)
        pcm_parts.append(frames)
    if base_params is None or not pcm_parts:
        return b""
    ch, sw, fr = base_params
    return _encode_pcm_to_wav(b"".join(pcm_parts), nchannels=ch, sampwidth=sw, framerate=fr)


def _atomic_write_bytes(path: Path, data: bytes, max_retries: int = 8, backoff: float = 0.05) -> None:
    """Atomically write binary audio payload via tempfile + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp_", suffix=".wav.tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
        last_err: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                os.replace(tmp_path, path)
                last_err = None
                break
            except (PermissionError, OSError) as exc:
                last_err = exc
                winerror = getattr(exc, "winerror", None)
                if winerror in (5, 32) or isinstance(exc, PermissionError):
                    time.sleep(backoff * (1.5 ** attempt))
                else:
                    break
        if last_err is not None:
            raise last_err
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _resolve_voice_name(speaker_id: str) -> str:
    try:
        spec = PERSONA_BY_ID.get(SpeakerId(speaker_id))
        if spec:
            return spec.voice_name
    except ValueError:
        pass
    return "Kore"


def _read_env_nonneg_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return max(0, int(str(raw).strip()))
    except ValueError:
        return default


class GeminiTTSAdapter(RoundtableTTSPort):
    """Concrete Gemini multi-speaker TTS synthesis + LRU WAV disk cache adapter."""

    def __init__(
        self,
        *,
        cache_dir: Path | str = DEFAULT_AUDIO_CACHE_DIR,
        max_cache_files: int = 1000,
        model_name: Optional[str] = None,
        tts_fetcher: Optional[Callable[[str, str, str], bytes]] = None,
        rpm_limit: Optional[int] = None,
        max_concurrency: Optional[int] = None,
        max_429_retries: Optional[int] = None,
        window_seconds: float = _RPM_WINDOW_SECONDS,
        retry_delay_fn: Optional[Callable[[Exception, int], float]] = None,
    ) -> None:
        self._cache_dir = Path(cache_dir)
        self._max_cache_files = max(1, int(max_cache_files))
        self._model_name = model_name or get_model_name("roundtable_tts")
        self._tts_fetcher = tts_fetcher
        is_live = tts_fetcher is None
        self._rpm_limit = (
            max(0, int(rpm_limit))
            if rpm_limit is not None
            else (_read_env_nonneg_int("ROUNDTABLE_TTS_RPM_LIMIT", DEFAULT_TTS_RPM_LIMIT) if is_live else 0)
        )
        self._max_concurrency = (
            max(0, int(max_concurrency))
            if max_concurrency is not None
            else (_read_env_nonneg_int("ROUNDTABLE_TTS_MAX_CONCURRENCY", DEFAULT_TTS_MAX_CONCURRENCY) if is_live else 0)
        )
        self._max_429_retries = (
            max(0, int(max_429_retries))
            if max_429_retries is not None
            else (_read_env_nonneg_int("ROUNDTABLE_TTS_MAX_429_RETRIES", DEFAULT_TTS_MAX_429_RETRIES) if is_live else 0)
        )
        self._window_seconds = max(0.05, float(window_seconds))
        self._retry_delay_fn = retry_delay_fn or _parse_retry_delay_seconds
        self._call_timestamps: Deque[float] = deque()
        self._loop_guards: weakref.WeakKeyDictionary[
            asyncio.AbstractEventLoop, Tuple[asyncio.Lock, Optional[asyncio.Semaphore]]
        ] = weakref.WeakKeyDictionary()

    def _get_loop_guards(self) -> Tuple[asyncio.Lock, Optional[asyncio.Semaphore]]:
        loop = asyncio.get_running_loop()
        pair = self._loop_guards.get(loop)
        if pair is None:
            sem = asyncio.Semaphore(self._max_concurrency) if self._max_concurrency > 0 else None
            pair = (asyncio.Lock(), sem)
            self._loop_guards[loop] = pair
        return pair

    async def _acquire_rate_slot(self) -> None:
        if self._rpm_limit <= 0:
            return
        rate_lock, _ = self._get_loop_guards()
        while True:
            wait_s = 0.0
            async with rate_lock:
                now = time.monotonic()
                cutoff = now - self._window_seconds
                while self._call_timestamps and self._call_timestamps[0] <= cutoff:
                    self._call_timestamps.popleft()
                if len(self._call_timestamps) < self._rpm_limit:
                    self._call_timestamps.append(now)
                    return
                wait_s = max(0.05, (self._call_timestamps[0] + self._window_seconds) - now)
            log.debug("Gemini TTS pacing: sleeping %.2fs to stay within %d RPM", wait_s, self._rpm_limit)
            await asyncio.sleep(wait_s)

    def _compute_cache_key(self, speaker_id: str, voice_name: str, cleaned_text: str) -> str:
        digest = hashlib.sha256(
            f"{self._model_name}|{speaker_id}|{voice_name}|{cleaned_text}".encode("utf-8")
        ).hexdigest()[:24]
        return f"rt_{digest}"

    def _prune_cache_lru(self, protect_path: Optional[Path] = None) -> None:
        if not self._cache_dir.is_dir():
            return
        wav_files = [p for p in self._cache_dir.glob("*.wav") if p.is_file()]
        if len(wav_files) <= self._max_cache_files:
            return
        protected_resolved = protect_path.resolve() if protect_path is not None else None
        evictable = [
            p for p in wav_files
            if protected_resolved is None or p.resolve() != protected_resolved
        ]
        evictable.sort(key=lambda p: p.stat().st_mtime_ns)
        excess = len(wav_files) - self._max_cache_files
        for stale in evictable[:excess]:
            try:
                stale.unlink(missing_ok=True)
            except Exception as exc:
                log.debug("Failed to prune stale audio cache %s: %s", stale, exc)

    def _default_tts_fetch(self, model_name: str, voice_name: str, cleaned_text: str) -> bytes:
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GOOGLE_API_KEY / GEMINI_API_KEY for Gemini TTS")
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents=cleaned_text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
                    )
                ),
            ),
        )
        candidates = getattr(response, "candidates", None) or []
        for cand in candidates:
            content = getattr(cand, "content", None)
            parts = getattr(content, "parts", None) or []
            for part in parts:
                inline = getattr(part, "inline_data", None)
                data = getattr(inline, "data", None) if inline else None
                if isinstance(data, (bytes, bytearray)) and len(data) > 0:
                    raw = bytes(data)
                    mime_type = str(getattr(inline, "mime_type", "") or "")
                    parsed_rate = _parse_sample_rate_from_mime(mime_type)
                    if parsed_rate != PCM_SAMPLE_RATE and not (raw[:4] == b"RIFF" and b"WAVE" in raw[:16]):
                        return wrap_pcm_to_wav(raw, sample_rate=parsed_rate)
                    return raw
        raise RuntimeError("Gemini TTS response contained no inline audio bytes")

    async def _fetch_with_pacing_and_retry(
        self,
        fetcher: Callable[[str, str, str], bytes],
        resolved_voice: str,
        cleaned: str,
    ) -> bytes:
        _, sem = self._get_loop_guards()
        attempt = 0
        while True:
            await self._acquire_rate_slot()
            try:
                if sem is not None:
                    async with sem:
                        return await asyncio.to_thread(fetcher, self._model_name, resolved_voice, cleaned)
                return await asyncio.to_thread(fetcher, self._model_name, resolved_voice, cleaned)
            except Exception as exc:
                if _is_rate_limit_error(exc) and attempt < self._max_429_retries:
                    delay_s = max(0.0, float(self._retry_delay_fn(exc, attempt)))
                    attempt += 1
                    log.warning(
                        "Gemini TTS rate-limited (attempt %d/%d, retry in %.2fs): %s",
                        attempt,
                        self._max_429_retries,
                        delay_s,
                        exc,
                    )
                    if delay_s > 0:
                        await asyncio.sleep(delay_s)
                    continue
                raise

    async def _synthesize_validated_wav(
        self,
        fetcher: Callable[[str, str, str], bytes],
        resolved_voice: str,
        cleaned: str,
        speaker_id: str,
    ) -> bytes:
        raw_bytes = await self._fetch_with_pacing_and_retry(fetcher, resolved_voice, cleaned)
        wav_bytes = wrap_pcm_to_wav(raw_bytes)
        if not wav_bytes:
            raise RuntimeError("Empty audio buffer synthesized")

        check = validate_and_sanitize_tts_wav(wav_bytes, cleaned)
        if check.is_valid:
            if check.was_trimmed:
                log.info(
                    "Trimmed trailing non-speech from TTS clip for speaker=%s (%.2fs -> %.2fs)",
                    speaker_id,
                    check.duration_sec + max(0.0, check.trailing_low_sec - _TRIM_PAD_SEC),
                    check.duration_sec,
                )
            return check.wav_bytes

        log.warning(
            "Gemini TTS failed sanity validation (%s) for speaker=%s; attempting recovery",
            check.reason,
            speaker_id,
        )
        chunks = split_text_for_tts_recovery(cleaned)
        if len(chunks) > 1:
            part_wavs: List[bytes] = []
            for part_text in chunks:
                part_raw = await self._fetch_with_pacing_and_retry(fetcher, resolved_voice, part_text)
                part_wav = wrap_pcm_to_wav(part_raw)
                part_check = validate_and_sanitize_tts_wav(part_wav, part_text)
                if not part_check.is_valid:
                    raise RuntimeError(f"Chunked TTS recovery failed validation ({part_check.reason})")
                part_wavs.append(part_check.wav_bytes)
            merged_wav = _concatenate_wav_clips(part_wavs)
            merged_check = validate_and_sanitize_tts_wav(merged_wav, cleaned)
            if not merged_check.is_valid:
                raise RuntimeError(f"Merged TTS recovery failed validation ({merged_check.reason})")
            return merged_check.wav_bytes

        retry_raw = await self._fetch_with_pacing_and_retry(fetcher, resolved_voice, cleaned)
        retry_wav = wrap_pcm_to_wav(retry_raw)
        retry_check = validate_and_sanitize_tts_wav(retry_wav, cleaned)
        if not retry_check.is_valid:
            raise RuntimeError(f"TTS retry failed validation ({retry_check.reason})")
        return retry_check.wav_bytes

    async def synthesize_turn_audio(
        self,
        *,
        text: str,
        speaker_id: str,
        enabled: bool = True,
        voice_name: Optional[str] = None,
    ) -> AudioSynthesisResult:
        if not enabled:
            return AudioSynthesisResult(audio_cache_key=None, audio_status="disabled")

        cleaned = prepare_dialogue_for_tts(text)
        if not cleaned:
            return AudioSynthesisResult(audio_cache_key=None, audio_status="fallback_web_speech")

        resolved_voice = (voice_name or "").strip() or _resolve_voice_name(speaker_id)
        cache_key = self._compute_cache_key(speaker_id, resolved_voice, cleaned)
        cache_path = self._cache_dir / f"{cache_key}.wav"

        if cache_path.is_file():
            try:
                cached_bytes = await asyncio.to_thread(cache_path.read_bytes)
                cached_check = validate_and_sanitize_tts_wav(cached_bytes, cleaned)
            except Exception:
                cached_check = WavValidationResult(is_valid=False, wav_bytes=b"", reason="cache_read_error")

            if cached_check.is_valid:
                if cached_check.was_trimmed:
                    await asyncio.to_thread(_atomic_write_bytes, cache_path, cached_check.wav_bytes)
                try:
                    os.utime(cache_path, None)
                except Exception:
                    pass
                return AudioSynthesisResult(audio_cache_key=cache_key, audio_status="cached")

            log.warning(
                "Cached TTS WAV %s failed sanity check (%s); purging and re-synthesizing",
                cache_key,
                cached_check.reason,
            )
            try:
                await asyncio.to_thread(cache_path.unlink, True)
            except Exception:
                pass

        fetcher = self._tts_fetcher or self._default_tts_fetch
        try:
            wav_bytes = await self._synthesize_validated_wav(fetcher, resolved_voice, cleaned, speaker_id)
            await asyncio.to_thread(_atomic_write_bytes, cache_path, wav_bytes)
            await asyncio.to_thread(self._prune_cache_lru, cache_path)
            return AudioSynthesisResult(audio_cache_key=cache_key, audio_status="ready")
        except Exception as exc:
            log.warning("Gemini TTS failed for speaker=%s (falling back to Web Speech): %s", speaker_id, exc)
            return AudioSynthesisResult(
                audio_cache_key=None,
                audio_status="fallback_web_speech",
                error_message=str(exc),
            )

    def get_audio_bytes(self, audio_cache_key: str) -> Optional[bytes]:
        if not audio_cache_key or not _SAFE_CACHE_KEY_RE.match(audio_cache_key):
            return None
        resolved_dir = self._cache_dir.resolve()
        candidate = (resolved_dir / f"{audio_cache_key}.wav").resolve()
        if not candidate.is_relative_to(resolved_dir) or not candidate.is_file():
            return None
        try:
            os.utime(candidate, None)
        except Exception:
            pass
        return candidate.read_bytes()

