from langsmith import traceable
import os
import re
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool
from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeTranscriptApi,
)

from core.logger import get_logger
from .core import _call_extractor_llm

log = get_logger(__name__)

_TRANSCRIPT_CHAR_LIMIT = 20_000
_ytt_api = YouTubeTranscriptApi()
_YOUTUBE_SUMMARIES_PATH = (
    Path(os.getenv("OBSIDIAN_VAULT_PATH", "./memories"))
    / "30_Knowledge_Base/YouTube_Summaries"
)


class TranscriptUnavailable(Exception):
    """ไม่มี Transcript ในคลิป (ทั้งภาษาเป้าหมายและ fallback) — แยกจาก library exception เพื่อหลีกเลี่ยง constructor signature ของ v1.x"""


def _extract_video_id(url_or_id: str) -> str | None:
    """สกัด Video ID (11 chars) จาก YouTube URL ทุกรูปแบบ หรือ ID ตรงๆ"""
    s = url_or_id.strip()
    if re.match(r'^[\w-]{11}$', s):
        return s
    m = re.search(r'(?:v=|/v/|youtu\.be/|/embed/|/shorts/|/live/)([\w-]{11})', s)
    return m.group(1) if m else None


def _find_existing_insight(video_id: str) -> Path | None:
    """คืน Path ของไฟล์ insight ที่มีอยู่แล้ว หรือ None ถ้ายังไม่เคย ingest"""
    if not _YOUTUBE_SUMMARIES_PATH.exists():
        return None
    matches = list(_YOUTUBE_SUMMARIES_PATH.rglob(f"YouTube_Insight_{video_id}_*.md"))
    return matches[0] if matches else None


def _entries_to_text(entries) -> str:
    """แปลง transcript entries เป็น plain text — รองรับทั้ง dict และ Snippet object (v1.x)"""
    parts = []
    for e in entries:
        text = e.get("text", "") if isinstance(e, dict) else getattr(e, "text", "")
        if text:
            parts.append(text)
    return " ".join(parts)


def _get_channel_name(video_id: str) -> str:
    """สกัดชื่อช่องจากหน้าวิดีโอ"""
    import urllib.request
    url = f"https://www.youtube.com/watch?v={video_id}"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        html = urllib.request.urlopen(req).read().decode('utf-8', errors='ignore')
        import re
        m = re.search(r'<link itemprop="name" content="([^"]+)">', html)
        return m.group(1).strip() if m else "Unknown_Channel"
    except Exception:
        return "Unknown_Channel"


def _get_raw_transcript(video_id: str) -> str:
    """ดึง Transcript ดิบ — ลอง Thai/English ก่อน แล้ว fallback ไปภาษาแรกที่มี (v1.x API)"""
    try:
        fetched = _ytt_api.fetch(video_id, languages=["th", "en"])
        return _entries_to_text(fetched)
    except NoTranscriptFound:
        pass

    # Fallback: ใช้ภาษาแรกที่ระบบให้มา
    transcript_list = _ytt_api.list(video_id)
    for t in transcript_list:
        try:
            fetched = t.fetch()
            text = _entries_to_text(fetched)
            if text.strip():
                return text
        except Exception:
            continue

    raise TranscriptUnavailable(f"ไม่พบ Transcript ในคลิปนี้ (ID: {video_id})")


@tool
@traceable(run_type="tool")
def ingest_youtube_transcript(url: str) -> str:
    """ดึง Transcript จากคลิป YouTube แล้วสกัดข้อมูลการลงทุนด้วย LLM (LLM-as-a-Tool pattern)
    รองรับ URL ทุกรูปแบบ (youtube.com/watch, youtu.be, /shorts/) และ Video ID ตรงๆ
    ใช้ OPENROUTER_API_KEY ใน .env — Return Markdown พร้อม YAML frontmatter ไม่บันทึกไฟล์เอง

    Args:
        url: YouTube URL หรือ Video ID เช่น 'https://youtu.be/dQw4w9WgXcQ' หรือ 'dQw4w9WgXcQ'
    """
    today = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1. สกัด Video ID
    video_id = _extract_video_id(url)
    if not video_id:
        return f"ERROR: ไม่สามารถสกัด Video ID จาก '{url}' — ตรวจสอบว่า URL ถูกต้อง"

    # 2. ตรวจ Duplicate — หยุดทันทีถ้าเคย ingest แล้ว
    existing = _find_existing_insight(video_id)
    if existing:
        return f"[DUPLICATE] | video_id={video_id} | ไฟล์ {existing.name} มีอยู่แล้วใน Vault — ข้ามการประมวลผล"

    # 3. ดึง Transcript
    try:
        raw = _get_raw_transcript(video_id)
    except TranscriptsDisabled:
        log.warning("YouTube transcript disabled | video_id=%s", video_id)
        return (
            f"ERROR: คลิปนี้ปิดใช้งาน Transcript/Subtitle (ID: {video_id}) — "
            "ผู้อัปโหลดได้ปิดฟีเจอร์นี้ไว้"
        )
    except (NoTranscriptFound, TranscriptUnavailable):
        log.warning("YouTube transcript not found | video_id=%s", video_id)
        return (
            f"ERROR: ไม่พบ Transcript ในคลิปนี้ (ID: {video_id}) — "
            "ลองคลิปอื่นที่มีซับไตเติลอัตโนมัติหรือที่ผู้สร้างเพิ่มไว้"
        )
    except VideoUnavailable as e:
        if "live event" in str(e).lower() or "premieres" in str(e).lower():
            log.warning("YouTube video is upcoming live | video_id=%s", video_id)
            return f"ข้าม: วิดีโอนี้ยังไม่ถึงเวลา Live หรือกำลังรอ Premiere (ID: {video_id})"
        log.warning("YouTube video unavailable | video_id=%s", video_id)
        return f"ERROR: ไม่พบวิดีโอ (ID: {video_id}) — อาจถูกลบ, เป็น Private, หรือ Region-locked"
    except Exception as e:
        if "live event" in str(e).lower() or "premieres" in str(e).lower():
            log.warning("YouTube video is upcoming live | video_id=%s", video_id)
            return f"ข้าม: วิดีโอนี้ยังไม่ถึงเวลา Live หรือกำลังรอ Premiere (ID: {video_id})"
        log.warning("YouTube transcript fetch failed | video_id=%s: %s", video_id, e)
        return f"ERROR: ดึง Transcript ล้มเหลว (ID: {video_id}): {e}"

    if not raw.strip():
        log.warning("YouTube transcript empty | video_id=%s", video_id)
        return f"ERROR: Transcript ว่างเปล่า ไม่มีเนื้อหาที่สกัดได้ (ID: {video_id})"

    # 4. ตัดทอน Transcript ถ้ายาวเกิน token limit
    if len(raw) > _TRANSCRIPT_CHAR_LIMIT:
        raw = raw[:_TRANSCRIPT_CHAR_LIMIT] + "\n...[ตัดทอน — Transcript เกิน 20,000 ตัวอักษร]"

    # 5. เรียก LLM ผ่าน OpenRouter สกัดข้อมูล
    try:
        extracted = _call_extractor_llm(raw, f"YouTube: {video_id}")
    except ValueError as e:
        return f"ERROR: {e}"
    except Exception as e:
        log.warning("YouTube LLM extraction failed | video_id=%s: %s", video_id, e)
        return f"ERROR: LLM Extraction ล้มเหลว (OpenRouter): {e}"

    channel_name = _get_channel_name(video_id)

    # 6. Build Markdown output พร้อม YAML frontmatter
    thumbnail = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
    content = "\n".join([
        "---",
        f"title: YouTube Insight {video_id} {today}",
        "entity_type: youtube_insight",
        f"channel: {channel_name}",
        f"video_id: {video_id}",
        f"source_url: {url}",
        f"image: {thumbnail}",
        f"date: {today}",
        f"last_updated: {now_time}",
        "tags: [youtube, transcript, investment_insight]",
        "---",
        "",
        f"# YouTube Investment Insight — `{video_id}`",
        f"> แหล่งที่มา: {url} | ช่อง: {channel_name}",
        "",
        f'<iframe width="560" height="315" src="https://www.youtube.com/embed/{video_id}" frameborder="0" allowfullscreen></iframe>',
        "",
        extracted,
        "",
        "## หมายเหตุ",
        "",
        "> สกัดข้อมูลจาก YouTube Transcript ผ่าน LLM — ตรวจสอบความถูกต้องก่อนนำไปใช้ตัดสินใจลงทุน",
        "",
    ])

    return content
