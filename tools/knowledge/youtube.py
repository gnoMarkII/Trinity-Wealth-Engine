from langsmith import traceable
import os
import re
import subprocess
import tempfile
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


def _get_ytdlp_subtitles(video_id: str) -> str | None:
    """Tier 2: ดึงข้อความซับไตเติลผ่าน yt-dlp โดยไม่ดาวน์โหลดไฟล์วิดีโอ (--skip-download)"""
    url = f"https://www.youtube.com/watch?v={video_id}"
    with tempfile.TemporaryDirectory() as tmp_dir:
        cmd = [
            "yt-dlp",
            "--skip-download",
            "--write-sub",
            "--write-auto-sub",
            "--sub-langs",
            "th,en,.*",
            "--sub-format",
            "vtt/srt/best",
            "-o",
            f"{tmp_dir}/%(id)s",
            url,
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if res.returncode != 0:
                log.warning("yt-dlp subtitle fetch failed | video_id=%s: %s", video_id, res.stderr)
                return None
            sub_files = list(Path(tmp_dir).glob(f"{video_id}*.*"))
            if not sub_files:
                return None
            sub_files.sort(key=lambda p: 0 if ".th." in p.name else (1 if ".en." in p.name else 2))
            content = sub_files[0].read_text(encoding="utf-8", errors="ignore")
            lines = []
            for line in content.splitlines():
                line = line.strip()
                if not line or "-->" in line or line.startswith("WEBVTT") or re.match(r"^\d+$", line):
                    continue
                clean_line = re.sub(r"<[^>]+>", "", line).strip()
                if clean_line and (not lines or lines[-1] != clean_line):
                    lines.append(clean_line)
            text = " ".join(lines)
            return text if text.strip() else None
        except Exception as e:
            log.warning("yt-dlp subtitle execution error | video_id=%s: %s", video_id, e)
            return None


def _extract_via_gemini_url_direct(url: str, video_id: str) -> str:
    """Tier 3: ส่ง YouTube URL ให้ Google Gemini ดู/ฟังโดยตรง (Zero-Download Multimodal)"""
    from google import genai
    from google.genai.types import Part
    from .core import _EXTRACTOR_SYSTEM_PROMPT

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("ไม่พบ GOOGLE_API_KEY หรือ GEMINI_API_KEY สำหรับใช้งาน Gemini Multimodal")

    extractor_model = os.getenv("EXTRACTOR_MODEL", "gemini-2.0-flash")
    model_name = extractor_model if "gemini" in extractor_model.lower() else "gemini-2.0-flash"

    client = genai.Client(api_key=api_key)
    prompt = f"Source: YouTube: {video_id}\n\nกรุณาดู/ฟังวิดีโอ YouTube นี้ และสกัดข้อมูลการลงทุนตามคำสั่งใน System Prompt"

    response = client.models.generate_content(
        model=model_name,
        contents=[
            Part.from_uri(file_uri=url, mime_type="video/mp4"),
            prompt,
        ],
        config={
            "system_instruction": _EXTRACTOR_SYSTEM_PROMPT,
            "temperature": 0.0,
        },
    )
    return str(response.text).strip()


@tool
def ingest_youtube_transcript(url: str) -> str:
    """ดึงซับไตเติ้ล (Transcript) จากวิดีโอ YouTube และแปลงเป็น Markdown

    [Usage/When to use]
    ใช้เมื่อผู้ใช้ส่ง URL ของ YouTube ให้สรุป หรือสั่งให้ดึงข้อมูลจากคลิป YouTube
    - สามารถดึงข้อมูล Channel Name, วันที่เผยแพร่, และ Title มาใส่ใน YAML ให้อัตโนมัติ

    [Caution]
    - **Live Event**: หากวิดีโอยังเป็น Live Stream ที่ยังไม่จบ หรือรอ Live จะไม่สามารถดึง Transcript ได้
    - เครื่องมือนี้แค่ส่งคืนข้อความ Markdown (ไม่บันทึกไฟล์เอง)

    Args:
        url (str): ลิงก์วิดีโอ YouTube หรือ Video ID (เช่น 'https://www.youtube.com/watch?v=RjOEkDIZFZU' หรือ 'RjOEkDIZFZU')

    Returns:
        str: เนื้อหา Transcript ในรูปแบบ Markdown พร้อม YAML Frontmatter
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

    # 3. ดึงเนื้อหาและสกัดข้อมูล (3-Tier Zero-Download Pipeline)
    extracted = ""
    extraction_method = ""

    # Tier 1: youtube-transcript-api
    try:
        raw = _get_raw_transcript(video_id)
        if raw and raw.strip():
            if len(raw) > _TRANSCRIPT_CHAR_LIMIT:
                raw = raw[:_TRANSCRIPT_CHAR_LIMIT] + "\n...[ตัดทอน — Transcript เกิน 20,000 ตัวอักษร]"
            extracted = _call_extractor_llm(raw, f"YouTube: {video_id}")
            extraction_method = "Tier 1: YouTube Transcript API"
    except VideoUnavailable as e:
        if "live event" in str(e).lower() or "premieres" in str(e).lower():
            log.warning("YouTube video is upcoming live | video_id=%s", video_id)
            return f"ข้าม: วิดีโอนี้ยังไม่ถึงเวลา Live หรือกำลังรอ Premiere (ID: {video_id})"
        log.warning("YouTube video unavailable | video_id=%s", video_id)
        return f"ERROR: ไม่พบวิดีโอ (ID: {video_id}) — อาจถูกลบ, เป็น Private, หรือ Region-locked"
    except (NoTranscriptFound, TranscriptsDisabled, TranscriptUnavailable):
        log.info("Tier 1 transcript not found/disabled, trying Tier 2 yt-dlp subtitle scraper")
    except Exception as e:
        msg = str(e)
        if "live event" in msg.lower() or "premieres" in msg.lower():
            log.warning("YouTube video is upcoming live | video_id=%s", video_id)
            return f"ข้าม: วิดีโอนี้ยังไม่ถึงเวลา Live หรือกำลังรอ Premiere (ID: {video_id})"
        log.warning("Tier 1 transcript fetch failed (%s), trying Tier 2 yt-dlp subtitle scraper", e)

    # Tier 2: yt-dlp Subtitle Scraper
    if not extracted:
        try:
            ytdlp_text = _get_ytdlp_subtitles(video_id)
            if ytdlp_text and ytdlp_text.strip():
                if len(ytdlp_text) > _TRANSCRIPT_CHAR_LIMIT:
                    ytdlp_text = ytdlp_text[:_TRANSCRIPT_CHAR_LIMIT] + "\n...[ตัดทอน — Transcript เกิน 20,000 ตัวอักษร]"
                extracted = _call_extractor_llm(ytdlp_text, f"YouTube: {video_id}")
                extraction_method = "Tier 2: yt-dlp Subtitle Scraper"
        except Exception as e:
            log.warning("Tier 2 yt-dlp subtitle fetch failed (%s), trying Tier 3 Gemini Multimodal", e)

    # Tier 3: Zero-Download Gemini Direct YouTube URL Understanding
    if not extracted:
        try:
            extracted = _extract_via_gemini_url_direct(url, video_id)
            extraction_method = "Tier 3: Gemini Direct YouTube Multimodal (Zero-Download)"
        except Exception as e:
            log.warning("YouTube 3-Tier extraction all failed | video_id=%s: %s", video_id, e)
            return f"ERROR: ดึงข้อมูลและสรุปเนื้อหาจาก YouTube ล้มเหลวทั้ง 3 ขั้นตอน (ID: {video_id}): {e}"

    if not extracted or not extracted.strip():
        return f"ERROR: ไม่สามารถสกัดข้อมูลจากวิดีโอนี้ได้ (ID: {video_id})"

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
        f"> แหล่งที่มา: {url} | ช่อง: {channel_name} | วิธีสกัดข้อมูล: {extraction_method}",
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
