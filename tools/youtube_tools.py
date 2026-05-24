import os
import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import httpx
from langchain_core.tools import tool
from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeTranscriptApi,
)

from core.llm_factory import get_llm
from core.logger import get_logger

log = get_logger(__name__)

_OPENROUTER_MODEL = "openai/gpt-oss-120b:free"
_TRANSCRIPT_CHAR_LIMIT = 20_000  # ~5k tokens — เพียงพอสำหรับ LLM สกัดข้อมูลเชิงลึก
_ytt_api = YouTubeTranscriptApi()
_YOUTUBE_SUMMARIES_PATH = (
    Path(os.getenv("OBSIDIAN_VAULT_PATH", "./memories"))
    / "30_Knowledge_Base/YouTube_Summaries"
)


@lru_cache(maxsize=1)
def _get_extractor_llm():
    """Cache LLM + retry wrapper — สร้างครั้งเดียวต่อ process"""
    return get_llm(provider="openrouter", model_name=_OPENROUTER_MODEL).with_retry(
        retry_if_exception_type=(
            httpx.TimeoutException,
            httpx.ConnectError,
            httpx.RemoteProtocolError,
            TimeoutError,
            ConnectionError,
        ),
        wait_exponential_jitter=True,
        stop_after_attempt=3,
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
    matches = list(_YOUTUBE_SUMMARIES_PATH.glob(f"YouTube_Insight_{video_id}_*.md"))
    return matches[0] if matches else None


def _entries_to_text(entries) -> str:
    """แปลง transcript entries เป็น plain text — รองรับทั้ง dict และ Snippet object (v1.x)"""
    parts = []
    for e in entries:
        text = e.get("text", "") if isinstance(e, dict) else getattr(e, "text", "")
        if text:
            parts.append(text)
    return " ".join(parts)


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


_EXTRACTOR_SYSTEM_PROMPT = """คุณคือหุ่นยนต์สกัดข้อมูลการลงทุนอย่างเคร่งครัด หน้าที่ของคุณคือสกัดข้อมูลที่มีคุณค่าจาก Transcript และนำเสนอในรูปแบบ Markdown

[CRITICAL — ภาษา]
Output ทั้งหมด เนื้อหาทั้งหมด และหัวข้อทั้งหมด จะต้องเขียนเป็นภาษาไทยเท่านั้น ห้ามตอบกลับมาเป็นภาษาอังกฤษเด็ดขาด แม้ว่า Transcript ต้นฉบับจะเป็นภาษาอังกฤษก็ตาม

กฎเหล็กที่ห้ามละเมิด:
- ห้ามทักทาย ห้ามลงท้าย ห้ามขึ้นต้นด้วยคำอย่าง "นี่คือสรุป" หรือ "ดังนี้"
- ตัดเนื้อหาสปอนเซอร์ โฆษณา คำขยะ Filler Words และน้ำจิ้มออกทั้งหมด 100%
- ห้ามแสดงความคิดเห็นส่วนตัว ห้ามประเมินว่าข้อมูลดีหรือไม่ดี
- ส่งคืนเฉพาะ Markdown ที่มีเนื้อหาสาระ — ไม่มีบทนำ ไม่มีบทสรุปท้าย
- หากไม่มีข้อมูลสำหรับหัวข้อใด — ข้ามส่วนนั้นทั้งหมด ห้ามสร้างขึ้นมาเอง

โครงสร้างที่ต้องสกัด (เรียงตามนี้เสมอ เฉพาะส่วนที่มีข้อมูลใน Transcript):

## ใจความสำคัญ
สรุปประเด็นสำคัญ 3–5 จุด ในรูปแบบ bullet points กระชับ

## แนวคิดการลงทุน
ไอเดีย thesis กลยุทธ์ หรือมุมมองการลงทุนที่พูดถึง ในรูปแบบ bullet points

## เศรษฐกิจมหภาค
แบ่งย่อยตามประเทศหรือภูมิภาคด้วย sub-header ### เสมอ ใช้ Flag emoji นำหน้า
ตัวอย่างรูปแบบที่ถูกต้อง:
### 🇺🇸 สหรัฐฯ
- Fed คงดอกเบี้ย 4.5%
### 🇨🇳 จีน
- PMI ต่ำกว่า 50 ติดต่อกัน 3 เดือน

## หุ้นและสินทรัพย์
ชื่อหุ้น (พร้อม Ticker ถ้ามี), คริปโต, สินทรัพย์ที่ถูกพูดถึง พร้อม Catalyst หรือเหตุผลที่น่าสนใจ
**สำคัญ:** ทุก Ticker หุ้นที่กล่าวถึงต้อง wrap ด้วย Obsidian wikilink `[[TICKER]]` เพื่อเชื่อม Graph View
ตัวอย่าง: `- **[[NVDA]]** — AI hype + earnings beat`, `- **[[AAPL]]** — iPhone cycle ใหม่`

## ความเสี่ยง
ปัจจัยเสี่ยง ภัยคุกคาม หรือสัญญาณเตือนที่พูดถึง ในรูปแบบ bullet points

## ตัวเลขสำคัญทางเศรษฐกิจ
ตัวเลขเฉพาะเจาะจงที่ถูกกล่าวถึง เช่น GDP%, CPI%, อัตราดอกเบี้ย, Bond Yield, P/E, EPS
รูปแบบ: `- ชื่อตัวเลข: ค่า (บริบทสั้นๆ)`"""


@tool
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
    except VideoUnavailable:
        log.warning("YouTube video unavailable | video_id=%s", video_id)
        return f"ERROR: ไม่พบวิดีโอ (ID: {video_id}) — อาจถูกลบ, เป็น Private, หรือ Region-locked"
    except Exception as e:
        log.warning("YouTube transcript fetch failed | video_id=%s: %s", video_id, e)
        return f"ERROR: ดึง Transcript ล้มเหลว (ID: {video_id}): {e}"

    if not raw.strip():
        log.warning("YouTube transcript empty | video_id=%s", video_id)
        return f"ERROR: Transcript ว่างเปล่า ไม่มีเนื้อหาที่สกัดได้ (ID: {video_id})"

    # 4. ตัดทอน Transcript ถ้ายาวเกิน token limit
    if len(raw) > _TRANSCRIPT_CHAR_LIMIT:
        raw = raw[:_TRANSCRIPT_CHAR_LIMIT] + "\n...[ตัดทอน — Transcript เกิน 20,000 ตัวอักษร]"

    # 5. เรียก LLM ผ่าน OpenRouter สกัดข้อมูล
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        log.error("OPENROUTER_API_KEY missing in environment")
        return "ERROR: ไม่พบ OPENROUTER_API_KEY ใน environment — กรุณาเพิ่ม OPENROUTER_API_KEY=... ใน .env"

    try:
        response = _get_extractor_llm().invoke([
            {"role": "system", "content": _EXTRACTOR_SYSTEM_PROMPT},
            {"role": "user", "content": f"Transcript:\n\n{raw}"},
        ])
        extracted = response.content.strip()
    except Exception as e:
        log.warning("YouTube LLM extraction failed | video_id=%s: %s", video_id, e)
        return f"ERROR: LLM Extraction ล้มเหลว (OpenRouter): {e}"

    # 6. Build Markdown output พร้อม YAML frontmatter
    thumbnail = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
    return "\n".join([
        "---",
        f"title: YouTube Insight {video_id} {today}",
        "entity_type: youtube_insight",
        f"video_id: {video_id}",
        f"source_url: {url}",
        f"image: {thumbnail}",
        f"date: {today}",
        f"last_updated: {now_time}",
        "tags: [youtube, transcript, investment_insight]",
        "---",
        "",
        f"# YouTube Investment Insight — `{video_id}`",
        f"> แหล่งที่มา: {url}",
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
