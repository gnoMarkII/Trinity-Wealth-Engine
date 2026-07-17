import os
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

import httpx

from core.llm_factory import get_llm, detect_provider
from core.logger import get_logger
from core.security import anonymize_pii

log = get_logger(__name__)

_EXTRACTOR_MODEL = os.getenv("EXTRACTOR_MODEL", "gemini-3.1-flash-lite-preview")
_CONTENT_CHAR_LIMIT = 20_000

@lru_cache(maxsize=1)
def _get_extractor_llm():
    """Cache LLM + retry wrapper — สร้างครั้งเดียวต่อ process"""
    provider = detect_provider(_EXTRACTOR_MODEL)
    return get_llm(provider=provider, model_name=_EXTRACTOR_MODEL).with_retry(
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


_EXTRACTOR_SYSTEM_PROMPT = """คุณคือหุ่นยนต์สกัดข้อมูลการลงทุนอย่างเคร่งครัด หน้าที่ของคุณคือสกัดข้อมูลที่มีคุณค่าจากเนื้อหาที่ได้รับและนำเสนอในรูปแบบ Markdown

[CRITICAL — ภาษา]
Output ทั้งหมด เนื้อหาทั้งหมด และหัวข้อทั้งหมด จะต้องเขียนเป็นภาษาไทยเท่านั้น ห้ามตอบกลับมาเป็นภาษาอังกฤษเด็ดขาด แม้ว่าต้นฉบับจะเป็นภาษาอังกฤษก็ตาม

กฎเหล็กที่ห้ามละเมิด:
- ห้ามทักทาย ห้ามลงท้าย ห้ามขึ้นต้นด้วยคำอย่าง "นี่คือสรุป" หรือ "ดังนี้"
- ตัดเนื้อหาสปอนเซอร์ โฆษณา คำขยะ Filler Words และน้ำจิ้มออกทั้งหมด 100%
- ห้ามแสดงความคิดเห็นส่วนตัว ห้ามประเมินว่าข้อมูลดีหรือไม่ดี
- ส่งคืนเฉพาะ Markdown ที่มีเนื้อหาสาระ — ไม่มีบทนำ ไม่มีบทสรุปท้าย
- หากไม่มีข้อมูลสำหรับหัวข้อใด — ข้ามส่วนนั้นทั้งหมด ห้ามสร้างขึ้นมาเอง

โครงสร้างที่ต้องสกัด (เรียงตามนี้เสมอ เฉพาะส่วนที่มีข้อมูลใน content):

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


def _call_extractor_llm(raw_content: str, source_label: str) -> str:
    """เรียก LLM สกัดข้อมูลการลงทุน — shared logic สำหรับทั้ง URL, PDF, และ YouTube Transcript"""
    raw_content, _ = anonymize_pii(raw_content)

    if len(raw_content) > _CONTENT_CHAR_LIMIT:
        raw_content = raw_content[:_CONTENT_CHAR_LIMIT] + f"\n...[ตัดทอน — เนื้อหาเกิน {_CONTENT_CHAR_LIMIT:,} ตัวอักษร]"

    log.info("LLM Call | purpose=article_extraction | model=%s", _EXTRACTOR_MODEL)

    response = _get_extractor_llm().invoke([
        {"role": "system", "content": _EXTRACTOR_SYSTEM_PROMPT},
        {"role": "user", "content": f"Source: {source_label}\n\nContent:\n\n{raw_content}"},
    ])
    
    content = response.content
    if isinstance(content, list):
        content = content[0].get("text", "") if len(content) > 0 and isinstance(content[0], dict) else str(content)
    
    return str(content).strip()


def _build_article_md(extracted: str, source_url: str, title: str, today: str, now_time: str, image: str | None = None) -> str:
    safe_title = title.replace(":", " -").replace("/", "-")[:80]
    image_line = [f"image: {image}"] if image else []
    return "\n".join([
        "---",
        f"title: {safe_title}",
        "entity_type: article_note",
        f"source_url: {source_url}",
        f"publisher: {urlparse(source_url).netloc.replace('www.', '')}",
        *image_line,
        f"date: {today}",
        f"last_updated: {now_time}",
        "tags: [article, investment_insight]",
        "---",
        "",
        f"# {safe_title}",
        f"> แหล่งที่มา: {source_url}",
        "",
        extracted,
        "",
        "## หมายเหตุ",
        "",
        "> สกัดข้อมูลจากบทความผ่าน LLM — ตรวจสอบความถูกต้องก่อนนำไปใช้ตัดสินใจลงทุน",
        "",
    ])
