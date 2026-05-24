import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import httpx
from langchain_core.tools import tool

from core.llm_factory import get_llm
from core.logger import get_logger

log = get_logger(__name__)

_OPENROUTER_MODEL = "openai/gpt-oss-120b:free"
_CONTENT_CHAR_LIMIT = 20_000

_BOOKS_PATH = Path(os.getenv("OBSIDIAN_VAULT_PATH", "./memories")) / "30_Knowledge_Base/Books"
_ARTICLES_PATH = Path(os.getenv("OBSIDIAN_VAULT_PATH", "./memories")) / "30_Knowledge_Base/Articles"


@lru_cache(maxsize=1)
def _get_extractor_llm():
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


def _find_existing_article(source_id: str) -> Path | None:
    """คืน Path ของไฟล์ที่มีอยู่แล้ว หรือ None ถ้ายังไม่เคย ingest"""
    for folder in (_ARTICLES_PATH,):
        if not folder.exists():
            continue
        matches = list(folder.glob(f"*{source_id}*.md"))
        if matches:
            return matches[0]
    return None


def _call_extractor_llm(raw_content: str, source_label: str) -> str:
    """เรียก LLM สกัดข้อมูลการลงทุน — shared logic สำหรับทั้ง URL และ PDF"""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("ไม่พบ OPENROUTER_API_KEY ใน environment")

    if len(raw_content) > _CONTENT_CHAR_LIMIT:
        raw_content = raw_content[:_CONTENT_CHAR_LIMIT] + f"\n...[ตัดทอน — เนื้อหาเกิน {_CONTENT_CHAR_LIMIT:,} ตัวอักษร]"

    response = _get_extractor_llm().invoke([
        {"role": "system", "content": _EXTRACTOR_SYSTEM_PROMPT},
        {"role": "user", "content": f"Source: {source_label}\n\nContent:\n\n{raw_content}"},
    ])
    return response.content.strip()


def _build_article_md(extracted: str, source_url: str, title: str, today: str, now_time: str, image: str | None = None) -> str:
    safe_title = title.replace(":", " -").replace("/", "-")[:80]
    image_line = [f"image: {image}"] if image else []
    return "\n".join([
        "---",
        f"title: {safe_title}",
        "entity_type: article_note",
        f"source_url: {source_url}",
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


@tool
def ingest_article_url(url: str) -> str:
    """ดึงเนื้อหาจาก URL บทความและสกัดข้อมูลการลงทุนด้วย LLM
    รองรับบทความทั่วไป, สื่อการเงิน, บล็อก — Return Markdown พร้อม YAML frontmatter ไม่บันทึกไฟล์เอง

    Args:
        url: URL ของบทความที่ต้องการ ingest เช่น 'https://example.com/article'
    """
    today = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        import trafilatura
    except ImportError:
        return "ERROR: ไม่พบ library 'trafilatura' — กรุณา install ด้วย: uv add trafilatura"

    # ดึงและ extract เนื้อหาบทความ
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return f"ERROR: ไม่สามารถดึงเนื้อหาจาก URL: {url} — ตรวจสอบว่า URL ถูกต้องและเข้าถึงได้"
    except Exception as e:
        log.warning("Article fetch failed | url=%s: %s", url, e)
        return f"ERROR: ดึงเนื้อหาล้มเหลว: {e}"

    metadata = trafilatura.extract_metadata(downloaded)
    title = (metadata.title if metadata and metadata.title else None) or url
    og_image = (metadata.image if metadata and metadata.image else None)

    raw_text = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=True,
        no_fallback=False,
    )

    if not raw_text or not raw_text.strip():
        return f"ERROR: ไม่สามารถสกัดเนื้อหาจาก URL: {url} — เว็บอาจ block crawler หรือใช้ JavaScript rendering"

    # เรียก LLM สกัดข้อมูล
    try:
        extracted = _call_extractor_llm(raw_text, f"Article: {title}")
    except ValueError as e:
        return f"ERROR: {e}"
    except Exception as e:
        log.warning("Article LLM extraction failed | url=%s: %s", url, e)
        return f"ERROR: LLM Extraction ล้มเหลว (OpenRouter): {e}"

    return _build_article_md(extracted, url, title, today, now_time, image=og_image)


@tool
def ingest_pdf(file_path: str) -> str:
    """อ่าน PDF และสกัดข้อมูลการลงทุนด้วย LLM
    รองรับ PDF รายงานบริษัท, งบการเงิน, บทวิเคราะห์ — Return Markdown พร้อม YAML frontmatter ไม่บันทึกไฟล์เอง

    Args:
        file_path: path ของไฟล์ PDF เช่น 'C:/Downloads/annual_report_2024.pdf' หรือ './reports/analysis.pdf'
    """
    today = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        import pypdf
    except ImportError:
        return "ERROR: ไม่พบ library 'pypdf' — กรุณา install ด้วย: uv add pypdf"

    pdf_path = Path(file_path)
    if not pdf_path.exists():
        return f"ERROR: ไม่พบไฟล์ PDF: '{file_path}'"
    if pdf_path.suffix.lower() != ".pdf":
        return f"ERROR: ไฟล์ไม่ใช่ PDF: '{file_path}'"

    # อ่าน PDF
    try:
        reader = pypdf.PdfReader(str(pdf_path))
        pages_text = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                pages_text.append(text)
        raw_text = "\n".join(pages_text)
    except Exception as e:
        log.warning("PDF read failed | path=%s: %s", file_path, e)
        return f"ERROR: อ่าน PDF ล้มเหลว: {e}"

    if not raw_text.strip():
        return f"ERROR: PDF ไม่มีข้อความที่สกัดได้ (อาจเป็น scanned image) — ไฟล์: {pdf_path.name}"

    title = pdf_path.stem.replace("_", " ").replace("-", " ")

    # เรียก LLM สกัดข้อมูล
    try:
        extracted = _call_extractor_llm(raw_text, f"PDF: {pdf_path.name}")
    except ValueError as e:
        return f"ERROR: {e}"
    except Exception as e:
        log.warning("PDF LLM extraction failed | path=%s: %s", file_path, e)
        return f"ERROR: LLM Extraction ล้มเหลว (OpenRouter): {e}"

    return _build_article_md(extracted, f"file://{pdf_path.resolve()}", title, today, now_time)
