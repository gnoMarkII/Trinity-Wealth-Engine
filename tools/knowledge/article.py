from langsmith import traceable
import os
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool

from core.logger import get_logger
from core.security import anonymize_pii
from .core import _call_extractor_llm, _build_article_md

log = get_logger(__name__)

_ARTICLES_PATH = Path(os.getenv("OBSIDIAN_VAULT_PATH", "./memories")) / "30_Knowledge_Base/Articles"

def _find_existing_article(source_id: str) -> Path | None:
    """คืน Path ของไฟล์ที่มีอยู่แล้ว หรือ None ถ้ายังไม่เคย ingest"""
    for folder in (_ARTICLES_PATH,):
        if not folder.exists():
            continue
        matches = list(folder.glob(f"*{source_id}*.md"))
        if matches:
            return matches[0]
    return None

@tool
@traceable(run_type="tool")
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
    raw_title = (metadata.title if metadata and metadata.title else None) or url
    title, _ = anonymize_pii(raw_title)
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
