from langsmith import traceable
import os
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool

from core.logger import get_logger
from core.security import anonymize_pii
from core.retry import with_retry
from .core import _call_extractor_llm, _build_article_md

log = get_logger(__name__)

_ARTICLES_PATH = Path(os.getenv("OBSIDIAN_VAULT_PATH", "./memories")) / "30_Knowledge_Base/Articles"

def _is_url_already_processed(url: str) -> bool:
    """ตรวจสอบว่า URL นี้เคยถูกอ่านและเซฟลง Knowledge Base แล้วหรือยัง"""
    vault = Path(os.getenv("OBSIDIAN_VAULT_PATH", "./memories"))
    news_dir = vault / "30_Knowledge_Base/News"
    if not news_dir.exists():
        return False
    for md_file in news_dir.rglob("*.md"):
        if "Inbox" in md_file.parts:
            continue
        try:
            content = md_file.read_text(encoding="utf-8")
            if url in content:
                return True
        except Exception:
            pass
    return False

@tool
def ingest_article_url(url: str) -> str:
    """ดึงเนื้อหาจาก URL ของบทความ/บล็อก/ข่าวการลงทุน และแปลงเป็น Markdown

    [Usage/When to use]
    ใช้เมื่อต้องการสรุปเนื้อหาจาก URL หรือเมื่อผู้ใช้ส่งลิงก์บทความมาให้สรุป
    - ดึงเนื้อหาโดยใช้ 3-Tier Fallback (Firecrawl → Trafilatura → BeautifulSoup)
    - สามารถดึง Title มาใช้ตั้งชื่อไฟล์ได้อัตโนมัติ

    [Caution]
    - เครื่องมือนี้แค่ส่งคืนข้อความ Markdown (ไม่บันทึกไฟล์เอง)
    - **ต้อง** นำผลลัพธ์ที่ได้ไปส่งให้ Bookkeeper หรือ Archivist บันทึกไฟล์ต่อด้วย `write_raw_markdown` เท่านั้น

    Args:
        url (str): ลิงก์ URL ของบทความที่ต้องการสกัดข้อมูล

    Returns:
        str: เนื้อหาในรูปแบบ Markdown พร้อม YAML Frontmatter (หรือข้อความ Error)
    """
    if _is_url_already_processed(url):
        return f"ข้าม: ข่าวนี้เคยถูกดึงและบันทึกไปแล้ว ({url})"

    today = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        import trafilatura
    except ImportError:
        return "ERROR: ไม่พบ library 'trafilatura' — กรุณา install ด้วย: uv add trafilatura"

    # พยายามดึง metadata (เร็วที่สุดด้วย trafilatura)
    downloaded = None
    try:
        downloaded = trafilatura.fetch_url(url)
    except Exception as e:
        log.warning("Article metadata fetch failed | url=%s: %s", url, e)

    metadata = trafilatura.extract_metadata(downloaded) if downloaded else None
    raw_title = (metadata.title if metadata and metadata.title else None) or url
    title, _ = anonymize_pii(raw_title)
    og_image = (metadata.image if metadata and metadata.image else None)

    def _is_valid_length(text: str | None) -> bool:
        if not text:
            return False
        if len(text) >= 800:
            return True
        if len(text.split()) >= 150:
            return True
        return False

    def fetch_tier1(u: str) -> tuple[str | None, str | None]:
        d = trafilatura.fetch_url(u)
        if not d: return None, None
        return trafilatura.extract(d, include_comments=False, include_tables=True, no_fallback=False), None

    def fetch_tier2(u: str) -> tuple[str | None, str | None]:
        import requests
        from bs4 import BeautifulSoup
        res = requests.get(u, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, 'html.parser')
        page_title = soup.title.string if soup.title else None
        for script in soup(["script", "style"]): 
            script.extract()
        return soup.get_text(separator=' ', strip=True), page_title

    def fetch_tier3(u: str) -> tuple[str | None, str | None]:
        from playwright.sync_api import sync_playwright
        from playwright_stealth.stealth import Stealth
        import time
        from bs4 import BeautifulSoup
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            page = browser.new_page()
            Stealth().apply_stealth_sync(page)
            page.goto(u, wait_until="domcontentloaded", timeout=20000)
            time.sleep(3)  # Give Cloudflare/JS time to evaluate stealth signals
            content = page.content()
            page_title = page.title()
            browser.close()
        soup = BeautifulSoup(content, 'html.parser')
        for script in soup(["script", "style"]): 
            script.extract()
        return soup.get_text(separator=' ', strip=True), page_title

    raw_text = None
    fetched_title = None
    
    # 3-Tier Fallback Strategy
    try:
        raw_text, fetched_title = with_retry(fetch_tier1, url)
    except Exception as e:
        log.warning(f"Tier1 (trafilatura) failed for {url}: {e}")
        
    if not _is_valid_length(raw_text):
        log.info(f"Tier 1 insufficient or failed, falling back to Tier 2 for {url}")
        try:
            raw_text, fetched_title = with_retry(fetch_tier2, url)
        except Exception as e:
            log.warning(f"Tier2 (bs4) failed for {url}: {e}")
            
    if not _is_valid_length(raw_text):
        log.info(f"Tier 2 insufficient or failed, falling back to Tier 3 for {url}")
        try:
            raw_text, fetched_title = with_retry(fetch_tier3, url)
        except Exception as e:
            log.error(f"Tier3 (playwright) failed for {url}: {e}")

    # Update title if tier 2/3 found a better one
    if fetched_title and title == url:
        title, _ = anonymize_pii(fetched_title.strip())

    # Sanity Check
    if not _is_valid_length(raw_text):
        log.error(f"Failed to fetch sufficient content for {url} (Length check failed). Possibly Paywalled/Bot Protection.")
        return f"ดึงข้อมูลไม่สำเร็จ: เนื้อหาที่ดึงได้สั้นเกินไป (อาจติด Paywall หรือระบบป้องกัน Bot) สำหรับ URL {url}"

    # เรียก LLM สกัดข้อมูล
    try:
        extracted = _call_extractor_llm(raw_text, f"Article: {title}")
    except ValueError as e:
        return f"ERROR: {e}"
    except Exception as e:
        log.warning("Article LLM extraction failed | url=%s: %s", url, e)
        return f"ERROR: LLM Extraction ล้มเหลว (OpenRouter): {e}"

    return _build_article_md(extracted, url, title, today, now_time, image=og_image)

