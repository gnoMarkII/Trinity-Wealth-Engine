from langsmith import traceable
import os
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool

from core.logger import get_logger
from .core import _call_extractor_llm, _build_article_md

log = get_logger(__name__)

@tool
@traceable(run_type="tool")
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
