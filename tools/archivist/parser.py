import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Optional, Any

import frontmatter as fm
from filelock import FileLock
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.logger import get_logger
from schemas.pkm_models import MemoryEntry

log = get_logger(__name__)
_ASSET_TICKER_RE = re.compile(r"(?i)\b(?:[A-Z]+-[A-Z]+|[A-Z]{2,5})\b")
_H2_SECTION_RE = re.compile(r"^##\s+(.*)$", re.MULTILINE)
_H3_SECTION_RE = re.compile(r"^###\s+(.*)$", re.MULTILINE)

_TICKER_FRONTMATTER_RE = re.compile(r"^tickers?:\s*\[?[\"']?([A-Z0-9.-]+)[\"']?", re.MULTILINE | re.IGNORECASE)
_VIDEO_ID_FRONTMATTER_RE = re.compile(r"^video_id:\s*[\"']?([a-zA-Z0-9_-]+)[\"']?", re.MULTILINE | re.IGNORECASE)
_SOURCE_URL_FRONTMATTER_RE = re.compile(r"^source_url:\s*[\"']?(https?://[^\s\"']+)[\"']?", re.MULTILINE | re.IGNORECASE)

from tools.archivist.metadata import (
    parse_note,
    normalize_legacy_metadata,
    validate_note,
    dump_note,
)




VAULT_PATH = Path(os.getenv("OBSIDIAN_VAULT_PATH", "./memories"))
INDEX_PATH = VAULT_PATH / ".system" / "master_index.json"
INDEX_LOCK = str(INDEX_PATH) + ".lock"


def extract_yaml_frontmatter_value(content: str, key: str) -> Optional[str]:
    """สกัด YAML frontmatter จากระหว่างกรอบ --- เท่านั้น พร้อมดึงค่า key โดย strip quotes (" และ ') ออก"""
    if not content.strip().startswith("---"):
        return None
    parts = content.split("---", 2)
    if len(parts) < 3:
        return None
    fm_block = parts[1]
    pattern = rf"^\s*{re.escape(key)}\s*:\s*(?:[\"'](?P<qval>[^\"']+)[\"']|(?P<uval>[^\r\n#]+))"
    match = re.search(pattern, fm_block, re.MULTILINE)
    if match:
        val = match.group("qval") or match.group("uval")
        return val.strip() if val else None
    return None


def parse_frontmatter_metadata(content: str) -> dict[str, Any]:
    """แปลง YAML frontmatter เป็น dict โดยใช้ python-frontmatter เพื่อได้ชนิดข้อมูลที่แท้จริง (เช่น list[str])"""
    try:
        post = fm.loads(content)
        return dict(post.metadata)
    except Exception as e:
        log.warning("parse_frontmatter_metadata failed: %s", e)
        return {}


def _strip_frontmatter(content: str) -> str:
    """ตัด YAML frontmatter (--- ... ---) ออก คืนเฉพาะ body"""
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            return parts[2].strip()
    return content.strip()


def _parse_h2_sections(body: str) -> dict[str, str]:
    """สกัด ## headers → {heading: content} dict"""
    result: dict[str, str] = {}
    matches = list(_H2_SECTION_RE.finditer(body))
    for i, m in enumerate(matches):
        heading = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        result[heading] = body[start:end].strip()
    return result


def _parse_h3_subsections(text: str) -> dict[str, str]:
    """สกัด ### sub-headers → {heading: content}, fallback 'ทั่วไป' ถ้าไม่มี sub-header"""
    result: dict[str, str] = {}
    matches = list(_H3_SECTION_RE.finditer(text))
    if not matches:
        stripped = text.strip()
        return {"ทั่วไป": stripped} if stripped else {}
    for i, m in enumerate(matches):
        heading = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            result[heading] = body
    return result


def _split_bullets(text: str, max_per_node: int = 3) -> list[str]:
    """แบ่ง text เป็น chunks ไม่เกิน max_per_node บรรทัดต่อ chunk"""
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return []
    return ["\n".join(lines[i:i + max_per_node]) for i in range(0, len(lines), max_per_node)]


def _extract_asset_tickers(text: str) -> list[tuple[str, str]]:
    """สกัด (ticker, description) จากบรรทัดที่มี [[TICKER]] wikilink"""
    result = []
    for line in text.splitlines():
        stripped = line.strip().lstrip("- *")
        m = re.search(r"\[\[([A-Z][A-Z0-9.\-]{1,9})\]\]", stripped)
        if not m:
            continue
        ticker = m.group(1)
        desc = re.sub(r"\*+", "", stripped)
        desc = re.sub(r"\[\[[^\]]+\]\]", f"#{ticker}", desc)
        desc = re.sub(r"^[#\s—\-]+", "", desc).strip()
        result.append((ticker, desc[:120]))
    return result


def _chunk_file(file_path: Path, splitter: RecursiveCharacterTextSplitter) -> tuple[list[str], list[dict], list[str]]:
    """Chunk หนึ่งไฟล์ → คืน (texts, metas, ids) ที่พร้อม upsert เข้า Chroma"""
    content = file_path.read_text(encoding="utf-8")
    rel = str(file_path.relative_to(VAULT_PATH))
    chunks = splitter.split_text(content)
    texts = chunks
    metas = [{"source": rel} for _ in chunks]
    ids = [f"{rel}::{i}" for i in range(len(chunks))]
    return texts, metas, ids


def parse_company_news_items(content: str) -> dict[str, Any]:
    """Parse Company_News markdown content into structured dictionary."""
    ticker = extract_yaml_frontmatter_value(content, "ticker") or ""
    market = extract_yaml_frontmatter_value(content, "market") or "US"
    date_str = extract_yaml_frontmatter_value(content, "date")
    last_updated_str = extract_yaml_frontmatter_value(content, "last_updated")

    # Base reference datetime for calculating published_at fallback
    ref_dt = None
    if last_updated_str:
        try:
            ref_dt = datetime.strptime(last_updated_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        except Exception:
            try:
                ref_dt = datetime.fromisoformat(last_updated_str.replace("Z", "+00:00"))
            except Exception:
                ref_dt = None
    if not ref_dt and date_str:
        try:
            ref_dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except Exception:
            ref_dt = None

    items = []
    # Pattern to match item headlines like: 1. **Title** ⚠️ [STALE] <!-- published_at: 2026-08-05T14:30:00Z -->
    item_blocks = re.split(r"(?m)^(?=\s*\d+\.\s+\*\*)", content)
    for block in item_blocks:
        block = block.strip()
        if not block or not re.match(r"^\s*\d+\.\s+\*\*", block):
            continue

        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        headline_line = lines[0]

        # Extract title
        m_head = re.match(r"^\s*\d+\.\s+\*\*(.+?)\*\*", headline_line)
        if not m_head:
            continue

        title = m_head.group(1).strip()
        # Remove any ⚠️ [STALE] flag or comments from title if captured
        title = re.sub(r"\s+⚠️\s*\[STALE\]", "", title).strip()

        # Extract published_at HTML comment if present anywhere in headline_line
        m_pub = re.search(r"<!--\s*published_at:\s*([^\s>]+)\s*-->", headline_line)
        pub_iso = m_pub.group(1).strip() if m_pub else None

        source = "N/A"
        sources_count = 1
        link = ""
        age_hours_fallback = 0
        freshness_reason_fallback = "Unknown age"

        for ln in lines[1:]:
            if "ที่มา:" in ln:
                m_src = re.search(r"ที่มา:\s*(.*?)(?:\s*\(Reported by (\d+) sources\))?$", ln)
                if m_src:
                    source = m_src.group(1).strip()
                    if m_src.group(2):
                        try:
                            sources_count = int(m_src.group(2))
                        except Exception:
                            pass
            elif "อายุข่าว:" in ln:
                m_age = re.search(r"อายุข่าว:\s*(\d+)\s*ชั่วโมง\s*(?:\((.*?)\))?", ln)
                if m_age:
                    try:
                        age_hours_fallback = int(m_age.group(1))
                    except Exception:
                        pass
                    if m_age.group(2):
                        freshness_reason_fallback = m_age.group(2).strip()
            elif "[อ่านต่อ]" in ln:
                m_link = re.search(r"\[อ่านต่อ\]\((.*?)\)", ln)
                if m_link:
                    link = m_link.group(1).strip()

        # If pub_iso is missing, fallback calculate from ref_dt - age_hours_fallback
        if not pub_iso and ref_dt and age_hours_fallback < 9000:
            pub_iso = (ref_dt - timedelta(hours=age_hours_fallback)).isoformat()

        items.append({
            "title": title,
            "source": source,
            "link": link,
            "published_at": pub_iso,
            "age_hours": age_hours_fallback,
            "freshness_reason": freshness_reason_fallback,
            "sources_count": sources_count,
            "is_stale": age_hours_fallback > 48
        })

    return {
        "ticker": ticker,
        "market": market,
        "date": date_str,
        "last_updated": last_updated_str,
        "items": items
    }


