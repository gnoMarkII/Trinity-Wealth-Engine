import json
import os
import re
import shutil
import tempfile
from datetime import datetime
from functools import lru_cache
from pathlib import Path

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




VAULT_PATH = Path(os.getenv("OBSIDIAN_VAULT_PATH", "./memories"))
INDEX_PATH = VAULT_PATH / ".system" / "master_index.json"
INDEX_LOCK = str(INDEX_PATH) + ".lock"


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


