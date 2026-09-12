import json
import hashlib
import os
import re
import shutil
import tempfile
import time
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


_INDEX_EXCLUDE = ("00_Inbox", "01_Daily_Logs")
_VAULT_SYSTEM_FILES = {
    "index.md",
    "Portfolio_Holdings.md",
    "Portfolio_Dashboard.md",
    "Watchlist.md",
    "Trading_Journal.md",
}
_INVALID_FILE_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_READ_FILE_LIMIT = 8000
_LINKED_CONTENT_LIMIT = 1500
_SEMANTIC_CONTENT_LIMIT = 2000
_DEFAULT_VAULT_FOLDERS = [
    "00_Inbox",
    "01_Daily_Logs",
    "10_Projects",
    "20_Areas",
    "30_Knowledge_Base/Stocks",
    "30_Knowledge_Base/Crypto",
    "30_Knowledge_Base/Concepts",
    "40_Archive",
]
VAULT_PATH = Path(os.getenv("OBSIDIAN_VAULT_PATH", "./memories"))
INDEX_PATH = VAULT_PATH / ".master_index.json"
INDEX_LOCK = VAULT_PATH / ".master_index.lock"


from typing import Any, Optional, Union
from core.utils import normalize_content
from tools.archivist.maintenance_guard import assert_write_allowed
from tools.archivist.vault_paths import VaultPaths


def get_vault_paths(root: Optional[Union[str, Path]] = None) -> VaultPaths:
    """Returns an injected or environment-configured VaultPaths instance."""
    return VaultPaths(root=root)


def get_note_lock(
    file_path: Union[str, Path],
    timeout: float = 10.0,
    vault_root: Optional[Union[str, Path]] = None,
) -> FileLock:
    """Returns a process-safe FileLock for a specific note, storing lockfiles in .system/locks/."""
    import hashlib
    fp = Path(file_path).resolve()
    v_root = Path(vault_root).resolve() if vault_root else Path(os.getenv("OBSIDIAN_VAULT_PATH", str(VAULT_PATH))).resolve()
    assert_write_allowed(v_root)
    locks_dir = v_root / ".system" / "locks"
    locks_dir.mkdir(parents=True, exist_ok=True)
    path_hash = hashlib.sha256(str(fp).lower().encode("utf-8")).hexdigest()[:16]
    lock_file = locks_dir / f"{fp.stem}_{path_hash}.lock"
    return FileLock(str(lock_file), timeout=timeout)


def _atomic_write_text(path: Path, content: Any, max_retries: int = 8, backoff: float = 0.05) -> None:
    """เขียนไฟล์แบบ atomic: temp file ใน folder เดียวกัน → os.replace()
    os.replace() เป็น atomic บนทั้ง Windows และ POSIX เมื่ออยู่ filesystem เดียวกัน
    บน Windows เพิ่ม retry loop เพื่อจัดการ transient file lock (WinError 5 / WinError 32)
    จาก file indexing, antivirus หรือ concurrent readers
    """
    if not isinstance(content, str):
        content = normalize_content(content) if isinstance(content, list) else str(content)
    # Enforce the vault maintenance lease before creating the temporary file.
    # Scratch evidence and external runtime paths are outside the vault and
    # are intentionally unaffected.
    assert_write_allowed(path)
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.stem}_", suffix=f"{path.suffix}.tmp", dir=str(parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
        
        last_err: Exception | None = None
        for attempt in range(max_retries):
            try:
                os.replace(tmp_path, path)
                last_err = None
                break
            except (PermissionError, OSError) as e:
                last_err = e
                winerror = getattr(e, "winerror", None)
                if winerror in (5, 32) or isinstance(e, PermissionError):
                    time.sleep(backoff * (1.5 ** attempt))
                else:
                    break
        if last_err is not None:
            raise last_err
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _sanitize_filename(name: str) -> str:
    """แทนอักขระต้องห้ามบน Windows/POSIX และตัดช่องว่าง/จุดท้ายชื่อ"""
    cleaned = _INVALID_FILE_CHARS.sub("-", name).strip(" .")
    cleaned = re.sub(r'-{2,}', '-', cleaned).strip('-')
    return cleaned or "untitled"


from langsmith import traceable

def _vault_folders() -> list[str]:
    """รวม default folders + extras จาก VAULT_EXTRA_FOLDERS env (comma-separated)
    ตัวอย่าง: VAULT_EXTRA_FOLDERS='50_Crypto,60_Research/Drafts'
    """
    extras = os.getenv("VAULT_EXTRA_FOLDERS", "").strip()
    if not extras:
        return _DEFAULT_VAULT_FOLDERS
    extra_list = [p.strip() for p in extras.split(",") if p.strip()]
    return _DEFAULT_VAULT_FOLDERS + extra_list


@traceable(run_type="tool")
def init_vault_structure() -> None:
    for folder in _vault_folders():
        (VAULT_PATH / folder).mkdir(parents=True, exist_ok=True)


def _safe_read_path(filepath: str) -> Path:
    """Resolve a reader path inside the configured Vault only."""
    return get_vault_paths(VAULT_PATH).safe_resolve(str(filepath))


@tool
def read_file(filepath: str) -> str:
    """อ่านเนื้อหาไฟล์ .md จาก Obsidian Vault (ดึงข้อมูลดิบเต็มไฟล์)

    [Usage/When to use]
    ใช้เมื่อต้องการอ่านข้อมูลจากไฟล์ที่ทราบ path หรือชื่อไฟล์ชัดเจน
    - ควรเริ่มด้วยการเรียก read_file('.system/master_index.json') ก่อนเสมอ หากต้องการภาพรวมของ Vault
    - ใช้เพื่อดึงเนื้อหาที่ถูกระบุในหน้า Index มาอ่านแบบเต็มๆ

    [Caution]
    - ไม่เหมาะกับการค้นหาข้อมูลที่ไม่ทราบชื่อไฟล์ ให้ใช้ `search_all_memories` แทน
    - ข้อมูลอาจถูกตัดทอนหากไฟล์มีความยาวเกินกำหนด

    Args:
        filepath (str): path ของไฟล์ภายใน Vault (อ้างอิงจาก root ของ Vault) เช่น '30_Knowledge_Base/Macroeconomics/GDP.md' หรือ '.system/master_index.json'

    Returns:
        str: เนื้อหาของไฟล์พร้อม header ระบุชื่อไฟล์ (หรือข้อความแจ้งเตือนหากไม่พบไฟล์)
    """
    try:
        file_path = _safe_read_path(filepath)
    except ValueError:
        return f"ไม่พบไฟล์: {filepath}"
    if not file_path.is_file():
        return f"ไม่พบไฟล์: {filepath}"

    content = file_path.read_text(encoding="utf-8")
    if len(content) > _READ_FILE_LIMIT:
        content = content[:_READ_FILE_LIMIT] + f"\n\n...[ตัดทอน — ไฟล์ยาว {len(content)} ตัวอักษร]"
    return f"=== {filepath} ===\n\n{content}"


@tool
def read_note_chunk(
    filepath: str,
    cursor: str = "",
    max_chars: int = _READ_FILE_LIMIT,
    expected_content_sha256: str = "",
) -> str:
    """Read one hash-bound page of a Vault note without silent truncation.

    ``cursor`` is the JSON value returned by the previous call.  A plain
    integer cursor is accepted for compatibility, but the normal response
    binds the offset to the complete-note SHA-256 and returns a stale-cursor
    error if the note changed between pages.
    """
    try:
        file_path = _safe_read_path(filepath)
    except ValueError:
        return json.dumps(
            {"status": "NOT_FOUND", "filepath": filepath, "error": "path_outside_vault"},
            ensure_ascii=False,
        )
    if not file_path.is_file():
        return json.dumps(
            {"status": "NOT_FOUND", "filepath": filepath, "error": "file_not_found"},
            ensure_ascii=False,
        )
    try:
        content = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return json.dumps(
            {"status": "ERROR", "filepath": filepath, "error": str(exc)},
            ensure_ascii=False,
        )

    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    token = str(cursor or "").strip()
    offset = 0
    cursor_hash = ""
    if token:
        try:
            if token.isdigit():
                offset = int(token)
            else:
                parsed = json.loads(token)
                if isinstance(parsed, dict):
                    offset = int(parsed.get("offset", 0) or 0)
                    cursor_hash = str(parsed.get("content_sha256") or "")
        except (TypeError, ValueError, json.JSONDecodeError):
            return json.dumps(
                {"status": "ERROR", "filepath": filepath, "error": "invalid_cursor"},
                ensure_ascii=False,
            )
    expected = str(expected_content_sha256 or cursor_hash).strip()
    if expected and expected != content_hash:
        return json.dumps(
            {
                "status": "STALE_CURSOR",
                "filepath": filepath,
                "content_sha256": content_hash,
                "error": "note_changed_between_pages; restart_from_cursor_empty",
            },
            ensure_ascii=False,
        )
    try:
        page_size = max(1, min(int(max_chars), 8000))
    except (TypeError, ValueError):
        page_size = _READ_FILE_LIMIT
    offset = max(0, min(offset, len(content)))
    end = min(len(content), offset + page_size)
    page = content[offset:end]
    eof = end >= len(content)
    next_cursor = "" if eof else json.dumps(
        {"offset": end, "content_sha256": content_hash},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    revision_id = ""
    try:
        from tools.archivist.metadata import parse_note

        metadata, _, _ = parse_note(content)
        revision_id = str(
            metadata.get("current_revision_id")
            or metadata.get("revision_id")
            or ""
        )
    except Exception:
        revision_id = ""
    return json.dumps(
        {
            "status": "PASS",
            "filepath": str(file_path.relative_to(VAULT_PATH.resolve()).as_posix()),
            "content": page,
            "offset": offset,
            "next_offset": None if eof else end,
            "next_cursor": next_cursor,
            "eof": eof,
            "content_sha256": content_hash,
            "revision_id": revision_id,
            "page_chars": len(page),
            "total_chars": len(content),
        },
        ensure_ascii=False,
    )


