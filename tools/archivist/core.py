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


def _atomic_write_text(path: Path, content: str) -> None:
    """เขียนไฟล์แบบ atomic: temp file ใน folder เดียวกัน → os.replace()
    os.replace() เป็น atomic บนทั้ง Windows และ POSIX เมื่ออยู่ filesystem เดียวกัน
    """
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.stem}_", suffix=f"{path.suffix}.tmp", dir=str(parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_path, path)
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


@tool
@traceable(run_type="retriever")
def read_file(filepath: str) -> str:
    """อ่านเนื้อหาไฟล์ .md จาก Vault

    Args:
        filepath: path ของไฟล์ภายใน Vault เช่น '30_Knowledge_Base/Macroeconomics/GDP.md'
    """
    file_path = VAULT_PATH / filepath
    if not file_path.exists():
        return f"ไม่พบไฟล์: {filepath}"

    content = file_path.read_text(encoding="utf-8")
    if len(content) > _READ_FILE_LIMIT:
        content = content[:_READ_FILE_LIMIT] + f"\n\n...[ตัดทอน — ไฟล์ยาว {len(content)} ตัวอักษร]"
    return f"=== {filepath} ===\n\n{content}"


