from langsmith import traceable
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
from .core import VAULT_PATH, read_file, _VAULT_SYSTEM_FILES, _INDEX_EXCLUDE, _SEMANTIC_CONTENT_LIMIT
from .parser import _chunk_file
from .search import search_graph_context





VAULT_PATH = Path(os.getenv("OBSIDIAN_VAULT_PATH", "./memories"))
INDEX_PATH = VAULT_PATH / ".system" / "master_index.json"
INDEX_LOCK = str(INDEX_PATH) + ".lock"


@tool
@traceable(run_type="tool")
def lint_structural_health() -> str:
    """ตรวจสุขภาพเชิงโครงสร้างของ Vault: ค้นหา Orphan files (ไม่มีไฟล์ใด Wikilink โยงมา)
    และ Empty files (ไม่มีเนื้อหา) ส่งรายงานกลับมาให้ทราบว่าไฟล์ไหนต้องจัดการ
    """
    all_files = [
        f for f in VAULT_PATH.rglob("*.md")
        if f.name not in _VAULT_SYSTEM_FILES
        and not any(excl in f.parts for excl in _INDEX_EXCLUDE)
    ]

    if not all_files:
        return "ไม่มีไฟล์ใดใน Vault ที่จะตรวจสอบ"

    stem_to_path = {f.stem: f for f in all_files}
    file_contents = {f: f.read_text(encoding="utf-8") for f in all_files}

    inbound: dict[str, set[str]] = {f.stem: set() for f in all_files}
    empty_files: list[str] = []

    for file_path in all_files:
        content = file_contents[file_path]

        for raw in re.findall(r'\[\[(.*?)\]\]', content):
            target = raw.split("|")[0].strip()
            if target in stem_to_path:
                inbound[target].add(file_path.stem)

        stripped = content.strip()
        parts = stripped.split("---", 2) if stripped.startswith("---") else ["", "", stripped]
        body = parts[2].strip() if len(parts) >= 3 else stripped
        if not body:
            empty_files.append(file_path.stem)

    orphans = sorted(s for s in stem_to_path if not inbound[s])

    lines = [
        "# Vault Health Report",
        "",
        f"**ไฟล์ทั้งหมด:** {len(all_files)}",
        f"**Orphan notes:** {len(orphans)}",
        f"**Empty files:** {len(empty_files)}",
        "",
    ]
    if orphans:
        lines += ["## Orphan Notes (ไม่มี Link เชื่อมโยง)"]
        lines += [f"- [[{s}]] ({stem_to_path[s].relative_to(VAULT_PATH).parent})" for s in orphans]
        lines.append("")
    if empty_files:
        lines += ["## Empty Files (ไม่มีเนื้อหา)"]
        lines += [f"- [[{s}]] ({stem_to_path[s].relative_to(VAULT_PATH).parent})" for s in sorted(empty_files)]
        lines.append("")
    if not orphans and not empty_files:
        lines.append("Vault อยู่ในสุขภาพดี ไม่พบปัญหาใดๆ")

    return "\n".join(lines)


@tool
@traceable(run_type="tool")
def lint_semantic_conflict(target_folder_or_entity: str) -> str:
    """ดึงเนื้อหาไฟล์ใน Folder หรือ Entity ที่ระบุ เพื่อให้ LLM ตรวจหาความขัดแย้งของข้อมูล
    Python ทำหน้าที่ดึงและจำกัดขอบเขต ส่วน LLM อ่านผลลัพธ์เพื่อวิเคราะห์ความสอดคล้อง

    Args:
        target_folder_or_entity: โฟลเดอร์ (เช่น '30_Knowledge_Base/Stocks')
                                  หรือชื่อ Entity (เช่น 'PTT') ที่ต้องการตรวจสอบ
    """
    target_path = VAULT_PATH / target_folder_or_entity

    if target_path.is_dir():
        files = sorted(target_path.glob("*.md"))
        source_label = f"Folder: {target_folder_or_entity}"
    else:
        all_files = list(VAULT_PATH.rglob("*.md"))
        name_lower = target_folder_or_entity.lower()
        files = sorted(f for f in all_files if name_lower in f.stem.lower())
        source_label = f"Entity match: {target_folder_or_entity}"

    if not files:
        return f"ไม่พบไฟล์ที่ตรงกับ '{target_folder_or_entity}'"

    parts = [
        f"# Semantic Conflict Check — {source_label}",
        f"ไฟล์ที่ตรวจ: {len(files)} ไฟล์",
        "",
    ]
    for file_path in files:
        content = file_path.read_text(encoding="utf-8")
        if len(content) > _SEMANTIC_CONTENT_LIMIT:
            content = content[:_SEMANTIC_CONTENT_LIMIT] + "\n...[ตัดทอน]"
        rel = file_path.relative_to(VAULT_PATH)
        parts.append(f"## [{rel}]\n{content}\n")

    return "\n".join(parts)


