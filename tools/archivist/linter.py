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
def lint_structural_health() -> str:
    """ตรวจสอบสุขภาพระดับโครงสร้างของ Vault (Orphan/Empty files)

    [Usage/When to use]
    ใช้เมื่อถูกสั่งให้ตรวจสุขภาพของ Vault โดยรวม หรือตรวจสอบโครงสร้างไฟล์
    - ตรวจหาไฟล์ที่ไม่มีใครเชื่อมโยงถึง (Orphan) และไฟล์ที่ไม่มีเนื้อหาหรือสั้นเกินไป (Empty/Stub)
    
    [Caution]
    - เครื่องมือนี้ไม่รับพารามิเตอร์ใดๆ และจะสแกนทั้ง Vault ทันที
    - ไม่ได้ตรวจสอบความถูกต้องของเนื้อหา (ใช้ `lint_semantic_conflict` สำหรับเนื้อหา)

    Returns:
        str: รายงานผลการสแกนในรูปแบบ Markdown
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
def lint_semantic_conflict(target_folder_or_entity: str) -> str:
    """ตรวจสอบความขัดแย้งเชิงความหมายและเนื้อหาซ้ำซ้อนภายในโฟลเดอร์เป้าหมาย

    [Usage/When to use]
    ใช้เมื่อต้องการตรวจความถูกต้องของเนื้อหา หาจุดที่ขัดแย้งกัน หรือความซ้ำซ้อนของข้อมูล
    - ตัวอย่าง: "มีข้อมูลของ PTT ขัดแย้งกันไหม", "ช่วยตรวจ folder Strategy ว่ามีข้อมูลซ้ำซ้อนกันหรือเปล่า"
    
    [Caution]
    - บังคับระบุเป้าหมาย `target_path` ให้แคบที่สุดเสมอ (เช่น '30_Knowledge_Base/Stocks/PTT' หรือเจาะจงโฟลเดอร์)
    - ห้ามรันสแกนที่ Root level หรือทั้ง Vault เพราะอาจทำให้เปลือง Token และเวลาประมวลผลสูง

    Args:
        target_path (str): path ภายใน Vault ที่ต้องการสแกน เช่น '30_Knowledge_Base/Stocks/PTT' หรือระบุเฉพาะเจาะจง

    Returns:
        str: รายงานผลการวิเคราะห์ความขัดแย้งและความซ้ำซ้อน (หรือแจ้งว่าไม่พบปัญหา)
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


