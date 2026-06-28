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
from .core import VAULT_PATH, INDEX_PATH, INDEX_LOCK, _atomic_write_text, _VAULT_SYSTEM_FILES, _INDEX_EXCLUDE, _LINKED_CONTENT_LIMIT
from .parser import _chunk_file
CHROMA_PATH = VAULT_PATH / ".chroma_index"
_CHROMA_MTIME_FILE = VAULT_PATH / ".chroma_mtime"
_vs_cache: dict = {}  # {"cache_key": str, "vs": Chroma}





VAULT_PATH = Path(os.getenv("OBSIDIAN_VAULT_PATH", "./memories"))
INDEX_PATH = VAULT_PATH / ".system" / "master_index.json"
INDEX_LOCK = str(INDEX_PATH) + ".lock"


@lru_cache(maxsize=1)
def get_embeddings():
    log.info("กำลังโหลด embedding model สำหรับ Semantic Search (ครั้งแรกอาจใช้เวลา)")
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


def _load_index_state() -> dict:
    """โหลด per-file state: {rel: {mtime, chunks}}"""
    if not _CHROMA_MTIME_FILE.exists():
        return {}
    try:
        data = json.loads(_CHROMA_MTIME_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "files" in data:
            return data["files"]
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def _save_index_state(files: dict) -> None:
    _atomic_write_text(
        _CHROMA_MTIME_FILE,
        json.dumps({"version": 1, "files": files}, ensure_ascii=False),
    )


def _searchable_files() -> list[Path]:
    return [
        f for f in VAULT_PATH.rglob("*.md")
        if f.name not in _VAULT_SYSTEM_FILES
        and not any(excl in f.parts for excl in _INDEX_EXCLUDE)
    ]


@tool
def search_all_memories(keyword: str) -> str:
    """ค้นหาความจำทั้งหมดใน Vault ด้วย Semantic Search (Vector RAG) แบบ Local

    [Usage/When to use]
    ใช้เมื่อต้องการค้นหาข้อมูลจากคลังความรู้แต่ไม่ทราบชื่อไฟล์ชัดเจน
    - เหมาะสำหรับคำถามที่ต้องการความเข้าใจความหมายและบริบท ไม่ใช่แค่ Keyword ตรงๆ (เช่น 'กลยุทธ์เมื่อดอกเบี้ยขึ้น', 'หุ้นสื่อสารที่น่าสนใจ')
    - เมื่อหาไฟล์จาก `read_file('index.md')` ไม่พบ

    [Caution]
    - ผลลัพธ์ที่ได้จะเป็นการสรุปและดึงเฉพาะส่วนที่เกี่ยวข้อง (Top-K snippets) ไม่ใช่เนื้อหาเต็มทั้งไฟล์
    - หากต้องการข้อมูลแบบเจาะลึกทั้ง Entity พร้อมความสัมพันธ์ ให้ใช้ `search_graph_context` แทน

    Args:
        keyword (str): คำถาม ประโยค หรือวลีที่ต้องการค้นหาความหมายจากคลังข้อมูล

    Returns:
        str: ผลลัพธ์การค้นหาที่ถูกจัดรูปแบบ (รวมรายชื่อไฟล์และเนื้อหาที่สกัดมา)
    """
    md_files = _searchable_files()
    if not md_files:
        return "ยังไม่มีไฟล์ความจำใดใน Vault"

    current: dict[str, dict] = {
        str(f.relative_to(VAULT_PATH)): {"mtime": f.stat().st_mtime}
        for f in md_files
    }
    stored = _load_index_state()

    added_or_changed = [
        rel for rel, info in current.items()
        if rel not in stored or abs(stored[rel].get("mtime", 0) - info["mtime"]) > 1e-3
    ]
    removed = [rel for rel in stored if rel not in current]

    cache_valid = _vs_cache.get("cache_signature") == (len(current), tuple(sorted(current)))
    needs_update = bool(added_or_changed or removed) or not cache_valid

    if "vs" in _vs_cache and not needs_update:
        vectorstore = _vs_cache["vs"]
    else:
        try:
            vectorstore = _vs_cache.get("vs") or Chroma(
                persist_directory=str(CHROMA_PATH),
                embedding_function=get_embeddings(),
            )
        except Exception:
            # corrupted store → wipe and start fresh
            if CHROMA_PATH.exists():
                shutil.rmtree(CHROMA_PATH)
            stored = {}
            added_or_changed = list(current)
            removed = []
            try:
                vectorstore = Chroma(
                    persist_directory=str(CHROMA_PATH),
                    embedding_function=get_embeddings(),
                )
            except Exception as e:
                return f"เกิดข้อผิดพลาดในการเปิด vectorstore: {e}"

        # Delete removed + changed (changed gets re-added below)
        ids_to_delete: list[str] = []
        for rel in removed:
            n = stored.get(rel, {}).get("chunks", 0)
            ids_to_delete.extend(f"{rel}::{i}" for i in range(n))
        for rel in added_or_changed:
            n = stored.get(rel, {}).get("chunks", 0)
            if n:
                ids_to_delete.extend(f"{rel}::{i}" for i in range(n))
        if ids_to_delete:
            try:
                vectorstore.delete(ids=ids_to_delete)
            except Exception as e:
                log.warning("Chroma delete failed (continuing): %s", e)

        # Re-chunk and add changed
        if added_or_changed:
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            all_texts: list[str] = []
            all_metas: list[dict] = []
            all_ids: list[str] = []
            for rel in added_or_changed:
                fp = VAULT_PATH / rel
                if not fp.exists():
                    continue
                texts, metas, ids = _chunk_file(fp, splitter)
                all_texts.extend(texts)
                all_metas.extend(metas)
                all_ids.extend(ids)
                current[rel]["chunks"] = len(texts)
            if all_texts:
                try:
                    vectorstore.add_texts(texts=all_texts, metadatas=all_metas, ids=all_ids)
                except Exception as e:
                    return f"เกิดข้อผิดพลาดในการเพิ่ม vectorstore: {e}"

        # Preserve chunk count for unchanged files
        for rel in current:
            if "chunks" not in current[rel] and rel in stored:
                current[rel]["chunks"] = stored[rel].get("chunks", 0)

        _save_index_state(current)
        _vs_cache["vs"] = vectorstore
        _vs_cache["cache_signature"] = (len(current), tuple(sorted(current)))

    try:
        results = vectorstore.similarity_search(keyword, k=5)
    except Exception as e:
        return f"เกิดข้อผิดพลาดในการค้นหา: {e}"

    if not results:
        return f"ไม่พบความจำที่เกี่ยวข้องกับ '{keyword}'"

    parts = [f"ผลการค้นหาเชิงความหมายสำหรับ '{keyword}' ({len(results)} ผลลัพธ์):\n"]
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "ไม่ทราบแหล่งที่มา")
        parts.append(f"--- ผลลัพธ์ที่ {i} | แหล่งที่มา: [{source}] ---\n{doc.page_content}\n")

    return "\n".join(parts)


def _find_file_by_name(name: str, all_files: list[Path]) -> Path | None:
    """ค้นหาไฟล์โดย stem ตรงทั้งหมดก่อน แล้ว fallback ไป partial match"""
    name_lower = name.lower()
    exact = next((f for f in all_files if f.stem.lower() == name_lower), None)
    if exact:
        return exact
    return next((f for f in all_files if name_lower in f.stem.lower()), None)


@tool
def search_graph_context(entity_name: str) -> str:
    """ค้นหาข้อมูล Entity พร้อมดึงเนื้อหาจาก Linked Entities ที่เชื่อมโยงกัน (GraphRAG)

    [Usage/When to use]
    ใช้เมื่อต้องการวิเคราะห์บริษัท, บุคคล, กลยุทธ์, หรือเหตุการณ์แบบเจาะลึก 360 องศา
    - เครื่องมือนี้จะดึงเนื้อหาจาก "ไฟล์เป้าหมาย" และ "ไฟล์ทั้งหมดที่เป้าหมายนั้นทำ Wikilink โยงไปหา" มาให้ในครั้งเดียว
    - เหมาะสำหรับการดูภาพรวมเครือข่ายความสัมพันธ์ของ Entity ใด Entity หนึ่ง

    [Caution]
    - ไม่เหมาะสำหรับการค้นหากว้างๆ หรือ Semantic Search (ให้ใช้ `search_all_memories` แทน)
    - ต้องระบุชื่อ Entity ที่มีแนวโน้มเป็นชื่อไฟล์จริงๆ ในระบบ

    Args:
        entity_name (str): ชื่อ Entity หรือชื่อไฟล์เป้าหมายที่ต้องการเจาะลึก เช่น 'PTT', 'Somchai', 'Interest_Rate_Hike'

    Returns:
        str: เนื้อหาของ Entity หลัก พร้อมกับเนื้อหาแบบตัดทอนของไฟล์ทั้งหมดที่เชื่อมโยงอยู่ (หรือแจ้งเตือนหากไม่พบไฟล์)
    """
    all_files = list(VAULT_PATH.rglob("*.md"))
    if not all_files:
        return "ยังไม่มีไฟล์ความจำใดใน Vault"

    # Step 1-2: หาและอ่านไฟล์หลัก
    main_file = _find_file_by_name(entity_name, all_files)
    if main_file is None:
        return f"ไม่พบไฟล์สำหรับ entity '{entity_name}' ใน Vault"

    main_content = main_file.read_text(encoding="utf-8")

    # Step 3: สกัด Wikilinks
    wikilinks = re.findall(r'\[\[(.*?)\]\]', main_content)

    output = f"--- Main Entity: {main_file.stem} ---\n{main_content}\n"

    # Step 4-5: อ่าน linked files
    if wikilinks:
        output += "\n--- Linked Connections ---\n"
        seen = set()
        for link_name in wikilinks:
            # Wikilink อาจมีรูป [[File|Alias]] — ดึงเฉพาะชื่อไฟล์
            link_target = link_name.split("|")[0].strip()
            if link_target in seen:
                continue
            seen.add(link_target)

            linked_file = _find_file_by_name(link_target, all_files)
            if linked_file:
                link_content = linked_file.read_text(encoding="utf-8")
                if len(link_content) > _LINKED_CONTENT_LIMIT:
                    link_content = link_content[:_LINKED_CONTENT_LIMIT] + "\n...[ตัดทอน]"
                output += f"\n- [{link_target}]:\n{link_content}\n"
            else:
                output += f"\n- [{link_target}]: (ไม่พบไฟล์ใน Vault)\n"

    return output


