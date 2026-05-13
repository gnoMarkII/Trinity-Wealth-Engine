import os
import re
import shutil
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter

from schemas.pkm_models import MemoryEntry

load_dotenv()

VAULT_PATH = Path(os.getenv("OBSIDIAN_VAULT_PATH", "./memories"))
CHROMA_PATH = VAULT_PATH / ".chroma_index"
_CHROMA_MTIME_FILE = VAULT_PATH / ".chroma_mtime"

_vs_cache: dict = {}  # {"mtime_sum": float, "vs": Chroma}

_VAULT_FOLDERS = [
    "00_Inbox",
    "01_Daily_Logs",
    "10_System_Agents",
    "20_Portfolio_Management/Current_Holdings",
    "20_Portfolio_Management/Trading_Journals",
    "30_Knowledge_Base/Macroeconomics",
    "30_Knowledge_Base/Stocks",
    "30_Knowledge_Base/Strategies",
    "40_Finance_and_Tax/Tax_Deductions",
    "40_Finance_and_Tax/Capital_Gains",
    "99_Templates",
]


def init_vault_structure() -> None:
    for folder in _VAULT_FOLDERS:
        (VAULT_PATH / folder).mkdir(parents=True, exist_ok=True)


init_vault_structure()



@lru_cache(maxsize=1)
def get_embeddings():
    print("\n[System]: ⏳ กำลังเตรียมระบบความจำ (Semantic Search)... (อาจใช้เวลาโหลดโมเดลสักครู่ในครั้งแรก)")
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


@tool
def save_memory(
    title: str,
    content: str,
    folder_path: str,
    tags: list[str],
    entity_type: str,
    aliases: list[str] | None = None,
    linked_files: list[str] | None = None,
) -> str:
    """บันทึก MemoryEntry ลง Obsidian Vault พร้อม YAML frontmatter และ Wikilinks

    Args:
        title: ชื่อ entity หรือหัวข้อ ใช้เป็นชื่อไฟล์ .md
        content: เนื้อหาหลักในรูปแบบ Markdown เกี่ยวกับ entity นี้โดยตรง
        folder_path: โฟลเดอร์ปลายทาง เช่น '30_Knowledge_Base/Stocks'
        tags: รายการ tag เช่น ['energy', 'SET100']
        entity_type: ประเภทของ entity เช่น 'Company', 'Executive', 'Macro_Event'
        aliases: ชื่อเรียกอื่นๆ เช่น ['ปตท.', 'PTT PCL']
        linked_files: ชื่อไฟล์ที่ต้องการสร้าง Wikilinks (ไม่รวม .md)
    """
    entry = MemoryEntry(
        title=title,
        content=content,
        folder_path=folder_path,
        tags=tags,
        entity_type=entity_type,
        aliases=aliases or [],
        linked_files=linked_files or [],
    )

    target_dir = VAULT_PATH / entry.folder_path
    target_dir.mkdir(parents=True, exist_ok=True)

    safe_title = entry.title.replace("/", "-").replace("\\", "-")
    file_path = target_dir / f"{safe_title}.md"

    date_str = datetime.now().strftime("%Y-%m-%d")
    tags_yaml = "[" + ", ".join(entry.tags) + "]"
    aliases_yaml = "[" + ", ".join(f'"{a}"' for a in entry.aliases) + "]" if entry.aliases else "[]"
    frontmatter = (
        f"---\n"
        f"title: {entry.title}\n"
        f"entity_type: {entry.entity_type}\n"
        f"aliases: {aliases_yaml}\n"
        f"tags: {tags_yaml}\n"
        f"date: {date_str}\n"
        f"---\n\n"
    )

    body = entry.content
    if entry.linked_files:
        wikilinks = "\n".join(f"- [[{f}]]" for f in entry.linked_files)
        body += f"\n\n## Related\n{wikilinks}\n"

    if file_path.exists():
        existing = file_path.read_text(encoding="utf-8")
        if existing.startswith("---"):
            parts = existing.split("---", 2)
            if len(parts) >= 3:
                fm = re.sub(r'\nlast_updated: .+', '', parts[1]).rstrip()
                fm += f"\nlast_updated: {date_str}\n"
                existing = "---" + fm + "---" + parts[2]
        update_header = f"## Update — {date_str}\n\n"
        file_path.write_text(existing + "\n\n---\n\n" + update_header + body, encoding="utf-8")
        _rebuild_index()
        return f"เพิ่มข้อมูลสำเร็จ (append): {file_path}"

    file_path.write_text(frontmatter + body, encoding="utf-8")
    _rebuild_index()
    return f"บันทึกสำเร็จ (new): {file_path}"


@tool
def write_raw_markdown(content: str, folder_path: str, filename: str) -> str:
    """บันทึกไฟล์ Markdown ดิบลง Vault โดยตรง ไม่แปลงหรือประมวลผลเนื้อหา
    ใช้สำหรับข้อมูลที่มี YAML frontmatter พร้อมแล้ว เช่น ผลลัพธ์จาก Researcher tools

    Args:
        content: เนื้อหา Markdown พร้อม frontmatter ที่ต้องการบันทึก
        folder_path: โฟลเดอร์ปลายทาง เช่น '30_Knowledge_Base/Macroeconomics'
        filename: ชื่อไฟล์ไม่รวมนามสกุล เช่น 'Macro_Snapshot_2025-01-15'
    """
    target_dir = VAULT_PATH / folder_path
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_name = filename.replace("/", "-").replace("\\", "-")
    file_path = target_dir / f"{safe_name}.md"
    existed = file_path.exists()
    file_path.write_text(content, encoding="utf-8")
    _rebuild_index()
    action = "overwritten" if existed else "new"
    return f"บันทึกสำเร็จ (raw, {action}): {file_path}"


@tool
def read_file(filepath: str) -> str:
    """อ่านเนื้อหาไฟล์ .md จาก Vault

    Args:
        filepath: path ของไฟล์ภายใน Vault เช่น '30_Knowledge_Base/Macroeconomics/GDP.md'
    """
    file_path = VAULT_PATH / filepath
    if not file_path.exists():
        return f"ไม่พบไฟล์: {filepath}"

    _READ_FILE_LIMIT = 8000
    content = file_path.read_text(encoding="utf-8")
    if len(content) > _READ_FILE_LIMIT:
        content = content[:_READ_FILE_LIMIT] + f"\n\n...[ตัดทอน — ไฟล์ยาว {len(content)} ตัวอักษร]"
    return f"=== {filepath} ===\n\n{content}"


_LINKED_CONTENT_LIMIT = 1500
_INDEX_EXCLUDE = ("00_Inbox", "01_Daily_Logs")
_VAULT_SYSTEM_FILES = {"log.md", "index.md"}


@tool
def write_log(action: str, target: str, summary: str) -> str:
    """บันทึกประวัติการทำงานของ Archivist ลงใน log.md ที่ Root ของ Vault (append)

    Args:
        action: ประเภทการกระทำ เช่น 'SAVE', 'READ', 'UPDATE', 'LINK'
        target: ชื่อไฟล์หรือ entity ที่เกี่ยวข้อง
        summary: สรุปสั้นๆ ว่าทำอะไรไปและผลลัพธ์เป็นอย่างไร
    """
    log_path = VAULT_PATH / "log.md"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    entry = f"\n## [{timestamp}] {action} | {target}\n{summary}\n"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(entry)
    return f"บันทึก log สำเร็จ: [{timestamp}] {action} | {target}"


def _rebuild_index() -> str:
    """สแกน Vault และเขียน index.md ใหม่ — เรียกได้ทั้งจาก tool และจาก save functions โดยตรง"""
    all_files = [
        f for f in sorted(VAULT_PATH.rglob("*.md"))
        if f.name not in _VAULT_SYSTEM_FILES
        and not any(excl in f.parts for excl in _INDEX_EXCLUDE)
    ]

    folder_map: dict[str, list[tuple[str, str]]] = {}
    for file_path in all_files:
        rel = file_path.relative_to(VAULT_PATH)
        folder = str(rel.parent) if rel.parent != Path(".") else "Root"
        content = file_path.read_text(encoding="utf-8")
        m = re.search(r'^entity_type:\s*(.+)$', content, re.MULTILINE)
        entity_type = m.group(1).strip() if m else "—"
        folder_map.setdefault(folder, []).append((file_path.stem, entity_type))

    if not folder_map:
        return "ไม่มีไฟล์ที่จะ index ใน Vault"

    lines = [
        "---",
        "title: Master Index",
        f"date: {datetime.now().strftime('%Y-%m-%d')}",
        "---",
        "",
        "# Master Index",
        "",
    ]
    for folder in sorted(folder_map):
        lines += [f"## {folder}", "", "| File | Entity Type |", "|------|-------------|"]
        for stem, etype in sorted(folder_map[folder]):
            lines.append(f"| [[{stem}]] | {etype} |")
        lines.append("")

    (VAULT_PATH / "index.md").write_text("\n".join(lines), encoding="utf-8")
    total = sum(len(v) for v in folder_map.values())
    return f"อัปเดต index.md สำเร็จ: {total} ไฟล์ใน {len(folder_map)} โฟลเดอร์"


@tool
def update_master_index() -> str:
    """สแกน Vault และสร้าง/อัปเดตสารบัญ index.md อัตโนมัติ จัดกลุ่มตาม Folder
    ข้ามโฟลเดอร์ 00_Inbox และ Daily_Logs รวมถึงไฟล์ระบบ log.md และ index.md
    """
    return _rebuild_index()


@tool
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


_SEMANTIC_CONTENT_LIMIT = 2000


@tool
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


@tool
def search_all_memories(keyword: str) -> str:
    """ค้นหาความจำทั้งหมดใน Vault ด้วย Semantic Search (Vector RAG) แบบ Local
    เหมาะสำหรับคำถามที่ต้องการความเข้าใจความหมาย ไม่ใช่แค่ keyword ตรงๆ

    Args:
        keyword: คำถามหรือวลีที่ต้องการค้นหาตามความหมาย
    """
    _SEARCH_SYSTEM_FILES = {"log.md", "index.md"}
    md_files = [
        f for f in VAULT_PATH.rglob("*.md")
        if f.name not in _SEARCH_SYSTEM_FILES
        and not any(excl in f.parts for excl in _INDEX_EXCLUDE)
    ]
    if not md_files:
        return "ยังไม่มีไฟล์ความจำใดใน Vault"

    mtime_sum = sum(f.stat().st_mtime for f in md_files)

    # 1) In-memory cache hit — ไม่ต้องทำอะไรเพิ่ม
    if _vs_cache.get("mtime_sum") == mtime_sum and "vs" in _vs_cache:
        vectorstore = _vs_cache["vs"]
    else:
        # 2) On-disk cache hit — โหลดจาก persist_directory ถ้า mtime ตรงกัน
        stored_mtime: float | None = None
        if _CHROMA_MTIME_FILE.exists():
            try:
                stored_mtime = float(_CHROMA_MTIME_FILE.read_text().strip())
            except (ValueError, OSError):
                pass

        vectorstore = None
        if stored_mtime == mtime_sum and CHROMA_PATH.exists():
            try:
                vectorstore = Chroma(
                    persist_directory=str(CHROMA_PATH),
                    embedding_function=get_embeddings(),
                )
            except Exception:
                vectorstore = None  # ล้มเหลว → rebuild

        # 3) Rebuild จาก scratch แล้ว persist ลงดิสก์
        if vectorstore is None:
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            split_texts: list[str] = []
            split_metas: list[dict] = []
            for file_path in md_files:
                content = file_path.read_text(encoding="utf-8")
                rel = str(file_path.relative_to(VAULT_PATH))
                for chunk in splitter.split_text(content):
                    split_texts.append(chunk)
                    split_metas.append({"source": rel})

            if not split_texts:
                return "ไม่พบเนื้อหาในไฟล์ความจำที่สามารถค้นหาได้"

            if CHROMA_PATH.exists():
                shutil.rmtree(CHROMA_PATH)

            try:
                vectorstore = Chroma.from_texts(
                    texts=split_texts,
                    metadatas=split_metas,
                    embedding=get_embeddings(),
                    persist_directory=str(CHROMA_PATH),
                )
                _CHROMA_MTIME_FILE.write_text(str(mtime_sum))
            except Exception as e:
                return f"เกิดข้อผิดพลาดในการสร้าง vectorstore: {e}"

        _vs_cache["mtime_sum"] = mtime_sum
        _vs_cache["vs"] = vectorstore

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
    """ค้นหาข้อมูล entity พร้อม linked entities ในรูปแบบ Graph (GraphRAG)
    ใช้เมื่อต้องการวิเคราะห์บริษัท, บุคคล, หรือเหตุการณ์แบบเจาะลึกพร้อมบริบทโดยรอบ

    Args:
        entity_name: ชื่อ entity ที่ต้องการค้นหา เช่น 'PTT', 'Somchai'
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
