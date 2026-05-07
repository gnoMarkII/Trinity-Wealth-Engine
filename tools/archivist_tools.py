import os
import re
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.tools import tool

from schemas.pkm_models import MemoryEntry

load_dotenv()

VAULT_PATH = Path(os.getenv("OBSIDIAN_VAULT_PATH", "./memories"))

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
        file_path.write_text(existing + "\n\n---\n\n" + body, encoding="utf-8")
        return f"เพิ่มข้อมูลสำเร็จ (append): {file_path}"

    file_path.write_text(frontmatter + body, encoding="utf-8")
    return f"บันทึกสำเร็จ (new): {file_path}"


@tool
def search_knowledge(query: str, strategy: str = "keyword") -> str:
    """ค้นหาข้อมูลในทุกไฟล์ .md ภายใน Vault รวมโฟลเดอร์ย่อย พร้อมแสดง context รอบๆ

    Args:
        query: คำหรือวลีที่ต้องการค้นหา
        strategy: 'keyword' (default) หรือ 'semantic'
    """
    # TODO: Vector RAG สำหรับ strategy='semantic'

    md_files = list(VAULT_PATH.rglob("*.md"))
    if not md_files:
        return "ยังไม่มีไฟล์ความจำใดใน Vault"

    results = []
    query_lower = query.lower()
    context_window = 2

    for file_path in md_files:
        lines = file_path.read_text(encoding="utf-8").splitlines()
        matches = []
        for i, line in enumerate(lines):
            if query_lower in line.lower():
                start = max(0, i - context_window)
                end = min(len(lines), i + context_window + 1)
                excerpt = "\n".join(
                    f"  {'>' if j == i else ' '} {lines[j]}" for j in range(start, end)
                )
                matches.append(f"  บรรทัด {i + 1}:\n{excerpt}")
        if matches:
            rel_path = file_path.relative_to(VAULT_PATH)
            results.append(f"[{rel_path}]\n" + "\n".join(matches))

    if not results:
        return f"ไม่พบคำว่า '{query}' ในไฟล์ความจำใดเลย"

    return f"ผลการค้นหา '{query}':\n\n" + "\n\n".join(results)


@tool
def read_file(filepath: str) -> str:
    """อ่านเนื้อหาไฟล์ .md จาก Vault

    Args:
        filepath: path ของไฟล์ภายใน Vault เช่น '30_Knowledge_Base/Macroeconomics/GDP.md'
    """
    file_path = VAULT_PATH / filepath
    if not file_path.exists():
        return f"ไม่พบไฟล์: {filepath}"

    content = file_path.read_text(encoding="utf-8")
    return f"=== {filepath} ===\n\n{content}"


_LINKED_CONTENT_LIMIT = 1500
_INDEX_EXCLUDE = ("00_Inbox", "01_Daily_Logs")


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


@tool
def update_master_index() -> str:
    """สแกน Vault และสร้าง/อัปเดตสารบัญ index.md อัตโนมัติ จัดกลุ่มตาม Folder
    ข้ามโฟลเดอร์ 00_Inbox และ Daily_Logs รวมถึงไฟล์ระบบ log.md และ index.md
    """
    _SYSTEM_FILES = {"log.md", "index.md"}
    all_files = sorted(VAULT_PATH.rglob("*.md"))

    folder_map: dict[str, list[tuple[str, str]]] = {}
    for file_path in all_files:
        rel = file_path.relative_to(VAULT_PATH)
        if rel.name in _SYSTEM_FILES:
            continue
        if any(excl in rel.parts for excl in _INDEX_EXCLUDE):
            continue

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
def lint_structural_health() -> str:
    """ตรวจสุขภาพเชิงโครงสร้างของ Vault: ค้นหา Orphan files (ไม่มีไฟล์ใด Wikilink โยงมา)
    และ Empty files (ไม่มีเนื้อหา) ส่งรายงานกลับมาให้ทราบว่าไฟล์ไหนต้องจัดการ
    """
    _SYSTEM_FILES = {"log.md", "index.md"}
    all_files = [f for f in VAULT_PATH.rglob("*.md") if f.name not in _SYSTEM_FILES]

    if not all_files:
        return "ไม่มีไฟล์ใดใน Vault ที่จะตรวจสอบ"

    stem_to_path = {f.stem: f for f in all_files}

    outbound: dict[str, set[str]] = {f.stem: set() for f in all_files}
    inbound: dict[str, set[str]] = {f.stem: set() for f in all_files}
    for file_path in all_files:
        for raw in re.findall(r'\[\[(.*?)\]\]', file_path.read_text(encoding="utf-8")):
            target = raw.split("|")[0].strip()
            if target in stem_to_path:
                outbound[file_path.stem].add(target)
                inbound[target].add(file_path.stem)

    orphans = sorted(s for s in stem_to_path if not outbound[s] and not inbound[s])

    empty_files = []
    for file_path in all_files:
        raw = file_path.read_text(encoding="utf-8").strip()
        parts = raw.split("---", 2) if raw.startswith("---") else ["", "", raw]
        body = parts[2].strip() if len(parts) >= 3 else raw
        if not body:
            empty_files.append(file_path.stem)

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


def _find_file_by_name(name: str, all_files: list[Path]) -> Path | None:
    """ค้นหาไฟล์โดย stem ตรงทั้งหมดก่อน แล้ว fallback ไป partial match"""
    name_lower = name.lower()
    exact = next((f for f in all_files if f.stem.lower() == name_lower), None)
    if exact:
        return exact
    return next((f for f in all_files if name_lower in f.stem.lower()), None)


@tool
def search_graph_context(entity_name: str, depth: int = 1) -> str:
    """ค้นหาข้อมูล entity พร้อม linked entities ในรูปแบบ Graph (GraphRAG)
    ใช้เมื่อต้องการวิเคราะห์บริษัท, บุคคล, หรือเหตุการณ์แบบเจาะลึกพร้อมบริบทโดยรอบ

    Args:
        entity_name: ชื่อ entity ที่ต้องการค้นหา เช่น 'PTT', 'Somchai'
        depth: ความลึกของการ traverse (1 = entity หลัก + linked entities ชั้นแรก)
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
    if depth >= 1 and wikilinks:
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
