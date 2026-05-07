import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

VAULT_PATH = Path(os.getenv("OBSIDIAN_VAULT_PATH", "./memories"))


def _ensure_vault() -> None:
    VAULT_PATH.mkdir(parents=True, exist_ok=True)


@tool
def write_agent_memory(agent_name: str, topic: str, content: str) -> str:
    """บันทึกหรือเพิ่มข้อมูลลงไฟล์ {agent_name}.md ใน Obsidian Vault
    ถ้าไฟล์มีอยู่แล้วจะ append หัวข้อใหม่ต่อท้าย ถ้าไม่มีจะสร้างใหม่

    Args:
        agent_name: ชื่อ Agent เช่น Risk_Officer, Portfolio_Manager
        topic: หัวข้อที่จะบันทึก
        content: เนื้อหาที่จะบันทึก
    """
    _ensure_vault()
    file_path = VAULT_PATH / f"{agent_name}.md"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if file_path.exists():
        existing = file_path.read_text(encoding="utf-8")
        separator = "\n\n---\n\n"
    else:
        existing = f"# {agent_name}\n\n"
        separator = ""

    new_section = f"## {topic}\n*บันทึกเมื่อ: {timestamp}*\n\n{content}\n"
    file_path.write_text(existing + separator + new_section, encoding="utf-8")

    return f"บันทึกสำเร็จ: [{agent_name}] หัวข้อ '{topic}' → {file_path}"


@tool
def read_agent_memory(agent_name: str) -> str:
    """อ่านข้อมูลทั้งหมดในไฟล์ความจำของ Agent ที่ระบุ

    Args:
        agent_name: ชื่อ Agent เช่น Risk_Officer, Portfolio_Manager
    """
    _ensure_vault()
    file_path = VAULT_PATH / f"{agent_name}.md"

    if not file_path.exists():
        return f"ไม่พบไฟล์ความจำสำหรับ Agent '{agent_name}' (ยังไม่มีการบันทึก)"

    content = file_path.read_text(encoding="utf-8")
    return f"=== ความจำของ {agent_name} ===\n\n{content}"


@tool
def search_all_memories(keyword: str) -> str:
    """ค้นหาคำหรือวลีในไฟล์ความจำทุกไฟล์ใน Vault

    Args:
        keyword: คำหรือวลีที่ต้องการค้นหา
    """
    _ensure_vault()
    md_files = list(VAULT_PATH.glob("*.md"))

    if not md_files:
        return "ยังไม่มีไฟล์ความจำใดใน Vault"

    results = []
    keyword_lower = keyword.lower()

    for file_path in md_files:
        content = file_path.read_text(encoding="utf-8")
        matching_lines = [
            f"  บรรทัด {i + 1}: {line.strip()}"
            for i, line in enumerate(content.splitlines())
            if keyword_lower in line.lower()
        ]
        if matching_lines:
            results.append(f"[{file_path.stem}]\n" + "\n".join(matching_lines))

    if not results:
        return f"ไม่พบคำว่า '{keyword}' ในไฟล์ความจำใดเลย"

    return f"ผลการค้นหา '{keyword}':\n\n" + "\n\n".join(results)
