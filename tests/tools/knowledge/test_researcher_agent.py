import pytest
from pathlib import Path
from agents.manager_agent import _sanitize_researcher_instruction


def test_sanitize_researcher_instruction_strips_save_language():
    inst1 = "ดึงข้อมูลข่าวหุ้น AAPL แล้วให้บันทึกใน Vault ทันที"
    sanitized1 = _sanitize_researcher_instruction(inst1)
    assert "บันทึก" not in sanitized1
    assert "Vault" not in sanitized1
    assert "AAPL" in sanitized1
    
    inst2 = "Please ingest financial health for MSFT and save to vault"
    sanitized2 = _sanitize_researcher_instruction(inst2)
    assert "save to vault" not in sanitized2.lower()
    assert "MSFT" in sanitized2
    
    inst3 = "แล้วให้บันทึกลงวอลท์ด้วยนะ"
    sanitized3 = _sanitize_researcher_instruction(inst3)
    assert "บันทึก" not in sanitized3
    assert "วอลท์" not in sanitized3


def test_researcher_prompt_contains_absolute_no_save_rule():
    prompt_path = Path(__file__).parent.parent.parent.parent / "prompts" / "skills" / "researcher" / "SKILL.md"
    content = prompt_path.read_text(encoding="utf-8")
    assert "ห้ามเรียกเครื่องมือ write_raw_markdown" in content
    assert "[กฎเหล็กอันดับ 1 — ห้ามมีข้อยกเว้น]" in content
