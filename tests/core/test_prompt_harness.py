import os
import time
from pathlib import Path
import pytest

from core.prompt_harness import TOOLS_PROMPTS_ROOT, PromptHarness, get_harness


def test_mustache_formatting(tmp_path: Path):
    harness = PromptHarness("test_agent", skills_root=tmp_path)
    template = "Hello {{ name }}, here is your JSON: {\"key\": \"{{val}}\", \"other\": 123} and untouched {{missing}}."
    
    # Non-strict mode leaves missing variables untouched
    formatted = harness.format_mustache(template, strict=False, name="World", val="value456")
    assert "Hello World, here is your JSON:" in formatted
    assert "{\"key\": \"value456\", \"other\": 123}" in formatted
    assert "untouched {{missing}}." in formatted
    
    # Strict mode (default) raises ValueError when variables are missing
    with pytest.raises(ValueError, match="Missing required Mustache variable"):
        harness.format_mustache(template, strict=True, name="World", val="value456")


def test_multi_file_composition(tmp_path: Path):
    agent_dir = tmp_path / "test_agent"
    agent_dir.mkdir(parents=True)
    
    (agent_dir / "SKILL.md").write_text("# Core Skill for {{name}}", encoding="utf-8")
    (agent_dir / "pillars.md").write_text("## Pillars Section", encoding="utf-8")
    (agent_dir / "guardrails.md").write_text("## Guardrails: Do not guess", encoding="utf-8")
    
    harness = PromptHarness("test_agent", skills_root=tmp_path)
    system_prompt = harness.get_system_prompt(name="Allocator")
    
    assert "# Core Skill for Allocator" in system_prompt
    assert "---" in system_prompt
    assert "## Pillars Section" in system_prompt
    assert "## Guardrails: Do not guess" in system_prompt


def test_hot_reload_cache(tmp_path: Path):
    agent_dir = tmp_path / "cache_agent"
    agent_dir.mkdir(parents=True)
    
    skill_file = agent_dir / "SKILL.md"
    skill_file.write_text("Initial Content v1", encoding="utf-8")
    
    harness = PromptHarness("cache_agent", skills_root=tmp_path)
    content1 = harness.get_system_prompt()
    assert content1 == "Initial Content v1"
    
    # Modify file and update mtime
    time.sleep(0.01)
    skill_file.write_text("Updated Content v2", encoding="utf-8")
    # Force timestamp change for OS file system granularity if needed
    current_mtime = os.path.getmtime(skill_file)
    os.utime(skill_file, (current_mtime + 2, current_mtime + 2))
    
    content2 = harness.get_system_prompt()
    assert content2 == "Updated Content v2"


def test_few_shots_feedback(tmp_path: Path):
    agent_dir = tmp_path / "shot_agent"
    agent_dir.mkdir(parents=True)
    
    (agent_dir / "SKILL.md").write_text("Main skill", encoding="utf-8")
    (agent_dir / "few_shots.md").write_text("{\"example\": \"{{ex}}\"}", encoding="utf-8")
    
    harness = PromptHarness("shot_agent", skills_root=tmp_path)
    feedback = harness.get_few_shots_feedback("Error: missing field", ex="valid_data")
    assert "Error: missing field" in feedback
    assert "[ตัวอย่างรูปแบบข้อมูลและ JSON ที่ถูกต้อง (Few-Shot Reference)]" in feedback
    assert "{\"example\": \"valid_data\"}" in feedback


class TestToolsPromptsRoot:
    """prompts/tools/ เป็น root แยกจาก prompts/skills/ — ใช้ skills_root=TOOLS_PROMPTS_ROOT
    เข้าถึงได้โดยไม่ต้องแก้ PromptHarness class เลย (ใช้ constructor param ที่มีอยู่แล้ว)"""

    def test_tools_prompts_root_points_at_prompts_tools_dir(self):
        assert TOOLS_PROMPTS_ROOT.name == "tools"
        assert TOOLS_PROMPTS_ROOT.parent.name == "prompts"

    def test_extractor_skill_md_loads_from_real_prompts_tools_dir(self):
        """โหลดไฟล์จริงที่ core.py ใช้ตอน production — regression guard ถ้ามีคนย้าย/ลบไฟล์ไป"""
        prompt = get_harness("extractor", skills_root=TOOLS_PROMPTS_ROOT).get_system_prompt()
        assert "หุ่นยนต์สกัดข้อมูลการลงทุน" in prompt
        assert "## ใจความสำคัญ" in prompt

    def test_dynamic_list_variable_stays_at_templated_position(self, tmp_path: Path):
        """จำลองปัญหาที่เจอตอนรีวิว: ถ้า concat dynamic list ต่อท้าย get_skill_text() แทนที่จะ
        ใช้ Mustache variable ตรงตำแหน่งเดิม จะทำให้ลำดับ prompt ผิดเพี้ยนจากของเดิม (เช่น
        candidate_list ต้องอยู่ก่อนกฎ ไม่ใช่หลังกฎ) — เทสต์นี้ล็อกว่าตำแหน่งต้องถูกต้องตาม template"""
        agent_dir = tmp_path / "order_agent"
        agent_dir.mkdir(parents=True)
        (agent_dir / "SKILL.md").write_text(
            "intro\n{{dynamic_list}}\nrules after list", encoding="utf-8"
        )
        harness = PromptHarness("order_agent", skills_root=tmp_path)
        result = harness.get_skill_text("SKILL.md", dynamic_list="item1\nitem2")

        intro_pos = result.index("intro")
        list_pos = result.index("item1")
        rules_pos = result.index("rules after list")
        assert intro_pos < list_pos < rules_pos


def test_selective_sections_and_metrics(tmp_path: Path):
    agent_dir = tmp_path / "selective_agent"
    agent_dir.mkdir(parents=True)

    (agent_dir / "SKILL.md").write_text("# Core Skill", encoding="utf-8")
    (agent_dir / "pillars.md").write_text("## Pillars", encoding="utf-8")
    (agent_dir / "guardrails.md").write_text("## Guardrails", encoding="utf-8")
    (agent_dir / "extra.md").write_text("## Extra Section", encoding="utf-8")

    harness = PromptHarness("selective_agent", skills_root=tmp_path)

    # 1. Include only pillars (exclude guardrails)
    p_only = harness.get_system_prompt(include_pillars=True, include_guardrails=False)
    assert "# Core Skill" in p_only
    assert "## Pillars" in p_only
    assert "## Guardrails" not in p_only

    # 2. Include extra sections
    with_extra = harness.get_system_prompt(
        include_pillars=False, include_guardrails=False, extra_sections=["extra.md"]
    )
    assert "## Extra Section" in with_extra
    assert "## Pillars" not in with_extra

    # 3. Test prompt metrics
    metrics = harness.get_prompt_metrics()
    assert "SKILL.md_chars" in metrics
    assert "pillars.md_chars" in metrics
    assert "total_chars" in metrics
    assert metrics["total_chars"] > 0
    assert metrics["total_est_tokens"] > 0
