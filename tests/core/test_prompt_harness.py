import os
import time
from pathlib import Path
import pytest

from core.prompt_harness import PromptHarness, get_harness


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
