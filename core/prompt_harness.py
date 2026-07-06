import os
import re
import logging
from typing import Dict, Any, Optional
from pathlib import Path

from core.text_utils import repair_mojibake

log = logging.getLogger(__name__)

class PromptHarness:
    """
    Skill Harness Loader & Dynamic Prompt Injector.
    
    Features:
    1. In-memory caching with mtime checking (Hot-reloading without restarting app).
    2. Mustache-style template formatting (`{{variable}}`), safe from JSON single brace `{ }` collisions.
    3. Mojibake repair for TIS-620/Latin-1 encoding issues.
    4. Composes multi-file skills (e.g. SKILL.md + pillars.md + guardrails.md).
    5. Dynamic few-shot injection for structured output retry layers.
    6. Strict variable validation preventing unformatted Mustache placeholders in production prompts.
    """
    def __init__(self, agent_name: str, skills_root: Optional[str | Path] = None):
        self.agent_name = agent_name
        if skills_root is None:
            # Default: <project_root>/prompts/skills/<agent_name>
            base_dir = Path(__file__).resolve().parent.parent
            self.agent_dir = base_dir / "prompts" / "skills" / agent_name
        else:
            self.agent_dir = Path(skills_root) / agent_name
            
        # Cache format: {filename: (mtime, content)}
        self._cache: Dict[str, tuple[float, str]] = {}

    def _load_file_with_cache(self, filename: str, required: bool = True) -> str:
        filepath = self.agent_dir / filename
        if not filepath.exists():
            if required:
                raise FileNotFoundError(f"Skill file missing: {filepath}")
            return ""
            
        mtime = os.path.getmtime(filepath)
        if filename in self._cache:
            cached_mtime, cached_content = self._cache[filename]
            if cached_mtime == mtime:
                return cached_content
                
        log.info(f"Loading/hot-reloading skill file: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            raw_content = f.read()
            
        content = repair_mojibake(raw_content).strip()
        self._cache[filename] = (mtime, content)
        return content

    def format_mustache(self, template: str, strict: bool = True, **kwargs: Any) -> str:
        """
        Replaces {{variable_name}} with kwargs[variable_name].
        If strict is True and variable_name is not provided in kwargs, raises ValueError.
        If strict is False, leaves {{variable_name}} unchanged.
        """
        def replace_match(match: re.Match) -> str:
            key = match.group(1).strip()
            if key in kwargs:
                val = kwargs[key]
                return repair_mojibake(str(val)) if isinstance(val, str) else str(val)
            if strict:
                raise ValueError(f"Missing required Mustache variable: '{key}' in template for agent '{self.agent_name}'")
            return match.group(0)
            
        return re.sub(r'\{\{\s*(\w+)\s*\}\}', replace_match, template)

    def get_skill_text(self, filename: str, required: bool = True, strict: bool = True, **kwargs: Any) -> str:
        """Loads a specific skill file and formats it with kwargs."""
        content = self._load_file_with_cache(filename, required=required)
        if not content:
            return ""
        return self.format_mustache(content, strict=strict, **kwargs)

    def get_system_prompt(self, strict: bool = True, **kwargs: Any) -> str:
        """
        Composes the main system prompt.
        Looks for SKILL.md (or <AGENT_NAME>_SKILL.md).
        If pillars.md and/or guardrails.md exist, appends them in sequence.
        """
        # Try SKILL.md first, fallback to <AGENT_NAME.upper()>_SKILL.md
        main_text = self._load_file_with_cache("SKILL.md", required=False)
        if not main_text:
            alt_name = f"{self.agent_name.upper()}_SKILL.md"
            main_text = self._load_file_with_cache(alt_name, required=True)
            
        sections = [main_text]
        
        # Check for optional sub-modules (pillars, guardrails)
        pillars_text = self._load_file_with_cache("pillars.md", required=False)
        if pillars_text:
            sections.append(pillars_text)
            
        guardrails_text = self._load_file_with_cache("guardrails.md", required=False)
        if guardrails_text:
            sections.append(guardrails_text)
            
        combined = "\n\n---\n\n".join(sections)
        return self.format_mustache(combined, strict=strict, **kwargs)

    def get_few_shots_feedback(self, base_feedback: str, strict: bool = True, **kwargs: Any) -> str:
        """
        Appends few_shots.md to retry feedback if available.
        Used by invoke_with_retry for Dynamic Few-Shot Injection.
        """
        few_shots = self._load_file_with_cache("few_shots.md", required=False)
        if not few_shots:
            return base_feedback
        formatted_shots = self.format_mustache(few_shots, strict=strict, **kwargs)
        return f"{base_feedback}\n\n[ตัวอย่างรูปแบบข้อมูลและ JSON ที่ถูกต้อง (Few-Shot Reference)]\n{formatted_shots}"


_harness_instances: Dict[str, PromptHarness] = {}

def get_harness(agent_name: str, skills_root: Optional[str | Path] = None) -> PromptHarness:
    """Returns a singleton PromptHarness instance for the specified agent."""
    key = f"{agent_name}:{skills_root}"
    if key not in _harness_instances:
        _harness_instances[key] = PromptHarness(agent_name, skills_root)
    return _harness_instances[key]
