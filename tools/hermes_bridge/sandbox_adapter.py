"""Sandbox Adapter implementing SandboxAdapterPort for manager_sandbox/."""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from filelock import FileLock

from core.sandbox.ast_guard import validate_scratch_python
from core.sandbox.memory_hygiene import sanitize_memory_markdown
from core.sandbox.models import (
    SandboxExecutionResult,
    SandboxStateSnapshot,
    SkillManifest,
)
from tools._atomic_io import _atomic_write_to

log = logging.getLogger(__name__)

DEFAULT_SANDBOX_ROOT = Path("manager_sandbox")


def parse_skill_frontmatter(markdown_text: str) -> tuple[dict[str, str], str]:
    """Parse YAML frontmatter (--- ... ---) from SKILL.md.

    Returns:
        (metadata_dict, body_text)
    """
    metadata: dict[str, str] = {}
    lines = markdown_text.splitlines()
    body_lines = lines

    if len(lines) >= 3 and lines[0].strip() == "---":
        end_idx = -1
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                end_idx = i
                break
        if end_idx != -1:
            fm_lines = lines[1:end_idx]
            body_lines = lines[end_idx + 1 :]
            for line in fm_lines:
                if ":" in line:
                    key, val = line.split(":", 1)
                    metadata[key.strip().lower()] = val.strip()

    # Determine description
    desc = metadata.get("description", "")
    if not desc:
        for bl in body_lines:
            stripped = bl.strip()
            if stripped and not stripped.startswith("#") and stripped != "---":
                desc = stripped[:80]
                break
    if not desc:
        desc = "User or agent skill"
    metadata["description"] = desc

    body_text = "\n".join(body_lines).strip()
    return metadata, body_text


class SandboxAdapter:
    """Concrete adapter for managing file operations inside manager_sandbox/."""

    def __init__(self, sandbox_root: Path = DEFAULT_SANDBOX_ROOT) -> None:
        self._root = Path(sandbox_root)
        self._memory_dir = self._root / "memory"
        self._skills_dir = self._root / "skills"
        self._quarantine_dir = self._skills_dir / "_quarantine"
        self._scratch_dir = self._root / "scratch"
        self._audit_file = self._root / "audit" / "execution_audit.jsonl"
        self._lock_path = self._root / ".sandbox.lock"

        # Ensure directories exist
        for directory in (self._memory_dir, self._skills_dir, self._quarantine_dir, self._scratch_dir, self._audit_file.parent):
            directory.mkdir(parents=True, exist_ok=True)

        # Ensure default memory files exist
        default_files = {
            "USER.md": (
                "# 👤 ข้อมูลนักลงทุน (Investor Profile)\n\n"
                "- เป้าหมาย: สร้างความมั่งคั่งระยะยาวและรักษาเงินต้น\n"
                "- ระดับความเสี่ยงที่ยอมรับได้: ปานกลางค่อนข้างสูง (Moderate-Aggressive)\n"
                "- กรอบการจัดสรรเงิน: เน้น 3 ถังสภาพคล่อง (Liquidity, Core Compounder, Growth)\n"
            ),
            "MEMORY.md": (
                "# 🧠 ข้อสรุปพอร์ตและบทเรียนในอดีต (Portfolio Memory)\n\n"
                "- ตรวจสอบความเบี่ยงเบนของพอร์ตสม่ำเสมอ\n"
                "- ให้ความสำคัญกับ Margin of Safety และหลีกเลี่ยง Value Trap\n"
                "- ติดตามสัญญาของผู้บริหาร (Say-Do Ratio) ของหุ้นรายตัว\n"
            ),
            "DREAMS.md": (
                "# 🌌 ความรู้ที่ตกผลึก (Synthesized Insights)\n\n"
                "- วิเคราะห์ปัจจัยมหภาคควบคู่กับพื้นฐานหุ้นรายตัว (Top-Down & Bottom-Up)\n"
                "- รักษาวินัยการตัดขาดทุนเมื่อเกิด Invalidation Triggers\n"
            ),
        }
        for fname, default_content in default_files.items():
            fpath = self._memory_dir / fname
            if not fpath.is_file():
                _atomic_write_to(fpath, default_content)

    def _append_audit(self, action: str, details: dict) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "action": action,
            **details,
        }
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        with FileLock(str(self._lock_path)):
            existing = ""
            if self._audit_file.is_file():
                try:
                    existing = self._audit_file.read_text(encoding="utf-8")
                except Exception:
                    existing = ""
            _atomic_write_to(self._audit_file, existing + line)

    def get_snapshot(self) -> SandboxStateSnapshot:
        """Scan manager_sandbox/ and return an immutable domain snapshot."""
        # Memory files
        memory_files = tuple(
            p.name for p in sorted(self._memory_dir.glob("*.md")) if not p.name.startswith(".")
        )

        # Active skills
        active_skills: list[SkillManifest] = []
        if self._skills_dir.is_dir():
            for skill_path in sorted(self._skills_dir.iterdir()):
                if skill_path.is_dir() and skill_path.name != "_quarantine":
                    skill_file = skill_path / "SKILL.md"
                    desc = "User or agent skill"
                    if skill_file.is_file():
                        try:
                            fm, _ = parse_skill_frontmatter(skill_file.read_text(encoding="utf-8"))
                            desc = fm.get("description", desc)
                        except Exception:
                            pass
                    active_skills.append(
                        SkillManifest(
                            name=skill_path.name,
                            description=desc,
                            is_quarantined=False,
                            path=str(skill_path),
                        )
                    )

        # Quarantined skills
        quarantined_skills: list[SkillManifest] = []
        if self._quarantine_dir.is_dir():
            for skill_path in sorted(self._quarantine_dir.iterdir()):
                if skill_path.is_dir():
                    skill_file = skill_path / "SKILL.md"
                    desc = "Quarantined pending verification"
                    if skill_file.is_file():
                        try:
                            fm, _ = parse_skill_frontmatter(skill_file.read_text(encoding="utf-8"))
                            desc = fm.get("description", desc)
                        except Exception:
                            pass
                    quarantined_skills.append(
                        SkillManifest(
                            name=skill_path.name,
                            description=desc,
                            is_quarantined=True,
                            path=str(skill_path),
                        )
                    )

        # Scratch files count
        scratch_files = [
            p for p in self._scratch_dir.iterdir() if p.is_file() and not p.name.startswith(".")
        ]

        # Last audit line
        last_audit: Optional[str] = None
        if self._audit_file.is_file():
            try:
                lines = self._audit_file.read_text(encoding="utf-8").splitlines()
                if lines:
                    last_audit = lines[-1]
            except Exception:
                pass

        return SandboxStateSnapshot(
            memory_files=memory_files,
            active_skills=tuple(active_skills),
            quarantined_skills=tuple(quarantined_skills),
            scratch_files_count=len(scratch_files),
            last_audit_entry=last_audit,
        )

    def execute_scratch_script(self, filename: str, code: str) -> SandboxExecutionResult:
        """Run Python code in scratchpad through AST whitelist gate."""
        start_time = time.monotonic()

        # 1. AST Validation Gate
        validation = validate_scratch_python(code)
        if not validation.is_safe:
            self._append_audit("ast_blocked", {"filename": filename, "violations": list(validation.violations)})
            return SandboxExecutionResult(
                filename=filename,
                is_success=False,
                output="AST Validation Failed:\n" + "\n".join(validation.violations),
                exit_code=1,
                duration_seconds=time.monotonic() - start_time,
                violations=validation.violations,
            )

        # 2. Write script to scratchpad
        safe_name = Path(filename).name
        if not safe_name.endswith(".py"):
            safe_name += ".py"
        target_path = self._scratch_dir / safe_name
        _atomic_write_to(target_path, code)

        # 3. Execute in isolated subprocess
        cmd = [sys.executable, "-I", "-B", str(target_path)]
        env = {
            **os.environ,
            "HERMES_HOME": str(self._root.resolve()),
            "PYTHONIOENCODING": "utf-8",
        }
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
                env=env,
            )
            stdout = proc.stdout[:65536]  # 64 KB limit
            stderr = proc.stderr[:65536]
            output = stdout if proc.returncode == 0 else (stderr or stdout)
            is_success = proc.returncode == 0
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            output = "Execution timed out (10s limit exceeded)"
            is_success = False
            exit_code = 124
        except Exception as exc:
            output = f"Execution error: {exc}"
            is_success = False
            exit_code = 1
        finally:
            # Clean up temporary scratch script
            target_path.unlink(missing_ok=True)

        duration = time.monotonic() - start_time
        self._append_audit("script_executed", {
            "filename": filename,
            "success": is_success,
            "exit_code": exit_code,
            "duration": round(duration, 4),
        })

        return SandboxExecutionResult(
            filename=filename,
            is_success=is_success,
            output=output,
            exit_code=exit_code,
            duration_seconds=duration,
            violations=(),
        )

    def update_memory(self, target: str, markdown_content: str) -> None:
        """Sanitize and write markdown into manager_sandbox/memory/."""
        sanitization = sanitize_memory_markdown(markdown_content)
        if not sanitization.is_safe:
            self._append_audit("memory_rejected", {"target": target, "violations": list(sanitization.violations)})
            raise ValueError(f"Memory update rejected: {', '.join(sanitization.violations)}")

        target_name = Path(target).name
        if not target_name.endswith(".md"):
            target_name += ".md"
        dest = self._memory_dir / target_name
        with FileLock(str(self._lock_path)):
            _atomic_write_to(dest, sanitization.sanitized_content)

        self._append_audit("memory_updated", {
            "target": target_name,
            "bytes": len(sanitization.sanitized_content.encode("utf-8")),
            "redacted_pii": any("PII" in v for v in sanitization.violations),
        })

    def save_skill(self, skill_name: str, skill_content: str) -> bool:
        """Save new procedural skill into quarantine pending safety check."""
        clean_name = re_safe_name = "".join(c for c in skill_name if c.isalnum() or c in ("_", "-")).strip()
        if not clean_name:
            return False

        skill_dir = self._quarantine_dir / clean_name
        skill_dir.mkdir(parents=True, exist_ok=True)
        dest = skill_dir / "SKILL.md"

        with FileLock(str(self._lock_path)):
            _atomic_write_to(dest, skill_content)

        self._append_audit("skill_quarantined", {"skill": clean_name})
        return True

    def promote_quarantined_skill(self, skill_name: str) -> bool:
        """Promote a verified quarantined skill to active status after safety inspection."""
        clean_name = "".join(c for c in skill_name if c.isalnum() or c in ("_", "-")).strip()
        src = self._quarantine_dir / clean_name
        if not src.is_dir():
            return False

        skill_file = src / "SKILL.md"
        if not skill_file.is_file():
            return False

        # Verify skill content safety (Max size 64KB, no prompt injections, and read-only DB invariants)
        try:
            content = skill_file.read_text(encoding="utf-8")
            if len(content.encode("utf-8")) > 65536:
                log.warning("Skill %s exceeds 64KB limit", clean_name)
                return False
            sanitization = sanitize_memory_markdown(content)
            if not sanitization.is_safe:
                log.warning("Skill %s contains unsafe patterns: %s", clean_name, sanitization.violations)
                return False

            # Check for forbidden database mutation / DDL commands
            forbidden_sql_patterns = [
                r"\bDROP\s+TABLE\b",
                r"\bDELETE\s+FROM\b",
                r"\bINSERT\s+INTO\b",
                r"\bUPDATE\s+\w+\s+SET\b",
                r"\.commit\(\)",
                r"\.rollback\(\)",
            ]
            for pat in forbidden_sql_patterns:
                if re.search(pat, content, re.IGNORECASE):
                    log.warning("Skill %s contains forbidden database modification statement: %s", clean_name, pat)
                    return False
        except Exception as exc:
            log.warning("Failed to verify skill %s: %s", clean_name, exc)
            return False

        dest = self._skills_dir / clean_name
        with FileLock(str(self._lock_path)):
            if dest.exists():
                shutil.rmtree(dest)
            shutil.move(str(src), str(dest))

        self._append_audit("skill_promoted", {"skill": clean_name})
        return True

