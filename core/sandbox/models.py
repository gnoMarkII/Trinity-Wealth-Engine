"""Pure domain models for Manager Chat and Hermes Sandbox."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class AdvisorChatMessage:
    id: str
    session_id: str
    role: str  # "user" | "assistant" | "system"
    content: str
    citations: tuple[str, ...] = field(default_factory=tuple)
    created_at: str = ""


@dataclass(frozen=True)
class AdvisorSessionSummary:
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int = 0


@dataclass(frozen=True)
class SkillManifest:
    name: str
    description: str
    is_quarantined: bool
    path: str
    created_at: str = ""


@dataclass(frozen=True)
class SandboxExecutionResult:
    filename: str
    is_success: bool
    output: str
    exit_code: int
    duration_seconds: float
    violations: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class SandboxStateSnapshot:
    memory_files: tuple[str, ...]
    active_skills: tuple[SkillManifest, ...]
    quarantined_skills: tuple[SkillManifest, ...]
    scratch_files_count: int
    last_audit_entry: Optional[str] = None


@dataclass(frozen=True)
class HermesSyncStatus:
    is_installed: bool
    current_commit: str
    upstream_commit: str
    has_updates: bool
    last_synced_at: Optional[str] = None

