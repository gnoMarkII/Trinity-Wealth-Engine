"""Application service for managing NousResearch/hermes-agent sync status."""
from __future__ import annotations

import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from core.sandbox.models import HermesSyncStatus

HERMES_DIR = Path("services/hermes_agent")


class HermesSyncService:
    """Manages upstream status and updates for NousResearch/hermes-agent."""

    def __init__(self, hermes_path: Path = HERMES_DIR) -> None:
        self._hermes_path = hermes_path
        self._last_synced_at: Optional[str] = None

    def get_sync_status(self) -> HermesSyncStatus:
        if not (self._hermes_path / ".git").is_dir():
            return HermesSyncStatus(
                is_installed=False,
                current_commit="none",
                upstream_commit="none",
                has_updates=False,
                last_synced_at=None,
            )

        try:
            # Get current HEAD commit short hash
            head_proc = subprocess.run(
                ["git", "-C", str(self._hermes_path), "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            current_commit = head_proc.stdout.strip()
        except Exception:
            current_commit = "unknown"

        return HermesSyncStatus(
            is_installed=True,
            current_commit=current_commit,
            upstream_commit="upstream/main",
            has_updates=False,
            last_synced_at=self._last_synced_at,
        )

    def trigger_sync(self) -> tuple[bool, str]:
        if not self._hermes_path.is_dir():
            return False, "services/hermes_agent directory does not exist"
        try:
            proc = subprocess.run(
                ["git", "-C", str(self._hermes_path), "pull", "--ff-only"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if proc.returncode == 0:
                self._last_synced_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            return proc.returncode == 0, proc.stdout or proc.stderr
        except Exception as exc:
            return False, str(exc)

