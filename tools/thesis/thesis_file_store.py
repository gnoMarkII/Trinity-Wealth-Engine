"""Append-Only File Store & Obsidian Markdown Mirror for Theses.

Architecture Rule (Layer 3 - Tools & Adapters):
Uses _atomic_io.py for safe atomic writes and maintains SHA-256 verification.
Syncs human-readable notes with YAML frontmatter to Obsidian Vault V2.
"""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from filelock import FileLock

from core.security import validate_ticker
from core.thesis.evidence_taxonomy import EvidenceTier
from core.thesis.models import (
    CatalystEvent,
    FalsifierRule,
    ThesisClaimRecord,
)
from tools._atomic_io import _atomic_write_to

log = logging.getLogger(__name__)


def _get_default_data_dir() -> Path:
    return Path(os.environ.get("THESIS_DATA_DIR", "data/theses"))


def _get_default_vault_dir() -> Path:
    return Path(os.environ.get("THESIS_VAULT_DIR", "memories/20_Portfolio_Management/theses"))


class ThesisFileStore:
    """Manages append-only JSON files and Obsidian Markdown mirror for stock theses."""

    def __init__(
        self,
        data_dir: Optional[Path | str] = None,
        vault_dir: Optional[Path | str] = None,
    ) -> None:
        self.data_dir = (Path(data_dir) if data_dir is not None else _get_default_data_dir()).resolve()
        self.vault_dir = (Path(vault_dir) if vault_dir is not None else _get_default_vault_dir()).resolve()

    def _get_ticker_json_path(self, ticker: str) -> Path:
        clean = validate_ticker(ticker)
        path = (self.data_dir / f"{clean}_theses.json").resolve()
        if not str(path).startswith(str(self.data_dir)):
            raise ValueError(f"Path traversal detected: {path} is outside data_dir {self.data_dir}")
        return path

    def _get_vault_md_path(self, ticker: str) -> Path:
        clean = validate_ticker(ticker)
        path = (self.vault_dir / f"{clean}.md").resolve()
        if not str(path).startswith(str(self.vault_dir)):
            raise ValueError(f"Path traversal detected: {path} is outside vault_dir {self.vault_dir}")
        return path

    def _get_lock_path(self, ticker: str) -> Path:
        clean = validate_ticker(ticker)
        return self.data_dir / f".{clean}_theses.lock"

    def save_claim(self, claim: ThesisClaimRecord) -> None:
        """Appends a new thesis claim to the ticker's history and updates the Obsidian mirror under process lock."""
        clean_ticker = validate_ticker(claim.ticker)
        if not claim.verify_integrity():
            raise ValueError(f"Cannot save corrupted or unverified claim: {claim.claim_id}")

        self.data_dir.mkdir(parents=True, exist_ok=True)
        lock_file = self._get_lock_path(clean_ticker)
        lock = FileLock(str(lock_file), timeout=15)

        with lock:
            json_path = self._get_ticker_json_path(clean_ticker)
            claims = self.load_claims(clean_ticker)

            # Check for duplicate claim_id
            if any(c.claim_id == claim.claim_id for c in claims):
                raise ValueError(f"Claim ID already exists in append-only store: {claim.claim_id}")

            # Verify optimistic parent version linkage if prior_claim_id specified
            if claim.prior_claim_id:
                prior = next((c for c in claims if c.claim_id == claim.prior_claim_id), None)
                if not prior:
                    raise ValueError(f"Referenced prior claim {claim.prior_claim_id} not found in history for {clean_ticker}")
                if prior.ticker != clean_ticker:
                    raise ValueError(f"Cross-ticker parent reference detected: {prior.ticker} != {clean_ticker}")
                latest_active = next((c for c in reversed(claims) if c.status.upper() == "ACTIVE"), None)
                if latest_active and latest_active.claim_id != claim.prior_claim_id:
                    raise ValueError(
                        f"Stale parent conflict: prior claim {claim.prior_claim_id} is superseded by latest active claim {latest_active.claim_id}"
                    )
            elif len(claims) > 0 and any(c.status.upper() == "ACTIVE" for c in claims):
                raise ValueError(
                    f"Initial create rejected: active thesis already exists for {clean_ticker}. Must specify prior_claim_id to revise."
                )

            claims.append(claim)
            serialized = [c.to_dict() for c in claims]
            json_content = json.dumps(serialized, indent=2, ensure_ascii=False)
            _atomic_write_to(json_path, json_content)

            # Sync active version to Obsidian Vault inside lock to avoid race conditions
            self.sync_to_obsidian(claim)

    def load_claims(self, ticker: str) -> List[ThesisClaimRecord]:
        """Loads and verifies all historical claims for a specific ticker."""
        json_path = self._get_ticker_json_path(ticker)
        if not json_path.exists():
            return []

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if not isinstance(raw, list):
                raise ValueError(f"Corrupted thesis file structure (expected list): {json_path}")
            claims: List[ThesisClaimRecord] = []
            has_corrupt_claim = False
            for item in raw:
                c = ThesisClaimRecord.from_dict(item)
                if not c.verify_integrity():
                    log.error("Integrity check failed for claim %s in %s", c.claim_id, json_path)
                    has_corrupt_claim = True
                else:
                    claims.append(c)
            if has_corrupt_claim:
                ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                corrupt_backup = json_path.with_suffix(f".corrupt_{ts}.json")
                try:
                    import shutil
                    shutil.copy2(json_path, corrupt_backup)
                    log.critical("Thesis with invalid checksum for %s quarantined to %s", ticker, corrupt_backup)
                except Exception as copy_err:
                    log.error("Failed to copy corrupt thesis file: %s", copy_err)
                raise ValueError(f"Integrity check failed: corrupted thesis record detected for {ticker}. History preserved in quarantine.")
            return claims
        except json.JSONDecodeError as exc:
            # Quarantine corrupted file to preserve evidence rather than quietly returning [] and overwriting!
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            corrupt_backup = json_path.with_suffix(f".corrupt_{ts}.json")
            try:
                import shutil
                shutil.copy2(json_path, corrupt_backup)
                log.critical("Corrupted thesis JSON file for %s quarantined to %s: %s", ticker, corrupt_backup, exc)
            except Exception as copy_err:
                log.error("Failed to copy corrupt thesis file: %s", copy_err)
            raise ValueError(f"Corrupted thesis store detected for {ticker}. History preserved in quarantine.") from exc

    def get_active_claim(self, ticker: str) -> Optional[ThesisClaimRecord]:
        """Returns the latest ACTIVE, verified claim for the ticker."""
        claims = self.load_claims(ticker)
        active = [c for c in claims if c.status.upper() == "ACTIVE" and c.verify_integrity()]
        return active[-1] if active else None

    def list_all_active_claims(self) -> List[ThesisClaimRecord]:
        """Scans data_dir for all active claims across all tickers."""
        if not self.data_dir.exists():
            return []
        active_list: List[ThesisClaimRecord] = []
        for file in self.data_dir.glob("*_theses.json"):
            ticker = file.stem.replace("_theses", "")
            act = self.get_active_claim(ticker)
            if act:
                active_list.append(act)
        return active_list

    def sync_to_obsidian(self, claim: ThesisClaimRecord) -> None:
        """Generates an Obsidian Markdown mirror with YAML frontmatter."""
        md_path = self._get_vault_md_path(claim.ticker)

        # Build YAML frontmatter
        lines = [
            "---",
            f"ticker: {claim.ticker}",
            f"claim_id: {claim.claim_id}",
            f"version: {claim.version}",
            f"status: {claim.status}",
            f"as_of_utc: '{claim.as_of_utc}'",
            f"evidence_tier: {claim.evidence_tier.value}",
            f"author_role: {claim.author_role}",
            f"checksum_sha256: '{claim.checksum_sha256}'",
            "tags:",
            "  - investment-thesis",
            "  - institutional-research",
            "---",
            "",
            f"# 🏛️ Investment Thesis: {claim.ticker} (v{claim.version})",
            "",
            f"> **As Of:** `{claim.as_of_utc}` | **Evidence:** `{claim.evidence_tier.value}` | **Status:** `{claim.status}`",
            "",
            "## 📌 Core Thesis Claim",
            f"{claim.claim_statement}",
            "",
            "## 🛑 Falsifier Floor Rules",
        ]

        if claim.falsifiers:
            for idx, f in enumerate(claim.falsifiers, 1):
                lines.extend([
                    f"### {idx}. {f.metric_name} ({f.invalidation_impact})",
                    f"- **Falsifier ID:** `{f.falsifier_id}`",
                    f"- **Condition:** `{f.operator} {f.threshold_value} {f.unit}` ({f.period_basis})",
                    f"- **Narrative:** {f.narrative_explanation}",
                    "",
                ])
        else:
            lines.append("*No explicit quantitative falsifiers specified.*")
            lines.append("")

        lines.append("## 🗓️ Catalyst Milestones")
        if claim.linked_catalysts:
            for idx, c in enumerate(claim.linked_catalysts, 1):
                lines.extend([
                    f"### {idx}. {c.event_name} (`{c.status}`)",
                    f"- **Expected Date:** `{c.expected_date_str}` (Window: $\\pm${c.confidence_window_days} days)",
                    f"- **Source:** {c.source_locator or 'N/A'}",
                    "",
                ])
        else:
            lines.append("*No catalysts linked.*")
            lines.append("")

        content = "\n".join(lines)
        try:
            _atomic_write_to(md_path, content)
        except Exception as exc:
            log.warning("Could not sync thesis markdown to Obsidian vault: %s", exc)

