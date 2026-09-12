import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict

from core.logger import get_logger
from tools._atomic_io import _atomic_write_to
from tools.portfolio.domain.constants import _CASH_SYMBOLS
from tools.portfolio.domain.events import _normalize_journal_timestamp
from tools.portfolio.domain.validator import validate_portfolio_id
from tools.portfolio.ports.journal_port import TradeJournalPort
from .journal_format import inject_journal_links, inject_journal_wikilinks, serialize_journal
from .paths import get_journal_filepath, get_vault_path
from .repository_adapter import _get_portfolio_lock

log = get_logger(__name__)

_TRADE_TITLE_RE = re.compile(r'(\*\*\[[\w\s]+\]\*\*\s+)([A-Z][\w.\-]*)([^\]]*\]\*\*)(?!\s*—\s*\[\[)')
_JOURNAL_BLOCK_RE = re.compile(
    r"^##\s+\[(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s*\n(?P<body>.*?)(?=\n##\s+\[\d{4}-\d{2}-\d{2}|\Z)",
    re.DOTALL | re.MULTILINE,
)


def _inject_journal_wikilinks(content: str) -> str:
    """Backward-compatible wrapper for the portable journal renderer."""
    return inject_journal_links(content, vault_root=get_vault_path())


class JournalVaultAdapter(TradeJournalPort):
    """Obsidian Markdown Vault adapter for Trade Journal."""

    def append_journal(
        self, entry: str, date_str: Optional[str] = None, portfolio_id: str = "default"
    ) -> List[Dict]:
        content = (entry or "").strip()
        if not content:
            raise ValueError("entry ต้องไม่ว่าง")

        pid = validate_portfolio_id(portfolio_id)
        lock = _get_portfolio_lock(pid)
        with lock:
            jpath = get_journal_filepath(pid)
            from tools.archivist.maintenance_guard import assert_write_allowed
            assert_write_allowed(jpath)
            jpath.parent.mkdir(parents=True, exist_ok=True)
            timestamp = _normalize_journal_timestamp(date_str)
            linked = inject_journal_links(
                content,
                vault_root=get_vault_path(),
                source_path=jpath,
            )
            block = f"\n## [{timestamp}]\n\n{linked}\n"
            existing = jpath.read_text(encoding="utf-8") if jpath.exists() else ""
            _atomic_write_to(
                jpath,
                serialize_journal(
                    existing,
                    block,
                    portfolio_id=pid,
                    vault_root=get_vault_path(),
                ),
            )

        return self.read_journal(days=365, limit=100, portfolio_id=pid)

    def append_system_entry(
        self, entry: str, date_str: Optional[str] = None, portfolio_id: str = "default"
    ) -> None:
        self.append_journal(entry, date_str=date_str, portfolio_id=portfolio_id)

    def read_journal(
        self, days: Optional[int] = 365, keyword: Optional[str] = None, limit: int = 100, portfolio_id: str = "default"
    ) -> List[Dict]:
        pid = validate_portfolio_id(portfolio_id)
        lock = _get_portfolio_lock(pid)
        with lock:
            jpath = get_journal_filepath(pid)
            if not jpath.exists():
                return []
            content = jpath.read_text(encoding="utf-8")

        entries: List[Dict] = []
        cutoff = datetime.now() - timedelta(days=days) if days else None
        kw_lower = keyword.strip().lower() if keyword else None

        for match in _JOURNAL_BLOCK_RE.finditer(content):
            ts_str = match.group("ts")
            body = match.group("body").strip()
            try:
                dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue

            if cutoff and dt < cutoff:
                continue
            if kw_lower and kw_lower not in body.lower():
                continue

            entries.append({"timestamp": ts_str, "content": body})

        # Most recent first
        entries.reverse()
        return entries[:limit]
