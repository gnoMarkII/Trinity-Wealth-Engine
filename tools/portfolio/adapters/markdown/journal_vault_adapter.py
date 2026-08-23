import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict

from core.logger import get_logger
from tools._atomic_io import _atomic_write_to
from tools.portfolio.domain.constants import _CASH_SYMBOLS
from tools.portfolio.domain.validator import validate_portfolio_id
from tools.portfolio.ports.journal_port import TradeJournalPort
from .paths import get_journal_filepath
from .repository_adapter import _get_portfolio_lock

log = get_logger(__name__)

_TRADE_TITLE_RE = re.compile(r'(\*\*\[[\w\s]+\]\*\*\s+)([A-Z][\w.\-]*)([^\]]*\]\*\*)(?!\s*—\s*\[\[)')
_JOURNAL_BLOCK_RE = re.compile(
    r"^##\s+\[(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s*\n(?P<body>.*?)(?=\n##\s+\[\d{4}-\d{2}-\d{2}|\Z)",
    re.DOTALL | re.MULTILINE,
)


def _inject_journal_wikilinks(content: str) -> str:
    """Inject wikilinks to holding symbol notes."""
    def _replace(m: re.Match) -> str:
        symbol = m.group(2)
        if symbol in _CASH_SYMBOLS:
            return m.group(0)
        return f"{m.group(1)}{symbol}{m.group(3)} — [[{symbol}]]"
    return _TRADE_TITLE_RE.sub(_replace, content)


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
            jpath.parent.mkdir(parents=True, exist_ok=True)
            if date_str:
                if len(date_str) == 10:
                    timestamp = f"{date_str} 12:00:00"
                else:
                    timestamp = date_str
            else:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            linked = _inject_journal_wikilinks(content)
            block = f"\n## [{timestamp}]\n\n{linked}\n"
            existing = jpath.read_text(encoding="utf-8") if jpath.exists() else ""
            _atomic_write_to(jpath, existing + block)

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
