"""PortfolioJournalService — Journal Append/Read, String Formatting & Markdown."""
from typing import Optional, List, Dict
from tools.portfolio.domain.models import _now_iso
from tools.portfolio.ports.journal_port import TradeJournalPort


class PortfolioJournalService:
    """Handles Journal note creation, markdown reading, and search."""

    def __init__(self, journal_provider: TradeJournalPort) -> None:
        self.journal_provider = journal_provider

    def append_trading_journal(self, entry: str, portfolio_id: str = "default") -> str:
        try:
            self.journal_provider.append_journal(entry, portfolio_id=portfolio_id)
            return f"[JOURNAL] บันทึกสำเร็จ | [{_now_iso()}] | {len(entry)} chars"
        except Exception as e:
            return f"Error: {e}"

    def read_trading_journal(
        self, days: int = 30, keyword: Optional[str] = None, limit: int = 20, portfolio_id: str = "default"
    ) -> str:
        entries = self.get_structured_journal(days=days, keyword=keyword, limit=limit, portfolio_id=portfolio_id)
        if not entries:
            return "ไม่พบบันทึกการเทรดตามเงื่อนไข"
        lines = []
        for e in entries:
            lines.append(f"## [{e.get('timestamp')}]\n\n{e.get('content')}\n")
        return "\n".join(lines)

    def get_structured_journal(
        self, days: Optional[int] = 365, keyword: Optional[str] = None, limit: int = 100, portfolio_id: str = "default"
    ) -> List[Dict]:
        return self.journal_provider.read_journal(days=days, keyword=keyword, limit=limit, portfolio_id=portfolio_id)

    def structured_append_journal(self, entry: str, portfolio_id: str = "default") -> List[Dict]:
        return self.journal_provider.append_journal(entry, portfolio_id=portfolio_id)
