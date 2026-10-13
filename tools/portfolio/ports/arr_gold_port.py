"""ARR Gold Trading Parser Port (Hexagonal Architecture).

Defines abstract interface for parsing ARR Gold Trading trade confirmations (HTML & PDF)
into canonical domain TradeImportItem records.
"""
from abc import ABC, abstractmethod
from typing import List, Optional
from tools.portfolio.domain.models import TradeImportItem


class ArrGoldParserPort(ABC):
    """Port interface for parsing ARR Gold Trading emails and statements."""

    @abstractmethod
    def parse_trade_html(self, html_content: str, email_subject: str = "") -> List[TradeImportItem]:
        """Parse raw HTML content of an ARR Gold trade execution email into canonical TradeImportItem records."""
        pass

    @abstractmethod
    def parse_daily_statement_pdf(self, pdf_bytes: bytes) -> List[TradeImportItem]:
        """Parse raw PDF bytes of an ARR Gold Daily Report Statement into canonical TradeImportItem records."""
        pass
