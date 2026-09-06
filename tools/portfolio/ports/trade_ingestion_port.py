"""Trade Ingestion Ports (Hexagonal Architecture).

Defines abstract interfaces for:
1. TradeEmailSourcePort: Fetching confirmation emails & raw PDF attachments.
2. TradeDocumentParserPort: Isolated parsing of PDF bytes into canonical TradeImportItem.
3. TradeStagingPort: Temporary session-bound staging of parsed items before user review/commit.
"""
from abc import ABC, abstractmethod
from typing import Optional, List
from pydantic import BaseModel, ConfigDict

from tools.portfolio.domain.models import TradeImportItem


class TradeDocumentMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    message_id: str
    attachment_id: str
    subject: str
    sender: str
    received_at: str
    filename: str
    size_bytes: int
    x_gm_msgid: str = ""
    uid: str = ""
    account_email: str = ""


class TradeEmailSourcePort(ABC):
    """Port for fetching trade confirmation documents from email (e.g. Gmail IMAP)."""

    @abstractmethod
    def search_dime_emails(self, query: str = "", limit: int = 20) -> List[TradeDocumentMetadata]:
        """Search email inbox for Dime confirmation emails."""
        pass

    @abstractmethod
    def search_wealthx_emails(self, query: str = "", limit: Optional[int] = None) -> List[TradeDocumentMetadata]:
        """Search email inbox for WealthX confirmation emails (Trade Confirmations only)."""
        pass

    @abstractmethod
    def search_scbam_emails(self, query: str = "", limit: Optional[int] = None) -> List[TradeDocumentMetadata]:
        """Search email inbox for SCBAM Fund Click confirmation emails."""
        pass

    @abstractmethod
    def fetch_pdf_attachment(self, message_id: str, attachment_id: str) -> bytes:
        """Fetch raw bytes of a PDF attachment given message and attachment ID."""
        pass

    @abstractmethod
    def fetch_email_html_body(self, message_id: str) -> str:
        """Fetch raw HTML body of an email given message ID."""
        pass


class TradeDocumentParserPort(ABC):
    """Port for parsing confirmation PDF documents into domain TradeImportItem records."""

    @abstractmethod
    def parse_confirmation_pdf(self, pdf_bytes: bytes, password: Optional[str] = None) -> List[TradeImportItem]:
        """Parse raw PDF bytes into a list of canonical TradeImportItem."""
        pass


class TradeStagingPort(ABC):
    """Port for staging parsed trade import items in-memory prior to commitment."""

    @abstractmethod
    def stage_items(
        self,
        items: List[TradeImportItem],
        ttl_seconds: int = 1800,
        session_id: Optional[str] = None,
    ) -> str:
        """Stage parsed items and return a unique scan_id."""
        pass

    @abstractmethod
    def get_staged_items(
        self,
        scan_id: str,
        session_id: Optional[str] = None,
    ) -> List[TradeImportItem]:
        """Retrieve staged items for a given scan_id and session_id."""
        pass

    @abstractmethod
    def delete_staged(
        self,
        scan_id: str,
        session_id: Optional[str] = None,
    ) -> None:
        """Delete staged items for a given scan_id and session_id."""
        pass
