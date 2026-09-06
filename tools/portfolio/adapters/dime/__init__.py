"""Dime Confirmation Ingestion Adapters."""
from .inmemory_staging_adapter import InMemoryStagingAdapter
from .isolated_parser_adapter import IsolatedDimePdfParserAdapter
from .gmail_imap_adapter import GmailImapSourceAdapter

__all__ = [
    "InMemoryStagingAdapter",
    "IsolatedDimePdfParserAdapter",
    "GmailImapSourceAdapter",
]
