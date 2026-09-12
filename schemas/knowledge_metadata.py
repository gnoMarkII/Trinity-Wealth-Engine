"""Pydantic Models for Obsidian Vault V2 Knowledge Metadata.

Defines schemas for common and domain-specific note metadata with strict validation:
- schema_version, note_id, document_key, entity_type, title are universally required.
- Extra/custom fields are preserved (extra='allow').
- Numerical strings like ticker '005930' are strictly preserved as strings.
- Separate source_verification_status vs content_verification_status gates.
- Earnings calls support url and pasted_transcript, quarterly and annual periods.
- Newer schema versions (schema_version > 2) are recognized for read-only handling.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator


VerificationStatus = Literal["verified", "unverified", "not_reviewed", "disputed"]
TrustTier = Literal["T1", "T2", "T3", "TX"]
InputKind = Literal["url", "pasted_transcript"]
PeriodType = Literal["quarter", "annual", "other"]

CURRENT_SCHEMA_VERSION = 2


class CommonNoteMetadata(BaseModel):
    """Core metadata schema required for all Obsidian Vault V2 notes."""
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    schema_version: int = Field(default=CURRENT_SCHEMA_VERSION, description="Schema revision number")
    note_id: str = Field(..., description="Unique immutable note identifier")
    document_key: str = Field(..., description="Deterministic logical document identity")
    entity_type: str = Field(..., description="Canonical entity type name")
    title: str = Field(..., description="Human-readable title")
    date: Optional[str] = Field(default=None, description="Primary date (YYYY-MM-DD)")
    tags: list[str] = Field(default_factory=list, description="Categorization tags")
    aliases: list[str] = Field(default_factory=list, description="Alternative names or tickers")
    revision: int = Field(default=1, description="Sequential human revision counter")
    entity_id: Optional[str] = Field(default=None, description="Stable identifier for financial asset or entity")

    # Cross-application governance fields.  They are intentionally plain
    # portable scalars so Obsidian, API workers, scripts, and other clients can
    # round-trip them without depending on a plugin-specific vocabulary.
    document_role: Optional[str] = Field(default=None, description="knowledge | navigation | projection | capture")
    search_scope: str = Field(default="included", description="included | excluded")
    content_status: str = Field(default="published", description="draft | reviewed | published | superseded | retired | generated")
    sensitivity: str = Field(default="internal", description="public | internal | confidential | restricted")
    retention_class: Optional[str] = Field(default=None, description="permanent | long_term | operational | ephemeral")
    verification_status: Optional[str] = Field(default=None, description="Compatibility provenance gate for indexing policy")
    
    # Distinct verification gates
    source_verification_status: VerificationStatus = Field(
        default="not_reviewed",
        description="Whether source authenticity and origin are verified",
    )
    content_verification_status: VerificationStatus = Field(
        default="not_reviewed",
        description="Whether content accuracy/fact-checking has been reviewed",
    )
    verification_method: str = Field(
        default="not_reviewed",
        description="Auditable method or workflow that produced verification status",
    )
    verified_at: Optional[str] = Field(
        default=None,
        description="UTC timestamp of the latest completed verification",
    )
    evidence_refs: list[str] = Field(
        default_factory=list,
        description="Stable evidence paths, URLs, artifact IDs, or digests",
    )
    trust_tier: TrustTier = Field(
        default="T3",
        description="T1 source-backed, T2 derived with evidence, T3 unreviewed, TX unavailable",
    )
    production_eligible: bool = Field(
        default=False,
        description="Explicit gate for AI answers and downstream production decisions",
    )
    source_unavailable_reason: Optional[str] = Field(
        default=None,
        description="Why a source could not be verified or recovered",
    )

    @field_validator("date", mode="before")
    @classmethod
    def _normalize_date(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        if isinstance(v, (datetime, date)):
            return v.strftime("%Y-%m-%d")
        return str(v).strip()

    @field_validator("tags", mode="before")
    @classmethod
    def _normalize_tags(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            return [t.strip() for t in v.split(",") if t.strip()]
        return [str(t).strip() for t in v if str(t).strip()]


class StockHubMetadata(CommonNoteMetadata):
    entity_type: Literal["stock_hub"] = "stock_hub"
    ticker: str = Field(..., description="Ticker symbol strictly preserved as text")
    company_name: Optional[str] = None
    market: Optional[str] = None
    sector: Optional[str] = None

    @field_validator("ticker", mode="before")
    @classmethod
    def _preserve_ticker_str(cls, v: Any) -> str:
        return str(v).strip().upper()


class EquityAnalysisMetadata(CommonNoteMetadata):
    entity_type: Literal["equity_analysis"] = "equity_analysis"
    ticker: str
    composite_score: Optional[float] = None
    valuation_status: Optional[str] = None
    analyst: Optional[str] = None

    @field_validator("ticker", mode="before")
    @classmethod
    def _preserve_ticker_str(cls, v: Any) -> str:
        return str(v).strip().upper()


class QuantSnapshotMetadata(CommonNoteMetadata):
    entity_type: Literal["quant_snapshot"] = "quant_snapshot"
    ticker: str
    composite_score: Optional[float] = None
    as_of: Optional[str] = None

    @field_validator("ticker", mode="before")
    @classmethod
    def _preserve_ticker_str(cls, v: Any) -> str:
        return str(v).strip().upper()


class EarningsCallMetadata(CommonNoteMetadata):
    entity_type: Literal["earnings_call"] = "earnings_call"
    ticker: str
    period: str = Field(..., description="Reporting period string, e.g. 2026-Q1 or 2025-FY")
    period_type: PeriodType = Field(default="quarter", description="quarter | annual | other")
    fiscal_quarter: Optional[int] = Field(default=None, description="1..4 for quarterly calls")
    fiscal_year: Optional[int] = Field(default=None, description="Fiscal year e.g. 2026")
    input_kind: InputKind = Field(default="url", description="url or pasted_transcript")
    source_url: Optional[str] = Field(default=None, description="Source URL if input_kind == 'url'")
    transcript_hash: Optional[str] = Field(default=None, description="SHA-256 of transcript artifact")
    call_date: Optional[str] = Field(default=None, description="Date call took place")

    @field_validator("ticker", mode="before")
    @classmethod
    def _preserve_ticker_str(cls, v: Any) -> str:
        return str(v).strip().upper()


class CompanyNewsMetadata(CommonNoteMetadata):
    entity_type: Literal["company_news"] = "company_news"
    tickers: list[str] = Field(default_factory=list)
    published_date: Optional[str] = None
    source_url: Optional[str] = None
    publisher: Optional[str] = None
    authors: list[str] = Field(default_factory=list)


class YouTubeSummaryMetadata(CommonNoteMetadata):
    entity_type: Literal["youtube_summary"] = "youtube_summary"
    video_id: str
    source_url: Optional[str] = None
    channel: Optional[str] = None
    published_date: Optional[str] = None


class MacroStrategyMetadata(CommonNoteMetadata):
    entity_type: Literal["macro_strategy"] = "macro_strategy"
    as_of: Optional[str] = None
    horizon: Optional[str] = None
    direction: Optional[str] = None


class MacroSnapshotMetadata(CommonNoteMetadata):
    entity_type: Literal["macro_snapshot"] = "macro_snapshot"
    as_of: Optional[str] = None


class BriefingBookMetadata(CommonNoteMetadata):
    entity_type: Literal["briefing_book"] = "briefing_book"
    authored_date: Optional[str] = None
    source_count: Optional[int] = None


class BookNoteMetadata(CommonNoteMetadata):
    entity_type: Literal["book_note"] = "book_note"
    author: Optional[str] = None
    genre: Optional[str] = None
    date_read: Optional[str] = None
    rating: Optional[Union[int, float, str]] = None


class GenericKnowledgeMetadata(CommonNoteMetadata):
    """Fallback model for user concepts, books, and other arbitrary notes."""
    entity_type: str = "concept"
