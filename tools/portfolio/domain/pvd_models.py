"""Typed PVD statement contracts. No storage, API, or PDF concerns live here."""
from datetime import date
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PvdSubFundHolding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    name: str
    units: Decimal = Field(ge=0)
    statement_nav: Decimal = Field(ge=0)
    statement_market_value: Decimal = Field(ge=0)
    weight_percent: Decimal = Field(ge=0)
    reconciliation_adjustment: Decimal = Decimal("0.00")
    allocated_cost: Decimal = Decimal("0.00")


class PvdContributionSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    member_principal: Decimal = Field(ge=0)
    member_benefit: Decimal
    employer_principal: Decimal = Field(ge=0)
    employer_benefit: Decimal
    total_contribution: Decimal = Field(ge=0)
    total_nav: Decimal = Field(ge=0)
    employer_vesting_percent: Decimal = Field(ge=0, le=100)
    member_vested_amount: Decimal = Field(ge=0)
    employer_vested_amount: Decimal = Field(ge=0)


class PvdStatement(BaseModel):
    """Parsed statement; raw member ID is transient and never persisted."""
    model_config = ConfigDict(extra="forbid")

    fund_code: str = "BBLAM"
    member_id: str
    member_hash: str
    masked_member_id: str = ""
    as_of_date: date
    period_start: Optional[date] = None
    period_end: Optional[date] = None
    summary: PvdContributionSummary
    holdings: list[PvdSubFundHolding]
    source_pdf_sha256: str = ""


class PvdHoldingValuation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    payload_version: int = 1
    fund_code: str
    sub_fund_code: str
    units: Decimal = Field(ge=0)
    statement_nav: Decimal = Field(ge=0)
    statement_market_value: Decimal = Field(ge=0)
    reconciliation_adjustment: Decimal = Decimal("0.00")
    cost_basis: Decimal = Field(ge=0)
    as_of_date: date
    statement_fingerprint: str
    archive_id: str


class PvdStatementMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    payload_version: int = 1
    fund_code: str
    member_hash: str
    masked_member_id: str = ""
    as_of_date: date
    period_start: Optional[date] = None
    period_end: Optional[date] = None
    total_contribution: Decimal
    total_nav: Decimal
    member_principal: Decimal
    member_benefit: Decimal
    employer_principal: Decimal
    employer_benefit: Decimal
    employer_vesting_percent: Decimal
    member_vested_amount: Decimal
    employer_vested_amount: Decimal
    source_pdf_sha256: str
    semantic_fingerprint: str
    archive_id: str
    reconciliation_adjustment: Decimal = Decimal("0.00")

    @field_validator("as_of_date", mode="before")
    @classmethod
    def parse_as_of_date(cls, value):
        return date.fromisoformat(value) if isinstance(value, str) else value


class PvdSnapshotReplayResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    as_of_date: date
    statement_fingerprint: str
    active_rows: list[dict]
    closed_symbols: list[str] = Field(default_factory=list)
    latest_metadata: PvdStatementMetadata

