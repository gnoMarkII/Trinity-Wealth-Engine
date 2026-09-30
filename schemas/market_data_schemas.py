"""Market Data Schemas

Defines contracts for macroeconomic observations and financial market data,
ensuring central source of truth for downstream analytics and validators.
"""
from typing import Dict, Literal, Optional
from pydantic import BaseModel, Field


class MacroObservation(BaseModel):
    series_id: str = Field(description="Ticker or FRED Series ID e.g. BZ=F, CPIAUCSL")
    category: Literal[
        "inflation",
        "rates",
        "energy",
        "equity",
        "sector",
        "commodity",
        "fx",
        "yield",
        "other",
    ] = Field(description="หมวดหมู่ของดัชนีสำหรับใช้อ้างอิงใน Validator อย่างรัดกุม")
    label: str = Field(description="ชื่อภาษาไทย/อังกฤษของดัชนี")
    value: float = Field(description="ค่าตัวเลขล่าสุด")
    unit: str = Field(description="หน่วยของดัชนี e.g. USD/bbl, %, Points")
    observed_at: str = Field(description="วันที่และเวลาของค่าล่าสุด (ISO format)")
    provider: str = Field(description="ผู้ให้บริการข้อมูล e.g. Yahoo Finance, FRED")
    source_url: str | None = Field(default=None, description="URL อ้างอิง")
    previous_value: Optional[float] = Field(default=None, description="ค่าก่อนหน้า")
    change_pct: Optional[float] = Field(default=None, description="% เปลี่ยนแปลงล่าสุด")
    returns: Dict[str, Optional[float]] = Field(default_factory=dict, description="ผลตอบแทน 1D, 5D, 30D, 90D (%)")
    is_stale: bool = Field(default=False, description="ข้อมูลเก่าเกิน 5 วันทำการหรือไม่")
    confidence: Literal["high", "medium", "low"] = Field(default="high")
    frequency: Optional[str] = Field(default=None, description="Provider reporting frequency")
    provider_updated_at: Optional[str] = Field(
        default=None,
        description="Provider publication/update time, distinct from the economic observation period",
    )
    freshness_reason: str = Field(default="", description="Auditable freshness decision")
    region: str = Field(default="United States", description="Geographic region e.g. United States, Thailand, Global")
    raw_unit: Optional[str] = Field(default=None, description="Original raw unit from provider before conversion")
    normalized_unit: Optional[str] = Field(default=None, description="Canonical normalized unit e.g. %, % YoY, bps, USD")
    transform: Optional[str] = Field(default=None, description="Transform applied e.g. pc1, diff_bps, ratio, none")
    period: Optional[str] = Field(default=None, description="Economic observation period e.g. 2026-08, 2026-Q2")
    fetched_at: Optional[str] = Field(default=None, description="Timestamp when observation was retrieved by system")
    source_file: Optional[str] = Field(default=None, description="Source provenance file or endpoint identifier")
    revision: Optional[int] = Field(default=None, description="Revision or announcement number if applicable")
    status: Literal["verified", "stale", "unverified", "mock", "missing"] = Field(
        default="verified",
        description="Data verification status. Only 'verified' is admissible into scoring."
    )
    reason: str = Field(default="", description="Exclusion or downgrade reason if status is not verified")
    input_ids: list[str] = Field(default_factory=list, description="IDs of underlying raw observables for derived metrics")
