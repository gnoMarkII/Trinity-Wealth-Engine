"""Thai Hard Data Feasibility & Status Adapter (Dual-Track Macro Intelligence - Phase 4).

Tracks feasibility, status, and verified release sources for Thai official macroeconomic hard data:
- Real GDP: Office of the National Economic and Social Development Council (NESDC / สภาพัฒน์)
- CPI / Inflation: Trade Policy and Strategy Office, Ministry of Commerce (TPSO / MOC / กระทรวงพาณิชย์)
- Policy Interest Rate: Monetary Policy Committee, Bank of Thailand (BOT / ธปท.)

Fail-Closed Policy:
When automated official feeds are unavailable, the adapter emits structured data gaps
rather than generating synthetic or mock numbers.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional
from schemas.macro_schemas import MarketObservable


@dataclass(frozen=True)
class ThaiHardDataRecord:
    series_id: str
    indicator_name: str
    source_authority: str
    frequency: str
    unit: str
    value: Optional[float] = None
    prev: Optional[float] = None
    ma: Optional[float] = None
    period: Optional[str] = None
    observed_at: Optional[str] = None
    published_at: Optional[str] = None
    is_verified: bool = False
    status: str = "missing"
    gap_reason: str = ""


class ThaiHardDataAdapter:
    """Provides status, metadata, and structured data gaps for Thai macroeconomic releases."""

    def __init__(self, override_records: Optional[dict[str, ThaiHardDataRecord]] = None):
        self._records = override_records or {}

    def get_thai_gdp_status(self) -> ThaiHardDataRecord:
        if "TH_REAL_GDP" in self._records:
            return self._records["TH_REAL_GDP"]
        return ThaiHardDataRecord(
            series_id="TH_REAL_GDP",
            indicator_name="Thailand Real GDP YoY",
            source_authority="Office of the National Economic and Social Development Council (NESDC / สภาพัฒน์)",
            frequency="Quarterly",
            unit="% YoY",
            value=None,
            is_verified=False,
            status="missing",
            gap_reason="Direct NESDC automated API integration not yet active; synthetic mocks disabled by production guardrail.",
        )

    def get_thai_cpi_status(self) -> ThaiHardDataRecord:
        if "TH_CPI_YOY" in self._records:
            return self._records["TH_CPI_YOY"]
        return ThaiHardDataRecord(
            series_id="TH_CPI_YOY",
            indicator_name="Thailand CPI Inflation YoY",
            source_authority="Trade Policy and Strategy Office, Ministry of Commerce (TPSO / MOC / กระทรวงพาณิชย์)",
            frequency="Monthly",
            unit="% YoY",
            value=None,
            is_verified=False,
            status="missing",
            gap_reason="Direct MOC/TPSO automated API integration not yet active; synthetic mocks disabled by production guardrail.",
        )

    def get_hard_data_gaps(self) -> list[str]:
        gaps = []
        gdp = self.get_thai_gdp_status()
        if not gdp.is_verified or gdp.value is None:
            gaps.append(f"{gdp.indicator_name} ({gdp.source_authority})")
        cpi = self.get_thai_cpi_status()
        if not cpi.is_verified or cpi.value is None:
            gaps.append(f"{cpi.indicator_name} ({cpi.source_authority})")
        return gaps

    def as_observables(self, as_of_date: Optional[str] = None) -> list[MarketObservable]:
        today_str = as_of_date or datetime.now().strftime("%Y-%m-%d")
        observables: list[MarketObservable] = []

        gdp = self.get_thai_gdp_status()
        gdp_meta: dict[str, Any] = {}
        if gdp.value is not None:
            gdp_meta["val"] = gdp.value
        if gdp.prev is not None:
            gdp_meta["prev"] = gdp.prev
        if gdp.ma is not None:
            gdp_meta["ma"] = gdp.ma

        observables.append(MarketObservable(
            observable_id="obs_th_gdp_nesdc",
            asset_bucket="equities",
            region="Thailand",
            indicator=gdp.indicator_name,
            value=f"{gdp.value:.2f}" if gdp.value is not None else "N/A",
            unit=gdp.unit,
            observed_at=gdp.observed_at or today_str,
            published_at=gdp.published_at,
            source_file="NESDC_Official_Releases",
            provider="NESDC",
            confidence="high" if gdp.is_verified else "low",
            is_valid=gdp.is_verified and gdp.value is not None,
            status="verified" if (gdp.is_verified and gdp.value is not None) else "missing",
            stale_reason="" if gdp.is_verified else gdp.gap_reason,
            period=gdp.period,
            metadata=gdp_meta,
        ))

        cpi = self.get_thai_cpi_status()
        cpi_meta: dict[str, Any] = {}
        if cpi.value is not None:
            cpi_meta["val"] = cpi.value
        if cpi.prev is not None:
            cpi_meta["prev"] = cpi.prev
        if cpi.ma is not None:
            cpi_meta["ma"] = cpi.ma

        observables.append(MarketObservable(
            observable_id="obs_th_cpi_moc",
            asset_bucket="cash",
            region="Thailand",
            indicator=cpi.indicator_name,
            value=f"{cpi.value:.2f}" if cpi.value is not None else "N/A",
            unit=cpi.unit,
            observed_at=cpi.observed_at or today_str,
            published_at=cpi.published_at,
            source_file="MOC_Official_Releases",
            provider="MOC TPSO",
            confidence="high" if cpi.is_verified else "low",
            is_valid=cpi.is_verified and cpi.value is not None,
            status="verified" if (cpi.is_verified and cpi.value is not None) else "missing",
            stale_reason="" if cpi.is_verified else cpi.gap_reason,
            period=cpi.period,
            metadata=cpi_meta,
        ))

        return observables
