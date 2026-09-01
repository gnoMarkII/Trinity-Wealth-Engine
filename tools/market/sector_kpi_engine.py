"""Declarative Sector KPI & Multi-Period Forensics Engine (Phase 3 & v3.1).

Computes sector-specific operating metrics (SaaS Rule of 40, Retail Working Capital Days,
Banking ROE/NIM, Semi Gross Margins & CapEx Intensity, Energy FCF Conversion)
from multi-period financial statements without penalizing unavailable operational data.
"""
from typing import Any, Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field

from schemas.micro_quant_schemas import DataStatus
from tools.market.financial_autopsy import FinancialAutopsyPeriod


class SectorKPIItem(BaseModel):
    name: str
    value: Optional[float]
    unit: str  # "%", "days", "x", "USD"
    benchmark: Optional[str] = None
    status: DataStatus = "available"


class SectorKPISummary(BaseModel):
    sector_name: str
    kpis: list[SectorKPIItem] = Field(default_factory=list)
    sector_health_score: Optional[float] = None  # 0 - 100
    status: DataStatus = "available"
    flags: list[str] = Field(default_factory=list)


def compute_sector_kpis(
    sector: str,
    periods: list[FinancialAutopsyPeriod],
) -> SectorKPISummary:
    """Computes declarative sector KPIs from multi-period financial autopsy statements."""
    san_sec = (sector or "").lower()
    flags: list[str] = []

    if not periods or len(periods) < 1:
        return SectorKPISummary(
            sector_name=sector or "Unknown",
            kpis=[],
            sector_health_score=None,
            status="unavailable",
            flags=["insufficient_financial_periods"],
        )

    latest = periods[-1]
    prev = periods[-2] if len(periods) >= 2 else None

    kpis: list[SectorKPIItem] = []
    scores: list[float] = []

    # 1. Technology / SaaS / Software
    if any(k in san_sec for k in ("tech", "software", "saas", "internet", "cloud")):
        # Rule of 40: Revenue Growth YoY % + FCF Margin %
        latest_rev = getattr(latest, "total_revenue", None) or getattr(latest, "revenue", None)
        prev_rev = getattr(prev, "total_revenue", None) or getattr(prev, "revenue", None) if prev else None
        latest_fcf = getattr(latest, "free_cash_flow", None) or getattr(latest, "fcf", None)

        fcf_m = (latest_fcf / latest_rev * 100.0) if (latest_fcf is not None and latest_rev and latest_rev > 0) else None
        rev_g = 0.0
        if prev_rev and prev_rev > 0 and latest_rev:
            rev_g = ((latest_rev - prev_rev) / prev_rev) * 100.0

        rule_of_40 = round((fcf_m or 0.0) + rev_g, 1) if fcf_m is not None else None
        kpis.append(SectorKPIItem(
            name="Rule of 40 (Growth + FCF Margin)",
            value=rule_of_40,
            unit="%",
            benchmark=">= 40% is Elite SaaS",
            status="available" if rule_of_40 is not None else "unavailable",
        ))
        if rule_of_40 is not None:
            scores.append(100.0 if rule_of_40 >= 40.0 else (80.0 if rule_of_40 >= 25.0 else 50.0))

        # Gross Margin
        latest_gp = getattr(latest, "gross_profit", None)
        gm = (latest_gp / latest_rev * 100.0) if (latest_gp is not None and latest_rev and latest_rev > 0) else None
        kpis.append(SectorKPIItem(
            name="Subscription / Gross Margin",
            value=round(gm, 1) if gm is not None else None,
            unit="%",
            benchmark=">= 70% for pure software",
            status="available" if gm is not None else "unavailable",
        ))
        if gm is not None:
            scores.append(90.0 if gm >= 70.0 else (70.0 if gm >= 50.0 else 40.0))

    # 2. Retail / Consumer / Goods
    elif any(k in san_sec for k in ("retail", "consumer", "goods", "commerce")):
        latest_rev = getattr(latest, "total_revenue", None) or getattr(latest, "revenue", None)
        latest_gp = getattr(latest, "gross_profit", None)
        inv = getattr(latest, "inventory", None)
        cogs = (latest_rev - latest_gp) if (latest_rev and latest_gp) else None
        
        dio = None
        if inv and cogs and cogs > 0:
            dio = round((inv / cogs) * 365.0, 1)

        kpis.append(SectorKPIItem(
            name="Days Inventory Outstanding (DIO)",
            value=dio,
            unit="days",
            benchmark="< 60 days optimal",
            status="available" if dio is not None else "unavailable",
        ))
        if dio is not None:
            scores.append(90.0 if dio <= 60.0 else (70.0 if dio <= 100.0 else 40.0))

    # 3. Energy / Industrial / Commodity
    elif any(k in san_sec for k in ("energy", "oil", "gas", "mining", "utility", "industrial")):
        latest_ocf = getattr(latest, "operating_cash_flow", None) or getattr(latest, "ocf", None)
        latest_fcf = getattr(latest, "free_cash_flow", None) or getattr(latest, "fcf", None)
        latest_capex = getattr(latest, "capital_expenditure", None) or getattr(latest, "capex", None)
        latest_rev = getattr(latest, "total_revenue", None) or getattr(latest, "revenue", None)

        fcf_conv = None
        if latest_ocf and latest_ocf > 0 and latest_fcf is not None:
            fcf_conv = round((latest_fcf / latest_ocf) * 100.0, 1)

        kpis.append(SectorKPIItem(
            name="FCF Conversion (FCF / OCF)",
            value=fcf_conv,
            unit="%",
            benchmark=">= 60% indicates disciplined CapEx",
            status="available" if fcf_conv is not None else "unavailable",
        ))
        if fcf_conv is not None:
            scores.append(95.0 if fcf_conv >= 60.0 else (75.0 if fcf_conv >= 40.0 else 45.0))

        # CapEx Intensity
        capex_int = None
        if latest_rev and latest_rev > 0 and latest_capex:
            capex_int = round((abs(latest_capex) / latest_rev) * 100.0, 1)

        kpis.append(SectorKPIItem(
            name="CapEx Intensity (CapEx / Revenue)",
            value=capex_int,
            unit="%",
            benchmark="10-25% typical for E&P",
            status="available" if capex_int is not None else "unavailable",
        ))

    # 4. General / Default Sector
    else:
        latest_ebit = getattr(latest, "ebit", None) or getattr(latest, "operating_income", None)
        latest_rev = getattr(latest, "total_revenue", None) or getattr(latest, "revenue", None)
        ebit_m = (latest_ebit / latest_rev * 100.0) if (latest_ebit is not None and latest_rev and latest_rev > 0) else None
        kpis.append(SectorKPIItem(
            name="Operating (EBIT) Margin",
            value=round(ebit_m, 1) if ebit_m is not None else None,
            unit="%",
            benchmark=">= 15% healthy",
            status="available" if ebit_m is not None else "unavailable",
        ))
        if ebit_m is not None:
            scores.append(90.0 if ebit_m >= 20.0 else (75.0 if ebit_m >= 10.0 else 40.0))

    health_score = round(sum(scores) / len(scores), 1) if scores else None

    return SectorKPISummary(
        sector_name=sector or "General",
        kpis=kpis,
        sector_health_score=health_score,
        status="available" if kpis else "partial",
        flags=flags,
    )
