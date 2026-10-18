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

    # Ensure periods are ordered newest to oldest
    has_dates = all(getattr(p, "fiscal_period_end", None) for p in periods)
    if has_dates:
        sorted_periods = sorted(periods, key=lambda p: str(getattr(p, "fiscal_period_end")), reverse=True)
    else:
        # If periods[0] has data or no dates, treat periods as either newest-first or oldest-first
        # In financial_autopsy.py, periods[0] is typically the latest period.
        sorted_periods = list(periods)

    latest = sorted_periods[0]
    prev = sorted_periods[1] if len(sorted_periods) >= 2 else None

    kpis: list[SectorKPIItem] = []
    scores: list[float] = []

    # 1. Banking / Financial Services / Insurance
    if any(k in san_sec for k in ("financial", "bank", "insurance")):
        latest_net_income = getattr(latest, "net_income", None)
        latest_equity = getattr(latest, "stockholders_equity", None) or getattr(latest, "total_equity", None)
        latest_rev = getattr(latest, "total_revenue", None) or getattr(latest, "revenue", None)
        latest_assets = getattr(latest, "total_assets", None)

        # ROE %
        roe = (latest_net_income / latest_equity * 100.0) if (latest_net_income is not None and latest_equity and latest_equity > 0) else None
        kpis.append(SectorKPIItem(
            name="Return on Equity (ROE %)",
            value=round(roe, 1) if roe is not None else None,
            unit="%",
            benchmark=">= 12% healthy bank ROE",
            status="available" if roe is not None else "unavailable",
        ))
        if roe is not None:
            scores.append(95.0 if roe >= 15.0 else (75.0 if roe >= 10.0 else 45.0))

        # Net Profit Margin %
        net_margin = (latest_net_income / latest_rev * 100.0) if (latest_net_income is not None and latest_rev and latest_rev > 0) else None
        kpis.append(SectorKPIItem(
            name="Net Profit Margin (%)",
            value=round(net_margin, 1) if net_margin is not None else None,
            unit="%",
            benchmark=">= 25% strong banking margin",
            status="available" if net_margin is not None else "unavailable",
        ))
        if net_margin is not None:
            scores.append(90.0 if net_margin >= 30.0 else (75.0 if net_margin >= 20.0 else 45.0))

        # Equity-to-Assets %
        eq_assets = (latest_equity / latest_assets * 100.0) if (latest_equity is not None and latest_assets and latest_assets > 0) else None
        kpis.append(SectorKPIItem(
            name="Equity-to-Assets Capital Buffer (%)",
            value=round(eq_assets, 1) if eq_assets is not None else None,
            unit="%",
            benchmark=">= 8% solid capital buffer",
            status="available" if eq_assets is not None else "unavailable",
        ))
        if eq_assets is not None:
            scores.append(90.0 if eq_assets >= 10.0 else (75.0 if eq_assets >= 7.0 else 40.0))

    # 2. Semiconductors / Hardware
    elif any(k in san_sec for k in ("semiconductor", "semi", "hardware")):
        latest_rev = getattr(latest, "total_revenue", None) or getattr(latest, "revenue", None)
        latest_gp = getattr(latest, "gross_profit", None)
        latest_capex = getattr(latest, "capital_expenditure", None) or getattr(latest, "capex", None)
        latest_fcf = getattr(latest, "free_cash_flow", None) or getattr(latest, "fcf", None)
        latest_ocf = getattr(latest, "operating_cash_flow", None) or getattr(latest, "ocf", None)

        # Gross Margin %
        gm = (latest_gp / latest_rev * 100.0) if (latest_gp is not None and latest_rev and latest_rev > 0) else None
        kpis.append(SectorKPIItem(
            name="Gross Margin (%)",
            value=round(gm, 1) if gm is not None else None,
            unit="%",
            benchmark=">= 50% fabless/semi moat",
            status="available" if gm is not None else "unavailable",
        ))
        if gm is not None:
            scores.append(95.0 if gm >= 55.0 else (75.0 if gm >= 40.0 else 45.0))

        # CapEx Intensity %
        capex_int = (abs(latest_capex) / latest_rev * 100.0) if (latest_capex is not None and latest_rev and latest_rev > 0) else None
        kpis.append(SectorKPIItem(
            name="CapEx Intensity (CapEx / Revenue %)",
            value=round(capex_int, 1) if capex_int is not None else None,
            unit="%",
            benchmark="< 20% asset-light / 20-35% foundry",
            status="available" if capex_int is not None else "unavailable",
        ))
        if capex_int is not None:
            scores.append(90.0 if capex_int <= 20.0 else (75.0 if capex_int <= 35.0 else 50.0))

        # FCF Conversion %
        fcf_conv = (latest_fcf / latest_ocf * 100.0) if (latest_fcf is not None and latest_ocf and latest_ocf > 0) else None
        kpis.append(SectorKPIItem(
            name="FCF Conversion (FCF / OCF %)",
            value=round(fcf_conv, 1) if fcf_conv is not None else None,
            unit="%",
            benchmark=">= 50% strong cash conversion",
            status="available" if fcf_conv is not None else "unavailable",
        ))
        if fcf_conv is not None:
            scores.append(90.0 if fcf_conv >= 60.0 else (75.0 if fcf_conv >= 40.0 else 45.0))

    # 3. Real Estate / REITs
    elif any(k in san_sec for k in ("real estate", "reit", "property")):
        latest_rev = getattr(latest, "total_revenue", None) or getattr(latest, "revenue", None)
        latest_ocf = getattr(latest, "operating_cash_flow", None) or getattr(latest, "ocf", None)
        latest_div = getattr(latest, "dividends_paid", None)

        # OCF Margin %
        ocf_m = (latest_ocf / latest_rev * 100.0) if (latest_ocf is not None and latest_rev and latest_rev > 0) else None
        kpis.append(SectorKPIItem(
            name="Operating Cash Flow Margin (OCF / Revenue %)",
            value=round(ocf_m, 1) if ocf_m is not None else None,
            unit="%",
            benchmark=">= 40% strong rental cash flow",
            status="available" if ocf_m is not None else "unavailable",
        ))
        if ocf_m is not None:
            scores.append(90.0 if ocf_m >= 45.0 else (70.0 if ocf_m >= 25.0 else 40.0))

        # Dividend / OCF Coverage %
        div_cov = (abs(latest_div) / latest_ocf * 100.0) if (latest_div is not None and latest_ocf and latest_ocf > 0) else None
        kpis.append(SectorKPIItem(
            name="Dividend / OCF Coverage (%)",
            value=round(div_cov, 1) if div_cov is not None else None,
            unit="%",
            benchmark="<= 90% sustainable distribution",
            status="available" if div_cov is not None else "unavailable",
        ))
        if div_cov is not None:
            scores.append(90.0 if div_cov <= 85.0 else (70.0 if div_cov <= 100.0 else 35.0))

    # 4. Technology / SaaS / Software
    elif any(k in san_sec for k in ("tech", "software", "saas", "internet", "cloud")):
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

    # 5. Retail / Consumer / Goods
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

    # 6. Energy / Industrial / Commodity
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

    # 7. General / Default Sector
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
