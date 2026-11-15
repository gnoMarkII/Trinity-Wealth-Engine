"""Declarative Sector KPI & Multi-Period Forensics Engine (Phase 3 & v3.1).

Computes sector-specific operating metrics (SaaS Rule of 40, Retail Working Capital Days,
Banking ROE/NIM, Semi Gross Margins & CapEx Intensity, Energy FCF Conversion)
from multi-period financial statements without penalizing unavailable operational data.
"""
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field

from schemas.micro_quant_schemas import DataStatus
from tools.market.financial_autopsy import FinancialAutopsyPeriod


def _gross_margin_percent(gross_profit: Any, revenue: Any) -> Optional[Decimal]:
    if gross_profit is None or revenue is None:
        return None
    try:
        gross_profit_decimal = Decimal(str(gross_profit))
        revenue_decimal = Decimal(str(revenue))
        if not gross_profit_decimal.is_finite() or not revenue_decimal.is_finite() or revenue_decimal <= 0:
            return None
        return gross_profit_decimal / revenue_decimal * Decimal("100")
    except (InvalidOperation, TypeError, ValueError):
        return None


class SectorKPIItem(BaseModel):
    name: str
    value: Optional[float]
    unit: str  # "%", "days", "x", "USD"
    benchmark: Optional[str] = None
    status: DataStatus = "available"
    period_basis: Optional[Literal["annual", "quarterly", "ttm", "unknown"]] = None


class SectorKPISummary(BaseModel):
    sector_name: str
    kpis: list[SectorKPIItem] = Field(default_factory=list)
    sector_health_score: Optional[float] = None  # 0 - 100
    status: DataStatus = "available"
    flags: list[str] = Field(default_factory=list)


def compute_sector_kpis(
    sector: str,
    periods: list[FinancialAutopsyPeriod],
    *,
    industry: Optional[str] = None,
    ttm_period: Optional[FinancialAutopsyPeriod] = None,
) -> SectorKPISummary:
    """Computes declarative sector KPIs from multi-period financial autopsy statements."""
    san_sec = f"{industry or ''} {sector or ''}".strip().lower()
    flags: list[str] = []

    if (not periods or len(periods) < 1) and ttm_period is None:
        return SectorKPISummary(
            sector_name=sector or industry or "Unknown",
            kpis=[],
            sector_health_score=None,
            status="unavailable",
            flags=["insufficient_financial_periods"],
        )

    # Ensure periods are ordered newest to oldest
    has_dates = bool(periods) and all(getattr(p, "fiscal_period_end", None) for p in periods)
    if has_dates:
        sorted_periods = sorted(periods, key=lambda p: str(getattr(p, "fiscal_period_end")), reverse=True)
    else:
        # If periods[0] has data or no dates, treat periods as either newest-first or oldest-first
        # In financial_autopsy.py, periods[0] is typically the latest period.
        sorted_periods = list(periods or [])

    latest = sorted_periods[0] if sorted_periods else ttm_period
    prev = sorted_periods[1] if len(sorted_periods) >= 2 else None

    def _rev(p: Optional[FinancialAutopsyPeriod]) -> Optional[float]:
        if p is None:
            return None
        v = getattr(p, "total_revenue", None)
        return v if v is not None else getattr(p, "revenue", None)

    def _capex(p: Optional[FinancialAutopsyPeriod]) -> Optional[float]:
        if p is None:
            return None
        v = getattr(p, "capital_expenditure", None)
        return v if v is not None else getattr(p, "capex", None)

    def _fcf(p: Optional[FinancialAutopsyPeriod]) -> Optional[float]:
        if p is None:
            return None
        v = getattr(p, "free_cash_flow", None)
        return v if v is not None else getattr(p, "fcf", None)

    def _ocf(p: Optional[FinancialAutopsyPeriod]) -> Optional[float]:
        if p is None:
            return None
        v = getattr(p, "operating_cash_flow", None)
        return v if v is not None else getattr(p, "ocf", None)

    def _period_basis(period: Optional[FinancialAutopsyPeriod]) -> Literal["annual", "quarterly", "ttm", "unknown"]:
        basis = getattr(period, "period_type", None) if period is not None else None
        return basis if basis in ("annual", "quarterly", "ttm") else "unknown"

    kpis: list[SectorKPIItem] = []
    scores: list[float] = []

    # 1. Banking / Financial Services / Insurance
    if any(k in san_sec for k in ("financial", "bank", "insurance")):
        latest_net_income = getattr(latest, "net_income", None)
        latest_equity = getattr(latest, "stockholders_equity", None) or getattr(latest, "total_equity", None)
        latest_rev = _rev(latest)
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
        # Gross Margin % must use one period basis; a partial TTM GP stays unavailable.
        gm_src = ttm_period if ttm_period is not None else latest
        gm = _gross_margin_percent(getattr(gm_src, "gross_profit", None), _rev(gm_src))
        gm_value = float(gm.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)) if gm is not None else None
        kpis.append(SectorKPIItem(
            name="Gross Margin (%)",
            value=gm_value,
            unit="%",
            benchmark=">= 50% fabless/semi moat",
            status="available" if gm is not None else "unavailable",
            period_basis=_period_basis(gm_src),
        ))
        if gm is not None:
            scores.append(95.0 if gm >= Decimal("55.0") else (75.0 if gm >= Decimal("40.0") else 45.0))

        # CapEx Intensity % (period-matched: prefer TTM)
        cap_src = ttm_period if (ttm_period and _capex(ttm_period) is not None and (_rev(ttm_period) or 0) > 0) else latest
        cap_rev = _rev(cap_src)
        cap_val = _capex(cap_src)
        capex_int = (abs(cap_val) / cap_rev * 100.0) if (cap_val is not None and cap_rev and cap_rev > 0) else None
        kpis.append(SectorKPIItem(
            name="CapEx Intensity (CapEx / Revenue %)",
            value=round(capex_int, 1) if capex_int is not None else None,
            unit="%",
            benchmark="< 20% asset-light / 20-35% foundry",
            status="available" if capex_int is not None else "unavailable",
        ))
        if capex_int is not None:
            scores.append(90.0 if capex_int <= 20.0 else (75.0 if capex_int <= 35.0 else 50.0))

        # FCF Conversion % (period-matched: prefer TTM)
        fcf_src = ttm_period if (ttm_period and _fcf(ttm_period) is not None and _ocf(ttm_period) is not None) else latest
        fcf_val = _fcf(fcf_src)
        ocf_val = _ocf(fcf_src)
        fcf_conv = (fcf_val / ocf_val * 100.0) if (fcf_val is not None and ocf_val and ocf_val > 0) else None
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
        ocf_src = ttm_period if (ttm_period and _ocf(ttm_period) is not None and (_rev(ttm_period) or 0) > 0) else latest
        latest_rev = _rev(ocf_src)
        latest_ocf = _ocf(ocf_src)
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
        div_ocf = _ocf(latest)
        div_cov = (abs(latest_div) / div_ocf * 100.0) if (latest_div is not None and div_ocf and div_ocf > 0) else None
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
        # Rule of 40: Annual Revenue Growth YoY % + TTM (or latest annual) FCF Margin %
        fcf_src = ttm_period if (ttm_period and _fcf(ttm_period) is not None and (_rev(ttm_period) or 0) > 0) else latest
        fcf_rev = _rev(fcf_src)
        fcf_val = _fcf(fcf_src)
        fcf_m = (fcf_val / fcf_rev * 100.0) if (fcf_val is not None and fcf_rev and fcf_rev > 0) else None

        annual_periods = [p for p in sorted_periods if getattr(p, "period_type", None) not in ("quarterly", "ttm")]
        growth_curr = annual_periods[0] if len(annual_periods) >= 2 else latest
        growth_prev = annual_periods[1] if len(annual_periods) >= 2 else prev
        latest_rev_g = _rev(growth_curr)
        prev_rev_g = _rev(growth_prev)
        rev_g = 0.0
        if prev_rev_g and prev_rev_g > 0 and latest_rev_g is not None:
            rev_g = ((latest_rev_g - prev_rev_g) / prev_rev_g) * 100.0

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

        # Gross Margin uses the TTM period when present; incomplete TTM data stays unavailable.
        gm_src = ttm_period if ttm_period is not None else latest
        gm = _gross_margin_percent(getattr(gm_src, "gross_profit", None), _rev(gm_src))
        gm_value = float(gm.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)) if gm is not None else None
        kpis.append(SectorKPIItem(
            name="Subscription / Gross Margin",
            value=gm_value,
            unit="%",
            benchmark=">= 70% for pure software",
            status="available" if gm is not None else "unavailable",
            period_basis=_period_basis(gm_src),
        ))
        if gm is not None:
            scores.append(90.0 if gm >= Decimal("70.0") else (70.0 if gm >= Decimal("50.0") else 40.0))

    # 5. Retail / Consumer / Goods
    elif any(k in san_sec for k in ("retail", "consumer", "goods", "commerce")):
        latest_rev = _rev(latest)
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
        conv_src = ttm_period if (ttm_period and _fcf(ttm_period) is not None and _ocf(ttm_period) is not None) else latest
        latest_ocf = _ocf(conv_src)
        latest_fcf = _fcf(conv_src)

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
        cap_src = ttm_period if (ttm_period and _capex(ttm_period) is not None and (_rev(ttm_period) or 0) > 0) else latest
        latest_capex = _capex(cap_src)
        latest_rev = _rev(cap_src)
        capex_int = None
        if latest_rev and latest_rev > 0 and latest_capex is not None:
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
        ebit_src = (
            ttm_period
            if (
                ttm_period
                and (getattr(ttm_period, "ebit", None) is not None or getattr(ttm_period, "operating_income", None) is not None)
                and (_rev(ttm_period) or 0) > 0
            )
            else latest
        )
        latest_ebit = getattr(ebit_src, "ebit", None)
        if latest_ebit is None:
            latest_ebit = getattr(ebit_src, "operating_income", None)
        latest_rev = _rev(ebit_src)
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
        sector_name=sector or industry or "General",
        kpis=kpis,
        sector_health_score=health_score,
        status="available" if kpis else "partial",
        flags=flags,
    )
