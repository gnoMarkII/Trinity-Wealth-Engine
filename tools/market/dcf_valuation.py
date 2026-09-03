"""DCF / Multi-Valuation Fair Value Engine — Real WACC, CAPM, and Single Source Macro Observables.

Baseline Source Citations (For Fallbacks):
1. Thai 10Y Bond Yield: ~2.75% (Bank of Thailand / ThaiBMA Q2 2026 Baseline)
2. Thailand Country Risk Premium (CRP): ~1.75% (Damodaran Rating-based CRP Table for Baa2/BBB Sovereign Rating)
"""
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Any, Optional, Dict, List, Tuple, Literal
import math
from pathlib import Path

from langsmith import traceable

from core.logger import get_logger
from schemas.macro_schemas import MarketObservable
from schemas.micro_quant_schemas import DCFResult, DCFScenario, ExplicitFCFProjection, ReverseDCFResult
from tools.macro.valuation import _find_dgs10_in_observables, VALUATION_RICH_ERP_THRESHOLD

log = get_logger(__name__)


@traceable(run_type="parser")
def _extract_obs_erp(macro_registry: Dict[str, MarketObservable]) -> Tuple[float, Optional[str]]:
    """Helper ดึง ERP จาก macro_registry พร้อม ID โดยแยก Fed Model Yield Gap ออกจาก CAPM ERP"""
    # 1. Look for explicit institutional CAPM / Damodaran ERP first
    for key in ("obs_us_damodaran_erp", "obs_damodaran_erp", "obs_capm_erp"):
        obs = macro_registry.get(key)
        if obs and getattr(obs, "is_valid", True):
            try:
                val = float(obs.value)
                if val > 0:
                    return val, obs.observable_id
            except ValueError:
                pass

    # 2. Check obs_erp_gspc
    obs_gspc = macro_registry.get("obs_erp_gspc")
    if obs_gspc and getattr(obs_gspc, "is_valid", True):
        try:
            val = float(obs_gspc.value)
            calc_method = getattr(obs_gspc, "calculation_method", "") or ""
            meta = getattr(obs_gspc, "metadata", {}) or {}
            is_fed_model = "forwardpe" in calc_method.lower() or "is_rich" in meta
            if is_fed_model and val <= 1.5:
                # Fed Model indicates equity yields are rich vs 10Y bonds, but for CAPM discounting,
                # equity cost of capital requires positive equity risk premium (Damodaran US baseline 4.60%)
                return 4.60, obs_gspc.observable_id
            return val, obs_gspc.observable_id
        except ValueError:
            pass

    # Fallback to scanning indicators
    for obs in macro_registry.values():
        if obs and getattr(obs, "is_valid", True):
            if "erp" in obs.observable_id.lower() or "erp" in obs.indicator.lower() or "equity risk premium" in obs.indicator.lower():
                try:
                    val = float(obs.value)
                    calc_method = getattr(obs, "calculation_method", "") or ""
                    meta = getattr(obs, "metadata", {}) or {}
                    is_fed_model = "forwardpe" in calc_method.lower() or "is_rich" in meta
                    if is_fed_model and val <= 1.5:
                        return 4.60, obs.observable_id
                    return val, obs.observable_id
                except ValueError:
                    pass
    return 4.60, None


def validate_macro_observables_freshness(
    macro_registry: Dict[str, MarketObservable],
    as_of_dt: Optional[datetime] = None,
) -> Tuple[bool, List[str]]:
    """Validates macro observable publication dates against trading calendars and source cadences."""
    now_dt = as_of_dt or datetime.now(timezone.utc)
    invalidation_flags: List[str] = []

    # 1. US 10Y Treasury (SIFMA calendar: 5 publication days max, published_at <= as_of)
    # Check all candidate 10Y observables and take the freshest (newest by date)
    dgs10_candidates = [
        obs for obs in macro_registry.values()
        if obs and getattr(obs, "is_valid", True) and (
            "10y" in obs.observable_id.lower() or "dgs10" in obs.observable_id.lower()
            or "10y" in obs.indicator.lower() or "dgs10" in obs.indicator.lower()
        ) and not (
            "spread" in obs.indicator.lower() or "breakeven" in obs.indicator.lower()
            or "t10y2y" in obs.observable_id.lower() or "t10yie" in obs.observable_id.lower()
            or "dfii10" in obs.observable_id.lower()
        )
    ]
    if dgs10_candidates:
        def _get_obs_date(obs: MarketObservable) -> datetime.date:
            d_str = getattr(obs, "observed_at", None) or getattr(obs, "as_of_date", None)
            if d_str:
                try:
                    return datetime.strptime(d_str[:10], "%Y-%m-%d").date()
                except Exception:
                    pass
            return datetime(2000, 1, 1).date()

        dgs10_candidates.sort(key=_get_obs_date, reverse=True)
        dgs10_obs = dgs10_candidates[0]
        pub_d = _get_obs_date(dgs10_obs)
        if pub_d > now_dt.date():
            invalidation_flags.append("macro_publication_future_leak:us_10y")
        elif (now_dt.date() - pub_d).days > 7:  # > 5 trading days
            invalidation_flags.append("macro_stale_publication:us_10y")

    # 2. ERP Cadence check
    erp_obs = next((obs for obs in macro_registry.values() if obs and "erp" in obs.observable_id.lower()), None)
    if erp_obs and getattr(erp_obs, "is_valid", True):
        pub_date_str = getattr(erp_obs, "observed_at", None) or getattr(erp_obs, "as_of_date", None)
        cadence = (erp_obs.metadata.get("publication_cadence") if (hasattr(erp_obs, "metadata") and erp_obs.metadata) else None) or getattr(erp_obs, "publication_cadence", "monthly")
        cadence = str(cadence).lower()
        max_days = 35 if cadence == "monthly" else (105 if cadence == "quarterly" else 380)
        if pub_date_str:
            try:
                pub_d = datetime.strptime(pub_date_str[:10], "%Y-%m-%d").date()
                if (now_dt.date() - pub_d).days > max_days:
                    invalidation_flags.append("macro_stale_publication:erp")
            except Exception:
                pass

    is_fresh = len(invalidation_flags) == 0
    return is_fresh, invalidation_flags


@traceable(run_type="parser")
def compute_dcf_valuation(
    ticker: str,
    market: str,
    current_price: float,
    beta: Optional[float],
    fcf_per_share: float,
    market_cap: float,
    total_debt: float,
    interest_expense: Optional[float],
    tax_rate: float,
    fcf_cagr_3y: Optional[float],
    macro_registry: Dict[str, MarketObservable],
    forward_eps: Optional[float] = None,
    trailing_eps: Optional[float] = None,
    is_us_domestic: bool = True,
    cash_and_equivalents: float = 0.0,
) -> Tuple[Optional[DCFResult], List[str]]:
    """คำนวณ DCF Multi-Scenario (Bull/Base/Bear) ด้วย Real WACC, Observable Refs, และ Tax Policy Guards"""
    flags: List[str] = []
    observable_refs: List[str] = []

    # 1. Guards: Negative FCF or Missing Beta
    if fcf_per_share <= 0:
        flags.append("negative_fcf_dcf_unavailable:dcf")
        return None, flags

    if beta is None:
        flags.append("beta_unavailable_dcf_unavailable:dcf")
        return None, flags

    if current_price <= 0:
        return None, flags

    # Macro Freshness Guard
    is_fresh, freshness_flags = validate_macro_observables_freshness(macro_registry)
    if not is_fresh:
        flags.extend(freshness_flags)

    # 2. Dynamic Observable Refs Resolution & Market CAPM
    if market == "TH":
        th_rf_obs = macro_registry.get("obs_th_10y_yield")  # Literal key
        if th_rf_obs and getattr(th_rf_obs, "is_valid", True):
            try:
                risk_free_rate = float(th_rf_obs.value)
                observable_refs.append(th_rf_obs.observable_id)
            except ValueError:
                risk_free_rate = 2.75
                flags.append("hardcoded_th_risk_free:dcf")
        else:
            risk_free_rate = 2.75  # Primary Operational Path
            flags.append("hardcoded_th_risk_free:dcf")

        th_crp_obs = macro_registry.get("obs_th_crp")
        if th_crp_obs and getattr(th_crp_obs, "is_valid", True):
            try:
                crp = float(th_crp_obs.value)
                observable_refs.append(th_crp_obs.observable_id)
            except ValueError:
                crp = 1.75
                flags.append("hardcoded_country_risk_premium:dcf")
        else:
            crp = 1.75  # Primary Operational Path (Damodaran Baa2/BBB baseline)
            flags.append("hardcoded_country_risk_premium:dcf")

        us_erp, erp_id = _extract_obs_erp(macro_registry)
        if erp_id:
            observable_refs.append(erp_id)
        erp_pct = us_erp + crp
    else:
        # US Market: Use shared _find_dgs10_in_observables with exclusions
        dgs10_val, dgs10_id = _find_dgs10_in_observables(list(macro_registry.values()))
        if dgs10_val is not None:
            risk_free_rate = dgs10_val
            if dgs10_id:
                observable_refs.append(dgs10_id)
        else:
            risk_free_rate = 4.25
            flags.append("hardcoded_us_risk_free:dcf")

        us_erp, erp_id = _extract_obs_erp(macro_registry)
        if erp_id:
            observable_refs.append(erp_id)
        erp_pct = us_erp

    # Check ERP richness threshold (1.5%)
    if erp_pct < (VALUATION_RICH_ERP_THRESHOLD * 100.0):
        flags.append("rich_market_valuation_low_erp:dcf")

    # Upfront Domain Validation for Decimal fractions
    rf_dec = Decimal(str(risk_free_rate)) / Decimal("100")
    erp_dec = Decimal(str(erp_pct)) / Decimal("100")
    
    # Tax Policy application: normalize statutory 21% fallback only for US domestic
    if tax_rate is None or tax_rate <= 0:
        if market == "US" and is_us_domestic:
            tax_rate_dec = Decimal("0.21")
            flags.append("tax_policy:US_FEDERAL_STATUTORY_21PCT")
        else:
            tax_rate_dec = Decimal("0.20")
    else:
        tax_rate_dec = Decimal(str(tax_rate)) if tax_rate <= 1.0 else (Decimal(str(tax_rate)) / Decimal("100"))
    
    beta_dec = Decimal(str(beta))

    domain_errors: List[str] = []
    if not (Decimal("0.0") <= rf_dec <= Decimal("0.20")):
        domain_errors.append("rate_out_of_domain:rf")
    if not (Decimal("-0.10") <= erp_dec <= Decimal("0.20")):
        domain_errors.append("rate_out_of_domain:erp")
    if not (Decimal("0.0") <= tax_rate_dec <= Decimal("0.60")):
        domain_errors.append("rate_out_of_domain:tax_rate")

    if domain_errors:
        flags.extend(["invalid_rate_scale:dcf"] + domain_errors)
        return DCFResult(
            wacc_pct=None,
            cost_of_equity_pct=None,
            cost_of_debt_pct=None,
            risk_free_rate_pct=float(rf_dec * 100),
            erp_pct=float(erp_dec * 100),
            observable_refs=observable_refs,
            scenarios={},
            valuation_verdict="unavailable",
            is_actionable=False,
            actionability_reason=domain_errors[0],
            invalidation_reasons=["invalid_rate_scale:dcf"] + domain_errors,
        ), flags

    # 3. Cost of Equity (Ke) & Cost of Debt (Kd) in Decimal fractions
    ke_dec = rf_dec + (beta_dec * erp_dec)
    ke = float(ke_dec * 100)

    if total_debt > 0 and interest_expense is not None and interest_expense != 0:
        raw_kd = (abs(interest_expense) / total_debt) * 100.0
        kd = max(2.0, min(15.0, raw_kd))
        if raw_kd < 2.0 or raw_kd > 15.0:
            flags.append("kd_clamped:dcf")
    else:
        kd = 5.0
        flags.append("hardcoded_cost_of_debt:dcf")

    kd_dec = Decimal(str(kd)) / Decimal("100")

    # 4. Capital Structure Weighting (Real WACC)
    v = market_cap + total_debt if (market_cap is not None and market_cap > 0) else total_debt
    e_weight_dec = Decimal(str(market_cap / v)) if (market_cap and v > 0) else Decimal("1.0")
    d_weight_dec = Decimal(str(total_debt / v)) if (total_debt and v > 0) else Decimal("0.0")

    wacc_dec = (e_weight_dec * ke_dec) + (d_weight_dec * kd_dec * (Decimal("1.0") - tax_rate_dec))
    wacc_pct = round(float(wacc_dec * 100), 2)

    # 5. Projection & Gordon Growth Model
    g_terminal_dec = min(Decimal("0.025"), rf_dec)
    g_terminal = float(g_terminal_dec)

    # Invalidation checking (Multi-Reason)
    invalidation_reasons: List[str] = []
    if erp_dec <= Decimal("0.0"):
        invalidation_reasons.append("erp_non_positive")
    if wacc_dec <= (g_terminal_dec + Decimal("0.005")):
        invalidation_reasons.append("wacc_terminal_spread_insufficient")
    if beta_dec > Decimal("0.0") and ke_dec < rf_dec:
        invalidation_reasons.append("cost_of_equity_below_risk_free")
    if not is_fresh:
        invalidation_reasons.append("macro_inputs_stale_or_misaligned")

    if wacc_dec <= g_terminal_dec:
        flags.append("wacc_below_terminal_growth:dcf")

    if invalidation_reasons:
        for r in invalidation_reasons:
            flags.append(f"invalidation:{r}")
        return DCFResult(
            wacc_pct=wacc_pct,
            cost_of_equity_pct=round(ke, 2),
            cost_of_debt_pct=round(kd, 2),
            risk_free_rate_pct=risk_free_rate,
            erp_pct=erp_pct,
            observable_refs=observable_refs,
            scenarios={},  # UNIVERSAL TARGET SUPPRESSION: No distorted targets when invalid
            valuation_verdict="unavailable",
            is_actionable=False,
            actionability_reason=invalidation_reasons[0],
            invalidation_reasons=invalidation_reasons,
        ), flags

    # 5.1 Base Growth Rate
    if (
        forward_eps is not None and trailing_eps is not None
        and trailing_eps != 0
        and abs(trailing_eps) > 0.01
    ):
        yoy_eps_growth = (forward_eps - trailing_eps) / abs(trailing_eps)
        base_g = max(-0.10, min(0.25, yoy_eps_growth))
        flags.append("eps_proxy_base_growth:dcf")
    else:
        base_g = 0.05
        flags.append("generic_base_growth_assumption:dcf")

    # 5.2 Bull Growth Rate (guaranteed > base_g)
    fcf_cagr_cand = (fcf_cagr_3y / 100.0) if (fcf_cagr_3y is not None and fcf_cagr_3y > 0) else None
    if fcf_cagr_cand is not None:
        bull_g = max(base_g + 0.03, min(0.35, max(base_g * 1.25, fcf_cagr_cand)))
    else:
        bull_g = max(base_g + 0.03, base_g * 1.25 if base_g > 0 else base_g + 0.05)

    # 5.3 Bear Growth Rate (guaranteed < base_g)
    if base_g > 0:
        bear_g = max(-0.15, min(base_g - 0.03, base_g * 0.50))
    else:
        bear_g = max(-0.25, base_g - 0.05)

    # Explicit runtime guard to guarantee growth ordering
    if not (bear_g <= base_g <= bull_g):
        bear_g = min(bear_g, base_g - 0.02)
        bull_g = max(bull_g, base_g + 0.02)

    # Net cash per share adjustment for Equity Value (Cash - Debt)
    net_cash_per_share = 0.0
    if cash_and_equivalents > 0 and current_price > 0 and market_cap > 0:
        shares_out = market_cap / current_price
        if shares_out > 0:
            net_cash_per_share = (cash_and_equivalents - total_debt) / shares_out

    scenarios: Dict[str, DCFScenario] = {}
    wacc_f = float(wacc_dec)
    for key, g in [("bull", bull_g), ("base", base_g), ("bear", bear_g)]:
        pv_fcf = sum([(fcf_per_share * ((1.0 + g) ** t)) / ((1.0 + wacc_f) ** t) for t in range(1, 6)])
        fcf_5 = fcf_per_share * ((1.0 + g) ** 5)

        if wacc_f > g_terminal:
            terminal_val = (fcf_5 * (1.0 + g_terminal)) / (wacc_f - g_terminal)
        else:
            terminal_val = 0.0

        pv_terminal = terminal_val / ((1.0 + wacc_f) ** 5)
        target_price = max(0.0, round(pv_fcf + pv_terminal + net_cash_per_share, 2))
        upside = round(((target_price - current_price) / current_price) * 100.0, 1)
        mos = round(((target_price - current_price) / target_price) * 100.0, 1) if target_price > 0 else 0.0
        scenarios[key] = DCFScenario(target_price=target_price, upside_pct=upside, margin_of_safety_pct=mos)

    # 6. Verdict via MoS
    base_tp = scenarios["base"].target_price
    if base_tp is not None:
        if current_price <= base_tp * 0.80:
            verdict = "undervalued"
        elif current_price >= base_tp * 1.15:
            verdict = "overvalued"
        else:
            verdict = "fairly_valued"
    else:
        verdict = "unavailable"

    result = DCFResult(
        wacc_pct=wacc_pct,
        cost_of_equity_pct=round(ke, 2),
        cost_of_debt_pct=round(kd, 2),
        risk_free_rate_pct=risk_free_rate,
        erp_pct=erp_pct,
        observable_refs=observable_refs,
        scenarios=scenarios,
        valuation_verdict=verdict,
        is_actionable=True,
        actionability_reason=None,
        invalidation_reasons=[],
    )
    return result, flags


def _calculate_per_share_value(
    revenue_growth: float,
    ebit_margin: float,
    base_revenue: float,
    tax_rate: float,
    reinvestment_rate: float,
    wacc_dec: float,
    g_terminal: float,
    net_cash_debt: float,
    shares_out: float,
    exit_multiple: Optional[float] = None,
) -> Tuple[float, List[ExplicitFCFProjection], float, float, Optional[float]]:
    """Helper คำนวณ Intrinsic Value per share จาก Forecast 5 ปีเต็ม พร้อม Gordon Growth และ Exit Multiple"""
    projections: List[ExplicitFCFProjection] = []
    curr_rev = base_revenue
    sum_pv = 0.0

    for t in range(1, 6):
        curr_rev = curr_rev * (1.0 + revenue_growth)
        ebit = curr_rev * ebit_margin
        nopat = ebit * (1.0 - tax_rate)
        # Damodaran standard: Reinvestment Rate is a percentage of NOPAT
        reinvest_currency = nopat * reinvestment_rate
        fcf = max(0.0, nopat - reinvest_currency)
        df = (1.0 + wacc_dec) ** (-t)
        pv = fcf * df
        sum_pv += pv
        projections.append(
            ExplicitFCFProjection(
                year_index=t,
                projected_revenue=round(curr_rev, 2),
                projected_ebit=round(ebit, 2),
                projected_ebit_margin_pct=round(ebit_margin * 100.0, 2),
                projected_nopat=round(nopat, 2),
                projected_reinvestment_currency=round(reinvest_currency, 2),
                projected_fcf=round(fcf, 2),
                discount_factor=round(df, 4),
                pv_fcf=round(pv, 2),
            )
        )

    last_fcf = projections[-1].projected_fcf
    if wacc_dec > g_terminal and last_fcf > 0:
        tv_undisc = (last_fcf * (1.0 + g_terminal)) / (wacc_dec - g_terminal)
    else:
        tv_undisc = 0.0
    tv_pv = tv_undisc * ((1.0 + wacc_dec) ** -5)
    ev = sum_pv + tv_pv
    eq_val = ev + net_cash_debt
    per_share = (eq_val / shares_out) if shares_out > 0 else 0.0

    per_share_exit = None
    if exit_multiple is not None and exit_multiple > 0 and last_fcf > 0:
        tv_exit_undisc = last_fcf * exit_multiple
        tv_exit_pv = tv_exit_undisc * ((1.0 + wacc_dec) ** -5)
        ev_exit = sum_pv + tv_exit_pv + net_cash_debt
        per_share_exit = round((ev_exit / shares_out), 2) if shares_out > 0 else 0.0

    return per_share, projections, sum_pv, tv_pv, per_share_exit


def compute_institutional_reverse_dcf(
    ticker: str,
    market: str,
    current_price: float,
    shares_outstanding: float,
    base_revenue: float,
    base_ebit_margin_pct: Optional[float] = None,
    tax_rate: float = 0.21,
    reinvestment_rate_pct: float = 10.0,
    beta: Optional[float] = None,
    total_debt: float = 0.0,
    cash_and_equivalents: float = 0.0,
    interest_expense: Optional[float] = None,
    macro_registry: Optional[Dict[str, MarketObservable]] = None,
    sector: Optional[str] = None,
    forecast_revenue_growth_pct: Optional[float] = None,
    expected_dividend_per_share: float = 0.0,
    ebit_margin_fiscal_period: Optional[str] = None,
    ebit_margin_period_type: Optional[Literal["annual", "quarterly", "ttm", "unknown"]] = None,
    ebit_margin_source_tier: Optional[Literal["filing_authoritative", "primary_best_effort", "fallback", "unknown"]] = None,
    consensus_target_price: Optional[float] = None,
    consensus_target_high: Optional[float] = None,
    consensus_target_low: Optional[float] = None,
    analyst_count: Optional[int] = None,
    base_revenue_period_type: Optional[Literal["annual", "quarterly", "ttm", "unknown"]] = None,
) -> Tuple[ReverseDCFResult, List[str]]:
    """คำนวณ Explicit 5-Year Forward DCF และ Bounded Reverse DCF Solver พร้อม Sector Exclusion และ Robust Margin Guards"""
    flags: List[str] = []

    # 1. Missing / Non-positive Margin Guard (Policy Guard before arithmetic)
    if base_ebit_margin_pct is None or base_ebit_margin_pct <= 0 or not math.isfinite(base_ebit_margin_pct):
        return ReverseDCFResult(
            status="unavailable",
            valuation_verdict="unavailable",
            is_eligible=False,
            exclusion_reason="missing_or_non_positive_ebit_margin",
            reported_ebit_margin_pct=base_ebit_margin_pct,
            ebit_margin_fiscal_period=ebit_margin_fiscal_period,
            ebit_margin_period_type=ebit_margin_period_type,
            ebit_margin_source_tier=ebit_margin_source_tier,
        ), ["missing_operating_margin:dcf"]

    # 2. Sector Exclusion
    excluded_sectors = ["financials", "financial services", "real estate", "banks", "insurance", "reit"]
    if sector and sector.strip().lower() in excluded_sectors:
        return ReverseDCFResult(
            status="not_applicable",
            valuation_verdict="unavailable",
            is_eligible=False,
            exclusion_reason=f"Sector '{sector}' is excluded from generic FCF DCF (financial/REIT structure)",
            solver_status="not_applicable",
            reported_ebit_margin_pct=base_ebit_margin_pct,
            ebit_margin_fiscal_period=ebit_margin_fiscal_period,
            ebit_margin_period_type=ebit_margin_period_type,
            ebit_margin_source_tier=ebit_margin_source_tier,
        ), ["sector_excluded_from_dcf:dcf"]

    # 3. Basic Guards
    if current_price <= 0 or shares_outstanding <= 0 or base_revenue <= 0 or beta is None:
        if beta is None:
            flags.append("beta_unavailable_dcf_unavailable:dcf")
        return ReverseDCFResult(
            status="unavailable",
            valuation_verdict="unavailable",
            is_eligible=True,
            exclusion_reason="Missing critical valuation inputs (price, shares, revenue, or beta)",
            solver_status="no_solution",
            reported_ebit_margin_pct=base_ebit_margin_pct,
            ebit_margin_fiscal_period=ebit_margin_fiscal_period,
            ebit_margin_period_type=ebit_margin_period_type,
            ebit_margin_source_tier=ebit_margin_source_tier,
        ), flags

    # 4. Macro & WACC Resolution
    macro_reg = macro_registry or {}
    if market == "TH":
        th_rf_obs = macro_reg.get("obs_th_10y_yield")
        rf = float(th_rf_obs.value) if (th_rf_obs and getattr(th_rf_obs, "is_valid", True)) else 2.75
        th_crp_obs = macro_reg.get("obs_th_crp")
        crp = float(th_crp_obs.value) if (th_crp_obs and getattr(th_crp_obs, "is_valid", True)) else 1.75
        us_erp, _ = _extract_obs_erp(macro_reg)
        erp = us_erp + crp
    else:
        dgs10_val, _ = _find_dgs10_in_observables(list(macro_reg.values()))
        rf = dgs10_val if dgs10_val is not None else 4.25
        us_erp, _ = _extract_obs_erp(macro_reg)
        erp = us_erp

    ke = rf + (beta * erp)
    if total_debt > 0 and interest_expense is not None and interest_expense != 0:
        raw_kd = (abs(interest_expense) / total_debt) * 100.0
        kd = max(2.0, min(15.0, raw_kd))
    else:
        kd = 5.0

    market_cap = current_price * shares_outstanding
    v = market_cap + total_debt
    e_weight = market_cap / v if v > 0 else 1.0
    d_weight = total_debt / v if v > 0 else 0.0
    wacc_pct = (e_weight * ke) + (d_weight * kd * (1.0 - tax_rate))
    wacc_dec = max(0.04, wacc_pct / 100.0)
    g_terminal = min(0.025, rf / 100.0)

    net_cash_debt = cash_and_equivalents - total_debt
    base_ebit_margin = max(0.01, base_ebit_margin_pct / 100.0)
    reinvest_rate = max(0.0, min(0.40, reinvestment_rate_pct / 100.0))
    # 4.1 Actionability Check
    is_actionable = True
    actionability_reason = None
    if erp <= 0.0 or ke < rf:
        is_actionable = False
        actionability_reason = f"Macro parameter anomaly: ERP ({round(erp, 2)}%) <= 0 or Ke ({round(ke, 2)}%) < Rf ({round(rf, 2)}%)"
        flags.append("non_actionable_macro_anomaly:reverse_dcf")
    elif wacc_dec <= g_terminal:
        is_actionable = False
        actionability_reason = f"WACC ({round(wacc_pct, 2)}%) <= Terminal Growth ({round(g_terminal * 100.0, 2)}%)"
        flags.append("wacc_below_terminal_growth:reverse_dcf")

    if not is_actionable:
        # UNIVERSAL TARGET SUPPRESSION INVARIANT:
        # If valuation is non-actionable, NEVER emit target price, upside, intrinsic value, or projections
        return ReverseDCFResult(
            explicit_forecast_5y=[],
            sum_pv_5y_fcf=None,
            terminal_value_undiscounted=None,
            terminal_value_pv=None,
            enterprise_value=None,
            net_cash_debt=round(net_cash_debt, 2),
            equity_value=None,
            intrinsic_value_today=None,
            target_price_12m=None,
            upside_12m_pct=None,
            market_implied_growth_pct=None,
            market_implied_margin_pct=None,
            solver_status="no_solution",
            fixed_parameters={
                "wacc_pct": round(wacc_pct, 2),
                "cost_of_equity_pct": round(ke, 2),
                "terminal_growth_pct": round(g_terminal * 100.0, 2),
            },
            valuation_horizon_months=12,
            status="unavailable",
            is_eligible=True,
            valuation_verdict="unavailable",
            is_actionable=False,
            actionability_reason=actionability_reason,
            raw_wacc_pct=round(wacc_pct, 2),
            effective_wacc_pct=round(wacc_dec * 100.0, 2),
            reported_ebit_margin_pct=round(base_ebit_margin_pct, 2) if base_ebit_margin_pct is not None else None,
            ebit_margin_fiscal_period=ebit_margin_fiscal_period,
            ebit_margin_period_type=ebit_margin_period_type,
            ebit_margin_source_tier=ebit_margin_source_tier,
            consensus_target_price=consensus_target_price,
            consensus_target_high=consensus_target_high,
            consensus_target_low=consensus_target_low,
            analyst_count=analyst_count,
        ), flags

    base_growth = (forecast_revenue_growth_pct / 100.0) if forecast_revenue_growth_pct is not None else 0.08

    # Sector-aware exit multiple benchmark
    sec = (sector or "").lower()
    if any(k in sec for k in ["tech", "software", "information technology", "cloud", "security", "saas"]):
        default_exit_mult = 25.0
    elif any(k in sec for k in ["health", "consumer", "pharma"]):
        default_exit_mult = 18.0
    elif any(k in sec for k in ["industr", "material", "energy", "utilit"]):
        default_exit_mult = 12.0
    else:
        default_exit_mult = 20.0

    # 5. Explicit 5-Year Forward DCF Calculation
    intrinsic_today, projections, sum_pv_fcf, tv_pv, intrinsic_exit = _calculate_per_share_value(
        revenue_growth=base_growth,
        ebit_margin=base_ebit_margin,
        base_revenue=base_revenue,
        tax_rate=tax_rate,
        reinvestment_rate=reinvest_rate,
        wacc_dec=wacc_dec,
        g_terminal=g_terminal,
        net_cash_debt=net_cash_debt,
        shares_out=shares_outstanding,
        exit_multiple=default_exit_mult,
    )

    ke_dec = ke / 100.0
    target_price_12m = max(0.0, round((intrinsic_today * (1.0 + ke_dec)) - expected_dividend_per_share, 2))
    target_price_exit_12m = (
        max(0.0, round((intrinsic_exit * (1.0 + ke_dec)) - expected_dividend_per_share, 2))
        if intrinsic_exit is not None else None
    )
    upside_12m = round(((target_price_12m - current_price) / current_price) * 100.0, 1) if current_price > 0 else 0.0

    # 6. Bounded Reverse DCF Solver (Solve for Implied Revenue Growth)
    # Target: P_model(g) - current_price = 0 over bound g in [-0.20, +0.60]
    low_g, high_g = -0.20, 0.60
    val_low, _, _, _, _ = _calculate_per_share_value(low_g, base_ebit_margin, base_revenue, tax_rate, reinvest_rate, wacc_dec, g_terminal, net_cash_debt, shares_outstanding)
    val_high, _, _, _, _ = _calculate_per_share_value(high_g, base_ebit_margin, base_revenue, tax_rate, reinvest_rate, wacc_dec, g_terminal, net_cash_debt, shares_outstanding)

    implied_growth: Optional[float] = None
    solver_status = "converged"

    if current_price < val_low:
        solver_status = "bounded_extreme"
        implied_growth = low_g * 100.0
        flags.append("implied_growth_below_lower_bound:dcf")
    elif current_price > val_high:
        solver_status = "bounded_extreme"
        implied_growth = high_g * 100.0
        flags.append("implied_growth_above_upper_bound:dcf")
    else:
        # Bisection Root-Finding (50 iterations for precision < 1e-6)
        a, b = low_g, high_g
        for _ in range(50):
            mid_g = (a + b) / 2.0
            mid_val, _, _, _, _ = _calculate_per_share_value(mid_g, base_ebit_margin, base_revenue, tax_rate, reinvest_rate, wacc_dec, g_terminal, net_cash_debt, shares_outstanding)
            if abs(mid_val - current_price) < 0.01:
                break
            if mid_val < current_price:
                a = mid_g
            else:
                b = mid_g
        implied_growth = round(((a + b) / 2.0) * 100.0, 2)

    # 7. Solve for Implied Operating Margin (Bound: 0.0% to 60.0%)
    low_m, high_m = 0.01, 0.60
    val_low_m, _, _, _, _ = _calculate_per_share_value(base_growth, low_m, base_revenue, tax_rate, reinvest_rate, wacc_dec, g_terminal, net_cash_debt, shares_outstanding)
    val_high_m, _, _, _, _ = _calculate_per_share_value(base_growth, high_m, base_revenue, tax_rate, reinvest_rate, wacc_dec, g_terminal, net_cash_debt, shares_outstanding)

    implied_margin: Optional[float] = None
    if current_price >= val_low_m and current_price <= val_high_m:
        a_m, b_m = low_m, high_m
        for _ in range(50):
            mid_m = (a_m + b_m) / 2.0
            mid_val, _, _, _, _ = _calculate_per_share_value(base_growth, mid_m, base_revenue, tax_rate, reinvest_rate, wacc_dec, g_terminal, net_cash_debt, shares_outstanding)
            if abs(mid_val - current_price) < 0.01:
                break
            if mid_val < current_price:
                a_m = mid_m
            else:
                b_m = mid_m
        implied_margin = round(((a_m + b_m) / 2.0) * 100.0, 2)

    # 8. Single Source of Truth Valuation Verdict (Policy Methodology v2.1.0)
    if upside_12m > 15.0:
        verdict: Literal["overvalued", "fairly_valued", "undervalued", "unavailable"] = "undervalued"
    elif upside_12m < -15.0:
        verdict = "overvalued"
    else:
        verdict = "fairly_valued"

    fixed_params = {
        "wacc_pct": round(wacc_pct, 2),
        "cost_of_equity_pct": round(ke, 2),
        "terminal_growth_pct": round(g_terminal * 100.0, 2),
        "tax_rate_pct": round(tax_rate * 100.0, 2),
        "base_revenue_usd": base_revenue,
        "base_ebit_margin_pct": round(base_ebit_margin * 100.0, 2),
        "net_cash_debt_usd": net_cash_debt,
    }

    raw_wacc = round(wacc_pct, 2)
    effective_wacc = round(wacc_dec * 100.0, 2)
    wacc_adj_reason = "WACC clamped to minimum 4.0% floor" if (wacc_pct / 100.0) < 0.04 else None

    result = ReverseDCFResult(
        explicit_forecast_5y=projections,
        sum_pv_5y_fcf=round(sum_pv_fcf, 2),
        terminal_value_undiscounted=round(tv_pv * ((1.0 + wacc_dec) ** 5), 2),
        terminal_value_pv=round(tv_pv, 2),
        enterprise_value=round(sum_pv_fcf + tv_pv, 2),
        net_cash_debt=round(net_cash_debt, 2),
        equity_value=round(sum_pv_fcf + tv_pv + net_cash_debt, 2),
        intrinsic_value_today=round(intrinsic_today, 2),
        target_price_12m=target_price_12m,
        upside_12m_pct=upside_12m,
        market_implied_growth_pct=implied_growth,
        market_implied_margin_pct=implied_margin,
        solver_status=solver_status,
        fixed_parameters=fixed_params,
        valuation_horizon_months=12,
        status="available",
        is_eligible=True,
        valuation_verdict=verdict,
        is_actionable=is_actionable,
        actionability_reason=actionability_reason,
        raw_wacc_pct=raw_wacc,
        effective_wacc_pct=effective_wacc,
        wacc_adjustment_reason=wacc_adj_reason,
        reported_ebit_margin_pct=round(base_ebit_margin_pct, 2) if base_ebit_margin_pct is not None else None,
        ebit_margin_fiscal_period=ebit_margin_fiscal_period,
        ebit_margin_period_type=ebit_margin_period_type,
        ebit_margin_source_tier=ebit_margin_source_tier,
        target_price_exit_multiple_12m=target_price_exit_12m,
        exit_multiple_used=default_exit_mult,
        intrinsic_value_exit_multiple=round(intrinsic_exit, 2) if intrinsic_exit is not None else None,
        consensus_target_price=consensus_target_price,
        consensus_target_high=consensus_target_high,
        consensus_target_low=consensus_target_low,
        analyst_count=analyst_count,
        base_revenue=base_revenue,
        base_revenue_period_type=base_revenue_period_type,
    )
    return result, flags
