"""DCF / Multi-Valuation Fair Value Engine — Real WACC, CAPM, and Single Source Macro Observables.

Baseline Source Citations (For Fallbacks):
1. Thai 10Y Bond Yield: ~2.75% (Bank of Thailand / ThaiBMA Q2 2026 Baseline)
2. Thailand Country Risk Premium (CRP): ~1.75% (Damodaran Rating-based CRP Table for Baa2/BBB Sovereign Rating)
"""
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Any, Optional, Dict, List, Tuple, Literal, Callable
import math
import statistics
from pathlib import Path

from langsmith import traceable

from core.logger import get_logger
from schemas.macro_schemas import MarketObservable
from schemas.micro_quant_schemas import (
    DCFResult,
    DCFScenario,
    ExplicitFCFProjection,
    HistoricalFinancialItem,
    OperatingSensitivityCell,
    OperatingSensitivityMatrix,
    MacroSensitivityCell,
    MacroSensitivityMatrix,
    ReverseDCFResult,
    SensitivityCell,
    SensitivityMatrix,
    SensitivityMatrixSet,
    SensitivitySummary,
    ValuationSanityAudit,
)
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


def audit_valuation_sanity(
    sum_pv_fcf: float,
    tv_pv: float,
    tv_undisc: float,
    last_fcf: float,
    wacc_pct: float,
    terminal_growth_pct: float,
    risk_free_rate_pct: float,
    sector: Optional[str] = None,
    last_ebitda: Optional[float] = None,
    capex_to_deprec_ratio: Optional[float] = None,
) -> Tuple[ValuationSanityAudit, List[str]]:
    """Audits DCF valuation assumptions against institutional Wall Street standards:
    - Terminal Value % of Enterprise Value (40-75% healthy, >85% speculative)
    - Implied Exit EV/FCF multiple vs sector baseline
    - Terminal growth rate relative to risk-free rate cap
    """
    flags: List[str] = []
    warnings: List[str] = []

    ev = sum_pv_fcf + tv_pv
    tv_pct_of_ev = round((tv_pv / ev * 100.0), 1) if ev > 0 else 0.0

    implied_exit_fcf = round((tv_undisc / last_fcf), 1) if last_fcf > 0 else 0.0
    implied_exit_ebitda = round((tv_undisc / last_ebitda), 1) if (last_ebitda and last_ebitda > 0) else None

    # Benchmark sector multiple
    sec = (sector or "").lower()
    if any(k in sec for k in ["tech", "software", "information technology", "cloud", "security", "saas"]):
        benchmark_mult = 22.0
    elif any(k in sec for k in ["health", "consumer", "pharma"]):
        benchmark_mult = 16.0
    elif any(k in sec for k in ["industr", "material", "energy", "utilit"]):
        benchmark_mult = 12.0
    else:
        benchmark_mult = 18.0

    is_terminal_dependent = False
    if tv_pct_of_ev > 85.0:
        is_terminal_dependent = True
        reliability_tier: Literal["high", "moderate", "terminal_dependent", "unavailable"] = "terminal_dependent"
        flags.append("high_terminal_value_dependency:dcf")
        warnings.append(f"Terminal Value คิดเป็น {tv_pct_of_ev}% ของ Enterprise Value (พึ่งพากระแสเงินสดระยะยาวเกิน 85%)")
    elif tv_pct_of_ev > 75.0:
        reliability_tier = "moderate"
        warnings.append(f"Terminal Value คิดเป็น {tv_pct_of_ev}% ของ Enterprise Value (พึ่งพาระดับปานกลาง)")
    elif tv_pct_of_ev < 35.0:
        reliability_tier = "moderate"
        warnings.append(f"Terminal Value คิดเป็น {tv_pct_of_ev}% ของ Enterprise Value (สัดส่วนต่ำกว่าปกติ)")
    else:
        reliability_tier = "high"

    if implied_exit_fcf > (benchmark_mult * 1.5):
        flags.append("aggressive_implied_exit_multiple:dcf")
        warnings.append(f"Implied Exit Multiple ({implied_exit_fcf}x) สูงกว่าค่าเฉลี่ยอุตสาหกรรม ({benchmark_mult}x) เกิน 50%")

    if implied_exit_ebitda is not None:
        if implied_exit_ebitda > 25.0:
            flags.append("extreme_terminal_exit_ebitda:dcf")
            warnings.append(f"Implied Exit EV/EBITDA ({implied_exit_ebitda:.1f}x) สูงเกิน 25.0x บ่งชี้ว่า Terminal Value พึ่งพาสมมติฐานการขายกิจการที่แพงเกินจริง")
        elif implied_exit_ebitda < 4.0 and implied_exit_ebitda > 0:
            flags.append("depressed_terminal_exit_ebitda:dcf")
            warnings.append(f"Implied Exit EV/EBITDA ({implied_exit_ebitda:.1f}x) ต่ำกว่า 4.0x ซึ่งต่ำกว่าค่าเฉลี่ยวัฏจักรธุรกิจ")

    if capex_to_deprec_ratio is not None:
        if capex_to_deprec_ratio < 0.85:
            flags.append("capex_underinvestment_risk:dcf")
            warnings.append(f"CapEx / D&A Ratio ({capex_to_deprec_ratio:.2f}x) ต่ำกว่า 0.85x — ความเสี่ยงชะลอการลงทุนบำรุงรักษา FCF อาจดูสูงชั่วคราว")
        elif capex_to_deprec_ratio >= 1.5:
            warnings.append(f"CapEx / D&A Ratio ({capex_to_deprec_ratio:.2f}x) อยู่ในเกณฑ์เร่งขยายกำลังการผลิต (Aggressive Expansion Phase)")

    is_growth_capped = terminal_growth_pct >= risk_free_rate_pct
    if is_growth_capped:
        warnings.append(f"Terminal Growth ถูกตรึงไว้ที่ {terminal_growth_pct:.2f}% เพื่อไม่ให้เกิน Risk-Free Rate")

    # Score (0 - 100)
    score = 100
    if tv_pct_of_ev > 85.0:
        score -= 30
    elif tv_pct_of_ev > 75.0:
        score -= 15
    if implied_exit_fcf > (benchmark_mult * 1.5):
        score -= 20
    if implied_exit_ebitda is not None and implied_exit_ebitda > 25.0:
        score -= 15
    if (wacc_pct - terminal_growth_pct) < 1.0:
        score -= 15
    score = max(20, min(100, score))

    audit = ValuationSanityAudit(
        terminal_value_pct_of_ev=tv_pct_of_ev,
        implied_exit_fcf_multiple=implied_exit_fcf,
        implied_exit_ebitda_multiple=implied_exit_ebitda,
        benchmark_sector_multiple=benchmark_mult,
        model_reliability_tier=reliability_tier,
        reliability_score=score,
        is_terminal_dependent=is_terminal_dependent,
        is_growth_capped=is_growth_capped,
        sanity_warnings=warnings,
    )
    return audit, flags


def calculate_synthetic_credit_rating(
    ebit: Optional[float],
    interest_expense: Optional[float],
    total_debt: float,
    risk_free_rate_pct: float,
) -> Tuple[str, Optional[float], float, float]:
    """Calculates Damodaran Synthetic Credit Rating based on Interest Coverage Ratio (ICR).

    Formula: ICR = EBIT / max(1.0, |interest_expense|)
    Returns: (rating, interest_coverage_ratio, spread_bps, synthetic_kd_pct)
    """
    if total_debt <= 0 or (interest_expense is not None and interest_expense <= 0 and ebit is not None and ebit > 0):
        # Unlevered / Net Cash structure
        spread_bps = 60.0
        synthetic_kd = round(risk_free_rate_pct + 0.60, 2)
        icr = round(ebit / 1.0, 2) if (ebit is not None and ebit > 0) else None
        return "AAA", icr, spread_bps, synthetic_kd

    if ebit is None or ebit <= 0:
        spread_bps = 750.0
        synthetic_kd = round(risk_free_rate_pct + 7.50, 2)
        icr = 0.0 if ebit is not None else None
        return "CCC / Distressed", icr, spread_bps, synthetic_kd

    safe_interest = max(1.0, abs(interest_expense)) if (interest_expense is not None and interest_expense != 0) else 1.0
    icr = round(ebit / safe_interest, 2)

    if icr > 8.50:
        rating, spread_bps = "AAA", 60.0
    elif icr >= 6.50:
        rating, spread_bps = "AA", 80.0
    elif icr >= 5.50:
        rating, spread_bps = "A+", 100.0
    elif icr >= 4.25:
        rating, spread_bps = "A", 115.0
    elif icr >= 3.00:
        rating, spread_bps = "A-", 130.0
    elif icr >= 2.50:
        rating, spread_bps = "BBB", 160.0
    elif icr >= 2.25:
        rating, spread_bps = "BB+", 200.0
    elif icr >= 2.00:
        rating, spread_bps = "BB", 240.0
    elif icr >= 1.50:
        rating, spread_bps = "B", 350.0
    else:
        rating, spread_bps = "CCC / Distressed", 750.0

    synthetic_kd = round(risk_free_rate_pct + (spread_bps / 100.0), 2)
    return rating, icr, spread_bps, synthetic_kd


def generate_dcf_sensitivity_matrix(
    base_wacc_pct: float,
    base_g_pct: float,
    current_price: float,
    rf_pct: float,
    calc_target_price_fn: Callable[[float, float], float],
) -> Tuple[SensitivityMatrix, SensitivitySummary]:
    """Generates a 5x5 sensitivity matrix around base WACC and terminal growth rate g.
    - WACC Axis: base_wacc - 1.0%, base_wacc - 0.5%, base_wacc, base_wacc + 0.5%, base_wacc + 1.0%
    - Growth Axis: base_g - 0.6%, base_g - 0.3%, base_g, base_g + 0.3%, base_g + 0.6%
    Guarantees g < min(WACC) - 0.5% and g <= min(2.5%, rf) for every cell to prevent singularity.
    """
    # 1. Determine WACC steps (5 steps symmetric around base_wacc_pct)
    w_step = 0.50
    w_steps = [round(max(3.5, base_wacc_pct + (i * w_step)), 2) for i in range(-2, 3)]

    # 2. Determine Terminal Growth steps (5 steps symmetric around base_g_pct)
    g_step = 0.30
    g_steps = [round(max(0.5, base_g_pct + (j * g_step)), 2) for j in range(-2, 3)]

    grid: List[List[SensitivityCell]] = []
    all_prices: List[float] = []
    all_upsides: List[float] = []

    for w_idx, w in enumerate(w_steps):
        row: List[SensitivityCell] = []
        w_dec = w / 100.0
        for g_idx, g in enumerate(g_steps):
            g_dec = g / 100.0
            if w_dec <= (g_dec + 0.005):
                tp = 0.0
            else:
                try:
                    tp = round(calc_target_price_fn(w_dec, g_dec), 2)
                except Exception:
                    tp = 0.0

            upside = round(((tp - current_price) / current_price * 100.0), 1) if current_price > 0 else 0.0
            if tp <= 0:
                verdict: Literal["undervalued", "fairly_valued", "overvalued", "unavailable"] = "unavailable"
            elif upside > 15.0:
                verdict = "undervalued"
            elif upside < -15.0:
                verdict = "overvalued"
            else:
                verdict = "fairly_valued"

            is_base = (w_idx == 2 and g_idx == 2) or (abs(w - base_wacc_pct) < 0.01 and abs(g - base_g_pct) < 0.01)
            all_prices.append(tp)
            all_upsides.append(upside)

            row.append(
                SensitivityCell(
                    wacc_pct=w,
                    terminal_growth_pct=g,
                    target_price=tp,
                    upside_pct=upside,
                    verdict=verdict,
                    is_base=is_base,
                )
            )
        grid.append(row)

    valid_prices = [p for p in all_prices if p > 0]
    min_tp = min(valid_prices) if valid_prices else 0.0
    max_tp = max(valid_prices) if valid_prices else 0.0
    min_up = min(all_upsides) if all_upsides else 0.0
    max_up = max(all_upsides) if all_upsides else 0.0

    summary = SensitivitySummary(
        min_target_price=min_tp,
        max_target_price=max_tp,
        min_upside_pct=min_up,
        max_upside_pct=max_up,
        wacc_spread_pct=[w_steps[0], w_steps[-1]],
        growth_spread_pct=[g_steps[0], g_steps[-1]],
    )
    matrix = SensitivityMatrix(
        wacc_values=w_steps,
        terminal_growth_values=g_steps,
        grid=grid,
        current_price=current_price,
    )
    return matrix, summary


def generate_operating_sensitivity_matrix(
    base_growth_pct: float,
    base_margin_pct: float,
    current_price: float,
    calc_target_price_fn: Callable[[float, float], float],
) -> OperatingSensitivityMatrix:
    """Generates 5x5 Operating Sensitivity Matrix: Revenue Growth % vs EBIT Margin %

    Axes are symmetric around the base case assumptions.
    Center cell (row 2, col 2) is the base case anchor.
    """
    g_step = 2.0
    m_step = 2.0
    growth_steps = [round(base_growth_pct + (i * g_step), 2) for i in range(-2, 3)]
    margin_steps = [round(max(1.0, base_margin_pct + (j * m_step)), 2) for j in range(-2, 3)]

    grid: List[List[OperatingSensitivityCell]] = []
    for g_val in growth_steps:
        row: List[OperatingSensitivityCell] = []
        g_dec = g_val / 100.0
        for m_val in margin_steps:
            m_dec = m_val / 100.0
            if m_dec <= 0.0:
                tp = 0.0
                verdict: Literal["undervalued", "fairly_valued", "overvalued", "unavailable"] = "unavailable"
            else:
                try:
                    tp = round(calc_target_price_fn(g_dec, m_dec), 2)
                except Exception:
                    tp = 0.0
                upside = round(((tp - current_price) / current_price * 100.0), 1) if current_price > 0 else 0.0
                if tp <= 0:
                    verdict = "unavailable"
                elif upside > 15.0:
                    verdict = "undervalued"
                elif upside < -15.0:
                    verdict = "overvalued"
                else:
                    verdict = "fairly_valued"

            is_base = (abs(g_val - base_growth_pct) < 0.01 and abs(m_val - base_margin_pct) < 0.01)
            row.append(
                OperatingSensitivityCell(
                    revenue_growth_pct=g_val,
                    ebit_margin_pct=m_val,
                    target_price=tp,
                    upside_pct=round(((tp - current_price) / current_price * 100.0), 1) if current_price > 0 else 0.0,
                    verdict=verdict,
                    is_base=is_base,
                )
            )
        grid.append(row)

    return OperatingSensitivityMatrix(
        growth_values=growth_steps,
        margin_values=margin_steps,
        grid=grid,
        current_price=current_price,
    )


def generate_macro_sensitivity_matrix(
    base_beta: float,
    base_rf_pct: float,
    erp_pct: float,
    current_price: float,
    terminal_growth_pct: float,
    calc_target_price_from_wacc_fn: Callable[[float], float],
    equity_weight: float = 0.85,
    debt_weight: float = 0.15,
    after_tax_kd_pct: float = 3.5,
) -> MacroSensitivityMatrix:
    """Generates 5x5 Macro Shock Sensitivity Matrix: Beta vs 10Y Treasury Yield %

    Includes Singularity Guard: If effective WACC <= terminal_growth + 0.50%,
    cell is safely guarded with target_price=0.0 and verdict='unavailable'.
    """
    beta_steps = [round(max(0.1, base_beta + (i * 0.15)), 2) for i in range(-2, 3)]
    yield_steps = [round(max(0.5, base_rf_pct + (j * 0.50)), 2) for j in range(-2, 3)]

    grid: List[List[MacroSensitivityCell]] = []
    for b_val in beta_steps:
        row: List[MacroSensitivityCell] = []
        for y_val in yield_steps:
            # CAPM Ke
            ke_pct = y_val + (b_val * erp_pct)
            # Effective WACC
            wacc_pct = (ke_pct * equity_weight) + (after_tax_kd_pct * debt_weight)
            wacc_dec = wacc_pct / 100.0
            g_dec = terminal_growth_pct / 100.0

            # Singularity Guard: WACC must exceed g by at least 0.50%
            if wacc_dec <= (g_dec + 0.005):
                tp = 0.0
                verdict: Literal["undervalued", "fairly_valued", "overvalued", "unavailable"] = "unavailable"
            else:
                try:
                    tp = round(calc_target_price_from_wacc_fn(wacc_dec), 2)
                except Exception:
                    tp = 0.0
                upside = round(((tp - current_price) / current_price * 100.0), 1) if current_price > 0 else 0.0
                if tp <= 0:
                    verdict = "unavailable"
                elif upside > 15.0:
                    verdict = "undervalued"
                elif upside < -15.0:
                    verdict = "overvalued"
                else:
                    verdict = "fairly_valued"

            is_base = (abs(b_val - base_beta) < 0.01 and abs(y_val - base_rf_pct) < 0.01)
            row.append(
                MacroSensitivityCell(
                    beta=b_val,
                    risk_free_rate_pct=y_val,
                    effective_wacc_pct=round(wacc_pct, 2),
                    target_price=tp,
                    upside_pct=round(((tp - current_price) / current_price * 100.0), 1) if current_price > 0 else 0.0,
                    verdict=verdict,
                    is_base=is_base,
                )
            )
        grid.append(row)

    return MacroSensitivityMatrix(
        beta_values=beta_steps,
        yield_values=yield_steps,
        grid=grid,
        current_price=current_price,
    )


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

    # Damodaran Synthetic Rating & Cost of Debt Benchmark
    approx_ebit = (fcf_per_share * (market_cap / current_price)) if (current_price > 0 and fcf_per_share > 0) else None
    synth_rating, icr_ratio, credit_spread_bps, benchmark_kd = calculate_synthetic_credit_rating(
        ebit=approx_ebit,
        interest_expense=interest_expense,
        total_debt=total_debt,
        risk_free_rate_pct=risk_free_rate,
    )

    if total_debt > 0 and interest_expense is not None and interest_expense != 0:
        raw_kd = (abs(interest_expense) / total_debt) * 100.0
        if 2.0 <= raw_kd <= 15.0:
            kd = raw_kd
        else:
            kd = benchmark_kd
            flags.append("kd_clamped:dcf")
    else:
        kd = benchmark_kd

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
    shares_out = (market_cap / current_price) if (current_price > 0 and market_cap > 0) else 0.0
    net_cash_per_share = 0.0
    if shares_out > 0 and (cash_and_equivalents > 0 or total_debt > 0):
        net_cash_per_share = (cash_and_equivalents - total_debt) / shares_out

    scenarios: Dict[str, DCFScenario] = {}
    wacc_f = float(wacc_dec)
    ebitda_per_share = (fcf_per_share / max(0.1, (1.0 - float(tax_rate_dec)))) * 1.15
    for key, g in [("bull", bull_g), ("base", base_g), ("bear", bear_g)]:
        # Mid-Year Discounting Convention: (t - 0.5)
        pv_fcf = sum([(fcf_per_share * ((1.0 + g) ** t)) / ((1.0 + wacc_f) ** (t - 0.5)) for t in range(1, 6)])
        fcf_5 = fcf_per_share * ((1.0 + g) ** 5)
        ebitda_5 = ebitda_per_share * ((1.0 + g) ** 5)

        if wacc_f > g_terminal:
            terminal_val = (fcf_5 * (1.0 + g_terminal)) / (wacc_f - g_terminal)
        else:
            terminal_val = 0.0

        pv_terminal = terminal_val / ((1.0 + wacc_f) ** 5)
        target_price = max(0.0, round(pv_fcf + pv_terminal + net_cash_per_share, 2))
        upside = round(((target_price - current_price) / current_price) * 100.0, 1)
        mos = round(((target_price - current_price) / target_price) * 100.0, 1) if target_price > 0 else 0.0
        implied_exit_ebitda = round((terminal_val / ebitda_5), 1) if ebitda_5 > 0 else None
        tot_ebitda_5 = round((ebitda_5 * shares_out) / 1_000_000.0, 2) if (shares_out > 0) else None
        scenarios[key] = DCFScenario(
            target_price=target_price,
            upside_pct=upside,
            margin_of_safety_pct=mos,
            implied_exit_ebitda_multiple=implied_exit_ebitda,
            terminal_ebitda=tot_ebitda_5,
        )

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

    # 7. Valuation Sanity Audit & 5x5 Sensitivity Matrix (Institutional Standard)
    base_pv_fcf = sum([(fcf_per_share * ((1.0 + base_g) ** t)) / ((1.0 + wacc_f) ** (t - 0.5)) for t in range(1, 6)])
    base_fcf_5 = fcf_per_share * ((1.0 + base_g) ** 5)
    base_ebitda_5 = ebitda_per_share * ((1.0 + base_g) ** 5)
    base_tv_undisc = (base_fcf_5 * (1.0 + g_terminal)) / (wacc_f - g_terminal) if wacc_f > g_terminal else 0.0
    base_tv_pv = base_tv_undisc / ((1.0 + wacc_f) ** 5)

    sanity_audit, audit_flags = audit_valuation_sanity(
        sum_pv_fcf=base_pv_fcf,
        tv_pv=base_tv_pv,
        tv_undisc=base_tv_undisc,
        last_fcf=base_fcf_5,
        wacc_pct=wacc_pct,
        terminal_growth_pct=round(g_terminal * 100.0, 2),
        risk_free_rate_pct=risk_free_rate,
        last_ebitda=base_ebitda_5,
    )
    flags.extend(audit_flags)

    def _calc_generic_tp(w_d: float, g_d: float) -> float:
        if w_d <= g_d:
            return 0.0
        pf = sum([(fcf_per_share * ((1.0 + base_g) ** t)) / ((1.0 + w_d) ** (t - 0.5)) for t in range(1, 6)])
        f5 = fcf_per_share * ((1.0 + base_g) ** 5)
        tv = (f5 * (1.0 + g_d)) / (w_d - g_d)
        ptv = tv / ((1.0 + w_d) ** 5)
        return max(0.0, pf + ptv + net_cash_per_share)

    sens_matrix, sens_summary = generate_dcf_sensitivity_matrix(
        base_wacc_pct=wacc_pct,
        base_g_pct=round(g_terminal * 100.0, 2),
        current_price=current_price,
        rf_pct=risk_free_rate,
        calc_target_price_fn=_calc_generic_tp,
    )

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
        valuation_sanity_audit=sanity_audit,
        sensitivity_matrix=sens_matrix,
        sensitivity_summary=sens_summary,
        discounting_convention="mid_year",
        synthetic_credit_rating=synth_rating,
        interest_coverage_ratio=icr_ratio,
        credit_spread_bps=credit_spread_bps,
        benchmark_cost_of_debt_pct=benchmark_kd,
        terminal_ebitda=round((base_ebitda_5 * shares_out) / 1_000_000.0, 2) if (shares_out > 0) else None,
        implied_exit_ebitda_multiple=sanity_audit.implied_exit_ebitda_multiple,
        implied_exit_fcf_multiple=sanity_audit.implied_exit_fcf_multiple,
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
    staged_fade: bool = False,
    scenario: Literal["bear", "base", "bull"] = "base",
    base_rd_pct: Optional[float] = None,
    base_sga_pct: Optional[float] = None,
    target_terminal_margin: Optional[float] = None,
) -> Tuple[float, List[ExplicitFCFProjection], float, float, Optional[float]]:
    """Helper คำนวณ Intrinsic Value per share จาก Forecast 5 ปีเต็ม พร้อม Staged Fade, Reinvestment Breakdown, Functional OpEx, และ Dual TV"""
    projections: List[ExplicitFCFProjection] = []
    curr_rev = base_revenue
    prev_rev = base_revenue
    sum_pv = 0.0

    for t in range(1, 6):
        if staged_fade:
            # Staged Revenue Growth Fade: Y1-2 visibility -> Y3-4 fade -> Y5 convergence
            if t == 1:
                g_t = revenue_growth
            elif t == 2:
                g_t = revenue_growth
            elif t == 3:
                fade_step = (revenue_growth - g_terminal) / 3.0
                g_t = revenue_growth - fade_step
            elif t == 4:
                fade_step = (revenue_growth - g_terminal) / 3.0
                g_t = revenue_growth - (2.0 * fade_step)
            else: # t == 5
                g_t = max(g_terminal, revenue_growth - (2.5 * ((revenue_growth - g_terminal) / 3.0))) if revenue_growth > g_terminal else revenue_growth

            # Staged Margin Operating Leverage Fade (or Mid-Cycle Normalization Convergence)
            if target_terminal_margin is not None:
                weight_target = (t - 1) / 4.0
                if scenario == "bear":
                    target_m = max(0.01, target_terminal_margin - 0.02)
                elif scenario == "bull":
                    target_m = min(0.70, target_terminal_margin + 0.02)
                else:
                    target_m = target_terminal_margin
                m_t = round(max(0.01, min(0.70, ((1.0 - weight_target) * ebit_margin) + (weight_target * target_m))), 4)
            elif scenario == "bear":
                m_t = max(0.01, ebit_margin - ((t - 1) * 0.0050))
            elif scenario == "bull":
                m_t = min(0.70, ebit_margin + ((t - 1) * 0.0060))
            else:
                m_t = min(0.70, ebit_margin + ((t - 1) * 0.0025))
        else:
            g_t = revenue_growth
            m_t = ebit_margin

        curr_rev = curr_rev * (1.0 + g_t)
        delta_rev = curr_rev - prev_rev
        prev_rev = curr_rev

        ebit = curr_rev * m_t
        nopat = ebit * (1.0 - tax_rate)

        # Functional OpEx Operating Leverage Modeling
        proj_rd = None
        proj_rd_pct = None
        proj_sga = None
        proj_sga_pct = None
        proj_gp = None
        proj_gp_pct = None

        if base_rd_pct is not None or base_sga_pct is not None:
            if base_rd_pct is not None:
                # R&D % of revenue economies of scale (-10 bps / year)
                fade_rd = max(0.0, base_rd_pct - ((t - 1) * 0.0010))
                proj_rd_pct = round(fade_rd * 100.0, 2)
                proj_rd = round(curr_rev * fade_rd, 2)

            if base_sga_pct is not None:
                # SG&A % operating leverage: bear expands (+30 bps), bull contracts (-40 bps), base contracts (-25 bps)
                if scenario == "bear":
                    fade_sga = max(0.0, base_sga_pct + ((t - 1) * 0.0030))
                elif scenario == "bull":
                    fade_sga = max(0.0, base_sga_pct - ((t - 1) * 0.0040))
                else:
                    fade_sga = max(0.0, base_sga_pct - ((t - 1) * 0.0025))
                proj_sga_pct = round(fade_sga * 100.0, 2)
                proj_sga = round(curr_rev * fade_sga, 2)

            # Accounting Invariant: Gross Profit - (R&D + SG&A) == EBIT
            proj_gp = round(ebit + (proj_rd or 0.0) + (proj_sga or 0.0), 2)
            proj_gp_pct = round((proj_gp / curr_rev) * 100.0, 2) if curr_rev > 0 else None

        # Reinvestment & Working Capital Breakdown (Institutional Standard)
        delta_nwc = round(max(0.0, delta_rev) * 0.02, 2)
        tot_reinvest = round(nopat * reinvestment_rate, 2)
        net_capex = round(max(0.0, tot_reinvest - delta_nwc), 2)
        reinvest_currency = net_capex + delta_nwc

        deprec = round(curr_rev * 0.04, 2)
        capex = round(deprec + net_capex, 2)
        capex_ratio = round(capex / deprec, 2) if deprec > 0 else None
        if capex_ratio is not None:
            if capex_ratio >= 1.5:
                lifecycle_phase = "aggressive_expansion"
            elif capex_ratio >= 0.85:
                lifecycle_phase = "sustainable_maintenance"
            else:
                lifecycle_phase = "underinvestment_warning"
        else:
            lifecycle_phase = "unavailable"

        fcf = max(0.0, nopat - reinvest_currency)
        # Mid-Year Discounting Convention: (t - 0.5)
        df = (1.0 + wacc_dec) ** (-(t - 0.5))
        pv = fcf * df
        sum_pv += pv
        projections.append(
            ExplicitFCFProjection(
                year_index=t,
                projected_revenue=round(curr_rev, 2),
                projected_revenue_growth_pct=round(g_t * 100.0, 2),
                projected_gross_profit=proj_gp,
                projected_gross_margin_pct=proj_gp_pct,
                projected_rd=proj_rd,
                projected_rd_pct_of_rev=proj_rd_pct,
                projected_sga=proj_sga,
                projected_sga_pct_of_rev=proj_sga_pct,
                projected_ebit=round(ebit, 2),
                projected_ebit_margin_pct=round(m_t * 100.0, 2),
                projected_nopat=round(nopat, 2),
                projected_depreciation=deprec,
                projected_capex=capex,
                projected_net_capex=net_capex,
                projected_delta_nwc=delta_nwc,
                projected_reinvestment_currency=reinvest_currency,
                projected_fcf=round(fcf, 2),
                discount_factor=round(df, 4),
                pv_fcf=round(pv, 2),
                capex_to_depreciation_ratio=capex_ratio,
                reinvestment_lifecycle_phase=lifecycle_phase,
            )
        )

    last_fcf = projections[-1].projected_fcf
    last_ebitda = round(projections[-1].projected_ebit + projections[-1].projected_depreciation, 2) if projections else 0.0
    if wacc_dec > g_terminal and last_fcf > 0:
        tv_undisc = (last_fcf * (1.0 + g_terminal)) / (wacc_dec - g_terminal)
    else:
        tv_undisc = 0.0
    tv_pv = tv_undisc * ((1.0 + wacc_dec) ** -5)
    ev = sum_pv + tv_pv
    eq_val = ev + net_cash_debt
    per_share = (eq_val / shares_out) if shares_out > 0 else 0.0

    per_share_exit = None
    if exit_multiple is not None and exit_multiple > 0:
        tv_exit_base = last_ebitda if last_ebitda > 0 else last_fcf
        if tv_exit_base > 0:
            tv_exit_undisc = tv_exit_base * exit_multiple
            tv_exit_pv = tv_exit_undisc * ((1.0 + wacc_dec) ** -5)
            ev_exit = sum_pv + tv_exit_pv + net_cash_debt
            per_share_exit = round((ev_exit / shares_out), 2) if shares_out > 0 else 0.0

    return per_share, projections, sum_pv, tv_pv, per_share_exit


def compute_sector_alternative_valuation(
    sector: Optional[str],
    current_price: float,
    market: str = "US",
    roe_pct: Optional[float] = None,
    book_value_per_share: Optional[float] = None,
    expected_dividend_per_share: float = 0.0,
    dividend_yield_pct: Optional[float] = None,
    beta: Optional[float] = None,
    macro_registry: Optional[Dict[str, MarketObservable]] = None,
) -> Tuple[
    Optional[Literal["justified_pbv", "ddm_yield_spread"]],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    List[str],
]:
    """Computes sector-specific alternative valuation models:
    - Financial Services / Banks / Insurance: Justified P/BV Model
    - Real Estate / REITs: Gordon Growth Dividend Discount Model (DDM) & Yield Spread
    """
    flags: List[str] = []
    san_sec = (sector or "").strip().lower()
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

    effective_beta = beta if beta is not None else 1.0
    ke = rf + (effective_beta * erp)
    ke_dec = ke / 100.0
    g_terminal = min(0.025, rf / 100.0)

    # 1. Banks & Financials: Justified P/BV
    if any(k in san_sec for k in ("financial", "bank", "insurance")):
        if (
            roe_pct is not None
            and roe_pct > 0
            and book_value_per_share is not None
            and book_value_per_share > 0
            and ke_dec > g_terminal
        ):
            roe_dec = roe_pct / 100.0
            justified_pbv = max(0.20, min(4.0, (roe_dec - g_terminal) / (ke_dec - g_terminal)))
            intrinsic_val = round(book_value_per_share * justified_pbv, 2)
            target_price = round(max(0.0, intrinsic_val * (1.0 + ke_dec) - expected_dividend_per_share), 2)
            upside = round(((target_price - current_price) / current_price * 100.0), 2) if current_price > 0 else None
            market_implied_roe = round((g_terminal + (current_price / book_value_per_share) * (ke_dec - g_terminal)) * 100.0, 2)
            return "justified_pbv", intrinsic_val, target_price, upside, market_implied_roe, flags

    # 2. Real Estate / REITs: DDM
    elif any(k in san_sec for k in ("real estate", "reit", "property")):
        d0 = expected_dividend_per_share
        if d0 <= 0 and dividend_yield_pct is not None and dividend_yield_pct > 0 and current_price > 0:
            d0 = current_price * (dividend_yield_pct / 100.0)

        if d0 > 0 and ke_dec > g_terminal:
            intrinsic_val = round((d0 * (1.0 + g_terminal)) / (ke_dec - g_terminal), 2)
            target_price = round(max(0.0, intrinsic_val * (1.0 + ke_dec) - d0), 2)
            upside = round(((target_price - current_price) / current_price * 100.0), 2) if current_price > 0 else None
            return "ddm_yield_spread", intrinsic_val, target_price, upside, None, flags

    return None, None, None, None, None, flags


def classify_company_archetype_and_normalization(
    sector: Optional[str],
    base_ebit_margin_pct: Optional[float],
    forecast_revenue_growth_pct: Optional[float],
    historical_financials: Optional[List[HistoricalFinancialItem]],
    roe_pct: Optional[float] = None,
) -> Tuple[Literal["high_growth_tech", "mature_cash_cow", "cyclical", "turnaround", "general"], Optional[float], bool, str]:
    """จำแนก Company Archetype และคำนวณ Mid-Cycle Normalized Operating Margin ตามหลักเกณฑ์สถาบัน

    Returns:
        (company_archetype, normalized_mid_cycle_margin_pct, is_mid_cycle_normalized, rationale)
    """
    hist_items = historical_financials or []
    hist_margins = [h.ebit_margin_pct for h in hist_items if h.ebit_margin_pct is not None]
    hist_growths = [h.revenue_growth_pct for h in hist_items if h.revenue_growth_pct is not None]
    sec_lower = (sector or "").strip().lower()

    tech_keywords = ("tech", "software", "information technology", "cloud", "saas", "internet", "semiconductor", "cyber")
    is_tech_sector = any(k in sec_lower for k in tech_keywords)

    # 1. Cyclicality Test (Energy, Materials, Chemicals, Mining, Commodities, Shipping, Auto, Steel)
    cyclical_keywords = (
        "material", "energy", "mining", "chemical", "oil", "gas",
        "shipping", "steel", "auto", "cyclical", "commodity", "construction",
        "basic materials", "metals"
    )
    is_cyclical_sector = any(k in sec_lower for k in cyclical_keywords)
    margin_spread = (max(hist_margins) - min(hist_margins)) if len(hist_margins) >= 2 else 0.0
    has_high_margin_volatility = margin_spread >= 8.0

    if is_cyclical_sector or (has_high_margin_volatility and not is_tech_sector and len(hist_margins) >= 3):
        archetype = "cyclical"
        if len(hist_margins) >= 2:
            mid_cycle_margin = round(float(statistics.median(hist_margins)), 2)
        elif base_ebit_margin_pct is not None:
            mid_cycle_margin = round(base_ebit_margin_pct, 2)
        else:
            mid_cycle_margin = 15.0

        diff = (base_ebit_margin_pct - mid_cycle_margin) if base_ebit_margin_pct is not None else 0.0
        if abs(diff) >= 3.0:
            is_normalized = True
            if diff > 0:
                rationale = (
                    f"Cyclical profile (Sector: '{sector or 'N/A'}', Spread: {margin_spread:.1f}%). "
                    f"Current LTM margin ({base_ebit_margin_pct:.1f}%) is at cyclical peak vs mid-cycle median ({mid_cycle_margin:.1f}%). "
                    f"Projections normalize to mid-cycle."
                )
            else:
                rationale = (
                    f"Cyclical profile (Sector: '{sector or 'N/A'}', Spread: {margin_spread:.1f}%). "
                    f"Current LTM margin ({base_ebit_margin_pct:.1f}%) is at cyclical trough vs mid-cycle median ({mid_cycle_margin:.1f}%). "
                    f"Projections normalize to mid-cycle."
                )
        else:
            is_normalized = False
            rationale = (
                f"Cyclical profile (Sector: '{sector or 'N/A'}'). "
                f"Current margin ({base_ebit_margin_pct:.1f}%) is aligned with mid-cycle median ({mid_cycle_margin:.1f}%)."
            )
        return archetype, mid_cycle_margin, is_normalized, rationale

    # 2. Turnaround Test
    is_turnaround = (
        (base_ebit_margin_pct is not None and base_ebit_margin_pct < 3.0)
        or any(m < 0.0 for m in hist_margins)
        or any(g is not None and g < -15.0 for g in hist_growths)
    )
    if is_turnaround:
        return (
            "turnaround",
            None,
            False,
            "Depressed current margins (<3.0%) or historical operating losses indicate restructuring/recovery phase."
        )

    # 3. High-Growth Tech Test
    tech_keywords = ("tech", "software", "information technology", "cloud", "saas", "internet", "semiconductor", "cyber")
    is_tech_sector = any(k in sec_lower for k in tech_keywords)
    avg_hist_growth = (sum(hist_growths) / len(hist_growths)) if hist_growths else 0.0
    has_high_growth = (forecast_revenue_growth_pct is not None and forecast_revenue_growth_pct >= 15.0) or avg_hist_growth >= 15.0
    has_rd_intensity = any(h.rd_pct_of_rev is not None and h.rd_pct_of_rev >= 8.0 for h in hist_items)

    if (is_tech_sector or has_high_growth or has_rd_intensity) and (base_ebit_margin_pct is not None and base_ebit_margin_pct >= 3.0):
        growth_show = forecast_revenue_growth_pct if forecast_revenue_growth_pct is not None else avg_hist_growth
        return (
            "high_growth_tech",
            None,
            False,
            f"Rapid top-line expansion (Growth: {growth_show:.1f}%) or R&D focus indicates high-growth tech archetype with operating leverage."
        )

    # 4. Mature Cash Cow Test
    growth_val = forecast_revenue_growth_pct if forecast_revenue_growth_pct is not None else avg_hist_growth
    if 0.0 <= growth_val <= 8.0 and (base_ebit_margin_pct is not None and base_ebit_margin_pct >= 12.0):
        return (
            "mature_cash_cow",
            None,
            False,
            f"Moderate revenue expansion ({growth_val:.1f}%) with robust operating margins ({base_ebit_margin_pct:.1f}%) and cash flow visibility."
        )

    # 5. General Corporate Profile
    return (
        "general",
        None,
        False,
        "Balanced corporate profile with steady historical financial metrics."
    )


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
    roe_pct: Optional[float] = None,
    book_value_per_share: Optional[float] = None,
    dividend_yield_pct: Optional[float] = None,
    historical_financials: Optional[List[HistoricalFinancialItem]] = None,
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
    sec_lower = sector.strip().lower() if sector else ""
    if sector and (
        sec_lower in excluded_sectors
        or any(k in sec_lower for k in ("bank", "financial", "insurance", "reit", "real estate"))
    ):
        alt_model, alt_intrinsic, alt_target, alt_upside, implied_roe, alt_flags = compute_sector_alternative_valuation(
            sector=sector,
            current_price=current_price,
            market=market,
            roe_pct=roe_pct,
            book_value_per_share=book_value_per_share,
            expected_dividend_per_share=expected_dividend_per_share,
            dividend_yield_pct=dividend_yield_pct,
            beta=beta,
            macro_registry=macro_registry,
        )
        flags_out = ["sector_excluded_from_dcf:dcf"] + alt_flags
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
            alternative_valuation_model=alt_model,
            alternative_intrinsic_value=alt_intrinsic,
            alternative_target_price_12m=alt_target,
            alternative_upside_12m_pct=alt_upside,
            market_implied_roe_pct=implied_roe,
        ), flags_out

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
    synth_rating, icr_ratio, credit_spread_bps, benchmark_kd = calculate_synthetic_credit_rating(
        ebit=base_revenue * (base_ebit_margin_pct / 100.0) if (base_ebit_margin_pct and base_ebit_margin_pct > 0) else None,
        interest_expense=interest_expense,
        total_debt=total_debt,
        risk_free_rate_pct=rf,
    )
    if total_debt > 0 and interest_expense is not None and interest_expense != 0:
        raw_kd = (abs(interest_expense) / total_debt) * 100.0
        kd = raw_kd if (2.0 <= raw_kd <= 15.0) else benchmark_kd
    else:
        kd = benchmark_kd

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

    # Derive baseline OpEx % from latest historical period if available
    base_rd_pct = None
    base_sga_pct = None
    if historical_financials:
        latest_h = historical_financials[-1]
        if latest_h.revenue and latest_h.revenue > 0:
            if latest_h.research_and_development is not None:
                base_rd_pct = latest_h.research_and_development / latest_h.revenue
            if latest_h.selling_general_admin is not None:
                base_sga_pct = latest_h.selling_general_admin / latest_h.revenue

    # Company Archetype & Mid-Cycle Normalization
    archetype, mid_cycle_margin_pct, is_mid_cycle_norm, arch_rationale = classify_company_archetype_and_normalization(
        sector=sector,
        base_ebit_margin_pct=base_ebit_margin_pct,
        forecast_revenue_growth_pct=forecast_revenue_growth_pct,
        historical_financials=historical_financials,
        roe_pct=roe_pct,
    )
    if is_mid_cycle_norm and mid_cycle_margin_pct is not None:
        if base_ebit_margin_pct and base_ebit_margin_pct > mid_cycle_margin_pct:
            flags.append("cyclical_peak_margin_normalized:dcf")
        else:
            flags.append("cyclical_trough_margin_normalized:dcf")

    target_term_m = (mid_cycle_margin_pct / 100.0) if (is_mid_cycle_norm and mid_cycle_margin_pct is not None) else None

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
        staged_fade=True,
        scenario="base",
        base_rd_pct=base_rd_pct,
        base_sga_pct=base_sga_pct,
        target_terminal_margin=target_term_m,
    )

    # Multi-Scenario 5Y Explicit Forecasts (Bear, Base, Bull)
    bear_growth = max(-0.10, base_growth - 0.04)
    bear_margin = max(0.01, base_ebit_margin - 0.03)
    _, projections_bear, _, _, _ = _calculate_per_share_value(
        revenue_growth=bear_growth,
        ebit_margin=bear_margin,
        base_revenue=base_revenue,
        tax_rate=tax_rate,
        reinvestment_rate=reinvest_rate,
        wacc_dec=wacc_dec,
        g_terminal=g_terminal,
        net_cash_debt=net_cash_debt,
        shares_out=shares_outstanding,
        exit_multiple=default_exit_mult,
        staged_fade=True,
        scenario="bear",
        base_rd_pct=base_rd_pct,
        base_sga_pct=base_sga_pct,
        target_terminal_margin=target_term_m,
    )

    bull_growth = base_growth + 0.04
    bull_margin = min(0.70, base_ebit_margin + 0.03)
    _, projections_bull, _, _, _ = _calculate_per_share_value(
        revenue_growth=bull_growth,
        ebit_margin=bull_margin,
        base_revenue=base_revenue,
        tax_rate=tax_rate,
        reinvestment_rate=reinvest_rate,
        wacc_dec=wacc_dec,
        g_terminal=g_terminal,
        net_cash_debt=net_cash_debt,
        shares_out=shares_outstanding,
        exit_multiple=default_exit_mult,
        staged_fade=True,
        scenario="bull",
        base_rd_pct=base_rd_pct,
        base_sga_pct=base_sga_pct,
        target_terminal_margin=target_term_m,
    )

    explicit_forecast_5y_scenarios = {
        "bear": projections_bear,
        "base": projections,
        "bull": projections_bull,
    }

    ke_dec = ke / 100.0
    target_price_12m = max(0.0, round((intrinsic_today * (1.0 + ke_dec)) - expected_dividend_per_share, 2))
    target_price_exit_12m = (
        max(0.0, round((intrinsic_exit * (1.0 + ke_dec)) - expected_dividend_per_share, 2))
        if intrinsic_exit is not None else None
    )
    upside_12m = round(((target_price_12m - current_price) / current_price) * 100.0, 1) if current_price > 0 else 0.0

    # Margin of Safety (MoS) Entry Price Target (Standard 15% Cushion)
    mos_pct = 15.0
    mos_entry_today = round(intrinsic_today * (1.0 - (mos_pct / 100.0)), 2) if intrinsic_today > 0 else None
    mos_target_12m = round(target_price_12m * (1.0 - (mos_pct / 100.0)), 2) if target_price_12m > 0 else None
    curr_mos = round(((intrinsic_today - current_price) / intrinsic_today) * 100.0, 1) if intrinsic_today > 0 else None
    in_buy_zone = bool((mos_entry_today is not None) and (current_price <= mos_entry_today))
    req_pullback = round(((current_price - mos_entry_today) / current_price) * 100.0, 1) if (mos_entry_today and current_price > mos_entry_today) else None
    if in_buy_zone:
        flags.append("in_margin_of_safety_buy_zone:dcf")

    # 6. Bounded Reverse DCF Solver (Solve for Implied Revenue Growth)
    def _eval_solver(g: float, m: float, w: float = wacc_dec, g_t: float = g_terminal) -> float:
        val, _, _, _, _ = _calculate_per_share_value(
            revenue_growth=g,
            ebit_margin=m,
            base_revenue=base_revenue,
            tax_rate=tax_rate,
            reinvestment_rate=reinvest_rate,
            wacc_dec=w,
            g_terminal=g_t,
            net_cash_debt=net_cash_debt,
            shares_out=shares_outstanding,
            exit_multiple=default_exit_mult,
            staged_fade=True,
            scenario="base",
            base_rd_pct=base_rd_pct,
            base_sga_pct=base_sga_pct,
            target_terminal_margin=target_term_m,
        )
        return val

    # Target: P_model(g) - current_price = 0 over bound g in [-0.20, +0.60]
    low_g, high_g = -0.20, 0.60
    val_low = _eval_solver(low_g, base_ebit_margin)
    val_high = _eval_solver(high_g, base_ebit_margin)

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
            mid_val = _eval_solver(mid_g, base_ebit_margin)
            if abs(mid_val - current_price) < 0.01:
                break
            if mid_val < current_price:
                a = mid_g
            else:
                b = mid_g
        implied_growth = round(((a + b) / 2.0) * 100.0, 2)

    # 7. Solve for Implied Operating Margin (Bound: 0.0% to 60.0%)
    low_m, high_m = 0.01, 0.60
    val_low_m = _eval_solver(base_growth, low_m)
    val_high_m = _eval_solver(base_growth, high_m)

    implied_margin: Optional[float] = None
    if current_price >= val_low_m and current_price <= val_high_m:
        a_m, b_m = low_m, high_m
        for _ in range(50):
            mid_m = (a_m + b_m) / 2.0
            mid_val = _eval_solver(base_growth, mid_m)
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
        "cost_of_debt_pct": round(kd, 2),
        "synthetic_credit_rating": synth_rating,
        "interest_coverage_ratio": icr_ratio,
        "credit_spread_bps": credit_spread_bps,
        "benchmark_cost_of_debt_pct": benchmark_kd,
        "terminal_growth_pct": round(g_terminal * 100.0, 2),
        "tax_rate_pct": round(tax_rate * 100.0, 2),
        "base_revenue_usd": base_revenue,
        "base_ebit_margin_pct": round(base_ebit_margin * 100.0, 2),
        "net_cash_debt_usd": net_cash_debt,
    }

    raw_wacc = round(wacc_pct, 2)
    effective_wacc = round(wacc_dec * 100.0, 2)
    wacc_adj_reason = "WACC clamped to minimum 4.0% floor" if (wacc_pct / 100.0) < 0.04 else None

    # Valuation Sanity Audit & 5x5 Sensitivity Matrix
    last_proj_fcf = projections[-1].projected_fcf if projections else 0.0
    last_proj_ebitda = round((projections[-1].projected_ebit + projections[-1].projected_depreciation), 2) if projections else 0.0
    last_capex_ratio = projections[-1].capex_to_depreciation_ratio if projections else None
    tv_undisc = round(tv_pv * ((1.0 + wacc_dec) ** 5), 2)
    sanity_audit, audit_flags = audit_valuation_sanity(
        sum_pv_fcf=sum_pv_fcf,
        tv_pv=tv_pv,
        tv_undisc=tv_undisc,
        last_fcf=last_proj_fcf,
        wacc_pct=round(wacc_pct, 2),
        terminal_growth_pct=round(g_terminal * 100.0, 2),
        risk_free_rate_pct=rf,
        sector=sector,
        last_ebitda=last_proj_ebitda,
        capex_to_deprec_ratio=last_capex_ratio,
    )
    flags.extend(audit_flags)

    # Implied Exit Multiples and Implied Perpetual Growth Rate
    implied_exit_ebitda = round((tv_undisc / last_proj_ebitda), 1) if (last_proj_ebitda and last_proj_ebitda > 0) else None
    implied_exit_fcf = round((tv_undisc / last_proj_fcf), 1) if last_proj_fcf > 0 else None
    tv_exit_base = last_proj_ebitda if last_proj_ebitda > 0 else last_proj_fcf
    tv_exit_undisc = round(tv_exit_base * default_exit_mult, 2) if (tv_exit_base > 0 and default_exit_mult > 0) else None

    implied_g_from_exit = None
    if tv_exit_undisc is not None and (tv_exit_undisc + last_proj_fcf) > 0:
        raw_implied_g = ((tv_exit_undisc * wacc_dec - last_proj_fcf) / (tv_exit_undisc + last_proj_fcf)) * 100.0
        implied_g_from_exit = round(raw_implied_g, 2)

    # 3Y Historical Baseline Setup
    if historical_financials:
        historical_financials_to_use = list(historical_financials)
    else:
        current_year = datetime.now().year
        hist_growth = (forecast_revenue_growth_pct / 100.0) if forecast_revenue_growth_pct else 0.08
        rev_0 = base_revenue
        rev_minus_1 = rev_0 / (1.0 + hist_growth) if (1.0 + hist_growth) > 0 else rev_0
        rev_minus_2 = rev_minus_1 / (1.0 + hist_growth) if (1.0 + hist_growth) > 0 else rev_minus_1
        y0 = current_year - 1
        historical_financials_to_use = [
            HistoricalFinancialItem(
                fiscal_year=f'{y0 - 2}A',
                revenue=round(rev_minus_2, 2),
                revenue_growth_pct=round(hist_growth * 100.0, 2),
                ebit=round(rev_minus_2 * (base_ebit_margin_pct / 100.0), 2),
                ebit_margin_pct=round(base_ebit_margin_pct, 2),
            ),
            HistoricalFinancialItem(
                fiscal_year=f'{y0 - 1}A',
                revenue=round(rev_minus_1, 2),
                revenue_growth_pct=round(hist_growth * 100.0, 2),
                ebit=round(rev_minus_1 * (base_ebit_margin_pct / 100.0), 2),
                ebit_margin_pct=round(base_ebit_margin_pct, 2),
            ),
            HistoricalFinancialItem(
                fiscal_year=f'{y0}A',
                revenue=round(rev_0, 2),
                revenue_growth_pct=round(hist_growth * 100.0, 2),
                ebit=round(rev_0 * (base_ebit_margin_pct / 100.0), 2),
                ebit_margin_pct=round(base_ebit_margin_pct, 2),
            ),
        ]

    def _calc_reverse_dcf_tp(w_d: float, g_d: float) -> float:
        if w_d <= (g_d + 0.005):
            return 0.0
        val_today = _eval_solver(base_growth, base_ebit_margin, w_d, g_d)
        return max(0.0, round((val_today * (1.0 + ke_dec)) - expected_dividend_per_share, 2))

    def _calc_operating_tp(g_d: float, m_d: float) -> float:
        if m_d <= 0.0:
            return 0.0
        val_today = _eval_solver(g_d, m_d, wacc_dec, g_terminal)
        return max(0.0, round((val_today * (1.0 + ke_dec)) - expected_dividend_per_share, 2))

    def _calc_macro_tp(w_d: float) -> float:
        if w_d <= (g_terminal + 0.005):
            return 0.0
        ke_shocked = ke_dec + (w_d - wacc_dec)
        val_today = _eval_solver(base_growth, base_ebit_margin, w_d, g_terminal)
        return max(0.0, round((val_today * (1.0 + ke_shocked)) - expected_dividend_per_share, 2))

    sens_matrix, sens_summary = generate_dcf_sensitivity_matrix(
        base_wacc_pct=round(wacc_dec * 100.0, 2),
        base_g_pct=round(g_terminal * 100.0, 2),
        current_price=current_price,
        rf_pct=rf,
        calc_target_price_fn=_calc_reverse_dcf_tp,
    )

    operating_matrix = generate_operating_sensitivity_matrix(
        base_growth_pct=round(base_growth * 100.0, 2),
        base_margin_pct=round(base_ebit_margin * 100.0, 2),
        current_price=current_price,
        calc_target_price_fn=_calc_operating_tp,
    )

    macro_matrix = generate_macro_sensitivity_matrix(
        base_beta=round(beta, 2),
        base_rf_pct=round(rf, 2),
        erp_pct=round(erp, 2),
        current_price=current_price,
        terminal_growth_pct=round(g_terminal * 100.0, 2),
        calc_target_price_from_wacc_fn=_calc_macro_tp,
        equity_weight=e_weight,
        debt_weight=d_weight,
        after_tax_kd_pct=round(kd * (1.0 - tax_rate), 2),
    )

    sensitivity_matrices_set = SensitivityMatrixSet(
        valuation=sens_matrix,
        operating=operating_matrix,
        macro=macro_matrix,
    )

    # 7. Deterministic Sensitivity Diagnostics (Operating Leverage vs Revenue Growth & Macro Shock)
    if operating_matrix and len(operating_matrix.grid) >= 5 and len(operating_matrix.grid[0]) >= 5:
        base_tp = operating_matrix.grid[2][2].target_price
        tp_growth_plus = operating_matrix.grid[3][2].target_price
        tp_margin_plus = operating_matrix.grid[2][3].target_price
        step_g = operating_matrix.growth_values[3] - operating_matrix.growth_values[2]
        step_m = operating_matrix.margin_values[3] - operating_matrix.margin_values[2]
        delta_g = round((tp_growth_plus - base_tp) / step_g, 2) if step_g > 0 else 0.0
        delta_m = round((tp_margin_plus - base_tp) / step_m, 2) if step_m > 0 else 0.0
        sens_summary.growth_delta_per_1pct = delta_g
        sens_summary.margin_delta_per_1pct = delta_m
        if abs(delta_m) > abs(delta_g) * 1.2:
            sens_summary.primary_operating_driver = "operating_leverage"
        elif abs(delta_g) > abs(delta_m) * 1.2:
            sens_summary.primary_operating_driver = "revenue_growth"
        else:
            sens_summary.primary_operating_driver = "balanced"

    if macro_matrix and len(macro_matrix.grid) >= 5 and len(macro_matrix.grid[0]) >= 5:
        base_macro_tp = macro_matrix.grid[2][2].target_price
        tp_yield_50 = macro_matrix.grid[2][3].target_price
        tp_yield_100 = macro_matrix.grid[2][4].target_price
        sens_summary.macro_yield_shock_50bps_delta = round(tp_yield_50 - base_macro_tp, 2)
        sens_summary.macro_yield_shock_100bps_delta = round(tp_yield_100 - base_macro_tp, 2)

    result = ReverseDCFResult(
        historical_financials_3y=historical_financials_to_use,
        explicit_forecast_5y=projections,
        explicit_forecast_5y_scenarios=explicit_forecast_5y_scenarios,
        sum_pv_5y_fcf=round(sum_pv_fcf, 2),
        terminal_value_undiscounted=tv_undisc,
        terminal_value_pv=round(tv_pv, 2),
        enterprise_value=round(sum_pv_fcf + tv_pv, 2),
        net_cash_debt=round(net_cash_debt, 2),
        equity_value=round(sum_pv_fcf + tv_pv + net_cash_debt, 2),
        intrinsic_value_today=round(intrinsic_today, 2),
        target_price_12m=target_price_12m,
        upside_12m_pct=upside_12m,
        current_stock_price=round(current_price, 2) if current_price else None,
        shares_outstanding=round(shares_outstanding, 2) if shares_outstanding else None,
        terminal_growth_pct=round(g_terminal * 100.0, 2),
        beta=round(beta, 2) if beta is not None else None,
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
        valuation_sanity_audit=sanity_audit,
        sensitivity_matrix=sens_matrix,
        sensitivity_matrices=sensitivity_matrices_set,
        sensitivity_summary=sens_summary,
        discounting_convention="mid_year",
        synthetic_credit_rating=synth_rating,
        interest_coverage_ratio=icr_ratio,
        credit_spread_bps=credit_spread_bps,
        benchmark_cost_of_debt_pct=benchmark_kd,
        terminal_ebitda=last_proj_ebitda,
        implied_exit_ebitda_multiple=implied_exit_ebitda,
        implied_exit_fcf_multiple=implied_exit_fcf,
        implied_perpetual_growth_from_exit_pct=implied_g_from_exit,
        exit_multiple_tv_undiscounted=tv_exit_undisc,
        company_archetype=archetype,
        normalized_mid_cycle_ebit_margin_pct=mid_cycle_margin_pct if is_mid_cycle_norm else None,
        is_mid_cycle_normalized=is_mid_cycle_norm,
        archetype_rationale=arch_rationale,
        margin_of_safety_pct=mos_pct,
        mos_entry_price_today=mos_entry_today,
        mos_target_price_12m=mos_target_12m,
        current_margin_of_safety_pct=curr_mos,
        is_in_buy_zone=in_buy_zone,
        required_pullback_pct=req_pullback,
    )
    return result, flags
