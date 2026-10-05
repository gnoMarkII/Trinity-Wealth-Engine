import math
from typing import Optional, Any
from datetime import datetime
import re
from .parsers import _parse_markdown_with_context, _parse_float_from_str
from schemas.macro_schemas import EconomicState

_CANONICAL_REGION_MAPPING: dict[str, str] = {
    "united states": "United States",
    "usa": "United States",
    "us": "United States",
    "thailand": "Thailand",
    "thai": "Thailand",
    "th": "Thailand",
    "euro area": "Euro Area",
    "europe": "Euro Area",
    "eu": "Euro Area",
    "china": "China",
    "cn": "China",
    "japan": "Japan",
    "jp": "Japan",
    "india": "India",
    "in": "India",
    "latin america": "Latin America",
    "latam": "Latin America",
    "global": "Global",
}


def _get_global_risk_sentiment(global_md: str = "", observables: list[Any] | None = None) -> float:
    """Calculate global market risk sentiment from VIX volatility observable.

    Note: This measures market risk/volatility sentiment (calm vs panic),
    not subjective political or geopolitical headline opinions.
    VIX > 20: Elevated market volatility / fear (-1.0)
    VIX <= 20: Calm / stable volatility (+1.0)
    Default: 0.0 (neutral if no VIX observable available)
    """
    vix = None
    if observables:
        for o in observables:
            id_str = getattr(o, "observable_id", "").lower()
            ind_str = getattr(o, "indicator", "").lower()
            if "vix" in id_str or "vix" in ind_str:
                try:
                    vix = float(str(getattr(o, "value", "")).replace(",", ""))
                    break
                except (ValueError, TypeError):
                    pass
    if vix is None and global_md:
        rows = _parse_markdown_with_context(global_md)
        for r in rows:
            idx = r.get("ดัชนี", "").replace("**", "").strip()
            if "VIX" in idx:
                vix = _parse_float_from_str(r.get("ค่าล่าสุด", ""))
                break

    if vix is not None:
        return -1.0 if vix > 20.0 else 1.0
    return 0.0


def _get_global_geopolitics(global_md: str) -> float:
    """Legacy alias for _get_global_risk_sentiment."""
    return _get_global_risk_sentiment(global_md=global_md)


def _determine_economic_state(growth_score: Optional[float], inflation_score: Optional[float]) -> str:
    """Determine economic regime based on Growth and Inflation momentum scores.

    Fail-closed policy: If either growth or inflation score is missing (None),
    the state is strictly UNKNOWN.
    Growth > 0 => Expansion, Growth <= 0 => Contraction
    Inflation Score >= 0 => Low inflation, Inflation Score < 0 => High inflation
    """
    if growth_score is None or inflation_score is None:
        return EconomicState.UNKNOWN.value

    if growth_score > 0 and inflation_score >= 0:
        return EconomicState.GOLDILOCKS.value
    elif growth_score > 0 and inflation_score < 0:
        return EconomicState.REFLATION.value
    elif growth_score <= 0 and inflation_score < 0:
        return EconomicState.STAGFLATION.value
    else:
        return EconomicState.RECESSION.value


def _calculate_matrix_scores_from_markdown(country_md: str) -> dict:
    rows = _parse_markdown_with_context(country_md)
    regions_data = {}
    for r in rows:
        # Extract region name by stripping leading flag emoji / non-word chars (e.g. '🇹🇭 Thailand' -> 'Thailand')
        h1_raw = str(r.get("_H1", "Unknown")).strip()
        cleaned_h1 = re.sub(r"^[^\w]+", "", h1_raw).strip() or h1_raw
        region_name = _CANONICAL_REGION_MAPPING.get(cleaned_h1.lower(), cleaned_h1)
        if region_name not in regions_data:
            regions_data[region_name] = {}

        idx = r.get("ดัชนี", "").replace("**", "").strip()
        val = _parse_float_from_str(r.get("ค่าล่าสุด", ""))
        prev = _parse_float_from_str(r.get("ก่อนหน้า", ""))
        ma = _parse_float_from_str(r.get("MA ย้อนหลัง", ""))
        if val is not None:
            regions_data[region_name][idx] = {
                "val": val,
                "prev": prev,
                "ma": ma,
            }

    results = {}
    for region, data in regions_data.items():
        def get_metric(key_fragment: str) -> dict | None:
            for k, v in data.items():
                k_lower = k.lower()
                # Ignore mock or staticproxy metrics
                if "[mock]" in k_lower or "staticproxy" in k_lower or "[staticproxy]" in k_lower:
                    continue
                if key_fragment.lower() in k_lower:
                    return v
            return None

        # Helper for scoring momentum & MA
        def score_momentum(metric: dict, is_inverse: bool = False) -> Optional[float]:
            val = metric.get("val")
            ma = metric.get("ma")
            prev = metric.get("prev")
            if val is None or ma is None or prev is None:
                return None  # Insufficient history for momentum calculation

            score = 0.0
            if val > ma:
                score += 0.5 if not is_inverse else -0.5
            elif val < ma:
                score -= 0.5 if not is_inverse else -0.5

            if val > prev:
                score += 0.5 if not is_inverse else -0.5
            elif val < prev:
                score -= 0.5 if not is_inverse else -0.5
            return score

        data_gaps = []

        if region == "Thailand":
            # 1. Thai Growth Pillar (Real GDP YoY required, corroborated by OIE MPI YoY if present)
            gdp = get_metric("Real GDP") or get_metric("GDP")
            mpi = get_metric("Manufacturing Production") or get_metric("Industrial Production") or get_metric("MPI")

            m_gdp = score_momentum(gdp) if gdp else None
            m_mpi = score_momentum(mpi) if mpi else None

            if m_gdp is not None and m_mpi is not None:
                final_growth = (m_gdp * 0.4) + (m_mpi * 0.6)
            elif m_gdp is not None:
                final_growth = m_gdp
            else:
                final_growth = None
                data_gaps.append("Thailand Growth (สศช. Real GDP)")

            # 2. Thai Inflation Pillar (Headline CPI YoY required, Core CPI diagnostic)
            cpi = get_metric("CPI")
            m_cpi = score_momentum(cpi, is_inverse=True) if cpi else None
            final_inflation = m_cpi
            if final_inflation is None:
                data_gaps.append("Thailand Inflation (สนค. Headline CPI)")

            # 3. Thai Monetary Pillar (Real-rate proxy: Policy Rate - Headline CPI YoY)
            policy_rate = get_metric("Policy Rate") or get_metric("Repo Rate")
            spread = get_metric("10Y-2Y") or get_metric("Yield Spread")

            if policy_rate is not None:
                if cpi is not None and cpi["val"] < 30.0:
                    real_rate = policy_rate["val"] - cpi["val"]
                    if real_rate > 1.0:
                        monetary_score = -1.0
                    elif real_rate <= 0.0:
                        monetary_score = 1.0
                    else:
                        monetary_score = 0.0
                else:
                    monetary_score = None
                    data_gaps.append(
                        f"Thailand Monetary: อัตราดอกเบี้ยนโยบายพร้อมใช้งาน ({policy_rate['val']:.2f}%) แต่ขาด Headline CPI เพื่อคำนวณอัตราดอกเบี้ยจริง (Real Policy Rate)"
                    )
            else:
                monetary_score = None
                data_gaps.append("Thailand Monetary (ธปท. Policy Rate / Yield Curve)")

        else:
            # 1. Growth Pillar (United States / Euro Area / General)
            gdp = get_metric("Real GDP")
            indpro = get_metric("Industrial Production")
            retail = get_metric("Retail Sales")
            unemp = get_metric("Unemployment Rate")

            growth_pmi_score = score_momentum(indpro) if indpro is not None else None
            lag_score = 0.0
            lag_count = 0
            if gdp is not None:
                s = score_momentum(gdp)
                if s is not None:
                    lag_score += s
                    lag_count += 1
            if retail is not None:
                s = score_momentum(retail)
                if s is not None:
                    lag_score += s
                    lag_count += 1
            if unemp is not None:
                s = score_momentum(unemp, is_inverse=True)
                if s is not None:
                    lag_score += s
                    lag_count += 1

            if growth_pmi_score is None and lag_count == 0:
                final_growth = None
            elif growth_pmi_score is not None and lag_count == 0:
                final_growth = growth_pmi_score
            elif growth_pmi_score is None and lag_count > 0:
                final_growth = lag_score / lag_count
            else:
                final_growth = (growth_pmi_score * 0.6) + ((lag_score / lag_count) * 0.4)

            # 2. Inflation Pillar
            cpi = get_metric("CPI")
            pce = get_metric("Core PCE") or get_metric("PCE")

            inf_score = 0.0
            inf_count = 0
            for inf_metric in [cpi, pce]:
                if inf_metric is not None:
                    s = score_momentum(inf_metric, is_inverse=True)
                    if s is not None:
                        inf_score += s
                        inf_count += 1
            if inf_count == 0:
                final_inflation = None
            else:
                final_inflation = inf_score / inf_count

            # 3. Monetary Pillar
            fed = get_metric("Fed Funds Rate") or get_metric("Policy Rate")
            spread = get_metric("10Y-2Y") or get_metric("10-Year Minus 2-Year")

            monetary_score = 0.0
            mon_count = 0
            inf_candidate = pce or cpi
            if fed is not None and inf_candidate is not None:
                if inf_candidate["val"] < 30.0:
                    real_rate = fed["val"] - inf_candidate["val"]
                    if real_rate > 1.0:
                        monetary_score -= 1.0
                    elif real_rate <= 0.0:
                        monetary_score += 1.0
                    mon_count += 1
            if spread is not None:
                if spread["val"] < 0:
                    monetary_score -= 1.0
                else:
                    monetary_score += 1.0
                mon_count += 1

            if mon_count == 0:
                monetary_score = None
            else:
                monetary_score = monetary_score / mon_count

            if final_growth is None:
                data_gaps.append(f"{region} Growth (GDP/IP/Retail/Unemployment)")
            if final_inflation is None:
                data_gaps.append(f"{region} Inflation (CPI/PCE)")
            if monetary_score is None:
                data_gaps.append(f"{region} Monetary (Policy Rate/Yield Curve)")

        coverage = round((3 - len(data_gaps)) / 3.0, 2)
        state = _determine_economic_state(final_growth, final_inflation)

        if state == EconomicState.UNKNOWN.value or coverage < 0.5:
            confidence = 0.0
        else:
            confidence = round(0.5 + (coverage * 0.4), 2)

        results[region] = {
            "growth": final_growth,
            "inflation": final_inflation,
            "monetary": monetary_score,
            "state": state,
            "confidence": confidence,
            "coverage": coverage,
            "data_gaps": data_gaps,
        }
    return results


def _is_eligible_observable(o: Any) -> bool:
    """Check if an observable is eligible for deterministic macro scoring.

    Contradictory or unverified states (is_valid=False, status='mock', 'missing', 'stale', 'unverified', 'blocked', 'mismatched_date')
    are strictly excluded. Only status == 'verified' is eligible.
    """
    if not getattr(o, "is_valid", False):
        return False
    status = getattr(o, "status", "verified")
    if status != "verified":
        return False
    return True


def _calculate_matrix_scores_from_observables(observables: list[Any]) -> dict[str, dict]:
    """Calculate macro regime matrix scores strictly from validated observables.

    Observables that are invalid (is_valid=False, status='mock', 'missing', 'stale', 'unverified')
    are strictly excluded from scoring and coverage calculations.
    """
    valid_obs = [o for o in observables if _is_eligible_observable(o)]

    # Group valid observables by region (canonicalized)
    by_region: dict[str, list[Any]] = {}
    for o in valid_obs:
        raw_reg = getattr(o, "region", "Global") or "Global"
        reg = _CANONICAL_REGION_MAPPING.get(raw_reg.strip().lower(), raw_reg.strip())
        by_region.setdefault(reg, []).append(o)

    # Always ensure canonical regions are evaluated even if data is completely missing (fail-closed)
    for canonical in ["United States", "Thailand", "Euro Area"]:
        if canonical not in by_region:
            by_region[canonical] = []

    results = {}
    for region, obs_list in by_region.items():
        if region in ("Global", "Unknown"):
            continue

        def find_metric(*key_fragments: str) -> dict | None:
            for o in obs_list:
                ind = getattr(o, "indicator", "").lower()
                obs_id = getattr(o, "observable_id", "").lower()
                text = f"{ind} {obs_id}"
                if all(kf.lower() in text for kf in key_fragments):
                    val = _parse_float_from_str(str(getattr(o, "value", "")))
                    if val is None:
                        continue
                    meta = getattr(o, "metadata", {}) or {}
                    prev = meta.get("prev")
                    ma = meta.get("ma")
                    return {
                        "val": val,
                        "prev": prev,
                        "ma": ma,
                    }
            return None

        def score_momentum(metric: dict, is_inverse: bool = False) -> Optional[float]:
            val = metric.get("val")
            ma = metric.get("ma")
            prev = metric.get("prev")
            if val is None or ma is None or prev is None:
                return None  # Insufficient history for momentum calculation

            score = 0.0
            if val > ma:
                score += 0.5 if not is_inverse else -0.5
            elif val < ma:
                score -= 0.5 if not is_inverse else -0.5

            if val > prev:
                score += 0.5 if not is_inverse else -0.5
            elif val < prev:
                score -= 0.5 if not is_inverse else -0.5
            return score

        data_gaps = []
        fiscal_health = None
        thai_yield_curve = None

        if region == "Thailand":
            # 1. Thai Growth Pillar (Real GDP YoY required, corroborated by OIE MPI YoY if present)
            gdp = find_metric("real gdp") or find_metric("gdp")
            mpi = find_metric("manufacturing production") or find_metric("mpi") or find_metric("indpro")

            m_gdp = score_momentum(gdp) if gdp else None
            m_mpi = score_momentum(mpi) if mpi else None

            if m_gdp is not None and m_mpi is not None:
                final_growth = (m_gdp * 0.4) + (m_mpi * 0.6)
            elif m_gdp is not None:
                final_growth = m_gdp
            else:
                final_growth = None
                data_gaps.append("Thailand Growth (สศช. Real GDP)")

            # 2. Thai Inflation Pillar (Headline CPI YoY required, Core CPI diagnostic)
            cpi = find_metric("cpi")
            m_cpi = score_momentum(cpi, is_inverse=True) if cpi else None
            final_inflation = m_cpi
            if final_inflation is None:
                data_gaps.append("Thailand Inflation (สนค. Headline CPI)")

            # 3. Thai Monetary Pillar (Real-rate proxy: Policy Rate - Headline CPI YoY)
            policy_rate = (
                find_metric("policy rate")
                or find_metric("policy_rate")
                or find_metric("repo rate")
            )
            spread = find_metric("10y", "2y") or find_metric("yield spread")
            y2_val = find_metric("2y gov bond yield") or find_metric("gov_yield_2y")
            y10_val = find_metric("10y gov bond yield") or find_metric("gov_yield_10y")
            thai_yield_curve = {
                "spread_10y_2y_bps": spread["val"] if spread else None,
                "yield_2y": y2_val["val"] if y2_val else None,
                "yield_10y": y10_val["val"] if y10_val else None,
                "is_available": spread is not None,
                "status": "verified" if spread else "blocked",
                "reason": "" if spread else "ThaiBMA authorized access required (separate access gate)",
            }

            if policy_rate is not None:
                if cpi is not None and cpi["val"] < 30.0:
                    real_rate = policy_rate["val"] - cpi["val"]
                    if real_rate > 1.0:
                        monetary_score = -1.0
                    elif real_rate <= 0.0:
                        monetary_score = 1.0
                    else:
                        monetary_score = 0.0
                else:
                    monetary_score = None
                    data_gaps.append(
                        f"Thailand Monetary: อัตราดอกเบี้ยนโยบายพร้อมใช้งาน ({policy_rate['val']:.2f}%) แต่ขาด Headline CPI เพื่อคำนวณอัตราดอกเบี้ยจริง (Real Policy Rate)"
                    )
            else:
                monetary_score = None
                data_gaps.append("Thailand Monetary (ธปท. Policy Rate / Yield Curve)")

            # 4. Thai Fiscal Health Diagnostic (MOF Public Debt to GDP vs 70% statutory limit)
            debt_to_gdp = find_metric("debt to gdp") or find_metric("debt_to_gdp")
            public_debt = find_metric("public debt") or find_metric("public_debt")
            fiscal_health = {
                "debt_to_gdp_pct": debt_to_gdp["val"] if debt_to_gdp else None,
                "public_debt_million_thb": public_debt["val"] if public_debt else None,
                "statutory_limit_pct": 70.0,
                "status": (
                    "within_ceiling"
                    if (debt_to_gdp and debt_to_gdp["val"] <= 70.0)
                    else ("exceeds_ceiling" if debt_to_gdp else "unknown")
                ),
            }

        else:
            # 1. Growth Pillar (United States / Euro Area / General)
            indpro = find_metric("industrial production") or find_metric("indpro") or find_metric("pmi")
            gdp = find_metric("real gdp") or find_metric("gdp")
            retail = find_metric("retail") or find_metric("rsafs")
            unemp = find_metric("unemployment")

            growth_pmi_score = score_momentum(indpro) if indpro is not None else None
            lag_score = 0.0
            lag_count = 0
            if gdp is not None:
                s = score_momentum(gdp)
                if s is not None:
                    lag_score += s
                    lag_count += 1
            if retail is not None:
                s = score_momentum(retail)
                if s is not None:
                    lag_score += s
                    lag_count += 1
            if unemp is not None:
                s = score_momentum(unemp, is_inverse=True)
                if s is not None:
                    lag_score += s
                    lag_count += 1

            if growth_pmi_score is None and lag_count == 0:
                final_growth = None
            elif growth_pmi_score is not None and lag_count == 0:
                final_growth = growth_pmi_score
            elif growth_pmi_score is None and lag_count > 0:
                final_growth = lag_score / lag_count
            else:
                final_growth = (growth_pmi_score * 0.6) + ((lag_score / lag_count) * 0.4)

            # 2. Inflation Pillar
            cpi = find_metric("cpi")
            pce = find_metric("core pce") or find_metric("pce")

            inf_score = 0.0
            inf_count = 0
            for inf_metric in [cpi, pce]:
                if inf_metric is not None:
                    s = score_momentum(inf_metric, is_inverse=True)
                    if s is not None:
                        inf_score += s
                        inf_count += 1
            if inf_count == 0:
                final_inflation = None
            else:
                final_inflation = inf_score / inf_count

            # 3. Monetary Pillar
            fed = (
                find_metric("fed funds")
                or find_metric("policy rate")
                or find_metric("policy_rate")
            )
            spread = (
                find_metric("10y", "2y")
                or find_metric("t10y2y")
                or find_metric("yield spread")
            )

            monetary_score = 0.0
            mon_count = 0
            inf_candidate = pce or cpi
            if fed is not None and inf_candidate is not None:
                if inf_candidate["val"] < 30.0:
                    real_rate = fed["val"] - inf_candidate["val"]
                    if real_rate > 1.0:
                        monetary_score -= 1.0
                    elif real_rate <= 0.0:
                        monetary_score += 1.0
                    mon_count += 1
            if spread is not None:
                if spread["val"] < 0:
                    monetary_score -= 1.0
                else:
                    monetary_score += 1.0
                mon_count += 1

            if mon_count == 0:
                monetary_score = None
            else:
                monetary_score = monetary_score / mon_count

            if final_growth is None:
                data_gaps.append(f"{region} Growth (GDP/IP/Retail/Unemployment)")
            if final_inflation is None:
                data_gaps.append(f"{region} Inflation (CPI/PCE)")
            if monetary_score is None:
                data_gaps.append(f"{region} Monetary (Policy Rate/Yield Curve)")

        coverage = round((3 - len(data_gaps)) / 3.0, 2)
        state = _determine_economic_state(final_growth, final_inflation)

        if state == EconomicState.UNKNOWN.value or coverage < 0.5:
            confidence = 0.0
        else:
            confidence = round(0.5 + (coverage * 0.4), 2)

        def _safe_float(v: Any) -> Optional[float]:
            if v is None:
                return None
            try:
                flt = float(v)
                return flt if math.isfinite(flt) else None
            except (ValueError, TypeError):
                return None

        result_entry: dict[str, Any] = {
            "growth": _safe_float(final_growth),
            "inflation": _safe_float(final_inflation),
            "monetary": _safe_float(monetary_score),
            "state": state,
            "confidence": _safe_float(confidence) or 0.0,
            "coverage": _safe_float(coverage) or 0.0,
            "data_gaps": data_gaps,
        }
        if fiscal_health is not None:
            result_entry["fiscal_health"] = fiscal_health
        if thai_yield_curve is not None:
            result_entry["thai_yield_curve"] = thai_yield_curve

        results[region] = result_entry
    return results


def _blend_optional(val1: Optional[float], val2: Optional[float], w1: float = 1.0, w2: float = 0.5) -> Optional[float]:
    if val1 is None and val2 is None:
        return None
    if val1 is not None and val2 is None:
        return val1
    if val1 is None and val2 is not None:
        return val2
    return ((val1 * w1) + (val2 * w2)) / (w1 + w2)


def _calculate_matrix_scores(
    country_md: str = "",
    regional_md: str = "",
    observables: list[Any] | None = None,
) -> dict:
    if observables is not None:
        return _calculate_matrix_scores_from_observables(observables)
    country_results = _calculate_matrix_scores_from_markdown(country_md)
    if not regional_md:
        return country_results

    regional_results = _calculate_matrix_scores_from_markdown(regional_md)
    if not regional_results:
        return country_results

    blended = dict(country_results)
    for region, regional in regional_results.items():
        if region not in blended:
            blended[region] = regional
            continue
        country = blended[region]
        growth = _blend_optional(country.get("growth"), regional.get("growth"), 1.0, 0.5)
        inflation = _blend_optional(country.get("inflation"), regional.get("inflation"), 1.0, 0.5)
        monetary = _blend_optional(country.get("monetary"), regional.get("monetary"), 1.0, 0.5)

        data_gaps = []
        if growth is None:
            data_gaps.append(f"{region} Growth (GDP/IP/Retail/Unemployment)")
        if inflation is None:
            data_gaps.append(f"{region} Inflation (CPI/PCE)")
        if monetary is None:
            data_gaps.append(f"{region} Monetary (Policy Rate/Yield Curve)")

        coverage = round((3 - len(data_gaps)) / 3.0, 2)
        state = _determine_economic_state(growth, inflation)
        confidence = 0.0 if (state == EconomicState.UNKNOWN.value or coverage < 0.5) else round(0.5 + (coverage * 0.4), 2)

        blended[region] = {
            "growth": growth,
            "inflation": inflation,
            "monetary": monetary,
            "state": state,
            "confidence": confidence,
            "coverage": coverage,
            "data_gaps": data_gaps,
        }
    return blended


def _format_trend(score: Optional[float]) -> str:
    if score is None:
        return "N/A"
    if score > 0:
        return f"{(score):.2f} ↗️"
    elif score < 0:
        return f"{(score):.2f} ↘️"
    else:
        return f"{(score):.2f} ➡️"


def _calculate_us_recession_risk(
    us_growth: Optional[float],
    us_monetary: Optional[float],
    risk_sentiment: float,
) -> Optional[float]:
    """Calculate US recession risk heuristic score from US Growth and Monetary conditions.

    Does NOT blend other countries (e.g. Thailand or Euro Area).
    Returns None if US Growth or Monetary conditions are missing.
    """
    if us_growth is None or us_monetary is None:
        return None
    raw = (-us_growth * 0.5) + (-us_monetary * 0.3) + (-risk_sentiment * 0.2)
    return max(0.0, min(1.0, (raw + 1.0) / 2.0))


def _calculate_recession_probability(matrices: dict, geo_score: float) -> float:
    """Legacy wrapper for backward compatibility: returns US recession risk or 0.5 default."""
    us = matrices.get("United States")
    if us and us.get("growth") is not None and us.get("monetary") is not None:
        risk = _calculate_us_recession_risk(us["growth"], us["monetary"], geo_score)
        return risk if risk is not None else 0.5
    return 0.5

