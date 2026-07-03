from .parsers import _parse_markdown_with_context, _parse_float_from_str
from datetime import datetime

def _get_global_geopolitics(global_md: str) -> float:
    rows = _parse_markdown_with_context(global_md)
    vix = None
    for r in rows:
        idx = r.get("ดัชนี", "").replace("**", "").strip()
        if "VIX" in idx:
            vix = _parse_float_from_str(r.get("ค่าล่าสุด", ""))

    if vix is not None:
        return -1.0 if vix > 20.0 else 1.0
    return 0.0

def _determine_economic_state(growth_score: float, inflation_score: float) -> str:
    # Growth > 0 => Expansion, Growth <= 0 => Contraction
    # Inflation Score >= 0 => Low inflation, Inflation Score < 0 => High inflation
    if growth_score > 0 and inflation_score >= 0:
        return "Goldilocks"
    elif growth_score > 0 and inflation_score < 0:
        return "Reflation"
    elif growth_score <= 0 and inflation_score < 0:
        return "Stagflation"
    else:
        return "Recession"

def _calculate_matrix_scores_from_markdown(country_md: str) -> dict:
    rows = _parse_markdown_with_context(country_md)
    regions_data = {}
    for r in rows:
        # Extract region name by stripping emoji (assume format '🇹🇭 Thailand')
        h1_raw = r.get("_H1", "Unknown")
        region_name = h1_raw.split(" ", 1)[-1].strip() if " " in h1_raw else h1_raw
        if region_name not in regions_data:
            regions_data[region_name] = {}

        idx = r.get("ดัชนี", "").replace("**", "").strip()
        val = _parse_float_from_str(r.get("ค่าล่าสุด", ""))
        prev = _parse_float_from_str(r.get("ก่อนหน้า", ""))
        ma = _parse_float_from_str(r.get("MA ย้อนหลัง", ""))
        if val is not None:
            regions_data[region_name][idx] = {
                "val": val,
                "prev": prev if prev is not None else val,
                "ma": ma if ma is not None else val
            }

    results = {}
    for region, data in regions_data.items():
        def get_metric(key_fragment: str) -> dict | None:
            for k, v in data.items():
                if key_fragment.lower() in k.lower():
                    return v
            return None

        # Helper for scoring momentum & MA
        def score_momentum(metric: dict, is_inverse: bool = False) -> float:
            score = 0.0
            if metric["val"] > metric["ma"]:
                score += 0.5 if not is_inverse else -0.5
            elif metric["val"] < metric["ma"]:
                score -= 0.5 if not is_inverse else -0.5

            if metric["val"] > metric["prev"]:
                score += 0.5 if not is_inverse else -0.5
            elif metric["val"] < metric["prev"]:
                score -= 0.5 if not is_inverse else -0.5
            return score

        # Growth
        gdp = get_metric("Real GDP")
        indpro = get_metric("Industrial Production")
        retail = get_metric("Retail Sales")
        unemp = get_metric("Unemployment Rate")

        growth_score = 0.0
        if indpro is not None:
            growth_score += score_momentum(indpro)
            weight_pmi = 0.6
            weight_lag = 0.4
        else:
            weight_pmi = 0.0
            weight_lag = 1.0

        lag_score = 0.0
        lag_count = 0
        if gdp is not None:
            lag_score += score_momentum(gdp)
            lag_count += 1
        if retail is not None:
            lag_score += score_momentum(retail)
            lag_count += 1
        if unemp is not None:
            lag_score += score_momentum(unemp, is_inverse=True)
            lag_count += 1

        if lag_count > 0:
            lag_score = lag_score / lag_count

        final_growth = (growth_score * weight_pmi) + (lag_score * weight_lag)

        # Inflation
        cpi = get_metric("CPI")
        pce = get_metric("Core PCE")

        inf_score = 0.0
        inf_count = 0
        for inf_metric in [cpi, pce]:
            if inf_metric is not None:
                inf_score += score_momentum(inf_metric, is_inverse=True)
                inf_count += 1
        if inf_count > 0:
            inf_score = inf_score / inf_count

        # Monetary (M)
        fed = get_metric("Fed Funds Rate") or get_metric("Policy Rate")
        spread = get_metric("10Y-2Y") or get_metric("10-Year Minus 2-Year")

        monetary_score = 0.0
        mon_count = 0
        if fed is not None and pce is not None:
            real_rate = fed["val"] - pce["val"]
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

        if mon_count > 0:
            monetary_score = monetary_score / mon_count

        state = _determine_economic_state(final_growth, inf_score)

        results[region] = {
            "growth": final_growth,
            "inflation": inf_score,
            "monetary": monetary_score,
            "state": state
        }
    return results


def _calculate_matrix_scores(country_md: str, regional_md: str = "") -> dict:
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
        growth = ((country["growth"] * 1.0) + (regional["growth"] * 0.5)) / 1.5
        inflation = ((country["inflation"] * 1.0) + (regional["inflation"] * 0.5)) / 1.5
        monetary = ((country["monetary"] * 1.0) + (regional["monetary"] * 0.5)) / 1.5
        blended[region] = {
            "growth": growth,
            "inflation": inflation,
            "monetary": monetary,
            "state": _determine_economic_state(growth, inflation),
        }
    return blended

def _format_trend(score: float) -> str:
    if score > 0:
        return f"{(score):.2f} ↗️"
    elif score < 0:
        return f"{(score):.2f} ↘️"
    else:
        return f"{(score):.2f} ➡️"

def _calculate_recession_probability(matrices: dict, geo_score: float) -> float:
    """คำนวณความน่าจะเป็น Recession จาก matrix scores (deterministic)"""
    us = matrices.get("United States")
    if us:
        growth, monetary = us["growth"], us["monetary"]
    else:
        all_g = [m["growth"] for m in matrices.values()]
        all_m = [m["monetary"] for m in matrices.values()]
        growth = sum(all_g) / len(all_g) if all_g else 0.0
        monetary = sum(all_m) / len(all_m) if all_m else 0.0

    raw = (-growth * 0.5) + (-monetary * 0.3) + (-geo_score * 0.2)
    return max(0.0, min(1.0, (raw + 1.0) / 2.0))
