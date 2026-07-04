from langsmith import traceable
import concurrent.futures

import os

from datetime import datetime

import yfinance as yf

from fredapi import Fred

from langchain_core.tools import tool

from core.logger import get_logger

from core.retry import with_retry as _with_retry


from core.logger import get_logger
log = get_logger(__name__)

from .parsers import *
from .scoring import *

import os
from pathlib import Path
from datetime import datetime
from langchain_core.tools import tool
from core.logger import get_logger
from .parsers import _parse_float_from_str, _parse_markdown_table_rows, _parse_markdown_with_context
from .scoring import _calculate_matrix_scores, _get_global_geopolitics, _calculate_recession_probability
from schemas.macro_schemas import QuantScore, RegionQuantMetrics, EconomicState, MarketObservable
import json
import re

log = get_logger(__name__)


def _slug(value: str) -> str:
    value = re.sub(r"`([^`]+)`", r"\1", str(value)).lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")[:64] or "unknown"


def _clean_indicator(value: str) -> str:
    return re.sub(r"\*\*", "", str(value)).strip()


def _extract_symbol(indicator: str) -> str:
    match = re.search(r"`([^`]+)`", str(indicator))
    return match.group(1) if match else ""


def _extract_observed_at(raw_date: str, fallback_date: str) -> str:
    match = re.search(r"\d{4}-\d{2}-\d{2}", str(raw_date))
    return match.group(0) if match else fallback_date


def _infer_asset_bucket(indicator: str, source_key: str) -> str:
    text = indicator.lower()
    symbol = _extract_symbol(indicator).lower()
    if any(k in text or k in symbol for k in ["dgs", "tnx", "tyx", "fvx", "t10y", "yield", "treasury", "bond", "spread", "lqd", "hyg", "dfii10", "tips"]):
        return "fixed_income"
    if any(k in text or k in symbol for k in ["dxy", "dollar", "usd", "eur", "jpy", "cny", "thb", "=x", "dtwexbgs", "current account", "tourist arrivals"]):
        return "fx"
    if any(k in text or k in symbol for k in ["gold", "oil", "wti", "brent", "copper", "gas", "gc=f", "cl=f", "hg=f", "ng=f"]):
        return "commodities"
    if any(k in text or k in symbol for k in ["vix", "bitcoin", "btc", "credit", "sentiment"]):
        return "risk"
    if any(k in text for k in ["fed funds", "policy rate", "t-bill", "m2", "cpi", "pce", "ppi", "inflation"]):
        return "cash"
    if any(k in text or k in symbol for k in ["gdp", "industrial", "retail", "unemployment", "pmi", "sp500", "s&p", "nasdaq", "set", "russell", "msci", "etf", "^gspc", "^ndx", "^set"]):
        return "equities"
    return "risk" if source_key == "Global_Macro_Snapshot" else "equities"


def _infer_provider(indicator: str, source_key: str) -> str:
    text = indicator.lower()
    symbol = _extract_symbol(indicator)
    if "staticproxy" in text or "static proxy" in text or "mock" in text or symbol.upper() in ["TH10Y", "CURRENT ACCOUNT", "TOURIST ARRIVALS"]:
        return "StaticProxy"
    if symbol and any(token in symbol for token in ["=", "^", "-USD"]):
        return "Yahoo"
    if source_key == "Regional_Macro_Snapshot":
        return "Yahoo"
    return "FRED"


def _infer_unit(value: str, indicator: str) -> str:
    raw = f"{value} {indicator}".lower()
    if "%" in raw or "yield" in raw or "rate" in raw:
        return "%"
    if "bps" in raw:
        return "bps"
    if "pts" in raw or "point" in raw:
        return "pts"
    if "usd" in raw or "$" in raw:
        return "USD"
    return ""


def _is_macro_stat(indicator: str) -> bool:
    text = indicator.lower()
    return any(k in text for k in ["gdp", "cpi", "pce", "ppi", "pmi", "unemployment", "policy rate", "fed funds", "m2"])


def _apply_validity(obs: MarketObservable, today_str: str) -> MarketObservable:
    if obs.provider == "StaticProxy":
        obs.is_valid = False
        obs.confidence = "low"
        obs.stale_reason = "Mock/static proxy without live market feed"
        return obs
    try:
        today = datetime.strptime(today_str, "%Y-%m-%d")
        observed = datetime.strptime(obs.observed_at, "%Y-%m-%d")
    except ValueError:
        obs.is_valid = False
        obs.confidence = "low"
        obs.stale_reason = "Invalid observed_at date"
        return obs
    age_days = (today - observed).days
    cutoff = 180 if _is_macro_stat(obs.indicator) else 7
    if age_days > cutoff:
        obs.is_valid = False
        obs.confidence = "low"
        obs.stale_reason = f"Exceeded indicator-specific stale cutoff ({age_days} days > {cutoff} days)"
    return obs


def _extract_region(row: dict, source_key: str) -> str:
    for key in ["_H1", "_H2"]:
        raw = str(row.get(key, "")).strip()
        if raw:
            cleaned = re.sub(r"^[^\w]+", "", raw).strip()
            return cleaned.split(" ", 1)[-1].strip() if " " in cleaned else cleaned
    return source_key.replace("_Macro_Snapshot", "")


def _extract_market_observables(
    contents: dict[str, str],
    resolved_files: dict[str, str],
    today_str: str,
) -> list[MarketObservable]:
    observables: list[MarketObservable] = []
    used_ids: set[str] = set()
    for source_key in ["Global_Macro_Snapshot", "Country_Macro_Snapshot", "Regional_Macro_Snapshot"]:
        for row in _parse_markdown_with_context(contents.get(source_key, "")):
            keys = [key for key in row.keys() if not key.startswith("_")]
            if len(keys) < 2:
                continue
            indicator = _clean_indicator(row.get(keys[0], ""))
            value = str(row.get(keys[1], "")).strip()
            if not indicator or not value or _parse_float_from_str(value) is None:
                continue
            observed_at = _extract_observed_at(row.get(keys[4], "") if len(keys) > 4 else "", today_str)
            symbol = _extract_symbol(indicator)
            base_id = f"obs_{_slug(source_key)}_{_slug(symbol or indicator)}_{today_str.replace('-', '')}"
            observable_id = base_id
            suffix = 2
            while observable_id in used_ids:
                observable_id = f"{base_id}_{suffix}"
                suffix += 1
            used_ids.add(observable_id)
            obs = MarketObservable(
                observable_id=observable_id,
                asset_bucket=_infer_asset_bucket(indicator, source_key),
                region=_extract_region(row, source_key),
                indicator=indicator,
                value=value,
                unit=_infer_unit(value, indicator),
                observed_at=observed_at,
                source_file=resolved_files.get(source_key, f"{source_key}_{today_str}.md"),
                source_section=" / ".join(str(row.get(k, "")).strip() for k in ["_H1", "_H2", "_H3"] if str(row.get(k, "")).strip()),
                provider=_infer_provider(indicator, source_key),
            )
            observables.append(_apply_validity(obs, today_str))
    _add_relative_observables(observables, today_str)
    return observables


def _find_observable(observables: list[MarketObservable], *needles: str) -> MarketObservable | None:
    lowered = [needle.lower() for needle in needles]
    for obs in observables:
        text = f"{obs.indicator} {obs.observable_id}".lower()
        if all(needle in text for needle in lowered):
            return obs
    return None


def _add_relative_observables(observables: list[MarketObservable], today_str: str) -> None:
    def append_relative(obs_id: str, indicator: str, value: float, unit: str, sources: list[MarketObservable]) -> None:
        if any(obs.observable_id == obs_id for obs in observables):
            return
        is_valid = all(obs.is_valid for obs in sources)
        source_file = sources[0].source_file if sources else f"Derived_{today_str}.md"
        observables.append(MarketObservable(
            observable_id=obs_id,
            asset_bucket="risk",
            region=sources[0].region if sources else "Global",
            indicator=indicator,
            value=f"{value:.2f}",
            unit=unit,
            observed_at=max((obs.observed_at for obs in sources), default=today_str),
            source_file=source_file,
            source_section="Derived relative observable",
            provider="Derived",
            confidence="high" if is_valid else "low",
            is_valid=is_valid,
            stale_reason="" if is_valid else "Derived from invalid or stale observables",
        ))

    spread = _find_observable(observables, "10y", "2y") or _find_observable(observables, "t10y2y")
    if spread:
        parsed = _parse_float_from_str(spread.value)
        if parsed is not None:
            append_relative("obs_spread_us_10y_2y", "US 10Y-2Y Yield Spread", parsed, "% pts", [spread])

    fed = _find_observable(observables, "fed funds") or _find_observable(observables, "fed", "rate")
    thai_policy = None
    for obs in observables:
        text = f"{obs.region} {obs.indicator}".lower()
        if "thai" in text and "policy rate" in text:
            thai_policy = obs
            break
    if fed and thai_policy:
        fed_val = _parse_float_from_str(fed.value)
        thai_val = _parse_float_from_str(thai_policy.value)
        if fed_val is not None and thai_val is not None:
            append_relative(
                "obs_diff_us_th_policy_rate",
                "US-Thailand Policy Rate Differential",
                fed_val - thai_val,
                "% pts",
                [fed, thai_policy],
            )

    hyg = _find_observable(observables, "hyg")
    lqd = _find_observable(observables, "lqd")
    if hyg and lqd:
        hyg_val = _parse_float_from_str(hyg.value)
        lqd_val = _parse_float_from_str(lqd.value)
        if hyg_val is not None and lqd_val not in (None, 0):
            append_relative("obs_ratio_hyg_lqd", "HYG/LQD Relative Price Ratio", hyg_val / lqd_val, "ratio", [hyg, lqd])

@tool
def evaluate_macro_matrix() -> str:
    """ประเมินข้อมูลเศรษฐกิจมหภาคและคืน QuantScore JSON

    [Usage/When to use]
    ใช้เมื่อต้องการวิเคราะห์สภาวะเศรษฐกิจ (Economic State) และคำนวณ Macro Matrix Score
    - ดึงข้อมูลจากไฟล์ Daily Snapshots ล่าสุดเพื่อประเมินสถานการณ์
    - ส่งกลับค่าตัวเลขและสภาวะเศรษฐกิจเป็น JSON string

    Returns:
        str: JSON string ของ Pydantic model `QuantScore`
    """
    today_str = os.environ.get("EVAL_DATE", datetime.now().strftime("%Y-%m-%d"))

    # Paths based on test_macro.py expectations
    vault_path = Path(os.environ.get("OBSIDIAN_VAULT_PATH", "C:/ChinoDoc/Projects/Claude/invest-agents/memories")).resolve()
    snapshots_dir = vault_path / "30_Knowledge_Base" / "Macroeconomics" / "Daily_Snapshots"
    print(f"DEBUG: vault_path = {vault_path}")
    print(f"DEBUG: snapshots_dir = {snapshots_dir}")

    # 1. Read files
    files_to_check_map = {
        "Global_Macro_Snapshot": f"Global_Macro_Snapshot_{today_str}.md",
        "Regional_Macro_Snapshot": f"Regional_Macro_Snapshot_{today_str}.md",
        "Country_Macro_Snapshot": f"Country_Macro_Snapshot_{today_str}.md"
    }

    contents = {}
    resolved_files = {}
    for key, f in files_to_check_map.items():
        path = snapshots_dir / f
        if not path.exists():
            # Fallback 1: check old nested folder path (pre-migration)
            fallback_nested = snapshots_dir / today_str / f
            if fallback_nested.exists():
                path = fallback_nested
            else:
                # Fallback 2: names without date suffix for test mocks
                fallback_path = snapshots_dir / f"{key}.md"
                if fallback_path.exists():
                    path = fallback_path
                    f = f"{key}.md"
                else:
                    # Final fallback: test mocks in old nested folder
                    fallback_path_nested = snapshots_dir / today_str / f"{key}.md"
                    if fallback_path_nested.exists():
                        path = fallback_path_nested
                        f = f"{key}.md"
                    else:
                        return f"Error: ข้อมูลไม่ครบถ้วน ไม่พบไฟล์ {f}"

        content = path.read_text(encoding="utf-8")
        if "ไม่พบข้อมูล" in content or "ERROR:" in content:
            return f"Error: ข้อมูลไม่ครบถ้วนในไฟล์ {f} ระบบจะไม่สร้างรายงานเพื่อป้องกันความผิดพลาด"

        contents[key] = content
        resolved_files[key] = f

    # 2. Score them
    try:
        global_md = contents.get("Global_Macro_Snapshot", "")
        country_md = contents.get("Country_Macro_Snapshot", "")
        regional_md = contents.get("Regional_Macro_Snapshot", "")

        matrices = _calculate_matrix_scores(country_md, regional_md)
        geo_score = _get_global_geopolitics(global_md)
        recession_prob = _calculate_recession_probability(matrices, geo_score)
        market_observables = _extract_market_observables(contents, resolved_files, today_str)

        regions_dict = {}
        for region, m in matrices.items():
            regions_dict[region] = RegionQuantMetrics(
                growth_score=m["growth"],
                inflation_score=m["inflation"],
                monetary_score=m["monetary"],
                economic_state=EconomicState(m["state"]),
                confidence=0.8  # Default value for purely quant scoring
            )

        quant = QuantScore(
            evaluated_at=datetime.now().isoformat(),
            regions=regions_dict,
            global_geopolitics_score=geo_score,
            recession_probability=recession_prob,
            data_freshness_note=f"Snapshot: {today_str}",
            market_observables=market_observables,
        )

        return json.dumps(quant.model_dump(mode="json"), ensure_ascii=False, indent=2)
    except Exception as e:
        log.error(f"Failed to evaluate macro matrix: {e}")
        return f"Error: Failed to evaluate macro matrix - {str(e)}"
