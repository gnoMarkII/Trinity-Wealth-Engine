from typing import Optional, Dict, Any, List
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
from .scoring import (
    _calculate_matrix_scores,
    _calculate_matrix_scores_from_observables,
    _get_global_geopolitics,
    _get_global_risk_sentiment,
    _calculate_recession_probability,
    _calculate_us_recession_risk,
)
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


def _extract_observed_at(row: dict, keys: list[str], today_str: str) -> tuple[str, bool]:
    """Extract YYYY-MM-DD from row. Return (date_str, has_real_date).

    Does NOT fallback to today's evaluation date if the real date is missing.
    Missing dates return ('1970-01-01', False) so they can be marked invalid.
    """
    for date_key in ["วันที่", "วันสังเกต", "ประกาศ ณ / เปลี่ยนแปลง", "date", "observed_at"]:
        for col_name, col_val in row.items():
            if date_key in col_name.lower():
                m = re.search(r"\d{4}-\d{2}-\d{2}", str(col_val))
                if m:
                    return m.group(0), True

    if len(keys) > 4:
        val = str(row.get(keys[4], ""))
        m = re.search(r"\d{4}-\d{2}-\d{2}", val)
        if m:
            return m.group(0), True

    for k, v in row.items():
        if k.startswith("_"):
            continue
        m = re.search(r"\d{4}-\d{2}-\d{2}", str(v))
        if m:
            return m.group(0), True

    return "1970-01-01", False


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
    symbol = _extract_symbol(indicator).upper()
    if symbol in ("THB=X", "USDTHB", "USD/THB") or "usd/thb" in raw or "usd to thb" in raw:
        return "THB per USD"
    if symbol in ("T10Y2Y", "T10Y3M") or "10y-2y" in raw or "10y-3m" in raw:
        return "percentage points"
    if symbol == "GC=F" or "gold futures" in raw:
        return "USD/oz"
    if "%" in raw or "yield" in raw or "rate" in raw:
        return "%"
    if "bps" in raw:
        return "bps"
    if "pts" in raw or "point" in raw or "vix" in raw:
        return "pts"
    if "usd" in raw or "$" in raw:
        return "USD"
    return ""


def _get_series_stale_cutoff(indicator: str, symbol: str) -> int:
    """Return maximum allowed age in days based on series frequency in ticker_config."""
    from .ticker_config import FRED_SERIES_SPECS

    sym_upper = symbol.upper()
    if sym_upper in FRED_SERIES_SPECS:
        freq = FRED_SERIES_SPECS[sym_upper].frequency.lower()
        if freq == "daily":
            return 7
        elif freq == "weekly":
            return 21
        elif freq == "monthly":
            return 90
        elif freq == "quarterly":
            return 180
        elif freq == "annual":
            return 450

    text = f"{indicator} {symbol}".lower()
    if any(k in text for k in ["gdp", "quarterly"]):
        return 180
    if any(k in text for k in ["cpi", "pce", "ppi", "pmi", "unemployment", "retail", "industrial", "housing", "m2", "monthly"]):
        return 90
    if any(k in text for k in ["weekly", "claims"]):
        return 21
    return 7


def _apply_validity(obs: MarketObservable, today_str: str) -> MarketObservable:
    if obs.provider == "StaticProxy" or "staticproxy" in obs.indicator.lower() or "mock" in obs.indicator.lower() or "[mock]" in obs.indicator.lower():
        obs.is_valid = False
        obs.confidence = "low"
        obs.status = "mock"
        obs.stale_reason = "Mock/static proxy without live market feed"
        return obs

    if obs.observed_at == "1970-01-01" or not obs.observed_at:
        obs.is_valid = False
        obs.confidence = "low"
        obs.status = "missing"
        obs.stale_reason = "Missing real observation date from source"
        return obs

    try:
        today = datetime.strptime(today_str, "%Y-%m-%d")
        observed = datetime.strptime(obs.observed_at, "%Y-%m-%d")
    except ValueError:
        obs.is_valid = False
        obs.confidence = "low"
        obs.status = "unverified"
        obs.stale_reason = "Invalid observed_at date format"
        return obs

    age_days = (today - observed).days
    symbol = _extract_symbol(obs.indicator)
    cutoff = _get_series_stale_cutoff(obs.indicator, symbol)

    if age_days > cutoff:
        obs.is_valid = False
        obs.confidence = "low"
        obs.status = "stale"
        obs.stale_reason = f"Exceeded indicator-specific stale cutoff ({age_days} days > {cutoff} days)"
    else:
        obs.is_valid = True
        obs.status = "verified"
    return obs


_CANONICAL_REGION_MAPPING: dict[str, str] = {
    "united states": "United States",
    "usa": "United States",
    "us": "United States",
    "thailand": "Thailand",
    "thai": "Thailand",
    "euro area": "Euro Area",
    "europe": "Euro Area",
    "china": "China",
    "japan": "Japan",
    "india": "India",
    "latin america": "Latin America",
    "latam": "Latin America",
    "global": "Global",
}


def _extract_region(row: dict, source_key: str) -> str:
    """Extract canonical region from heading hierarchy (_H1, _H2) or source_key."""
    for key in ["_H1", "_H2"]:
        raw = str(row.get(key, "")).strip()
        if not raw:
            continue
        cleaned = re.sub(r"^[^\w]+", "", raw).strip()
        if not cleaned:
            continue
        cleaned_lower = cleaned.lower()
        for pattern, canon in _CANONICAL_REGION_MAPPING.items():
            if cleaned_lower == pattern or cleaned_lower.startswith(pattern) or pattern in cleaned_lower:
                return canon

    if source_key == "Global_Macro_Snapshot":
        return "Global"

    log.warning("Could not map region from heading %s in %s, falling back to 'Global'", row.get("_H1"), source_key)
    return "Global"


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
            parsed_val = _parse_float_from_str(value)
            if not indicator or not value or parsed_val is None:
                continue
            prev_val = _parse_float_from_str(row.get(keys[2], "")) if len(keys) > 2 else None
            ma_val = _parse_float_from_str(row.get(keys[3], "")) if len(keys) > 3 else None
            obs_date, has_real_date = _extract_observed_at(row, keys, today_str)
            symbol = _extract_symbol(indicator)
            base_id = f"obs_{_slug(source_key)}_{_slug(symbol or indicator)}_{today_str.replace('-', '')}"
            observable_id = base_id
            suffix = 2
            while observable_id in used_ids:
                observable_id = f"{base_id}_{suffix}"
                suffix += 1
            used_ids.add(observable_id)

            obs_meta: dict[str, Any] = {"val": parsed_val}
            if prev_val is not None:
                obs_meta["prev"] = prev_val
            if ma_val is not None:
                obs_meta["ma"] = ma_val

            obs = MarketObservable(
                observable_id=observable_id,
                asset_bucket=_infer_asset_bucket(indicator, source_key),
                region=_extract_region(row, source_key),
                indicator=indicator,
                value=value,
                unit=_infer_unit(value, indicator),
                observed_at=obs_date,
                source_file=resolved_files.get(source_key, f"{source_key}_{today_str}.md"),
                source_section=" / ".join(str(row.get(k, "")).strip() for k in ["_H1", "_H2", "_H3"] if str(row.get(k, "")).strip()),
                provider=_infer_provider(indicator, source_key),
                metadata=obs_meta,
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
    def append_relative(
        obs_id: str,
        indicator: str,
        value: float,
        unit: str,
        sources: list[MarketObservable],
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        if any(obs.observable_id == obs_id for obs in observables):
            return
        is_valid = all(obs.is_valid for obs in sources) if sources else False
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
            status="verified" if is_valid else "unverified",
            stale_reason="" if is_valid else "Derived from invalid or stale observables",
            input_observable_ids=[obs.observable_id for obs in sources],
            metadata=metadata or {},
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
            diff_pct = fed_val - thai_val
            diff_bps = diff_pct * 100.0
            append_relative(
                "obs_diff_us_th_policy_rate",
                "US-Thailand Policy Rate Differential",
                diff_pct,
                "% pts",
                [fed, thai_policy],
                metadata={"diff_pct": round(diff_pct, 4), "diff_bps": round(diff_bps, 2)},
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
        candidates = [
            snapshots_dir / f,
            (snapshots_dir / today_str[:4] / today_str[5:7] / f) if len(today_str) >= 7 else None,
            snapshots_dir / today_str / f,
            snapshots_dir / f"{key}.md",
            snapshots_dir / today_str / f"{key}.md",
        ]
        found_path = None
        for cand in candidates:
            if cand and cand.exists():
                found_path = cand
                f = cand.name
                break

        if not found_path:
            return f"Error: ข้อมูลไม่ครบถ้วน ไม่พบไฟล์ {f}"

        path = found_path

        content = path.read_text(encoding="utf-8")
        if "ไม่พบข้อมูล" in content or "ERROR:" in content:
            return f"Error: ข้อมูลไม่ครบถ้วนในไฟล์ {f} ระบบจะไม่สร้างรายงานเพื่อป้องกันความผิดพลาด"

        contents[key] = content
        resolved_files[key] = f

    # 2. Build Observables Registry (Parse -> Validate -> Score)
    try:
        global_md = contents.get("Global_Macro_Snapshot", "")
        country_md = contents.get("Country_Macro_Snapshot", "")
        regional_md = contents.get("Regional_Macro_Snapshot", "")

        market_observables = _extract_market_observables(contents, resolved_files, today_str)

        is_test_env = "PYTEST_CURRENT_TEST" in os.environ or os.environ.get("MACRO_OFFLINE_EVAL") == "true"

        # Build & merge Valuation Observables (Pillar 1)
        try:
            from .valuation import build_valuation_observables, build_credit_spread_observable
            val_info_getter = (lambda sym: {}) if is_test_env else None
            val_obs = build_valuation_observables(
                existing_observables=market_observables,
                ticker_info_getter=val_info_getter,
            )
            market_observables.extend(val_obs)
            hy_obs = build_credit_spread_observable(existing_observables=market_observables)
            if hy_obs and not any(o.observable_id == hy_obs.observable_id for o in market_observables):
                market_observables.append(hy_obs)
        except Exception as e:
            log.warning(f"Could not build valuation observables: {e}")

        # Build & merge Derived Pair Trade Ratios (Pillar 3)
        try:
            from .derived_ratios import build_derived_pair_observables, _default_price_getter
            pair_price_getter = (lambda s: None) if is_test_env else _default_price_getter
            pair_obs = build_derived_pair_observables(
                existing_observables=market_observables,
                price_getter=pair_price_getter,
                today_str=today_str,
                use_mock_fallback=False
            )
            for po in pair_obs:
                if not any(o.observable_id == po.observable_id for o in market_observables):
                    market_observables.append(po)
        except Exception as e:
            log.warning(f"Could not build derived pair observables: {e}")

        # Build & merge Risk Correlation Analytics (Pillar 4)
        try:
            from .risk_analytics import build_risk_correlation_observables, _default_correlation_calculator
            corr_calc = (lambda *a, **k: None) if is_test_env else _default_correlation_calculator
            corr_obs = build_risk_correlation_observables(
                correlation_calculator=corr_calc,
                today_str=today_str,
                use_mock_fallback=False
            )
            for co in corr_obs:
                if not any(o.observable_id == co.observable_id for o in market_observables):
                    market_observables.append(co)
        except Exception as e:
            log.warning(f"Could not build risk correlation observables: {e}")

        # Build & merge Terminal V2 Observables (Pillar 5 / Dual-Track)
        try:
            from .terminal_observables import (
                build_thai_market_observables,
                build_rates_observables,
                build_thai_market_stance,
            )
            if not is_test_env:
                t2_thai = build_thai_market_observables(as_of_date=today_str)
                for to in t2_thai:
                    if not any(o.observable_id == to.observable_id for o in market_observables):
                        market_observables.append(to)

                t2_rates = build_rates_observables(as_of_date=today_str)
                for ro in t2_rates:
                    if not any(o.observable_id == ro.observable_id for o in market_observables):
                        market_observables.append(ro)
        except Exception as e:
            log.warning(f"Could not build Terminal V2 observables: {e}")

        # Build & merge Thai Hard Data Status Adapter (Fail-closed official feeds)
        try:
            from .adapters.thai_hard_data_adapter import ThaiHardDataAdapter
            thai_adapter = ThaiHardDataAdapter()
            for th_obs in thai_adapter.as_observables(as_of_date=today_str):
                if not any(o.observable_id == th_obs.observable_id for o in market_observables):
                    market_observables.append(th_obs)
        except Exception as e:
            log.warning(f"Could not build Thai Hard Data observables: {e}")

        thai_stance = None
        try:
            from .terminal_observables import build_thai_market_stance
            thai_stance = build_thai_market_stance(market_observables)
        except Exception as e:
            log.warning(f"Could not aggregate Thai market stance: {e}")

        # 3. Calculate Scores strictly from validated observables
        matrices = _calculate_matrix_scores_from_observables(market_observables)
        geo_score = _get_global_risk_sentiment(global_md=global_md, observables=market_observables)
        us_metrics = matrices.get("United States", {})
        us_growth = us_metrics.get("growth")
        us_monetary = us_metrics.get("monetary")
        us_recession_risk = _calculate_us_recession_risk(us_growth, us_monetary, geo_score)
        recession_prob = us_recession_risk if us_recession_risk is not None else 0.5

        regions_dict = {}
        for region, m in matrices.items():
            stance_val = thai_stance if ("thailand" in region.lower() and thai_stance and any(thai_stance.values())) else None
            regions_dict[region] = RegionQuantMetrics(
                growth_score=m.get("growth"),
                inflation_score=m.get("inflation"),
                monetary_score=m.get("monetary"),
                economic_state=EconomicState(m.get("state", "Unknown")),
                confidence=m.get("confidence", 0.0),
                coverage=m.get("coverage", 0.0),
                data_gaps=m.get("data_gaps", []),
                market_stance=stance_val,
            )

        all_gaps = []
        coverage_dict = {}
        for r_name, r_metrics in regions_dict.items():
            all_gaps.extend(r_metrics.data_gaps)
            coverage_dict[r_name] = r_metrics.coverage

        quant = QuantScore(
            evaluated_at=datetime.now().isoformat(),
            regions=regions_dict,
            global_geopolitics_score=geo_score,
            global_risk_sentiment_score=geo_score,
            recession_probability=recession_prob,
            us_recession_risk_score=us_recession_risk,
            data_freshness_note=f"Snapshot: {today_str}",
            coverage=coverage_dict,
            data_gaps=list(dict.fromkeys(all_gaps)),
            formula_version="2.0.0",
            market_observables=market_observables,
        )

        try:
            _write_macro_observables_json_sidecar(market_observables, today_str, snapshots_dir)
        except Exception as e:
            log.warning("Could not write macro observables JSON sidecar: %s", e)

        return json.dumps(quant.model_dump(mode="json"), ensure_ascii=False, indent=2)
    except Exception as e:
        log.error(f"Failed to evaluate macro matrix: {e}")
        return f"Error: Failed to evaluate macro matrix - {str(e)}"


def _write_macro_observables_json_sidecar(observables: list[MarketObservable], today_str: str, snapshots_dir: Path) -> Path:
    """บันทึก list[MarketObservable] เป็น JSON Sidecar ป้องกันการ regex parse จาก prose Markdown"""
    from tools._atomic_io import _atomic_write_to
    sidecar_path = snapshots_dir / f"Macro_Observables_Snapshot_{today_str}.json"
    from tools.archivist.maintenance_guard import assert_write_allowed
    assert_write_allowed(sidecar_path)
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    payload = [o.model_dump(mode="json") for o in observables]
    _atomic_write_to(sidecar_path, json.dumps(payload, ensure_ascii=False, indent=2))
    return sidecar_path


def load_latest_macro_observables(vault_path: Optional[Path] = None) -> dict[str, MarketObservable]:
    """โหลด MarketObservables ล่าสุดจากไฟล์ Macro_Observables_Snapshot_YYYY-MM-DD.json
    หากไม่พบ ให้ fallback ไปรัน build_valuation_observables() พร้อมเตือนใน log
    """
    if vault_path is None:
        vault_path = Path(os.environ.get("OBSIDIAN_VAULT_PATH", "./memories")).resolve()

    snapshots_dir = vault_path / "30_Knowledge_Base" / "Macroeconomics" / "Daily_Snapshots"

    if snapshots_dir.exists():
        json_files = sorted(
            [p for p in snapshots_dir.rglob("Macro_Observables_Snapshot_*.json") if "Revisions" not in p.parts],
            reverse=True,
        )
        if json_files:
            try:
                latest_json = json_files[0]
                with open(latest_json, "r", encoding="utf-8") as f:
                    raw_list = json.load(f)
                from schemas.macro_schemas import MarketObservable
                obs_list = [MarketObservable.model_validate(item) for item in raw_list]
                return {o.observable_id: o for o in obs_list if getattr(o, "is_valid", True)}
            except Exception as e:
                log.warning("Failed to parse macro JSON sidecar %s: %s", json_files[0], e)

    from .valuation import build_valuation_observables
    log.info("[DCF Engine] No Daily Macro Snapshot JSON found. Seeding fresh Macro Observables.")
    try:
        obs_list = build_valuation_observables()
        return {o.observable_id: o for o in obs_list if getattr(o, "is_valid", True)}
    except Exception as e:
        log.warning("Could not load macro observables: %s", e)
        return {}
