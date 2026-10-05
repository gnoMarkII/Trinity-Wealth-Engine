from langsmith import traceable
import concurrent.futures
import os
from datetime import datetime
import yfinance as yf
from fredapi import Fred
from langchain_core.tools import tool
from core.logger import get_logger
from core.retry import with_retry as _with_retry

log = get_logger(__name__)

from .ticker_config import (
    _FETCH_TIMEOUT, _PRICE_FORMAT, _MACRO_TICKERS, _GLOBAL_GROUPS,
    _US_SECTORS, _REGIONAL_TICKERS, _REGIONAL_GROUPS_MAP,
    _FRED_SERIES, _FRED_YOY_SERIES, _FRED_UNIT_DISPLAY, _US_GROUPS,
    _THAI_INDICATORS, _THAI_GROUPS, _EURO_GROUPS, _CHINA_GROUPS,
    _JAPAN_GROUPS, _INDIA_GROUPS, _LATAM_GROUPS
)

def _fetch_price_once(symbol: str) -> tuple[float | None, float | None, str | None]:
    ticker = yf.Ticker(symbol)
    # Read value and observation date from the same bar. fast_info has no
    # observation timestamp, so stamping its price with today's date is unsafe.
    hist = ticker.history(period="5d", auto_adjust=False, timeout=_FETCH_TIMEOUT)
    if hist.empty:
        return None, None, None
    closes = hist["Close"].dropna()
    if closes.empty:
        return None, None, None
    last = float(closes.iloc[-1])
    prev = float(closes.iloc[-2]) if len(closes) > 1 else None
    return last, prev, closes.index[-1].strftime("%Y-%m-%d")

def _fetch_price(symbol: str) -> tuple[float | None, float | None, str | None]:
    return _with_retry(_fetch_price_once, symbol)

def _fetch_fred_once(fred: Fred, series_id: str):
    if series_id in _FRED_YOY_SERIES:
        return fred.get_series(series_id, units="pc1").dropna()
    return fred.get_series(series_id).dropna()

def _fetch_fred_series(args: tuple):
    fred, series_id = args
    return _with_retry(_fetch_fred_once, fred, series_id)

@tool
def ingest_global_macro() -> str:
    """ดึงข้อมูลเศรษฐกิจระดับโลก จัดกลุ่ม 4 มิติ (Monetary Policy, Growth, Inflation, Geopolitics)

    [Usage/When to use]
    ใช้เมื่อต้องการภาพรวมของสภาวะตลาดโลก (Global Macro)
    - ดึงข้อมูลดัชนีสำคัญเช่น Yield Curve, VIX, DXY, ทองคำ, น้ำมัน, Bitcoin

    [Caution]
    - เครื่องมือนี้แค่ส่งคืนข้อความ Markdown (ไม่บันทึกไฟล์เอง)
    - ผลลัพธ์จะถูกนำไปส่งให้ Archivist บันทึกไฟล์ต่อโดยอัตโนมัติ

    Returns:
        str: ข้อมูล Global Macro Snapshot ในรูปแบบ Markdown พร้อม YAML Frontmatter
    """
    today = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    all_symbols = list(_MACRO_TICKERS.keys())
    rows_by_symbol: dict[str, dict] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(all_symbols)) as executor:
        futures = {executor.submit(_fetch_price, sym): sym for sym in all_symbols}
        for future in concurrent.futures.as_completed(futures, timeout=_FETCH_TIMEOUT * 2):
            sym = futures[future]
            name, description = _MACRO_TICKERS[sym]
            try:
                last, prev, observed_at = future.result()
                if last is not None:
                    change_pct = ((last - prev) / prev * 100) if prev else 0.0
                    rows_by_symbol[sym] = {
                        "symbol": sym, "name": name, "description": description,
                        "price": last, "change_pct": change_pct,
                        "previous_price": prev, "observed_at": observed_at,
                        "direction": "▲" if change_pct >= 0 else "▼",
                    }
            except Exception:
                pass

    md_lines = [
        "---",
        "schema_version: 2",
        f"title: Global Macro Snapshot {today}",
        "entity_type: macro_global",
        f"date: {today}",
        f"last_updated: {now_time}",
        "tags: [macro, global, snapshot]",
        "---",
        "",
        f"# 🌍 Global Macro Snapshot ({today})",
        "",
    ]

    for group_name, symbols in _GLOBAL_GROUPS:
        group_rows = [rows_by_symbol[sym] for sym in symbols if sym in rows_by_symbol]
        if not group_rows: continue
        md_lines += [
            f"## {group_name}", "",
            "| ดัชนี | ค่าล่าสุด | ก่อนหน้า | เปลี่ยนแปลง | วันสังเกต | ความหมาย |",
            "|-------|----------|----------|-------------|----------|---------|"
        ]
        for r in group_rows:
            fmt, suffix = _PRICE_FORMAT.get(r["symbol"], (".2f", ""))
            price_str = f"{r['price']:{fmt}}{suffix}"
            change_str = f"{r['direction']}{abs(r['change_pct']):.2f}%"
            prev_str = f"{r['previous_price']:{fmt}}{suffix}" if r['previous_price'] is not None else "—"
            md_lines.append(f"| **{r['name']}** (`{r['symbol']}`) | {price_str} | {prev_str} | {change_str} | {r['observed_at'] or '—'} | {r['description']} |")
        md_lines.append("")

    return "\n".join(md_lines)

@tool
def ingest_regional_macro() -> str:
    """ดึงข้อมูล Regional Proxy ETF จัดกลุ่มตามภูมิภาค และ 4 มิติ

    [Usage/When to use]
    ใช้เมื่อต้องการภาพรวมของสภาวะตลาดรายภูมิภาค (Regional Macro)
    - ดึงข้อมูล ETF ที่เป็นตัวแทนของภูมิภาคต่างๆ เช่น LatAm, EU, EM, Asia

    [Caution]
    - เครื่องมือนี้แค่ส่งคืนข้อความ Markdown (ไม่บันทึกไฟล์เอง)
    - ผลลัพธ์จะถูกนำไปส่งให้ Archivist บันทึกไฟล์ต่อโดยอัตโนมัติ

    Returns:
        str: ข้อมูล Regional Macro Snapshot ในรูปแบบ Markdown พร้อม YAML Frontmatter
    """
    today = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    symbols = list(_REGIONAL_TICKERS.keys())

    rows_by_symbol: dict[str, dict] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(symbols)) as executor:
        futures = {executor.submit(_fetch_price, sym): sym for sym in symbols}
        for future in concurrent.futures.as_completed(futures, timeout=_FETCH_TIMEOUT * 2):
            sym = futures[future]
            name, description = _REGIONAL_TICKERS[sym]
            try:
                last, prev, observed_at = future.result()
                if last is not None:
                    change_pct = ((last - prev) / prev * 100) if prev else 0.0
                    rows_by_symbol[sym] = {
                        "symbol": sym, "name": name, "description": description,
                        "price": last, "change_pct": change_pct,
                        "previous_price": prev, "observed_at": observed_at,
                        "direction": "▲" if change_pct >= 0 else "▼",
                    }
            except Exception:
                pass

    md_lines = [
        "---",
        "schema_version: 2",
        f"title: Regional Macro Snapshot {today}",
        "entity_type: macro_regional",
        f"date: {today}",
        f"last_updated: {now_time}",
        "tags: [macro, regional, etf_proxy]",
        "---",
        "",
        f"# 🗺️ Regional Macro Snapshot ({today})",
        "",
    ]

    for region, pillars in _REGIONAL_GROUPS_MAP.items():
        md_lines.append(f"## {region}")
        for pillar, syms in pillars.items():
            group_rows = [rows_by_symbol[s] for s in syms if s in rows_by_symbol]
            if not group_rows: continue
            md_lines += [
                f"### {pillar}", "",
                "| ดัชนี | ค่าล่าสุด | ก่อนหน้า | เปลี่ยนแปลง | วันสังเกต | ความหมาย |",
                "|-------|----------|----------|-------------|----------|---------|"
            ]
            for r in group_rows:
                change_str = f"{r['direction']}{abs(r['change_pct']):.2f}%"
                prev_str = f"{r['previous_price']:.2f}" if r['previous_price'] is not None else "—"
                md_lines.append(f"| **{r['name']}** (`{r['symbol']}`) | {r['price']:.2f} | {prev_str} | {change_str} | {r['observed_at'] or '—'} | {r['description']} |")
            md_lines.append("")

    return "\n".join(md_lines)

@tool
def ingest_country_macro() -> str:
    """ดึงตัวเลขเศรษฐกิจพื้นฐานของประเทศสหรัฐฯ (FRED) และไทย จัดกลุ่มเป็น 4 มิติ

    [Usage/When to use]
    ใช้เมื่อต้องการตัวเลขเศรษฐกิจพื้นฐาน (Hard Data) แบบรายประเทศ
    - ดึงข้อมูลจากฐานข้อมูล FRED สำหรับสหรัฐฯ และจำลองข้อมูลสำหรับประเทศไทย

    [Caution]
    - เครื่องมือนี้แค่ส่งคืนข้อความ Markdown (ไม่บันทึกไฟล์เอง)
    - ผลลัพธ์จะถูกนำไปส่งให้ Archivist บันทึกไฟล์ต่อโดยอัตโนมัติ

    Returns:
        str: ข้อมูล Country Macro Snapshot ในรูปแบบ Markdown พร้อม YAML Frontmatter
    """
    today = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    api_key = os.getenv("FRED_API_KEY")
    rows_by_id: dict[str, dict] = {}
    if api_key:
        try:
            fred = Fred(api_key=api_key)
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(_FRED_SERIES)) as executor:
                futures = {executor.submit(_fetch_fred_series, (fred, sid)): sid for sid in _FRED_SERIES}
                for future in concurrent.futures.as_completed(futures):
                    sid = futures[future]
                    name, description = _FRED_SERIES[sid]
                    try:
                        raw = future.result()
                        if not raw.empty:
                            unit = _FRED_UNIT_DISPLAY.get(sid, "")
                            days_diff = (raw.index[-1] - raw.index[-2]).days if len(raw) > 1 else 30
                            if days_diff > 180:
                                ma_period = 3  # Annual
                            elif days_diff > 70:
                                ma_period = 4  # Quarterly
                            else:
                                ma_period = 12  # Monthly

                            val = float(raw.iloc[-1])
                            prev_val = float(raw.iloc[-2]) if len(raw) > 1 else val
                            ma_val = float(raw.tail(ma_period).mean()) if len(raw) >= ma_period else val

                            rows_by_id[sid] = {
                                "series_id": sid, "name": name, "description": description,
                                "value": val, "prev": prev_val, "ma": ma_val, "unit": unit,
                                "date": raw.index[-1].strftime("%Y-%m-%d"),
                            }
                    except Exception:
                        pass
        except Exception:
            pass

    for sym, (name, description) in _THAI_INDICATORS.items():
        try:
            last, prev, observed_at = _fetch_price(sym)
            if last is not None:
                change_pct = ((last - prev) / prev * 100) if prev else 0.0
                rows_by_id[sym] = {
                    "series_id": sym, "name": name, "description": description,
                    "value": last, "prev": prev if prev is not None else last, "ma": last,
                    "unit": "", "date": observed_at or "—",
                    "change": f"{'▲' if change_pct >= 0 else '▼'}{abs(change_pct):.2f}%"
                }
        except Exception:
            pass

    # In production path, do not inject mock indicators.
    # Offline test suites can enable mock injection explicitly via ALLOW_MOCK_MACRO_INGEST=true.
    if os.getenv("ALLOW_MOCK_MACRO_INGEST", "false").lower() == "true":
        rows_by_id["Policy Rate"] = {"series_id": "Policy Rate", "name": "Policy Rate [Mock]", "description": "อัตราดอกเบี้ยนโยบาย (Mock)", "value": 2.50, "prev": 2.50, "ma": 2.50, "unit": "%", "date": today, "change": "-", "provider": "Mock", "is_valid": False, "confidence": "low"}
        rows_by_id["TH10Y"] = {"series_id": "TH10Y", "name": "Thailand 10Y Gov Bond Yield [StaticProxy]", "description": "StaticProxy: Thailand 10-Year Government Bond Yield proxy", "value": 2.65, "prev": 2.65, "ma": 2.65, "unit": "%", "date": today, "change": "StaticProxy", "provider": "Mock", "is_valid": False, "confidence": "low"}
        rows_by_id["CPI Inflation"] = {"series_id": "CPI Inflation", "name": "CPI Inflation [Mock]", "description": "อัตราเงินเฟ้อทั่วไป (Mock)", "value": 1.0, "prev": 1.0, "ma": 1.0, "unit": "%", "date": today, "change": "-", "provider": "Mock", "is_valid": False, "confidence": "low"}
        rows_by_id["Exports Growth"] = {"series_id": "Exports Growth", "name": "Exports Growth [Mock]", "description": "การส่งออก (Mock)", "value": 2.0, "prev": 2.0, "ma": 2.0, "unit": "%", "date": today, "change": "-", "provider": "Mock", "is_valid": False, "confidence": "low"}
        rows_by_id["Tourism Growth"] = {"series_id": "Tourism Growth", "name": "Tourism Growth [Mock]", "description": "การท่องเที่ยว (Mock)", "value": 5.0, "prev": 5.0, "ma": 5.0, "unit": "%", "date": today, "change": "-", "provider": "Mock", "is_valid": False, "confidence": "low"}
        rows_by_id["Domestic Stimulus"] = {"series_id": "Domestic Stimulus", "name": "Domestic Stimulus [Mock]", "description": "นโยบายกระตุ้นเศรษฐกิจ (Mock)", "value": 1.0, "prev": 1.0, "ma": 1.0, "unit": "", "date": today, "change": "-", "provider": "Mock", "is_valid": False, "confidence": "low"}
        rows_by_id["Current Account"] = {"series_id": "Current Account", "name": "Thailand Current Account [StaticProxy]", "description": "StaticProxy: ดุลบัญชีเดินสะพัดไทย proxy", "value": 1.2, "prev": 1.2, "ma": 1.2, "unit": "B USD", "date": today, "change": "StaticProxy", "provider": "Mock", "is_valid": False, "confidence": "low"}
        rows_by_id["Tourist Arrivals"] = {"series_id": "Tourist Arrivals", "name": "Foreign Tourist Arrivals [StaticProxy]", "description": "StaticProxy: จำนวนนักท่องเที่ยวต่างชาติเข้าไทย proxy", "value": 2.85, "prev": 2.85, "ma": 2.85, "unit": "M persons", "date": today, "change": "StaticProxy", "provider": "Mock", "is_valid": False, "confidence": "low"}
        thai_groups = [
            ("🏦 Monetary Policy & Liquidity", ["THB=X", "Policy Rate", "TH10Y", "Current Account"]),
            ("📈 Economic Growth", ["^SET.BK", "Exports Growth", "Tourism Growth", "Tourist Arrivals"]),
            ("💰 Inflation", ["CPI Inflation"]),
            ("🛡️ Geopolitics & Risk Sentiment", ["Domestic Stimulus"])
        ]
    else:
        thai_groups = _THAI_GROUPS

    md_lines = [
        "---",
        "schema_version: 2",
        f"title: Country Macro Snapshot {today}",
        "entity_type: macro_country",
        f"date: {today}",
        f"last_updated: {now_time}",
        "tags: [macro, country, fred, hard_data]",
        "---",
        "",
    ]

    regions = [
        ("🇺🇸 United States", _US_GROUPS),
        ("🇹🇭 Thailand", thai_groups),
        ("🇪🇺 Euro Area", _EURO_GROUPS),
        ("🇨🇳 China", _CHINA_GROUPS),
        ("🇯🇵 Japan", _JAPAN_GROUPS),
        ("🇮🇳 India", _INDIA_GROUPS),
        ("🌎 Latin America", _LATAM_GROUPS)
    ]

    for region_name, group_list in regions:
        md_lines.append(f"# {region_name}")
        md_lines.append("")
        for group_name, series_ids in group_list:
            group_rows = []
            for sid in series_ids:
                if sid in rows_by_id:
                    group_rows.append(rows_by_id[sid])
            if not group_rows: continue
            md_lines += [
                f"### {group_name}", "",
                "| ดัชนี | ค่าล่าสุด | ก่อนหน้า | MA ย้อนหลัง | ประกาศ ณ / เปลี่ยนแปลง | ความหมาย |",
                "|-------|----------|----------|------------|-----------------------|---------|"
            ]
            for r in group_rows:
                val_str = f"{r['value']:.2f} {r['unit']}".strip()
                prev_str = f"{r['prev']:.2f} {r['unit']}".strip()
                ma_str = f"{r['ma']:.2f} {r['unit']}".strip()
                change = f"{r['date']} / {r['change']}" if r.get('change') else r['date']
                md_lines.append(f"| **{r['name']}** (`{r['series_id']}`) | {val_str} | {prev_str} | {ma_str} | {change} | {r['description']} |")
            md_lines.append("")

    return "\n".join(md_lines)

@tool
def ingest_us_sectors() -> str:
    """Return the shared deterministic US sector snapshot as compact JSON.

    The dashboard and macro agents use the same adjusted-price inputs, formula
    versions, missing-data reasons, and immutable snapshot identity. This is a
    market-strength proxy; it does not measure ETF fund flows.
    """
    try:
        from tools.macro.sector_rotation.bootstrap import get_sector_rotation_service
        from tools.macro.sector_rotation.domain.claims import compact_ai_context
        from tools.macro.sector_rotation.run_context import current_sector_run_id

        run_id = current_sector_run_id()
        if not run_id:
            return json.dumps({"status": "unavailable", "reason": "macro_run_binding_required"}, ensure_ascii=False)
        service = get_sector_rotation_service()
        snapshot, binding_status = service.pin_for_run(run_id)
        if snapshot is None:
            return json.dumps(binding_status, ensure_ascii=False)
        return json.dumps(compact_ai_context(snapshot), ensure_ascii=False, allow_nan=False)
    except Exception as exc:
        return json.dumps(
            {"status": "unavailable", "reason": f"sector_snapshot_unavailable:{type(exc).__name__}"},
            ensure_ascii=False,
        )

@traceable(run_type="chain")
def fetch_and_save_macro_snapshots() -> None:
    """ดึงข้อมูล Snapshots 3 ระดับ (Global, Regional, Country) และบันทึกลง Daily_Snapshots โดยตรง"""
    today_str = os.environ.get("EVAL_DATE", datetime.now().strftime("%Y-%m-%d"))
    from tools.archivist.writer import write_raw_markdown

    try:
        from tools.macro.adapters.thai_hard_data_adapter import sync_thai_macro_data
        sync_thai_macro_data()
    except Exception as exc:
        log.warning(f"Failed to sync Thai macro data during fetch_and_save_macro_snapshots: {exc}")

    folder = "30_Knowledge_Base/Macroeconomics/Daily_Snapshots"
    for content, filename in (
        (ingest_global_macro.invoke({}), f"Global_Macro_Snapshot_{today_str}"),
        (ingest_regional_macro.invoke({}), f"Regional_Macro_Snapshot_{today_str}"),
        (ingest_country_macro.invoke({}), f"Country_Macro_Snapshot_{today_str}"),
    ):
        write_raw_markdown.invoke({
            "content": content,
            "folder_path": folder,
            "filename": filename,
        })
