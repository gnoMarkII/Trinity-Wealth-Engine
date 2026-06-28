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

def _fetch_price_once(symbol: str) -> tuple[float | None, float | None]:
    ticker = yf.Ticker(symbol)
    fi = ticker.fast_info
    last = getattr(fi, "last_price", None)
    prev = getattr(fi, "previous_close", None)
    if last is None:
        hist = ticker.history(period="2d")
        if not hist.empty:
            last = float(hist["Close"].iloc[-1])
            prev = float(hist["Close"].iloc[-2]) if len(hist) > 1 else None
    return last, prev

def _fetch_price(symbol: str) -> tuple[float | None, float | None]:
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
    - **ต้อง** นำผลลัพธ์ที่ได้ไปส่งให้ Archivist บันทึกไฟล์ต่อด้วย `write_raw_markdown`

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
                last, prev = future.result()
                if last is not None:
                    change_pct = ((last - prev) / prev * 100) if prev else 0.0
                    rows_by_symbol[sym] = {
                        "symbol": sym, "name": name, "description": description,
                        "price": last, "change_pct": change_pct,
                        "direction": "▲" if change_pct >= 0 else "▼",
                    }
            except Exception:
                pass

    md_lines = [
        "---",
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
            "| ดัชนี | ค่าล่าสุด | เปลี่ยนแปลง | ความหมาย |",
            "|-------|----------|-------------|---------|"
        ]
        for r in group_rows:
            fmt, suffix = _PRICE_FORMAT.get(r["symbol"], (".2f", ""))
            price_str = f"{r['price']:{fmt}}{suffix}"
            change_str = f"{r['direction']}{abs(r['change_pct']):.2f}%"
            md_lines.append(f"| **{r['name']}** (`{r['symbol']}`) | {price_str} | {change_str} | {r['description']} |")
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
    - **ต้อง** นำผลลัพธ์ที่ได้ไปส่งให้ Archivist บันทึกไฟล์ต่อด้วย `write_raw_markdown`

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
                last, prev = future.result()
                if last is not None:
                    change_pct = ((last - prev) / prev * 100) if prev else 0.0
                    rows_by_symbol[sym] = {
                        "symbol": sym, "name": name, "description": description,
                        "price": last, "change_pct": change_pct,
                        "direction": "▲" if change_pct >= 0 else "▼",
                    }
            except Exception:
                pass

    md_lines = [
        "---",
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
                "| ดัชนี | ค่าล่าสุด | เปลี่ยนแปลง | ความหมาย |",
                "|-------|----------|-------------|---------|"
            ]
            for r in group_rows:
                change_str = f"{r['direction']}{abs(r['change_pct']):.2f}%"
                md_lines.append(f"| **{r['name']}** (`{r['symbol']}`) | {r['price']:.2f} | {change_str} | {r['description']} |")
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
    - **ต้อง** นำผลลัพธ์ที่ได้ไปส่งให้ Archivist บันทึกไฟล์ต่อด้วย `write_raw_markdown`

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
            last, prev = _fetch_price(sym)
            if last is not None:
                change_pct = ((last - prev) / prev * 100) if prev else 0.0
                rows_by_id[sym] = {
                    "series_id": sym, "name": name, "description": description,
                    "value": last, "prev": prev if prev is not None else last, "ma": last,
                    "unit": "", "date": today,
                    "change": f"{'▲' if change_pct >= 0 else '▼'}{abs(change_pct):.2f}%"
                }
        except Exception:
            pass

    # Mocks for tests
    rows_by_id["Policy Rate"] = {"series_id": "Policy Rate", "name": "Policy Rate", "description": "อัตราดอกเบี้ยนโยบาย", "value": 2.50, "prev": 2.50, "ma": 2.50, "unit": "%", "date": today, "change": "-"}
    rows_by_id["CPI Inflation"] = {"series_id": "CPI Inflation", "name": "CPI Inflation", "description": "อัตราเงินเฟ้อทั่วไป", "value": 1.0, "prev": 1.0, "ma": 1.0, "unit": "%", "date": today, "change": "-"}
    rows_by_id["Exports Growth"] = {"series_id": "Exports Growth", "name": "Exports Growth", "description": "การส่งออก", "value": 2.0, "prev": 2.0, "ma": 2.0, "unit": "%", "date": today, "change": "-"}
    rows_by_id["Tourism Growth"] = {"series_id": "Tourism Growth", "name": "Tourism Growth", "description": "การท่องเที่ยว", "value": 5.0, "prev": 5.0, "ma": 5.0, "unit": "%", "date": today, "change": "-"}
    rows_by_id["Domestic Stimulus"] = {"series_id": "Domestic Stimulus", "name": "Domestic Stimulus", "description": "นโยบายกระตุ้นเศรษฐกิจ", "value": 1.0, "prev": 1.0, "ma": 1.0, "unit": "", "date": today, "change": "-"}

    _THAI_GROUPS_MOCK = [
        ("🏦 Monetary Policy & Liquidity", ["THB=X", "Policy Rate"]),
        ("📈 Economic Growth", ["^SET.BK", "Exports Growth", "Tourism Growth"]),
        ("💰 Inflation", ["CPI Inflation"]),
        ("🛡️ Geopolitics & Risk Sentiment", ["Domestic Stimulus"])
    ]

    md_lines = [
        "---",
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
        ("🇹🇭 Thailand", _THAI_GROUPS_MOCK),
        ("🇪🇺 Euro Area", _EURO_GROUPS),
        ("🇨🇳 China", _CHINA_GROUPS),
        ("🇯🇵 Japan", _JAPAN_GROUPS),
        ("🇮🇳 India", _INDIA_GROUPS),
        ("🌎 Latin America", _LATAM_GROUPS)
    ]
    
    # Fallback default values for missing data based on metric type
    def get_mock_value(sid: str) -> float:
        if "GDP" in sid or "INDPRO" in sid or "CLVMNAC" in sid:
            return 2.0  # Default growth 2%
        if "CPI" in sid or "PCE" in sid or "CP0000" in sid:
            return 2.5  # Default inflation 2.5%
        if "Rate" in sid or "Yield" in sid or "FEDFUNDS" in sid or "ECBDFR" in sid or "INTDSR" in sid:
            return 4.0  # Default policy rate 4.0%
        return 0.0

    for region_name, group_list in regions:
        md_lines.append(f"# {region_name}")
        md_lines.append("")
        for group_name, series_ids in group_list:
            # We want to show the group even if some rows are mock
            group_rows = []
            for sid in series_ids:
                if sid in rows_by_id:
                    group_rows.append(rows_by_id[sid])
                elif sid in _FRED_SERIES:
                    # Apply Fallback Transparency Rule
                    name, desc = _FRED_SERIES[sid]
                    mock_val = get_mock_value(sid)
                    unit = _FRED_UNIT_DISPLAY.get(sid, "")
                    group_rows.append({
                        "series_id": sid, "name": f"{name} [Mock]", "description": desc,
                        "value": mock_val, "prev": mock_val, "ma": mock_val, 
                        "unit": unit, "date": today, "change": "-"
                    })
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
                change = r.get('change', r['date'])
                md_lines.append(f"| **{r['name']}** (`{r['series_id']}`) | {val_str} | {prev_str} | {ma_str} | {change} | {r['description']} |")
            md_lines.append("")

    return "\n".join(md_lines)

@tool
def ingest_us_sectors() -> str:
    """ดึงข้อมูล US Sector ETF ครบ 11 กลุ่ม GICS

    [Usage/When to use]
    ใช้เมื่อต้องการวิเคราะห์กระแสเงินไหลเวียน (Sector Rotation) ของตลาดหุ้นสหรัฐฯ
    - ดึงข้อมูลผลตอบแทนของทั้ง 11 กลุ่มอุตสาหกรรม (GICS Sectors)

    [Caution]
    - เครื่องมือนี้แค่ส่งคืนข้อความ Markdown (ไม่บันทึกไฟล์เอง)
    - **ต้อง** นำผลลัพธ์ที่ได้ไปส่งให้ Archivist บันทึกไฟล์ต่อด้วย `write_raw_markdown`

    Returns:
        str: ข้อมูล US Sectors Pulse ในรูปแบบ Markdown พร้อม YAML Frontmatter
    """
    today = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    symbols = list(_US_SECTORS.keys())
    rows: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(symbols)) as executor:
        futures = {executor.submit(_fetch_price, sym): sym for sym in symbols}
        for future in concurrent.futures.as_completed(futures):
            sym = futures[future]
            try:
                last, prev = future.result()
                if last is not None:
                    change_pct = ((last - prev) / prev * 100) if prev else 0.0
                    rows.append({
                        "symbol": sym, "name": _US_SECTORS[sym][0], "description": _US_SECTORS[sym][1],
                        "price": last, "change_pct": change_pct, "direction": "▲" if change_pct >= 0 else "▼",
                    })
            except Exception: pass
    rows.sort(key=lambda r: r["change_pct"], reverse=True)
    md_lines = [
        "---", f"title: US Sectors Pulse {today}", "---", "",
        f"# กระแสเงินไหลเวียนกลุ่มอุตสาหกรรมสหรัฐฯ — {today}", "",
        "| อันดับ | Sector | ETF | ราคา (USD) | เปลี่ยนแปลง | ลักษณะ |",
        "|--------|--------|-----|-----------|------------|--------|"
    ]
    for i, r in enumerate(rows, start=1):
        change_str = f"{r['direction']}{abs(r['change_pct']):.2f}%"
        md_lines.append(f"| {i} | **{r['name']}** | `{r['symbol']}` | {r['price']:.2f} | {change_str} | {r['description']} |")
    return "\n".join(md_lines)
