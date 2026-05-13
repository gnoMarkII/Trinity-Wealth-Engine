import concurrent.futures
import os
from datetime import datetime

import yfinance as yf
from fredapi import Fred
from langchain_core.tools import tool

_FETCH_TIMEOUT = 10  # seconds per symbol

# Price display format per symbol: (format_spec, suffix)
_PRICE_FORMAT: dict[str, tuple[str, str]] = {
    "^IRX": (".4f", "%"), "^FVX": (".4f", "%"), "^TNX": (".4f", "%"), "^TYX": (".4f", "%"),
    "^VIX": (".2f", ""),
    "HYG": (".2f", ""), "LQD": (".2f", ""),
    "DX-Y.NYB": (".2f", ""),
    "EURUSD=X": (".4f", ""), "USDJPY=X": (".2f", ""), "USDCNY=X": (".4f", ""),
    "GC=F": (",.2f", ""), "CL=F": (".2f", ""), "NG=F": (".3f", ""), "HG=F": (".4f", ""),
    "^GSPC": (",.2f", ""), "^NDX": (",.2f", ""), "^RUT": (",.2f", ""),
    "BTC-USD": (",.0f", ""),
}

_MACRO_TICKERS: dict[str, tuple[str, str]] = {
    # --- Yield Curve (เรียงอายุสั้น → ยาว เพื่ออ่านรูปร่าง Curve ได้ทันที) ---
    "^IRX": (
        "13-Week T-Bill Yield",
        "อัตราผลตอบแทนพันธบัตร 3 เดือน — จุดเริ่มต้น Yield Curve ใช้เทียบ 10Y เพื่อดู Inversion",
    ),
    "^FVX": (
        "5-Year Treasury Yield",
        "อัตราผลตอบแทนพันธบัตร 5 ปี — จุดกึ่งกลาง Curve สะท้อนคาดการณ์ดอกเบี้ยระยะกลาง",
    ),
    "^TNX": (
        "10-Year Treasury Yield",
        "อัตราผลตอบแทนพันธบัตร 10 ปี — Risk-Free Rate หลักของโลก กำหนด Discount Rate ทุกสินทรัพย์",
    ),
    "^TYX": (
        "30-Year Treasury Yield",
        "อัตราผลตอบแทนพันธบัตร 30 ปี — Long-end สะท้อนคาดการณ์เงินเฟ้อและการเติบโตระยะยาว",
    ),
    # --- Risk Sentiment ---
    "^VIX": (
        "VIX Fear Index",
        "ดัชนีความผันผวนของตลาด — ค่า >30 = ความกลัวรุนแรง, ค่า <20 = ตลาดสงบ",
    ),
    # --- Credit Market (สัญญาณ Financial Stress) ---
    "HYG": (
        "High Yield Bond ETF (HYG)",
        "ตราสารหนี้ High Yield — ตกก่อนตลาดหุ้นเสมอ ใช้เป็น Early Warning ของ Credit Stress",
    ),
    "LQD": (
        "Investment Grade Bond ETF (LQD)",
        "ตราสารหนี้ Investment Grade — สะท้อนต้นทุนกู้ยืมของบริษัทใหญ่ อ่อนไหวต่อ Rate ขึ้น",
    ),
    # --- สกุลเงิน / FX (DXY ก่อน แล้วตามด้วยคู่สกุลหลัก) ---
    "DX-Y.NYB": (
        "US Dollar Index (DXY)",
        "ความแข็งแกร่งของดอลลาร์เทียบ 6 สกุลเงินหลัก — ค่าสูงกดดัน EM Assets และสินค้าโภคภัณฑ์",
    ),
    "EURUSD=X": (
        "EUR/USD",
        "ค่าเงินยูโรต่อดอลลาร์ — สะท้อน ECB vs Fed Policy Divergence คู่ที่มีสภาพคล่องสูงสุดในโลก",
    ),
    "USDJPY=X": (
        "USD/JPY",
        "ค่าเงินดอลลาร์ต่อเยน — สะท้อน BOJ Policy และ Carry Trade ค่าสูง = เยนอ่อน",
    ),
    "USDCNY=X": (
        "USD/CNY",
        "ค่าเงินดอลลาร์ต่อหยวน — ชี้วัดแรงกดดันเศรษฐกิจจีนและทิศทางนโยบาย PBOC",
    ),
    # --- Commodities (Safe Haven → Energy → Industrial) ---
    "GC=F": (
        "Gold Futures (USD/oz)",
        "ทองคำล่วงหน้า — Safe Haven ที่มักผกผันกับ Real Interest Rate และ DXY",
    ),
    "CL=F": (
        "WTI Crude Oil (USD/bbl)",
        "น้ำมันดิบ WTI — สะท้อนอุปสงค์เศรษฐกิจโลกและต้นทุนพลังงานภาคการผลิต",
    ),
    "NG=F": (
        "Natural Gas (USD/MMBtu)",
        "ก๊าซธรรมชาติ — ต้นทุนพลังงานอุตสาหกรรม อ่อนไหวต่อสภาพอากาศและภูมิรัฐศาสตร์",
    ),
    "HG=F": (
        "Copper Futures (USD/lb)",
        "ทองแดง (Dr. Copper) — ตัวชี้วัดล่วงหน้าของเศรษฐกิจภาคการผลิตและอุตสาหกรรมโลก",
    ),
    # --- US Equities (Broad → Growth → Small-cap) ---
    "^GSPC": (
        "S&P 500",
        "ตัวแทนตลาดหุ้นสหรัฐฯ ภาพรวม 500 บริษัทชั้นนำ",
    ),
    "^NDX": (
        "Nasdaq 100",
        "ตัวแทนหุ้นเทคโนโลยีสหรัฐฯ — ไวต่อ Real Rate มากกว่า S&P",
    ),
    "^RUT": (
        "Russell 2000",
        "ตัวแทนบริษัทขนาดเล็กสหรัฐฯ — สะท้อนเศรษฐกิจในประเทศ ไวต่อ Credit Condition",
    ),
    # --- Digital Assets ---
    "BTC-USD": (
        "Bitcoin",
        "ตัวชี้วัดสภาพคล่องโลกและความเสี่ยงของสินทรัพย์ดิจิทัล",
    ),
}

_MACRO_GROUPS: list[tuple[str, list[str]]] = [
    ("1. Yield Curve (อัตราผลตอบแทนพันธบัตร — อ่านจากซ้ายไปขวา = สั้น → ยาว)", ["^IRX", "^FVX", "^TNX", "^TYX"]),
    ("2. Risk Sentiment (ความผันผวน)", ["^VIX"]),
    ("3. Credit Market (สัญญาณความเครียดตลาดสินเชื่อ)", ["HYG", "LQD"]),
    ("4. สกุลเงิน / FX", ["DX-Y.NYB", "EURUSD=X", "USDJPY=X", "USDCNY=X"]),
    ("5. สินค้าโภคภัณฑ์ (Commodities)", ["GC=F", "CL=F", "NG=F", "HG=F"]),
    ("6. ดัชนีหุ้นสหรัฐฯ (US Equities)", ["^GSPC", "^NDX", "^RUT"]),
    ("7. สินทรัพย์ดิจิทัล", ["BTC-USD"]),
]


def _fetch_price(symbol: str) -> tuple[float | None, float | None]:
    """คืน (last_price, previous_close) พร้อม timeout ป้องกันเน็ตค้าง"""
    def _do_fetch():
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

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_do_fetch)
        try:
            return future.result(timeout=_FETCH_TIMEOUT)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Timeout >{_FETCH_TIMEOUT}s")


@tool
def ingest_daily_macro() -> str:
    """ดึงข้อมูลเศรษฐกิจมหภาค 19 ดัชนี จัดกลุ่มเป็น 7 หมวด แบบ Real-time จาก Yahoo Finance
    หมวด: Yield Curve (4), Risk Sentiment (1), Credit Market (2), FX (4), Commodities (4), US Equities (3), Digital (1)
    ดึงแบบ Parallel พร้อมกันทุก symbol — Return Markdown พร้อม YAML frontmatter
    ไม่บันทึกไฟล์ด้วยตัวเอง — Archivist เป็นผู้จัดการเซฟไฟล์
    """
    today = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    all_symbols = list(_MACRO_TICKERS.keys())
    rows_by_symbol: dict[str, dict] = {}
    errors: list[str] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(all_symbols)) as executor:
        futures = {executor.submit(_fetch_price, sym): sym for sym in all_symbols}
        for future in concurrent.futures.as_completed(futures):
            sym = futures[future]
            name, description = _MACRO_TICKERS[sym]
            try:
                last, prev = future.result()
                if last is not None:
                    change_pct = ((last - prev) / prev * 100) if prev else 0.0
                    rows_by_symbol[sym] = {
                        "symbol": sym,
                        "name": name,
                        "description": description,
                        "price": last,
                        "change_pct": change_pct,
                        "direction": "▲" if change_pct >= 0 else "▼",
                    }
                else:
                    errors.append(f"{sym}: ไม่พบข้อมูลราคา")
            except Exception as e:
                errors.append(f"{sym}: {e}")

    md_lines = [
        "---",
        f"title: Macro Snapshot {today}",
        "entity_type: macro_daily",
        f"date: {today}",
        f"last_updated: {now_time}",
        "tags: [macro, daily_snapshot, market_conditions]",
        "---",
        "",
        f"# สภาวะเศรษฐกิจมหภาค — {today}",
        "",
    ]

    for group_name, symbols in _MACRO_GROUPS:
        group_rows = [rows_by_symbol[sym] for sym in symbols if sym in rows_by_symbol]
        if not group_rows:
            continue
        md_lines += [
            f"## {group_name}",
            "",
            "| ดัชนี | ค่าล่าสุด | เปลี่ยนแปลง | ความหมาย |",
            "|-------|----------|-------------|---------|",
        ]
        for r in group_rows:
            fmt, suffix = _PRICE_FORMAT.get(r["symbol"], (".2f", ""))
            price_str = f"{r['price']:{fmt}}{suffix}"
            change_str = f"{r['direction']}{abs(r['change_pct']):.2f}%"
            md_lines.append(
                f"| **{r['name']}** (`{r['symbol']}`) | {price_str} | {change_str} | {r['description']} |"
            )
        md_lines.append("")

    md_lines += [
        "## หมายเหตุ",
        "",
        "> ข้อมูลจาก Yahoo Finance — ใช้ประเมินทิศทางตลาดเท่านั้น ไม่ใช่ราคาซื้อขายจริง",
        "",
    ]

    if errors:
        md_lines += ["## ข้อผิดพลาด", ""]
        md_lines += [f"- {e}" for e in errors]
        md_lines.append("")

    return "\n".join(md_lines)


# GICS standard order (11 sectors ครบทุกกลุ่ม)
_US_SECTORS: dict[str, tuple[str, str]] = {
    "XLC": (
        "Communication Services (สื่อสาร)",
        "Meta/Alphabet/Netflix — อ่อนไหวต่อ Ad Revenue Cycle และ Streaming Competition",
    ),
    "XLY": (
        "Consumer Discretionary (สินค้าฟุ่มเฟือย)",
        "Amazon/Tesla — ไวต่อ Consumer Confidence และ Interest Rate",
    ),
    "XLP": (
        "Consumer Staples (สินค้าจำเป็น)",
        "Walmart/P&G/Coca-Cola — Defensive หนีเข้าช่วง Risk-Off ทนต่อ Recession",
    ),
    "XLE": (
        "Energy (พลังงาน)",
        "Exxon/Chevron — เคลื่อนไหวตาม WTI/Brent และ Geopolitical Risk",
    ),
    "XLF": (
        "Financials (การเงิน/ธนาคาร)",
        "JPMorgan/Berkshire — ได้ประโยชน์เมื่อ Yield Curve ชัน เสี่ยงจาก Credit Cycle",
    ),
    "XLV": (
        "Healthcare (สุขภาพ)",
        "J&J/UnitedHealth — Defensive ทนต่อ Recession เหมาะช่วงตลาดผันผวน",
    ),
    "XLI": (
        "Industrials (อุตสาหกรรม)",
        "Caterpillar/Boeing/Honeywell — เคลื่อนไหวตาม Manufacturing PMI และ CapEx Cycle",
    ),
    "XLB": (
        "Materials (วัสดุ)",
        "เคมี/เหมืองแร่/บรรจุภัณฑ์ — สะท้อนอุปสงค์ภาคการผลิตและราคาสินค้าโภคภัณฑ์",
    ),
    "XLRE": (
        "Real Estate (อสังหาริมทรัพย์)",
        "REIT — อ่อนไหวสูงต่อ Interest Rate ได้ประโยชน์เมื่อ Fed ลด Rate",
    ),
    "XLK": (
        "Technology (เทคโนโลยี)",
        "Apple/Microsoft/Nvidia — ไวต่อ Real Rate และ Growth Expectations",
    ),
    "XLU": (
        "Utilities (สาธารณูปโภค)",
        "NextEra/Duke — Yield-sensitive แข่งกับพันธบัตร แข็งแกร่งเมื่อ Fed ลด Rate",
    ),
}


@tool
def ingest_us_sectors() -> str:
    """ดึงข้อมูล US Sector ETF ครบ 11 กลุ่ม GICS (XLC, XLY, XLP, XLE, XLF, XLV, XLI, XLB, XLRE, XLK, XLU)
    แบบ Real-time จาก Yahoo Finance ด้วย Parallel fetch พร้อม % การเปลี่ยนแปลง
    เรียงจาก Sector ที่ขึ้นมากที่สุดไปหาลดมากที่สุด เพื่อดู Sector Rotation
    Return เป็น Markdown พร้อม YAML frontmatter — ไม่บันทึกไฟล์ด้วยตัวเอง
    """
    today = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    symbols = list(_US_SECTORS.keys())

    rows: list[dict] = []
    errors: list[str] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(symbols)) as executor:
        futures = {executor.submit(_fetch_price, sym): sym for sym in symbols}
        for future in concurrent.futures.as_completed(futures):
            sym = futures[future]
            name, description = _US_SECTORS[sym]
            try:
                last, prev = future.result()
                if last is not None:
                    change_pct = ((last - prev) / prev * 100) if prev else 0.0
                    rows.append({
                        "symbol": sym,
                        "name": name,
                        "description": description,
                        "price": last,
                        "change_pct": change_pct,
                        "direction": "▲" if change_pct >= 0 else "▼",
                    })
                else:
                    errors.append(f"{sym}: ไม่พบข้อมูลราคา")
            except Exception as e:
                errors.append(f"{sym}: {e}")

    rows.sort(key=lambda r: r["change_pct"], reverse=True)

    md_lines = [
        "---",
        f"title: US Sectors Pulse {today}",
        "entity_type: us_sectors_pulse",
        f"date: {today}",
        f"last_updated: {now_time}",
        "tags: [macro, sector_rotation, us_sectors, etf_proxy]",
        "---",
        "",
        f"# กระแสเงินไหลเวียนกลุ่มอุตสาหกรรมสหรัฐฯ — {today}",
        "",
        "> เรียงจาก Sector ที่ขึ้นมากที่สุด → ลดมากที่สุด (อันดับ 1 = เงินไหลเข้ามากสุดวันนี้)",
        "",
        "| อันดับ | Sector | ETF | ราคา (USD) | เปลี่ยนแปลง | ลักษณะ |",
        "|--------|--------|-----|-----------|------------|--------|",
    ]

    for i, r in enumerate(rows, start=1):
        change_str = f"{r['direction']}{abs(r['change_pct']):.2f}%"
        md_lines.append(
            f"| {i} | **{r['name']}** | `{r['symbol']}` | {r['price']:.2f} | {change_str} | {r['description']} |"
        )

    md_lines += [
        "",
        "## หมายเหตุ",
        "",
        "> ETF ใช้เป็น Proxy กลุ่มอุตสาหกรรม (GICS Standard) — ราคาจาก Yahoo Finance ไม่ใช่ราคาซื้อขายจริง",
        "",
    ]

    if errors:
        md_lines += ["## ข้อผิดพลาด", ""]
        md_lines += [f"- {e}" for e in errors]
        md_lines.append("")

    return "\n".join(md_lines)


# Geographic order: Americas → Europe → EM Broad → Asia (Japan → India → China → AsiaPac)
_REGIONAL_TICKERS: dict[str, tuple[str, str]] = {
    "ILF": (
        "Latin America (iShares S&P Lat Am 40)",
        "ละตินอเมริกา (บราซิล/เม็กซิโก/ชิลี) — อ่อนไหวต่อ Commodity Prices และ DXY แข็งค่า",
    ),
    "VGK": (
        "Europe (Vanguard FTSE Europe)",
        "ยุโรป — ผลกระทบจาก ECB Policy วิกฤตพลังงาน และค่าเงิน EUR/USD",
    ),
    "EEM": (
        "Emerging Markets (iShares MSCI EM)",
        "ตลาดเกิดใหม่รวม — อ่อนไหวต่อ DXY แข็งค่าและ Fed Rate ขึ้น",
    ),
    "EWJ": (
        "Japan (iShares MSCI Japan)",
        "ญี่ปุ่น — ผูกพันกับ BOJ Yield Curve Control และค่าเงินเยน (USD/JPY)",
    ),
    "INDA": (
        "India (iShares MSCI India)",
        "อินเดีย — ตลาดเกิดใหม่ที่เติบโตเร็วสุด ได้ประโยชน์จาก Supply Chain Shift จากจีน",
    ),
    "MCHI": (
        "China (iShares MSCI China)",
        "จีน — สะท้อนนโยบายปักกิ่ง ความตึงเครียด US-China และสภาวะ Consumer/Tech จีน",
    ),
    "EPP": (
        "Asia Pacific ex-Japan (iShares MSCI)",
        "เอเชียแปซิฟิกยกเว้นญี่ปุ่น — ออสเตรเลีย/เกาหลีใต้/HK/สิงคโปร์",
    ),
}


@tool
def ingest_regional_pulse() -> str:
    """ดึงข้อมูล Regional Proxy ETF 7 ภูมิภาคหลัก เรียงตามภูมิศาสตร์ (Americas → Europe → EM → Asia)
    ครอบคลุม: ลาตินอเมริกา, ยุโรป, EM รวม, ญี่ปุ่น, อินเดีย, จีน, เอเชียแปซิฟิก
    แบบ Real-time จาก Yahoo Finance ด้วย Parallel fetch — Return Markdown พร้อม YAML frontmatter
    ไม่บันทึกไฟล์ด้วยตัวเอง
    """
    today = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    symbols = list(_REGIONAL_TICKERS.keys())

    rows: list[dict] = []
    errors: list[str] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(symbols)) as executor:
        futures = {executor.submit(_fetch_price, sym): sym for sym in symbols}
        for future in concurrent.futures.as_completed(futures):
            sym = futures[future]
            name, description = _REGIONAL_TICKERS[sym]
            try:
                last, prev = future.result()
                if last is not None:
                    change_pct = ((last - prev) / prev * 100) if prev else 0.0
                    rows.append({
                        "symbol": sym,
                        "name": name,
                        "description": description,
                        "price": last,
                        "change_pct": change_pct,
                        "direction": "▲" if change_pct >= 0 else "▼",
                    })
                else:
                    errors.append(f"{sym}: ไม่พบข้อมูลราคา")
            except Exception as e:
                errors.append(f"{sym}: {e}")

    order = {sym: i for i, sym in enumerate(symbols)}
    rows.sort(key=lambda r: order.get(r["symbol"], 99))

    md_lines = [
        "---",
        f"title: Regional Pulse {today}",
        "entity_type: regional_macro",
        f"date: {today}",
        f"last_updated: {now_time}",
        "tags: [macro, regional, etf_proxy]",
        "---",
        "",
        f"# ภาพรวมเศรษฐกิจรายภูมิภาค — {today}",
        "",
        "| ภูมิภาค | ETF | ราคา (USD) | เปลี่ยนแปลง | หมายเหตุ |",
        "|---------|-----|-----------|------------|---------|",
    ]

    for r in rows:
        change_str = f"{r['direction']}{abs(r['change_pct']):.2f}%"
        md_lines.append(
            f"| **{r['name']}** | `{r['symbol']}` | {r['price']:.2f} | {change_str} | {r['description']} |"
        )

    md_lines += [
        "",
        "## หมายเหตุ",
        "",
        "> ETF ใช้เป็น Proxy ภูมิภาค — ไม่ใช่ดัชนีตลาดโดยตรง ราคาจาก Yahoo Finance",
        "",
    ]

    if errors:
        md_lines += ["## ข้อผิดพลาด", ""]
        md_lines += [f"- {e}" for e in errors]
        md_lines.append("")

    return "\n".join(md_lines)


_FRED_SERIES: dict[str, tuple[str, str]] = {
    # --- Monetary Policy ---
    "FEDFUNDS": (
        "Fed Funds Rate",
        "อัตราดอกเบี้ยนโยบายสหรัฐฯ (%) — ต้นทุนการเงินโลก กำหนดโดย FOMC",
    ),
    "DGS2": (
        "2-Year Treasury Yield",
        "อัตราผลตอบแทนพันธบัตร 2 ปี — ไวต่อ Fed Policy มากสุด ใช้คู่กับ 10Y เพื่อดู Yield Curve",
    ),
    "T10Y2Y": (
        "10Y-2Y Yield Spread",
        "ส่วนต่างผลตอบแทน 10Y ลบ 2Y — ค่าติดลบ = Inverted Yield Curve สัญญาณ Recession ล่วงหน้า",
    ),
    # --- Inflation & Expectations ---
    "CPIAUCSL": (
        "CPI (YoY %)",
        "ดัชนีราคาผู้บริโภค YoY — ตัวชี้วัดเงินเฟ้อที่สาธารณชนรับรู้ ใช้กำหนด COLA",
    ),
    "PCEPI": (
        "PCE Inflation (YoY %)",
        "Personal Consumption Expenditures YoY — Headline PCE ติดตามควบคู่กับ Core PCE",
    ),
    "PCEPILFE": (
        "Core PCE Inflation (YoY %)",
        "PCE หัก Food & Energy YoY — ตัวชี้วัดเงินเฟ้อที่ Fed ใช้เป็น Primary Target จริงๆ (Target 2%)",
    ),
    "PPIACO": (
        "PPI (YoY %)",
        "ดัชนีราคาผู้ผลิต YoY — แรงกดดันเงินเฟ้อต้นน้ำ บอกก่อน CPI ประมาณ 1-3 เดือน",
    ),
    "T5YIE": (
        "5Y Breakeven Inflation Rate",
        "คาดการณ์เงินเฟ้อ 5 ปีของตลาด (TIPS spread) — forward-looking กว่า CPI สะท้อนความเชื่อมั่นต่อ Fed",
    ),
    "T10YIE": (
        "10Y Breakeven Inflation Rate",
        "คาดการณ์เงินเฟ้อ 10 ปีของตลาด — ถ้าสูงกว่า CPI = ตลาดคาดว่าเงินเฟ้อยังคงอยู่ยาวนาน",
    ),
    # --- Credit Market ---
    "BAA10Y": (
        "BAA Corporate Bond Spread",
        "ส่วนต่างพันธบัตรองค์กร Moody BAA เหนือ 10Y Treasury — ค่าสูง = ตลาดกลัว Credit Risk",
    ),
    "BAMLH0A0HYM2": (
        "High Yield Bond Spread",
        "ส่วนต่างผลตอบแทนหุ้นกู้ขยะ (ICE BofA) — ดัชนีชี้วัดความตื่นตระหนกในตลาดสินเชื่อ (Credit Risk)",
    ),
    # --- Labor Market ---
    "UNRATE": (
        "Unemployment Rate",
        "อัตราการว่างงานสหรัฐฯ (%) — ชี้วัดตลาดแรงงาน ส่วนหนึ่งของ Fed Dual Mandate",
    ),
    "ICSA": (
        "Initial Jobless Claims (K/week)",
        "ยื่นขอสวัสดิการว่างงานครั้งแรกต่อสัปดาห์ (พันคน) — Leading Indicator ตลาดแรงงาน",
    ),
    # --- Growth & Consumption ---
    "GDPC1": (
        "Real GDP (YoY %)",
        "ผลิตภัณฑ์มวลรวมแบบหักเงินเฟ้อ YoY — ชี้วัดการเติบโตจริงของเศรษฐกิจ",
    ),
    "INDPRO": (
        "Industrial Production (YoY %)",
        "ดัชนีการผลิตอุตสาหกรรม YoY — proxy ที่ดีที่สุดสำหรับ PMI ในข้อมูลฟรี ชี้ภาคการผลิต",
    ),
    "RSAFS": (
        "Retail Sales (YoY %)",
        "ยอดขายปลีก YoY — สะท้อนการบริโภคภาคเอกชน ซึ่งเป็น ~70% ของ GDP สหรัฐฯ",
    ),
    "HOUST": (
        "Housing Starts (K units/yr)",
        "จำนวนบ้านที่เริ่มก่อสร้าง (พันหลัง/ปี SAAR) — Leading Indicator Real Estate Cycle และ Recession",
    ),
    # --- Liquidity & Sentiment ---
    "M2SL": (
        "M2 Money Supply (B USD)",
        "ปริมาณเงินในระบบ M2 (พันล้านดอลลาร์) — สะท้อน Monetary Condition และ Liquidity Cycle",
    ),
    "UMCSENT": (
        "Consumer Sentiment (Index)",
        "ดัชนีความเชื่อมั่นผู้บริโภค U of Michigan — Leading Indicator การบริโภคและ Recession Risk",
    ),
}

# Series ที่ดึงเป็น % YoY ผ่าน units="pc1"
_FRED_YOY_SERIES = {"CPIAUCSL", "GDPC1", "PCEPI", "PCEPILFE", "PPIACO", "INDPRO", "RSAFS"}

# หน่วยแสดงผลต่อ series
_FRED_UNIT_DISPLAY: dict[str, str] = {
    "FEDFUNDS": "%",
    "DGS2": "%",
    "T10Y2Y": "% pts",
    "CPIAUCSL": "% YoY",
    "PCEPI": "% YoY",
    "PCEPILFE": "% YoY",
    "PPIACO": "% YoY",
    "T5YIE": "%",
    "T10YIE": "%",
    "BAA10Y": "% pts",
    "BAMLH0A0HYM2": "% pts",
    "UNRATE": "%",
    "ICSA": "K",
    "GDPC1": "% YoY",
    "INDPRO": "% YoY",
    "RSAFS": "% YoY",
    "HOUST": "K units",
    "M2SL": "B USD",
    "UMCSENT": "",
}

_FRED_GROUPS: list[tuple[str, list[str]]] = [
    ("1. นโยบายการเงิน (Monetary Policy)", ["FEDFUNDS", "DGS2", "T10Y2Y"]),
    ("2. เงินเฟ้อและคาดการณ์ (Inflation & Expectations)", ["CPIAUCSL", "PCEPI", "PCEPILFE", "PPIACO", "T5YIE", "T10YIE"]),
    ("3. ตลาดสินเชื่อ (Credit Market)", ["BAA10Y", "BAMLH0A0HYM2"]),
    ("4. ตลาดแรงงาน (Labor Market)", ["UNRATE", "ICSA"]),
    ("5. การเติบโตและการบริโภค (Growth & Consumption)", ["GDPC1", "INDPRO", "RSAFS", "HOUST"]),
    ("6. สภาพคล่องและความเชื่อมั่น (Liquidity & Sentiment)", ["M2SL", "UMCSENT"]),
]


def _fetch_fred_series(args: tuple):
    """ดึง FRED series เดี่ยว — ออกแบบมาสำหรับ ThreadPoolExecutor"""
    fred, series_id = args
    if series_id in _FRED_YOY_SERIES:
        return fred.get_series(series_id, units="pc1").dropna()
    return fred.get_series(series_id).dropna()


@tool
def ingest_economic_fundamentals() -> str:
    """ดึงตัวเลขเศรษฐกิจพื้นฐาน 19 ดัชนี จาก FRED จัดกลุ่มเป็น 6 หมวด:
    นโยบายการเงิน (Fed Rate/2Y Yield/Spread), เงินเฟ้อและคาดการณ์ (CPI/PCE/Core PCE/PPI/Breakeven 5Y+10Y),
    สินเชื่อ (BAA Spread), แรงงาน (Unemployment/Claims), การเติบโต (GDP/INDPRO/Retail/Housing),
    สภาพคล่องและความเชื่อมั่น (M2/Consumer Sentiment)
    ต้องตั้งค่า FRED_API_KEY ใน .env — Return Markdown พร้อม YAML frontmatter
    ไม่บันทึกไฟล์ด้วยตัวเอง
    """
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        return (
            "ERROR: ไม่พบ FRED_API_KEY ใน .env\n"
            "กรุณาสมัคร API Key ฟรีที่ https://fred.stlouisfed.org/docs/api/api_key.html\n"
            "แล้วเพิ่ม FRED_API_KEY=your_key_here ในไฟล์ .env"
        )

    try:
        fred = Fred(api_key=api_key)
    except Exception as e:
        return f"ERROR: ไม่สามารถเชื่อมต่อ FRED API ได้: {e}"

    today = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows_by_id: dict[str, dict] = {}
    errors: list[str] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(_FRED_SERIES)) as executor:
        futures = {
            executor.submit(_fetch_fred_series, (fred, sid)): sid
            for sid in _FRED_SERIES
        }
        for future in concurrent.futures.as_completed(futures):
            series_id = futures[future]
            name, description = _FRED_SERIES[series_id]
            try:
                raw = future.result()
                if raw.empty:
                    errors.append(f"{series_id}: ไม่พบข้อมูล")
                    continue
                latest_value = float(raw.iloc[-1])
                latest_date = raw.index[-1].strftime("%Y-%m-%d")
                unit = _FRED_UNIT_DISPLAY.get(series_id, "")
                rows_by_id[series_id] = {
                    "series_id": series_id,
                    "name": name,
                    "description": description,
                    "value": latest_value,
                    "unit": unit,
                    "date": latest_date,
                }
            except Exception as e:
                errors.append(f"{series_id}: {e}")

    md_lines = [
        "---",
        f"title: Economic Fundamentals {today}",
        "entity_type: economic_fundamentals",
        f"date: {today}",
        f"last_updated: {now_time}",
        "tags: [macro, fundamentals, fred, hard_data]",
        "---",
        "",
        f"# ตัวเลขเศรษฐกิจพื้นฐานสหรัฐฯ — {today}",
        "",
        "> ข้อมูลจาก FRED (Federal Reserve Bank of St. Louis) — ตัวเลขประกาศล่าสุดที่มีในฐานข้อมูล",
        "",
    ]

    for group_name, series_ids in _FRED_GROUPS:
        group_rows = [rows_by_id[sid] for sid in series_ids if sid in rows_by_id]
        if not group_rows:
            continue
        md_lines += [
            f"## {group_name}",
            "",
            "| ดัชนี | ค่าล่าสุด | ประกาศ ณ | ความหมาย |",
            "|-------|----------|----------|---------|",
        ]
        for r in group_rows:
            value_str = f"{r['value']:.2f} {r['unit']}".strip()
            md_lines.append(
                f"| **{r['name']}** (`{r['series_id']}`) | {value_str} | {r['date']} | {r['description']} |"
            )
        md_lines.append("")

    md_lines += [
        "## หมายเหตุ",
        "",
        "> ตัวเลข FRED มีความล่าช้า (Lagging) ตามรอบประกาศของหน่วยงาน ไม่ใช่ Real-time",
        "> CPI / PCE / PPI / GDP / INDPRO / Retail Sales แสดงเป็น % YoY",
        "> T10Y2Y = Spread 10Y − 2Y (% pts) ค่าติดลบ = Inverted Yield Curve — BAA10Y = Credit Spread เหนือ 10Y",
        "> T5YIE / T10YIE = Breakeven Inflation คาดการณ์ตลาด — Core PCE (`PCEPILFE`) คือ Primary Target ของ Fed (2%) | PCE Headline ติดตามควบคู่",
        "> ICSA = พันคน/สัปดาห์ — HOUST = พันหลัง/ปี (SAAR) — M2SL = พันล้าน USD — UMCSENT = Index (1966=100)",
        "",
    ]

    if errors:
        md_lines += ["## ข้อผิดพลาด", ""]
        md_lines += [f"- {e}" for e in errors]
        md_lines.append("")

    return "\n".join(md_lines)