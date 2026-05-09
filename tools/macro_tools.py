import concurrent.futures
from datetime import datetime

import yfinance as yf
from langchain_core.tools import tool

_FETCH_TIMEOUT = 10  # seconds per symbol

_MACRO_TICKERS: dict[str, tuple[str, str]] = {
    "^TNX": (
        "US 10-Year Treasury Yield",
        "อัตราผลตอบแทนพันธบัตรรัฐบาลสหรัฐ 10 ปี — ตัวชี้วัดหลักของ Risk-Free Rate และ Monetary Tightening (หน่วย: %)",
    ),
    "^VIX": (
        "VIX Fear Index",
        "ดัชนีความผันผวนของตลาด — ค่า >30 แสดงถึงความกลัวรุนแรง ค่า <20 แสดงถึงตลาดสงบ",
    ),
    "DX-Y.NYB": (
        "US Dollar Index (DXY)",
        "ความแข็งแกร่งของดอลลาร์เทียบ 6 สกุลเงินหลัก — ค่าสูงกดดัน EM Assets และสินค้าโภคภัณฑ์",
    ),
    "GC=F": (
        "Gold Futures (USD/oz)",
        "ทองคำล่วงหน้า — Safe Haven ที่มักผกผันกับ Real Interest Rate และ DXY",
    ),
    "CL=F": (
        "WTI Crude Oil (USD/bbl)",
        "น้ำมันดิบ WTI — สะท้อนอุปสงค์เศรษฐกิจโลกและต้นทุนพลังงานภาคการผลิต",
    ),
}


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
    """ดึงข้อมูลเศรษฐกิจมหภาค 5 ดัชนีหลัก (10Y Yield, VIX, DXY, Gold, Oil) แบบ Real-time
    จาก Yahoo Finance และ Return เป็น Markdown พร้อม YAML frontmatter
    ไม่บันทึกไฟล์ด้วยตัวเอง — Archivist เป็นผู้จัดการเซฟไฟล์
    """
    today = datetime.now().strftime("%Y-%m-%d")

    rows: list[dict] = []
    errors: list[str] = []

    try:
        for symbol, (name, description) in _MACRO_TICKERS.items():
            try:
                last, prev = _fetch_price(symbol)
                if last is not None:
                    change_pct = ((last - prev) / prev * 100) if prev else 0.0
                    rows.append({
                        "symbol": symbol,
                        "name": name,
                        "description": description,
                        "price": last,
                        "change_pct": change_pct,
                        "direction": "▲" if change_pct >= 0 else "▼",
                    })
                else:
                    errors.append(f"{symbol}: ไม่พบข้อมูลราคา")
            except Exception as e:
                errors.append(f"{symbol}: {e}")
    except Exception as e:
        return f"เกิดข้อผิดพลาดร้ายแรงในการเชื่อมต่อ Yahoo Finance: {e}"

    md_lines = [
        "---",
        f"title: Macro Snapshot {today}",
        "entity_type: macro_daily",
        f"date: {today}",
        "tags: [macro, daily_snapshot, market_conditions]",
        "---",
        "",
        f"# สภาวะเศรษฐกิจมหภาค — {today}",
        "",
        "| ดัชนี | ราคาล่าสุด | เปลี่ยนแปลง | ความหมาย |",
        "|-------|-----------|-------------|----------|",
    ]

    for r in rows:
        price_str = f"{r['price']:.4f}" if "Yield" in r["name"] else f"{r['price']:.2f}"
        change_str = f"{r['direction']}{abs(r['change_pct']):.2f}%"
        md_lines.append(
            f"| **{r['name']}** (`{r['symbol']}`) | {price_str} | {change_str} | {r['description']} |"
        )

    md_lines += [
        "",
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


_REGIONAL_TICKERS: dict[str, tuple[str, str]] = {
    "MCHI": (
        "China (iShares MSCI China)",
        "ตัวแทนตลาดหุ้นจีน — สะท้อนนโยบายปักกิ่งและเศรษฐกิจ Consumer/Tech จีน",
    ),
    "VGK": (
        "Europe (Vanguard FTSE Europe)",
        "ตัวแทนตลาดหุ้นยุโรป — ได้รับผลกระทบจาก ECB Policy และวิกฤตพลังงาน",
    ),
    "EEM": (
        "Emerging Markets (iShares MSCI EM)",
        "ตลาดเกิดใหม่รวม — อ่อนไหวต่อ DXY แข็งค่าและ Fed Rate ขึ้น",
    ),
    "EWJ": (
        "Japan (iShares MSCI Japan)",
        "ตัวแทนตลาดหุ้นญี่ปุ่น — ผูกพันกับ BOJ Yield Curve Control และค่าเงินเยน",
    ),
    "EPP": (
        "Asia Pacific ex-Japan",
        "เอเชียแปซิฟิกยกเว้นญี่ปุ่น — ครอบคลุมออสเตรเลีย, เกาหลีใต้, HK, สิงคโปร์",
    ),
}


@tool
def ingest_regional_pulse() -> str:
    """ดึงข้อมูล Regional Proxy ETF 5 ภูมิภาคหลัก (จีน, ยุโรป, EM, ญี่ปุ่น, เอเชียแปซิฟิก)
    แบบ Real-time จาก Yahoo Finance ด้วย Parallel fetch พร้อม % การเปลี่ยนแปลง
    Return เป็น Markdown พร้อม YAML frontmatter — ไม่บันทึกไฟล์ด้วยตัวเอง
    """
    today = datetime.now().strftime("%Y-%m-%d")
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
