"""Terminal V2 Agent Tools for Macro Quant and Equity Quant.

Provides LangChain tools for agents to query live market flows, options max pain,
benchmark rates, US Treasury yield curves, Thai public debt, prediction markets, and ETF flows.
Backed by Keyless Terminal V2 Hexagonal Architecture.
Strict Rule: Metadata provenance, delays, report dates, and limitations must be clearly reported.
"""
from typing import Optional
from langchain_core.tools import tool

from tools.market.terminal_v2.bootstrap import (
    get_terminal_data_service,
    get_terminal_service,
)
from tools.market.terminal_v2.domain.errors import (
    DataUnavailableError,
    ProviderError,
    SymbolMarketMismatchError,
)


# ============================================================================
# Phase 1 Agent Tools
# ============================================================================

@tool
def fetch_thai_market_flow(market: str = "SET") -> str:
    """ดึงข้อมูลยอดซื้อขายสุทธิ 4 กลุ่มนักลงทุน (Foreign, Institution, Proprietary, Retail) ของตลาดหุ้นไทย (SET/mai)

    [Usage/When to use]
    ใช้เมื่อต้องการวิเคราะห์กระแสเงินทุน (Fund Flow) ของตลาดหุ้นไทยแบบ Real-time
    มีข้อมูลสถิติ P/E, P/BV, Yield และ Market Breadth ร่วมด้วย
    ดึงสดผ่าน Settrade API โดยไม่ต้องใช้ API key

    Args:
        market (str): 'SET' หรือ 'mai' (ค่าเริ่มต้น 'SET')

    Returns:
        str: ข้อความสรุป Fund Flow และ Market Breadth ในรูปแบบ Markdown
    """
    service = get_terminal_service()
    try:
        flow = service.get_investor_flow(market=market)
        stats = service.get_market_valuation(market=market)
        breadth = service.get_market_breadth(market=market)
    except (DataUnavailableError, ProviderError) as exc:
        return f"⚠️ ไม่สามารถดึงข้อมูลตลาดไทยได้ในขณะนี้: {exc}"

    stale_warn = " *(Stale Snapshot)*" if flow.is_stale else ""
    lines = [
        f"### 🇹🇭 สรุปตลาดหุ้นไทย {flow.market} ({flow.as_of}){stale_warn}",
        f"- **มูลค่าการซื้อขายรวม:** {flow.total_value:,.2f} บาท",
        f"- **Breadth:** หุ้นขึ้น {breadth.gainers} | หุ้นลง {breadth.losers} | ไม่เปลี่ยนแปลง {breadth.unchanged}",
        f"- **Valuation:** P/E: {stats.pe_ratio or 'N/A'}x | P/BV: {stats.pbv_ratio or 'N/A'}x | Yield: {stats.dividend_yield or 'N/A'}%",
        "",
        "| กลุ่มนักลงทุน | ซื้อ (บาท) | ขาย (บาท) | สุทธิ (บาท) |",
        "| :--- | :---: | :---: | :---: |",
    ]
    for row in flow.investors:
        lines.append(f"| {row.name_en} | {row.buy_value:,.2f} | {row.sell_value:,.2f} | {row.net_value:,.2f} |")

    return "\n".join(lines)


@tool
def fetch_thai_retail_gold() -> str:
    """ดึงข้อมูลราคาทองคำแท่งและทองรูปพรรณ 96.5% ประกาศโดยสมาคมค้าทองคำแห่งประเทศไทย

    [Usage/When to use]
    ใช้เมื่อต้องการทราบราคาทองคำกายภาพค้าปลีกในไทยที่คนไทยซื้อขายจริงตามตู้ทอง
    (ไม่ใช่ราคา Gold Futures GC=F หรือ LBMA spot)

    Returns:
        str: รายงานราคาทองคำแท่งและรูปพรรณ (รับซื้อ/ขายออก) พร้อมรอบที่ประกาศ
    """
    service = get_terminal_service()
    try:
        gold = service.get_retail_gold()
    except (DataUnavailableError, ProviderError) as exc:
        return f"⚠️ ไม่สามารถดึงราคาทองคำสมาคมฯ ได้ในขณะนี้: {exc}"

    stale_warn = " *(Stale Snapshot)*" if gold.is_stale else ""
    rev_str = f" (ครั้งที่ {gold.revision})" if gold.revision else ""

    lines = [
        f"### 🪙 ราคาทองคำสมาคมค้าทองคำไทย{rev_str}{stale_warn}",
        f"- **ประกาศเมื่อ:** {gold.announced_at}",
        f"- **หน่วย:** {gold.unit}",
        "",
        "| ชนิดทอง 96.5% | รับซื้อ (บาท) | ขายออก (บาท) |",
        "| :--- | :---: | :---: |",
        f"| ทองคำแท่ง | {gold.bar.buy:,.2f} | {gold.bar.sell:,.2f} |",
        f"| ทองรูปพรรณ | {gold.ornament.buy:,.2f} | {gold.ornament.sell:,.2f} |",
    ]
    return "\n".join(lines)


@tool
def fetch_fred_macro_series(series_id: str) -> str:
    """ดึงข้อมูลตัวเลขเศรษฐกิจมหภาคจาก FRED (Keyless CSV)

    [Usage/When to use]
    ใช้เมื่อต้องการดึงข้อมูล Macro Time-Series สำคัญ เช่น:
    - 'CPIAUCSL' (CPI เงินเฟ้อสหรัฐฯ)
    - 'FEDFUNDS' (อัตราดอกเบี้ย Fed Funds)
    - 'DGS10' (ผลตอบแทนพันธบัตร 10 ปี)
    - 'T10Y2Y' (Yield Curve Inversion 10Y-2Y)
    - 'MORTGAGE30US' (อัตราดอกเบี้ยกู้ซื้อบ้าน 30 ปี)
    - 'VIXCLS' (ดัชนีความผันผวน VIX)

    Args:
        series_id (str): รหัส Series ของ FRED เช่น 'CPIAUCSL', 'FEDFUNDS'

    Returns:
        str: สรุปข้อมูลตัวเลขล่าสุดและย้อนหลัง 5 จุด
    """
    service = get_terminal_service()
    try:
        macro = service.get_macro_series(series_id=series_id)
    except (DataUnavailableError, ProviderError) as exc:
        return f"⚠️ ไม่สามารถดึงข้อมูล FRED Series '{series_id}' ได้ในขณะนี้: {exc}"

    if not macro.points:
        return f"ℹ️ ไม่พบข้อมูลสำหรับ Series '{series_id}'"

    latest = macro.points[-1]
    recent = macro.points[-5:] if len(macro.points) >= 5 else macro.points

    lines = [
        f"### 📊 ข้อมูล FRED: {macro.label} ({macro.series_id})",
        f"- **ความถี่:** {macro.frequency} | **หน่วย:** {macro.unit}",
        f"- **ค่าล่าสุด ณ วันที่ {latest.date}:** {latest.value:,.4f}",
        "",
        "**ประวัติย้อนหลังล่าสุด:**",
    ]
    for pt in reversed(recent):
        lines.append(f"- `{pt.date}`: {pt.value:,.4f}")

    return "\n".join(lines)


@tool
def fetch_hyperliquid_perps(symbol: str) -> str:
    """ดึงราคา Perpetual Futures สดจาก Hyperliquid (Crypto และ HIP-3 Builder DEX)

    [Usage/When to use]
    ใช้เมื่อต้องการดูราคา Real-time ของสัญญาอนุพันธ์ (Perpetuals)
    - เหรียญ Crypto ทั่วไป: 'BTC', 'ETH', 'SOL'
    - หุ้นสังเคราะห์ HIP-3 Perps: 'xyz:TSLA', 'xyz:NVDA', 'km:US500'
    (ข้อควรระวัง: สัญญา 'xyz:TSLA' เป็นอนุพันธ์ Synthetic Perp ไม่ใช่หุ้นสามัญ NASDAQ)

    Args:
        symbol (str): สัญลักษณ์สัญญา เช่น 'BTC', 'ETH', 'xyz:TSLA'

    Returns:
        str: รายงาน Mark Price, Funding Rate, และ Open Interest
    """
    service = get_terminal_service()
    try:
        perp = service.get_perps_quote(symbol=symbol)
    except SymbolMarketMismatchError as exc:
        return f"❌ ข้อผิดพลาดทางชนิดสินทรัพย์: {exc}"
    except (DataUnavailableError, ProviderError) as exc:
        return f"⚠️ ไม่สามารถดึงราคา Perpetual ของ '{symbol}' ได้ในขณะนี้: {exc}"

    lines = [
        f"### ⚡ Hyperliquid Perp: {perp.symbol}",
        f"- **Asset Class:** `{perp.asset_class}` ({perp.contract_type})",
        f"- **Mark Price:** ${perp.mark_price:,.4f}",
    ]
    if perp.funding_rate is not None:
        lines.append(f"- **Funding Rate (1h):** {perp.funding_rate * 100:.6f}%")
    if perp.open_interest is not None:
        lines.append(f"- **Open Interest:** {perp.open_interest:,.2f}")
    if perp.day_ntl_vlm is not None:
        lines.append(f"- **24h Volume (Notional):** ${perp.day_ntl_vlm:,.2f}")

    return "\n".join(lines)


# ============================================================================
# Phase 2 Agent Tools (Institutional Data Engine)
# ============================================================================

@tool
def fetch_finra_short_volume(symbol: str) -> str:
    """ดึงข้อมูลยอดขายชอร์ตรายวันของหุ้นสหรัฐฯ จาก FINRA Consolidated TRF/ADF

    [Usage/When to use]
    ใช้สำหรับตรวจสอบ Daily Short Sale Volume ของหุ้นสหรัฐฯ เช่น 'AAPL', 'NVDA', 'TSLA'
    (ข้อพึงระวังตามเกณฑ์กำกับ: ข้อมูลนี้คือ Short Sale Volume ประจำวัน ไม่ใช่ Short Interest คงค้าง
    และไม่ใช่หลักฐานการสะสมสถานะของสถาบัน)

    Args:
        symbol (str): รหัสหุ้นสหรัฐฯ เช่น 'AAPL', 'TSLA'

    Returns:
        str: สรุปปริมาณ Short Volume, Total Volume, และสัดส่วน Short %
    """
    service = get_terminal_data_service()
    try:
        res = service.get_short_volume([symbol])
        clean_sym = symbol.strip().upper()
        snap = res.get(symbol) or res.get(clean_sym)
        if not snap:
            return f"ℹ️ ไม่พบข้อมูล FINRA Short Volume สำหรับหุ้น '{symbol}'"
    except (DataUnavailableError, ProviderError) as exc:
        return f"⚠️ ไม่สามารถดึงข้อมูล FINRA Short Volume ได้ในขณะนี้: {exc}"

    pct_str = f"{snap.short_pct:.2f}%" if snap.short_pct is not None else "N/A"
    stale_str = " *(Stale Snapshot)*" if snap.is_stale else ""

    lines = [
        f"### 📉 FINRA Daily Short Volume: {snap.symbol} ({snap.report_date}){stale_str}",
        f"- **Short Volume:** {snap.short_volume:,} หุ้น",
        f"- **Short Exempt Volume:** {snap.short_exempt_volume:,} หุ้น",
        f"- **FINRA Reported Total Volume:** {snap.finra_reported_total_volume:,} หุ้น",
        f"- **Short Volume %:** {pct_str}",
        f"- **Coverage:** {snap.coverage}",
        f"- *หมายเหตุข้อจำกัด:* {snap.limitations}",
    ]
    return "\n".join(lines)


@tool
def fetch_cboe_options_max_pain(symbol: str, expiry: Optional[str] = None) -> str:
    """คำนวณ Options Max Pain และอัตราส่วน Put/Call จาก Cboe Delayed Quotes (~15 min)

    [Usage/When to use]
    ใช้ดูแนวรับแนวต้านเชิงอนุพันธ์ (Options Expiry Pain Point) และ Put/Call Ratio
    (ข้อพึงระวัง: Max Pain เป็นตัวชี้วัดประกอบคำนวณจาก Open Interest คงที่ ไม่ใช่ราคาคาดการณ์
    และไม่ใช่หลักฐานผลกำไรจริงของ Market Maker)

    Args:
        symbol (str): รหัสหุ้นสหรัฐฯ เช่น 'AAPL', 'SPY', 'NVDA'
        expiry (str, optional): วันหมดอายุที่ต้องการดู ISO YYYY-MM-DD (เว้นว่างเพื่อเลือกวันหมดอายุที่ใกล้ที่สุด)

    Returns:
        str: ค่า Strike ที่เป็น Max Pain, Spot Price, ส่วนต่าง, และ Put/Call Ratio
    """
    service = get_terminal_data_service()
    try:
        pain = service.get_options_max_pain(symbol=symbol, expiry=expiry)
        pcr = service.get_options_put_call_ratios(symbol=symbol, expiry=pain.expiry)
    except (DataUnavailableError, ProviderError, ValueError) as exc:
        return f"⚠️ ไม่สามารถคำนวณ Options Max Pain ได้ในขณะนี้: {exc}"

    spot_str = f"${pain.spot_price:,.2f}" if pain.spot_price is not None else "N/A"
    gap_str = f"{pain.distance_from_spot:+,.2f} ({pain.distance_pct:+.2f}%)" if pain.distance_from_spot is not None else "N/A"
    vol_ratio = f"{pcr.volume_ratio:.2f}" if pcr.volume_ratio is not None else "N/A"
    oi_ratio = f"{pcr.oi_ratio:.2f}" if pcr.oi_ratio is not None else "N/A"

    lines = [
        f"### 🎯 Cboe Options Max Pain: {pain.underlying} (Expiry: {pain.expiry})",
        f"- **Max Pain Strike:** ${pain.strike:,.2f}",
        f"- **Spot Price:** {spot_str} | **Gap (Spot - Max Pain):** {gap_str}",
        f"- **Put/Call Volume Ratio:** {vol_ratio} (P: {pcr.put_volume:,} / C: {pcr.call_volume:,})",
        f"- **Put/Call OI Ratio:** {oi_ratio} (P: {pcr.put_open_interest:,} / C: {pcr.call_open_interest:,})",
        f"- **Candidate Strikes Evaluated:** {pain.candidate_count} (ตัด {pain.excluded_contract_count} contracts ที่ไม่มาตรฐาน)",
        f"- *คำเตือน:* {pain.limitations}",
    ]
    return "\n".join(lines)


@tool
def fetch_nyfed_reference_rates() -> str:
    """ดึงข้อมูลอัตราดอกเบี้ยอ้างอิงข้ามคืน (Overnight Reference Rates) จาก New York Fed (SOFR, EFFR, TGCR)

    [Usage/When to use]
    ใช้สำหรับติดตามสภาพคล่องและอัตราดอกเบี้ยนโยบายตลาดการเงินสหรัฐฯ พร้อม Spread (SOFR-EFFR)

    Returns:
        str: อัตราดอกเบี้ย SOFR, EFFR, TGCR, OBFR และค่า Spreads ในหน่วย basis points (bps)
    """
    service = get_terminal_data_service()
    try:
        rates_snap = service.get_reference_rates()
    except (DataUnavailableError, ProviderError) as exc:
        return f"⚠️ ไม่สามารถดึงข้อมูล NY Fed Reference Rates ได้ในขณะนี้: {exc}"

    stale_str = " *(Stale Snapshot)*" if rates_snap.is_stale else ""
    lines = [
        f"### 🏛️ New York Fed Reference Rates ({rates_snap.as_of}){stale_str}",
        "| Benchmark | Rate (%) | Volume ($B) | Description |",
        "| :--- | :---: | :---: | :--- |",
    ]
    for r in rates_snap.rates:
        rate_val = f"{r.rate_percent:.2f}%" if r.rate_percent is not None else "N/A"
        vol_val = f"${r.volume_in_billions:,.1f}B" if r.volume_in_billions is not None else "-"
        lines.append(f"| **{r.code}** | {rate_val} | {vol_val} | {r.label} |")

    if rates_snap.spreads_bps:
        lines.append("\n**Key Spreads (Basis Points):**")
        for pair, bps in rates_snap.spreads_bps.items():
            lines.append(f"- **{pair}:** {bps:+.2f} bps")

    lines.append(f"\n*หมายเหตุ:* {rates_snap.limitations}")
    return "\n".join(lines)


@tool
def fetch_treasury_yield_curve(month: Optional[str] = None) -> str:
    """ดึงข้อมูลเส้นอัตราผลตอบแทนพันธบัตรรัฐบาลสหรัฐฯ (US Treasury Yield Curve) จาก Treasury.gov

    [Usage/When to use]
    ใช้ดู Par Yield Curve ครบทุก Tenor (1M ถึง 30Y) และ Yield Spread 10Y-2Y, 10Y-3M สำหรับประเมินภาวะ Recession

    Args:
        month (str, optional): เดือนที่ต้องการสังเกตการณ์ รูปแบบ 'YYYYMM' เช่น '202609' (ค่าเริ่มต้นเดือนปัจจุบัน)

    Returns:
        str: ตาราง Yields ครบทุก Tenor พร้อมค่า 10Y-2Y และ 10Y-3M Spreads
    """
    service = get_terminal_data_service()
    try:
        yc = service.get_treasury_yield_curve(month_yyyymm=month)
    except (DataUnavailableError, ProviderError) as exc:
        return f"⚠️ ไม่สามารถดึงข้อมูล US Treasury Yield Curve ได้ในขณะนี้: {exc}"

    lines = [
        f"### 📈 US Treasury Par Yield Curve ({yc.observation_date})",
    ]
    if yc.spread_10y_2y_bps is not None:
        lines.append(f"- **10Y - 2Y Spread:** {yc.spread_10y_2y_bps:+.1f} bps {'⚠️ (Inverted)' if yc.spread_10y_2y_bps < 0 else ''}")
    if yc.spread_10y_3m_bps is not None:
        lines.append(f"- **10Y - 3M Spread:** {yc.spread_10y_3m_bps:+.1f} bps {'⚠️ (Inverted)' if yc.spread_10y_3m_bps < 0 else ''}")

    lines.append("\n| Tenor | Yield (%) |")
    lines.append("| :--- | :---: |")
    for pt in yc.yields:
        val_str = f"{pt.yield_percent:.2f}%" if pt.yield_percent is not None else "-"
        lines.append(f"| {pt.maturity} | {val_str} |")

    return "\n".join(lines)


@tool
def fetch_thai_fund_asset_allocation() -> str:
    """ดึงสัดส่วนการจัดสรรสินทรัพย์ของอุตสาหกรรมกองทุนรวมไทย จากสำนักงาน ก.ล.ต. (SEC Thailand)

    [Usage/When to use]
    ใช้ดูโครงสร้างการลงทุนของกองทุนไทยตามประเภทสินทรัพย์ (หุ้นสามัญ, ตราสารหนี้รัฐบาล, หุ้นกู้, เงินฝาก, สินทรัพย์ต่างประเทศ)
    (ข้อพึงระวัง: รายงานนี้แสดงสัดส่วนตามประเภทสินทรัพย์ ไม่ใช่สัดส่วนรายกลุ่มอุตสาหกรรม เช่น แบงก์/พลังงาน)

    Returns:
        str: สัดส่วนการลงทุนตามประเภทสินทรัพย์ในประเทศและต่างประเทศของอุตสาหกรรมกองทุนรวม
    """
    service = get_terminal_data_service()
    try:
        alloc = service.get_thai_fund_asset_allocation()
    except (DataUnavailableError, ProviderError) as exc:
        return f"⚠️ ไม่สามารถดึงข้อมูลพอร์ตกองทุนรวมจาก ก.ล.ต. ได้ในขณะนี้: {exc}"

    nav_str = f"{alloc.total_nav_thb / 1e12:,.2f} ล้านล้านบาท" if alloc.total_nav_thb else "N/A"
    lines = [
        f"### 💼 สัดส่วนสินทรัพย์กองทุนรวมไทย ({alloc.reporting_period})",
        f"- **มูลค่าทรัพย์สินสุทธิรวม (NAV):** {nav_str}",
        "",
        "| ประเภทสินทรัพย์ | ตลาด | มูลค่า (ล้านบาท) | สัดส่วน (% NAV) |",
        "| :--- | :---: | :---: | :---: |",
    ]
    for row in alloc.allocations:
        share_str = f"{row.share_of_nav_pct:.2f}%" if row.share_of_nav_pct is not None else "-"
        lines.append(f"| {row.asset_class} | {row.domestic_or_foreign} | {row.value_thb / 1e6:,.1f} | {share_str} |")

    lines.append(f"\n*หมายเหตุ:* {alloc.limitations}")
    return "\n".join(lines)


@tool
def fetch_thai_public_debt() -> str:
    """ดึงข้อมูลหนี้สาธารณะและสัดส่วนหนี้ต่อ GDP ของประเทศไทย จากกระทรวงการคลัง (MOF Thailand)

    [Usage/When to use]
    ใช้สำหรับติดตามภาระหนี้ภาครัฐ กรอบวินัยการคลัง และสัดส่วนหนี้สาธารณะต่อ GDP รายเดือน

    Returns:
        str: ยอดหนี้สาธารณะรวม, สัดส่วนต่อ GDP, และรายละเอียด 5 หมวดหนี้ตามที่กระทรวงการคลังประกาศ
    """
    service = get_terminal_data_service()
    try:
        debt = service.get_thai_public_debt()
    except (DataUnavailableError, ProviderError) as exc:
        return f"⚠️ ไม่สามารถดึงข้อมูลหนี้สาธารณะจากกระทรวงการคลังได้ในขณะนี้: {exc}"

    gdp_pct_str = f"{debt.debt_to_gdp_pct:.2f}%" if debt.debt_to_gdp_pct is not None else "N/A"
    fx_str = f"{debt.fx_rate_usd_thb:.2f} THB/USD" if debt.fx_rate_usd_thb is not None else "N/A"

    lines = [
        f"### 🇹🇭 รายงานหนี้สาธารณะคงค้าง ({debt.reporting_month})",
        f"- **หนี้สาธารณะรวม:** {debt.total_debt_thb / 1e12:,.2f} ล้านล้านบาท",
        f"- **สัดส่วนหนี้สาธารณะต่อ GDP:** {gdp_pct_str}",
        f"- **อัตราแลกเปลี่ยนอ้างอิง:** {fx_str}",
        "",
        "| หมวดหนี้ | รายละเอียดภาษาไทย | ยอดคงค้าง (ล้านบาท) |",
        "| :--- | :--- | :---: |",
    ]
    for c in debt.components:
        lines.append(f"| {c.label_en} | {c.label_th} | {c.amount_thb / 1e6:,.1f} |")

    lines.append(f"\n*แหล่งข้อมูล:* {debt.source}")
    return "\n".join(lines)


@tool
def fetch_polymarket_prediction_markets(limit: int = 5) -> str:
    """ดึงข้อมูลตลาดทำนายผล (Prediction Markets) และราคาความน่าจะเป็นแฝงจาก Polymarket

    [Usage/When to use]
    ใช้สำหรับติดตาม Market-Implied Odds ของเหตุการณ์สำคัญ เช่น นโยบายดอกเบี้ย การเมือง และเศรษฐกิจโลก
    (ข้อพึงระวัง: ราคาคือ Market Odds ไม่ใช่การพยากรณ์อย่างเป็นทางการ)

    Args:
        limit (int): จำนวนตลาดที่ต้องการดึง (ค่าเริ่มต้น 5)

    Returns:
        str: คำถามของตลาด, ผลลัพธ์ที่เป็นไปได้, และราคา Market-Implied Probability (0-1)
    """
    service = get_terminal_data_service()
    try:
        markets = service.get_prediction_markets(limit=limit)
    except (DataUnavailableError, ProviderError) as exc:
        return f"⚠️ ไม่สามารถดึงข้อมูล Polymarket ได้ในขณะนี้: {exc}"

    lines = [
        "### 🔮 Polymarket Prediction Markets (Active Volume Ranked)",
    ]
    for m in markets:
        vol_str = f"${m.volume_24h_usd:,.0f}" if m.volume_24h_usd else "-"
        lines.append(f"\n**Q: {m.question}** (24h Vol: {vol_str})")
        for o in m.outcomes:
            prob_pct = o.price * 100.0
            lines.append(f"  - `{o.label}`: {prob_pct:.1f}% (Implied Odds: {o.price:.3f})")

    return "\n".join(lines)


@tool
def fetch_spot_etf_flows(asset: str = "BTC") -> str:
    """ดึงข้อมูลกระแสเงินทุนสุทธิ (Net Flows) ของกองทุน US Spot ETF (BTC หรือ ETH) จาก SoSoValue

    [Usage/When to use]
    ใช้ดูยอด Net Inflow/Outflow รายวัน ยอดสะสม และส่วนแบ่งราย Issuer เช่น IBIT, FBTC

    Args:
        asset (str): 'BTC' หรือ 'ETH' (ค่าเริ่มต้น 'BTC')

    Returns:
        str: ยอดสุทธิประจำวัน ยอดสะสม และรายละเอียดรายกองทุน
    """
    service = get_terminal_data_service()
    try:
        flow = service.get_spot_etf_flows(asset=asset)
    except (DataUnavailableError, ProviderError, ValueError) as exc:
        return f"⚠️ ไม่สามารถดึงข้อมูล ETF Flows สำหรับ {asset} ได้ในขณะนี้: {exc}"

    daily_str = f"${flow.daily_total_usd / 1e6:+,.2f}M" if flow.daily_total_usd is not None else "N/A"
    cum_str = f"${flow.cumulative_total_usd / 1e6:+,.2f}M" if flow.cumulative_total_usd is not None else "N/A"
    partial_str = " *(Partial Data)*" if flow.is_partial else ""

    lines = [
        f"### 🪙 US Spot {flow.asset} ETF Net Flows ({flow.report_date}){partial_str}",
        f"- **Daily Net Total:** {daily_str}",
        f"- **Cumulative Net Total:** {cum_str}",
        "",
        "| Ticker | Issuer | Daily Net ($M) | Net Assets ($M) |",
        "| :--- | :--- | :---: | :---: |",
    ]
    for i in flow.issuers:
        d_val = f"${i.daily_net_inflow_usd / 1e6:+,.2f}M" if i.daily_net_inflow_usd is not None else "-"
        a_val = f"${i.total_net_assets_usd / 1e6:,.1f}M" if i.total_net_assets_usd is not None else "-"
        lines.append(f"| **{i.ticker}** | {i.institute} | {d_val} | {a_val} |")

    lines.append(f"\n*หมายเหตุ:* {flow.limitations}")
    return "\n".join(lines)


# ============================================================================
# Phase 3 Agent Tools
# ============================================================================

@tool
def fetch_ofr_financial_stress() -> str:
    """ดึงข้อมูลดัชนีความตึงเครียดของระบบการเงินสหรัฐฯ (OFR Financial Stress Index - FSI)

    [Usage/When to use]
    ใช้เมื่อต้องการวิเคราะห์ความเสี่ยงเชิงระบบ (Systemic Risk) และสภาพคล่องในตลาดการเงิน
    ครอบคลุม 5 เสาหลัก: Credit, Equity Valuation, Safe Assets, Funding, และ Volatility

    [Caution / ข้อจำกัดข้อมูล]
    - ข้อมูล FSI มี Lag ย้อนหลัง 2 วันทำการ (T-2 business days) ณ เวลาที่รายงาน
    - ค่า 0 คือค่าเฉลี่ยทางประวัติศาสตร์ (บวก = เครียดกว่าปกติ, ลบ = สงบกว่าปกติ)
    - เป็นมาตรวัดสภาวะตลาด ณ เวลาหนึ่ง ไม่ใช่แบบจำลองพยากรณ์ความน่าจะเป็นที่จะเกิด Recession โดยตรง

    Returns:
        str: ค่าดัชนี FSI, สภาวะความเสี่ยง, และส่วนแบ่งแรงกดดันรายหมวดหมู่
    """
    service = get_terminal_data_service()
    try:
        snap = service.get_financial_stress()
    except (DataUnavailableError, ProviderError) as exc:
        return f"⚠️ ไม่สามารถดึงข้อมูล OFR Financial Stress ได้ในขณะนี้: {exc}"

    stale_str = " *(Stale Snapshot)*" if snap.is_stale else ""
    lines = [
        f"### 🛡️ US OFR Financial Stress Index (สถานะ ณ วันที่ {snap.as_of_date}){stale_str}",
        f"- **ระดับดัชนี FSI รวม:** `{snap.fsi_value:+.2f}` (เกณฑ์ 0.0 = ค่าเฉลี่ยปกติ)",
        f"- **ความล่าช้าของข้อมูล:** T-{snap.data_lag_days} วันทำการ (เผยแพร่วันที่: {snap.published_at})",
        "",
        "| เสาหลักความเสี่ยง (Category) | ค่าดัชนีเฉพาะส่วน | การประเมิน |",
        "| :--- | :---: | :--- |",
    ]
    for c in snap.categories:
        eval_str = "ตึงเครียดสูง" if c.value > 0.2 else ("ปกติ/ผ่อนคลาย" if c.value <= 0 else "เริ่มตึงตัว")
        lines.append(f"| {c.label} | `{c.value:+.2f}` | {eval_str} |")

    lines.append("\n*ข้อจำกัดข้อมูล:* ข้อมูลนี้สะท้อนสภาวะตลาดการเงินสหรัฐฯ ย้อนหลัง 2 วันทำการ ไม่ใช่การันตีวิกฤตเศรษฐกิจถดถอย")
    return "\n".join(lines)


@tool
def fetch_cftc_metals_positioning(commodity: str = "gold") -> str:
    """ดึงข้อมูลสถานะสะสมสัญญาซื้อขายล่วงหน้า (Disaggregated COT) ของตลาดโลหะมีค่า (ทองคำ, เงิน, ฯลฯ) จาก CFTC

    [Usage/When to use]
    ใช้ติดตามสถานะสะสม Long/Short ของกลุ่ม Hedge Funds (Managed Money) และผู้ผลิตจริง (Producer/Merchant)
    ในตลาดอนุพันธ์ CME/NYMEX

    [Caution / ข้อจำกัดข้อมูล]
    - ข้อมูลเป็นสถานะ ณ สิ้นวันอังคาร ซึ่งเผยแพร่ช่วงบ่ายวันศุกร์ (มีความล่าช้า 3 วันทำการ)
    - รายงาน Disaggregated แยกหมวด Managed Money (Hedge Funds/CTAs) ชัดเจน โดยไม่มีหมวดรวม Commercials ตรง ๆ
    - เป็นสัญญาซื้อขายล่วงหน้าบนตลาดอนุพันธ์ ไม่ใช่ปริมาณทองคำกายภาพในคลังจริง

    Args:
        commodity (str): 'gold', 'silver', 'copper', หรือ 'platinum' (ค่าเริ่มต้น 'gold')

    Returns:
        str: สรุปสถานะ Managed Money, ส่วนต่าง Net Long/Short, และ Percentile 52 สัปดาห์
    """
    service = get_terminal_data_service()
    try:
        snap = service.get_metals_cot(commodity=commodity)
    except (DataUnavailableError, ProviderError, ValueError) as exc:
        return f"⚠️ ไม่สามารถดึงข้อมูล CFTC COT สำหรับ {commodity} ได้ในขณะนี้: {exc}"

    mm = snap.managed_money
    prod = snap.producer_merchant
    swap = snap.swap_dealers
    stale_str = " *(Stale Snapshot)*" if snap.is_stale else ""

    lines = [
        f"### 🪙 CFTC Disaggregated COT Positioning: {snap.commodity} ({snap.as_of_date}){stale_str}",
        f"- **สัญญาคงค้างทั้งหมด (Open Interest):** {snap.open_interest:,} สัญญา",
        f"- **สถานะสุทธิ Managed Money (Hedge Funds):** `{mm.net_contracts:+d}` สัญญา",
        f"- **ระดับ Percentile 52 สัปดาห์:** `{snap.percentile_52w:.1f}%`",
        f"- **รอบการเผยแพร่:** ข้อมูลวันอังคาร {snap.as_of_date} (เผยแพร่ {snap.published_at})",
        "",
        "| กลุ่มผู้มีส่วนได้ส่วนเสีย (Trader Class) | สัญญา Long | สัญญา Short | สถานะสุทธิ (Net) | การเปลี่ยนแปลงรายสัปดาห์ |",
        "| :--- | :---: | :---: | :---: | :---: |",
        f"| **Managed Money (Hedge Funds)** | {mm.long_contracts:,} | {mm.short_contracts:,} | `{mm.net_contracts:+d}` | Long {mm.change_long:+d} / Short {mm.change_short:+d} |",
        f"| **Producer/Merchant/User** | {prod.long_contracts:,} | {prod.short_contracts:,} | `{prod.net_contracts:+d}` | - |",
        f"| **Swap Dealers** | {swap.long_contracts:,} | {swap.short_contracts:,} | `{swap.net_contracts:+d}` | - |",
    ]
    lines.append("\n*ข้อจำกัดข้อมูล:* ข้อมูลนี้แสดงสถานะสัญญาอนุพันธ์ CME/NYMEX ของนักเก็งกำไร ไม่ใช่ปริมาณทองคำกายภาพในคลัง")
    return "\n".join(lines)


@tool
def fetch_global_policy_rates() -> str:
    """ดึงข้อมูลอัตราดอกเบี้ยนโยบายของธนาคารกลางหลัก 12 ประเทศทั่วโลกจาก BIS

    [Usage/When to use]
    ใช้เปรียบเทียบดอกเบี้ยนโยบายของไทย (BOT) เทียบกับสหรัฐฯ (Fed), ยุโรป (ECB), ญี่ปุ่น (BOJ) และประเทศในเอเชีย
    เพื่อวิเคราะห์ส่วนต่างดอกเบี้ย (Interest Rate Differential) และแรงกดดันค่าเงินบาท

    [Caution / ข้อจำกัดข้อมูล]
    - วันที่มีผลบังคับใช้ (effective_date) อาจไม่ใช่วันเดียวกับวันประชุมธนาคารกลาง
    - เครื่องมือดอกเบี้ยแต่ละประเทศมีชื่อเรียกและลักษณะเฉพาะตัว (เช่น US = Fed Funds Target, TH = 1-Day Repo)
    - สิงคโปร์ (MAS) และเวียดนามไม่มีดอกเบี้ยนโยบายแบบกำหนดอัตรา (MAS ใช้กรอบ S$NEER)

    Returns:
        str: ตารางอัตราดอกเบี้ย 12 ประเทศ, วันที่มีผล, และส่วนต่าง (Spreads) เทียบกับดอกเบี้ยไทย (bps)
    """
    service = get_terminal_data_service()
    try:
        snap = service.get_global_policy_rates()
    except (DataUnavailableError, ProviderError) as exc:
        return f"⚠️ ไม่สามารถดึงข้อมูล BIS Policy Rates ได้ในขณะนี้: {exc}"

    stale_str = " *(Stale Snapshot)*" if snap.is_stale else ""
    lines = [
        f"### 🌐 BIS Central Bank Policy Rate Board ({snap.as_of_date}){stale_str}",
        "",
        "| ประเทศ | อัตราดอกเบี้ย | ตราสารนโยบาย | วันมีผลจริง | ส่วนต่างเทียบ BOT (bps) |",
        "| :--- | :---: | :--- | :---: | :---: |",
    ]
    for r in snap.rates:
        diff_bps = snap.spreads_vs_bot_repo.get(r.country, 0.0)
        spread_str = f"`{diff_bps:+.1f}` bps" if r.country != "TH" else "*(Benchmark)*"
        lines.append(f"| **{r.country}** ({r.currency}) | `{r.rate_value:.2f}%` | {r.rate_type} | {r.effective_date} | {spread_str} |")

    lines.append("\n*หมายเหตุ:* สิงคโปร์ (MAS) และเวียดนามไม่รวมอยู่ในตารางเนื่องจากไม่มีการกำหนดอัตราดอกเบี้ยคงที่")
    return "\n".join(lines)


@tool
def fetch_nasdaq_equity_consensus(symbol: str) -> str:
    """ดึงข้อมูลปฏิทินงบการเงิน, ประวัติ EPS Surprise และบทวิเคราะห์ Sell-side Consensus จาก Nasdaq

    [Usage/When to use]
    ใช้เมื่อต้องการวิเคราะห์ความแม่นยำของผลประกอบการหุ้นสหรัฐฯ (Beat/Miss Rate) และฉันทามติของนักวิเคราะห์

    [Caution / ข้อจำกัดข้อมูล]
    - ไม่รับประกันว่าจะมีข้อมูลครบทุกหุ้น (หุ้นขนาดเล็ก, ADR หรือหุ้นเข้าใหม่อาจไม่มีข้อมูลบทวิเคราะห์)
    - วันประกาศงบที่แสดงจะระบุชัดเจนว่าบริษัทยืนยันแล้ว (confirmed) หรือเป็นการประเมินตามปฏิทิน (estimated)
    - Consensus เป็นค่าเฉลี่ยคาดการณ์ของนักวิเคราะห์ ไม่ใช่ Fair Value หรือเป้าหมายที่รับประกัน

    Args:
        symbol (str): สัญลักษณ์หุ้นสหรัฐฯ เช่น 'NVDA', 'AAPL', 'MSFT'

    Returns:
        str: สรุปสถานะ Coverage, ฉันทามติ Buy/Hold/Sell, และประวัติ EPS Surprise ย้อนหลัง
    """
    clean_sym = symbol.strip().upper()
    service = get_terminal_data_service()
    try:
        snap = service.get_nasdaq_consensus(symbol=clean_sym)
    except (DataUnavailableError, ProviderError, ValueError) as exc:
        return f"⚠️ ไม่สามารถดึงข้อมูล Nasdaq Consensus สำหรับ {clean_sym} ได้ในขณะนี้: {exc}"

    if snap.coverage_status == "no_coverage":
        return f"ℹ️ หุ้น **{clean_sym}** ไม่มีข้อมูลบทวิเคราะห์หรือ EPS Surprise บน Nasdaq ในขณะนี้ (Coverage: No Coverage)"

    lines = [
        f"### 📈 Nasdaq Equity Consensus: **{clean_sym}** (Coverage: {snap.coverage_status.capitalize()})",
    ]

    if snap.ratings:
        r = snap.ratings
        lines.extend([
            f"- **Sell-side Consensus:** `{r.consensus}` (จากนักวิเคราะห์ {r.analyst_count} ราย)",
            f"- **สถาบันที่ร่วมประเมิน:** {', '.join(r.broker_names[:4]) if r.broker_names else 'N/A'}",
        ])

    if snap.upcoming_earnings:
        ue = snap.upcoming_earnings
        lines.append(f"- **วันประกาศงบรอบถัดไป:** `{ue.earnings_date}` (สถานะ: **{ue.date_status}** / ช่วงเวลา: {ue.report_time})")

    if snap.surprise_history:
        lines.extend([
            "",
            "| ไตรมาสสิ้นสุด | วันรายงานจริง | EPS จริง | Consensus คาด | Surprise % |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ])
        for s in snap.surprise_history:
            surp_fmt = f"`{s.surprise_pct:+.2f}%`" if s.surprise_pct is not None else "-"
            lines.append(f"| {s.fiscal_quarter_end} | {s.date_reported} | ${s.eps:.2f} | ${s.consensus_eps:.2f} | {surp_fmt} |")

    lines.append("\n*ข้อจำกัดข้อมูล:* ข้อมูลนี้รวบรวมจากนักวิเคราะห์ Sell-side ที่รายงานกับ Nasdaq ไม่ใช่การการันตีผลกำไรหรือมูลค่าพื้นฐาน")
    return "\n".join(lines)


# ============================================================================
# Phase 4 Agent Tools (Commodity Vol, Treasury Demand, SEC Facts & Form 4, News)
# ============================================================================

@tool
def fetch_commodity_volatility(symbol: str = "GVZ") -> str:
    """ดึงข้อมูลดัชนีความผันผวนของสินค้าโภคภัณฑ์ 30 วันจาก Cboe (GVZ=Gold, VXSLV=Silver, OVX=Crude Oil)

    [Usage/When to use]
    ใช้เมื่อต้องการวิเคราะห์ความเสี่ยงและความตื่นตระหนกในตลาดทองคำ เงิน หรือน้ำมันดิบ
    ผ่าน Implied Volatility (IV) 30 วัน พร้อมสถิติ 52-week Percentile และ Regime

    [Caution / ข้อจำกัดข้อมูล]
    - วัดจาก IV ของ Options บน ETF (GLD, SLV, USO) ไม่ใช่สัญญาฟิวเจอร์สสินค้าโภคภัณฑ์จริงโดยตรง
    - Regime Label (Complacent, Normal, Elevated, Extreme Panic) เป็นเกณฑ์สถิติ Heuristic จาก Percentile
    - Change 1D วัดเป็นหน่วย Index Points

    Args:
        symbol (str): สัญลักษณ์ดัชนี เช่น 'GVZ' (ทองคำ), 'VXSLV' (เงิน), 'OVX' (น้ำมันดิบ)

    Returns:
        str: ข้อมูล IV, การเปลี่ยนแปลง 1 วัน, 52-week Percentile, และ Regime Heuristic
    """
    clean_sym = symbol.strip().upper()
    service = get_terminal_data_service()
    try:
        snap = service.get_commodity_volatility(clean_sym)
    except (DataUnavailableError, ProviderError, ValueError) as exc:
        return f"⚠️ ไม่สามารถดึงข้อมูลความผันผวนสินค้าโภคภัณฑ์ {clean_sym} ได้ในขณะนี้: {exc}"

    stale_warn = " *(Stale Snapshot)*" if snap.is_stale else ""
    chg_str = f"{snap.change_1d_points:+.2f} pts" if snap.change_1d_points is not None else "N/A"
    pct_str = f"{snap.percentile_52w:.1f}%" if snap.percentile_52w is not None else "N/A (< 100 samples)"
    regime_str = snap.regime_label.replace("_", " ").title() if snap.regime_label else "N/A"

    lines = [
        f"### 🛢️ Cboe Commodity Volatility Index: **{snap.index_symbol}** ({snap.close_date}){stale_warn}",
        f"- **สินทรัพย์อ้างอิง:** {snap.underlying_instrument}",
        f"- **30-Day Annualized IV:** `{snap.implied_volatility:.2f}%`",
        f"- **การเปลี่ยนแปลง 1 วัน:** `{chg_str}`",
        f"- **52-Week Percentile:** `{pct_str}` (จากข้อมูล {snap.sample_count} วันซื้อขาย)",
        f"- **Volatility Regime (Heuristic):** `{regime_str}`",
        "",
        "**ข้อจำกัดข้อมูลและที่มา:**",
        f"- แหล่งข้อมูล: {snap.source} (As of {snap.as_of_date})",
    ]
    for lim in snap.limitations:
        lines.append(f"- *{lim}*")

    return "\n".join(lines)


@tool
def fetch_treasury_auction_demand(security_type: str = "Note", security_term: str = "10-Year") -> str:
    """ดึงข้อมูลผลการประมูลพันธบัตรสหรัฐฯ ล่าสุดและสรุปอุปสงค์ (Demand) เปรียบเทียบกับค่าเฉลี่ย 8 รอบก่อนหน้า

    [Usage/When to use]
    ใช้เมื่อต้องการวิเคราะห์ความต้องการซื้อพันธบัตรรัฐบาลสหรัฐฯ (Treasury Auction Demand)
    เช่น 10-Year Note, 2-Year Note, 13-Week Bill ผ่านอัตรา Bid-to-Cover และ High Yield

    [Caution / ข้อจำกัดข้อมูลที่สำคัญอย่างยิ่ง]
    - ไม่มีการคำนวณ Auction Tail เนื่องจากฐานข้อมูล US Treasury Fiscal Data ไม่มีข้อมูล Yield ของตลาดซื้อขายล่วงหน้า (When-Issued Market)
    - ค่าเฉลี่ยเคลื่อนที่ (Moving Average) คำนวณจากประวัติการประมูลที่เสร็จสิ้น 8 ครั้งก่อนหน้าของตราสารประเภทและอายุเดียวกัน (ต้องมีอย่างน้อย 3 ครั้ง)
    - Bill แสดงเป็น Discount/Investment Rate ส่วน Note/Bond แสดงเป็น High Yield

    Args:
        security_type (str): ประเภทพันธบัตร เช่น 'Note', 'Bill', 'Bond'
        security_term (str): อายุพันธบัตร เช่น '10-Year', '2-Year', '13-Week', '4-Week', '30-Year'

    Returns:
        str: สรุปผลการประมูลล่าสุด, Bid-to-Cover Ratio, ค่าเฉลี่ย 8 รอบก่อนหน้า และ Demand Delta
    """
    service = get_terminal_data_service()
    try:
        snap = service.get_auction_demand_summary(
            security_type=security_type.strip(),
            security_term=security_term.strip(),
        )
    except (DataUnavailableError, ProviderError, ValueError) as exc:
        return f"⚠️ ไม่สามารถดึงข้อมูลสรุปอุปสงค์การประมูลพันธบัตร {security_term} {security_type} ได้: {exc}"

    if not snap.latest_auction_date:
        return f"ℹ️ ไม่พบประวัติการประมูลสำหรับ {snap.security_term} {snap.security_type} ในฐานข้อมูล US Treasury"

    stale_warn = " *(Stale Snapshot)*" if snap.is_stale else ""
    btc_str = f"{snap.latest_bid_to_cover_ratio:.2f}x" if snap.latest_bid_to_cover_ratio is not None else "N/A"
    prior_mean_str = f"{snap.prior_mean_bid_to_cover:.2f}x" if snap.prior_mean_bid_to_cover is not None else "N/A (< 3 auctions)"
    delta_str = f"{snap.demand_delta:+.2f}x" if snap.demand_delta is not None else "N/A"

    yield_info = []
    if snap.latest_high_yield is not None:
        yield_info.append(f"- **High Yield ที่ประมูลได้:** `{snap.latest_high_yield:.3f}%`")
    if snap.latest_high_investment_rate is not None:
        yield_info.append(f"- **Investment Rate:** `{snap.latest_high_investment_rate:.3f}%`")
    if snap.latest_high_discount_rate is not None:
        yield_info.append(f"- **Discount Rate:** `{snap.latest_high_discount_rate:.3f}%`")

    lines = [
        f"### 🏛️ US Treasury Auction Demand: **{snap.security_term} {snap.security_type}** ({snap.latest_auction_date}){stale_warn}",
        f"- **Bid-to-Cover ล่าสุด:** `{btc_str}`",
        f"- **ค่าเฉลี่ย Bid-to-Cover (8 รอบก่อนหน้า):** `{prior_mean_str}` (จากตัวอย่าง {snap.sample_count} รอบ)",
        f"- **Demand Delta (ล่าสุด - ค่าเฉลี่ย):** `{delta_str}`",
    ]
    lines.extend(yield_info)
    if snap.latest_offering_amount_usd:
        lines.append(f"- **วงเงินที่เปิดประมูล:** ${snap.latest_offering_amount_usd:,.0f}")
    if snap.latest_total_accepted_usd:
        lines.append(f"- **วงเงินที่รับซื้อจริง:** ${snap.latest_total_accepted_usd:,.0f}")

    lines.extend([
        "",
        "**ข้อจำกัดข้อมูลและที่มา:**",
        f"- แหล่งข้อมูล: {snap.source} (As of {snap.as_of_date})",
    ])
    for lim in snap.limitations:
        lines.append(f"- *{lim}*")

    return "\n".join(lines)


@tool
def fetch_sec_financial_facts(symbol: str) -> str:
    """ดึงข้อมูลงบการเงินและอัตราส่วนทางการเงินจาก Company-filed XBRL Facts บน SEC EDGAR

    [Usage/When to use]
    ใช้เมื่อต้องการวิเคราะห์งบการเงินจริงของบริษัทจดทะเบียนสหรัฐฯ (Revenue, OCF, CapEx, FCF, หนี้ระยะยาว)
    พร้อมการคำนวณอัตราส่วนกระแสเงินสดบริสุทธิ์ (Pure Financial Ratios)

    [Caution / ข้อจำกัดข้อมูลที่ต้องระบุให้ชัดเจน]
    - เรียกว่า 'ข้อมูลที่บริษัทยื่นต่อ SEC' (company-filed XBRL facts) ไม่เหมารวมว่า audited เพราะรวมเอกสาร Form 10-Q ซึ่งผู้บริหารยื่นแบบ un-audited
    - หนี้สิน (Debt) เป็นตัวเลขคงค้าง ณ วันสิ้นงวด (Stock) ส่วนกระแสเงินสด (OCF/FCF) เป็นยอดสะสมระหว่างงวด (Flow)

    Args:
        symbol (str): สัญลักษณ์หุ้นสหรัฐฯ เช่น 'NVDA', 'AAPL', 'MSFT'

    Returns:
        str: สรุปตัวเลขรายได้, OCF, CapEx, FCF, FCF Margin, หนี้ระยะยาว และ Debt-to-OCF Ratio
    """
    clean_sym = symbol.strip().upper()
    service = get_terminal_data_service()
    try:
        snap = service.get_sec_financials(clean_sym)
    except (DataUnavailableError, ProviderError, ValueError) as exc:
        return f"⚠️ ไม่สามารถดึงข้อมูลงบ SEC EDGAR สำหรับ {clean_sym} ได้: {exc}"

    stale_warn = " *(Stale Snapshot)*" if snap.is_stale else ""
    rev_str = f"${snap.revenue_usd:,.0f}" if snap.revenue_usd is not None else "N/A"
    ocf_str = f"${snap.operating_cash_flow_usd:,.0f}" if snap.operating_cash_flow_usd is not None else "N/A"
    capex_str = f"${snap.capex_usd:,.0f}" if snap.capex_usd is not None else "N/A"
    fcf_str = f"${snap.free_cash_flow_usd:,.0f}" if snap.free_cash_flow_usd is not None else "N/A"
    fcf_margin_str = f"{snap.free_cash_flow_margin * 100:.1f}%" if snap.free_cash_flow_margin is not None else "N/A"
    debt_str = f"${snap.long_term_debt_usd:,.0f}" if snap.long_term_debt_usd is not None else "N/A"
    debt_ocf_str = f"{snap.debt_to_ocf_ratio:.2f}x" if snap.debt_to_ocf_ratio is not None else "N/A"

    lines = [
        f"### 📑 SEC EDGAR Company-Filed Facts: **{snap.entity_name}** ({snap.symbol} / CIK: {snap.cik}){stale_warn}",
        f"- **รายได้รวม (Revenue):** `{rev_str}`",
        f"- **กระแสเงินสดจากการดำเนินงาน (Operating Cash Flow):** `{ocf_str}`",
        f"- **รายจ่ายฝ่ายทุน (CapEx):** `{capex_str}`",
        f"- **กระแสเงินสดอิสระ (Free Cash Flow):** `{fcf_str}`",
        f"- **FCF Margin (FCF / Revenue):** `{fcf_margin_str}`",
        f"- **หนี้สินระยะยาว (Long-Term Debt):** `{debt_str}` (Stock ณ วันสิ้นงวด)",
        f"- **Debt-to-OCF Ratio:** `{debt_ocf_str}`",
        "",
        "**ข้อจำกัดข้อมูลและความหมาย:**",
        f"- แหล่งข้อมูล: {snap.source} (As of {snap.as_of_date})",
    ]
    for lim in snap.limitations:
        lines.append(f"- *{lim}*")

    return "\n".join(lines)


@tool
def fetch_sec_insider_trades(symbol: str, limit: int = 15) -> str:
    """ดึงข้อมูลธุรกรรมผู้บริหารและผู้ถือหุ้นรายใหญ่ (Insider Trades) จาก SEC Form 4 XML

    [Usage/When to use]
    ใช้เมื่อต้องการตรวจสอบการซื้อขายหุ้นของผู้บริหาร กรรมการ และผู้ถือหุ้น 10% (Insider Transactions)
    พร้อมการคำนวณอัตราส่วนการซื้อสุทธิ 90 วัน (90-day Net Buying Ratio)

    [Caution / ข้อจำกัดข้อมูลที่ต้องระบุให้ชัดเจน]
    - ถอดรหัสโดยตรงจากเอกสาร Form 4 XML Ownership Documents ใน SEC EDGAR
    - แยกสิทธิการถือครองของสถาบัน (Form 13F Institutional Holdings) ออกจากเครื่องมือนี้อย่างเด็ดขาด
    - การคำนวณ 90-day Net Buy Ratio นับเฉพาะธุรกรรมซื้อ (P) และขาย (S) ในตลาดที่มีราคาและจำนวนหุ้นชัดเจนเท่านั้น (ไม่นับ Award/Gift/Option Exercise)

    Args:
        symbol (str): สัญลักษณ์หุ้นสหรัฐฯ เช่น 'NVDA', 'AAPL'
        limit (int): จำนวนธุรกรรมสูงสุดที่ต้องการแสดง (ค่าเริ่มต้น 15)

    Returns:
        str: อัตราส่วน 90-day Net Buying และรายการธุรกรรมผู้บริหารล่าสุด
    """
    clean_sym = symbol.strip().upper()
    service = get_terminal_data_service()
    try:
        snap = service.get_sec_insider_trades(clean_sym, limit=limit)
    except (DataUnavailableError, ProviderError, ValueError) as exc:
        return f"⚠️ ไม่สามารถดึงข้อมูล Insider Trades สำหรับ {clean_sym} ได้: {exc}"

    stale_warn = " *(Stale Snapshot)*" if snap.is_stale else ""
    ratio_str = f"{snap.net_buy_ratio_90d:+.2f}" if snap.net_buy_ratio_90d is not None else "N/A (ไม่มีธุรกรรม P/S ที่เข้าเกณฑ์)"

    lines = [
        f"### 👔 SEC EDGAR Form 4 Insider Trades: **{snap.symbol}** (CIK: {snap.cik}){stale_warn}",
        f"- **90-Day Net Buying Ratio [-1.0 ถึง +1.0]:** `{ratio_str}`",
        f"- **มูลค่าการซื้อในตลาด (P Notional):** ${snap.p_notional_sum_90d:,.0f}",
        f"- **มูลค่าการขายในตลาด (S Notional):** ${snap.s_notional_sum_90d:,.0f}",
        f"- **จำนวนธุรกรรมที่นำมาคำนวณ:** {snap.eligible_transaction_count} ธุรกรรม",
        "",
        "| วันที่ทำรายการ | ผู้รายงาน | ตำแหน่ง | คำสั่ง | จำนวนหุ้น | ราคา/หุ้น | มูลค่า (USD) |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: |",
    ]

    for tx in snap.transactions[:limit]:
        title = tx.officer_title or ("Director" if tx.is_director else ("10% Owner" if tx.is_ten_percent_owner else "Insider"))
        shares_str = f"{tx.shares:,.0f}" if tx.shares is not None else "-"
        price_str = f"${tx.price_per_share:.2f}" if tx.price_per_share is not None else "-"
        notional_str = f"${tx.notional_usd:,.0f}" if tx.notional_usd is not None else "-"
        lines.append(
            f"| {tx.transaction_date} | {tx.reporting_owner} | {title} | **{tx.transaction_code}** | {shares_str} | {price_str} | {notional_str} |"
        )

    lines.extend([
        "",
        "**ข้อจำกัดข้อมูลและที่มา:**",
        f"- แหล่งข้อมูล: {snap.source} (As of {snap.as_of_date})",
    ])
    for lim in snap.limitations:
        lines.append(f"- *{lim}*")

    return "\n".join(lines)


@tool
def fetch_equity_news_discovery(symbol: str, limit: int = 10) -> str:
    """ค้นพบข่าวล่าสุดเกี่ยวกับหุ้นสหรัฐฯ ผ่าน RSS Discovery Aggregator

    [Usage/When to use]
    ใช้สำหรับสำรวจข่าวสารและพัฒนาการของบริษัทจดทะเบียนรายตัวเบื้องต้น
    พร้อมแสดงสำนักข่าวและเวลาที่เผยแพร่

    [Caution / ข้อจำกัดข้อมูลที่ต้องระบุให้ชัดเจน]
    - เป็นระบบค้นพบข่าวเบื้องต้น (Secondary Discovery Aggregator) ไม่ใช่สำนักข่าวต้นทาง (Primary Source)
    - ไม่รับประกันว่าเป็น Real-time และอาจเผชิญการจำกัดความถี่ (HTTP 429) ซึ่งระบบจะจัดการด้วย Graceful Degradation

    Args:
        symbol (str): สัญลักษณ์หุ้น เช่น 'NVDA', 'AAPL', 'TSLA'
        limit (int): จำนวนข่าวสูงสุดที่ต้องการ (ค่าเริ่มต้น 10)

    Returns:
        str: รายการหัวข้อข่าว แหล่งที่มา และลิงก์ไปยังบทความ
    """
    clean_sym = symbol.strip().upper()
    service = get_terminal_data_service()
    try:
        snap = service.get_ticker_news_discovery(clean_sym, limit=limit)
    except (DataUnavailableError, ProviderError, ValueError) as exc:
        return f"⚠️ ไม่สามารถค้นพบข่าวสำหรับ {clean_sym} ได้ในขณะนี้: {exc}"

    if snap.status == "rate_limited":
        return f"⚠️ ระบบค้นพบข่าวสำหรับ {clean_sym} ถูกจำกัดความถี่ชั่วคราว (Rate Limited / HTTP 429) กรุณารอสักครู่แล้วลองใหม่"

    if not snap.items:
        return f"ℹ️ ไม่พบข่าวสารที่เกี่ยวข้องกับหุ้น {clean_sym} จาก RSS Discovery ในช่วงเวลานี้"

    lines = [
        f"### 📰 News Discovery Feed: **{snap.query_symbol}** (สถานะ: {snap.status.upper()})",
    ]
    for item in snap.items[:limit]:
        stale_flag = " *(Stale)*" if item.is_stale else ""
        lines.append(f"- **[{item.publisher}]** {item.headline} ({item.published_at}){stale_flag}\n  🔗 {item.article_url}")

    lines.extend([
        "",
        "**ข้อจำกัดข้อมูลและที่มา:**",
        f"- แหล่งข้อมูล: {snap.source} (As of {snap.as_of_date})",
    ])
    for lim in snap.limitations:
        lines.append(f"- *{lim}*")

    return "\n".join(lines)


