from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
from langchain_core.tools import tool


def _fmt_number(value, fmt=".2f", suffix="") -> str:
    if value is None:
        return "N/A"
    try:
        v = float(value)
        if pd.isna(v):
            return "N/A"
        return f"{v:{fmt}}{suffix}"
    except (TypeError, ValueError):
        return "N/A"


def _fmt_large(value) -> str:
    """แปลง marketCap เป็น B/M/T เพื่ออ่านง่าย"""
    if value is None:
        return "N/A"
    try:
        v = float(value)
        if pd.isna(v):
            return "N/A"
    except (TypeError, ValueError):
        return "N/A"
    if v >= 1e12:
        return f"{v / 1e12:.2f}T USD"
    if v >= 1e9:
        return f"{v / 1e9:.2f}B USD"
    if v >= 1e6:
        return f"{v / 1e6:.2f}M USD"
    return f"{v:.0f} USD"


def _fmt_fin(value) -> str:
    """แปลงตัวเลขงบการเงินเป็น B/M USD — คืน N/A ถ้า None หรือ NaN"""
    if value is None:
        return "N/A"
    try:
        v = float(value)
        if pd.isna(v):
            return "N/A"
        neg = v < 0
        a = abs(v)
        if a >= 1e12:
            s = f"{a / 1e12:.2f}T"
        elif a >= 1e9:
            s = f"{a / 1e9:.2f}B"
        elif a >= 1e6:
            s = f"{a / 1e6:.2f}M"
        else:
            s = f"{a:.0f}"
        return f"-{s} USD" if neg else f"{s} USD"
    except (TypeError, ValueError):
        return "N/A"


def _summarize_insider_transactions(tk: yf.Ticker) -> str:
    """สรุปการซื้อ/ขายหุ้นของคนวงในใน 6 เดือนล่าสุดจาก insider_transactions"""
    try:
        df = tk.insider_transactions
        if df is None or df.empty:
            return "ไม่พบข้อมูล"

        date_col = next((c for c in ["startDate", "Start Date", "Date", "date"] if c in df.columns), None)
        tx_col = next((c for c in ["Transaction", "transaction", "Type", "type"] if c in df.columns), None)

        if date_col:
            cutoff = datetime.now() - timedelta(days=180)
            df = df.copy()
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df[df[date_col] >= cutoff]

        if df.empty:
            return "ไม่มีรายการใน 6 เดือนล่าสุด"

        if tx_col:
            tx_lower = df[tx_col].astype(str).str.lower()
            buys = int(tx_lower.str.contains(r"buy|purchase|acqui", na=False).sum())
            sells = int(tx_lower.str.contains(r"sell|sale|dispos", na=False).sum())
            if buys > sells:
                return f"ซื้อมากกว่าขาย ({buys} ซื้อ / {sells} ขาย — 6 เดือนล่าสุด)"
            if sells > buys:
                return f"ขายมากกว่าซื้อ ({sells} ขาย / {buys} ซื้อ — 6 เดือนล่าสุด)"
            return f"ซื้อและขายเท่ากัน ({buys} รายการ — 6 เดือนล่าสุด)"

        return f"มี {len(df)} รายการใน 6 เดือนล่าสุด (ไม่สามารถแยกประเภทได้)"
    except Exception:
        return "N/A"


@tool
def ingest_stock_fundamentals(ticker: str) -> str:
    """ดึงข้อมูลพื้นฐานของหุ้นสหรัฐฯ รายตัวจาก Yahoo Finance
    ครอบคลุม: ข้อมูลทั่วไป, มูลค่าบริษัท, Valuation, ประสิทธิภาพ, ราคา
    Return เป็น Markdown พร้อม YAML frontmatter — ไม่บันทึกไฟล์ด้วยตัวเอง

    Args:
        ticker: Ticker symbol เช่น 'AAPL', 'MSFT', 'NVDA'
    """
    ticker_upper = ticker.strip().upper()
    today = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        tk = yf.Ticker(ticker_upper)
        info = tk.info
    except Exception as e:
        return f"ERROR: ไม่สามารถดึงข้อมูล {ticker_upper} ได้: {e}"

    if not info or info.get("quoteType") is None:
        return f"ERROR: ไม่พบข้อมูลสำหรับ ticker '{ticker_upper}' — ตรวจสอบว่า Symbol ถูกต้อง"

    # --- General ---
    short_name = info.get("shortName") or ticker_upper
    sector = info.get("sector") or "N/A"
    industry = info.get("industry") or "N/A"

    # --- Valuation ---
    market_cap = _fmt_large(info.get("marketCap"))
    trailing_pe = _fmt_number(info.get("trailingPE"))
    forward_pe = _fmt_number(info.get("forwardPE"))
    price_to_book = _fmt_number(info.get("priceToBook"))
    ev_to_ebitda = _fmt_number(info.get("enterpriseToEbitda"))
    # trailingAnnualDividendYield คืนค่า decimal (0.0036 = 0.36%) ต่างจาก dividendYield ที่คืน 0.37 (= 0.37% แล้ว)
    dividend_yield_raw = info.get("trailingAnnualDividendYield")
    dividend_yield = _fmt_number(
        dividend_yield_raw * 100 if dividend_yield_raw is not None else None,
        fmt=".2f", suffix="%"
    )
    payout_ratio_raw = info.get("payoutRatio")
    payout_ratio = _fmt_number(
        payout_ratio_raw * 100 if payout_ratio_raw is not None else None,
        fmt=".2f", suffix="%"
    )

    # --- ESG Scores (optional — ไม่มีข้อมูลสำหรับหุ้นบางตัว) ---
    esg_section: list[str] = []
    try:
        sustain = tk.sustainability
        if sustain is not None and not sustain.empty:
            def _esg_val(df, keys: list[str]):
                for k in keys:
                    if k in df.index:
                        v = df.loc[k]
                        v = v.iloc[0] if hasattr(v, "iloc") else v
                        try:
                            return float(v)
                        except (TypeError, ValueError):
                            pass
                return None

            total_esg = _esg_val(sustain, ["totalEsg", "Total ESG Risk score"])
            env_score = _esg_val(sustain, ["environmentScore", "Environment Risk Score", "environmentalScore"])
            soc_score = _esg_val(sustain, ["socialScore", "Social Risk Score"])
            gov_score = _esg_val(sustain, ["governanceScore", "Governance Risk Score"])

            if any(v is not None for v in [total_esg, env_score, soc_score, gov_score]):
                def _fe(v): return f"{v:.1f}" if v is not None else "N/A"
                esg_section = [
                    "## คะแนนความยั่งยืน ESG (Sustainalytics Risk Score)",
                    "",
                    "| ดัชนี | คะแนน | ความหมาย |",
                    "|-------|-------|---------|",
                    f"| **Total ESG Risk** | {_fe(total_esg)} | คะแนนรวม — ต่ำ = ความเสี่ยงด้าน ESG น้อย |",
                    f"| **Environment** | {_fe(env_score)} | ความเสี่ยงด้านสิ่งแวดล้อม |",
                    f"| **Social** | {_fe(soc_score)} | ความเสี่ยงด้านสังคมและแรงงาน |",
                    f"| **Governance** | {_fe(gov_score)} | ความเสี่ยงด้านธรรมาภิบาล |",
                    "",
                    "> *Sustainalytics Risk Score: 0–10 = Negligible, 10–20 = Low, 20–30 = Medium, 30–40 = High, 40+ = Severe*",
                    "",
                ]
    except Exception:
        pass

    # --- Profitability ---
    gross_margins_raw = info.get("grossMargins")
    gross_margins = _fmt_number(
        gross_margins_raw * 100 if gross_margins_raw is not None else None,
        fmt=".2f", suffix="%"
    )
    profit_margins_raw = info.get("profitMargins")
    profit_margins = _fmt_number(
        profit_margins_raw * 100 if profit_margins_raw is not None else None,
        fmt=".2f", suffix="%"
    )
    roe_raw = info.get("returnOnEquity")
    roe = _fmt_number(
        roe_raw * 100 if roe_raw is not None else None,
        fmt=".2f", suffix="%"
    )
    rev_growth_raw = info.get("revenueGrowth")
    rev_growth = _fmt_number(
        rev_growth_raw * 100 if rev_growth_raw is not None else None,
        fmt=".2f", suffix="%"
    )

    # --- Price ---
    current_price = _fmt_number(info.get("currentPrice"), fmt=",.2f", suffix=" USD")
    week52_change_raw = info.get("52WeekChange")
    week52_change = _fmt_number(
        week52_change_raw * 100 if week52_change_raw is not None else None,
        fmt=".2f", suffix="%"
    )
    beta = _fmt_number(info.get("beta"), fmt=".2f")

    md_lines = [
        "---",
        f"title: {ticker_upper} Fundamentals {today}",
        "entity_type: Company",
        f"date: {today}",
        f"last_updated: {now_time}",
        f"tags: [stock_analysis, {ticker_upper.lower()}]",
        "---",
        "",
        f"# {short_name} ({ticker_upper})",
        "",
        "## ข้อมูลทั่วไป",
        "",
        f"| รายการ | ข้อมูล |",
        f"|--------|--------|",
        f"| **Sector** | {sector} |",
        f"| **Industry** | {industry} |",
        "",
    ]

    md_lines += [
        "## มูลค่าและ Valuation",
        "",
        "| ดัชนี | ค่า | ความหมาย |",
        "|-------|-----|---------|",
        f"| **Market Cap** | {market_cap} | มูลค่าตลาดรวม |",
        f"| **Trailing P/E** | {trailing_pe} | P/E จากกำไรย้อนหลัง 12 เดือน |",
        f"| **Forward P/E** | {forward_pe} | P/E จากกำไรคาดการณ์ล่วงหน้า |",
        f"| **Price/Book** | {price_to_book} | ราคาต่อมูลค่าทางบัญชี |",
        f"| **EV/EBITDA** | {ev_to_ebitda} | มูลค่ากิจการต่อ EBITDA — ต่ำกว่า Peer = Undervalued |",
        f"| **Dividend Yield** | {dividend_yield} | อัตราปันผลต่อราคาหุ้น |",
        f"| **Payout Ratio** | {payout_ratio} | สัดส่วนกำไรที่จ่ายเป็นปันผล — >100% = จ่ายเกินกำไร |",
        "",
        "## ประสิทธิภาพการดำเนินงาน",
        "",
        "| ดัชนี | ค่า | ความหมาย |",
        "|-------|-----|---------|",
        f"| **Gross Margin** | {gross_margins} | อัตรากำไรขั้นต้น — Pricing Power ก่อนหัก SG&A/ดอกเบี้ย |",
        f"| **Profit Margin** | {profit_margins} | อัตรากำไรสุทธิหลังหักทุกอย่างแล้ว |",
        f"| **Return on Equity (ROE)** | {roe} | ผลตอบแทนต่อส่วนของผู้ถือหุ้น |",
        f"| **Revenue Growth (YoY)** | {rev_growth} | การเติบโตของรายได้เทียบปีก่อน |",
        "",
        "## ราคาและความเคลื่อนไหว",
        "",
        "| ดัชนี | ค่า |",
        "|-------|-----|",
        f"| **Current Price** | {current_price} |",
        f"| **52-Week Change** | {week52_change} |",
        f"| **Beta** | {beta} |",
        "",
    ]

    if esg_section:
        md_lines += esg_section

    md_lines += [
        "## หมายเหตุ",
        "",
        "> ข้อมูลจาก Yahoo Finance — ใช้ประกอบการวิเคราะห์เท่านั้น ไม่ใช่คำแนะนำการลงทุน",
        "",
    ]

    return "\n".join(md_lines)


@tool
def ingest_stock_news(ticker: str) -> str:
    """ดึงพาดหัวข่าวล่าสุด 5 ข่าวของหุ้นสหรัฐฯ รายตัวจาก Yahoo Finance
    แสดง title, publisher, และ link — Return Markdown พร้อม YAML frontmatter
    ไม่บันทึกไฟล์ด้วยตัวเอง

    Args:
        ticker: Ticker symbol เช่น 'AAPL', 'TSLA', 'NVDA'
    """
    ticker_upper = ticker.strip().upper()
    today = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        news_list = yf.Ticker(ticker_upper).news
    except Exception as e:
        return f"ERROR: ไม่สามารถดึงข่าวของ {ticker_upper} ได้: {e}"

    if not news_list:
        return f"ERROR: ไม่พบข่าวสำหรับ {ticker_upper}"

    items = news_list[:5]

    md_lines = [
        "---",
        f"title: {ticker_upper} Latest News {today}",
        "entity_type: Company_News",
        f"ticker: {ticker_upper}",
        f"date: {today}",
        f"last_updated: {now_time}",
        f"tags: [news, {ticker_upper.lower()}, stock_analysis]",
        "---",
        "",
        f"# ข่าวล่าสุด: {ticker_upper}",
        "",
    ]

    for i, item in enumerate(items, start=1):
        content = item.get("content") or {}
        title = content.get("title") or item.get("title") or "ไม่มีชื่อข่าว"
        provider = (content.get("provider") or {}).get("displayName") or item.get("publisher") or "N/A"
        url = (content.get("canonicalUrl") or {}).get("url") or item.get("link") or ""
        link_md = f"[อ่านต่อ]({url})" if url else "N/A"
        md_lines += [
            f"{i}. **{title}**",
            f"   - ที่มา: {provider}",
            f"   - {link_md}",
            "",
        ]

    md_lines += [
        "## หมายเหตุ",
        "",
        "> ข่าวจาก Yahoo Finance — ใช้ประกอบการวิเคราะห์เท่านั้น",
        "",
    ]

    return "\n".join(md_lines)


@tool
def ingest_stock_consensus(ticker: str) -> str:
    """ดึงมุมมองนักวิเคราะห์ (Analyst Consensus) ของหุ้นสหรัฐฯ รายตัวจาก Yahoo Finance
    ครอบคลุม: ราคาเป้าหมาย (ต่ำ/เฉลี่ย/สูง), คำแนะนำ, จำนวนนักวิเคราะห์
    Return เป็น Markdown พร้อม YAML frontmatter — ไม่บันทึกไฟล์ด้วยตัวเอง

    Args:
        ticker: Ticker symbol เช่น 'AAPL', 'TSLA', 'NVDA'
    """
    ticker_upper = ticker.strip().upper()
    today = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        info = yf.Ticker(ticker_upper).info
    except Exception as e:
        return f"ERROR: ไม่สามารถดึงข้อมูล {ticker_upper} ได้: {e}"

    if not info or info.get("quoteType") is None:
        return f"ERROR: ไม่พบข้อมูลสำหรับ ticker '{ticker_upper}' — ตรวจสอบว่า Symbol ถูกต้อง"

    target_low = _fmt_number(info.get("targetLowPrice"), fmt=",.2f", suffix=" USD")
    target_mean = _fmt_number(info.get("targetMeanPrice"), fmt=",.2f", suffix=" USD")
    target_high = _fmt_number(info.get("targetHighPrice"), fmt=",.2f", suffix=" USD")
    current_price = _fmt_number(info.get("currentPrice"), fmt=",.2f", suffix=" USD")
    recommendation = (info.get("recommendationKey") or "N/A").upper()
    num_analysts = info.get("numberOfAnalystOpinions")
    num_analysts_str = str(num_analysts) if num_analysts is not None else "N/A"

    _cur_raw = info.get("currentPrice")
    _mean_raw = info.get("targetMeanPrice")
    if _cur_raw and _mean_raw and float(_cur_raw) != 0:
        _upside_val = (float(_mean_raw) - float(_cur_raw)) / float(_cur_raw) * 100
        upside_pct = f"{_upside_val:+.1f}%"
    else:
        upside_pct = "N/A"

    short_name = info.get("shortName") or ticker_upper

    md_lines = [
        "---",
        f"title: {ticker_upper} Analyst Consensus {today}",
        "entity_type: Analyst_Consensus",
        f"ticker: {ticker_upper}",
        f"date: {today}",
        f"last_updated: {now_time}",
        f"tags: [analyst_consensus, {ticker_upper.lower()}, stock_analysis]",
        "---",
        "",
        f"# มุมมองนักวิเคราะห์: {short_name} ({ticker_upper})",
        "",
        "## Analyst Consensus",
        "",
        "| รายการ | ค่า | ความหมาย |",
        "|--------|-----|---------|",
        f"| **ราคาปัจจุบัน** | {current_price} | ราคาตลาดล่าสุด |",
        f"| **Target Low** | {target_low} | ราคาเป้าหมายต่ำสุดของนักวิเคราะห์ |",
        f"| **Target Mean** | {target_mean} | ราคาเป้าหมายเฉลี่ยของนักวิเคราะห์ |",
        f"| **Upside to Target** | {upside_pct} | % จากราคาปัจจุบันถึง Target Mean (+ = Upside, − = Downside) |",
        f"| **Target High** | {target_high} | ราคาเป้าหมายสูงสุดของนักวิเคราะห์ |",
        f"| **Recommendation** | {recommendation} | คำแนะนำรวม (BUY/HOLD/SELL) |",
        f"| **จำนวนนักวิเคราะห์** | {num_analysts_str} คน | จำนวนนักวิเคราะห์ที่ให้ความเห็น |",
        "",
        "## หมายเหตุ",
        "",
        "> ข้อมูลจาก Yahoo Finance — Consensus มาจากนักวิเคราะห์ Wall Street ใช้ประกอบการตัดสินใจเท่านั้น",
        "",
    ]

    return "\n".join(md_lines)


@tool
def ingest_financial_trends(ticker: str) -> str:
    """ดึงแนวโน้มงบการเงินย้อนหลัง (รายได้รวม + กำไรสุทธิ) ของหุ้นสหรัฐฯ รายตัวจาก Yahoo Finance
    ครอบคลุม: Total Revenue และ Net Income ย้อนหลัง 4 ปีงบการเงิน
    Return เป็น Markdown พร้อม YAML frontmatter — ไม่บันทึกไฟล์ด้วยตัวเอง

    Args:
        ticker: Ticker symbol เช่น 'AAPL', 'MSFT', 'NVDA'
    """
    ticker_upper = ticker.strip().upper()
    today = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        tk = yf.Ticker(ticker_upper)
        financials = tk.financials
        short_name = tk.info.get("shortName") or ticker_upper
    except Exception as e:
        return f"ERROR: ไม่สามารถดึงข้อมูล {ticker_upper} ได้: {e}"

    if financials is None or financials.empty:
        return f"ERROR: ไม่พบข้อมูลงบการเงินสำหรับ {ticker_upper}"

    def _find_series(df: pd.DataFrame, candidates: list[str]):
        for key in candidates:
            if key in df.index:
                return df.loc[key]
        return None

    revenue_s = _find_series(financials, ["Total Revenue", "TotalRevenue"])
    net_income_s = _find_series(financials, [
        "Net Income", "NetIncome",
        "Net Income Common Stockholders", "NetIncomeCommonStockholders",
    ])

    cols = list(financials.columns[:4])

    md_lines = [
        "---",
        f"title: {ticker_upper} Financial Trends {today}",
        "entity_type: Financial_Trends",
        f"ticker: {ticker_upper}",
        f"date: {today}",
        f"last_updated: {now_time}",
        f"tags: [financial_trends, {ticker_upper.lower()}, stock_analysis]",
        "---",
        "",
        f"# แนวโน้มงบการเงิน: {short_name} ({ticker_upper})",
        "",
        "## รายได้รวมและกำไรสุทธิ (ย้อนหลัง 4 ปีงบการเงิน)",
        "",
        "| ปีงบการเงิน | รายได้รวม | กำไรสุทธิ |",
        "|------------|----------|----------|",
    ]

    for col in cols:
        year_str = col.strftime("%Y") if hasattr(col, "strftime") else str(col)[:4]
        rev = _fmt_fin(revenue_s[col] if revenue_s is not None else None)
        ni = _fmt_fin(net_income_s[col] if net_income_s is not None else None)
        md_lines.append(f"| {year_str} | {rev} | {ni} |")

    md_lines += [
        "",
        "## หมายเหตุ",
        "",
        "> ข้อมูลจาก Yahoo Finance — B = พันล้าน USD, M = ล้าน USD, T = ล้านล้าน USD",
        "",
    ]

    notices = []
    if revenue_s is None:
        notices.append(f"ไม่พบแถว 'Total Revenue' — Index ที่มี: {list(financials.index[:8])}")
    if net_income_s is None:
        notices.append(f"ไม่พบแถว 'Net Income' — Index ที่มี: {list(financials.index[:8])}")

    if notices:
        md_lines += ["## ข้อผิดพลาด", ""]
        md_lines += [f"- {n}" for n in notices]
        md_lines.append("")

    return "\n".join(md_lines)


@tool
def ingest_stock_momentum(ticker: str) -> str:
    """ดึงข้อมูลสัญญาณเทคนิค/โมเมนตัมราคา ข้อมูลคนวงใน สถาบัน และ Short Interest ของหุ้นสหรัฐฯ
    ครอบคลุม: MA50, MA200, 52W High/Low, % Insider/Institution Hold, Short Ratio, Short % Float
    Return เป็น Markdown พร้อม YAML frontmatter — ไม่บันทึกไฟล์ด้วยตัวเอง

    Args:
        ticker: Ticker symbol เช่น 'AAPL', 'MSFT', 'NVDA'
    """
    ticker_upper = ticker.strip().upper()
    today = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        tk = yf.Ticker(ticker_upper)
        info = tk.info
    except Exception as e:
        return f"ERROR: ไม่สามารถดึงข้อมูล {ticker_upper} ได้: {e}"

    if not info or info.get("quoteType") is None:
        return f"ERROR: ไม่พบข้อมูลสำหรับ ticker '{ticker_upper}' — ตรวจสอบว่า Symbol ถูกต้อง"

    short_name = info.get("shortName") or ticker_upper
    cur = info.get("currentPrice")
    m50 = info.get("fiftyDayAverage")
    m200 = info.get("twoHundredDayAverage")

    current_price = _fmt_number(cur, fmt=",.2f", suffix=" USD")
    ma50 = _fmt_number(m50, fmt=",.2f", suffix=" USD")
    ma200 = _fmt_number(m200, fmt=",.2f", suffix=" USD")
    high52 = _fmt_number(info.get("fiftyTwoWeekHigh"), fmt=",.2f", suffix=" USD")
    low52 = _fmt_number(info.get("fiftyTwoWeekLow"), fmt=",.2f", suffix=" USD")

    signal_parts = []
    if cur and m50:
        signal_parts.append("เหนือ MA50 ✓" if cur > m50 else "ใต้ MA50 ✗")
    if cur and m200:
        signal_parts.append("เหนือ MA200 ✓" if cur > m200 else "ใต้ MA200 ✗")
    signal_str = " | ".join(signal_parts) if signal_parts else "N/A"

    # Insider
    insider_held_raw = info.get("heldPercentInsiders")
    insider_held = _fmt_number(
        insider_held_raw * 100 if insider_held_raw is not None else None,
        fmt=".2f", suffix="%"
    )
    insider_tx_summary = _summarize_insider_transactions(tk)

    # Institution & Short Interest
    inst_held_raw = info.get("heldPercentInstitutions")
    inst_held = _fmt_number(
        inst_held_raw * 100 if inst_held_raw is not None else None,
        fmt=".2f", suffix="%"
    )
    short_ratio = _fmt_number(info.get("shortRatio"), fmt=".2f", suffix=" วัน")
    short_pct_raw = info.get("sharesPercentSharesOut")
    short_pct = _fmt_number(
        short_pct_raw * 100 if short_pct_raw is not None else None,
        fmt=".2f", suffix="%"
    )

    md_lines = [
        "---",
        f"title: {ticker_upper} Momentum Insider {today}",
        "entity_type: Stock_Momentum",
        f"ticker: {ticker_upper}",
        f"date: {today}",
        f"last_updated: {now_time}",
        f"tags: [stock_momentum, {ticker_upper.lower()}, stock_analysis, technical]",
        "---",
        "",
        f"# โมเมนตัมราคาและคนวงใน: {short_name} ({ticker_upper})",
        "",
        "## สัญญาณเทคนิค (Technical Signals)",
        "",
        "| ดัชนี | ค่า | ความหมาย |",
        "|-------|-----|---------|",
        f"| **ราคาปัจจุบัน** | {current_price} | ราคาตลาดล่าสุด |",
        f"| **MA50** | {ma50} | ค่าเฉลี่ยเคลื่อนที่ 50 วัน |",
        f"| **MA200** | {ma200} | ค่าเฉลี่ยเคลื่อนที่ 200 วัน |",
        f"| **52W High** | {high52} | ราคาสูงสุดใน 52 สัปดาห์ |",
        f"| **52W Low** | {low52} | ราคาต่ำสุดใน 52 สัปดาห์ |",
        "",
        f"> **สัญญาณ:** {signal_str}",
        "",
        "## ข้อมูลคนวงใน (Insider Activity)",
        "",
        "| รายการ | ค่า |",
        "|--------|-----|",
        f"| **% Insider Hold** | {insider_held} |",
        f"| **ซื้อ/ขาย (6 เดือน)** | {insider_tx_summary} |",
        "",
        "## พฤติกรรมสถาบันและการชอร์ตเซล (Institution & Short Interest)",
        "",
        "| ดัชนี | ค่า | ความหมาย |",
        "|-------|-----|---------|",
        f"| **% Institution Hold** | {inst_held} | สัดส่วนหุ้นที่สถาบัน (กองทุน/บริษัท) ถือครอง — สูง = ความเชื่อมั่นสถาบัน |",
        f"| **Short Ratio** | {short_ratio} | จำนวนวันที่ต้องใช้ปิด Short ทั้งหมด — >5 วัน = ความเสี่ยง Short Squeeze สูง |",
        f"| **Short % of Float** | {short_pct} | % หุ้นที่ถูก Short เทียบหุ้นที่ซื้อขายได้ — >10% = มี Bearish Sentiment สูง |",
        "",
        "## หมายเหตุ",
        "",
        "> ข้อมูลจาก Yahoo Finance — MA = Moving Average | Insider Hold = % หุ้นที่ผู้บริหารถือครอง",
        "> Institution Hold = กองทุน/บริษัทใหญ่ | Short Ratio = Days to Cover | Short % Float = Short Interest",
        "",
    ]

    return "\n".join(md_lines)


@tool
def ingest_financial_health(ticker: str) -> str:
    """ดึงข้อมูลสุขภาพการเงินและกระแสเงินสดของหุ้นสหรัฐฯ รายตัวจาก Yahoo Finance
    ครอบคลุม: Operating/Free Cash Flow, Total Cash, Total Debt, Debt/Equity Ratio, Current Ratio
    Return เป็น Markdown พร้อม YAML frontmatter — ไม่บันทึกไฟล์ด้วยตัวเอง

    Args:
        ticker: Ticker symbol เช่น 'AAPL', 'MSFT', 'NVDA'
    """
    ticker_upper = ticker.strip().upper()
    today = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        info = yf.Ticker(ticker_upper).info
    except Exception as e:
        return f"ERROR: ไม่สามารถดึงข้อมูล {ticker_upper} ได้: {e}"

    if not info or info.get("quoteType") is None:
        return f"ERROR: ไม่พบข้อมูลสำหรับ ticker '{ticker_upper}' — ตรวจสอบว่า Symbol ถูกต้อง"

    short_name = info.get("shortName") or ticker_upper

    op_cf = _fmt_fin(info.get("operatingCashflow"))
    free_cf = _fmt_fin(info.get("freeCashflow"))
    total_cash = _fmt_fin(info.get("totalCash"))
    total_debt = _fmt_fin(info.get("totalDebt"))
    debt_to_equity = _fmt_number(info.get("debtToEquity"), fmt=".2f", suffix="%")
    current_ratio = _fmt_number(info.get("currentRatio"), fmt=".2f", suffix="x")

    md_lines = [
        "---",
        f"title: {ticker_upper} Financial Health {today}",
        "entity_type: Financial_Health",
        f"ticker: {ticker_upper}",
        f"date: {today}",
        f"last_updated: {now_time}",
        f"tags: [health_check, {ticker_upper.lower()}, stock_analysis]",
        "---",
        "",
        f"# สุขภาพการเงินและกระแสเงินสด: {short_name} ({ticker_upper})",
        "",
        "## กระแสเงินสด (Cash Flow)",
        "",
        "| ดัชนี | ค่า | ความหมาย |",
        "|-------|-----|---------|",
        f"| **Operating Cash Flow** | {op_cf} | กระแสเงินสดจากการดำเนินงานประจำปี |",
        f"| **Free Cash Flow** | {free_cf} | กระแสเงินสดหลังหัก CapEx — ใช้คืนหนี้/ปันผล/ซื้อหุ้นคืน |",
        "",
        "## งบดุลและหนี้สิน (Balance Sheet & Debt)",
        "",
        "| ดัชนี | ค่า | ความหมาย |",
        "|-------|-----|---------|",
        f"| **Total Cash** | {total_cash} | เงินสดและรายการเทียบเท่าเงินสดรวม |",
        f"| **Total Debt** | {total_debt} | หนี้สินรวมทั้งระยะสั้นและระยะยาว |",
        f"| **Debt/Equity** | {debt_to_equity} | อัตราส่วนหนี้ต่อทุน — ค่าสูง = leverage สูง |",
        f"| **Current Ratio** | {current_ratio} | สภาพคล่องระยะสั้น — >1x = ปลอดภัย, <1x = เสี่ยง |",
        "",
        "## หมายเหตุ",
        "",
        "> ข้อมูลจาก Yahoo Finance — B = พันล้าน USD, M = ล้าน USD | ใช้ประกอบการวิเคราะห์เท่านั้น",
        "> Debt/Equity แสดงในรูป % (เช่น 150% = หนี้ 1.5 เท่าของทุน)",
        "",
    ]

    return "\n".join(md_lines)
