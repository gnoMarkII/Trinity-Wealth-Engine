from langsmith import traceable
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
from langchain_core.tools import tool
from core.logger import get_logger
from core.retry import with_retry as _with_retry
from .core import Market, _normalize_yf_ticker, _currency_for, _yf_info, _yf_news, _yf_financials, _fmt_number, _fmt_large, _fmt_fin

log = get_logger(__name__)

@tool
def ingest_stock_fundamentals(ticker: str, market: Market = "US") -> str:
    """ดึงข้อมูลพื้นฐานของหุ้นรายตัว (Fundamentals Snapshot) จาก Yahoo Finance (รองรับ TH/US)

    [Usage/When to use]
    ใช้เมื่อต้องการข้อมูลพื้นฐาน **ณ ปัจจุบัน** ของบริษัท
    - ครอบคลุม: ข้อมูลทั่วไป, มูลค่าบริษัท, Valuation (P/E, P/B), ประสิทธิภาพ (ROE), ราคา
    - คำค้นที่เกี่ยวข้อง: "วิเคราะห์หุ้น", "valuation", "P/E", "Market Cap"

    [Caution]
    - ห้ามใช้ดึงข้อมูลย้อนหลัง (ให้ใช้ `ingest_financial_trends` แทน)
    - เครื่องมือนี้แค่ส่งคืนข้อความ Markdown (ไม่บันทึกไฟล์เอง)
    - **ต้อง** นำผลลัพธ์ที่ได้ไปส่งให้ Archivist บันทึกไฟล์ต่อด้วย `write_raw_markdown`

    Args:
        ticker (str): Ticker symbol เช่น 'AAPL', 'PTT' (ห้ามมี .BK suffix — ระบบจะเติมให้)
        market (Market): 'TH' สำหรับหุ้นไทย (SET) หรือ 'US' สำหรับหุ้นอเมริกา (default)
    """
    display_sym = ticker.strip().upper().removesuffix(".BK")
    yf_sym = _normalize_yf_ticker(display_sym, market)
    currency = _currency_for(market)
    today = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        tk = yf.Ticker(yf_sym)
        info = _with_retry(lambda: tk.info)
    except Exception as e:
        log.warning("yfinance fetch failed | %s (%s): %s", display_sym, market, e)
        return f"ERROR: ไม่สามารถดึงข้อมูล {display_sym} ({market}) ได้: {e}"

    if not info or info.get("quoteType") is None:
        return f"ERROR: ไม่พบข้อมูลสำหรับ ticker '{display_sym}' market={market} — ตรวจสอบว่า Symbol ถูกต้อง"

    # --- General ---
    short_name = info.get("shortName") or display_sym
    sector = info.get("sector") or "N/A"
    industry = info.get("industry") or "N/A"

    # --- Valuation ---
    market_cap = _fmt_large(info.get("marketCap"), currency)
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
        sustain = _with_retry(lambda: tk.sustainability)
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
    current_price = _fmt_number(info.get("currentPrice"), fmt=",.2f", suffix=f" {currency}")
    week52_change_raw = info.get("52WeekChange")
    week52_change = _fmt_number(
        week52_change_raw * 100 if week52_change_raw is not None else None,
        fmt=".2f", suffix="%"
    )
    beta = _fmt_number(info.get("beta"), fmt=".2f")

    md_lines = [
        "---",
        f"title: {display_sym} Fundamentals {today}",
        "entity_type: Company",
        f"market: {market}",
        f"date: {today}",
        f"last_updated: {now_time}",
        f"tags: [stock_analysis, {display_sym.lower()}, market_{market.lower()}]",
        "---",
        "",
        f"# {short_name} ({display_sym}, {market})",
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
        "## Related",
        "",
        f"- [[{display_sym}]]",
        "",
        "## หมายเหตุ",
        "",
        "> ข้อมูลจาก Yahoo Finance — ใช้ประกอบการวิเคราะห์เท่านั้น ไม่ใช่คำแนะนำการลงทุน",
        "",
    ]

    return "\n".join(md_lines)


@tool
def ingest_financial_health(ticker: str, market: Market = "US") -> str:
    """ดึงข้อมูลสุขภาพการเงินและกระแสเงินสดของหุ้นรายตัวจาก Yahoo Finance (รองรับ TH/US)

    [Usage/When to use]
    ใช้เมื่อต้องการดูข้อมูลสุขภาพทางการเงิน **ณ ปัจจุบัน**
    - ครอบคลุม: Operating/Free Cash Flow, Total Cash, Total Debt, Debt/Equity Ratio, Current Ratio
    - คำค้นที่เกี่ยวข้อง: "สุขภาพการเงิน", "หนี้สิน", "กระแสเงินสด", "สภาพคล่อง"

    [Caution]
    - หากผู้ใช้ถามหา "กระแสเงินสด/หนี้สิน **ย้อนหลัง**" ให้ใช้ `ingest_financial_trends` แทน
    - เครื่องมือนี้แค่ส่งคืนข้อความ Markdown (ไม่บันทึกไฟล์เอง)
    - **ต้อง** นำผลลัพธ์ที่ได้ไปส่งให้ Archivist บันทึกไฟล์ต่อด้วย `write_raw_markdown`

    Args:
        ticker (str): Ticker symbol เช่น 'AAPL', 'PTT' (ห้ามมี .BK suffix — ระบบจะเติมให้)
        market (Market): 'TH' สำหรับหุ้นไทย (SET) หรือ 'US' สำหรับหุ้นอเมริกา (default)
    """
    display_sym = ticker.strip().upper().removesuffix(".BK")
    yf_sym = _normalize_yf_ticker(display_sym, market)
    currency = _currency_for(market)
    today = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        info = _yf_info(yf_sym)
    except Exception as e:
        log.warning("yfinance fetch failed | %s (%s): %s", display_sym, market, e)
        return f"ERROR: ไม่สามารถดึงข้อมูล {display_sym} ({market}) ได้: {e}"

    if not info or info.get("quoteType") is None:
        return f"ERROR: ไม่พบข้อมูลสำหรับ ticker '{display_sym}' market={market} — ตรวจสอบว่า Symbol ถูกต้อง"

    short_name = info.get("shortName") or display_sym

    op_cf = _fmt_fin(info.get("operatingCashflow"), currency)
    free_cf = _fmt_fin(info.get("freeCashflow"), currency)
    total_cash = _fmt_fin(info.get("totalCash"), currency)
    total_debt = _fmt_fin(info.get("totalDebt"), currency)
    debt_to_equity = _fmt_number(info.get("debtToEquity"), fmt=".2f", suffix="%")
    current_ratio = _fmt_number(info.get("currentRatio"), fmt=".2f", suffix="x")

    md_lines = [
        "---",
        f"title: {display_sym} Financial Health {today}",
        "entity_type: Financial_Health",
        f"ticker: {display_sym}",
        f"market: {market}",
        f"date: {today}",
        f"last_updated: {now_time}",
        f"tags: [health_check, {display_sym.lower()}, market_{market.lower()}, stock_analysis]",
        "---",
        "",
        f"# สุขภาพการเงินและกระแสเงินสด: {short_name} ({display_sym}, {market})",
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
        "## Related",
        "",
        f"- [[{display_sym}]]",
        "",
        "## หมายเหตุ",
        "",
        f"> ข้อมูลจาก Yahoo Finance — B = พันล้าน {currency}, M = ล้าน {currency} | ใช้ประกอบการวิเคราะห์เท่านั้น",
        "> Debt/Equity แสดงในรูป % (เช่น 150% = หนี้ 1.5 เท่าของทุน)",
        "",
    ]

    return "\n".join(md_lines)
