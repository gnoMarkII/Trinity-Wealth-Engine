import concurrent.futures

import os

from datetime import datetime

import yfinance as yf

from fredapi import Fred

from langchain_core.tools import tool

from core.logger import get_logger

from core.retry import with_retry as _with_retry


log = get_logger(__name__)

_FETCH_TIMEOUT = 10  # seconds per symbol

# --- Macro Strategy Threshold Constants ---
VALUATION_RICH_ERP_THRESHOLD: float = 0.015  # 1.5% ERP threshold for equity richness
CREDIT_SPREAD_DANGER_THRESHOLD: float = 5.00  # 5.0% HY Spread danger threshold
CREDIT_SPREAD_WIDENING_3M_BPS: float = 100.0  # 100 bps widening over 3 months
STOCK_BOND_CORRELATION_WARNING_THRESHOLD: float = 0.30  # Correlation > 0.30 triggers warning
MIN_CORRELATION_OBSERVATIONS: int = 45  # Must have >= 45 overlapping trading days

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

_GLOBAL_GROUPS: list[tuple[str, list[str]]] = [
    ("🏦 Monetary Policy & Liquidity", ["DX-Y.NYB", "EURUSD=X", "USDJPY=X", "USDCNY=X", "^IRX", "^FVX", "^TNX", "^TYX"]),
    ("📈 Economic Growth", ["^GSPC", "^NDX", "^RUT", "HG=F", "CL=F"]),
    ("💰 Inflation", ["GC=F"]),
    ("⚠️ Geopolitics & Risk Sentiment", ["^VIX", "HYG", "LQD", "BTC-USD"]),
]

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

_GLOBAL_GROUPS: list[tuple[str, list[str]]] = [
    ("🏦 Monetary Policy & Liquidity", ["DX-Y.NYB", "EURUSD=X", "USDJPY=X", "USDCNY=X", "^IRX", "^FVX", "^TNX", "^TYX"]),
    ("📈 Economic Growth", ["^GSPC", "^NDX", "^RUT", "HG=F", "CL=F"]),
    ("💰 Inflation", ["GC=F"]),
    ("⚠️ Geopolitics & Risk Sentiment", ["^VIX", "HYG", "LQD", "BTC-USD"]),
]

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
    "DFII10": (
        "10-Year Real Yield (TIPS)",
        "อัตราผลตอบแทนพันธบัตรที่แท้จริง 10 ปี (TIPS Yield) — ต้นทุนค่าเสียโอกาสที่สำคัญที่สุดของทองคำ",
    ),
    "DTWEXBGS": (
        "US Dollar Index (Nominal Broad - FRED)",
        "ดัชนีค่าเงินดอลลาร์สหรัฐฯ แบบกว้าง (Nominal Broad) — ตัวชี้วัดโมเมนตัมค่าเงินและแรงกดดันต่อสินทรัพย์ EM",
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
    "CCSA": (
        "Continued Jobless Claims (K/week)",
        "ผู้รับสวัสดิการว่างงานต่อเนื่อง (พันคน) — ชี้วัดความยากในการหางานใหม่ของตลาดแรงงาน",
    ),
    "NFCI": (
        "Chicago Fed National Financial Conditions Index",
        "ดัชนีสภาวะทางการเงินโลก (Chicago Fed) — ค่าติดลบ = สภาพคล่องผ่อนคลาย เป็น Leading Indicator",
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
    # --- Euro Area ---
    "CLVMNACSCAB1GQEA19": ("Euro Area Real GDP (YoY %)", "ผลิตภัณฑ์มวลรวมแบบหักเงินเฟ้อ (Euro Area)"),
    "CP0000EZ19M086NEST": ("Euro Area CPI (YoY %)", "ดัชนีราคาผู้บริโภค (Euro Area)"),
    "ECBDFR": ("Euro Area Policy Rate", "อัตราดอกเบี้ยนโยบาย ECB"),
    # --- China ---
    "NGDPRXDCCNA": ("China Real GDP (YoY %)", "ผลิตภัณฑ์มวลรวมแบบหักเงินเฟ้อ (China)"),
    "CHNCPIALLMINMEI": ("China CPI (YoY %)", "ดัชนีราคาผู้บริโภค (China)"),
    "INTDSRCNM193N": ("China Policy Rate", "อัตราดอกเบี้ยนโยบาย PBOC"),
    # --- Japan ---
    "JPNRGDPEXP": ("Japan Real GDP (YoY %)", "ผลิตภัณฑ์มวลรวมแบบหักเงินเฟ้อ (Japan)"),
    "JPNCPIALLMINMEI": ("Japan CPI (YoY %)", "ดัชนีราคาผู้บริโภค (Japan)"),
    "INTDSRJPM193N": ("Japan Policy Rate", "อัตราดอกเบี้ยนโยบาย BOJ"),
    # --- India ---
    "NGDPRNSAXDCINQ": ("India Real GDP (YoY %)", "ผลิตภัณฑ์มวลรวมแบบหักเงินเฟ้อ (India)"),
    "INDCPIALLMINMEI": ("India CPI (YoY %)", "ดัชนีราคาผู้บริโภค (India)"),
    "INTDSRINM193N": ("India Policy Rate", "อัตราดอกเบี้ยนโยบาย RBI"),
    # --- Brazil (Latin America Proxy) ---
    "NGDPRSAXDCBRQ": ("Brazil Real GDP (YoY %)", "ผลิตภัณฑ์มวลรวมแบบหักเงินเฟ้อ (Brazil)"),
    "BRACPIALLMINMEI": ("Brazil CPI (YoY %)", "ดัชนีราคาผู้บริโภค (Brazil)"),
    "INTDSRBRM193N": ("Brazil Policy Rate", "อัตราดอกเบี้ยนโยบาย Brazil"),
}

_FRED_YOY_SERIES = {
    "CPIAUCSL", "GDPC1", "PCEPI", "PCEPILFE", "PPIACO", "INDPRO", "RSAFS",
    "CLVMNACSCAB1GQEA19", "CP0000EZ19M086NEST", "NGDPRXDCCNA", "CHNCPIALLMINMEI",
    "JPNRGDPEXP", "JPNCPIALLMINMEI", "NGDPRNSAXDCINQ", "INDCPIALLMINMEI",
    "NGDPRSAXDCBRQ", "BRACPIALLMINMEI"
}

_FRED_UNIT_DISPLAY: dict[str, str] = {
    "FEDFUNDS": "%",
    "DGS2": "%",
    "T10Y2Y": "% pts",
    "DFII10": "%",
    "DTWEXBGS": "Index",
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
    "CCSA": "K",
    "NFCI": "Index",
    "GDPC1": "% YoY",
    "INDPRO": "% YoY",
    "RSAFS": "% YoY",
    "HOUST": "K units",
    "M2SL": "B USD",
    "UMCSENT": "",
    "CLVMNACSCAB1GQEA19": "% YoY", "CP0000EZ19M086NEST": "% YoY", "ECBDFR": "%",
    "NGDPRXDCCNA": "% YoY", "CHNCPIALLMINMEI": "% YoY", "INTDSRCNM193N": "%",
    "JPNRGDPEXP": "% YoY", "JPNCPIALLMINMEI": "% YoY", "INTDSRJPM193N": "%",
    "NGDPRNSAXDCINQ": "% YoY", "INDCPIALLMINMEI": "% YoY", "INTDSRINM193N": "%",
    "NGDPRSAXDCBRQ": "% YoY", "BRACPIALLMINMEI": "% YoY", "INTDSRBRM193N": "%",
}

_US_GROUPS: list[tuple[str, list[str]]] = [
    ("🏦 Monetary Policy & Liquidity", ["FEDFUNDS", "DGS2", "T10Y2Y", "DFII10", "DTWEXBGS", "M2SL", "BAA10Y", "BAMLH0A0HYM2", "NFCI"]),
    ("📈 Economic Growth", ["ICSA", "CCSA", "GDPC1", "INDPRO", "RSAFS", "HOUST", "UNRATE"]),
    ("💰 Inflation", ["CPIAUCSL", "PCEPI", "PCEPILFE", "PPIACO", "T5YIE", "T10YIE"]),
    ("🛡️ Geopolitics & Risk Sentiment", ["UMCSENT"]),
]


_THAI_INDICATORS = {
    "THB=X": ("USD/THB", "USD to THB"),
    "^SET.BK": ("SET Index", "SET Index")
}

_THAI_GROUPS: list[tuple[str, list[str]]] = [
    ("🏦 Monetary Policy & Liquidity", ["THB=X"]),
    ("📈 Economic Growth", ["^SET.BK"]),
    ("💰 Inflation", []),
    ("🛡️ Geopolitics & Risk Sentiment", [])
]

_EURO_GROUPS: list[tuple[str, list[str]]] = [
    ("🏦 Monetary Policy & Liquidity", ["ECBDFR"]),
    ("📈 Economic Growth", ["CLVMNACSCAB1GQEA19"]),
    ("💰 Inflation", ["CP0000EZ19M086NEST"])
]

_CHINA_GROUPS: list[tuple[str, list[str]]] = [
    ("🏦 Monetary Policy & Liquidity", ["INTDSRCNM193N"]),
    ("📈 Economic Growth", ["NGDPRXDCCNA"]),
    ("💰 Inflation", ["CHNCPIALLMINMEI"])
]

_JAPAN_GROUPS: list[tuple[str, list[str]]] = [
    ("🏦 Monetary Policy & Liquidity", ["INTDSRJPM193N"]),
    ("📈 Economic Growth", ["JPNRGDPEXP"]),
    ("💰 Inflation", ["JPNCPIALLMINMEI"])
]

_INDIA_GROUPS: list[tuple[str, list[str]]] = [
    ("🏦 Monetary Policy & Liquidity", ["INTDSRINM193N"]),
    ("📈 Economic Growth", ["NGDPRNSAXDCINQ"]),
    ("💰 Inflation", ["INDCPIALLMINMEI"])
]

_LATAM_GROUPS: list[tuple[str, list[str]]] = [
    ("🏦 Monetary Policy & Liquidity", ["INTDSRBRM193N"]),
    ("📈 Economic Growth", ["NGDPRSAXDCBRQ"]),
    ("💰 Inflation", ["BRACPIALLMINMEI"])
]

_REGIONAL_GROUPS_MAP: dict[str, dict[str, list[str]]] = {
    "🇪🇺 Europe": {"📈 Economic Growth": ["VGK"]},
    "🇨🇳 China": {"📈 Economic Growth": ["MCHI"]},
    "🇯🇵 Japan": {"📈 Economic Growth": ["EWJ"]},
    "🇮🇳 India": {"📈 Economic Growth": ["INDA"]},
    "🌎 Latin America": {"📈 Economic Growth": ["ILF"]},
    "🌏 Asia Pacific ex-Japan": {"📈 Economic Growth": ["EPP"]},
    "🌐 Emerging Markets": {"📈 Economic Growth": ["EEM"]}
}
