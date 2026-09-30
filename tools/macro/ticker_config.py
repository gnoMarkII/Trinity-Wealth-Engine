from typing import NamedTuple, Literal, Optional
from core.logger import get_logger

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
    # --- Yield Curve ---
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
    # --- Credit Market ---
    "HYG": (
        "High Yield Bond ETF (HYG)",
        "ตราสารหนี้ High Yield — ตกก่อนตลาดหุ้นเสมอ ใช้เป็น Early Warning ของ Credit Stress",
    ),
    "LQD": (
        "Investment Grade Bond ETF (LQD)",
        "ตราสารหนี้ Investment Grade — สะท้อนต้นทุนกู้ยืมของบริษัทใหญ่ อ่อนไหวต่อ Rate ขึ้น",
    ),
    # --- FX ---
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
    # --- Commodities ---
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
    # --- US Equities ---
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
        "UnitedHealth/JNJ/Eli Lilly — Defensive รายได้สม่ำเสมอ ไม่ผูกกับ Economic Cycle",
    ),
    "XLI": (
        "Industrials (อุตสาหกรรม)",
        "Caterpillar/GE/Boeing — ไวต่อ Global Capex และ PMI ภาคการผลิต",
    ),
    "XLB": (
        "Materials (วัสดุศาสตร์)",
        "Linde/Freeport — ชี้วัดอุปสงค์วัตถุดิบต้นน้ำ อ่อนไหวต่อเศรษฐกิจจีนและ Commodity Prices",
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

class FredSeriesSpec(NamedTuple):
    series_id: str
    name: str
    description: str
    raw_unit: str
    frequency: str
    transform: Literal["none", "pc1", "diff_bps", "yoy_12m", "yoy_4q"]
    display_unit: str
    direction: Literal["positive", "inverse", "neutral"]
    min_observations: int
    category: str

FRED_SERIES_SPECS: dict[str, FredSeriesSpec] = {
    # --- US Monetary Policy & Rates ---
    "FEDFUNDS": FredSeriesSpec("FEDFUNDS", "Fed Funds Rate", "อัตราดอกเบี้ยนโยบายสหรัฐฯ (%) — ต้นทุนการเงินโลก กำหนดโดย FOMC", "%", "monthly", "none", "%", "neutral", 1, "monetary"),
    "DGS2": FredSeriesSpec("DGS2", "2-Year Treasury Yield", "อัตราผลตอบแทนพันธบัตร 2 ปี — ไวต่อ Fed Policy มากสุด ใช้คู่กับ 10Y เพื่อดู Yield Curve", "%", "daily", "none", "%", "neutral", 1, "rates"),
    "T10Y2Y": FredSeriesSpec("T10Y2Y", "10Y-2Y Yield Spread", "ส่วนต่างผลตอบแทน 10Y ลบ 2Y — ค่าติดลบ = Inverted Yield Curve สัญญาณ Recession ล่วงหน้า", "% pts", "daily", "none", "% pts", "positive", 1, "rates"),
    "DFII10": FredSeriesSpec("DFII10", "10-Year Real Yield (TIPS)", "อัตราผลตอบแทนพันธบัตรที่แท้จริง 10 ปี (TIPS Yield) — ต้นทุนค่าเสียโอกาสที่สำคัญที่สุดของทองคำ", "%", "daily", "none", "%", "neutral", 1, "rates"),
    "DTWEXBGS": FredSeriesSpec("DTWEXBGS", "US Dollar Index (Nominal Broad)", "ดัชนีค่าเงินดอลลาร์สหรัฐฯ แบบกว้าง (Nominal Broad) — ตัวชี้วัดโมเมนตัมค่าเงินและแรงกดดันต่อสินทรัพย์ EM", "Index", "daily", "none", "Index", "neutral", 1, "fx"),

    # --- US Inflation & Expectations ---
    "CPIAUCSL": FredSeriesSpec("CPIAUCSL", "CPI (YoY %)", "ดัชนีราคาผู้บริโภค YoY — ตัวชี้วัดเงินเฟ้อที่สาธารณชนรับรู้ ใช้กำหนด COLA", "Index 1982-84=100", "monthly", "pc1", "% YoY", "inverse", 13, "inflation"),
    "PCEPI": FredSeriesSpec("PCEPI", "PCE Inflation (YoY %)", "Personal Consumption Expenditures YoY — Headline PCE ติดตามควบคู่กับ Core PCE", "Index 2017=100", "monthly", "pc1", "% YoY", "inverse", 13, "inflation"),
    "PCEPILFE": FredSeriesSpec("PCEPILFE", "Core PCE Inflation (YoY %)", "PCE หัก Food & Energy YoY — ตัวชี้วัดเงินเฟ้อที่ Fed ใช้เป็น Primary Target จริงๆ (Target 2%)", "Index 2017=100", "monthly", "pc1", "% YoY", "inverse", 13, "inflation"),
    "PPIACO": FredSeriesSpec("PPIACO", "PPI (YoY %)", "ดัชนีราคาผู้ผลิต YoY — แรงกดดันเงินเฟ้อต้นน้ำ บอกก่อน CPI ประมาณ 1-3 เดือน", "Index 1982=100", "monthly", "pc1", "% YoY", "inverse", 13, "inflation"),
    "T5YIE": FredSeriesSpec("T5YIE", "5Y Breakeven Inflation Rate", "คาดการณ์เงินเฟ้อ 5 ปีของตลาด (TIPS spread) — forward-looking กว่า CPI สะท้อนความเชื่อมั่นต่อ Fed", "%", "daily", "none", "%", "neutral", 1, "inflation"),
    "T10YIE": FredSeriesSpec("T10YIE", "10Y Breakeven Inflation Rate", "คาดการณ์เงินเฟ้อ 10 ปีของตลาด — ถ้าสูงกว่า CPI = ตลาดคาดว่าเงินเฟ้อยังคงอยู่ยาวนาน", "%", "daily", "none", "%", "neutral", 1, "inflation"),

    # --- US Credit Market ---
    "BAA10Y": FredSeriesSpec("BAA10Y", "BAA Corporate Bond Spread", "ส่วนต่างพันธบัตรองค์กร Moody BAA เหนือ 10Y Treasury — ค่าสูง = ตลาดกลัว Credit Risk", "% pts", "daily", "none", "% pts", "inverse", 1, "credit"),
    "BAMLH0A0HYM2": FredSeriesSpec("BAMLH0A0HYM2", "High Yield Bond Spread", "ส่วนต่างผลตอบแทนหุ้นกู้ขยะ (ICE BofA) — ดัชนีชี้วัดความตื่นตระหนกในตลาดสินเชื่อ (Credit Risk)", "% pts", "daily", "none", "% pts", "inverse", 1, "credit"),

    # --- US Labor Market ---
    "UNRATE": FredSeriesSpec("UNRATE", "Unemployment Rate", "อัตราการว่างงานสหรัฐฯ (%) — ชี้วัดตลาดแรงงาน ส่วนหนึ่งของ Fed Dual Mandate", "%", "monthly", "none", "%", "inverse", 2, "growth"),
    "ICSA": FredSeriesSpec("ICSA", "Initial Jobless Claims (K/week)", "ยื่นขอสวัสดิการว่างงานครั้งแรกต่อสัปดาห์ (พันคน) — Leading Indicator ตลาดแรงงาน", "K", "weekly", "none", "K", "inverse", 2, "growth"),
    "CCSA": FredSeriesSpec("CCSA", "Continued Jobless Claims (K/week)", "ผู้รับสวัสดิการว่างงานต่อเนื่อง (พันคน) — ชี้วัดความยากในการหางานใหม่ของตลาดแรงงาน", "K", "weekly", "none", "K", "inverse", 2, "growth"),
    "NFCI": FredSeriesSpec("NFCI", "Chicago Fed National Financial Conditions Index", "ดัชนีสภาวะทางการเงินโลก (Chicago Fed) — ค่าติดลบ = สภาพคล่องผ่อนคลาย เป็น Leading Indicator", "Index", "weekly", "none", "Index", "inverse", 2, "credit"),

    # --- US Growth & Consumption ---
    "GDPC1": FredSeriesSpec("GDPC1", "Real GDP (YoY %)", "ผลิตภัณฑ์มวลรวมแบบหักเงินเฟ้อ YoY — ชี้วัดการเติบโตจริงของเศรษฐกิจ", "Bil. Chained 2017 $", "quarterly", "pc1", "% YoY", "positive", 5, "growth"),
    "INDPRO": FredSeriesSpec("INDPRO", "Industrial Production (YoY %)", "ดัชนีการผลิตอุตสาหกรรม YoY — proxy ที่ดีที่สุดสำหรับ PMI ในข้อมูลฟรี ชี้ภาคการผลิต", "Index 2017=100", "monthly", "pc1", "% YoY", "positive", 13, "growth"),
    "RSAFS": FredSeriesSpec("RSAFS", "Retail Sales (YoY %)", "ยอดขายปลีก YoY — สะท้อนการบริโภคภาคเอกชน ซึ่งเป็น ~70% ของ GDP สหรัฐฯ", "Mil. $", "monthly", "pc1", "% YoY", "positive", 13, "growth"),
    "HOUST": FredSeriesSpec("HOUST", "Housing Starts (K units/yr)", "จำนวนบ้านที่เริ่มก่อสร้าง (พันหลัง/ปี SAAR) — Leading Indicator Real Estate Cycle และ Recession", "K units", "monthly", "none", "K units", "positive", 2, "growth"),

    # --- US Liquidity & Sentiment ---
    "M2SL": FredSeriesSpec("M2SL", "M2 Money Supply (B USD)", "ปริมาณเงินในระบบ M2 (พันล้านดอลลาร์) — สะท้อน Monetary Condition และ Liquidity Cycle", "B USD", "monthly", "none", "B USD", "neutral", 2, "liquidity"),
    "UMCSENT": FredSeriesSpec("UMCSENT", "Consumer Sentiment (Index)", "ดัชนีความเชื่อมั่นผู้บริโภค U of Michigan — Leading Indicator การบริโภคและ Recession Risk", "Index", "monthly", "none", "Index", "positive", 2, "sentiment"),

    # --- Euro Area ---
    "CLVMNACSCAB1GQEA19": FredSeriesSpec("CLVMNACSCAB1GQEA19", "Euro Area Real GDP (YoY %)", "ผลิตภัณฑ์มวลรวมแบบหักเงินเฟ้อ (Euro Area)", "Mil. Chained 2010 EUR", "quarterly", "pc1", "% YoY", "positive", 5, "growth"),
    "CP0000EZ19M086NEST": FredSeriesSpec("CP0000EZ19M086NEST", "Euro Area CPI (YoY %)", "ดัชนีราคาผู้บริโภค (Euro Area)", "Index 2015=100", "monthly", "pc1", "% YoY", "inverse", 13, "inflation"),
    "ECBDFR": FredSeriesSpec("ECBDFR", "Euro Area Policy Rate", "อัตราดอกเบี้ยนโยบาย ECB", "%", "daily", "none", "%", "neutral", 1, "monetary"),

    # --- China ---
    "NGDPRXDCCNA": FredSeriesSpec("NGDPRXDCCNA", "China Real GDP (YoY %)", "ผลิตภัณฑ์มวลรวมแบบหักเงินเฟ้อ (China)", "National Currency", "annual", "pc1", "% YoY", "positive", 2, "growth"),
    "CHNCPIALLMINMEI": FredSeriesSpec("CHNCPIALLMINMEI", "China CPI (YoY %)", "ดัชนีราคาผู้บริโภค (China)", "Index 2015=100", "monthly", "pc1", "% YoY", "inverse", 13, "inflation"),
    "INTDSRCNM193N": FredSeriesSpec("INTDSRCNM193N", "China Policy Rate", "อัตราดอกเบี้ยนโยบาย PBOC", "%", "monthly", "none", "%", "neutral", 1, "monetary"),

    # --- Japan ---
    "JPNRGDPEXP": FredSeriesSpec("JPNRGDPEXP", "Japan Real GDP (YoY %)", "ผลิตภัณฑ์มวลรวมแบบหักเงินเฟ้อ (Japan)", "Bil. Chained 2015 JPY", "quarterly", "pc1", "% YoY", "positive", 5, "growth"),
    "JPNCPIALLMINMEI": FredSeriesSpec("JPNCPIALLMINMEI", "Japan CPI (YoY %)", "ดัชนีราคาผู้บริโภค (Japan)", "Index 2015=100", "monthly", "pc1", "% YoY", "inverse", 13, "inflation"),
    "INTDSRJPM193N": FredSeriesSpec("INTDSRJPM193N", "Japan Policy Rate", "อัตราดอกเบี้ยนโยบาย BOJ", "%", "monthly", "none", "%", "neutral", 1, "monetary"),

    # --- India ---
    "NGDPRNSAXDCINQ": FredSeriesSpec("NGDPRNSAXDCINQ", "India Real GDP (YoY %)", "ผลิตภัณฑ์มวลรวมแบบหักเงินเฟ้อ (India)", "Bil. Domestic Currency", "quarterly", "pc1", "% YoY", "positive", 5, "growth"),
    "INDCPIALLMINMEI": FredSeriesSpec("INDCPIALLMINMEI", "India CPI (YoY %)", "ดัชนีราคาผู้บริโภค (India)", "Index 2015=100", "monthly", "pc1", "% YoY", "inverse", 13, "inflation"),
    "INTDSRINM193N": FredSeriesSpec("INTDSRINM193N", "India Policy Rate", "อัตราดอกเบี้ยนโยบาย RBI", "%", "monthly", "none", "%", "neutral", 1, "monetary"),

    # --- Brazil (Latin America Proxy) ---
    "NGDPRSAXDCBRQ": FredSeriesSpec("NGDPRSAXDCBRQ", "Brazil Real GDP (YoY %)", "ผลิตภัณฑ์มวลรวมแบบหักเงินเฟ้อ (Brazil)", "Mil. Chained 1995 BRL", "quarterly", "pc1", "% YoY", "positive", 5, "growth"),
    "BRACPIALLMINMEI": FredSeriesSpec("BRACPIALLMINMEI", "Brazil CPI (YoY %)", "ดัชนีราคาผู้บริโภค (Brazil)", "Index 2015=100", "monthly", "pc1", "% YoY", "inverse", 13, "inflation"),
    "INTDSRBRM193N": FredSeriesSpec("INTDSRBRM193N", "Brazil Policy Rate", "อัตราดอกเบี้ยนโยบาย Brazil", "%", "monthly", "none", "%", "neutral", 1, "monetary"),
}

_FRED_SERIES: dict[str, tuple[str, str]] = {
    spec.series_id: (spec.name, spec.description)
    for spec in FRED_SERIES_SPECS.values()
}

_FRED_YOY_SERIES: set[str] = {
    spec.series_id
    for spec in FRED_SERIES_SPECS.values()
    if spec.transform in ("pc1", "yoy_12m", "yoy_4q")
}

_FRED_UNIT_DISPLAY: dict[str, str] = {
    spec.series_id: spec.display_unit
    for spec in FRED_SERIES_SPECS.values()
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
