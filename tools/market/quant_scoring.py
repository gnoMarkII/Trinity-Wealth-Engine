"""Deterministic Value/Quality/Momentum scoring — normalize ตัวเลขดิบเป็นคะแนน 0-100 ด้วยสูตรตายตัว

ห้าม LLM คำนวณคะแนนพวกนี้เองเด็ดขาด (กฎ 4.2/4.6/Final Doctrine ข้อ 3) — ทุกฟังก์ชันในไฟล์นี้
รับตัวเลขดิบจาก yfinance/financial_autopsy แล้ว Return คะแนนที่ deterministic 100%
"""
from typing import Optional, Tuple

from langsmith import traceable


def _linear_score(value: Optional[float], best: float, worst: float) -> Optional[float]:
    """แปลง value เป็นคะแนน 0-100 เชิงเส้น — value เท่ากับ best = 100, worst = 0, ค่ากลาง interpolate

    ถ้า best < worst แปลว่า 'ค่ายิ่งต่ำยิ่งดี' (เช่น P/E, Volatility)
    ถ้า best > worst แปลว่า 'ค่ายิ่งสูงยิ่งดี' (เช่น ROE, Profit Margin)
    """
    if value is None:
        return None
    if best <= worst:
        if value <= best:
            return 100.0
        if value >= worst:
            return 0.0
        return 100.0 * (worst - value) / (worst - best)
    else:
        if value >= best:
            return 100.0
        if value <= worst:
            return 0.0
        return 100.0 * (value - worst) / (best - worst)


def _avg_sub_scores(sub_scores: list[float]) -> Optional[float]:
    if not sub_scores:
        return None
    return round(sum(sub_scores) / len(sub_scores), 1)


@traceable(run_type="parser")
def compute_value_score(
    pe: Optional[float],
    pb: Optional[float] = None,
    ev_ebitda: Optional[float] = None,
) -> Tuple[Optional[float], Optional[str]]:
    """Normalize P/E (หลัก) + P/B + EV/EBITDA (เสริม) เป็นคะแนน 0-100 — P/E ต่ำ = ถูก = คะแนนสูง

    Threshold: P/E<=10 → 100, P/E>=40 → 0 (เชิงเส้น) | P/B<=1 → 100, P/B>=5 → 0 | EV/EBITDA<=8 → 100, >=20 → 0

    Returns:
        (score, flag) — flag ไม่ใช่ None เมื่อข้อมูลไม่พอ/ผิดปกติ เช่น
        'negative_earnings:pe_undefined' ถ้า P/E<=0 (บริษัทขาดทุน — ห้ามตีความว่า 'ถูก')
    """
    if pe is not None and pe <= 0:
        return None, "negative_earnings:pe_undefined"

    sub_scores = []
    if pe is not None:
        s = _linear_score(pe, best=10, worst=40)
        if s is not None:
            sub_scores.append(s)
    if pb is not None and pb > 0:
        s = _linear_score(pb, best=1, worst=5)
        if s is not None:
            sub_scores.append(s)
    if ev_ebitda is not None and ev_ebitda > 0:
        s = _linear_score(ev_ebitda, best=8, worst=20)
        if s is not None:
            sub_scores.append(s)

    score = _avg_sub_scores(sub_scores)
    if score is None:
        return None, "missing_valuation_metrics"
    return score, None


@traceable(run_type="parser")
def compute_quality_score(
    roe_pct: Optional[float],
    profit_margin_pct: Optional[float] = None,
    fcf_debt_ratio: Optional[float] = None,
) -> Tuple[Optional[float], Optional[str]]:
    """Normalize ROE + Profit Margin + FCF/Debt เป็นคะแนน 0-100 — ยิ่งสูงยิ่งมีคุณภาพดี

    Threshold: ROE>=20% → 100, <=5% → 0 | Profit Margin>=20% → 100, <=0% → 0 | FCF/Debt>=0.5x → 100, <=0 → 0
    """
    sub_scores = []
    if roe_pct is not None:
        s = _linear_score(roe_pct, best=20, worst=5)
        if s is not None:
            sub_scores.append(s)
    if profit_margin_pct is not None:
        s = _linear_score(profit_margin_pct, best=20, worst=0)
        if s is not None:
            sub_scores.append(s)
    if fcf_debt_ratio is not None:
        s = _linear_score(fcf_debt_ratio, best=0.5, worst=0.0)
        if s is not None:
            sub_scores.append(s)

    score = _avg_sub_scores(sub_scores)
    if score is None:
        return None, "missing_quality_metrics"
    return score, None


@traceable(run_type="parser")
def compute_momentum_score(
    rsi_14: Optional[float],
    macd_signal: Optional[str] = None,
    ma50_vs_ma200: Optional[str] = None,
) -> Tuple[Optional[float], Optional[str]]:
    """Normalize RSI(14) + MACD signal + MA50/MA200 cross เป็นคะแนน 0-100

    หมายเหตุสำคัญ (ต้องส่งต่อให้ narrative LLM ทราบเสมอ): คะแนนนี้วัด 'ความแรงของโมเมนตัม
    ขาขึ้นเชิงเทคนิค' (momentum-chasing view) ไม่ใช่คำแนะนำซื้อ/ขาย — RSI>70 ตาม technical
    analysis convention ทั่วไปหมายถึง Overbought (ราคาขึ้นเร็ว เสี่ยงปรับฐาน) ไม่ใช่ 'สัญญาณซื้อ'
    momentum_score สูงจึงอาจหมายถึงทั้ง 'ขาขึ้นแข็งแกร่ง' และ 'เสี่ยง overbought' พร้อมกัน

    Threshold: RSI>=70 → 100, RSI<=30 → 0 (เชิงเส้น) | MACD bullish → 100, bearish → 0
    | MA50/MA200: golden_cross → 100, death_cross → 0, อื่นๆ → 50
    """
    # Guard: หากไม่มีทั้ง RSI และ MACD (มีเพียง MA Cross โดดๆ จาก summary info ซึ่งไม่พอสะท้อน technical momentum)
    # ให้ถือว่าข้อมูลไม่เพียงพอ เพื่อไม่ให้เกิดคะแนน false 100.0 หรือ false 0.0
    if rsi_14 is None and macd_signal is None:
        return None, "insufficient_technical_data:momentum"

    sub_scores = []
    if rsi_14 is not None:
        s = _linear_score(rsi_14, best=70, worst=30)
        if s is not None:
            sub_scores.append(s)
    if macd_signal is not None:
        sub_scores.append(100.0 if macd_signal == "bullish" else 0.0)
    if ma50_vs_ma200 is not None:
        if ma50_vs_ma200 == "golden_cross":
            sub_scores.append(100.0)
        elif ma50_vs_ma200 == "death_cross":
            sub_scores.append(0.0)
        else:
            sub_scores.append(50.0)

    score = _avg_sub_scores(sub_scores)
    if score is None:
        return None, "missing_momentum_inputs"
    return score, None


@traceable(run_type="parser")
def compute_price_target_outlook(
    current_price: Optional[float],
    target_mean: Optional[float] = None,
    target_high: Optional[float] = None,
    target_low: Optional[float] = None,
) -> Tuple[Optional[float], Optional[float]]:
    """คำนวณ upside_pct (จาก target_high, fallback target_mean) และ downside_pct (จาก target_low)

    คืน (None, None) ถ้าไม่มี current_price หรือไม่มี target price เลย (พบบ่อยกับหุ้นไทยใน yfinance)
    """
    if current_price is None or current_price <= 0:
        return None, None

    upside_pct = None
    if target_high is not None:
        upside_pct = round((target_high - current_price) / current_price * 100, 2)
    elif target_mean is not None:
        upside_pct = round((target_mean - current_price) / current_price * 100, 2)

    downside_pct = None
    if target_low is not None:
        downside_pct = round((target_low - current_price) / current_price * 100, 2)

    return upside_pct, downside_pct


@traceable(run_type="parser")
def compute_growth_score(
    revenue_growth_yoy_pct: Optional[float],
    net_income_growth_yoy_pct: Optional[float] = None,
) -> Tuple[Optional[float], Optional[str]]:
    """Normalize Revenue Growth YoY (หลัก) + Net Income Growth YoY (เสริม, weight 2:1) เป็นคะแนน 0-100

    Threshold: Revenue Growth >=20% → 100, <=-10% → 0 (เชิงเส้น) — Net Income ใช้ threshold เดียวกัน
    ถ้ามีทั้งคู่ ให้น้ำหนัก Revenue:NetIncome = 2:1 (Revenue เป็นตัวชี้วัดหลักที่ Volatile น้อยกว่า)
    """
    if revenue_growth_yoy_pct is None:
        return None, "missing_growth_data:growth"

    rev_score = _linear_score(revenue_growth_yoy_pct, best=20, worst=-10)
    if net_income_growth_yoy_pct is not None:
        ni_score = _linear_score(net_income_growth_yoy_pct, best=20, worst=-10)
        score = round((rev_score * 2 + ni_score) / 3, 1)
    else:
        score = round(rev_score, 1)
    return score, None


@traceable(run_type="parser")
def compute_dividend_score(
    dividend_yield_pct: Optional[float],
    payout_ratio_pct: Optional[float] = None,
) -> Tuple[Optional[float], Optional[str]]:
    """Normalize Dividend Yield เป็นคะแนน 0-100 — Threshold: Yield>=5% → 100, 0% → 0 (เชิงเส้น)

    หมายเหตุสำคัญ (ต้องส่งต่อให้ narrative LLM ทราบเสมอ): Yield สูงผิดปกติอาจเป็น 'Value Trap' —
    ราคาหุ้นร่วงหนักทำให้ yield (=dividend/price) พุ่งสูงลวงตา ไม่ใช่สัญญาณบวกเสมอไป ต้องพิจารณา
    ร่วมกับ payout_ratio และ momentum/value score เสมอ ห้ามตีความ yield สูง = น่าลงทุน โดยลำพัง

    Payout Ratio >=100% (จ่ายเกินกำไร) หรือ <0% (ผิดปกติ) → แนบ flag เตือนแต่ยังคำนวณคะแนนจาก yield
    ตามปกติ (ไม่ null คะแนนทิ้ง) เพื่อให้ downstream เห็นทั้งตัวเลขและคำเตือนพร้อมกัน
    """
    if dividend_yield_pct is None:
        return None, "missing_dividend_data:dividend"

    score = round(_linear_score(dividend_yield_pct, best=5, worst=0), 1)
    if payout_ratio_pct is not None and (payout_ratio_pct >= 100 or payout_ratio_pct < 0):
        return score, "unsustainable_payout:dividend"
    return score, None


@traceable(run_type="parser")
def compute_solvency_score(
    de_ratio_pct: Optional[float],
    current_ratio: Optional[float] = None,
) -> Tuple[Optional[float], Optional[str]]:
    """Normalize D/E Ratio + Current Ratio เป็นคะแนน 0-100 — Risk Gate แยกต่างหาก ไม่รวมใน Composite Score

    de_ratio_pct: ต้องเป็นค่าที่มาจาก yfinance `debtToEquity` ตรงๆ (สเกล % แบบเดียวกับที่
    tools/market/fundamentals.py ใช้แสดงผล เช่น 150.0 = หนี้ 1.5 เท่าของทุน) — ห้ามหาร/คูณ 100 เพิ่ม

    Threshold: D/E<=50% → 100, >=300% → 0 (เชิงเส้น) | Current Ratio>=2.0x → 100, <=0.5x → 0

    คะแนนต่ำกว่า 30 จะแนบ flag 'high_leverage_risk:solvency' เพื่อเตือนความเสี่ยง leverage ชัดเจน
    ถึงแม้คะแนนนี้จะไม่ถูกรวมเข้า Composite Score (เป็น risk indicator แยก ไม่ใช่ return driver)
    """
    sub_scores = []
    if de_ratio_pct is not None:
        s = _linear_score(de_ratio_pct, best=50, worst=300)
        if s is not None:
            sub_scores.append(s)
    if current_ratio is not None:
        s = _linear_score(current_ratio, best=2.0, worst=0.5)
        if s is not None:
            sub_scores.append(s)

    score = _avg_sub_scores(sub_scores)
    if score is None:
        return None, "missing_solvency_data:solvency"
    if score < 30:
        return score, "high_leverage_risk:solvency"
    return score, None


@traceable(run_type="parser")
def compute_trading_liquidity(
    avg_volume: Optional[float],
    avg_volume_10d: Optional[float],
    current_price: Optional[float],
    market: str,
) -> Tuple[Optional[float], Optional[str]]:
    """คำนวณ ADTV (Average Daily Trading Value) สกุลเงินท้องถิ่น — ไม่ใช่คะแนน 0-100

    ใช้ avg_volume_10d (สดกว่า) ก่อน fallback ไป avg_volume (เฉลี่ย 3 เดือน) ถ้าไม่มี
    ผลลัพธ์เป็นสกุลท้องถิ่นตาม market (THB สำหรับ TH, USD สำหรับ US) — ไม่ได้แปลงข้ามสกุล
    ห้ามเปรียบเทียบ ADTV ข้าม market โดยตรงโดยไม่แปลง FX ก่อน

    Threshold low_liquidity: US < 1,000,000 USD | TH < 5,000,000 THB
    """
    volume = avg_volume_10d if avg_volume_10d is not None else avg_volume
    if volume is None or current_price is None or current_price <= 0:
        return None, "missing_liquidity_data:liquidity"

    adtv = round(volume * current_price, 2)
    threshold = 5_000_000 if market == "TH" else 1_000_000
    if adtv < threshold:
        return adtv, "low_liquidity:liquidity"
    return adtv, None


_COMPOSITE_WEIGHTS = {"value": 0.25, "quality": 0.25, "growth": 0.25, "momentum": 0.15, "dividend": 0.10}


@traceable(run_type="parser")
def compute_composite_score(
    value_score: Optional[float],
    quality_score: Optional[float],
    growth_score: Optional[float],
    momentum_score: Optional[float],
    dividend_score: Optional[float],
) -> Tuple[Optional[float], Optional[str]]:
    """Weighted Composite Score (0-100) — Value 25% / Quality 25% / Growth 25% / Momentum 15% / Dividend 10%

    Solvency ไม่รวมในสูตรนี้โดยตั้งใจ — ดู compute_solvency_score docstring (เป็น Risk Gate แยก
    ไม่ใช่ Return Driver การผสมรวมกับคะแนนบวกจะบดบังความเสี่ยง leverage ที่ควรเห็นชัดแยกต่างหาก)

    ข้ามมิติที่เป็น None แล้ว re-normalize weight ที่เหลือ (ไม่บังคับให้มีครบทุกมิติ)
    ถ้ามีข้อมูลไม่ถึง 2 มิติ → None พร้อม flag (ไม่น่าเชื่อถือพอจะสรุปเป็นคะแนนเดียว)
    """
    scores = {
        "value": value_score,
        "quality": quality_score,
        "growth": growth_score,
        "momentum": momentum_score,
        "dividend": dividend_score,
    }
    available = {k: v for k, v in scores.items() if v is not None}
    if len(available) < 2:
        return None, "insufficient_dimensions:composite"

    total_weight = sum(_COMPOSITE_WEIGHTS[k] for k in available)
    composite = sum(v * _COMPOSITE_WEIGHTS[k] for k, v in available.items()) / total_weight
    return round(composite, 1), None


@traceable(run_type="parser")
def compute_fcf_quality_score(
    fcf_yield_pct: Optional[float],
    ocf_to_net_income: Optional[float] = None,
) -> Tuple[Optional[float], Optional[str]]:
    """Normalize FCF Yield % (หลัก) + OCF/NetIncome (คุณภาพกำไร) เป็นคะแนน 0-100
    Threshold: FCF Yield >= 6% -> 100, <= 0% -> 0 | OCF/NI >= 1.2 -> 100, <= 0.5 -> 0
    """
    sub_scores = []
    if fcf_yield_pct is not None:
        s = _linear_score(fcf_yield_pct, best=6.0, worst=0.0)
        if s is not None:
            sub_scores.append(s)
    if ocf_to_net_income is not None:
        s = _linear_score(ocf_to_net_income, best=1.2, worst=0.5)
        if s is not None:
            sub_scores.append(s)

    score = _avg_sub_scores(sub_scores)
    if score is None:
        return None, "missing_fcf_quality_data:fcf_quality"
    return score, None


@traceable(run_type="parser")
def compute_debt_quality_score(
    interest_coverage: Optional[float],
    net_debt_ebitda: Optional[float] = None,
) -> Tuple[Optional[float], Optional[str]]:
    """Normalize Interest Coverage + Net Debt / EBITDA เป็น Risk Score 0-100
    Threshold: Interest Coverage >= 5.0x -> 100, <= 1.5x -> 0 | Net Debt/EBITDA <= 1.5x -> 100, >= 4.0x -> 0
    """
    sub_scores = []
    if interest_coverage is not None:
        s = _linear_score(interest_coverage, best=5.0, worst=1.5)
        if s is not None:
            sub_scores.append(s)
    if net_debt_ebitda is not None:
        s = _linear_score(net_debt_ebitda, best=1.5, worst=4.0)
        if s is not None:
            sub_scores.append(s)

    score = _avg_sub_scores(sub_scores)
    if score is None:
        return None, "missing_debt_quality_data:debt_quality"
    flag = "high_debt_risk:debt_quality" if score < 30 else None
    return score, flag
