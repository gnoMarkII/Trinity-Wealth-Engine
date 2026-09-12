"""มิเรอร์ tools/macro/report_formatter.py::format_macro_strategy_report สำหรับ equity_intel pipeline

Python เขียนตัวเลขทั้งหมดจาก quant_signals ตรงๆ — LLM (equity_synthesizer) ไม่มีโอกาสแตะตัวเลข
ในขั้นตอนนี้เลย มีแค่ narrative_analysis/base_case_summary ที่เป็น text จาก LLM
"""
from datetime import datetime

from schemas.micro_quant_schemas import MicroQuantOutput
from .core import _fmt_large

_SENTIMENT_LABELS = {"bullish": "🟢 Bullish", "bearish": "🔴 Bearish", "neutral": "⚪ Neutral"}


def _fmt(value) -> str:
    return "N/A" if value is None else str(value)


DATA_QUALITY_FLAG_TRANSLATIONS: dict[str, tuple[str, str]] = {
    # dcf_valuation.py
    "negative_fcf_dcf_unavailable:dcf": (
        "ไม่สามารถคำนวณ DCF ได้ (FCF ติดลบ)",
        "กระแสเงินสดเสรีเป็นลบ จึงไม่สามารถประเมินมูลค่าด้วย DCF ได้",
    ),
    "beta_unavailable_dcf_unavailable:dcf": (
        "ไม่สามารถคำนวณ DCF ได้ (ไม่พบค่า Beta)",
        "ไม่มีข้อมูลความผันผวนเทียบกับตลาด (Beta) ทำให้คำนวณ Cost of Equity ไม่ได้",
    ),
    "hardcoded_th_risk_free:dcf": (
        "ใช้อัตราดอกเบี้ย Risk-Free ไทยอ้างอิง (2.75%)",
        "ไม่พบข้อมูลพันธบัตรรัฐบาลไทย 10 ปีล่าสุด จึงใช้อัตราดอกเบี้ยสำรอง",
    ),
    "hardcoded_country_risk_premium:dcf": (
        "ใช้ Country Risk Premium อ้างอิง (1.75%)",
        "ใช้อัตราความเสี่ยงประเทศสำรองตามตาราง Damodaran",
    ),
    "hardcoded_us_risk_free:dcf": (
        "ใช้อัตราดอกเบี้ย Risk-Free อ้างอิง (4.25%)",
        "ไม่พบข้อมูลอัตราดอกเบี้ยพันธบัตรรัฐบาลสหรัฐฯ 10 ปีในระบบ จึงใช้อัตราดอกเบี้ยสำรอง",
    ),
    "rich_market_valuation_low_erp:dcf": (
        "Equity Risk Premium (ERP) ของตลาดค่อนข้างต่ำ (<1.5%)",
        "ผลตอบแทนชดเชยความเสี่ยงหุ้นเทียบกับพันธบัตรแคบลง สะท้อนสภาวะตลาด Valuation ตึงตัว",
    ),
    "kd_clamped:dcf": (
        "จำกัดช่วง Cost of Debt (2%-15%)",
        "ปรับอัตราดอกเบี้ยจ่ายให้อยู่ในกรอบมาตรฐานการคำนวณ",
    ),
    "hardcoded_cost_of_debt:dcf": (
        "ใช้ Cost of Debt อ้างอิง (5.0%)",
        "ไม่พบข้อมูลดอกเบี้ยจ่ายจริง จึงใช้ค่าประมาณการสำรอง",
    ),
    "wacc_below_terminal_growth:dcf": (
        "WACC ต่ำกว่าอัตราเติบโต Terminal Growth",
        "WACC มีค่าน้อยกว่าหรือเท่ากับอัตราการเติบโตระยะยาว ไม่สามารถใช้ Gordon Growth Model โดยตรง",
    ),
    "eps_proxy_base_growth:dcf": (
        "ใช้ YoY EPS Growth เป็นตัวแทน FCF Base Growth",
        "ประมาณการการเติบโต 5 ปีของ Free Cash Flow จากอัตราการเติบโต EPS",
    ),
    "generic_base_growth_assumption:dcf": (
        "ใช้ Base Growth อ้างอิง 5.0%",
        "ไม่พบข้อมูล EPS Growth จึงใช้สมมติฐานการเติบโตมาตรฐาน 5%",
    ),
    # ownership.py
    "10b51_unfiltered:insider_signal": (
        "ข้อมูล Insider ไม่ได้แยกแผน 10b5-1",
        "รายการซื้อขายของผู้บริหารรวมทั้งการซื้อขายตามแผนล่วงหน้าและแบบสมัครใจ",
    ),
    "insider_date_unavailable:insider_signal": (
        "ไม่มีข้อมูลวันที่ในรายการ Insider",
        "ไม่สามารถกรองรายการเฉพาะ 90 วันล่าสุดได้",
    ),
    # quant_scoring.py
    "negative_earnings:pe_undefined": (
        "ไม่สามารถคำนวณ P/E ได้ (กำไรติดลบ)",
        "บริษัทมีผลขาดทุนสุทธิ ทำให้ไม่สามารถประเมินมูลค่าผ่าน P/E Ratio ได้",
    ),
    "missing_growth_data:growth": (
        "ข้อมูลการเติบโตไม่เพียงพอ",
        "ขาดข้อมูลรายได้หรือกำไรย้อนหลังสำหรับคำนวณ Growth Score",
    ),
    "missing_dividend_data:dividend": (
        "ข้อมูลเงินปันผลไม่เพียงพอ",
        "ขาดข้อมูลการจ่ายเงินปันผลหรืออัตราตอบแทนปันผล",
    ),
    "unsustainable_payout:dividend": (
        "อัตราจ่ายปันผลสูงเกินความยั่งยืน (>100%)",
        "เงินปันผลที่จ่ายสูงกว่ากำไรสุทธิ มีความเสี่ยงที่จะลดการจ่ายปันผลในอนาคต",
    ),
    "missing_solvency_data:solvency": (
        "ข้อมูลความมั่นคงทางการเงินไม่เพียงพอ",
        "ขาดข้อมูลงบการเงินสำหรับคำนวณ Solvency Score",
    ),
    "high_leverage_risk:solvency": (
        "ความเสี่ยงภาระหนี้สินสูง",
        "สัดส่วนหนี้สินต่อทุน (D/E) หรือ Net Debt/EBITDA อยู่ในระดับสูงกว่ามาตรฐาน",
    ),
    "missing_liquidity_data:liquidity": (
        "ข้อมูลสภาพคล่องการซื้อขายไม่เพียงพอ",
        "ไม่พบข้อมูลมูลค่าการซื้อขายเฉลี่ยรายวัน (ADTV)",
    ),
    "low_liquidity:liquidity": (
        "สภาพคล่องการซื้อขายค่อนข้างต่ำ",
        "มูลค่าการซื้อขายเฉลี่ยรายวันต่ำกว่าเกณฑ์ อาจมีขีดจำกัดในการเข้าซื้อหรือขายออก",
    ),
    "insufficient_dimensions:composite": (
        "มิติการประเมินไม่ครบถ้วน",
        "มิติคำนวณ Score ไม่ครบ จึงมีการปรับ re-normalize น้ำหนักที่เหลือ",
    ),
    "missing_fcf_quality_data:fcf_quality": (
        "ข้อมูลคุณภาพกระแสเงินสดไม่เพียงพอ",
        "ขาดข้อมูลงบกระแสเงินสดสำหรับประเมิน FCF Quality",
    ),
    "missing_debt_quality_data:debt_quality": (
        "ข้อมูลคุณภาพหนี้สินไม่เพียงพอ",
        "ขาดข้อมูลดอกเบี้ยจ่ายหรือภาระหนี้สำหรับประเมิน Debt Quality",
    ),
    "high_debt_risk:debt_quality": (
        "ความเสี่ยงคุณภาพหนี้สินสูง",
        "ความสามารถในการชำระดอกเบี้ย (Interest Coverage) ต่ำกว่าเกณฑ์ความปลอดภัย",
    ),
    # peer_valuation.py
    "missing_own_pe:peer_relative": (
        "ไม่สามารถเปรียบเทียบ P/E กับกลุ่มได้ (ไม่มี P/E ตนเอง)",
        "หุ้นมี P/E ติดลบหรือไม่พบค่า P/E จึงไม่สามารถเปรียบเทียบกับกลุ่มอุตสาหกรรมได้",
    ),
    "insufficient_peers:peer_relative": (
        "จำนวนหุ้นเปรียบเทียบในกลุ่มไม่เพียงพอ",
        "มีหุ้นคู่แข่งในกลุ่มเดียวกันน้อยกว่าเกณฑ์ที่ใช้ประเมิน Peer Relative Score",
    ),
    # quant_engine.py
    "insufficient_periods:growth": (
        "รอบข้อมูลงบการเงินย้อนหลังไม่เพียงพอ",
        "ข้อมูลงบการเงินในอดีตมีน้อยกว่า 2 ช่วงเวลา ไม่สามารถคำนวณ YoY Growth ได้",
    ),
    "non_annual_period_gap:growth": (
        "ระยะเวลาเปรียบเทียบงบการเงินไม่ใช่งวดปีเต็ม",
        "งวดเวลาของข้อมูลเปรียบเทียบไม่อยู่ในรอบ 12 เดือนเต็ม",
    ),
}

_STALE_REASON_LABELS = {
    "insufficient_trading_history": "ประวัติราคาซื้อขายไม่เพียงพอ",
    "latest_close_unavailable": "ราคาปิดล่าสุดไม่สมบูรณ์หรือไม่พร้อมใช้งาน (NaN/Non-finite)",
    "fetch_error": "ดึงข้อมูลย้อนหลังไม่สำเร็จ",
    "zero_benchmark_variance": "ความผันผวนของดัชนีอ้างอิงเป็นศูนย์",
}

_METRIC_LABELS_TH = {
    "beta": "Beta",
    "volatility": "Volatility",
    "mdd": "Max Drawdown",
    "technical_indicators": "ตัวชี้วัดทางเทคนิค (Momentum)",
    "price_percentile": "Price Percentile",
}


def _translate_flag(raw: str) -> str:
    """แปลง raw flag code ให้เป็นคำอธิบายภาษาไทยพร้อมต่อท้ายด้วย (`raw_code`)"""
    if raw in DATA_QUALITY_FLAG_TRANSLATIONS:
        label, subtext = DATA_QUALITY_FLAG_TRANSLATIONS[raw]
        return f"- **{label}**: {subtext} (`{raw}`)"

    if ":" in raw:
        code, domain = raw.split(":", 1)
        reason_th = _STALE_REASON_LABELS.get(code)
        metric_th = _METRIC_LABELS_TH.get(domain)
        if reason_th and metric_th:
            return f"- **ไม่สามารถคำนวณ {metric_th} ได้**: {reason_th} (`{raw}`)"

        clean_code = code.replace("_", " ").title()
        return f"- **{clean_code}** ({domain.upper()}) (`{raw}`)"

    return f"- **{raw.replace('_', ' ')}** (`{raw}`)"


def format_equity_analysis_report(output: MicroQuantOutput) -> str:
    """Build markdown report จาก MicroQuantOutput — ตัวเลขทั้งหมดมาจาก quant_signals โดยตรง (Python)"""
    today = output.analysis_date
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    q = output.quant_signals
    s = output.sentiment_context
    display_name = f"{q.company_name} ({output.ticker})" if q.company_name else output.ticker

    benchmark_label = "^SET.BK" if output.market == "TH" else "^GSPC"
    currency_label = "THB" if output.market == "TH" else "USD"

    lines = [
        "---",
        "schema_version: 2",
        f"title: {output.ticker} Equity Analysis {today}",
        "entity_type: equity_analysis",
        f"ticker: {output.ticker}",
        f"market: {output.market}",
        f"date: {today}",
        f"last_updated: {now}",
        f"generated_by: {output.generated_by}",
        f"tags: [stock_analysis, {output.ticker.lower()}, market_{output.market.lower()}, equity_quant]",
        "---\n",
        f"# 📊 บทวิเคราะห์เชิงปริมาณ: {display_name} ({output.market}) — {today}\n",
    ]

    if q.atomic_market_snapshot:
        snap = q.atomic_market_snapshot
        lines.extend([
            f"> **Analysis Price:** **${_fmt(snap.analysis_price)}** (as of {snap.analysis_price_as_of}, Source: `{snap.price_source}` | Sync: `{snap.price_sync_status}`)",
            f"> **Market Cap:** ${_fmt_large(snap.market_cap, '$')} | **Shares Outstanding:** {_fmt_large(snap.shares_outstanding, '')}",
        ])
        if snap.price_sync_status == "quote_ohlcv_mismatch":
            lines.append(f"> ⚠️ **Price Synchronization Warning:** Live quote mismatch detected. Analysis strictly bound to EOD unadjusted Close (${snap.latest_ohlcv_close:.2f} as of {snap.latest_ohlcv_date}).")
    lines.extend([
        f"> **Market Sentiment:** {_SENTIMENT_LABELS.get(s.market_sentiment, s.market_sentiment)}",
        f"> **ประเมินเมื่อ (Evaluated At):** {q.evaluated_at}\n",
    ])

    # -------------------------------------------------------------
    # 🏆 Institutional 4-Pillar Scorecard & Action Stance
    # -------------------------------------------------------------
    if q.deterministic_scorecard:
        sc = q.deterministic_scorecard
        usable_cov = getattr(sc, "usable_coverage_pct", sc.coverage_pct)
        verified_cov = getattr(sc, "verified_coverage_pct", sc.coverage_pct)
        lines.extend([
            f"## 🏛️ Institutional Investment Committee Scorecard\n",
            f"- **Action Stance:** `{sc.action_stance}`",
        ])
        if sc.action_stance_reason:
            lines.append(f"- **Stance Reason:** {sc.action_stance_reason}")
        lines.extend([
            f"- **Core Conviction Score:** **{sc.core_conviction_score} / 10.0** (Fundamental + Guidance + Valuation)",
            f"- **Execution Readiness Score:** **{sc.execution_readiness_score} / 10.0** (Technicals + Liquidity + Form 4)",
            f"- **Coverage:** Usable {usable_cov}% | Verified {verified_cov}% ({sc.applicable_pillars_count} Applicable Pillars)",
            "",
            "| Pillar | Score | Weight | หมายเหตุ |",
            "|---|---|---|---|",
            f"| **Pillar 1: Fundamental Quality** | {sc.fundamental_quality_score} / 100 | 40% | งบการเงิน, Piotroski F-Score, OCF/NI |",
            f"| **Pillar 2: Guidance & Expectation Gap** | {sc.guidance_expectation_score} / 100 | 30% | สรุป Guidance & EPS Revisions |",
            f"| **Pillar 3: Valuation Margin of Safety** | {sc.valuation_margin_score} / 100 | 30% | 12M DCF Upside & Implied Growth Gap |",
            "",
        ])

    if q.dcf_discrepancy_warning:
        lines.extend([
            f"> ⚠️ **คำเตือน Valuation Discrepancy:** {q.dcf_discrepancy_warning}\n",
        ])

    lines.extend([
        f"## 🏆 Composite Score: {_fmt(q.composite_score)} / 100\n",
        "> Weighted: Value 25% + Quality 25% + Growth 25% + Momentum 15% + Dividend 10%\n",
        "## 🔢 Quant Signals (Deterministic — คำนวณจาก Python ล้วน)\n",
        "| Metric | Value | หมายเหตุ |",
        "|---|---|---|",
        f"| **Value Score** | {_fmt(q.value_score)} | 0-100, ยิ่งสูงยิ่งถูก (Valuation) |",
        f"| **Quality Score** | {_fmt(q.quality_score)} | 0-100, ยิ่งสูงยิ่งมีคุณภาพกิจการดี |",
        f"| **Growth Score** | {_fmt(q.growth_score)} | 0-100, จาก Revenue/Net Income Growth YoY |",
        f"| **Momentum Score** | {_fmt(q.momentum_score)} | 0-100, วัดความแรงขาขึ้นเชิงเทคนิค |",
        f"| **Dividend Score** | {_fmt(q.dividend_score)} | 0-100 |",
        f"| **Beta** | {_fmt(q.beta)} | เทียบ {benchmark_label} |",
        f"| **Volatility (Annualized)** | {_fmt(q.volatility_pct)}% | |",
        f"| **Max Drawdown** | {_fmt(q.mdd_pct)}% | |",
        f"| **Price Percentile (5Y)** | {_fmt(q.price_percentile_5y)}{'%' if q.price_percentile_5y is not None else ''} | เทียบการกระจายราคา 5 ปี |",
        f"| **Price Z-Score (5Y)** | {_fmt(q.price_zscore_5y)} | |",
        "",
        "### 🎯 Price Target Outlook (Consensus)\n",
        f"- Target Upside: {_fmt(q.upside_pct)}%",
        f"- Target Downside: {_fmt(q.downside_pct)}%",
        "",
    ])

    # -------------------------------------------------------------
    # 🩺 Piotroski F-Score Breakdown
    # -------------------------------------------------------------
    if q.piotroski_breakdown:
        p = q.piotroski_breakdown
        lines.extend([
            "### 🩺 Piotroski F-Score Forensics\n",
            f"- **F-Score:** **{_fmt(p.f_score)} / 9** (Status: `{p.status.upper()}`)",
            f"- Profitability Points: {p.profitability_points} / 4 (ROA > 0, CFO > 0, $\\Delta$ROA > 0, CFO > Net Income)",
            f"- Leverage & Liquidity Points: {p.leverage_liquidity_points} / 3 ($\\Delta$Leverage, $\\Delta$Current Ratio, Dilution)",
            f"- Operating Efficiency Points: {p.operating_efficiency_points} / 2 ($\\Delta$Gross Margin, $\\Delta$Asset Turnover)",
            f"- Exclusion Reason: {_fmt(p.exclusion_reason)}",
            "",
        ])

    # -------------------------------------------------------------
    # 🎯 5-Year Explicit DCF & Reverse DCF
    # -------------------------------------------------------------
    if q.reverse_dcf_result:
        rd = q.reverse_dcf_result
        ebit_label = "Audited EBIT Margin" if rd.ebit_margin_source_tier == "filing_authoritative" else "Reported EBIT Margin"
        ebit_meta = f"{rd.ebit_margin_fiscal_period or 'FY'}"
        if rd.ebit_margin_period_type:
            ebit_meta += f", {rd.ebit_margin_period_type}"
        if rd.ebit_margin_source_tier:
            ebit_meta += f", Source: {rd.ebit_margin_source_tier}"
        actionable_note = f" (⚠️ Informational Only: {rd.actionability_reason})" if not rd.is_actionable else ""
        lines.extend([
            "### 🎯 5-Year Explicit DCF & Reverse DCF Expectation Gap\n",
            f"- **12-Month Target Price:** **${_fmt(rd.target_price_12m)}** (Upside: {_fmt(rd.upside_12m_pct)}%){actionable_note}",
            f"- **Reverse DCF Verdict:** `{rd.valuation_verdict.upper()}`",
            f"- **Intrinsic Value Today:** ${_fmt(rd.intrinsic_value_today)}",
            f"- **{ebit_label}:** **{_fmt(rd.reported_ebit_margin_pct)}%** ({ebit_meta})",
            f"- **Market Implied Revenue Growth:** **{_fmt(rd.market_implied_growth_pct)}%** (Solver Status: `{rd.solver_status}`)",
            f"- **Market Implied Operating Margin:** {_fmt(rd.market_implied_margin_pct)}%",
            f"- Enterprise Value: ${_fmt_large(rd.enterprise_value, '$')} | Equity Value: ${_fmt_large(rd.equity_value, '$')}",
            f"- PV of 5Y FCF: ${_fmt_large(rd.sum_pv_5y_fcf, '$')} | PV of Terminal Value: ${_fmt_large(rd.terminal_value_pv, '$')}",
            "",
        ])
    elif q.dcf_result:
        d = q.dcf_result
        lines.extend([
            "### 🎯 DCF Target Price & Fair Value Engine (Legacy Proxy)\n",
            f"- **Real WACC:** {d.wacc_pct}%",
            f"- **Valuation Verdict:** `{d.valuation_verdict.upper()}`",
            f"- Base Case Target Price: ${d.scenarios['base'].target_price} (Upside: {d.scenarios['base'].upside_pct}%)",
            "",
        ])

    # -------------------------------------------------------------
    # 📊 Tactical Setup & S/R Plan
    # -------------------------------------------------------------
    if q.tactical_setup:
        ts = q.tactical_setup
        bz_rr_str = f"{_fmt(ts.buy_zone_rr_min)} - {_fmt(ts.buy_zone_rr_max)} : 1" if (ts.buy_zone_rr_min is not None and ts.buy_zone_rr_max is not None) else "N/A"
        lines.extend([
            "### 📊 Tactical Setup & Execution Timing (1-3M Horizon)\n",
            f"- **Price Stage:** `{ts.price_stage}`",
            f"- **ATR (14D):** ${_fmt(ts.atr_14)}",
            f"- **Key Support Level:** ${_fmt(ts.key_support_level)} | **Key Resistance Level:** ${_fmt(ts.key_resistance_level)}",
            f"- **Optimal Buy Zone:** **${_fmt(ts.buy_zone_min)} - ${_fmt(ts.buy_zone_max)}** (In Zone: {ts.is_in_buy_zone})",
            f"- **Invalidation Stop Loss:** **${_fmt(ts.invalidation_stop_loss)}**",
            f"- **Tactical Target (1-3M):** **${_fmt(ts.tactical_target_price)}**",
            f"- **Current R:R Ratio:** **{_fmt(ts.current_rr_ratio)} : 1** | **Buy Zone R:R Range:** **{bz_rr_str}**",
        ])
        if ts.breakout_trigger_price:
            curr_rr_str = f"{_fmt(ts.breakout_current_rr)} : 1" if ts.breakout_current_rr is not None else "Pre-trigger (ยังไม่ถึงจุด Trigger)"
            lines.append(f"- **Breakout Setup:** Trigger ${_fmt(ts.breakout_trigger_price)} | Target ${_fmt(ts.breakout_target_price)} | Stop ${_fmt(ts.breakout_stop_loss)} | Status: `{ts.breakout_entry_status}` (Planned R:R: {_fmt(ts.breakout_planned_rr)} : 1 | Current R:R: {curr_rr_str})")
        lines.append("")

    # -------------------------------------------------------------
    # 🕵️ SEC Form 4 Insider Conviction
    # -------------------------------------------------------------
    if q.insider_conviction:
        ic = q.insider_conviction
        lines.extend([
            "### 🕵️ Canonical SEC Form 4 Insider Conviction (90 Days)\n",
            f"- **Insider Status:** `{ic.status.upper()}` (Data Status: `{ic.data_status}`)",
            f"- Open-Market Purchases (Code P): {ic.open_market_p_count_90d} transactions (${_fmt_large(ic.open_market_p_value_usd, '$')})",
            f"- C-Suite Purchases (CEO/CFO/COO): {ic.c_suite_p_count} transactions",
            f"- Contextual Selling (Code S): {ic.open_market_s_count_90d} transactions (${_fmt_large(ic.open_market_s_value_usd, '$')})",
            f"- Insider Purchase Range: ${_fmt(ic.insider_buy_range_min)} - ${_fmt(ic.insider_buy_range_max)}",
            "",
        ])

    # -------------------------------------------------------------
    # 🛑 Thesis Falsifiers (Kill-Switches)
    # -------------------------------------------------------------
    if q.thesis_falsifiers:
        lines.extend([
            "### 🛑 Thesis Falsifiers & Invalidation Criteria (Kill-Switches)\n",
            "| ID | Metric / Condition | Threshold | Source Reference | คำอธิบาย |",
            "|---|---|---|---|---|",
        ])
        for tf in q.thesis_falsifiers:
            th_str = f"{tf.threshold_value}" if tf.threshold_value is not None else "N/A"
            ref_str = f"`{tf.source_ref}`" if tf.source_ref else "N/A"
            lines.append(f"| `{tf.falsifier_id}` | {tf.metric_name}: {tf.condition} | {th_str} | {ref_str} | {tf.narrative_explanation} |")
        lines.append("")

    lines.extend([
        "### 📈 Growth Signals\n",
        f"- Revenue Growth YoY: {_fmt(q.revenue_growth_yoy_pct)}%",
        f"- Net Income Growth YoY: {_fmt(q.net_income_growth_yoy_pct)}%",
        "",
        "### 🎁 Dividend Quality\n",
        f"- Dividend Yield: {_fmt(q.dividend_yield_pct)}%",
        f"- Payout Ratio: {_fmt(q.payout_ratio_pct)}%",
        "> ข้อควรระวัง: หุ้นที่มี Payout Ratio สูงเกิน 100% อาจเป็น Value Trap หรือมีความเสี่ยงในการลดการจ่ายปันผล\n",
        "",
        "### 💵 Cash Flow & Capital Quality\n",
        f"- FCF Yield: {_fmt(q.fcf_yield_pct)}%",
        f"- FCF Margin: {_fmt(q.fcf_margin_pct)}%",
        f"- ROIC: {_fmt(q.roic_pct)}%",
        f"- OCF / Net Income: {_fmt(q.ocf_to_net_income)}",
        f"- FCF Quality Score: {_fmt(q.fcf_quality_score)} / 100",
        f"- Debt Quality Score: {_fmt(q.debt_quality_score)} / 100",
        "",
        "### ⚠️ Solvency (Risk Gate — ไม่รวมใน Composite Score)\n",
        f"- Solvency Score: {_fmt(q.solvency_score)} / 100",
        f"- D/E Ratio: {_fmt(q.de_ratio_pct)}%",
        f"- Current Ratio: {_fmt(q.current_ratio)}x",
        "",
        "### 👥 Peer/Sector Comparison (Contextual — ไม่รวมใน Composite Score)\n",
        f"- Peer Sector: {_fmt(q.peer_sector)}",
        f"- Peer Relative Score: {_fmt(q.peer_relative_score)} / 100",
        f"- P/E vs Peer Avg: {_fmt(q.pe_vs_peer_avg_pct)}%",
        f"- Peer Count: {_fmt(q.peer_count)}",
        "",
        "### ⏳ Historical Price Context (ไม่ใช่ Valuation Multiple — ไม่รวมใน Composite Score)\n",
        f"- Price Percentile (5Y): {_fmt(q.price_percentile_5y) + ('%' if q.price_percentile_5y is not None else '')}",
        f"- Price Z-Score (5Y): {_fmt(q.price_zscore_5y)}",
        "",
        "### 🚀 Earnings Momentum & Revisions (Contextual — ไม่รวมใน Composite Score)\n",
        f"- Earnings Momentum Score: {_fmt(q.earnings_momentum_score)} / 100",
        f"- EPS Revision Net (30D): {_fmt(q.eps_revision_net_30d)}",
        f"- EPS Estimate Change (30D): {_fmt(q.eps_estimate_change_30d_pct)}%",
        "",
        "### 💧 Trading Liquidity\n",
        f"- ADTV (Average Daily Trading Value): {_fmt_large(q.adtv_local_currency, currency_label)}",
        "",
    ])

    if q.data_quality_flags:
        lines.append("### ⚠️ Data Quality Flags\n")
        for flag in q.data_quality_flags:
            lines.append(_translate_flag(flag))
        lines.append("")

    lines.append("## 📰 Sentiment & Narrative Context\n")

    if s.key_themes:
        lines.append("**ธีมสำคัญ:** " + ", ".join(s.key_themes))
        lines.append("")
    if s.tail_risks:
        lines.append("**ความเสี่ยงแฝง (Tail Risks):**")
        for risk in s.tail_risks:
            sanitized_risk = risk
            if q.price_percentile_5y is None:
                import re
                sanitized_risk = re.sub(r"\(Price Percentile \d+(\.\d+)?%?\)", "", sanitized_risk)
                sanitized_risk = re.sub(r"Price Percentile \d+(\.\d+)?%?", "สถิติราคาในอดีต (ไม่ผ่าน Quality Check)", sanitized_risk)
                sanitized_risk = re.sub(r"\s+", " ", sanitized_risk).strip()
            lines.append(f"- {sanitized_risk}")
        lines.append("")
    lines.append(f"> {s.sources_summary}\n")

    lines.extend([
        "## 📝 บทวิเคราะห์ (Narrative)\n",
        output.narrative_analysis,
        "",
        "## 🎯 Base Case Summary\n",
        output.base_case_summary,
        "",
        "## Related\n",
        f"- {output.ticker}",
        "",
        "## หมายเหตุ\n",
        "> ตัวเลข Quant Signals ทั้งหมดคำนวณแบบ Deterministic จาก Yahoo Finance — LLM ไม่มีส่วนในการคำนวณตัวเลข",
        "> ใช้ประกอบการวิเคราะห์เท่านั้น ไม่ใช่คำแนะนำการลงทุน",
        "",
    ])

    return "\n".join(lines)
