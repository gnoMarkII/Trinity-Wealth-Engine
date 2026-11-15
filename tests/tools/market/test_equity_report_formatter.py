"""Unit tests สำหรับ tools/market/equity_report_formatter.py"""
from schemas.micro_quant_schemas import (
    QuantSignals,
    EquitySentimentContext,
    MicroQuantOutput,
)
from tools.market.equity_report_formatter import format_equity_analysis_report


def _make_output(**overrides) -> MicroQuantOutput:
    defaults = dict(
        ticker="AAPL",
        market="US",
        analysis_date="2026-07-20",
        quant_signals=QuantSignals(
            ticker="AAPL",
            market="US",
            value_score=73.3,
            quality_score=96.7,
            momentum_score=95.8,
            beta=1.2,
            volatility_pct=28.5,
            mdd_pct=-18.0,
            upside_pct=20.0,
            downside_pct=-5.0,
            evaluated_at="2026-07-20T10:00:00+00:00",
            data_quality_flags=[],
        ),
        sentiment_context=EquitySentimentContext(
            evaluated_at="2026-07-20T10:00:00+00:00",
            market_sentiment="bullish",
            key_themes=["AI capex cycle"],
            tail_risks=["Regulatory risk in EU"],
            sources_summary="Vault + latest news",
        ),
        narrative_analysis="ราคาหุ้นอยู่ในทิศทางขาขึ้นต่อเนื่อง",
        base_case_summary="คงมุมมองเชิงบวกในกรอบ 3-6 เดือน",
    )
    defaults.update(overrides)
    return MicroQuantOutput(**defaults)


def test_has_valid_researcher_frontmatter():
    report = format_equity_analysis_report(_make_output())
    stripped = report.lstrip()
    assert stripped.startswith("---")
    head = "\n".join(stripped.splitlines()[:10])
    assert "entity_type:" in head
    assert "entity_type: equity_analysis" in report
    assert "ticker: AAPL" in report


def test_numeric_values_appear_verbatim_not_rewritten():
    report = format_equity_analysis_report(_make_output())
    assert "73.3" in report
    assert "96.7" in report
    assert "95.8" in report
    assert "1.2" in report
    assert "28.5" in report
    assert "-18.0" in report
    assert "20.0" in report
    assert "-5.0" in report


def test_narrative_and_base_case_included():
    report = format_equity_analysis_report(_make_output())
    assert "ราคาหุ้นอยู่ในทิศทางขาขึ้นต่อเนื่อง" in report
    assert "คงมุมมองเชิงบวกในกรอบ 3-6 เดือน" in report


def test_none_values_render_as_na():
    output = _make_output(
        quant_signals=QuantSignals(
            ticker="LOSSCO",
            market="US",
            evaluated_at="2026-07-20T10:00:00+00:00",
            data_quality_flags=["negative_earnings:pe_undefined"],
        )
    )
    report = format_equity_analysis_report(output)
    assert "N/A" in report
    assert "negative_earnings:pe_undefined" in report


def test_sentiment_label_rendered():
    report = format_equity_analysis_report(_make_output())
    assert "Bullish" in report


def test_title_uses_company_name_when_available():
    output = _make_output()
    output.quant_signals.company_name = "Apple Inc."
    report = format_equity_analysis_report(output)
    assert "Apple Inc. (AAPL)" in report


def test_title_falls_back_to_ticker_when_no_company_name():
    report = format_equity_analysis_report(_make_output())
    assert "# 📊 บทวิเคราะห์เชิงปริมาณ: AAPL (US)" in report


def test_composite_score_shown_as_hero_number():
    output = _make_output()
    output.quant_signals.composite_score = 82.5
    report = format_equity_analysis_report(output)
    assert "Composite Score: 82.5 / 100" in report


def test_us_beta_benchmark_label_is_gspc():
    report = format_equity_analysis_report(_make_output())
    assert "เทียบ ^GSPC" in report


def test_th_beta_benchmark_label_is_set_index():
    output = _make_output(ticker="PTT", market="TH", quant_signals=QuantSignals(
        ticker="PTT", market="TH", beta=0.9, adtv_local_currency=50_000_000.0,
        evaluated_at="2026-07-20T10:00:00+00:00",
    ))
    report = format_equity_analysis_report(output)
    assert "เทียบ ^SET.BK" in report
    assert "THB" in report  # ADTV currency label ต้องเป็น THB สำหรับ TH


def test_adtv_formatted_as_abbreviated_large_number():
    output = _make_output()
    output.quant_signals.adtv_local_currency = 17_444_389_267.62
    report = format_equity_analysis_report(output)
    assert "17.44B USD" in report
    assert "17444389267" not in report  # ห้ามโชว์ตัวเลขดิบยาวๆ


def test_adtv_none_renders_as_na_without_currency_suffix():
    report = format_equity_analysis_report(_make_output())  # adtv_local_currency ไม่ตั้งค่า -> None
    assert "ADTV (Average Daily Trading Value): N/A" in report


def test_solvency_labeled_as_risk_gate_excluded_from_composite():
    output = _make_output()
    output.quant_signals.solvency_score = 15.0
    output.quant_signals.de_ratio_pct = 500.0
    report = format_equity_analysis_report(output)
    assert "Risk Gate" in report
    assert "ไม่รวมใน Composite Score" in report
    assert "15.0" in report


def test_growth_and_dividend_sections_render_values():
    output = _make_output()
    output.quant_signals.revenue_growth_yoy_pct = 12.5
    output.quant_signals.dividend_yield_pct = 3.2
    output.quant_signals.payout_ratio_pct = 45.0
    report = format_equity_analysis_report(output)
    assert "12.5" in report
    assert "3.2" in report
    assert "45.0" in report
    assert "Value Trap" in report  # คำเตือน dividend ต้องอยู่ในรายงาน


def test_peer_relative_section_renders_and_labeled_contextual():
    output = _make_output()
    output.quant_signals.peer_sector = "Technology"
    output.quant_signals.peer_count = 3
    output.quant_signals.pe_vs_peer_avg_pct = -12.5
    output.quant_signals.peer_relative_score = 80.0
    report = format_equity_analysis_report(output)
    assert "Peer/Sector Comparison" in report
    assert "Technology" in report
    assert "-12.5" in report
    assert "80.0" in report
    assert "ไม่รวมใน Composite Score" in report


def test_price_percentile_section_labeled_as_price_not_valuation():
    output = _make_output()
    output.quant_signals.price_percentile_5y = 88.0
    output.quant_signals.price_zscore_5y = 1.8
    report = format_equity_analysis_report(output)
    assert "Historical Price Context" in report
    assert "88.0" in report
    assert "1.8" in report
    assert "ไม่ใช่ Valuation Multiple" in report  # ต้องกันความเข้าใจผิดว่าเป็น P/E percentile


def test_earnings_momentum_section_renders_values():
    output = _make_output()
    output.quant_signals.eps_revision_net_30d = 3
    output.quant_signals.eps_estimate_change_30d_pct = 4.2
    output.quant_signals.earnings_momentum_score = 92.0
    report = format_equity_analysis_report(output)
    assert "Earnings Momentum" in report
    assert "4.2" in report
    assert "92.0" in report
    assert "ไม่รวมใน Composite Score" in report


def test_valuation_triangulation_renders_consensus_mean_and_exit_multiple():
    from schemas.micro_quant_schemas import ReverseDCFResult, ValuationSanityAudit

    output = _make_output()
    output.quant_signals.raw_analysis_price = 62.50
    output.quant_signals.data_quality_flags = ["tax_policy:normalized_statutory"]
    output.quant_signals.reverse_dcf_result = ReverseDCFResult(
        status="available",
        is_eligible=True,
        valuation_verdict="overvalued",
        intrinsic_value_today=12.0,
        target_price_12m=13.23,
        upside_12m_pct=-78.8,
        target_price_exit_multiple_12m=64.50,
        exit_multiple_used=25.0,
        implied_exit_ebitda_multiple=4.2,
        consensus_target_price=90.65,
        consensus_target_low=25.0,
        consensus_target_high=160.0,
        analyst_count=39,
        valuation_sanity_audit=ValuationSanityAudit(
            sanity_warnings=["Depressed terminal exit multiple"],
        ),
    )
    report = format_equity_analysis_report(output)
    assert "Consensus Target Mean:** **$90.65** (Upside: +45.0%) | Range: $25.00 - $160.00 | 39 analysts" in report
    assert "Exit-Multiple Target Price (12M, EV/EBITDA 25.0x):** **$64.50** (Upside: +3.2%)" in report
    assert "Implied Exit EV/EBITDA (Gordon TV):** 4.2x" in report
    assert "Valuation Sanity Notes:** Depressed terminal exit multiple" in report
    assert "tax_policy:normalized_statutory" in report
