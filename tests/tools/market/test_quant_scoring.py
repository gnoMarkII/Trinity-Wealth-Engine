"""Unit tests สำหรับ tools/market/quant_scoring.py"""
from tools.market.quant_scoring import (
    compute_value_score,
    compute_quality_score,
    compute_momentum_score,
    compute_price_target_outlook,
    compute_growth_score,
    compute_dividend_score,
    compute_solvency_score,
    compute_trading_liquidity,
    compute_composite_score,
)


class TestValueScore:
    def test_negative_pe_flags_undefined_not_cheap(self):
        score, flag = compute_value_score(pe=-5.0)
        assert score is None
        assert flag == "negative_earnings:pe_undefined"

    def test_cheap_pe_scores_high(self):
        score, flag = compute_value_score(pe=8.0)
        assert flag is None
        assert score == 100.0

    def test_expensive_pe_scores_zero(self):
        score, flag = compute_value_score(pe=50.0)
        assert flag is None
        assert score == 0.0

    def test_mid_pe_interpolates(self):
        score, flag = compute_value_score(pe=25.0)  # midpoint ของ 10-40
        assert flag is None
        assert 40.0 <= score <= 60.0

    def test_missing_all_metrics_returns_none_with_flag(self):
        score, flag = compute_value_score(pe=None, pb=None, ev_ebitda=None)
        assert score is None
        assert flag == "missing_valuation_metrics"

    def test_combines_multiple_metrics(self):
        score, flag = compute_value_score(pe=10.0, pb=1.0, ev_ebitda=8.0)  # ทุกตัวเข้าเกณฑ์ 'ถูก' สุด
        assert flag is None
        assert score == 100.0


class TestQualityScore:
    def test_high_roe_scores_high(self):
        score, flag = compute_quality_score(roe_pct=25.0)
        assert flag is None
        assert score == 100.0

    def test_low_roe_scores_low(self):
        score, flag = compute_quality_score(roe_pct=2.0)
        assert flag is None
        assert score == 0.0

    def test_missing_metrics_returns_none(self):
        score, flag = compute_quality_score(roe_pct=None, profit_margin_pct=None, fcf_debt_ratio=None)
        assert score is None
        assert flag == "missing_quality_metrics"


class TestMomentumScore:
    def test_high_rsi_bullish_macd_golden_cross_scores_high(self):
        score, flag = compute_momentum_score(rsi_14=75.0, macd_signal="bullish", ma50_vs_ma200="golden_cross")
        assert flag is None
        assert score == 100.0

    def test_low_rsi_bearish_macd_death_cross_scores_low(self):
        score, flag = compute_momentum_score(rsi_14=20.0, macd_signal="bearish", ma50_vs_ma200="death_cross")
        assert flag is None
        assert score == 0.0

    def test_missing_all_inputs_returns_none(self):
        score, flag = compute_momentum_score(rsi_14=None, macd_signal=None, ma50_vs_ma200=None)
        assert score is None
        assert flag == "insufficient_technical_data:momentum"

    def test_momentum_score_none_when_only_ma_cross_available(self):
        """MA cross alone without RSI or MACD must return None with insufficient data flag."""
        score, flag = compute_momentum_score(rsi_14=None, macd_signal=None, ma50_vs_ma200="golden_cross")
        assert score is None
        assert flag == "insufficient_technical_data:momentum"


class TestPriceTargetOutlook:
    def test_no_current_price_returns_none_none(self):
        upside, downside = compute_price_target_outlook(current_price=None, target_high=120.0, target_low=80.0)
        assert upside is None
        assert downside is None

    def test_no_target_prices_returns_none_none(self):
        # หุ้นไทยหลายตัวไม่มี analyst target price ใน yfinance
        upside, downside = compute_price_target_outlook(current_price=100.0)
        assert upside is None
        assert downside is None

    def test_computes_upside_from_target_high(self):
        upside, downside = compute_price_target_outlook(current_price=100.0, target_high=120.0, target_low=80.0)
        assert upside == 20.0
        assert downside == -20.0

    def test_upside_falls_back_to_target_mean_when_no_high(self):
        upside, _ = compute_price_target_outlook(current_price=100.0, target_mean=110.0)
        assert upside == 10.0


class TestGrowthScore:
    def test_missing_revenue_growth_returns_none(self):
        score, flag = compute_growth_score(revenue_growth_yoy_pct=None)
        assert score is None
        assert flag == "missing_growth_data:growth"

    def test_high_revenue_growth_scores_high(self):
        score, flag = compute_growth_score(revenue_growth_yoy_pct=25.0)
        assert flag is None
        assert score == 100.0

    def test_negative_revenue_growth_scores_low(self):
        score, flag = compute_growth_score(revenue_growth_yoy_pct=-15.0)
        assert flag is None
        assert score == 0.0

    def test_combines_revenue_and_net_income_weighted_2_to_1(self):
        # revenue=20(->100) net_income=-10(->0) => (100*2+0)/3 = 66.7
        score, flag = compute_growth_score(revenue_growth_yoy_pct=20.0, net_income_growth_yoy_pct=-10.0)
        assert flag is None
        assert score == 66.7


class TestDividendScore:
    def test_missing_yield_returns_none(self):
        score, flag = compute_dividend_score(dividend_yield_pct=None)
        assert score is None
        assert flag == "missing_dividend_data:dividend"

    def test_high_yield_scores_100(self):
        score, flag = compute_dividend_score(dividend_yield_pct=6.0)
        assert flag is None
        assert score == 100.0

    def test_zero_yield_scores_0(self):
        score, flag = compute_dividend_score(dividend_yield_pct=0.0)
        assert flag is None
        assert score == 0.0

    def test_unsustainable_payout_flagged_but_score_still_computed(self):
        score, flag = compute_dividend_score(dividend_yield_pct=6.0, payout_ratio_pct=150.0)
        assert score == 100.0  # คะแนนยังคำนวณจาก yield ตามปกติ ไม่ null ทิ้ง
        assert flag == "unsustainable_payout:dividend"

    def test_negative_payout_also_flagged(self):
        _, flag = compute_dividend_score(dividend_yield_pct=3.0, payout_ratio_pct=-5.0)
        assert flag == "unsustainable_payout:dividend"

    def test_normal_payout_no_flag(self):
        _, flag = compute_dividend_score(dividend_yield_pct=3.0, payout_ratio_pct=50.0)
        assert flag is None


class TestSolvencyScore:
    def test_healthy_de_ratio_scores_high(self):
        # de_ratio_pct มาจาก yfinance ตรงๆ (150.0 = D/E 1.5x) — 50% = D/E 0.5x ถือว่าดี
        score, flag = compute_solvency_score(de_ratio_pct=50.0)
        assert score == 100.0
        assert flag is None

    def test_high_leverage_de_ratio_scores_low(self):
        score, flag = compute_solvency_score(de_ratio_pct=300.0, current_ratio=0.5)
        assert score == 0.0
        assert flag == "high_leverage_risk:solvency"

    def test_missing_all_inputs_returns_none(self):
        score, flag = compute_solvency_score(de_ratio_pct=None, current_ratio=None)
        assert score is None
        assert flag == "missing_solvency_data:solvency"

    def test_combines_de_ratio_and_current_ratio(self):
        score, flag = compute_solvency_score(de_ratio_pct=50.0, current_ratio=2.0)
        assert score == 100.0
        assert flag is None


class TestTradingLiquidity:
    def test_prefers_10day_volume_over_3month_average(self):
        adtv, flag = compute_trading_liquidity(avg_volume=1_000_000, avg_volume_10d=2_000_000, current_price=10.0, market="US")
        assert adtv == 20_000_000.0  # ใช้ 10d ไม่ใช่ 3-month average
        assert flag is None

    def test_falls_back_to_avg_volume_when_no_10day(self):
        adtv, _ = compute_trading_liquidity(avg_volume=1_000_000, avg_volume_10d=None, current_price=10.0, market="US")
        assert adtv == 10_000_000.0

    def test_missing_volume_or_price_returns_none(self):
        adtv, flag = compute_trading_liquidity(avg_volume=None, avg_volume_10d=None, current_price=10.0, market="US")
        assert adtv is None
        assert flag == "missing_liquidity_data:liquidity"

    def test_low_liquidity_flagged_for_us_below_1m(self):
        adtv, flag = compute_trading_liquidity(avg_volume=None, avg_volume_10d=50_000, current_price=5.0, market="US")
        assert adtv == 250_000.0
        assert flag == "low_liquidity:liquidity"

    def test_low_liquidity_flagged_for_th_below_5m(self):
        adtv, flag = compute_trading_liquidity(avg_volume=None, avg_volume_10d=100_000, current_price=20.0, market="TH")
        assert adtv == 2_000_000.0
        assert flag == "low_liquidity:liquidity"

    def test_sufficient_liquidity_no_flag(self):
        adtv, flag = compute_trading_liquidity(avg_volume=None, avg_volume_10d=1_000_000, current_price=50.0, market="US")
        assert adtv == 50_000_000.0
        assert flag is None


class TestCompositeScore:
    def test_all_five_dimensions_weighted_correctly(self):
        # 100*0.25 + 80*0.25 + 60*0.25 + 40*0.15 + 20*0.10 = 25+20+15+6+2 = 68.0
        score, flag = compute_composite_score(
            value_score=100.0, quality_score=80.0, growth_score=60.0, momentum_score=40.0, dividend_score=20.0
        )
        assert flag is None
        assert score == 68.0

    def test_missing_one_dimension_renormalizes_weights(self):
        # ไม่มี dividend_score (weight 0.10) -> ใช้ weight ที่เหลือ 0.90 มา renormalize
        # value=100(0.25) quality=100(0.25) growth=100(0.25) momentum=100(0.15) รวม weight=0.90 ทุกตัว=100 -> composite=100
        score, flag = compute_composite_score(
            value_score=100.0, quality_score=100.0, growth_score=100.0, momentum_score=100.0, dividend_score=None
        )
        assert flag is None
        assert score == 100.0

    def test_fewer_than_two_dimensions_returns_none(self):
        score, flag = compute_composite_score(
            value_score=100.0, quality_score=None, growth_score=None, momentum_score=None, dividend_score=None
        )
        assert score is None
        assert flag == "insufficient_dimensions:composite"

    def test_exactly_two_dimensions_computes(self):
        # value=100(0.25) quality=0(0.25) รวม weight=0.5 -> composite = (100*0.25+0*0.25)/0.5 = 50.0
        score, flag = compute_composite_score(
            value_score=100.0, quality_score=0.0, growth_score=None, momentum_score=None, dividend_score=None
        )
        assert flag is None
        assert score == 50.0

    def test_composite_score_renormalization_when_momentum_none(self):
        """FTNT profile: value=0, quality=100, growth=71.7, dividend=0, momentum=None -> composite=50.5."""
        score, flag = compute_composite_score(
            value_score=0.0,
            quality_score=100.0,
            growth_score=71.7,
            momentum_score=None,
            dividend_score=0.0,
        )
        assert flag is None
        # weights: value 0.25, quality 0.25, growth 0.25, dividend 0.10 (sum=0.85)
        # sum = (0*0.25) + (100*0.25) + (71.7*0.25) + (0*0.10) = 42.925 / 0.85 = 50.5
        assert score == 50.5
