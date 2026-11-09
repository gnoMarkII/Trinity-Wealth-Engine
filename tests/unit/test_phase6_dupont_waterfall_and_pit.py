from decimal import Decimal
import pytest

from core.equity_analytics.dupont_accrual import (
    DuPont3FactorBreakdown,
    SloanAccrualAnalysis,
    compute_dupont_3factor,
    compute_sloan_accrual_ratio,
)
from core.equity_analytics.signal_audit import (
    audit_signal_maturity,
    compute_rank_ic,
    compute_newey_west_hac_tstat,
)
from tools.market.quant_scoring import compute_composite_score_waterfall


def test_dupont_3factor_exact_identity():
    # Net Income 10k, Rev 100k, Assets 200k, Equity 50k
    # Margin = 10%, Turnover = 0.5x, Leverage = 4.0x -> ROE = 10 * 0.5 * 4 = 20%
    dp = compute_dupont_3factor(
        net_income=Decimal("10000"),
        revenue=Decimal("100000"),
        total_assets=Decimal("200000"),
        stockholders_equity=Decimal("50000"),
    )
    assert isinstance(dp, DuPont3FactorBreakdown)
    assert dp.net_profit_margin_pct == Decimal("10.00")
    assert dp.asset_turnover == Decimal("0.5000")
    assert dp.equity_multiplier == Decimal("4.00")
    assert dp.roe_pct == Decimal("20.00")
    assert dp.closure_error <= Decimal("0.0001")


def test_sloan_accrual_ratings():
    # 1. High Accruals -> Warning
    # NI 20k, OCF 5k, Assets 100k -> Accrual = (20k - 5k)/100k = +0.15 > +0.10
    sl_warn = compute_sloan_accrual_ratio(
        net_income=Decimal("20000"),
        operating_cash_flow=Decimal("5000"),
        total_assets_current=Decimal("100000"),
    )
    assert sl_warn.earnings_quality_rating == "EARNINGS_MANIPULATION_WARNING"
    assert sl_warn.sloan_accrual_ratio == Decimal("0.1500")

    # 2. Strong Cash Flow -> High Quality
    # NI 10k, OCF 25k, Assets 100k -> Accrual = (10k - 25k)/100k = -0.15 < -0.10
    sl_good = compute_sloan_accrual_ratio(
        net_income=Decimal("10000"),
        operating_cash_flow=Decimal("25000"),
        total_assets_current=Decimal("100000"),
    )
    assert sl_good.earnings_quality_rating == "HIGH_QUALITY"
    assert sl_good.sloan_accrual_ratio == Decimal("-0.1500")

    # 3. Balanced Cash Flow -> Neutral
    sl_neutral = compute_sloan_accrual_ratio(
        net_income=Decimal("10000"),
        operating_cash_flow=Decimal("9500"),
        total_assets_current=Decimal("100000"),
    )
    assert sl_neutral.earnings_quality_rating == "NEUTRAL"
    assert sl_neutral.sloan_accrual_ratio == Decimal("0.0050")


def test_composite_score_waterfall():
    # All 5 dimensions present: Value 80, Quality 70, Growth 60, Momentum 50, Dividend 40
    # Weights: 0.25, 0.25, 0.25, 0.15, 0.10 (sum = 1.00)
    # Expected: 80*0.25 + 70*0.25 + 60*0.25 + 50*0.15 + 40*0.10 = 20 + 17.5 + 15 + 7.5 + 4 = 64.0
    waterfall = compute_composite_score_waterfall(
        value_score=80.0,
        quality_score=70.0,
        growth_score=60.0,
        momentum_score=50.0,
        dividend_score=40.0,
    )
    assert waterfall["composite_score"] == 64.0
    assert waterfall["pillar_contributions"]["value"] == 20.0
    assert waterfall["pillar_contributions"]["quality"] == 17.5
    assert waterfall["pillar_contributions"]["growth"] == 15.0
    assert waterfall["pillar_contributions"]["momentum"] == 7.5
    assert waterfall["pillar_contributions"]["dividend"] == 4.0
    assert waterfall["available_dimensions_count"] == 5

    # Partial dimensions (Dividend is None): re-normalize remaining weights
    waterfall_partial = compute_composite_score_waterfall(
        value_score=80.0,
        quality_score=80.0,
        growth_score=None,
        momentum_score=None,
        dividend_score=None,
    )
    # Value (0.25) + Quality (0.25) -> 50% each -> 80.0
    assert waterfall_partial["composite_score"] == 80.0
    assert waterfall_partial["available_dimensions_count"] == 2


def test_audit_signal_maturity():
    filing = "2026-03-01"
    # Same day or next day with 2-day embargo -> False (not matured yet)
    assert audit_signal_maturity(as_of_date="2026-03-01", filing_date=filing, embargo_days=2) is False
    assert audit_signal_maturity(as_of_date="2026-03-02", filing_date=filing, embargo_days=2) is False

    # Day 3 onwards -> True (mature for trading)
    assert audit_signal_maturity(as_of_date="2026-03-03", filing_date=filing, embargo_days=2) is True
    assert audit_signal_maturity(as_of_date="2026-03-10", filing_date=filing, embargo_days=2) is True


def test_rank_ic_and_newey_west():
    # Perfect monotonic alignment -> IC = 1.0
    factors = [10.0, 20.0, 30.0, 40.0, 50.0]
    returns = [0.01, 0.02, 0.03, 0.04, 0.05]
    ic = compute_rank_ic(factors, returns)
    assert ic == 1.0

    # Perfect inverse monotonic alignment -> IC = -1.0
    ic_inv = compute_rank_ic(factors, list(reversed(returns)))
    assert ic_inv == -1.0

    # Newey-West t-statistic on consistent positive IC series
    ic_history = [0.08, 0.06, 0.07, 0.09, 0.05, 0.07, 0.08, 0.06]
    t_stat = compute_newey_west_hac_tstat(ic_history)
    assert t_stat is not None
    assert t_stat > 2.0  # Statistically significant positive factor alpha


def test_r18_dupont_published_reconstruction_and_invalid_inputs():
    """R18 / W05: DuPont publishes true factor-reconstruction residual & rejects non-finite inputs."""
    # Exact R18 reproduction: NI=1, Rev=100, Assets=10,000,000, Equity=10
    # Margin=1.00%, Turnover=0.0000, Multiplier=1,000,000.00, ROE=10.00%
    dp = compute_dupont_3factor(
        net_income=Decimal("1"),
        revenue=Decimal("100"),
        total_assets=Decimal("10000000"),
        stockholders_equity=Decimal("10"),
    )
    assert dp is not None
    assert dp.net_profit_margin_pct == Decimal("1.00")
    assert dp.asset_turnover == Decimal("0.0000")
    assert dp.equity_multiplier == Decimal("1000000.00")
    assert dp.roe_pct == Decimal("10.00")
    # Published factor product is 0.0000 -> honest discrepancy of 10.0000 percentage points
    assert dp.published_roe_reconstructed_pct == Decimal("0.0000")
    assert dp.closure_error == Decimal("10.0000")
    assert dp.is_published_reconciled is False
    assert dp.internal_closure_error == Decimal("0.00000000")

    # Non-positive denominators return None; non-finite values raise ValueError
    assert compute_dupont_3factor(Decimal("1"), Decimal("0"), Decimal("10"), Decimal("10")) is None
    with pytest.raises(ValueError, match="finite"):
        compute_dupont_3factor(Decimal("NaN"), Decimal("100"), Decimal("200"), Decimal("50"))


def test_r18_sloan_raw_threshold_precision_and_display_invariance():
    """R18 / W05: Sloan classifies from unrounded canonical Decimal ratio regardless of display quantum."""
    # Exact R18 reproduction: +0.10004 and -0.10004 round to +/-0.1000 at 4dp
    # Strict rules > +0.10 and < -0.10 MUST NOT fall into NEUTRAL
    sl_pos = compute_sloan_accrual_ratio(
        net_income=Decimal("110004"),
        operating_cash_flow=Decimal("100000"),
        total_assets_current=Decimal("100000"),
    )
    assert sl_pos is not None
    assert sl_pos.sloan_accrual_ratio == Decimal("0.1000")
    assert sl_pos.raw_sloan_accrual_ratio == Decimal("0.10004")
    assert sl_pos.earnings_quality_rating == "EARNINGS_MANIPULATION_WARNING"

    sl_neg = compute_sloan_accrual_ratio(
        net_income=Decimal("100000"),
        operating_cash_flow=Decimal("110004"),
        total_assets_current=Decimal("100000"),
    )
    assert sl_neg is not None
    assert sl_neg.sloan_accrual_ratio == Decimal("-0.1000")
    assert sl_neg.raw_sloan_accrual_ratio == Decimal("-0.10004")
    assert sl_neg.earnings_quality_rating == "HIGH_QUALITY"

    # Changing display precision to 2dp (0.10) still preserves unrounded classification!
    sl_coarse = compute_sloan_accrual_ratio(
        net_income=Decimal("110004"),
        operating_cash_flow=Decimal("100000"),
        total_assets_current=Decimal("100000"),
        display_quantum=Decimal("0.01"),
    )
    assert sl_coarse is not None
    assert sl_coarse.sloan_accrual_ratio == Decimal("0.10")
    assert sl_coarse.earnings_quality_rating == "EARNINGS_MANIPULATION_WARNING"

    # Exact +/-0.100000 stays NEUTRAL; non-positive prior assets returns None
    sl_exact_pos = compute_sloan_accrual_ratio(
        net_income=Decimal("110000"),
        operating_cash_flow=Decimal("100000"),
        total_assets_current=Decimal("100000"),
    )
    assert sl_exact_pos is not None and sl_exact_pos.earnings_quality_rating == "NEUTRAL"
    sl_exact_neg = compute_sloan_accrual_ratio(
        net_income=Decimal("100000"),
        operating_cash_flow=Decimal("110000"),
        total_assets_current=Decimal("100000"),
    )
    assert sl_exact_neg is not None and sl_exact_neg.earnings_quality_rating == "NEUTRAL"
    assert compute_sloan_accrual_ratio(Decimal("1"), Decimal("0"), Decimal("100"), Decimal("0")) is None


def test_w03_signal_audit_negative_delay_alignment_and_zero_variance():
    """W03 / W05: Reject negative embargo delay, date garbage, mismatched IC vectors, and constant samples."""
    # Negative delay rejected
    with pytest.raises(ValueError, match="embargo_days"):
        audit_signal_maturity("2026-11-09", "2026-11-10", embargo_days=-2)
    with pytest.raises(ValueError, match="embargo_days"):
        audit_signal_maturity("2026-11-09", "2026-11-10", embargo_days=True)  # type: ignore[arg-type]

    # Malformed trailing garbage does not parse via [:10] slicing
    assert audit_signal_maturity("2026-03-10garbage", "2026-03-01", embargo_days=2) is False
    # Valid ISO timestamp parses cleanly
    assert audit_signal_maturity("2026-03-03T09:00:00Z", "2026-03-01", embargo_days=2) is True

    # Length mismatch rejected instead of prefix slicing
    with pytest.raises(ValueError, match="mismatch"):
        compute_rank_ic([1.0, 2.0, 3.0, 4.0], [0.1, 0.2, 0.3])

    # Constant factor returns None (undefined) rather than 0.0
    assert compute_rank_ic([5.0, 5.0, 5.0, 5.0], [0.01, 0.02, 0.03, 0.04]) is None

    # Negative max_lags rejected; constant IC series returns None rather than 0.0
    with pytest.raises(ValueError, match="max_lags"):
        compute_newey_west_hac_tstat([0.05] * 6, max_lags=-1)
    assert compute_newey_west_hac_tstat([0.05] * 6) is None
