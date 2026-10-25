"""Unit tests for Core Event Study Engine.

Verifies deterministic financial math, look-ahead bias prevention (entry at Open[t+1]),
horizon forward returns, daily trajectory calculations, and statistical guardrails.
"""
import pytest
from core.event_study.models import DailyPriceBar
from core.event_study.engine import EventStudyEngine


def _make_dummy_bars(count: int = 100, base_price: float = 100.0, step: float = 1.0) -> list[DailyPriceBar]:
    """Generates synthetic trading days (skipping weekends)."""
    bars = []
    # Dates starting from 2023-01-02
    from datetime import date, timedelta
    curr = date(2023, 1, 2)
    price = base_price
    while len(bars) < count:
        if curr.weekday() < 5:  # Monday to Friday
            bars.append(
                DailyPriceBar(
                    date=curr.isoformat(),
                    open=round(price, 2),
                    high=round(price + 2.0, 2),
                    low=round(price - 1.0, 2),
                    close=round(price + step, 2),
                    volume=10000.0,
                )
            )
            price += step
        curr += timedelta(days=1)
    return bars


def test_event_study_empty_bars():
    report = EventStudyEngine.evaluate(
        ticker="AAPL",
        event_name="Empty Test",
        bars=[],
        event_items=["2023-01-10"],
    )
    assert report.ticker == "AAPL"
    assert len(report.events) == 0
    assert any(w.code == "NO_DATA_AVAILABLE" for w in report.warnings)
    assert report.horizon_stats[5].sample_size == 0


def test_event_study_entry_at_next_trading_day_open():
    """Verifies that an event on day t enters at Open[t+1]."""
    bars = _make_dummy_bars(count=30, base_price=100.0, step=1.0)
    # Day 0: 2023-01-02, Open=100.0, Close=101.0
    # Day 1: 2023-01-03, Open=101.0, Close=102.0
    # Event on 2023-01-02 -> Entry is on 2023-01-03 at Open=101.0
    report = EventStudyEngine.evaluate(
        ticker="TEST",
        event_name="Product Launch",
        bars=bars,
        event_items=["2023-01-02"],
        horizons=[5, 10],
    )
    assert len(report.events) == 1
    ev = report.events[0]
    assert ev.event_date == "2023-01-02"
    assert ev.entry_date == "2023-01-03"
    assert ev.entry_price == 101.0

    # Horizon 5: Target bar is Day 1 + 4 = Day 5 (2023-01-09)
    # Day 5 close: 100 + 1*6 = 106.0
    # Expected return = (106.0 / 101.0) - 1 = +4.95%
    ret_5d = ev.forward_returns[5]
    assert ret_5d is not None
    assert round(ret_5d, 2) == 4.95
    assert ev.is_win_by_horizon[5] is True


def test_event_study_weekend_event_handling():
    """Event on a weekend should map entry to Monday (first available trading day)."""
    bars = _make_dummy_bars(count=30, base_price=100.0, step=1.0)
    # 2023-01-07 is Saturday, 2023-01-08 is Sunday, 2023-01-09 is Monday (Bar index 5)
    report = EventStudyEngine.evaluate(
        ticker="TEST",
        event_name="Weekend Event",
        bars=bars,
        event_items=["2023-01-07"],
        horizons=[5],
    )
    assert len(report.events) == 1
    ev = report.events[0]
    assert ev.event_date == "2023-01-07"
    assert ev.entry_date == "2023-01-09"
    assert ev.entry_price == bars[5].open


def test_event_study_distribution_metrics():
    """Tests Hit rate, Median, Best/Worst, and Risk/Reward with positive and negative events."""
    # Create 6 distinct events
    bars = _make_dummy_bars(count=80, base_price=100.0, step=0.0)
    # Manually tweak closes for specific event targets
    # We will pick 6 event dates
    event_dates = [bars[i * 10].date for i in range(6)]

    # Make 4 wins and 2 losses at 5d
    # Entry for event i is at bar i*10 + 1
    # 5d target is at bar i*10 + 5
    for i in range(4):
        # Win: price goes up 10%
        target_idx = i * 10 + 5
        entry_idx = i * 10 + 1
        entry_open = bars[entry_idx].open
        bars[target_idx] = DailyPriceBar(
            date=bars[target_idx].date,
            open=bars[target_idx].open,
            high=entry_open * 1.15,
            low=entry_open * 0.95,
            close=entry_open * 1.10,  # +10%
            volume=10000.0,
        )
    for i in range(4, 6):
        # Loss: price goes down 5%
        target_idx = i * 10 + 5
        entry_idx = i * 10 + 1
        entry_open = bars[entry_idx].open
        bars[target_idx] = DailyPriceBar(
            date=bars[target_idx].date,
            open=bars[target_idx].open,
            high=entry_open * 1.02,
            low=entry_open * 0.90,
            close=entry_open * 0.95,  # -5%
            volume=10000.0,
        )

    report = EventStudyEngine.evaluate(
        ticker="AAPL",
        event_name="Earnings",
        bars=bars,
        event_items=event_dates,
        horizons=[5],
    )

    stats = report.horizon_stats[5]
    assert stats.sample_size == 6
    # 4 wins out of 6 -> 66.67%
    assert stats.hit_rate_pct == 66.67
    assert stats.win_count == 4
    assert stats.loss_count == 2
    assert stats.best_return_pct == 10.0
    assert stats.worst_return_pct == -5.0
    assert stats.avg_win_pct == 10.0
    assert stats.avg_loss_pct == -5.0
    # Risk/Reward = 10.0 / 5.0 = 2.0
    assert stats.risk_reward_ratio == 2.0
    # Since N=6 >= 5, no UNRELIABLE_SAMPLE_SIZE warning
    assert not any(w.code == "UNRELIABLE_SAMPLE_SIZE" for w in report.warnings)


def test_event_study_sample_size_guardrail():
    """Tests that N < 5 triggers UNRELIABLE_SAMPLE_SIZE warning."""
    bars = _make_dummy_bars(count=40)
    event_dates = [bars[2].date, bars[10].date]  # Only 2 events
    report = EventStudyEngine.evaluate(
        ticker="NVDA",
        event_name="GTC",
        bars=bars,
        event_items=event_dates,
        horizons=[5],
    )
    assert len(report.events) == 2
    assert any(w.code == "UNRELIABLE_SAMPLE_SIZE" for w in report.warnings)


def test_event_study_daily_trajectory_consistency():
    """Tests that daily trajectory points are contiguous and cumulative return increases properly."""
    bars = _make_dummy_bars(count=30, base_price=100.0, step=2.0)
    report = EventStudyEngine.evaluate(
        ticker="TSLA",
        event_name="AI Day",
        bars=bars,
        event_items=[bars[2].date],
        horizons=[5, 10],
    )
    ev = report.events[0]
    assert len(ev.daily_trajectory) == 10
    assert ev.daily_trajectory[0].day_offset == 1
    assert ev.daily_trajectory[9].day_offset == 10
    # Day 5 return in trajectory matches forward_returns[5]
    traj_day5_ret = ev.daily_trajectory[4].cumulative_return_pct
    assert traj_day5_ret == ev.forward_returns[5]
    # Average trajectory matches single event
    assert len(report.average_trajectory) == 10
    assert report.average_trajectory[4].avg_return_pct == ev.forward_returns[5]
