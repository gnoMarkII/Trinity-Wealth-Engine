import pandas as pd
import pytest
from datetime import datetime, timezone
from tools.market.technical import compute_tactical_setup


def _get_test_dates() -> pd.DatetimeIndex:
    last_bday = pd.Timestamp.now(timezone.utc).floor("D")
    if last_bday.weekday() >= 5:
        last_bday -= pd.offsets.BDay(1)
    return pd.date_range(end=last_bday, periods=50, freq="B")


def test_tactical_breakout_volume_baseline_and_confirmation():
    # Construct 50 days of data
    dates = _get_test_dates()
    df = pd.DataFrame(index=dates)
    df["Close"] = [100.0] * 49 + [102.0]
    df["High"] = [102.0] * 50
    df["Low"] = [98.0] * 50
    df["Open"] = [100.0] * 50
    # Normal volume 1,000,000, last day 2,000,000 (2.0x 20D average)
    df["Volume"] = [1_000_000] * 49 + [2_000_000]

    # Pre-trigger price = 100.0 (below breakout trigger)
    tactical_pre, _ = compute_tactical_setup("TEST", market="US", current_price=100.0, price_history_df=df)
    assert tactical_pre.breakout_entry_status == "pre_trigger"
    assert tactical_pre.breakout_entry_eligible is False
    assert tactical_pre.breakout_volume_confirmed is False
    assert tactical_pre.breakout_volume_ratio == 2.0

    # In trigger window with high volume -> eligible
    trigger_p = tactical_pre.breakout_trigger_price
    tactical_trig, _ = compute_tactical_setup("TEST", market="US", current_price=trigger_p, price_history_df=df)
    assert tactical_trig.breakout_entry_status == "eligible"
    assert tactical_trig.breakout_entry_eligible is True
    assert tactical_trig.breakout_volume_confirmed is True


def test_tactical_pullback_strict_interval():
    dates = _get_test_dates()
    df = pd.DataFrame(index=dates)
    df["Close"] = [100.0] * 50
    df["High"] = [105.0] * 50
    df["Low"] = [95.0] * 50
    df["Open"] = [100.0] * 50
    df["Volume"] = [1_000_000] * 50

    tactical_norm, _ = compute_tactical_setup("TEST", market="US", current_price=100.0, price_history_df=df)
    assert tactical_norm.current_rr_ratio is not None
    assert tactical_norm.pullback_entry_status == "in_buy_zone"

    # Price below invalidation stop loss (60.0 < 87.5) -> pullback current R:R is None
    tactical_below, _ = compute_tactical_setup("TEST", market="US", current_price=60.0, price_history_df=df)
    assert tactical_below.current_rr_ratio is None
    assert tactical_below.pullback_entry_status == "below_stop"

    # Price at or above tactical target (105.0) -> pullback current R:R is None
    tactical_above, _ = compute_tactical_setup("TEST", market="US", current_price=105.0, price_history_df=df)
    assert tactical_above.current_rr_ratio is None
    assert tactical_above.pullback_entry_status == "at_or_above_target"
