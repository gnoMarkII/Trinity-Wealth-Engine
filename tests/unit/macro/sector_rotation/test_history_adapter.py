from __future__ import annotations

from datetime import date
from types import SimpleNamespace
import time

import pandas as pd
import pytest

from tools.macro.adapters import sector_history_adapter
from tools.macro.adapters.sector_history_adapter import SectorHistoryAdapter
from tools.market.ohlcv.adapters.yfinance_adapter import YFinanceOhlcvAdapter


class _SlowProvider:
    def fetch_history(self, _symbol, _period, _interval, _auto_adjust):
        time.sleep(0.4)
        return pd.DataFrame()


def test_history_batch_returns_at_deadline_without_waiting_for_slow_provider(monkeypatch):
    monkeypatch.setattr(sector_history_adapter, "get_last_completed_regular_session", lambda: date(2026, 10, 2))
    monkeypatch.setattr(sector_history_adapter, "is_us_trading_day", lambda day: day.weekday() < 5)
    adapter = SectorHistoryAdapter(_SlowProvider(), batch_timeout_seconds=0.05)

    started = time.monotonic()
    batch = adapter.fetch()
    elapsed = time.monotonic() - started

    assert elapsed < 0.25
    assert len(batch.prices) == 12
    assert set(batch.reasons.values()) == {"provider_timeout"}


def test_sector_yfinance_provider_passes_timeout_and_propagates_provider_error():
    calls = []

    class _Ticker:
        def history(self, **kwargs):
            calls.append(kwargs)
            raise TimeoutError("request timed out")

    provider = YFinanceOhlcvAdapter(
        SimpleNamespace(Ticker=lambda _symbol: _Ticker()),
        request_timeout=12,
        retry_attempts=1,
        raise_errors=True,
    )

    with pytest.raises(TimeoutError):
        provider.fetch_history("SPY", "5y", "1d", True)

    assert calls == [{
        "period": "5y", "interval": "1d", "auto_adjust": True,
        "timeout": 12.0, "raise_errors": True,
    }]
