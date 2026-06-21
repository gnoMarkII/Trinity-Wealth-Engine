import pytest

@pytest.fixture
def mock_yf_ticker(monkeypatch):
    """Mock yf.Ticker — track ticker arg ที่ส่งเข้าไป + คืน fake info"""
    captured = {}

    fake_info = {
        "quoteType": "EQUITY",
        "shortName": "Mock Company",
        "sector": "Energy",
        "industry": "Oil & Gas",
        "marketCap": 1_500_000_000_000,
        "trailingPE": 12.5,
        "currentPrice": 40.0,
        "operatingCashflow": 100_000_000_000,
        "freeCashflow": 80_000_000_000,
        "totalCash": 200_000_000_000,
        "totalDebt": 150_000_000_000,
        "targetMeanPrice": 45.0,
        "targetLowPrice": 38.0,
        "targetHighPrice": 50.0,
        "fiftyDayAverage": 39.0,
        "twoHundredDayAverage": 37.0,
        "fiftyTwoWeekHigh": 45.0,
        "fiftyTwoWeekLow": 30.0,
    }

    class FakeTicker:
        def __init__(self, ticker):
            captured["ticker"] = ticker
            self.info = fake_info
            self.news = [{"content": {"title": "Test news", "provider": {"displayName": "Test"},
                                       "canonicalUrl": {"url": "http://test"}}}]

        @property
        def sustainability(self):
            return None

        @property
        def financials(self):
            import pandas as pd
            from datetime import datetime
            return pd.DataFrame(
                {datetime(2024, 12, 31): [100e9, 30e9]},
                index=["Total Revenue", "Net Income"],
            )

        @property
        def insider_transactions(self):
            return None

    monkeypatch.setattr("yfinance.Ticker", FakeTicker)
    import core.retry
    monkeypatch.setattr(core.retry, "with_retry", lambda fn, *args, **kwargs: fn())
    return captured
