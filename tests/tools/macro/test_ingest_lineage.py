from unittest.mock import Mock

import pandas as pd
import pytest

from tools.macro import ingest
from tools.macro.evaluation import _extract_market_observables
from tools.macro.evaluation import _apply_validity, _infer_provider, _infer_unit
from schemas.macro_schemas import MarketObservable


@pytest.mark.parametrize('snapshot_kind', ['global', 'regional', 'country'])
def test_market_snapshot_preserves_bar_date_and_previous_price(monkeypatch, snapshot_kind):
    history = pd.DataFrame({'Close': [100.0, 105.0]}, index=pd.to_datetime(['2026-10-01', '2026-10-02']))
    ticker = Mock()
    ticker.history.return_value = history
    monkeypatch.setattr(ingest.yf, 'Ticker', lambda symbol: ticker)
    monkeypatch.delenv('FRED_API_KEY', raising=False)
    monkeypatch.setenv('ALLOW_MOCK_MACRO_INGEST', 'false')
    if snapshot_kind == 'global':
        monkeypatch.setattr(ingest, '_MACRO_TICKERS', {'^VIX': ('VIX Index', '30-day risk sentiment')})
        monkeypatch.setattr(ingest, '_GLOBAL_GROUPS', [('Risk', ['^VIX'])])
        markdown = ingest.ingest_global_macro.invoke({})
        source_key = 'Global_Macro_Snapshot'
    elif snapshot_kind == 'regional':
        monkeypatch.setattr(ingest, '_REGIONAL_TICKERS', {'EWJ': ('Japan ETF', '12-month proxy')})
        monkeypatch.setattr(ingest, '_REGIONAL_GROUPS_MAP', {'Japan': {'Growth': ['EWJ']}})
        markdown = ingest.ingest_regional_macro.invoke({})
        source_key = 'Regional_Macro_Snapshot'
    else:
        monkeypatch.setattr(ingest, '_THAI_INDICATORS', {'THB=X': ('USD/THB', 'FX')})
        monkeypatch.setattr(ingest, '_THAI_GROUPS', [('FX', ['THB=X'])])
        markdown = ingest.ingest_country_macro.invoke({})
        source_key = 'Country_Macro_Snapshot'

    observables = _extract_market_observables({source_key: markdown}, {}, '2026-10-04')
    assert len(observables) == 1
    observable = observables[0]
    assert observable.observed_at == '2026-10-02'
    assert observable.is_valid
    assert observable.metadata['prev'] == 100.0
    # Do not read percent change or digits in description as the moving average.
    if snapshot_kind != 'country':
        assert 'ma' not in observable.metadata


def test_empty_market_history_never_fabricates_observation_date(monkeypatch):
    ticker = Mock()
    ticker.history.return_value = pd.DataFrame()
    monkeypatch.setattr(ingest.yf, 'Ticker', lambda symbol: ticker)
    assert ingest._fetch_price_once('^VIX') == (None, None, None)


@pytest.mark.parametrize('symbol,unit', [('HYG', 'USD'), ('LQD', 'USD'), ('DX-Y.NYB', 'pts'), ('NG=F', 'USD/MMBtu')])
def test_market_provider_and_units_remain_correct(symbol, unit):
    indicator = f'Market (`{symbol}`)'
    assert _infer_provider(indicator, 'Global_Macro_Snapshot') == 'Yahoo'
    assert _infer_unit('100', indicator) == unit


def test_quarterly_fred_period_start_is_not_mistaken_for_release_age():
    observable = MarketObservable(observable_id='gdp', asset_bucket='equities', region='United States',
        indicator='Real GDP (`GDPC1`)', value='2.3', unit='% YoY', observed_at='2026-04-01', source_file='FRED')
    result = _apply_validity(observable, '2026-10-04')
    assert result.is_valid
    assert result.observed_at == '2026-04-01'
    assert result.metadata['freshness_basis'] == 'quarter_end'
