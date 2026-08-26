"""Unit tests for OhlcvService and Hexagonal Architecture isolation."""
import pytest
from unittest.mock import MagicMock
import pandas as pd
from tools.market.ohlcv.ports.ohlcv_port import OhlcvProviderPort, CorporateActionProviderPort
from tools.market.ohlcv.service import OhlcvService, _calculate_indicator_burn_in, _calculate_52w
from api.schemas import OHLCVCandleDTO


def test_ohlcv_service_burn_in_full_convergence():
    candles = [
        OHLCVCandleDTO(timestamp=1000 * i, open=100.0, high=105.0, low=95.0, close=102.0, volume=1000)
        for i in range(250)
    ]
    status, remaining, f_ts, f_idx, policy = _calculate_indicator_burn_in(
        indicator_key="EMA200",
        required_bars=200,
        actual_warmup_bars=200,
        candles=candles,
        display_start_ts=200000,
    )
    assert status == "full"
    assert remaining == 0
    assert policy.required_burn_in_bars == 200


def test_ohlcv_service_burn_in_partial():
    candles = [
        OHLCVCandleDTO(timestamp=1000 * i, open=100.0, high=105.0, low=95.0, close=102.0, volume=1000)
        for i in range(220)
    ]
    status, remaining, f_ts, f_idx, policy = _calculate_indicator_burn_in(
        indicator_key="EMA200",
        required_bars=200,
        actual_warmup_bars=150,
        candles=candles,
        display_start_ts=150000,
    )
    assert status == "partial"
    assert remaining == 50


def test_ohlcv_service_instantiation_with_mocks():
    mock_ohlcv = MagicMock(spec=OhlcvProviderPort)
    mock_actions = MagicMock(spec=CorporateActionProviderPort)

    service = OhlcvService(ohlcv_provider=mock_ohlcv, action_provider=mock_actions)
    assert service is not None
