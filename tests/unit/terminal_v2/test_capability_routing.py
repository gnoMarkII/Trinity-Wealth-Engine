"""Unit tests verifying Capability & Symbol-Market Routing and Perps Isolation."""
from unittest.mock import MagicMock
import pytest

from tools.market.terminal_v2.application.routing_service import DynamicRoutingService
from tools.market.terminal_v2.domain.errors import (
    InvalidCapabilityError,
    SymbolMarketMismatchError,
)
from tools.market.terminal_v2.domain.models import (
    LivePerpsQuote,
    MacroSeries,
    ThaiFundFlowSnapshot,
    ThaiRetailGoldQuote,
)


@pytest.fixture
def mock_service():
    thai_mock = MagicMock()
    gold_mock = MagicMock()
    macro_mock = MagicMock()
    perps_mock = MagicMock()
    legacy_macro_mock = MagicMock()

    service = DynamicRoutingService(
        thai_market=thai_mock,
        gold_price=gold_mock,
        macro_series=macro_mock,
        perps_quote=perps_mock,
        legacy_macro_fallback=legacy_macro_mock,
    )
    return service, thai_mock, gold_mock, macro_mock, perps_mock, legacy_macro_mock


def test_routing_by_capability_direct(mock_service):
    service, thai_mock, gold_mock, macro_mock, perps_mock, _ = mock_service

    # Thai Flow
    thai_mock.get_investor_type_flow.return_value = MagicMock(spec=ThaiFundFlowSnapshot)
    res = service.query_by_capability("investor_type_flow", market="SET")
    thai_mock.get_investor_type_flow.assert_called_once_with("SET")

    # Thai Gold
    gold_mock.get_retail_gold_quote.return_value = MagicMock(spec=ThaiRetailGoldQuote)
    res = service.query_by_capability("retail_gold_price")
    gold_mock.get_retail_gold_quote.assert_called_once()

    # FRED Macro
    macro_mock.get_macro_series.return_value = MagicMock(spec=MacroSeries)
    res = service.query_by_capability("macro_series", symbol="CPIAUCSL")
    macro_mock.get_macro_series.assert_called_once_with("CPIAUCSL")


def test_perps_cannot_be_routed_as_cash_equity(mock_service):
    service, _, _, _, _, _ = mock_service

    # Requesting cash equity quote for HIP-3 perp symbol must fail
    with pytest.raises(SymbolMarketMismatchError) as exc_info:
        service.query_by_capability("equity_cash_quote", symbol="xyz:TSLA")
    assert "synthetic perpetual" in str(exc_info.value).lower()


def test_bare_stock_symbol_rejected_for_perps_without_dex_namespace(mock_service):
    service, _, _, _, _, _ = mock_service

    # Requesting bare 'TSLA' for perps must be rejected with helpful error
    with pytest.raises(SymbolMarketMismatchError) as exc_info:
        service.query_by_capability("perps_quote", symbol="TSLA")
    assert "bare ticker" in str(exc_info.value).lower()
    assert "xyz:" in str(exc_info.value)


def test_namespaced_hip3_perp_and_crypto_allowed_for_perps(mock_service):
    service, _, _, _, perps_mock, _ = mock_service
    perps_mock.get_perps_quote.return_value = LivePerpsQuote(
        symbol="xyz:TSLA",
        mark_price=250.0,
        dex_namespace="xyz",
        asset_class="synthetic_crypto_perp",
        contract_type="perpetual_future",
        source="Hyperliquid",
    )

    # xyz:TSLA is allowed
    quote = service.query_by_capability("perps_quote", symbol="xyz:TSLA")
    assert quote.mark_price == 250.0
    assert quote.dex_namespace == "xyz"

    # BTC is known crypto, allowed without prefix
    perps_mock.get_perps_quote.return_value = LivePerpsQuote(
        symbol="BTC",
        mark_price=64000.0,
        dex_namespace="",
        asset_class="synthetic_crypto_perp",
        contract_type="perpetual_future",
        source="Hyperliquid",
    )
    quote_btc = service.query_by_capability("perps_quote", symbol="BTC")
    assert quote_btc.mark_price == 64000.0


def test_unknown_capability_raises_error(mock_service):
    service, _, _, _, _, _ = mock_service
    with pytest.raises(InvalidCapabilityError):
        service.query_by_capability("unknown_cap_xyz")
