"""Bootstrap Composition Root for Terminal V2 (Unified Canonical Architecture).

Assembles driven adapters, application services, and injects dependencies
into driving ports:
- MarketTerminalServicePort (Real-time Quote & Dynamic Routing)
- TerminalDataServicePort (Unified Institutional & Market Data Engine)
"""
from typing import Optional

from tools.market.terminal_v2.adapters.bis_adapter import BisPolicyRatesHttpAdapter
from tools.market.terminal_v2.adapters.cboe_commodity_vol_adapter import CboeCommodityVolAdapter
from tools.market.terminal_v2.adapters.cboe_options_adapter import CboeOptionsAdapter
from tools.market.terminal_v2.adapters.cftc_cot_adapter import CftcCotHttpAdapter
from tools.market.terminal_v2.adapters.crypto_benchmark_adapter import CryptoBenchmarkAdapter
from tools.market.terminal_v2.adapters.defillama_stablecoin_adapter import DefiLlamaStablecoinsAdapter
from tools.market.terminal_v2.adapters.etf_flows_adapter import SoSoValueEtfFlowsAdapter
from tools.market.terminal_v2.adapters.finra_adapter import FinraAdapter
from tools.market.terminal_v2.adapters.fred_csv_adapter import FredCsvAdapter
from tools.market.terminal_v2.adapters.goldtraders_adapter import GoldTradersAdapter
from tools.market.terminal_v2.adapters.hyperliquid_adapter import HyperliquidAdapter
from tools.market.terminal_v2.adapters.legacy_fred_fallback_adapter import LegacyFredFallbackAdapter
from tools.market.terminal_v2.adapters.mof_th_adapter import MofThailandAdapter
from tools.market.terminal_v2.adapters.nasdaq_adapter import NasdaqHttpAdapter
from tools.market.terminal_v2.adapters.nyfed_adapter import NyFedAdapter
from tools.market.terminal_v2.adapters.ofr_adapter import OfrHttpAdapter
from tools.market.terminal_v2.adapters.polymarket_adapter import PolymarketAdapter
from tools.market.terminal_v2.adapters.rss_news_adapter import RssNewsDiscoveryAdapter
from tools.market.terminal_v2.adapters.sec_edgar_adapter import SecEdgarAdapter
from tools.market.terminal_v2.adapters.sec_th_adapter import SecThailandAdapter
from tools.market.terminal_v2.adapters.settrade_adapter import SettradeAdapter
from tools.market.terminal_v2.adapters.thaibma_adapter import ThaiBmaPublicAdapter
from tools.market.terminal_v2.adapters.treasury_adapter import TreasuryAdapter
from tools.market.terminal_v2.application.cache import ThreadSafeTTLCache
from tools.market.terminal_v2.application.routing_service import DynamicRoutingService
from tools.market.terminal_v2.application.terminal_data_service import TerminalDataService
from tools.market.terminal_v2.ports.driving_ports import (
    MarketTerminalServicePort,
    TerminalDataServicePort,
)

_terminal_service_instance: Optional[MarketTerminalServicePort] = None
_terminal_data_service_instance: Optional[TerminalDataServicePort] = None


# ============================================================================
# 1. Real-Time Market Routing Composition Root
# ============================================================================

def create_terminal_service(
    shared_cache: Optional[ThreadSafeTTLCache] = None,
) -> MarketTerminalServicePort:
    """Factory creating a new fully-wired instance of MarketTerminalServicePort."""
    cache = shared_cache or ThreadSafeTTLCache()

    settrade_adapter = SettradeAdapter(cache=cache)
    goldtraders_adapter = GoldTradersAdapter(cache=cache)
    fred_adapter = FredCsvAdapter(cache=cache)
    hyperliquid_adapter = HyperliquidAdapter(cache=cache)
    legacy_fred_adapter = LegacyFredFallbackAdapter()

    return DynamicRoutingService(
        thai_market=settrade_adapter,
        gold_price=goldtraders_adapter,
        macro_series=fred_adapter,
        perps_quote=hyperliquid_adapter,
        legacy_macro_fallback=legacy_fred_adapter,
    )


def get_terminal_service() -> MarketTerminalServicePort:
    """Obtain or initialize the singleton MarketTerminalServicePort instance."""
    global _terminal_service_instance
    if _terminal_service_instance is None:
        _terminal_service_instance = create_terminal_service()
    return _terminal_service_instance


# ============================================================================
# 2. Unified Terminal Data Service Composition Root
# ============================================================================

def create_terminal_data_service(
    shared_cache: Optional[ThreadSafeTTLCache] = None,
) -> TerminalDataServicePort:
    """Factory creating a new fully-wired instance of TerminalDataServicePort."""
    cache = shared_cache or ThreadSafeTTLCache()

    # Adapters instantiation
    finra_adapter = FinraAdapter(cache=cache)
    cboe_options_adapter = CboeOptionsAdapter(cache=cache)
    cboe_vol_adapter = CboeCommodityVolAdapter(cache=cache)
    nyfed_adapter = NyFedAdapter(cache=cache)
    treasury_adapter = TreasuryAdapter(cache=cache)
    sec_th_adapter = SecThailandAdapter(cache=cache)
    mof_adapter = MofThailandAdapter(cache=cache)
    polymarket_adapter = PolymarketAdapter(cache=cache)
    soso_adapter = SoSoValueEtfFlowsAdapter(cache=cache)
    ofr_adapter = OfrHttpAdapter(cache=cache)
    cot_adapter = CftcCotHttpAdapter(cache=cache)
    bis_adapter = BisPolicyRatesHttpAdapter(cache=cache)
    nasdaq_adapter = NasdaqHttpAdapter(cache=cache)
    sec_edgar_adapter = SecEdgarAdapter(cache=cache)
    rss_news_adapter = RssNewsDiscoveryAdapter(cache=cache)
    thaibma_adapter = ThaiBmaPublicAdapter(cache=cache)
    stablecoins_adapter = DefiLlamaStablecoinsAdapter(cache=cache)
    crypto_benchmark_adapter = CryptoBenchmarkAdapter(cache=cache)

    return TerminalDataService(
        short_volume=finra_adapter,
        nasdaq_intelligence=nasdaq_adapter,
        sec_financials=sec_edgar_adapter,
        sec_insider_trades=sec_edgar_adapter,
        ticker_news=rss_news_adapter,
        reference_rates=nyfed_adapter,
        ofr_stress=ofr_adapter,
        global_policy_rates=bis_adapter,
        treasury_data=treasury_adapter,
        auction_history=treasury_adapter,
        thai_bond_market=sec_th_adapter,
        thai_public_debt=mof_adapter,
        thai_yield_curve=thaibma_adapter,
        options_chain=cboe_options_adapter,
        commodity_vol=cboe_vol_adapter,
        metals_cot=cot_adapter,
        prediction_market=polymarket_adapter,
        thai_fund_allocation=sec_th_adapter,
        spot_etf_flows=soso_adapter,
        stablecoin_supply=stablecoins_adapter,
        crypto_benchmark=crypto_benchmark_adapter,
    )


def get_terminal_data_service() -> TerminalDataServicePort:
    """Obtain or initialize the singleton TerminalDataServicePort instance."""
    global _terminal_data_service_instance
    if _terminal_data_service_instance is None:
        _terminal_data_service_instance = create_terminal_data_service()
    return _terminal_data_service_instance


