"""Canonical Macro Contracts and Coverage Specifications.

Defines the source contract, observable coverage, freshness policies,
and exit codes for macro audits and end-to-end acceptance.
Hexagonal rule: Pure data definitions with zero external network side-effects.
"""
from enum import IntEnum
from typing import Any, Dict, List, Set


class AcceptanceExitCode(IntEnum):
    """Standard exit codes for acceptance and audit runners."""
    PASS = 0
    FAIL = 1
    BLOCKED = 2
    ERROR = 3


# Standard UI Market Observable Coverage Groups
MARKET_OBSERVABLE_COVERAGE: Dict[str, List[str]] = {
    "us_yield_curve": [
        "obs_ust_2y_yield",
        "obs_ust_10y_yield",
    ],
    "financial_stress": [
        "obs_ofr_financial_stress",
    ],
    "metals_cot_gold": [
        "obs_cftc_gold_net_managed_money",
    ],
    "global_policy_rates": [
        "obs_us_policy_rate_bis",
        "obs_thai_policy_rate_bis",
        "obs_diff_us_th_policy_rate_bis",
    ],
    "commodity_volatility": [
        "obs_cboe_gold_volatility_gvz",
        "obs_cboe_silver_volatility_vxslv",
        "obs_cboe_oil_volatility_ovx",
    ],
    "auction_demand_note_10y": [
        "obs_treasury_auction_bid_to_cover_10y",
    ],
    "auction_demand_bill_13w": [
        "obs_treasury_auction_bid_to_cover_13w",
    ],
    "us_national_debt": [
        "obs_us_national_debt_trillion",
    ],
    "th_investor_flow": [
        "obs_set_flow_foreign",
        "obs_set_flow_prop",
        "obs_set_flow_institution",
        "obs_set_flow_retail",
    ],
    "th_retail_gold": [
        "obs_gta_gold_bar_sell",
    ],
    "th_market_valuation": [
        "obs_set_valuation_pe",
        "obs_set_valuation_pbv",
        "obs_set_valuation_dividend_yield",
    ],
    "th_market_breadth": [
        "obs_set_advance_decline_ratio",
    ],
    "crypto_liquidity": [
        "obs_crypto_stablecoin_supply_usd_b",
        "obs_crypto_stablecoin_supply_growth_30d",
        "obs_crypto_btc_gold_ratio",
    ],
    "thai_hard_data": [
        "obs_th_gdp_nesdc",
        "obs_th_cpi_moc",
        "obs_th_core_cpi_moc",
        "obs_th_mpi_oie",
    ],
    "thai_debt_bonds": [
        "obs_th_gov_yield_10y",
        "obs_th_gov_yield_2y",
        "obs_th_gov_10y_2y_spread",
    ],
}


# Source Families & SLA Freshness Requirements (days)
SOURCE_FAMILIES: Dict[str, Dict[str, Any]] = {
    "yahoo_market_bars": {
        "provider": "Yahoo",
        "frequency": "daily",
        "max_freshness_days": 4,  # accommodates 3-day holiday weekends
        "authority": "Direct Yahoo Finance Bar History",
        "is_required": True,
    },
    "fred_us_macro": {
        "provider": "FRED",
        "frequency": "mixed",
        "max_freshness_days": 45,  # monthly series with reporting lag
        "authority": "Federal Reserve Bank of St. Louis",
        "is_required": True,
    },
    "terminal_v2_treasury": {
        "provider": "Terminal V2 (US Treasury)",
        "frequency": "daily",
        "max_freshness_days": 5,
        "authority": "U.S. Department of the Treasury Fiscal Data",
        "is_required": True,
    },
    "terminal_v2_bis": {
        "provider": "Terminal V2 (BIS)",
        "frequency": "monthly",
        "max_freshness_days": 45,
        "authority": "Bank for International Settlements Central Bank Policy Rates",
        "is_required": True,
    },
    "terminal_v2_ofr": {
        "provider": "Terminal V2 (OFR)",
        "frequency": "daily",
        "max_freshness_days": 7,
        "authority": "Office of Financial Research Financial Stress Index",
        "is_required": True,
    },
    "terminal_v2_cboe": {
        "provider": "Terminal V2 (Cboe)",
        "frequency": "daily",
        "max_freshness_days": 5,
        "authority": "Chicago Board Options Exchange",
        "is_required": True,
    },
    "terminal_v2_cftc": {
        "provider": "Terminal V2 (CFTC)",
        "frequency": "weekly",
        "max_freshness_days": 10,  # released every Friday for prior Tuesday
        "authority": "CFTC Commitments of Traders Socrata Dataset",
        "is_required": True,
    },
    "terminal_v2_settrade": {
        "provider": "Terminal V2 (Settrade)",
        "frequency": "daily",
        "max_freshness_days": 4,
        "authority": "Stock Exchange of Thailand Settrade Public Feeds",
        "is_required": True,
    },
    "terminal_v2_gta": {
        "provider": "Terminal V2 (GTA)",
        "frequency": "daily",
        "max_freshness_days": 4,
        "authority": "Gold Traders Association Thailand",
        "is_required": True,
    },
    "terminal_v2_thaibma": {
        "provider": "ThaiBMA",
        "frequency": "daily",
        "max_freshness_days": 5,
        "authority": "Thai Bond Market Association",
        "is_required": True,
    },
    "thai_official_hard_data": {
        "provider": "NESDC / MOC / OIE",
        "frequency": "monthly_quarterly",
        "max_freshness_days": 120,  # quarterly GDP lag
        "authority": "Official Thai Economic Agencies with primary PDF evidence",
        "is_required": True,
    },
    "terminal_v2_crypto_liquidity": {
        "provider": "DeFiLlama / Benchmark / SoSoValue",
        "frequency": "daily",
        "max_freshness_days": 3,
        "authority": "DeFiLlama Stablecoins & Public Benchmark Feeds",
        "is_required": True,
    },
}


# Known Degraded / Discontinued Sources Policy (W04 / MC-11 / MC-13)
# These are legacy series that are discontinued upstream or have quarterly/semi-annual releases.
# The contract explicitly documents their degraded status and fallback policy so that
# the evaluation engine does not treat them as unexpected silent gaps.
KNOWN_DEGRADED_SERIES_POLICY: Dict[str, Dict[str, Any]] = {
    "DTWEXBGS": {
        "reason": "Nominal Broad U.S. Dollar Index (FRED discontinued in favor of RTWEXBGS)",
        "policy": "allow_stale_or_fallback",
        "replacement_series": "DX-Y.NYB",
    },
    "INTDSRCNM193N": {
        "reason": "China Discount Rate discontinued on FRED",
        "policy": "mark_unavailable_exclude_from_scoring",
        "replacement_series": None,
    },
    "NGDPRXDCCNA": {
        "reason": "China Real GDP Annual series with long reporting lag",
        "policy": "mark_unavailable_exclude_from_scoring",
        "replacement_series": None,
    },
    "CHNCPIALLMINMEI": {
        "reason": "China CPI discontinued on FRED OECD dataset",
        "policy": "mark_unavailable_exclude_from_scoring",
        "replacement_series": None,
    },
    "INTDSRJPM193N": {
        "reason": "Japan Discount Rate discontinued by Bank of Japan in 2017",
        "policy": "mark_unavailable_exclude_from_scoring",
        "replacement_series": None,
    },
    "JPNCPIALLMINMEI": {
        "reason": "Japan CPI discontinued on FRED OECD dataset",
        "policy": "mark_unavailable_exclude_from_scoring",
        "replacement_series": None,
    },
    "INTDSRINM193N": {
        "reason": "India Bank Rate discontinued on FRED",
        "policy": "mark_unavailable_exclude_from_scoring",
        "replacement_series": None,
    },
    "INDCPIALLMINMEI": {
        "reason": "India CPI discontinued on FRED OECD dataset",
        "policy": "mark_unavailable_exclude_from_scoring",
        "replacement_series": None,
    },
    "BRACPIALLMINMEI": {
        "reason": "Brazil CPI discontinued on FRED OECD dataset",
        "policy": "mark_unavailable_exclude_from_scoring",
        "replacement_series": None,
    },
    "obs_th_public_debt_mof": {
        "reason": "MOF Public Debt is published bi-annually / quarterly; snapshot period may be older than 90 days",
        "policy": "allow_documented_quarterly_period",
        "replacement_series": "obs_th_public_debt_gdp_pct",
    },
    "obs_th_debt_to_gdp_mof": {
        "reason": "MOF Debt to GDP is published bi-annually / quarterly; snapshot period may be older than 90 days",
        "policy": "allow_documented_quarterly_period",
        "replacement_series": "obs_th_debt_to_gdp_pct",
    },
}
