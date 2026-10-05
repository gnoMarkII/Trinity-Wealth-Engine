"""Terminal V2 Macro Observables Adapter (Dual-Track Macro Intelligence - Phase 3).

Maps Terminal V2 domain models (Settrade flow, market breadth, venue valuation multiples,
retail physical gold, US Treasury yield curve, and BIS policy rates) into typed MarketObservables.
Integrates via Hexagonal Architecture driving ports:
- MarketTerminalServicePort (Real-time Market Routing)
- TerminalDataServicePort (Institutional Macro Data)
"""
from datetime import datetime
from dataclasses import asdict, is_dataclass
import re
from typing import Any, Optional
from core.logger import get_logger
from schemas.macro_schemas import MarketObservable
from tools.market.terminal_v2.ports.driving_ports import (
    MarketTerminalServicePort,
    TerminalDataServicePort,
)

log = get_logger(__name__)


def normalize_observed_date(date_raw: Any, default_date: str) -> str:
    """Normalize arbitrary provider dates (Thai BE, DD/MM/YYYY, ISO timestamps, YYYY-MM) to YYYY-MM-DD."""
    if not date_raw:
        return default_date
    val = str(date_raw).strip()

    # 1. Matches YYYY-MM-DD already
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", val):
        return val

    # 2. ISO timestamp: YYYY-MM-DDTHH:MM:SS or YYYY-MM-DD HH:MM:SS
    m_iso = re.match(r"^(\d{4}-\d{2}-\d{2})", val)
    if m_iso:
        return m_iso.group(1)

    # 3. YYYY-MM (e.g. BIS monthly) -> YYYY-MM-01
    m_ym = re.fullmatch(r"(\d{4})-(\d{2})", val)
    if m_ym:
        return f"{m_ym.group(1)}-{m_ym.group(2)}-01"

    # 4. DD/MM/YYYY or DD/MM/YYYY HH:MM:SS (including Thai Buddhist Era e.g. 2569)
    m_dmy = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})", val)
    if m_dmy:
        d, m, y = int(m_dmy.group(1)), int(m_dmy.group(2)), int(m_dmy.group(3))
        if y > 2400:
            y -= 543
        return f"{y:04d}-{m:02d}-{d:02d}"

    # 5. YYYY/MM/DD
    m_ymd = re.match(r"^(\d{4})/(\d{1,2})/(\d{1,2})", val)
    if m_ymd:
        y, m, d = int(m_ymd.group(1)), int(m_ymd.group(2)), int(m_ymd.group(3))
        if y > 2400:
            y -= 543
        return f"{y:04d}-{m:02d}-{d:02d}"

    return default_date


def normalize_policy_rate_country(country_str: str) -> str:
    """Normalize country names/codes from BIS or fixtures to standard codes (US, TH, XM, etc.)."""
    c = (country_str or "").strip().lower()
    if c in ("us", "usa", "united states", "united states of america"):
        return "US"
    if c in ("th", "tha", "thailand", "thai"):
        return "TH"
    if c in ("xm", "euro area", "ea", "eurozone", "europe"):
        return "XM"
    if c in ("jp", "japan"):
        return "JP"
    if c in ("cn", "china"):
        return "CN"
    if c in ("in", "india"):
        return "IN"
    if c in ("gb", "united kingdom", "uk"):
        return "GB"
    return c.upper()


def build_thai_market_observables(
    terminal_service: Optional[MarketTerminalServicePort] = None,
    as_of_date: Optional[str] = None,
) -> list[MarketObservable]:
    """Build typed MarketObservables for the Thailand market radar from Terminal V2."""
    if terminal_service is None:
        try:
            from tools.market.terminal_v2.bootstrap import get_terminal_service
            terminal_service = get_terminal_service()
        except Exception as e:
            log.warning("Could not obtain MarketTerminalServicePort instance: %s", e)
            return []

    today_str = as_of_date or datetime.now().strftime("%Y-%m-%d")
    observables: list[MarketObservable] = []

    # 1. SET Investor Flow (Foreign, Institution, Proprietary, Retail)
    try:
        flow = terminal_service.get_investor_flow(market="SET")
        obs_date = normalize_observed_date(getattr(flow, "as_of", today_str), today_str)
        is_valid = not getattr(flow, "is_stale", False)

        flow_slug_map = {
            "foreign": ("obs_set_flow_foreign", "SET Foreign Investor Net Flow"),
            "institution": ("obs_set_flow_institution", "SET Local Institution Net Flow"),
            "proprietary": ("obs_set_flow_prop", "SET Proprietary Trading Net Flow"),
            "retail": ("obs_set_flow_retail", "SET Retail Investor Net Flow"),
        }

        for row in getattr(flow, "investors", []):
            name_en_lower = getattr(row, "name_en", "").lower()
            name_th = getattr(row, "investor_type", "")
            net_val = float(getattr(row, "net_value", 0.0))
            net_mb = round(net_val / 1_000_000.0, 2)

            key = None
            if "foreign" in name_en_lower or "ต่างชาติ" in name_th:
                key = "foreign"
            elif "institution" in name_en_lower or "สถาบัน" in name_th:
                key = "institution"
            elif "prop" in name_en_lower or "บล." in name_th:
                key = "proprietary"
            elif "retail" in name_en_lower or "individual" in name_en_lower or "ในประเทศ" in name_th or "รายย่อย" in name_th or "individual" in name_th.lower():
                key = "retail"

            if key and key in flow_slug_map:
                obs_id, indicator_title = flow_slug_map[key]
                observables.append(MarketObservable(
                    observable_id=obs_id,
                    asset_bucket="equities",
                    region="Thailand",
                    indicator=indicator_title,
                    value=f"{net_mb:.2f}",
                    unit="THB Mil",
                    observed_at=obs_date,
                    source_file="Terminal_V2_Settrade",
                    provider="Terminal V2 (Settrade)",
                    confidence="high" if is_valid else "low",
                    is_valid=is_valid,
                    status="verified" if is_valid else "stale",
                    metadata={
                        "net_thb": net_val,
                        "buy_thb": getattr(row, "buy_value", 0.0),
                        "sell_thb": getattr(row, "sell_value", 0.0),
                        "investor_name_en": getattr(row, "name_en", ""),
                    },
                ))
    except Exception as e:
        log.warning("Failed to collect SET investor flow from Terminal V2: %s", e)

    # 2. SET Market Breadth (Advances, Declines, Unchanged)
    try:
        breadth = terminal_service.get_market_breadth(market="SET")
        obs_date = normalize_observed_date(getattr(breadth, "as_of", today_str), today_str)
        is_valid = not getattr(breadth, "is_stale", False)
        gainers = int(getattr(breadth, "gainers", 0))
        losers = int(getattr(breadth, "losers", 0))
        unchanged = int(getattr(breadth, "unchanged", 0))

        ad_ratio = round(gainers / losers, 2) if losers > 0 else 1.0

        observables.append(MarketObservable(
            observable_id="obs_set_advance_decline_ratio",
            asset_bucket="equities",
            region="Thailand",
            indicator="SET Advance/Decline Ratio",
            value=f"{ad_ratio:.2f}",
            unit="ratio",
            observed_at=obs_date,
            source_file="Terminal_V2_Settrade",
            provider="Terminal V2 (Settrade)",
            confidence="high" if is_valid else "low",
            is_valid=is_valid,
            status="verified" if is_valid else "stale",
            metadata={"gainers": gainers, "losers": losers, "unchanged": unchanged},
        ))
    except Exception as e:
        log.warning("Failed to collect SET market breadth from Terminal V2: %s", e)

    # 3. SET Market Valuation (P/E, P/BV, Dividend Yield)
    try:
        valuation = terminal_service.get_market_valuation(market="SET")
        obs_date = normalize_observed_date(getattr(valuation, "as_of", today_str), today_str)
        is_valid = not getattr(valuation, "is_stale", False)

        pe = getattr(valuation, "pe_ratio", None)
        if pe is not None:
            observables.append(MarketObservable(
                observable_id="obs_set_valuation_pe",
                asset_bucket="equities",
                region="Thailand",
                indicator="SET Index P/E Ratio",
                value=f"{float(pe):.2f}",
                unit="ratio",
                observed_at=obs_date,
                source_file="Terminal_V2_Settrade",
                provider="Terminal V2 (Settrade)",
                confidence="high" if is_valid else "low",
                is_valid=is_valid,
                status="verified" if is_valid else "stale",
            ))

        pbv = getattr(valuation, "pbv_ratio", None)
        if pbv is not None:
            observables.append(MarketObservable(
                observable_id="obs_set_valuation_pbv",
                asset_bucket="equities",
                region="Thailand",
                indicator="SET Index P/BV Ratio",
                value=f"{float(pbv):.2f}",
                unit="ratio",
                observed_at=obs_date,
                source_file="Terminal_V2_Settrade",
                provider="Terminal V2 (Settrade)",
                confidence="high" if is_valid else "low",
                is_valid=is_valid,
                status="verified" if is_valid else "stale",
            ))

        div_yield = getattr(valuation, "dividend_yield", None)
        if div_yield is not None:
            observables.append(MarketObservable(
                observable_id="obs_set_valuation_dividend_yield",
                asset_bucket="equities",
                region="Thailand",
                indicator="SET Index Dividend Yield",
                value=f"{float(div_yield):.2f}",
                unit="%",
                observed_at=obs_date,
                source_file="Terminal_V2_Settrade",
                provider="Terminal V2 (Settrade)",
                confidence="high" if is_valid else "low",
                is_valid=is_valid,
                status="verified" if is_valid else "stale",
            ))
    except Exception as e:
        log.warning("Failed to collect SET valuation from Terminal V2: %s", e)

    # 4. Thai Retail Physical Gold (Gold Traders Association)
    try:
        gold = terminal_service.get_retail_gold()
        obs_date = normalize_observed_date(getattr(gold, "announced_at", today_str), today_str)
        is_valid = not getattr(gold, "is_stale", False)

        bar_sell = float(gold.bar.sell) if hasattr(gold, "bar") else 0.0
        bar_buy = float(gold.bar.buy) if hasattr(gold, "bar") else 0.0

        if bar_sell > 0:
            observables.append(MarketObservable(
                observable_id="obs_gta_gold_bar_sell",
                asset_bucket="commodities",
                region="Thailand",
                indicator="Thai Retail Gold Bar Sell Price (GTA 96.5%)",
                value=f"{bar_sell:.2f}",
                unit="THB/Baht-weight",
                observed_at=obs_date,
                source_file="Terminal_V2_GoldTraders",
                provider="Terminal V2 (GTA)",
                confidence="high" if is_valid else "low",
                is_valid=is_valid,
                status="verified" if is_valid else "stale",
                metadata={"purity": "96.5%", "bar_buy": bar_buy, "bar_sell": bar_sell},
            ))
    except Exception as e:
        log.warning("Failed to collect GTA gold from Terminal V2: %s", e)

    return observables


def build_rates_observables(
    terminal_data_service: Optional[TerminalDataServicePort] = None,
    as_of_date: Optional[str] = None,
) -> list[MarketObservable]:
    """Build typed MarketObservables for sovereign yield curves, BIS policy rates, and stress."""
    if terminal_data_service is None:
        try:
            from tools.market.terminal_v2.bootstrap import get_terminal_data_service
            terminal_data_service = get_terminal_data_service()
        except Exception as e:
            log.warning("Could not obtain TerminalDataServicePort instance: %s", e)
            return []

    today_str = as_of_date or datetime.now().strftime("%Y-%m-%d")
    observables: list[MarketObservable] = []

    # 1. US Treasury Yield Curve & Spreads
    try:
        curve = terminal_data_service.get_treasury_yield_curve()
        obs_date = normalize_observed_date(getattr(curve, "observation_date", today_str), today_str)
        is_valid = not getattr(curve, "is_stale", False)

        tenor_map = {
            "3 Mo": ("obs_ust_3m_yield", "US Treasury 3-Month Yield"),
            "2 Yr": ("obs_ust_2y_yield", "US Treasury 2-Year Yield"),
            "10 Yr": ("obs_ust_10y_yield", "US Treasury 10-Year Yield"),
        }

        for pt in getattr(curve, "yields", []):
            m = getattr(pt, "maturity", "")
            y_val = getattr(pt, "yield_percent", None)
            if m and y_val is not None:
                tenor_key = re.sub(r"[^a-z0-9]+", "_", m.lower()).strip("_")
                obs_id, ind_title = tenor_map.get(m, (f"obs_ust_{tenor_key}_yield", f"US Treasury {m} Yield"))
                observables.append(MarketObservable(
                    observable_id=obs_id,
                    asset_bucket="fixed_income",
                    region="United States",
                    indicator=ind_title,
                    value=f"{float(y_val):.2f}",
                    unit="%",
                    observed_at=obs_date,
                    source_file="Terminal_V2_USTreasury",
                    provider="Terminal V2 (US Treasury)",
                    confidence="high" if is_valid else "low",
                    is_valid=is_valid,
                    status="verified" if is_valid else "stale",
                ))

        spread_10y2y = getattr(curve, "spread_10y_2y_bps", None)
        if spread_10y2y is not None:
            observables.append(MarketObservable(
                observable_id="obs_spread_us_10y_2y_bps",
                asset_bucket="fixed_income",
                region="United States",
                indicator="US Treasury 10Y-2Y Spread",
                value=f"{float(spread_10y2y):.1f}",
                unit="bps",
                observed_at=obs_date,
                source_file="Terminal_V2_USTreasury",
                provider="Terminal V2 (US Treasury)",
                confidence="high" if is_valid else "low",
                is_valid=is_valid,
                status="verified" if is_valid else "stale",
                metadata={"spread_bps": float(spread_10y2y)},
            ))

        spread_10y3m = getattr(curve, "spread_10y_3m_bps", None)
        if spread_10y3m is not None:
            observables.append(MarketObservable(
                observable_id="obs_spread_us_10y_3m_bps",
                asset_bucket="fixed_income",
                region="United States",
                indicator="US Treasury 10Y-3M Spread",
                value=f"{float(spread_10y3m):.1f}",
                unit="bps",
                observed_at=obs_date,
                source_file="Terminal_V2_USTreasury",
                provider="Terminal V2 (US Treasury)",
                confidence="high" if is_valid else "low",
                is_valid=is_valid,
                status="verified" if is_valid else "stale",
                metadata={"spread_bps": float(spread_10y3m)},
            ))
    except Exception as e:
        log.warning("Failed to collect Treasury curve from Terminal V2: %s", e)

    # 2. BIS Global Policy Rates & US-TH Spread
    try:
        policy = terminal_data_service.get_global_policy_rates()
        obs_date = normalize_observed_date(getattr(policy, "as_of_date", today_str), today_str)
        policy_stale = getattr(policy, "is_stale", False)

        us_rate_val: Optional[float] = None
        us_rate_valid: bool = False
        th_rate_val: Optional[float] = None
        th_rate_valid: bool = False

        for item in getattr(policy, "rates", []):
            raw_country = getattr(item, "country", "")
            norm_country = normalize_policy_rate_country(raw_country)
            rate = float(getattr(item, "rate_value", 0.0))
            eff_date = normalize_observed_date(getattr(item, "effective_date", obs_date), obs_date)
            item_stale = getattr(item, "is_stale", False)
            item_valid = (not policy_stale) and (not item_stale)

            if norm_country == "US":
                us_rate_val = rate
                us_rate_valid = item_valid
                observables.append(MarketObservable(
                    observable_id="obs_us_policy_rate_bis",
                    asset_bucket="cash",
                    region="United States",
                    indicator="US Fed Funds Policy Rate (BIS)",
                    value=f"{rate:.3f}" if rate != round(rate, 2) else f"{rate:.2f}",
                    unit="%",
                    observed_at=eff_date,
                    source_file="Terminal_V2_BIS",
                    provider="Terminal V2 (BIS)",
                    confidence="high" if item_valid else "low",
                    is_valid=item_valid,
                    status="verified" if item_valid else "stale",
                    metadata={"raw_country": raw_country, "numeric_value": rate, "previous_rate": item.previous_rate, "rate_type": item.rate_type},
                ))
            elif norm_country == "TH":
                th_rate_val = rate
                th_rate_valid = item_valid
                observables.append(MarketObservable(
                    observable_id="obs_thai_policy_rate_bis",
                    asset_bucket="cash",
                    region="Thailand",
                    indicator="Bank of Thailand Policy Rate (BIS)",
                    value=f"{rate:.2f}",
                    unit="%",
                    observed_at=eff_date,
                    source_file="Terminal_V2_BIS",
                    provider="Terminal V2 (BIS)",
                    confidence="high" if item_valid else "low",
                    is_valid=item_valid,
                    status="verified" if item_valid else "stale",
                    metadata={"raw_country": raw_country, "numeric_value": rate, "previous_rate": item.previous_rate, "rate_type": item.rate_type},
                ))
            else:
                region = {"XM": "Euro Area", "JP": "Japan", "CN": "China", "IN": "India"}.get(norm_country, norm_country)
                observables.append(MarketObservable(
                    observable_id=f"obs_{norm_country.lower()}_policy_rate_bis",
                    asset_bucket="cash", region=region,
                    indicator=f"{item.central_bank} Policy Rate (BIS)",
                    value=f"{rate:.3f}" if rate != round(rate, 2) else f"{rate:.2f}",
                    unit="%", observed_at=eff_date,
                    source_file="Terminal_V2_BIS", provider="Terminal V2 (BIS)",
                    confidence="high" if item_valid else "low", is_valid=item_valid,
                    status="verified" if item_valid else "stale",
                    metadata={"raw_country": raw_country, "numeric_value": rate, "previous_rate": item.previous_rate, "rate_type": item.rate_type},
                ))

        if us_rate_val is not None and th_rate_val is not None and us_rate_valid and th_rate_valid:
            diff_pct = us_rate_val - th_rate_val
            diff_bps = round(diff_pct * 100.0, 1)
            observables.append(MarketObservable(
                observable_id="obs_diff_us_th_policy_rate_bis",
                asset_bucket="cash",
                region="Global",
                indicator="US-Thailand Policy Rate Differential (BIS)",
                value=f"{diff_bps:.1f}",
                unit="bps",
                observed_at=obs_date,
                source_file="Terminal_V2_BIS",
                provider="Terminal V2 (BIS)",
                confidence="high",
                is_valid=True,
                status="verified",
                metadata={
                    "diff_bps": diff_bps,
                    "diff_pct": round(diff_pct, 4),
                    "us_rate": us_rate_val,
                    "th_rate": th_rate_val,
                },
            ))
    except Exception as e:
        log.warning("Failed to collect BIS policy rates from Terminal V2: %s", e)

    # 3. OFR Financial Stress Index
    try:
        stress = terminal_data_service.get_financial_stress()
        obs_date = normalize_observed_date(
            getattr(stress, "as_of_date", None) or getattr(stress, "published_at", today_str),
            today_str,
        )
        is_valid = not getattr(stress, "is_stale", False)
        fsi_val = float(getattr(stress, "fsi_value", 0.0))

        observables.append(MarketObservable(
            observable_id="obs_ofr_financial_stress",
            asset_bucket="risk",
            region="United States",
            indicator="OFR Financial Stress Index",
            value=f"{fsi_val:.2f}",
            unit="pts",
            observed_at=obs_date,
            source_file="Terminal_V2_OFR",
            provider="Terminal V2 (OFR)",
            confidence="high" if is_valid else "low",
            is_valid=is_valid,
            status="verified" if is_valid else "stale",
            metadata={"numeric_value": fsi_val, "published_at": stress.published_at,
                      "categories": [asdict(category) for category in stress.categories if is_dataclass(category)]},
        ))
    except Exception as e:
        log.warning("Failed to collect OFR financial stress from Terminal V2: %s", e)

    # 4. US National Debt (Fiscal Context)
    try:
        debts = terminal_data_service.get_treasury_debt(limit=1)
        if debts:
            d_snap = debts[0]
            d_date = normalize_observed_date(getattr(d_snap, "record_date", today_str), today_str)
            d_val = round(float(getattr(d_snap, "total_public_debt_usd", 0.0)) / 1e12, 2)
            d_valid = not getattr(d_snap, "is_stale", False)

            observables.append(MarketObservable(
                observable_id="obs_us_national_debt_trillion",
                asset_bucket="fixed_income",
                region="United States",
                indicator="US National Debt (Debt to the Penny)",
                value=f"{d_val:.2f}",
                unit="$T",
                observed_at=d_date,
                source_file="Terminal_V2_USTreasury",
                provider="Terminal V2 (Fiscal Data)",
                confidence="high" if d_valid else "low",
                is_valid=d_valid,
                status="verified" if d_valid else "stale",
                metadata={"total_public_debt_usd": getattr(d_snap, "total_public_debt_usd", 0.0)},
            ))
    except Exception as e:
        log.warning("Failed to collect US National Debt from Terminal V2: %s", e)

    # 5. Treasury Auction Demand (10-Year Note Bid-to-Cover)
    for security_type, security_term, obs_id, label in (
        ("Note", "10-Year", "obs_treasury_auction_bid_to_cover_10y", "10-Year Treasury Note"),
        ("Bill", "13-Week", "obs_treasury_auction_bid_to_cover_13w", "13-Week Treasury Bill"),
    ):
        try:
            auc = terminal_data_service.get_auction_demand_summary(security_type=security_type, security_term=security_term)
            a_date = normalize_observed_date(
                getattr(auc, "latest_auction_date", None) or getattr(auc, "as_of_date", today_str),
                today_str,
            )
            raw_btc = getattr(auc, "latest_bid_to_cover_ratio", None)
            btc = float(raw_btc) if raw_btc is not None else 0.0
            a_valid = not getattr(auc, "is_stale", False)

            if btc > 0:
                observables.append(MarketObservable(
                    observable_id=obs_id,
                    asset_bucket="fixed_income",
                    region="United States",
                    indicator=f"US {label} Auction Bid-to-Cover Ratio",
                    value=f"{btc:.2f}",
                    unit="ratio",
                    observed_at=a_date,
                    source_file="Terminal_V2_USTreasury",
                    provider="Terminal V2 (Fiscal Data)",
                    confidence="high" if a_valid else "low",
                    is_valid=a_valid,
                    status="verified" if a_valid else "stale",
                    stale_reason=getattr(auc, "stale_reason", None) or "",
                    metadata=asdict(auc) if is_dataclass(auc) else {"demand_delta": getattr(auc, "demand_delta", None)},
                ))
        except Exception as e:
            log.warning("Failed to collect Treasury %s auction demand from Terminal V2: %s", security_term, e)

    # 6. CFTC Gold Positioning (Commitments of Traders)
    try:
        cot = terminal_data_service.get_metals_cot(commodity="gold")
        c_date = normalize_observed_date(getattr(cot, "as_of_date", today_str), today_str)
        c_val = int(getattr(cot, "net_managed_money", 0))
        c_valid = not getattr(cot, "is_stale", False)

        observables.append(MarketObservable(
            observable_id="obs_cftc_gold_net_managed_money",
            asset_bucket="commodities",
            region="Global",
            indicator="CFTC Gold Futures Net Managed Money",
            value=f"{c_val}",
            unit="contracts",
            observed_at=c_date,
            source_file="Terminal_V2_CFTC",
            provider="Terminal V2 (CFTC)",
            confidence="high" if c_valid else "low",
            is_valid=c_valid,
            status="verified" if c_valid else "stale",
            metadata=asdict(cot) if is_dataclass(cot) else {"open_interest": getattr(cot, "open_interest", 0)},
        ))
    except Exception as e:
        log.warning("Failed to collect CFTC Metals COT from Terminal V2: %s", e)

    # 7. All Cboe commodity volatility metrics shown on the Macro page.
    for symbol, name, obs_id in (
        ("GVZ", "Gold", "obs_cboe_gold_volatility_gvz"),
        ("VXSLV", "Silver", "obs_cboe_silver_volatility_vxslv"),
        ("OVX", "Oil", "obs_cboe_oil_volatility_ovx"),
    ):
        try:
            vol = terminal_data_service.get_commodity_volatility(symbol)
            v_date = normalize_observed_date(
                getattr(vol, "close_date", None) or getattr(vol, "as_of_date", today_str),
                today_str,
            )
            v_val = float(getattr(vol, "implied_volatility", 0.0))
            v_valid = not getattr(vol, "is_stale", False)

            if v_val > 0:
                observables.append(MarketObservable(
                    observable_id=obs_id,
                    asset_bucket="commodities",
                    region="Global",
                    indicator=f"Cboe 30-Day {name} Volatility Index ({symbol})",
                    value=f"{v_val:.2f}",
                    unit="pts",
                    observed_at=v_date,
                    source_file="Terminal_V2_Cboe",
                    provider="Terminal V2 (Cboe)",
                    confidence="high" if v_valid else "low",
                    is_valid=v_valid,
                    status="verified" if v_valid else "stale",
                    stale_reason=getattr(vol, "stale_reason", None) or "",
                    metadata=asdict(vol) if is_dataclass(vol) else {},
                ))
        except Exception as e:
            log.warning("Failed to collect Cboe %s from Terminal V2: %s", symbol, e)

    return observables


def build_crypto_liquidity_observables(
    terminal_data_service: Optional[TerminalDataServicePort] = None,
    as_of_date: Optional[str] = None,
) -> list[MarketObservable]:
    """Build typed MarketObservables for Level 1 crypto macro liquidity (Stablecoins, BTC/Gold ratio, ETF flows)."""
    if terminal_data_service is None:
        try:
            from tools.market.terminal_v2.bootstrap import get_terminal_data_service
            terminal_data_service = get_terminal_data_service()
        except Exception as e:
            log.warning("Could not obtain TerminalDataServicePort instance: %s", e)
            return []

    today_str = as_of_date or datetime.now().strftime("%Y-%m-%d")
    observables: list[MarketObservable] = []

    try:
        liq = terminal_data_service.get_crypto_macro_liquidity()
        obs_date = normalize_observed_date(liq.as_of_date or today_str, today_str)
        is_valid = not liq.is_stale

        # 1. Total Stablecoin Supply (Global Digital Dollar / Dry Powder)
        if liq.stablecoin_total_usd is not None and liq.stablecoin_total_usd > 0:
            supply_b = round(liq.stablecoin_total_usd / 1_000_000_000.0, 2)
            observables.append(MarketObservable(
                observable_id="obs_crypto_stablecoin_supply_usd_b",
                asset_bucket="cash",
                region="Global",
                indicator="Global USD Stablecoin Total Circulating Supply",
                value=f"{supply_b:.2f}",
                unit="USD Bil",
                observed_at=obs_date,
                source_file="Terminal_V2_DeFiLlama",
                provider="Terminal V2 (DeFiLlama)",
                confidence="high" if is_valid else "low",
                is_valid=is_valid,
                status="verified" if is_valid else "stale",
                metadata={
                    "total_usd": liq.stablecoin_total_usd,
                    "change_7d_pct": liq.stablecoin_change_7d_pct,
                    "change_30d_pct": liq.stablecoin_change_30d_pct,
                    "liquidity_regime": liq.liquidity_regime,
                },
            ))

        # 2. Stablecoin 30-Day Growth Rate (Liquidity Expansion / Contraction Indicator)
        if liq.stablecoin_change_30d_pct is not None:
            observables.append(MarketObservable(
                observable_id="obs_crypto_stablecoin_supply_growth_30d",
                asset_bucket="cash",
                region="Global",
                indicator="Global Stablecoin Supply 30-Day Growth Rate",
                value=f"{liq.stablecoin_change_30d_pct:.2f}",
                unit="%",
                observed_at=obs_date,
                source_file="Terminal_V2_DeFiLlama",
                provider="Terminal V2 (DeFiLlama)",
                confidence="high" if is_valid else "low",
                is_valid=is_valid,
                status="verified" if is_valid else "stale",
                metadata={
                    "change_7d_pct": liq.stablecoin_change_7d_pct,
                    "regime": liq.liquidity_regime,
                },
            ))

        # 3. BTC / Gold Valuation Ratio (Risk Appetite vs Safe Haven)
        if liq.btc_gold_ratio is not None and liq.btc_gold_ratio > 0:
            observables.append(MarketObservable(
                observable_id="obs_crypto_btc_gold_ratio",
                asset_bucket="commodities",
                region="Global",
                indicator="Bitcoin to Gold Price Ratio (Risk Appetite Barometer)",
                value=f"{liq.btc_gold_ratio:.2f}",
                unit="ratio",
                observed_at=obs_date,
                source_file="Terminal_V2_MarketBenchmark",
                provider="Terminal V2 (Benchmark)",
                confidence="high" if is_valid else "low",
                is_valid=is_valid,
                status="verified" if is_valid else "stale",
                metadata={
                    "btc_price_usd": liq.btc_price_usd,
                    "btc_change_24h_pct": liq.btc_change_24h_pct,
                    "btc_change_7d_pct": liq.btc_change_7d_pct,
                },
            ))

        # 4. Spot BTC ETF Daily Net Inflow (Institutional Capital Flow)
        if liq.etf_daily_net_inflow_usd is not None:
            flow_m = round(liq.etf_daily_net_inflow_usd / 1_000_000.0, 2)
            observables.append(MarketObservable(
                observable_id="obs_crypto_etf_daily_net_inflow_m",
                asset_bucket="equities",
                region="United States",
                indicator="US Spot Bitcoin ETF Daily Net Inflow",
                value=f"{flow_m:.2f}",
                unit="USD Mil",
                observed_at=obs_date,
                source_file="Terminal_V2_SoSoValue",
                provider="Terminal V2 (SoSoValue)",
                confidence="high" if is_valid else "low",
                is_valid=is_valid,
                status="verified" if is_valid else "stale",
                metadata={
                    "daily_usd": liq.etf_daily_net_inflow_usd,
                    "cumulative_usd": liq.etf_cumulative_total_usd,
                },
            ))

    except Exception as e:
        log.warning("Failed to collect Level 1 crypto macro liquidity observables: %s", e)

    return observables


def synthesize_thai_market_stance_narrative(
    stance: dict[str, Any],
    thai_assets: list[Any] | None = None,
) -> str:
    """Generate an institutional-grade Thai market stance narrative from verified microstructure data."""
    parts = []

    # 1. Valuation & Breadth
    val = stance.get("valuation") or {}
    pe = val.get("pe_ratio")
    div_y = val.get("dividend_yield")
    breadth = stance.get("market_breadth") or {}
    ad_ratio = breadth.get("advance_decline_ratio")
    ad_sent = breadth.get("sentiment")

    if pe is not None or ad_ratio is not None:
        val_components = []
        if pe is not None:
            val_components.append(f"P/E {pe:.2f} เท่า")
        if div_y is not None:
            val_components.append(f"Dividend Yield {div_y:.2f}%")
        val_desc = f" ด้วยระดับราคา {' และ '.join(val_components)}" if val_components else ""

        breadth_desc = ""
        if ad_ratio is not None:
            breadth_desc = f" ประกอบกับ Market Breadth สะท้อนทัศนะเชิงบวก (A/D Ratio {ad_ratio:.2f}x{', ' + ad_sent if ad_sent else ''})"

        parts.append(
            f"สภาวะตลาดทุนไทยอยู่ในช่วงฟื้นตัวเชิงคุณค่า (Valuation-Driven Recovery){val_desc}{breadth_desc}"
        )

    # 2. Investor Flow
    flow = stance.get("investor_flow") or {}
    foreign = flow.get("foreign_net_mb")
    inst = flow.get("institution_net_mb")
    if foreign is not None or inst is not None:
        flow_components = []
        if foreign is not None:
            f_act = "ซื้อสุทธิ" if foreign > 0 else "ขายสุทธิ"
            flow_components.append(f"นักลงทุนต่างชาติ{f_act} {foreign:+,.2f} ล้านบาท")
        if inst is not None:
            i_act = "ซื้อสุทธิ" if inst > 0 else "ขายสุทธิ"
            flow_components.append(f"สถาบันในประเทศ{i_act} {inst:+,.2f} ล้านบาท เข้ามาช่วยดูดซับแรงขาย")
        parts.append(f"ด้านกระแสเงินทุน: {', '.join(flow_components)}")

    # 3. Policy Rate Differential Spread
    spread = stance.get("policy_spread_bps")
    if spread is not None:
        sign = "+" if spread > 0 else ""
        parts.append(
            f"ส่วนต่างอัตราดอกเบี้ยนโยบายสหรัฐฯ-ไทย (Fed-BOT Spread {sign}{spread:.1f} bps) ยังคงเป็นปัจจัยกดดันและชี้นำทิศทางค่าเงินบาท (USD/THB)"
        )

    # 4. Thai Yield Curve Spread
    yc_spread = stance.get("yield_curve_spread_10y_2y_bps")
    if yc_spread is not None:
        sign = "+" if yc_spread > 0 else ""
        curve_type = "Steepening" if yc_spread > 50 else ("Inverted" if yc_spread < 0 else "Flat")
        parts.append(
            f"เส้นอัตราผลตอบแทนพันธบัตรรัฐบาลไทย (ThaiBMA 10Y-2Y Spread {sign}{yc_spread:.1f} bps, {curve_type}) สะท้อนมุมมองการฟื้นตัวของเศรษฐกิจระยะยาว"
        )

    # 5. Asset Allocation Stance
    if thai_assets:
        asset_summaries = []
        for a in thai_assets:
            a_class = getattr(a, "asset_class", None) or (a.get("asset_class") if isinstance(a, dict) else "")
            a_stance = getattr(a, "stance", None) or (a.get("stance") if isinstance(a, dict) else "")
            a_rat = getattr(a, "rationale", None) or (a.get("rationale") if isinstance(a, dict) else "")
            if a_class and a_stance:
                asset_summaries.append(f"{a_class}: {a_stance} ({a_rat})" if a_rat else f"{a_class}: {a_stance}")
        if asset_summaries:
            parts.append(f"กลยุทธ์จัดสรรสินทรัพย์: {'; '.join(asset_summaries)}")

    return " โดย".join(parts) if len(parts) <= 2 else " ".join(parts)


def build_thai_market_stance(observables: list[MarketObservable]) -> dict[str, Any]:
    """Aggregate Thailand market stance radar from verified observables."""
    stance: dict[str, Any] = {
        "investor_flow": {},
        "market_breadth": {},
        "valuation": {},
        "physical_gold": {},
        "policy_spread_bps": None,
    }

    for obs in observables:
        if not getattr(obs, "is_valid", True):
            continue
        oid = obs.observable_id
        if oid == "obs_set_flow_foreign":
            stance["investor_flow"]["foreign_net_mb"] = float(obs.value)
        elif oid == "obs_set_flow_institution":
            stance["investor_flow"]["institution_net_mb"] = float(obs.value)
        elif oid == "obs_set_flow_prop":
            stance["investor_flow"]["prop_net_mb"] = float(obs.value)
        elif oid == "obs_set_flow_retail":
            stance["investor_flow"]["retail_net_mb"] = float(obs.value)
        elif oid == "obs_set_advance_decline_ratio":
            ad = float(obs.value)
            stance["market_breadth"]["advance_decline_ratio"] = ad
            stance["market_breadth"]["sentiment"] = "bullish" if ad > 1.2 else ("bearish" if ad < 0.8 else "neutral")
        elif oid == "obs_set_valuation_pe":
            stance["valuation"]["pe_ratio"] = float(obs.value)
        elif oid == "obs_set_valuation_pbv":
            stance["valuation"]["pbv_ratio"] = float(obs.value)
        elif oid == "obs_set_valuation_dividend_yield":
            stance["valuation"]["dividend_yield"] = float(obs.value)
        elif oid == "obs_gta_gold_bar_sell":
            stance["physical_gold"]["bar_sell_thb"] = float(obs.value)
            stance["physical_gold"]["unit"] = obs.unit
        elif oid in ("obs_diff_us_th_policy_rate_bis", "obs_diff_us_th_policy_rate"):
            stance["policy_spread_bps"] = float(obs.value)
        elif oid == "obs_th_gov_10y_2y_spread":
            stance["yield_curve_spread_10y_2y_bps"] = float(obs.value)

    # Automatically synthesize institutional rationale if quantitative signals are present
    has_signals = (
        bool(stance["investor_flow"])
        or bool(stance["valuation"])
        or stance["policy_spread_bps"] is not None
        or bool(stance["market_breadth"])
    )
    if has_signals:
        stance["rationale"] = synthesize_thai_market_stance_narrative(stance)

    return stance
