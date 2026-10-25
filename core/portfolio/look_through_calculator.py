"""Look-Through Calculator.

Architecture Layer: core/portfolio/ (Pure Domain)
Performs deterministic math on direct holdings and underlying mutual fund constituents
using Decimal to compute true economic exposures, cross-fund overlaps, and single-stock caps.
Adheres strictly to Pure Domain isolation: no frameworks, no Pydantic, no tools imports.
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Literal, Optional, Tuple

from core.portfolio.look_through_models import (
    FundConstituentsDomain,
    LookThroughCoverage,
    LookThroughHoldingExposure,
    PortfolioLookThroughReport,
    TrueSectorExposure,
)
from core.portfolio.sector_taxonomy import lookup_ticker_sector, normalize_sector_name
from core.portfolio.security_identity import classify_asset_type, resolve_security_identity

_CAP_THRESHOLD_PCT = Decimal("15.0")
_CROWDED_SECTOR_THRESHOLD_PCT = Decimal("30.0")


def _to_decimal(val: Any) -> Decimal:
    """Safe conversion to Decimal for financial calculation."""
    if val is None:
        return Decimal("0")
    if isinstance(val, Decimal):
        return val
    try:
        return Decimal(str(val))
    except Exception:
        return Decimal("0")


class LookThroughCalculator:
    """Pure domain calculator for look-through equity & sector aggregation."""

    @staticmethod
    def calculate(
        holdings: List[Any],
        fund_constituents_map: Dict[str, Any],
        total_nav_thb: Optional[Decimal] = None,
        portfolio_id: str = "default",
    ) -> PortfolioLookThroughReport:
        """Calculate consolidated look-through exposures across all direct and fund holdings."""
        evaluated_at = datetime.now(timezone.utc).isoformat()

        # 1. Calculate NAV if not provided
        if total_nav_thb is None or total_nav_thb <= Decimal("0"):
            nav_sum = Decimal("0")
            for h in holdings:
                nav_sum += _to_decimal(getattr(h, "market_value_thb", 0))
            total_nav_thb = nav_sum

        if total_nav_thb <= Decimal("0"):
            return PortfolioLookThroughReport(
                portfolio_id=portfolio_id,
                evaluated_at=evaluated_at,
                total_nav_thb=Decimal("0"),
                coverage=LookThroughCoverage(),
            )

        # 2. Tracking structures
        direct_equities: Dict[str, dict] = {}
        indirect_equities: Dict[str, dict] = {}
        sectors_agg: Dict[str, dict] = {}

        total_fund_value_thb = Decimal("0")
        resolved_fund_value_thb = Decimal("0")
        weighted_resolved_security_val = Decimal("0")
        weighted_resolved_sector_val = Decimal("0")

        for h in holdings:
            raw_sym = str(getattr(h, "symbol", "")).strip()
            raw_asset_type = str(getattr(h, "asset_type", "")).strip()
            c_name_hint = str(getattr(h, "company_name", "") or getattr(h, "asset_name", "") or "").strip()
            val_thb = _to_decimal(getattr(h, "market_value_thb", 0))

            if val_thb <= Decimal("0"):
                continue

            # Canonical asset classification
            asset_class = classify_asset_type(raw_sym, asset_type_raw=raw_asset_type, name=c_name_hint)

            if asset_class == "CASH":
                sec_en, sec_th = normalize_sector_name("เงินสด")
                if sec_en not in sectors_agg:
                    sectors_agg[sec_en] = {"th": sec_th, "direct_val": Decimal("0"), "funds_val": Decimal("0")}
                sectors_agg[sec_en]["direct_val"] += val_thb
                continue

            if asset_class in ("GOLD", "BOND"):
                # Track in sectors but not in single-equity cap
                sec_raw = str(getattr(h, "sector", "") or getattr(h, "sector_name", "") or asset_class).strip()
                sec_en, sec_th = normalize_sector_name(sec_raw)
                if sec_en not in sectors_agg:
                    sectors_agg[sec_en] = {"th": sec_th, "direct_val": Decimal("0"), "funds_val": Decimal("0")}
                sectors_agg[sec_en]["direct_val"] += val_thb
                continue

            if asset_class not in ("FUND", "ETF"):
                # Direct equity or REIT holding
                sec_ident = resolve_security_identity(
                    raw_sym,
                    market=getattr(h, "market", None),
                    company_name=c_name_hint,
                    asset_type=asset_class,
                )
                sec_key = sec_ident.security_id or f"{sec_ident.market}:{sec_ident.symbol}"
                if sec_key in direct_equities:
                    direct_equities[sec_key]["value_thb"] += val_thb
                else:
                    direct_equities[sec_key] = {
                        "ticker": sec_ident.symbol,
                        "security_id": sec_key,
                        "company_name": sec_ident.company_name,
                        "market": sec_ident.market,
                        "value_thb": val_thb,
                        "asset_type": sec_ident.asset_type,
                    }

                # Sector mapping for direct holding
                sec_raw = str(getattr(h, "sector", "") or getattr(h, "sector_name", "") or "").strip()
                if not sec_raw and sec_ident.asset_type == "REIT":
                    sec_raw = "Real Estate"
                if not sec_raw:
                    sec_en, sec_th = lookup_ticker_sector(raw_sym)
                else:
                    sec_en, sec_th = normalize_sector_name(sec_raw)

                if sec_en and sec_en != "Other":
                    if sec_en not in sectors_agg:
                        sectors_agg[sec_en] = {"th": sec_th, "direct_val": Decimal("0"), "funds_val": Decimal("0")}
                    sectors_agg[sec_en]["direct_val"] += val_thb
                elif sec_raw:
                    if sec_en not in sectors_agg:
                        sectors_agg[sec_en] = {"th": sec_th, "direct_val": Decimal("0"), "funds_val": Decimal("0")}
                    sectors_agg[sec_en]["direct_val"] += val_thb

            else:
                # Fund / ETF holding: Look through constituents
                total_fund_value_thb += val_thb
                constituents = (
                    fund_constituents_map.get(raw_sym)
                    or fund_constituents_map.get(raw_sym.upper())
                    or fund_constituents_map.get(raw_sym.replace(" ", "-"))
                    or fund_constituents_map.get(raw_sym.replace("-", " "))
                )
                if not constituents:
                    continue

                resolved_fund_value_thb += val_thb

                # A. Look through top holdings
                holdings_resolved_sum = Decimal("0")
                for c_item in getattr(constituents, "top_holdings", []):
                    c_ticker = (getattr(c_item, "ticker", "") or "").strip().upper()
                    if not c_ticker:
                        continue
                    c_asset_type = getattr(c_item, "asset_type", "EQUITY")
                    if str(c_asset_type).upper() not in ("EQUITY", "STOCK"):
                        continue

                    pct_in_fund = _to_decimal(getattr(c_item, "percent", 0))
                    effective_val = (val_thb * pct_in_fund) / Decimal("100")
                    if effective_val <= Decimal("0"):
                        continue

                    holdings_resolved_sum += pct_in_fund
                    c_name = getattr(c_item, "name", c_ticker)
                    c_ident = resolve_security_identity(
                        c_ticker,
                        company_name=c_name,
                        asset_type=c_asset_type,
                    )
                    sec_key = c_ident.security_id or f"{c_ident.market}:{c_ident.symbol}"

                    if sec_key not in indirect_equities:
                        indirect_equities[sec_key] = {
                            "ticker": c_ident.symbol,
                            "security_id": sec_key,
                            "company_name": c_ident.company_name,
                            "market": c_ident.market,
                            "asset_type": c_ident.asset_type,
                            "sources": [(raw_sym, effective_val)],
                        }
                    else:
                        indirect_equities[sec_key]["sources"].append((raw_sym, effective_val))

                weighted_resolved_security_val += (val_thb * (min(Decimal("100.0"), holdings_resolved_sum) / Decimal("100.0")))

                # B. Look through sectors
                sector_resolved_sum = Decimal("0")
                for s_item in getattr(constituents, "sector_breakdown", []):
                    sec_pct = _to_decimal(getattr(s_item, "percent", 0))
                    sec_val = (val_thb * sec_pct) / Decimal("100")
                    sec_name_raw = getattr(s_item, "sector_name_en", "") or getattr(s_item, "sector_name_th", "")
                    sec_en, sec_th = normalize_sector_name(sec_name_raw)
                    if sec_en not in sectors_agg:
                        sectors_agg[sec_en] = {"th": sec_th, "direct_val": Decimal("0"), "funds_val": Decimal("0")}
                    sectors_agg[sec_en]["funds_val"] += sec_val
                    sector_resolved_sum += sec_pct

                weighted_resolved_sector_val += (val_thb * (min(Decimal("100.0"), sector_resolved_sum) / Decimal("100.0")))

        # 3. Consolidate direct and indirect stock exposures
        all_sec_keys = set(direct_equities.keys()) | set(indirect_equities.keys())
        consolidated: List[LookThroughHoldingExposure] = []
        hidden_warnings: List[str] = []
        cap_breached_count = 0

        for sec_key in all_sec_keys:
            d_info = direct_equities.get(sec_key)
            i_info = indirect_equities.get(sec_key)

            ticker = (d_info["ticker"] if d_info else i_info["ticker"]) if (d_info or i_info) else sec_key.split(":")[-1]
            d_val = d_info["value_thb"] if d_info else Decimal("0")
            i_sources = i_info["sources"] if i_info else []
            i_val = sum((s[1] for s in i_sources), Decimal("0")) if i_sources else Decimal("0")
            tot_val = d_val + i_val

            # Exact full-precision Decimal calculation before rounding (F04, P0-08)
            raw_d_weight = (d_val / total_nav_thb) * Decimal("100")
            raw_i_weight = (i_val / total_nav_thb) * Decimal("100")
            raw_tot_weight = (tot_val / total_nav_thb) * Decimal("100")

            # Check Cap with unrounded precision
            is_breached = raw_tot_weight > _CAP_THRESHOLD_PCT
            is_hidden = is_breached and (raw_d_weight <= _CAP_THRESHOLD_PCT) and (i_val > Decimal("0"))

            # Display quantized weights
            d_weight = raw_d_weight.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            i_weight = raw_i_weight.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            tot_weight = raw_tot_weight.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

            c_name = (d_info["company_name"] if d_info else i_info["company_name"]) if (d_info or i_info) else ticker
            mkt: Literal["US", "TH", "GLOBAL"] = d_info["market"] if d_info else (i_info["market"] if i_info else "US")
            a_type = d_info["asset_type"] if d_info else (i_info["asset_type"] if i_info else "EQUITY")

            if is_breached:
                cap_breached_count += 1

            if is_hidden:
                funds_str = ", ".join(f"{s[0]} ({s[1]:,.0f} ฿)" for s in i_sources)
                hidden_warnings.append(
                    f"Hidden Concentration Alert: {ticker} total economic exposure is {tot_weight:.1f}% "
                    f"(exceeds {_CAP_THRESHOLD_PCT}% Cap). Direct: {d_weight:.1f}%, Indirect: {i_weight:.1f}% via {funds_str}."
                )

            consolidated.append(
                LookThroughHoldingExposure(
                    ticker=ticker,
                    company_name=c_name,
                    market=mkt,
                    direct_value_thb=d_val.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
                    direct_weight_pct=d_weight,
                    indirect_value_thb=i_val.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
                    indirect_weight_pct=i_weight,
                    total_value_thb=tot_val.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
                    total_weight_pct=tot_weight,
                    source_funds=i_sources,
                    is_cap_breached=is_breached,
                    is_hidden_concentration=is_hidden,
                    issuer_id=ticker,
                    security_id=sec_key,
                    asset_type=a_type,
                    raw_total_weight_pct=raw_tot_weight,
                )
            )

        # Sort consolidated by total_value_thb descending
        consolidated.sort(key=lambda x: x.total_value_thb, reverse=True)

        # Filter top overlapped stocks
        overlapped = [
            item for item in consolidated
            if (item.direct_value_thb > Decimal("0") and item.indirect_value_thb > Decimal("0"))
            or len(item.source_funds) > 1
        ]

        # 4. Consolidate True Sector Breakdown
        sector_results: List[TrueSectorExposure] = []
        for s_en, s_data in sectors_agg.items():
            s_d_val = s_data["direct_val"]
            s_f_val = s_data["funds_val"]
            s_tot = s_d_val + s_f_val
            if s_tot <= Decimal("0"):
                continue

            raw_s_weight = (s_tot / total_nav_thb) * Decimal("100")
            s_weight = raw_s_weight.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            s_d_pct = ((s_d_val / total_nav_thb) * Decimal("100")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            s_f_pct = ((s_f_val / total_nav_thb) * Decimal("100")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            is_crowded = raw_s_weight > _CROWDED_SECTOR_THRESHOLD_PCT

            sector_results.append(
                TrueSectorExposure(
                    sector_name=s_en,
                    sector_name_th=s_data["th"],
                    total_value_thb=s_tot.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
                    weight_pct=s_weight,
                    direct_stocks_pct=s_d_pct,
                    funds_pct=s_f_pct,
                    is_crowded=is_crowded,
                )
            )

        sector_results.sort(key=lambda s: s.total_value_thb, reverse=True)

        # 5. Calculate Look-Through Coverage (P0-07)
        if total_fund_value_thb > Decimal("0"):
            fund_fetch_cov = ((resolved_fund_value_thb / total_fund_value_thb) * Decimal("100")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            sec_cov = ((weighted_resolved_security_val / total_fund_value_thb) * Decimal("100")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            sec_breakdown_cov = ((weighted_resolved_sector_val / total_fund_value_thb) * Decimal("100")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        else:
            fund_fetch_cov = Decimal("100.00")
            sec_cov = Decimal("100.00")
            sec_breakdown_cov = Decimal("100.00")

        unresolved_funds_val = total_fund_value_thb - resolved_fund_value_thb
        coverage = LookThroughCoverage(
            fund_fetch_coverage=fund_fetch_cov,
            security_look_through_coverage=sec_cov,
            sector_coverage=sec_breakdown_cov,
            unresolved_value_thb=unresolved_funds_val.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            total_fund_value_thb=total_fund_value_thb.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            resolved_fund_value_thb=resolved_fund_value_thb.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
        )

        return PortfolioLookThroughReport(
            portfolio_id=portfolio_id,
            evaluated_at=evaluated_at,
            total_nav_thb=total_nav_thb.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            holdings_exposure=consolidated,
            top_overlapped_stocks=overlapped,
            sector_exposure=sector_results,
            hidden_concentration_warnings=hidden_warnings,
            cap_breached_count=cap_breached_count,
            coverage=coverage,
            unresolved_value_thb=unresolved_funds_val.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
        )
