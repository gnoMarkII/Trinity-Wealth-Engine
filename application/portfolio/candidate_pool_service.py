"""Candidate Pool Discovery & Merging Service.

Architecture Layer: application/portfolio/ (Use Cases & Orchestration)
Discovers candidates from:
- Pool A: Stocks held within portfolio's mutual funds (via look-through)
- Pool B: Stocks with active Equity Sidecars in the Knowledge Vault
Merges duplicates via pure domain SecurityIdentity and tags holding / watchlist status.
"""
from __future__ import annotations

from decimal import Decimal
import logging
from typing import Any, Dict, List, Optional, Set

from application.portfolio.opportunity_ports import (
    EquityResearchReaderPort,
    FundConstituentsReaderPort,
    PortfolioSnapshotReaderPort,
    WatchlistReaderPort,
)
from core.portfolio.sector_taxonomy import lookup_ticker_sector, normalize_sector_name
from core.portfolio.security_identity import classify_asset_type, resolve_security_identity
from core.opportunity_finder.models import OpportunityCandidate

log = logging.getLogger(__name__)


class CandidatePoolService:
    """Discovers and consolidates candidates from Pool A and Pool B."""

    def __init__(
        self,
        portfolio_reader: PortfolioSnapshotReaderPort,
        fund_port: FundConstituentsReaderPort,
        equity_reader: EquityResearchReaderPort,
        watchlist_reader: WatchlistReaderPort,
    ) -> None:
        self._portfolio_reader = portfolio_reader
        self._fund_port = fund_port
        self._equity_reader = equity_reader
        self._watchlist_reader = watchlist_reader

    def discover_candidates(
        self,
        portfolio_id: str = "default",
        look_through_report: Optional[Any] = None,
    ) -> List[OpportunityCandidate]:
        """Aggregate, deduplicate, and enrich candidates across Pool A and Pool B."""
        # 1. Read Current Portfolio State
        try:
            portfolio_state = self._portfolio_reader.get_current_portfolio_state(portfolio_id=portfolio_id)
        except Exception as e:
            log.warning("Could not read portfolio state for %s: %s", portfolio_id, e)
            portfolio_state = None

        holdings = getattr(portfolio_state, "holdings", []) or []

        # Identify directly held stock tickers (case/suffix-normalized)
        direct_held_symbols: Set[str] = set()
        for h in holdings:
            sym = str(getattr(h, "symbol", "")).strip().upper()
            raw_type = str(getattr(h, "asset_type", "")).strip()
            a_type = classify_asset_type(sym, asset_type_raw=raw_type)
            if a_type in ("EQUITY", "REIT"):
                ident = resolve_security_identity(sym, market=getattr(h, "market", None))
                direct_held_symbols.add(ident.symbol)

        # Identify already watchlisted symbols
        watchlisted_symbols: Set[str] = set()
        try:
            wl = self._watchlist_reader.load_watchlist(portfolio_id=portfolio_id)
            items = getattr(wl, "items", []) or []
            for item in items:
                w_sym = str(getattr(item, "symbol", "")).strip().upper()
                w_ident = resolve_security_identity(w_sym)
                watchlisted_symbols.add(w_ident.symbol)
        except Exception as e:
            log.debug("Watchlist read fallback for %s: %s", portfolio_id, e)

        # Candidate mapping keyed by canonical security_id: "US:MSFT" or "TH:CPALL"
        candidate_map: Dict[str, OpportunityCandidate] = {}

        # 2. Pool A: Fund Look-Through Constituents
        # If pre-calculated look_through_report provided, reuse its exposures
        if look_through_report and hasattr(look_through_report, "holdings_exposure"):
            for exp in look_through_report.holdings_exposure:
                # Include indirect equities from funds
                if exp.indirect_value_thb > Decimal("0"):
                    ident = resolve_security_identity(
                        exp.ticker,
                        market=exp.market,
                        company_name=exp.company_name,
                        asset_type=getattr(exp, "asset_type", "EQUITY"),
                    )
                    raw_sec = getattr(exp, "sector", "") or getattr(exp, "sector_name", "")
                    sec_en, sec_th = normalize_sector_name(raw_sec)
                    if sec_en in ("Other", "อื่นๆ"):
                        sec_en, sec_th = lookup_ticker_sector(exp.ticker)
                    sources = [f"fund:{s[0]}" for s in getattr(exp, "source_funds", [])]

                    cand = OpportunityCandidate(
                        security_id=ident.security_id,
                        issuer_id=ident.issuer_id,
                        symbol=ident.symbol,
                        market=ident.market,
                        currency=ident.currency,
                        asset_type=ident.asset_type,
                        company_name=exp.company_name or ident.symbol,
                        sector_name=sec_en,
                        sector_name_th=sec_th,
                        source_pools=["POOL_A_FUND_LOOK_THROUGH"],
                        source_refs=sources,
                        already_directly_held=ident.symbol in direct_held_symbols,
                        already_watchlisted=ident.symbol in watchlisted_symbols,
                        current_direct_value_thb=exp.direct_value_thb,
                        current_indirect_value_thb=exp.indirect_value_thb,
                        current_total_exposure_thb=exp.total_value_thb,
                        current_total_weight_pct=exp.total_weight_pct,
                    )
                    candidate_map[ident.security_id] = cand
        else:
            # Fallback: scan fund holdings directly
            for h in holdings:
                sym = str(getattr(h, "symbol", "")).strip().upper()
                raw_type = str(getattr(h, "asset_type", "")).strip()
                if classify_asset_type(sym, asset_type_raw=raw_type) in ("FUND", "ETF"):
                    try:
                        c_data = self._fund_port.fetch_constituents(sym)
                        if not c_data:
                            continue
                        for top in getattr(c_data, "top_holdings", []):
                            t_sym = (getattr(top, "ticker", "") or "").strip().upper()
                            if not t_sym:
                                continue
                            t_ident = resolve_security_identity(
                                t_sym,
                                company_name=getattr(top, "name", ""),
                                asset_type=getattr(top, "asset_type", "EQUITY"),
                            )
                            f_sec_en, f_sec_th = lookup_ticker_sector(t_sym)
                            if t_ident.security_id not in candidate_map:
                                candidate_map[t_ident.security_id] = OpportunityCandidate(
                                    security_id=t_ident.security_id,
                                    issuer_id=t_ident.issuer_id,
                                    symbol=t_ident.symbol,
                                    market=t_ident.market,
                                    currency=t_ident.currency,
                                    asset_type=t_ident.asset_type,
                                    company_name=getattr(top, "name", "") or t_ident.symbol,
                                    sector_name=f_sec_en,
                                    sector_name_th=f_sec_th,
                                    source_pools=["POOL_A_FUND_LOOK_THROUGH"],
                                    source_refs=[f"fund:{sym}"],
                                    already_directly_held=t_ident.symbol in direct_held_symbols,
                                    already_watchlisted=t_ident.symbol in watchlisted_symbols,
                                )
                            else:
                                candidate_map[t_ident.security_id].source_refs.append(f"fund:{sym}")
                    except Exception as e:
                        log.debug("Error discovering fund constituents for %s: %s", sym, e)

        # 3. Pool B: Equity Sidecars
        try:
            sidecars = self._equity_reader.get_all_equity_sidecars()
            for s_ticker, sidecar in sidecars.items():
                clean_sym = s_ticker.replace(".BK", "").strip().upper()
                if not clean_sym:
                    continue

                mkt = str(sidecar.get("market") or ("TH" if s_ticker.endswith(".BK") else "US")).strip().upper()
                c_name = str(sidecar.get("company_name") or clean_sym).strip()
                quant = sidecar.get("quant_signals") if isinstance(sidecar.get("quant_signals"), dict) else {}
                sec_kpis = quant.get("sector_kpis") if isinstance(quant.get("sector_kpis"), dict) else {}
                sec_raw = str(
                    quant.get("peer_sector")
                    or sec_kpis.get("sector_name")
                    or quant.get("sector")
                    or sidecar.get("sector")
                    or ""
                ).strip()
                sec_en, sec_th = normalize_sector_name(sec_raw)
                if sec_en in ("Other", "อื่นๆ"):
                    sec_en, sec_th = lookup_ticker_sector(clean_sym)

                ident = resolve_security_identity(clean_sym, market=mkt, company_name=c_name)

                if ident.security_id in candidate_map:
                    # Merge with existing Pool A candidate
                    existing = candidate_map[ident.security_id]
                    if "POOL_B_EQUITY_SIDECAR" not in existing.source_pools:
                        existing.source_pools.append("POOL_B_EQUITY_SIDECAR")
                    existing.source_refs.append(f"sidecar:{clean_sym}")
                    if existing.sector_name in ("Other", "อื่นๆ") and sec_en not in ("Other", "อื่นๆ"):
                        existing.sector_name = sec_en
                        existing.sector_name_th = sec_th
                else:
                    # New Pool B candidate
                    cand = OpportunityCandidate(
                        security_id=ident.security_id,
                        issuer_id=ident.issuer_id,
                        symbol=ident.symbol,
                        market=ident.market,
                        currency=ident.currency,
                        asset_type=ident.asset_type,
                        company_name=c_name,
                        sector_name=sec_en,
                        sector_name_th=sec_th,
                        source_pools=["POOL_B_EQUITY_SIDECAR"],
                        source_refs=[f"sidecar:{clean_sym}"],
                        already_directly_held=ident.symbol in direct_held_symbols,
                        already_watchlisted=ident.symbol in watchlisted_symbols,
                    )
                    candidate_map[ident.security_id] = cand
        except Exception as e:
            log.warning("Could not read equity sidecars for Pool B: %s", e)

        candidates = list(candidate_map.values())
        return candidates
