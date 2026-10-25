"""Fund Look-Through Application Service.

Architecture Layer: application/portfolio/ (Use Cases & Orchestration)
Coordinates portfolio state retrieval, mutual fund constituent look-throughs via ports,
and deterministic portfolio-wide overlap & concentration calculations.
"""
from __future__ import annotations

from decimal import Decimal
import logging
from typing import Any, Dict, List, Optional

from application.portfolio.advisory_ports import PortfolioStateReaderPort
from core.portfolio.look_through_calculator import LookThroughCalculator
from core.portfolio.look_through_models import (
    FundConstituentHolding,
    FundConstituentSector,
    FundConstituentsDomain,
    PortfolioLookThroughReport,
)
from schemas.portfolio_advisory_schemas import (
    FundConstituentsDTO,
    FundHoldingItemDTO,
    FundSectorItemDTO,
    LookThroughHoldingExposureDTO,
    PortfolioLookThroughReportDTO,
    TrueSectorExposureDTO,
)
from tools.portfolio.ports.thai_fund_port import ThaiFundPricePort, FundConstituentsData

log = logging.getLogger(__name__)


class FundLookThroughService:
    """Application Service for Fund Look-Through and Cross-Portfolio Overlap."""

    def __init__(
        self,
        portfolio_reader: PortfolioStateReaderPort,
        fund_port: ThaiFundPricePort,
    ) -> None:
        self._portfolio_reader = portfolio_reader
        self._fund_port = fund_port

    def get_fund_constituents(self, symbol: str) -> Optional[FundConstituentsData]:
        """Fetch constituent holdings, sector breakdown, and asset allocation for a single fund."""
        try:
            return self._fund_port.fetch_constituents(symbol)
        except Exception as e:
            log.warning("Error fetching constituents for fund %s: %s", symbol, e)
            return None

    def get_fund_constituents_dto(self, symbol: str) -> Optional[FundConstituentsDTO]:
        """Fetch single fund constituents and convert to DTO."""
        data = self.get_fund_constituents(symbol)
        if not data:
            return None

        return FundConstituentsDTO(
            fund_id=data.fund_id,
            symbol=data.symbol,
            as_of_date=data.as_of_date,
            master_fund_name=data.master_fund_name,
            master_fund_ticker=data.master_fund_ticker,
            top_holdings=[
                FundHoldingItemDTO(
                    name=h.name,
                    percent=float(h.percent),
                    ticker=h.ticker,
                    asset_type=h.asset_type,
                )
                for h in data.top_holdings
            ],
            sector_breakdown=[
                FundSectorItemDTO(
                    sector_name_th=s.sector_name_th,
                    sector_name_en=s.sector_name_en,
                    percent=float(s.percent),
                )
                for s in data.sector_breakdown
            ],
            asset_allocation={k: float(v) for k, v in data.asset_allocation.items()},
        )

    def get_portfolio_look_through(self, portfolio_id: str = "default") -> PortfolioLookThroughReport:
        """Execute complete portfolio look-through across direct and indirect fund assets."""
        try:
            state = self._portfolio_reader.get_current_portfolio_state(portfolio_id=portfolio_id)
        except Exception as e:
            log.warning("Could not read portfolio state for '%s': %s", portfolio_id, e)
            return PortfolioLookThroughReport(
                portfolio_id=portfolio_id,
                evaluated_at="",
                total_nav_thb=Decimal("0"),
            )

        holdings = getattr(state, "holdings", []) or []
        summary = getattr(state, "summary", None)
        total_nav_val = getattr(summary, "market_value_thb", None) or getattr(summary, "total_value_thb", None)
        total_nav_thb = Decimal(str(total_nav_val)) if total_nav_val is not None else None

        # Identify all mutual funds / PVD holdings
        fund_constituents_map: Dict[str, FundConstituentsData] = {}
        for h in holdings:
            sym = str(getattr(h, "symbol", "")).strip().upper()
            asset_type = str(getattr(h, "asset_type", "")).strip().upper()
            is_fund = any(k in asset_type for k in ("FUND", "MUTUAL_FUND", "PVD", "RMF", "SSF", "THAIESG", "ETF")) or sym.startswith("PVD_")

            if is_fund and sym not in fund_constituents_map:
                try:
                    c_data = self._fund_port.fetch_constituents(sym)
                    if c_data:
                        fund_constituents_map[sym] = FundConstituentsDomain(
                            fund_id=c_data.fund_id,
                            symbol=c_data.symbol,
                            as_of_date=c_data.as_of_date,
                            master_fund_name=c_data.master_fund_name,
                            master_fund_ticker=c_data.master_fund_ticker,
                            top_holdings=[
                                FundConstituentHolding(
                                    name=item.name,
                                    percent=item.percent,
                                    ticker=item.ticker,
                                    asset_type=item.asset_type,
                                )
                                for item in c_data.top_holdings
                            ],
                            sector_breakdown=[
                                FundConstituentSector(
                                    sector_name_th=s.sector_name_th,
                                    sector_name_en=s.sector_name_en,
                                    percent=s.percent,
                                )
                                for s in c_data.sector_breakdown
                            ],
                            asset_allocation=c_data.asset_allocation,
                            is_curated=getattr(c_data, "is_curated", False),
                            source_date=getattr(c_data, "as_of_date", None),
                        )
                except Exception as e:
                    log.warning("Graceful degradation: failed to fetch constituents for fund %s: %s", sym, e)

        return LookThroughCalculator.calculate(
            holdings=holdings,
            fund_constituents_map=fund_constituents_map,
            total_nav_thb=total_nav_thb,
            portfolio_id=portfolio_id,
        )

    def get_portfolio_look_through_dto(self, portfolio_id: str = "default") -> PortfolioLookThroughReportDTO:
        """Calculate look-through and convert to API DTO."""
        report = self.get_portfolio_look_through(portfolio_id=portfolio_id)
        return PortfolioLookThroughReportDTO(
            portfolio_id=report.portfolio_id,
            evaluated_at=report.evaluated_at,
            total_nav_thb=float(report.total_nav_thb),
            holdings_exposure=[
                LookThroughHoldingExposureDTO(
                    ticker=h.ticker,
                    company_name=h.company_name,
                    market=h.market,
                    direct_value_thb=float(h.direct_value_thb),
                    direct_weight_pct=float(h.direct_weight_pct),
                    indirect_value_thb=float(h.indirect_value_thb),
                    indirect_weight_pct=float(h.indirect_weight_pct),
                    total_value_thb=float(h.total_value_thb),
                    total_weight_pct=float(h.total_weight_pct),
                    source_funds=[(s[0], float(s[1])) for s in h.source_funds],
                    is_cap_breached=h.is_cap_breached,
                    is_hidden_concentration=h.is_hidden_concentration,
                )
                for h in report.holdings_exposure
            ],
            top_overlapped_stocks=[
                LookThroughHoldingExposureDTO(
                    ticker=h.ticker,
                    company_name=h.company_name,
                    market=h.market,
                    direct_value_thb=float(h.direct_value_thb),
                    direct_weight_pct=float(h.direct_weight_pct),
                    indirect_value_thb=float(h.indirect_value_thb),
                    indirect_weight_pct=float(h.indirect_weight_pct),
                    total_value_thb=float(h.total_value_thb),
                    total_weight_pct=float(h.total_weight_pct),
                    source_funds=[(s[0], float(s[1])) for s in h.source_funds],
                    is_cap_breached=h.is_cap_breached,
                    is_hidden_concentration=h.is_hidden_concentration,
                )
                for h in report.top_overlapped_stocks
            ],
            sector_exposure=[
                TrueSectorExposureDTO(
                    sector_name=s.sector_name,
                    sector_name_th=s.sector_name_th,
                    total_value_thb=float(s.total_value_thb),
                    weight_pct=float(s.weight_pct),
                    direct_stocks_pct=float(s.direct_stocks_pct),
                    funds_pct=float(s.funds_pct),
                    is_crowded=s.is_crowded,
                )
                for s in report.sector_exposure
            ],
            hidden_concentration_warnings=report.hidden_concentration_warnings,
            cap_breached_count=report.cap_breached_count,
        )
