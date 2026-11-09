"""Opportunity Application Service.

Architecture Layer: application/portfolio/ (Use Cases & Orchestration)
Coordinates:
- Portfolio snapshot retrieval and look-through exposures
- Candidate pool discovery and deduplication
- Pure domain 3-Dimensional Opportunity Evaluation
- API DTO conversion and cache management
Adheres strictly to Hexagonal Architecture: depends only on ports and domain.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
import logging
import time
from typing import Any, Dict, List, Optional

from application.portfolio.candidate_pool_service import CandidatePoolService
from application.portfolio.opportunity_ports import (
    ConfirmedPolicyReaderPort,
    EquityResearchReaderPort,
    FundConstituentsReaderPort,
    MacroSnapshotReaderPort,
    PortfolioSnapshotReaderPort,
    WatchlistReaderPort,
)
from core.opportunity_finder.evaluator import OpportunityEvaluator
from core.opportunity_finder.models import OpportunityReport
from schemas.opportunity_schemas import (
    CandidateEvaluationResultDTO,
    OpportunityCandidateDTO,
    PortfolioOpportunitiesResponseDTO,
)

log = logging.getLogger(__name__)


def _to_decimal(val: Any) -> Decimal:
    if val is None:
        return Decimal("0")
    if isinstance(val, Decimal):
        return val
    try:
        return Decimal(str(val))
    except Exception:
        return Decimal("0")


class PortfolioOpportunityService:
    """Application Service for 3-Dimensional Smart Opportunity Finder."""

    def __init__(
        self,
        portfolio_reader: PortfolioSnapshotReaderPort,
        fund_port: FundConstituentsReaderPort,
        equity_reader: EquityResearchReaderPort,
        policy_reader: ConfirmedPolicyReaderPort,
        macro_reader: MacroSnapshotReaderPort,
        watchlist_reader: WatchlistReaderPort,
        candidate_pool_service: Optional[CandidatePoolService] = None,
        look_through_service: Optional[Any] = None,
    ) -> None:
        self._portfolio_reader = portfolio_reader
        self._fund_port = fund_port
        self._equity_reader = equity_reader
        self._policy_reader = policy_reader
        self._macro_reader = macro_reader
        self._watchlist_reader = watchlist_reader

        self._candidate_pool_service = candidate_pool_service or CandidatePoolService(
            portfolio_reader=portfolio_reader,
            fund_port=fund_port,
            equity_reader=equity_reader,
            watchlist_reader=watchlist_reader,
        )
        self._look_through_service = look_through_service

        # Cache: portfolio_id -> {"timestamp": float, "dto": PortfolioOpportunitiesResponseDTO}
        self._cache: Dict[str, dict] = {}
        self._cache_ttl_seconds: float = 30.0

    def get_opportunities(
        self,
        portfolio_id: str = "default",
        force_refresh: bool = False,
    ) -> PortfolioOpportunitiesResponseDTO:
        """Evaluate and return ranked investment opportunities for a portfolio."""
        now = time.time()
        cached = self._cache.get(portfolio_id)
        if not force_refresh and cached:
            if now - cached["timestamp"] < self._cache_ttl_seconds:
                return cached["dto"]

        # 1. Fetch Portfolio State & NAV
        try:
            portfolio_state = self._portfolio_reader.get_current_portfolio_state(portfolio_id=portfolio_id)
        except Exception as e:
            log.warning("Could not load portfolio state for '%s': %s", portfolio_id, e)
            portfolio_state = None

        summary = getattr(portfolio_state, "summary", None)
        total_nav_thb = _to_decimal(getattr(summary, "market_value_thb", 0) or getattr(summary, "total_value_thb", 0))

        # Available cash
        cash_item = getattr(summary, "cash_breakdown", None) or {}
        available_cash_thb = _to_decimal(getattr(summary, "available_cash_thb", 0))
        if available_cash_thb <= Decimal("0") and isinstance(cash_item, dict):
            available_cash_thb = _to_decimal(cash_item.get("THB", 0))

        # Check holdings for cash if summary cash is zero
        if available_cash_thb <= Decimal("0") and portfolio_state and getattr(portfolio_state, "holdings", None):
            cash_sum = Decimal("0")
            for h in portfolio_state.holdings:
                asset_type = str(getattr(h, "asset_type", "")).upper()
                sym = str(getattr(h, "symbol", "")).upper()
                if asset_type == "CASH" or sym in ("CASH_THB", "CASH_USD", "CASH", "THB", "USD"):
                    val = _to_decimal(getattr(h, "market_value_thb", 0))
                    if val > Decimal("0"):
                        cash_sum += val
            if cash_sum > Decimal("0"):
                available_cash_thb = cash_sum

        # 2. Look-Through Report & Sector Exposures
        look_through_report = None
        look_through_coverage_pct = Decimal("100.0")
        sector_exposures_thb: Dict[str, Decimal] = {}

        if self._look_through_service:
            try:
                look_through_report = self._look_through_service.get_portfolio_look_through(portfolio_id=portfolio_id)
                if look_through_report:
                    if look_through_report.total_nav_thb > Decimal("0"):
                        total_nav_thb = look_through_report.total_nav_thb
                    for s in getattr(look_through_report, "sector_exposure", []):
                        sector_exposures_thb[s.sector_name] = s.total_value_thb
                    cov = getattr(look_through_report, "coverage", None)
                    if cov:
                        look_through_coverage_pct = cov.fund_fetch_coverage
            except Exception as e:
                log.warning("Look-through calculation error in opportunity service for %s: %s", portfolio_id, e)

        # 3. Confirmed Policy & Buckets
        confirmed_axis = None
        applied_buckets: List[Dict[str, Any]] = []
        try:
            confirmed_axis = self._policy_reader.get_confirmed_investment_axis(portfolio_id=portfolio_id)
            applied_buckets = self._policy_reader.get_applied_bucket_policy(portfolio_id=portfolio_id) or []
        except Exception as e:
            log.debug("Policy reader error for %s: %s", portfolio_id, e)

        # Supplement applied_buckets from portfolio_state if empty
        if not applied_buckets and portfolio_state:
            pol = getattr(portfolio_state, "allocation_policy", None)
            if pol and isinstance(pol, dict):
                pb = pol.get("purpose_buckets") or pol.get("buckets")
                if pb and isinstance(pb, list):
                    applied_buckets = pb
            if not applied_buckets and getattr(portfolio_state, "allocation_targets", None):
                applied_buckets = [
                    {
                        "id": getattr(t, "bucket_id", ""),
                        "name": getattr(t, "name", ""),
                        "target_pct": float(getattr(t, "target_percent", 0.0)),
                    }
                    for t in portfolio_state.allocation_targets
                ]

        # 4. Macro Rotation Snapshot
        sector_rotation_data = None
        try:
            sector_rotation_data = self._macro_reader.get_latest_sector_rotation()
        except Exception as e:
            log.debug("Macro sector rotation reader error: %s", e)

        # 5. Discover Candidates
        candidates = self._candidate_pool_service.discover_candidates(
            portfolio_id=portfolio_id,
            look_through_report=look_through_report,
        )

        # 6. Read Sidecars
        sidecars_map: Dict[str, Any] = {}
        try:
            sidecars_map = self._equity_reader.get_all_equity_sidecars()
        except Exception as e:
            log.warning("Could not load equity sidecars for opportunity evaluation: %s", e)

        # 6.1 Resolve USD/THB FX rate for ADTV conversion
        usd_thb_fx_rate: Optional[Decimal] = Decimal("35.50")
        raw_fx = getattr(summary, "usd_thb_rate", None)
        if raw_fx is None and portfolio_state is not None:
            fx_dict = getattr(portfolio_state, "fx_rates", None)
            if isinstance(fx_dict, dict):
                raw_fx = fx_dict.get("USDTHB") if "USDTHB" in fx_dict else fx_dict.get("USD_THB")
        if raw_fx is not None:
            try:
                fx_dec = raw_fx if isinstance(raw_fx, Decimal) else Decimal(str(raw_fx))
                if isinstance(raw_fx, bool) or not fx_dec.is_finite() or fx_dec <= Decimal("0"):
                    usd_thb_fx_rate = Decimal("-1")
                else:
                    usd_thb_fx_rate = fx_dec
            except (InvalidOperation, ValueError, TypeError):
                usd_thb_fx_rate = Decimal("-1")

        # 7. Pure Domain Evaluation
        report = OpportunityEvaluator.evaluate_all(
            candidates=candidates,
            total_nav_thb=total_nav_thb,
            available_cash_thb=available_cash_thb,
            sector_exposures_thb=sector_exposures_thb,
            portfolio_id=portfolio_id,
            confirmed_axis=confirmed_axis,
            applied_buckets=applied_buckets,
            sidecars_map=sidecars_map,
            sector_rotation_data=sector_rotation_data,
            single_stock_cap_pct=Decimal("15.0"),
            sector_cap_pct=Decimal("30.0"),
            look_through_coverage_pct=look_through_coverage_pct,
            usd_thb_fx_rate=usd_thb_fx_rate,
        )

        # 8. Convert to API DTO
        dto = self._convert_report_to_dto(report)

        # Cache result
        self._cache[portfolio_id] = {
            "timestamp": now,
            "dto": dto,
        }

        return dto

    @staticmethod
    def _convert_report_to_dto(report: OpportunityReport) -> PortfolioOpportunitiesResponseDTO:
        """Map pure domain OpportunityReport to Pydantic Response DTO."""
        candidate_dtos: List[CandidateEvaluationResultDTO] = []
        for r in report.candidates:
            c = r.candidate
            c_dto = OpportunityCandidateDTO(
                security_id=c.security_id,
                issuer_id=c.issuer_id,
                symbol=c.symbol,
                market=c.market,
                currency=c.currency,
                asset_type=c.asset_type,
                company_name=c.company_name,
                sector_name=c.sector_name,
                sector_name_th=c.sector_name_th,
                source_pools=c.source_pools,
                source_refs=c.source_refs,
                already_directly_held=c.already_directly_held,
                already_watchlisted=c.already_watchlisted,
                current_direct_value_thb=float(c.current_direct_value_thb),
                current_indirect_value_thb=float(c.current_indirect_value_thb),
                current_total_exposure_thb=float(c.current_total_exposure_thb),
                current_total_weight_pct=float(c.current_total_weight_pct),
            )

            r_dto = CandidateEvaluationResultDTO(
                candidate=c_dto,
                policy_fit=r.policy_fit,
                policy_reasons=r.policy_reasons,
                suggested_bucket_id=r.suggested_bucket_id,
                applied_bucket_name=r.applied_bucket_name,
                evidence_readiness=r.evidence_readiness,
                composite_score=r.composite_score,
                valuation_upside_pct=float(r.valuation_upside_pct) if r.valuation_upside_pct is not None else None,
                target_price=float(r.target_price) if r.target_price is not None else None,
                target_price_basis=r.target_price_basis,
                moat_rating=r.moat_rating,
                evidence_reasons=r.evidence_reasons,
                macro_alignment=r.macro_alignment,
                macro_stage=r.macro_stage,
                macro_reasons=r.macro_reasons,
                capacity_status=r.capacity_status,
                remaining_capacity_thb=float(r.remaining_capacity_thb) if r.remaining_capacity_thb is not None else None,
                issuer_room_thb=float(r.issuer_room_thb) if r.issuer_room_thb is not None else None,
                sector_room_thb=float(r.sector_room_thb) if r.sector_room_thb is not None else None,
                capacity_reasons=r.capacity_reasons,
                research_priority=r.research_priority,
                priority_reasons=r.priority_reasons,
                is_actionable=r.is_actionable,
                liquidity_status=r.liquidity_status,
                adtv_thb=float(r.adtv_thb) if r.adtv_thb is not None else None,
                liquidity_room_thb=float(r.liquidity_room_thb) if r.liquidity_room_thb is not None else None,
            )
            candidate_dtos.append(r_dto)

        return PortfolioOpportunitiesResponseDTO(
            portfolio_id=report.portfolio_id,
            evaluation_id=report.evaluation_id,
            evaluated_at=report.evaluated_at,
            readiness=report.readiness,
            total_candidates_scanned=report.total_candidates_scanned,
            passed_candidates_count=report.passed_candidates_count,
            candidates=candidate_dtos,
            warnings=report.warnings,
            missing_requirements=report.missing_requirements,
            snapshot_meta=report.snapshot_meta,
        )
