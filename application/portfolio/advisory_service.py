"""Portfolio Advisory Application Service.

Architecture Layer: application/portfolio/ (Use Cases & Orchestration)
Adheres strictly to Hexagonal Architecture (AST Rule S1):
- Depends exclusively on abstract protocol ports.
- Never imports concrete infrastructure or adapters directly.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

from application.portfolio.advisory_ports import (
    AdvisorySynthesizerPort,
    AdvisoryVaultPort,
    EquitySidecarReaderPort,
    MacroContextReaderPort,
    PortfolioStateReaderPort,
)
from core.portfolio_advisory.engine import evaluate_portfolio_advisory
from core.portfolio_advisory.llm_payload_builder import build_advisory_llm_payload
from core.portfolio_advisory.models import PortfolioAdvisoryQuantReport
from schemas.portfolio_advisory_schemas import (
    AdvisorySidecarRecord,
    DataAnomalyWarningDTO,
    DualCurrencyCashPlanDTO,
    HoldingAdvisoryDTO,
    LotSizingDTO,
    PortfolioAdvisoryNarrativeDTO,
    PortfolioAdvisoryResponseDTO,
    PortfolioRiskRadarDTO,
    QuarantinedHoldingDTO,
)

log = logging.getLogger(__name__)


def _convert_report_to_dto(
    report: PortfolioAdvisoryQuantReport,
    narrative: Optional[PortfolioAdvisoryNarrativeDTO] = None,
    look_through_summary: Optional[Any] = None,
) -> PortfolioAdvisoryResponseDTO:
    """Pure domain mapper from PortfolioAdvisoryQuantReport to PortfolioAdvisoryResponseDTO."""
    cash = report.cash_plan
    cash_dto = DualCurrencyCashPlanDTO(
        cash_thb_available=float(cash.cash_thb_available),
        cash_usd_available=float(cash.cash_usd_available),
        usd_thb_fx_rate=float(cash.usd_thb_fx_rate),
        immediate_deploy_thb=float(cash.immediate_deploy_thb),
        immediate_deploy_usd=float(cash.immediate_deploy_usd),
        conditional_limit_order_thb=float(cash.conditional_limit_order_thb),
        conditional_limit_order_usd=float(cash.conditional_limit_order_usd),
        thb_surplus_after_immediate=float(cash.thb_surplus_after_immediate),
        usd_surplus_after_immediate=float(cash.usd_surplus_after_immediate),
        fx_conversion_needed_thb=float(cash.fx_conversion_needed_thb),
    )

    radar = report.risk_radar
    radar_dto = PortfolioRiskRadarDTO(
        weighted_beta=float(radar.weighted_beta) if radar.weighted_beta is not None else None,
        sector_crowding={k: float(v) for k, v in radar.sector_crowding.items()},
        fx_exposure_pct={k: float(v) for k, v in radar.fx_exposure_pct.items()},
        purpose_bucket_drift={k: float(v) for k, v in radar.purpose_bucket_drift.items()},
        unassigned_holdings_value_thb=float(radar.unassigned_holdings_value_thb),
        unassigned_holdings_nav_pct=float(radar.unassigned_holdings_nav_pct),
        unassigned_drift_guard_active=radar.unassigned_drift_guard_active,
        hhi=float(radar.hhi) if radar.hhi is not None else None,
        effective_holdings_count=float(radar.effective_holdings_count) if radar.effective_holdings_count is not None else None,
        top_1_weight_pct=float(radar.top_1_weight_pct) if getattr(radar, "top_1_weight_pct", None) is not None else 0.0,
        top_3_weight_pct=float(radar.top_3_weight_pct) if getattr(radar, "top_3_weight_pct", None) is not None else 0.0,
        top_5_weight_pct=float(radar.top_5_weight_pct) if getattr(radar, "top_5_weight_pct", None) is not None else 0.0,
        concentration_risk_level=getattr(radar, "concentration_risk_level", "LOW") or "LOW",
    )

    holdings_dtos = []
    for h in report.holdings_advisory:
        lot_dto = None
        if h.lot_sizing:
            lot_dto = LotSizingDTO(
                market=h.lot_sizing.market,
                board_lot_shares=h.lot_sizing.board_lot_shares,
                odd_lot_shares=h.lot_sizing.odd_lot_shares,
                whole_shares=h.lot_sizing.whole_shares,
                fractional_shares=float(h.lot_sizing.fractional_shares) if h.lot_sizing.fractional_shares else None,
                amount_thb=float(h.lot_sizing.amount_thb),
                amount_native=float(h.lot_sizing.amount_native),
                estimated_execution_price_native=float(h.lot_sizing.estimated_execution_price_native),
                native_currency=h.lot_sizing.native_currency,
            )

        holdings_dtos.append(HoldingAdvisoryDTO(
            symbol=h.symbol,
            asset_name=h.asset_name,
            asset_type=h.asset_type,
            market=h.market,
            native_currency=h.native_currency,
            evaluation_track=h.evaluation_track.value,
            bucket_id=h.bucket_id,
            bucket_name=h.bucket_name,
            current_units=float(h.current_units),
            current_price_native=float(h.current_price_native),
            current_value_thb=float(h.current_value_thb),
            current_weight_pct=float(h.current_weight_pct),
            target_weight_pct=float(h.target_weight_pct),
            weight_drift_pct=float(h.weight_drift_pct),
            action_code=h.action_code.value,
            action_rationale=h.action_rationale,
            lot_sizing=lot_dto,
            is_immediate_execution=h.is_immediate_execution,
            conditional_buy_zone_min=float(h.conditional_buy_zone_min) if h.conditional_buy_zone_min else None,
            conditional_buy_zone_max=float(h.conditional_buy_zone_max) if h.conditional_buy_zone_max else None,
            core_conviction_score=float(h.core_conviction_score) if h.core_conviction_score else None,
            business_conviction_score=float(h.business_conviction_score) if h.business_conviction_score else None,
            price_stage=h.price_stage,
            live_upside_12m_pct=float(h.live_upside_12m_pct) if h.live_upside_12m_pct else None,
            macro_alignment=h.macro_alignment,
            sector_quadrant=h.sector_quadrant,
            breached_falsifiers=h.breached_falsifiers,
            data_warnings=h.data_warnings,
        ))

    quarantined_dtos = [
        QuarantinedHoldingDTO(
            symbol=q.symbol,
            reason=q.reason,
            guardrail_code=q.guardrail_code,
            current_value_thb=float(q.current_value_thb),
            details=q.details,
        )
        for q in report.quarantined_holdings
    ]

    anomaly_dtos = [
        DataAnomalyWarningDTO(
            guardrail_code=w.guardrail_code,
            ticker=w.ticker,
            message=w.message,
            severity=w.severity,
            remedy_hint=w.remedy_hint,
        )
        for w in report.anomaly_warnings
    ]

    return PortfolioAdvisoryResponseDTO(
        evaluated_at=report.evaluated_at,
        data_mode=report.data_mode.value,
        portfolio_state_hash=report.portfolio_state_hash,
        total_nav_thb=float(report.total_nav_thb),
        cash_plan=cash_dto,
        risk_radar=radar_dto,
        holdings_advisory=holdings_dtos,
        quarantined_holdings=quarantined_dtos,
        anomaly_warnings=anomaly_dtos,
        macro_regime_summary=report.macro_regime_summary,
        cached_narrative=narrative,
        look_through_summary=look_through_summary,
    )


class PortfolioAdvisoryService:
    """Hexagonal Application Service for 2-Tier Portfolio Advisory."""

    def __init__(
        self,
        portfolio_reader: PortfolioStateReaderPort,
        macro_reader: MacroContextReaderPort,
        sidecar_reader: EquitySidecarReaderPort,
        vault_port: AdvisoryVaultPort,
        synthesizer_port: AdvisorySynthesizerPort,
        look_through_service: Optional[Any] = None,
    ) -> None:
        self._portfolio_reader = portfolio_reader
        self._macro_reader = macro_reader
        self._sidecar_reader = sidecar_reader
        self._vault_port = vault_port
        self._synthesizer_port = synthesizer_port
        self._look_through_service = look_through_service

    def get_advisory_quant_report(
        self,
        refresh_prices: bool = False,
        portfolio_id: str = "default",
    ) -> PortfolioAdvisoryQuantReport:
        """Executes Tier 1 Deterministic Engine (< 50ms)."""
        portfolio_state = self._portfolio_reader.get_current_portfolio_state(
            refresh_prices=refresh_prices,
            portfolio_id=portfolio_id,
        )
        macro_strategy = self._macro_reader.get_latest_macro_strategy()
        sector_rotation = self._macro_reader.get_latest_sector_rotation()
        equity_sidecars = self._sidecar_reader.get_all_equity_sidecars()
        watchlist_sidecars = self._sidecar_reader.get_watchlist_sidecars()

        lt_summary = None
        if self._look_through_service:
            try:
                lt_summary = self._look_through_service.get_portfolio_look_through(portfolio_id=portfolio_id)
            except Exception as e:
                log.warning("Could not calculate look-through summary for %s: %s", portfolio_id, e)

        return evaluate_portfolio_advisory(
            portfolio_state=portfolio_state,
            macro_direction=macro_strategy,
            sector_rotation_snapshot=sector_rotation,
            equity_sidecars=equity_sidecars,
            watchlist_sidecars=watchlist_sidecars,
            look_through_summary=lt_summary,
        )

    def get_portfolio_advisory(
        self,
        refresh_prices: bool = False,
        portfolio_id: str = "default",
    ) -> PortfolioAdvisoryResponseDTO:
        """Retrieves Tier 1 report and attaches cached Tier 2 narrative if valid for current portfolio hash."""
        report = self.get_advisory_quant_report(refresh_prices=refresh_prices, portfolio_id=portfolio_id)

        # Check for cached narrative in vault
        cached_record = self._vault_port.load_latest_advisory_sidecar(portfolio_id=portfolio_id)
        valid_narrative = None
        if cached_record and cached_record.narrative:
            # Hash matching: Ensure narrative was synthesized against the EXACT current portfolio state
            if cached_record.narrative.portfolio_state_hash == report.portfolio_state_hash:
                valid_narrative = cached_record.narrative

        # Calculate look-through summary if service is wired
        lt_summary = None
        if self._look_through_service:
            try:
                lt_summary = self._look_through_service.get_portfolio_look_through_dto(portfolio_id=portfolio_id)
            except Exception as e:
                log.warning("Could not calculate look-through summary for %s: %s", portfolio_id, e)

        return _convert_report_to_dto(report, narrative=valid_narrative, look_through_summary=lt_summary)

    def synthesize_playbook(
        self,
        force_refresh: bool = False,
        portfolio_id: str = "default",
    ) -> PortfolioAdvisoryNarrativeDTO:
        """Executes Tier 2 AI Strategic Playbook Synthesis and persists to Vault."""
        report = self.get_advisory_quant_report(refresh_prices=False, portfolio_id=portfolio_id)

        # Cache hit check
        if not force_refresh:
            cached_record = self._vault_port.load_latest_advisory_sidecar(portfolio_id=portfolio_id)
            if (
                cached_record
                and cached_record.narrative
                and cached_record.narrative.portfolio_state_hash == report.portfolio_state_hash
            ):
                return cached_record.narrative

        # Calculate look-through summary and attach to report
        lt_summary = None
        if self._look_through_service:
            try:
                lt_report = self._look_through_service.get_portfolio_look_through(portfolio_id=portfolio_id)
                report.look_through_summary = lt_report  # type: ignore[attr-defined]
                lt_summary = self._look_through_service.get_portfolio_look_through_dto(portfolio_id=portfolio_id)
            except Exception as e:
                log.warning("Could not calculate look-through report for playbook synthesis: %s", e)

        # Build token-efficient sanitized payload
        payload_dict = build_advisory_llm_payload(report)
        payload_json = json.dumps(payload_dict, ensure_ascii=False)

        # Synthesize via port
        narrative = self._synthesizer_port.synthesize_playbook(payload_json, report)

        # Persist full advisory sidecar to Vault
        response_dto = _convert_report_to_dto(report, narrative=narrative, look_through_summary=lt_summary)
        sidecar_record = AdvisorySidecarRecord(
            evaluated_at=response_dto.evaluated_at,
            portfolio_state_hash=response_dto.portfolio_state_hash,
            data_mode=response_dto.data_mode,
            total_nav_thb=response_dto.total_nav_thb,
            cash_plan=response_dto.cash_plan,
            risk_radar=response_dto.risk_radar,
            holdings_advisory=response_dto.holdings_advisory,
            quarantined_holdings=response_dto.quarantined_holdings,
            anomaly_warnings=response_dto.anomaly_warnings,
            narrative=narrative,
            look_through_summary=lt_summary,
        )
        try:
            self._vault_port.save_advisory_sidecar(sidecar_record, portfolio_id=portfolio_id)
        except Exception as e:
            log.warning("Failed to save advisory sidecar: %s", e)

        return narrative
