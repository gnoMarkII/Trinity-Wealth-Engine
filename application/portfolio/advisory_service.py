"""Portfolio Advisory Application Service.

Architecture Layer: application/portfolio/ (Use Cases & Orchestration)
Adheres strictly to Hexagonal Architecture (AST Rule S1):
- Depends exclusively on abstract protocol ports.
- Never imports concrete infrastructure or adapters directly.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Literal, Optional

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
from schemas.micro_quant_schemas import DeterministicScorecard, QuantSignals
from schemas.portfolio_policy_schemas import PortfolioPolicy
from tools.market.portfolio_action_engine import (
    PortfolioActionVerdict,
    build_action_portfolio_state,
    compute_portfolio_action,
)
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
        bucket_active_share_pct=(
            float(radar.bucket_active_share_pct)
            if getattr(radar, "bucket_active_share_pct", None) is not None
            else None
        ),
        bucket_active_share_status=getattr(radar, "bucket_active_share_status", "AVAILABLE"),
        risk_contributions_pct={k: float(v) for k, v in getattr(radar, "risk_contributions_pct", {}).items()},
        inverse_vol_weights_pct={k: float(v) for k, v in getattr(radar, "inverse_vol_weights_pct", {}).items()},
        total_portfolio_volatility_pct=(
            float(radar.total_portfolio_volatility_pct)
            if getattr(radar, "total_portfolio_volatility_pct", None) is not None
            else None
        ),
        risk_contribution_status=getattr(radar, "risk_contribution_status", "AVAILABLE"),
        volatility_estimated_symbols=list(getattr(radar, "volatility_estimated_symbols", [])),
        drawdown_pct=(
            float(radar.drawdown_pct)
            if getattr(radar, "drawdown_pct", None) is not None
            else None
        ),
        drawdown_posture_tier=getattr(radar, "drawdown_posture_tier", "NORMAL"),
        drawdown_status=getattr(radar, "drawdown_status", "UNAVAILABLE_UNKNOWN_PEAK"),
        drawdown_hurdle_adjustment=(
            float(radar.drawdown_hurdle_adjustment)
            if getattr(radar, "drawdown_hurdle_adjustment", None) is not None
            else 0.0
        ),
        drawdown_action_restrictions=list(getattr(radar, "drawdown_action_restrictions", [])),
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
                fractional_shares=float(h.lot_sizing.fractional_shares) if h.lot_sizing.fractional_shares is not None else None,
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
            conditional_buy_zone_min=float(h.conditional_buy_zone_min) if h.conditional_buy_zone_min is not None else None,
            conditional_buy_zone_max=float(h.conditional_buy_zone_max) if h.conditional_buy_zone_max is not None else None,
            core_conviction_score=float(h.core_conviction_score) if h.core_conviction_score is not None else None,
            business_conviction_score=float(h.business_conviction_score) if h.business_conviction_score is not None else None,
            price_stage=h.price_stage,
            live_upside_12m_pct=float(h.live_upside_12m_pct) if h.live_upside_12m_pct is not None else None,
            macro_alignment=h.macro_alignment,
            sector_quadrant=h.sector_quadrant,
            breached_falsifiers=h.breached_falsifiers,
            data_warnings=h.data_warnings,
            risk_contribution_pct=float(h.risk_contribution_pct) if getattr(h, "risk_contribution_pct", None) is not None else None,
            estimated_friction_bps=float(h.estimated_friction_bps) if getattr(h, "estimated_friction_bps", None) is not None else None,
            volatility_pct=float(h.volatility_pct) if getattr(h, "volatility_pct", None) is not None else None,
            volatility_source=getattr(h, "volatility_source", "unavailable"),
            is_volatility_estimated=getattr(h, "is_volatility_estimated", False),
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
        synthesis_input_hash=getattr(report, "synthesis_input_hash", ""),
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


def _is_cached_narrative_fresh(
    cached_record: Optional[AdvisorySidecarRecord],
    report: PortfolioAdvisoryQuantReport,
) -> bool:
    """Checks both portfolio composition hash and full synthesis input digest."""
    if not cached_record or not cached_record.narrative:
        return False
    if cached_record.narrative.portfolio_state_hash != report.portfolio_state_hash:
        return False
    expected_synth_hash = getattr(report, "synthesis_input_hash", "")
    if expected_synth_hash:
        cached_synth_hash = (
            getattr(cached_record.narrative, "synthesis_input_hash", "")
            or getattr(cached_record, "synthesis_input_hash", "")
        )
        return bool(cached_synth_hash) and cached_synth_hash == expected_synth_hash
    return True


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
        if _is_cached_narrative_fresh(cached_record, report):
            valid_narrative = cached_record.narrative  # type: ignore[union-attr]

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
            if _is_cached_narrative_fresh(cached_record, report):
                return cached_record.narrative  # type: ignore[union-attr, return-value]

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
        narrative.portfolio_state_hash = report.portfolio_state_hash
        narrative.synthesis_input_hash = getattr(report, "synthesis_input_hash", "")

        # Persist full advisory sidecar to Vault
        response_dto = _convert_report_to_dto(report, narrative=narrative, look_through_summary=lt_summary)
        sidecar_record = AdvisorySidecarRecord(
            evaluated_at=response_dto.evaluated_at,
            portfolio_state_hash=response_dto.portfolio_state_hash,
            synthesis_input_hash=response_dto.synthesis_input_hash,
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

    def evaluate_holding_action_policy(
        self,
        ticker: str,
        scorecard: DeterministicScorecard,
        quant_signals: QuantSignals,
        policy: Optional[PortfolioPolicy] = None,
        market: Optional[Literal["TH", "US"]] = None,
        current_price: Optional[float] = None,
        portfolio_id: str = "default",
        refresh_prices: bool = False,
    ) -> PortfolioActionVerdict:
        """Evaluates multi-constraint single-holding sizing policy against live portfolio state."""
        san_ticker = ticker.strip().upper()
        resolved_market: Literal["TH", "US"] = market or (
            "TH" if quant_signals.market == "TH" or san_ticker.endswith(".BK") else "US"
        )
        local_currency = "THB" if resolved_market == "TH" else "USD"
        if policy is None:
            return compute_portfolio_action(
                ticker=san_ticker,
                market=resolved_market,
                scorecard=scorecard,
                quant_signals=quant_signals,
                policy=None,
                portfolio_state=None,
                current_price=current_price,
            )

        portfolio_state = self._portfolio_reader.get_current_portfolio_state(
            refresh_prices=refresh_prices,
            portfolio_id=portfolio_id,
        )
        sidecars = self._sidecar_reader.get_all_equity_sidecars() or {}
        sector_map: dict[str, str] = {}
        for sc_sym, sc_data in sidecars.items():
            q = sc_data.get("quant_signals") if isinstance(sc_data, dict) else getattr(sc_data, "quant_signals", None)
            peer_sec = q.get("peer_sector") if isinstance(q, dict) else getattr(q, "peer_sector", None)
            if peer_sec:
                sector_map[str(sc_sym).strip().upper()] = str(peer_sec)

        effective_price = current_price
        if effective_price is None and portfolio_state is not None and not isinstance(portfolio_state, dict):
            base_target = san_ticker[:-3] if san_ticker.endswith(".BK") else san_ticker
            for h in getattr(portfolio_state, "holdings", None) or []:
                h_sym = str(getattr(h, "symbol", "") or "").strip().upper()
                h_base = h_sym[:-3] if h_sym.endswith(".BK") else h_sym
                if h_sym == san_ticker or h_base == base_target:
                    effective_price = (
                        getattr(h, "current_price_thb", None)
                        if resolved_market == "TH"
                        else getattr(h, "current_price_usd", None)
                    )
                    if effective_price and effective_price > 0:
                        break
                    effective_price = None
        if effective_price is None and quant_signals.raw_analysis_price and quant_signals.raw_analysis_price > 0:
            effective_price = float(quant_signals.raw_analysis_price)

        action_state, missing_fx_reason = build_action_portfolio_state(
            portfolio_state,
            market=resolved_market,
            target_ticker=san_ticker,
            target_sector=quant_signals.peer_sector,
            sector_map=sector_map,
        )
        if missing_fx_reason is not None or action_state is None:
            return PortfolioActionVerdict(
                ticker=san_ticker,
                portfolio_id=policy.portfolio_id,
                action="RESTRICTED_MISSING_FX",
                security_stance=scorecard.action_stance,
                currency=local_currency,
                action_reasoning=missing_fx_reason or "Missing valid FX rate for cross-currency action evaluation",
                status="unavailable",
            )

        return compute_portfolio_action(
            ticker=san_ticker,
            market=resolved_market,
            scorecard=scorecard,
            quant_signals=quant_signals,
            policy=policy,
            portfolio_state=action_state,
            current_price=effective_price,
        )
