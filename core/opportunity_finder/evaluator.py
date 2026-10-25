"""Opportunity Finder Pure Domain Evaluator.

Architecture Layer: core/opportunity_finder/ (Pure Domain)
Combines:
1. Macro Sector Alignment (Leading/Improving/Weakening/Lagging)
2. Investor Principles / Axis Exclusions & Bucket Fit
3. Equity Sidecar Quant & Valuation Evidence
4. Portfolio Look-Through Capacity
Produces deterministic research priority (1 to 4) with stable ties.
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from core.opportunity_finder.capacity import calculate_candidate_capacity
from core.opportunity_finder.models import (
    CandidateEvaluationResult,
    EvidenceReadiness,
    MacroAlignment,
    OpportunityCandidate,
    OpportunityReport,
    ReportReadiness,
)
from core.opportunity_finder.policy import evaluate_policy_admissibility
from core.portfolio.sector_taxonomy import normalize_sector_name


def _to_decimal(val: Any) -> Optional[Decimal]:
    if val is None:
        return None
    try:
        return Decimal(str(val))
    except Exception:
        return None


class OpportunityEvaluator:
    """Pure domain orchestrator for candidate evaluation."""

    @staticmethod
    def evaluate_candidate(
        candidate: OpportunityCandidate,
        total_nav_thb: Decimal,
        available_cash_thb: Decimal,
        current_sector_exposure_thb: Decimal,
        confirmed_axis: Optional[Dict[str, Any]] = None,
        applied_buckets: Optional[List[Dict[str, Any]]] = None,
        sidecar_data: Optional[Dict[str, Any]] = None,
        sector_rotation_data: Optional[Dict[str, Any]] = None,
        single_stock_cap_pct: Decimal = Decimal("15.0"),
        sector_cap_pct: Decimal = Decimal("30.0"),
        look_through_coverage_pct: Decimal = Decimal("100.0"),
    ) -> CandidateEvaluationResult:
        """Run 3-dimensional evaluation on a single candidate."""

        # 1. Dimension 1: Investor Principles / Axis Fit
        policy_fit, policy_reasons, suggested_bucket_id, bucket_name = evaluate_policy_admissibility(
            candidate=candidate,
            confirmed_axis=confirmed_axis,
            applied_buckets=applied_buckets,
            sidecar_data=sidecar_data,
        )

        # 2. Dimension 2: Equity Research Evidence Readiness
        evidence_readiness: EvidenceReadiness = "unavailable"
        composite_score: Optional[float] = None
        valuation_upside_pct: Optional[Decimal] = None
        target_price: Optional[Decimal] = None
        target_price_basis: Optional[str] = None
        moat_rating: Optional[str] = None
        evidence_reasons: List[str] = []

        if sidecar_data and isinstance(sidecar_data, dict):
            quant = sidecar_data.get("quant_signals")
            if not isinstance(quant, dict):
                quant = {}
            val_workbench = sidecar_data.get("valuation_workbench")
            if not isinstance(val_workbench, dict):
                val_workbench = {}
            narrative = sidecar_data.get("narrative_analysis")
            if not isinstance(narrative, dict):
                narrative = {}

            raw_score = quant.get("composite_score") or sidecar_data.get("composite_score")
            if raw_score is not None:
                try:
                    composite_score = round(float(raw_score), 1)
                except (ValueError, TypeError):
                    composite_score = None

            reverse_dcf = quant.get("reverse_dcf_result") if isinstance(quant.get("reverse_dcf_result"), dict) else {}
            dcf_res = quant.get("dcf_result") if isinstance(quant.get("dcf_result"), dict) else {}

            raw_upside = (
                val_workbench.get("upside_pct")
                or quant.get("upside_pct")
                or reverse_dcf.get("upside_12m_pct")
                or dcf_res.get("upside_12m_pct")
                or dcf_res.get("upside_pct")
            )
            valuation_upside_pct = _to_decimal(raw_upside)

            raw_target = (
                val_workbench.get("target_price")
                or quant.get("target_price")
                or reverse_dcf.get("target_price_12m")
                or reverse_dcf.get("intrinsic_value_today")
                or dcf_res.get("target_price_12m")
                or dcf_res.get("fair_value_per_share")
                or dcf_res.get("target_price")
            )
            target_price = _to_decimal(raw_target)
            target_price_basis = (
                val_workbench.get("valuation_method")
                or reverse_dcf.get("actionability_reason")
                or "Base Case DCF / Multiples"
            )

            scorecard = quant.get("scorecard") if isinstance(quant.get("scorecard"), dict) else {}
            moat_rating = (
                quant.get("moat_rating")
                or scorecard.get("moat_rating")
                or quant.get("economic_moat")
                or sidecar_data.get("moat_rating")
            )
            if not moat_rating and isinstance(narrative, dict):
                moat_rating = narrative.get("moat_rating")
            if not moat_rating:
                moat_rating = "None"

            # Check Freshness
            analysis_date = str(sidecar_data.get("analysis_date") or "").strip()
            is_stale_flag = (
                sidecar_data.get("is_stale") is True
                or sidecar_data.get("status") == "stale"
                or "stale" in str(sidecar_data.get("freshness_status", "")).lower()
                or "stale" in str(quant.get("data_quality_flags", [])).lower()
            )
            if analysis_date:
                # Check date age if valid ISO/date string
                is_outdated = False
                try:
                    from datetime import datetime, timezone
                    dt = datetime.strptime(analysis_date[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
                    now_utc = datetime.now(timezone.utc)
                    if (now_utc - dt).days > 365:
                        is_outdated = True
                except Exception:
                    pass

                if is_stale_flag or is_outdated:
                    evidence_readiness = "stale"
                    evidence_reasons.append(
                        f"Equity Sidecar is stale (Score: {composite_score or 'N/A'}, Date: {analysis_date})."
                    )
                else:
                    evidence_readiness = "sufficient"
                    evidence_reasons.append(
                        f"Active Equity Sidecar found (Score: {composite_score or 'N/A'}, "
                        f"Upside: {valuation_upside_pct or 0.0:.1f}%, Date: {analysis_date})."
                    )
            else:
                evidence_readiness = "partial"
                evidence_reasons.append("Equity Sidecar present but lacks verified analysis date.")
        else:
            evidence_readiness = "unavailable"
            evidence_reasons.append("No active Equity Sidecar in system; quantitative research required.")

        # 3. Dimension 3: Macro / Sector Alignment
        macro_alignment: MacroAlignment = "neutral"
        macro_stage: Optional[str] = None
        macro_reasons: List[str] = []

        if sector_rotation_data and isinstance(sector_rotation_data, dict):
            # Check quadrant from sector rotation snapshot
            # Supports both {"sectors": {sector_name: {"quadrant": "Leading", ...}}} and {"rows": [{ticker, name, quadrant, ...}]}
            sec_name_l = candidate.sector_name.lower()
            matched_stage = None

            # A. Check sectors dict
            sectors_map = sector_rotation_data.get("sectors")
            if isinstance(sectors_map, dict):
                for s_k, s_v in sectors_map.items():
                    norm_k, _ = normalize_sector_name(str(s_k))
                    if norm_k.lower() == sec_name_l or str(s_k).strip().lower() == sec_name_l:
                        if isinstance(s_v, dict):
                            matched_stage = str(s_v.get("quadrant") or s_v.get("stage") or "")
                        else:
                            matched_stage = str(s_v)
                        break

            # B. Check rows list from SectorRotationService snapshot
            if not matched_stage and "rows" in sector_rotation_data and isinstance(sector_rotation_data["rows"], list):
                for row in sector_rotation_data["rows"]:
                    if isinstance(row, dict):
                        row_name = str(row.get("name") or row.get("sector") or row.get("ticker") or "")
                        norm_row, _ = normalize_sector_name(row_name)
                        if norm_row.lower() == sec_name_l or row_name.strip().lower() == sec_name_l:
                            matched_stage = str(row.get("quadrant") or row.get("stage") or "")
                            break

            if matched_stage:
                macro_stage = matched_stage
                stage_l = matched_stage.lower()
                if "leading" in stage_l or "improving" in stage_l:
                    macro_alignment = "tailwind"
                    macro_reasons.append(f"Macro Tailwind: {candidate.sector_name} is in '{matched_stage}' rotation phase.")
                elif "lagging" in stage_l:
                    macro_alignment = "headwind"
                    macro_reasons.append(f"Macro Headwind: {candidate.sector_name} is in '{matched_stage}' rotation phase.")
                else:
                    macro_alignment = "neutral"
                    macro_reasons.append(f"Macro Neutral: {candidate.sector_name} is in '{matched_stage}' rotation phase.")
            else:
                macro_alignment = "neutral"
                macro_reasons.append(f"Sector '{candidate.sector_name}' has no active rotation signal; treated as neutral.")
        else:
            macro_alignment = "unavailable"
            macro_reasons.append("Sector rotation snapshot unavailable.")

        # 4. Dimension 4: Capacity & Exposure Guard
        cap_status, usable_capacity, issuer_room, sector_room, cap_reasons = calculate_candidate_capacity(
            candidate=candidate,
            total_nav_thb=total_nav_thb,
            available_cash_thb=available_cash_thb,
            current_sector_exposure_thb=current_sector_exposure_thb,
            single_stock_cap_pct=single_stock_cap_pct,
            sector_cap_pct=sector_cap_pct,
            coverage_pct=look_through_coverage_pct,
        )

        # 5. Deterministic Research Priority Scoring
        priority = 3
        priority_reasons: List[str] = []

        if policy_fit == "fail":
            priority = 4
            priority_reasons.append("Policy Fit Failed: Excluded by confirmed investor constraints.")
        elif policy_fit == "needs_review":
            priority = 4
            priority_reasons.append("Policy Fit Unconfirmed: Needs Investor Axis review.")
        elif cap_status == "constrained":
            priority = 3
            priority_reasons.append("Capacity Constrained: Portfolio room exhausted for single-stock/sector.")
        elif evidence_readiness == "sufficient" and (composite_score or 0) >= 70.0 and macro_alignment == "tailwind":
            priority = 1
            priority_reasons.append("High Conviction: Strong Quant score (>= 70), Macro tailwind, and Portfolio capacity available.")
        elif evidence_readiness in ("sufficient", "partial") and (composite_score is None or composite_score >= 50.0) and macro_alignment in ("tailwind", "neutral"):
            priority = 2
            priority_reasons.append("Secondary Opportunity: Favorable alignment and available portfolio capacity.")
        else:
            priority = 3
            priority_reasons.append("Watch / Research Required: Missing quant evidence, low quant score (< 50), or neutral/headwind macro stance.")

        is_actionable = (policy_fit == "pass") and (cap_status == "available") and (priority in (1, 2))

        return CandidateEvaluationResult(
            candidate=candidate,
            policy_fit=policy_fit,
            policy_reasons=policy_reasons,
            suggested_bucket_id=suggested_bucket_id,
            applied_bucket_name=bucket_name,
            evidence_readiness=evidence_readiness,
            composite_score=composite_score,
            valuation_upside_pct=valuation_upside_pct,
            target_price=target_price,
            target_price_basis=target_price_basis,
            moat_rating=moat_rating,
            evidence_reasons=evidence_reasons,
            macro_alignment=macro_alignment,
            macro_stage=macro_stage,
            macro_reasons=macro_reasons,
            capacity_status=cap_status,
            remaining_capacity_thb=usable_capacity,
            issuer_room_thb=issuer_room,
            sector_room_thb=sector_room,
            capacity_reasons=cap_reasons,
            research_priority=priority,
            priority_reasons=priority_reasons,
            is_actionable=is_actionable,
        )

    @classmethod
    def evaluate_all(
        cls,
        candidates: List[OpportunityCandidate],
        total_nav_thb: Decimal,
        available_cash_thb: Decimal,
        sector_exposures_thb: Dict[str, Decimal],
        portfolio_id: str = "default",
        evaluation_id: str = "",
        confirmed_axis: Optional[Dict[str, Any]] = None,
        applied_buckets: Optional[List[Dict[str, Any]]] = None,
        sidecars_map: Optional[Dict[str, Any]] = None,
        sector_rotation_data: Optional[Dict[str, Any]] = None,
        single_stock_cap_pct: Decimal = Decimal("15.0"),
        sector_cap_pct: Decimal = Decimal("30.0"),
        look_through_coverage_pct: Decimal = Decimal("100.0"),
    ) -> OpportunityReport:
        """Batch evaluate all candidates and build sorted OpportunityReport."""
        now_iso = datetime.now(timezone.utc).isoformat()
        sidecars = sidecars_map or {}

        results: List[CandidateEvaluationResult] = []
        warnings: List[str] = []
        missing_requirements: List[str] = []

        if not confirmed_axis:
            missing_requirements.append("Confirmed Investor Axis not found for portfolio.")

        if not sector_rotation_data:
            warnings.append("Macro Sector Rotation data unavailable; Macro alignment evaluated as neutral.")

        if look_through_coverage_pct < Decimal("60.0"):
            warnings.append(
                f"Fund Look-Through coverage is incomplete ({look_through_coverage_pct:.1f}%); "
                "capacity checks operate under uncertainty."
            )

        for cand in candidates:
            sec_exp = sector_exposures_thb.get(cand.sector_name, Decimal("0"))
            sidecar = sidecars.get(cand.symbol) or sidecars.get(cand.symbol.replace(".BK", ""))

            res = cls.evaluate_candidate(
                candidate=cand,
                total_nav_thb=total_nav_thb,
                available_cash_thb=available_cash_thb,
                current_sector_exposure_thb=sec_exp,
                confirmed_axis=confirmed_axis,
                applied_buckets=applied_buckets,
                sidecar_data=sidecar,
                sector_rotation_data=sector_rotation_data,
                single_stock_cap_pct=single_stock_cap_pct,
                sector_cap_pct=sector_cap_pct,
                look_through_coverage_pct=look_through_coverage_pct,
            )
            results.append(res)

        # Stable deterministic sort: Priority ascending (1 -> 4), composite_score desc, upside desc, symbol asc
        results.sort(
            key=lambda r: (
                r.research_priority,
                -(r.composite_score if r.composite_score is not None else -999.0),
                -(float(r.valuation_upside_pct) if r.valuation_upside_pct is not None else -999.0),
                r.candidate.symbol,
            )
        )

        passed_count = sum(1 for r in results if r.policy_fit == "pass")

        # Report readiness
        readiness: ReportReadiness = "ready"
        if not confirmed_axis or not candidates:
            readiness = "degraded"

        return OpportunityReport(
            portfolio_id=portfolio_id,
            evaluation_id=evaluation_id or f"opp_{now_iso[:19].replace(':', '')}",
            evaluated_at=now_iso,
            readiness=readiness,
            candidates=results,
            warnings=warnings,
            missing_requirements=missing_requirements,
            total_candidates_scanned=len(candidates),
            passed_candidates_count=passed_count,
            snapshot_meta={
                "nav_thb": float(total_nav_thb),
                "cash_thb": float(available_cash_thb),
                "single_stock_cap_pct": float(single_stock_cap_pct),
                "sector_cap_pct": float(sector_cap_pct),
                "look_through_coverage_pct": float(look_through_coverage_pct),
            },
        )
