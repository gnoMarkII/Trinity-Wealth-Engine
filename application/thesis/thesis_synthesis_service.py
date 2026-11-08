"""Application Service for Automated Thesis & Falsifiers Synthesis.

Architecture Rule (Layer 2 - Application):
Orchestrates conversion of research notes, valuation models, and portfolio holdings
into structured, verifiable ThesisClaimRecords with quantitative FalsifierRules and Catalysts.
Strictly zero imports from API, infrastructure, or external frameworks.
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Sequence

from core.security import validate_ticker
from core.thesis.evidence_taxonomy import EvidenceTier
from core.input_sanitizer import sanitize_untrusted_input
from core.thesis.claim_validator import (
    ClaimRecord,
    ClaimValidationResult,
    batch_validate_claims,
    validate_claim_evidence,
)
from core.thesis.models import (
    CatalystEvent,
    FalsifierRule,
    ThesisClaimRecord,
)
from core.portfolio_advisory.guardrails import resolve_holding_currency_and_market
from application.thesis.thesis_service import ThesisApplicationService

log = logging.getLogger(__name__)


class ThesisSynthesisApplicationService:
    """Automates synthesis of institutional investment theses and falsifier boundaries."""

    def __init__(
        self,
        thesis_service: ThesisApplicationService,
        sidecar_reader: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
        portfolio_reader: Optional[Callable[..., Any]] = None,
    ) -> None:
        self._thesis_service = thesis_service
        self._sidecar_reader = sidecar_reader
        self._portfolio_reader = portfolio_reader

    def synthesize_thesis_for_ticker(
        self,
        ticker: str,
        dry_run: bool = False,
        custom_stop_loss: Optional[float] = None,
        custom_growth_floor: Optional[float] = None,
        custom_margin_floor: Optional[float] = None,
        author_role: str = "HERMES_ADVISOR",
        portfolio_id: str = "default",
    ) -> Dict[str, Any]:
        """Synthesizes structured ThesisClaimRecord with verified quantitative falsifiers."""
        clean_ticker = validate_ticker(ticker)

        # 1. Check if an active thesis already exists
        existing = self._thesis_service.get_active_thesis(clean_ticker)
        if existing and existing.status == "ACTIVE":
            return {
                "status": "ALREADY_ACTIVE",
                "ticker": clean_ticker,
                "claim_id": existing.claim_id,
                "message": f"Active thesis {existing.claim_id} already exists for {clean_ticker}",
                "claim": existing.to_dict(),
            }

        # 2. Extract context from sidecar if available
        sidecar_data: Optional[Dict[str, Any]] = None
        if self._sidecar_reader:
            try:
                sidecar_data = self._sidecar_reader(clean_ticker)
            except Exception as exc:
                log.warning("Could not read sidecar for %s: %s", clean_ticker, exc)

        quant = (sidecar_data or {}).get("quant_signals") or {}
        company_name = quant.get("company_name") or clean_ticker
        market = str((sidecar_data or {}).get("market") or quant.get("market") or "").upper()
        is_th = "TH" in market or clean_ticker.endswith(".BK")
        unit_currency = "THB" if is_th else "USD"

        raw_price = quant.get("raw_analysis_price") or 0.0
        current_price = Decimal(str(raw_price)) if raw_price > 0 else Decimal("0")

        # Fallback to portfolio holdings if price missing from sidecar
        if current_price <= Decimal("0") and self._portfolio_reader:
            try:
                p_state = self._portfolio_reader(portfolio_id=portfolio_id, refresh_prices=False)
                for h in getattr(p_state, "holdings", []) or []:
                    sym = str(getattr(h, "symbol", "")).strip().upper()
                    if sym == clean_ticker or sym.replace(".BK", "") == clean_ticker.replace(".BK", ""):
                        _, native_curr = resolve_holding_currency_and_market(h)
                        if native_curr:
                            unit_currency = native_curr
                        p_val = getattr(h, "current_price_thb", None) if unit_currency == "THB" else getattr(h, "current_price_usd", None)
                        if p_val is None:
                            p_val = getattr(h, "current_price", None)
                        if p_val:
                            current_price = Decimal(str(p_val))
                            company_name = getattr(h, "company_name", None) or company_name
                        break
            except Exception as exc:
                log.warning("Could not resolve portfolio holding price for %s: %s", clean_ticker, exc)

        base_summary = str((sidecar_data or {}).get("base_case_summary") or "")
        sanitization = sanitize_untrusted_input(base_summary)
        if not sanitization.is_safe:
            log.warning("Untrusted instruction detected in sidecar for %s: %s", clean_ticker, [e.pattern_name for e in sanitization.exceptions])
            base_summary = sanitization.cleaned_text

        # 3. Determine Stop Loss threshold
        stop_loss: Optional[Decimal] = None
        if custom_stop_loss is not None:
            stop_loss = Decimal(str(custom_stop_loss))
        else:
            match_stop = re.search(r"ตัดขาดทุนที่\s*([\d\.,]+)", base_summary)
            if match_stop:
                try:
                    stop_loss = Decimal(match_stop.group(1).replace(",", ""))
                except Exception:
                    stop_loss = None
            if stop_loss is None or stop_loss <= Decimal("0"):
                dcf_bear = Decimal(str((quant.get("dcf_result") or {}).get("scenarios", {}).get("bear", {}).get("target_price", 0)))
                if dcf_bear > Decimal("0"):
                    stop_loss = dcf_bear
                elif current_price > Decimal("0"):
                    stop_loss = (current_price * Decimal("0.85")).quantize(Decimal("0.01"))

        # 4. Determine Growth Floor threshold
        growth_floor: Optional[Decimal] = None
        yoy_growth = quant.get("revenue_growth_yoy_pct")
        if custom_growth_floor is not None:
            growth_floor = Decimal(str(custom_growth_floor))
        elif yoy_growth is not None:
            growth_floor = max(Decimal("5.0"), Decimal(str(round(float(yoy_growth) * 0.70, 1))))

        # 5. Determine Margin Floor threshold (separating operating_margin vs fcf_margin)
        margin_floor: Optional[Decimal] = None
        margin_metric = "operating_margin"
        if custom_margin_floor is not None:
            margin_floor = Decimal(str(custom_margin_floor))
        else:
            match_margin = re.search(r"Margin\s*ไม่ให้ต่ำกว่า\s*([\d\.]+)%", base_summary)
            if match_margin:
                try:
                    margin_floor = Decimal(match_margin.group(1))
                except Exception:
                    margin_floor = None
            if margin_floor is None:
                op_margin = quant.get("operating_margin_pct") or quant.get("ebit_margin_pct")
                if op_margin is not None:
                    margin_metric = "operating_margin"
                    margin_floor = max(Decimal("5.0"), Decimal(str(round(float(op_margin) * 0.70, 1))))
                elif quant.get("fcf_margin_pct") is not None:
                    margin_metric = "fcf_margin"
                    margin_floor = max(Decimal("5.0"), Decimal(str(round(float(quant.get("fcf_margin_pct")) * 0.70, 1))))

        # 6. Formulate Falsifier Rules
        falsifiers: List[FalsifierRule] = []
        if stop_loss is not None and stop_loss > Decimal("0"):
            falsifiers.append(
                FalsifierRule(
                    falsifier_id=f"FAL-{clean_ticker}-PRICE-STOP",
                    metric_name="current_price",
                    operator="<",
                    threshold_value=stop_loss,
                    unit=unit_currency,
                    period_basis="DAILY",
                    invalidation_impact="FORCES_PORTFOLIO_TRIM",
                    narrative_explanation=f"Price falling below {unit_currency} {stop_loss} breaches stop-loss discipline.",
                )
            )

        if growth_floor is not None:
            falsifiers.append(
                FalsifierRule(
                    falsifier_id=f"FAL-{clean_ticker}-REV-GROWTH",
                    metric_name="revenue_growth_yoy",
                    operator="<",
                    threshold_value=growth_floor,
                    unit="PERCENT",
                    period_basis="QUARTERLY",
                    invalidation_impact="BREAKS_BUY_CASE",
                    narrative_explanation=f"Revenue growth decelerating below {growth_floor}% YoY indicates fundamental impairment.",
                )
            )

        if margin_floor is not None:
            falsifiers.append(
                FalsifierRule(
                    falsifier_id=f"FAL-{clean_ticker}-MARGIN-FLOOR",
                    metric_name=margin_metric,
                    operator="<",
                    threshold_value=margin_floor,
                    unit="PERCENT",
                    period_basis="QUARTERLY",
                    invalidation_impact="FORCES_PORTFOLIO_TRIM",
                    narrative_explanation=f"{margin_metric} compression below {margin_floor}% signals cost escalation.",
                )
            )

        # 7. Formulate Catalyst Event (Check real calendar if available)
        catalyst_date_str = str(quant.get("next_earnings_date") or quant.get("next_reporting_date") or "")
        if catalyst_date_str:
            catalysts: List[CatalystEvent] = [
                CatalystEvent(
                    catalyst_id=f"CAT-{clean_ticker}-EARNINGS",
                    event_name=f"{clean_ticker} Quarterly Earnings Release",
                    expected_date_str=catalyst_date_str,
                    confidence_window_days=7,
                    source_locator=f"{clean_ticker} IR Calendar",
                    status="PENDING",
                    direction="N/A",
                )
            ]
        else:
            catalysts: List[CatalystEvent] = [
                CatalystEvent(
                    catalyst_id=f"CAT-{clean_ticker}-DISCLOSURE",
                    event_name=f"{clean_ticker} Periodic Financial Disclosure",
                    expected_date_str="UNSCHEDULED",
                    confidence_window_days=14,
                    source_locator=f"{clean_ticker} SEC/SET Disclosures",
                    status="PENDING",
                    direction="N/A",
                )
            ]

        # 8. Fact extraction & Anti-Citation-Laundering Validation (Strict 100 bps tolerance)
        sidecar_dict = sidecar_data or {}
        source_artifact = sidecar_dict.get("source_artifact")
        raw_primary_facts = (
            source_artifact.get("verified_facts") or source_artifact.get("primary_facts")
            if isinstance(source_artifact, dict)
            else sidecar_dict.get("verified_source_facts")
        )
        artifact_sha = str(
            (source_artifact.get("content_sha256") or source_artifact.get("artifact_sha256") or "")
            if isinstance(source_artifact, dict)
            else (sidecar_dict.get("source_content_sha256") or "")
        ).strip()
        artifact_ticker = str(
            (source_artifact.get("ticker") or clean_ticker)
            if isinstance(source_artifact, dict)
            else clean_ticker
        ).strip().upper()

        raw_locator = str(
            (source_artifact.get("source_locator") or source_artifact.get("filing_locator") if isinstance(source_artifact, dict) else None)
            or sidecar_dict.get("filing_locator")
            or sidecar_dict.get("source_filing")
            or sidecar_dict.get("primary_source")
            or sidecar_dict.get("sec_accession")
            or ""
        ).strip()
        has_valid_artifact_hash = bool(re.fullmatch(r"[0-9a-fA-F]{64}", artifact_sha))
        has_independent_source = bool(
            raw_locator
            and has_valid_artifact_hash
            and artifact_ticker == clean_ticker
            and isinstance(raw_primary_facts, dict)
            and len(raw_primary_facts) > 0
        )
        primary_facts: Dict[str, Any] = dict(raw_primary_facts) if has_independent_source else {}
        source_citation = raw_locator if has_independent_source else f"Internal sidecar analysis for {clean_ticker} (unverified against primary regulatory filing)"

        claimed_margin_val = (
            quant.get("operating_margin_pct")
            if quant.get("operating_margin_pct") is not None
            else (
                quant.get("ebit_margin_pct")
                if quant.get("ebit_margin_pct") is not None
                else quant.get("fcf_margin_pct")
            )
        )

        # Build claim records to validate primary cited facts against disclosure
        claims_to_validate: List[ClaimRecord] = []
        if custom_growth_floor is None and yoy_growth is not None:
            claims_to_validate.append(
                ClaimRecord(
                    claim_id=f"CLM-{clean_ticker}-REV-GROWTH",
                    ticker=clean_ticker,
                    statement=f"{clean_ticker} revenue growth baseline is {yoy_growth}% YoY.",
                    metric_name="revenue_growth_yoy",
                    claimed_value=Decimal(str(yoy_growth)),
                    evidence_tier=EvidenceTier.DOCUMENTED if has_independent_source else EvidenceTier.INFERRED,
                    source_locator=source_citation,
                    tolerance_bps=100, # Strict 1% verification against disclosure
                )
            )
        if custom_margin_floor is None and claimed_margin_val is not None:
            claims_to_validate.append(
                ClaimRecord(
                    claim_id=f"CLM-{clean_ticker}-MARGIN-FLOOR",
                    ticker=clean_ticker,
                    statement=f"{clean_ticker} {margin_metric} baseline is {claimed_margin_val}%.",
                    metric_name=margin_metric,
                    claimed_value=Decimal(str(claimed_margin_val)),
                    evidence_tier=EvidenceTier.DOCUMENTED if has_independent_source else EvidenceTier.INFERRED,
                    source_locator=source_citation,
                    tolerance_bps=100,
                )
            )

        validation_results, all_claims_valid = batch_validate_claims(claims_to_validate, primary_facts)

        # Determine rigorous evidence tier
        if not sanitization.is_safe:
            resolved_evidence_tier = EvidenceTier.UNVERIFIED
        elif has_independent_source and not all_claims_valid:
            resolved_evidence_tier = EvidenceTier.UNVERIFIED
        elif (
            custom_growth_floor is not None
            or custom_margin_floor is not None
            or custom_stop_loss is not None
            or not has_independent_source
        ):
            resolved_evidence_tier = EvidenceTier.INFERRED
        elif all_claims_valid and len(claims_to_validate) > 0:
            resolved_evidence_tier = EvidenceTier.DOCUMENTED
        else:
            resolved_evidence_tier = EvidenceTier.INFERRED

        # 9. Investment Claim Statement
        thesis_statement = (
            base_summary.strip()
            if len(base_summary.strip()) > 20
            else f"Long position in {company_name} ({clean_ticker}) supported by structural secular growth and resilient margin profile."
        )

        now_utc = datetime.now(timezone.utc).isoformat()
        claim_record = ThesisClaimRecord.create(
            claim_id=f"TH-{clean_ticker}-v1",
            ticker=clean_ticker,
            version=1,
            as_of_utc=now_utc,
            claim_statement=thesis_statement,
            falsifiers=falsifiers,
            linked_catalysts=catalysts,
            status="ACTIVE",
            evidence_tier=resolved_evidence_tier,
            author_role=author_role,
        )

        if not dry_run:
            final_claim = self._thesis_service.record_new_thesis(
                ticker=clean_ticker,
                claim_statement=thesis_statement,
                falsifiers=falsifiers,
                catalysts=catalysts,
                evidence_tier=resolved_evidence_tier,
                author_role=author_role,
            )
        else:
            final_claim = claim_record

        res: Dict[str, Any] = {
            "status": "SYNTHESIZED",
            "ticker": clean_ticker,
            "dry_run": dry_run,
            "claim_id": final_claim.claim_id,
            "evidence_tier": resolved_evidence_tier.value,
            "is_untrusted_input_sanitized": not sanitization.is_safe,
            "claim_validations": [v.to_dict() for v in validation_results],
            "falsifiers": [f.to_dict() for f in falsifiers],
            "falsifiers_count": len(falsifiers),
            "catalysts": [c.to_dict() for c in catalysts],
            "catalysts_count": len(catalysts),
            "growth_floor_pct": float(growth_floor) if growth_floor is not None else None,
            "margin_floor_pct": float(margin_floor) if margin_floor is not None else None,
            "stop_loss": float(stop_loss) if stop_loss is not None else None,
            "claim": final_claim.to_dict(),
        }
        if not sanitization.is_safe:
            res["security_warning"] = "UNTRUSTED_INSTRUCTION_DETECTED"
            res["sanitization_exceptions"] = [e.pattern_name for e in sanitization.exceptions]
        return res

    def validate_thesis_claims(
        self,
        claims: Sequence[ClaimRecord],
        primary_source_facts: Dict[str, Any],
    ) -> tuple[List[ClaimValidationResult], bool]:
        """Validates quantitative claim batch against primary source facts."""
        return batch_validate_claims(claims, primary_source_facts)

    def batch_synthesize_portfolio_theses(
        self,
        portfolio_id: str = "default",
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Scans portfolio holdings and synthesizes theses for all uncovered assets."""
        if not self._portfolio_reader:
            return {"error": "portfolio_reader is not wired"}

        try:
            p_state = self._portfolio_reader(portfolio_id=portfolio_id, refresh_prices=False)
        except Exception as exc:
            return {"error": f"Failed to read portfolio state: {exc}"}

        raw_holdings = getattr(p_state, "holdings", []) or []
        portfolio_tickers = [
            str(getattr(h, "symbol", "")).strip().upper()
            for h in raw_holdings
            if getattr(h, "symbol", None) and "CASH" not in str(getattr(h, "symbol", "")).upper()
        ]

        if not portfolio_tickers:
            return {
                "portfolio_id": portfolio_id,
                "message": "No active non-cash holdings in portfolio",
                "prior_coverage_pct": 0.0,
                "newly_synthesized_count": 0,
                "uncovered_remaining": [],
            }

        coverage = self._thesis_service.get_thesis_coverage(portfolio_tickers)
        uncovered = coverage.get("uncovered_tickers", [])

        synthesized_results: List[Dict[str, Any]] = []
        errors: List[Dict[str, str]] = []

        for ticker in uncovered:
            try:
                res = self.synthesize_thesis_for_ticker(
                    ticker=ticker,
                    dry_run=dry_run,
                    author_role="HERMES_ADVISOR",
                    portfolio_id=portfolio_id,
                )
                synthesized_results.append(res)
            except Exception as err:
                log.exception("Error synthesizing thesis for %s: %s", ticker, err)
                errors.append({"ticker": ticker, "error": str(err)})

        new_coverage = self._thesis_service.get_thesis_coverage(portfolio_tickers) if not dry_run else coverage

        return {
            "portfolio_id": portfolio_id,
            "total_holdings": len(portfolio_tickers),
            "prior_coverage_pct": coverage.get("coverage_pct", 0.0),
            "prior_covered_count": coverage.get("covered_count", 0),
            "newly_synthesized_count": len(synthesized_results),
            "new_coverage_pct": new_coverage.get("coverage_pct", 0.0),
            "newly_synthesized": synthesized_results,
            "errors": errors,
        }
