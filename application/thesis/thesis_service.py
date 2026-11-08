"""Application use-case service for Thesis lifecycle & As-Of Research State.



Architecture Rule (Layer 2 - Application):

Orchestrates domain models (ThesisEvaluator, ThesisClaimRecord) and ports.

Strictly zero imports from infrastructure or external frameworks.

"""

from __future__ import annotations



from datetime import datetime, timezone

from decimal import Decimal

import logging

from typing import Any, Dict, List, Optional, Sequence



from core.provenance import is_available_as_of, normalize_utc_iso

from core.security import validate_ticker

from core.thesis.evaluator import ThesisEvaluator

from core.thesis.evidence_taxonomy import EvidenceTier

from core.thesis.models import (

    CatalystEvent,

    FalsifierRule,

    ThesisClaimRecord,

    ThesisExceptionRow,

)

from application.thesis.ports import (

    MarketObservationPort,

    ThesisExceptionStorePort,

    ThesisStorePort,

)



log = logging.getLogger(__name__)





class ThesisApplicationService:

    """Coordinates structured thesis lifecycle, As-Of history queries, and watchdog scans."""



    def __init__(

        self,

        store: ThesisStorePort,

        exception_store: Optional[ThesisExceptionStorePort] = None,

        market_observer: Optional[MarketObservationPort] = None,

    ) -> None:

        self._store = store

        self._exception_store = exception_store

        self._market_observer = market_observer



    def record_new_thesis(

        self,

        ticker: str,

        claim_statement: str,

        falsifiers: Sequence[FalsifierRule] = (),

        catalysts: Sequence[CatalystEvent] = (),

        evidence_tier: EvidenceTier = EvidenceTier.DOCUMENTED,

        author_role: str = "HUMAN_PM",

        as_of_utc: Optional[str] = None,

    ) -> ThesisClaimRecord:

        """Records initial version (v1) of an investment thesis claim."""

        clean_ticker = validate_ticker(ticker)

        now_utc = normalize_utc_iso(as_of_utc) if as_of_utc else datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        claim_id = f"TH-{clean_ticker}-v1"



        claim = ThesisClaimRecord.create(

            claim_id=claim_id,

            ticker=clean_ticker,

            version=1,

            as_of_utc=now_utc,

            claim_statement=claim_statement,

            falsifiers=falsifiers,

            linked_catalysts=catalysts,

            status="ACTIVE",

            evidence_tier=evidence_tier,

            author_role=author_role,

            prior_claim_id=None,

        )

        self._store.save_claim(claim)

        log.info("Recorded new thesis claim %s for %s", claim_id, clean_ticker)

        return claim



    def revise_thesis(

        self,

        ticker: str,

        claim_statement: str,

        falsifiers: Sequence[FalsifierRule] = (),

        catalysts: Sequence[CatalystEvent] = (),

        evidence_tier: EvidenceTier = EvidenceTier.DOCUMENTED,

        author_role: str = "HUMAN_PM",

        prior_claim_id: Optional[str] = None,

        as_of_utc: Optional[str] = None,

    ) -> ThesisClaimRecord:

        """Creates a new version (vN) in the append-only ledger linked to prior_claim_id."""

        clean_ticker = validate_ticker(ticker)

        history = self._store.load_claims(clean_ticker)

        new_version = (max([c.version for c in history], default=0) + 1)

        parent_id = prior_claim_id or (history[-1].claim_id if history else None)

        claim_id = f"TH-{clean_ticker}-v{new_version}"

        now_utc = normalize_utc_iso(as_of_utc) if as_of_utc else datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")



        # Archive active prior claim if present

        claim = ThesisClaimRecord.create(

            claim_id=claim_id,

            ticker=clean_ticker,

            version=new_version,

            as_of_utc=now_utc,

            claim_statement=claim_statement,

            falsifiers=falsifiers,

            linked_catalysts=catalysts,

            status="ACTIVE",

            evidence_tier=evidence_tier,

            author_role=author_role,

            prior_claim_id=parent_id,

        )

        self._store.save_claim(claim)

        log.info("Appended revised thesis claim %s (parent: %s) for %s", claim_id, parent_id, clean_ticker)

        return claim



    def get_active_thesis(self, ticker: str) -> Optional[ThesisClaimRecord]:

        """Loads latest active claim for the specified ticker."""

        return self._store.get_active_claim(validate_ticker(ticker))



    def get_thesis_as_of(self, ticker: str, as_of_utc: str) -> Optional[ThesisClaimRecord]:

        """Time-travel query: retrieves the active claim record effective as of the specified historical date."""

        history = self._store.load_claims(validate_ticker(ticker))

        eligible = [c for c in history if is_available_as_of(c.as_of_utc, as_of_utc)]

        if not eligible:

            return None

        # Sort by version descending and return latest applicable

        eligible.sort(key=lambda c: c.version, reverse=True)

        return eligible[0]



    def list_active_theses(self) -> List[ThesisClaimRecord]:

        """Returns all currently active thesis claims across the fund."""

        return self._store.list_all_active_claims()



    # R20 alias for backward-compatibility with callers/tools expecting list_all_theses

    def list_all_theses(self) -> List[ThesisClaimRecord]:

        """Alias for list_active_theses to satisfy callers/tools."""

        return self.list_active_theses()



    def get_thesis_coverage(self, portfolio_tickers: Sequence[str]) -> Dict[str, Any]:

        """Calculates thesis coverage metrics across given portfolio tickers."""

        clean_tickers = sorted(list({validate_ticker(t) for t in portfolio_tickers if t and str(t).strip()}))

        active_claims = self.list_active_theses()

        covered_tickers_set = {c.ticker.strip().upper() for c in active_claims}



        covered: List[str] = []

        uncovered: List[str] = []

        for t in clean_tickers:

            t_alt = t.replace(".BK", "") if t.endswith(".BK") else f"{t}.BK"

            if t in covered_tickers_set or t_alt in covered_tickers_set:

                covered.append(t)

            else:

                uncovered.append(t)



        total = len(clean_tickers)

        cov_count = len(covered)

        cov_pct = round((cov_count / total * 100.0), 1) if total > 0 else 0.0



        return {

            "total_holdings_count": total,

            "covered_count": cov_count,

            "uncovered_count": len(uncovered),

            "coverage_pct": cov_pct,

            "covered_tickers": covered,

            "uncovered_tickers": uncovered,

        }



    def run_watchdog_scan(

        self,

        ticker: Optional[str] = None,

        as_of_date_str: Optional[str] = None,

        observed_metrics_override: Optional[Dict[str, Decimal]] = None,

        observed_events_override: Optional[Dict[str, Dict[str, Any]]] = None,

        observed_metrics_by_ticker: Optional[Dict[str, Dict[str, Decimal]]] = None,

    ) -> List[ThesisExceptionRow]:

        """Scans active claims against market reality and records exceptions."""

        now_date = as_of_date_str or datetime.now(timezone.utc).strftime("%Y-%m-%d")

        now_utc = f"{now_date}T00:00:00Z" if as_of_date_str else datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        claims = [self.get_active_thesis(ticker)] if ticker else self.list_active_theses()

        valid_claims = [c for c in claims if c is not None]



        all_exceptions: List[ThesisExceptionRow] = []



        for claim in valid_claims:

            metrics: Dict[str, Decimal] = {}

            # Use overrides or query via market observation port
            if observed_metrics_by_ticker and claim.ticker in observed_metrics_by_ticker:
                metrics.update(observed_metrics_by_ticker[claim.ticker])
            if observed_metrics_override:
                for k, v in observed_metrics_override.items():
                    if k not in metrics:
                        metrics[k] = v

            if self._market_observer:
                for f in claim.falsifiers:
                    if f.metric_name not in metrics:
                        val = self._market_observer.get_financial_metric(claim.ticker, f.metric_name)
                        if val is not None:
                            try:
                                metrics[f.metric_name] = Decimal(str(val))
                            except Exception:
                                pass

            events: Dict[str, Dict[str, Any]] = {}
            if observed_events_override:
                events.update(observed_events_override)

            if self._market_observer:
                for cat in claim.linked_catalysts:
                    if cat.catalyst_id not in events:
                        sched = self._market_observer.get_upcoming_catalyst_date(claim.ticker, cat.event_name)
                        if sched:
                            events[cat.catalyst_id] = {"scheduled_date": sched}



            exceptions = ThesisEvaluator.evaluate_claim(

                claim=claim,

                observed_metrics=metrics,

                as_of_date_str=now_date,

                detected_at_utc=now_utc,

                observed_events=events,

            )



            for exc in exceptions:

                all_exceptions.append(exc)

                if self._exception_store:

                    try:

                        self._exception_store.insert_exception(exc)

                    except Exception as err:

                        log.warning("Could not persist exception %s: %s", exc.exception_id, err)



        return all_exceptions

