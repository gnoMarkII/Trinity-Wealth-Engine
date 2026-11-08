"""Deterministic Evaluator for Falsifier Rules & Catalyst Drift.

Architecture Rule (Layer 1 - Core):
Pure Python standard library only — strictly zero external dependencies (no FastAPI, Pydantic,
SQLite, or yfinance) per Hexagonal Architecture AST Domain Layer Isolation rules.
"""
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional
import hashlib
import uuid

from core.thesis.models import (
    CatalystEvent,
    FalsifierRule,
    ThesisClaimRecord,
    ThesisExceptionRow,
)


class ThesisEvaluator:
    """Pure domain engine that evaluates thesis falsifier rules and catalyst timelines."""

    @staticmethod
    def _parse_date(date_str: str) -> Optional[datetime]:
        """Safely parse ISO date strings YYYY-MM-DD."""
        try:
            clean = date_str.strip().split("T")[0]
            return datetime.strptime(clean, "%Y-%m-%d")
        except (ValueError, TypeError, AttributeError):
            return None

    @classmethod
    def evaluate_falsifier(
        cls,
        claim: ThesisClaimRecord,
        falsifier: FalsifierRule,
        observed_value: Decimal,
        detected_at_utc: str,
        source_citation: Optional[str] = None,
    ) -> Optional[ThesisExceptionRow]:
        """Evaluates whether an observed metric breaches a specific falsifier rule."""
        breached = False
        op = falsifier.operator.strip()
        thresh = falsifier.threshold_value

        try:
            if op == "<":
                breached = observed_value < thresh
            elif op == "<=":
                breached = observed_value <= thresh
            elif op == ">":
                breached = observed_value > thresh
            elif op == ">=":
                breached = observed_value >= thresh
            elif op == "==":
                breached = observed_value == thresh
            elif op == "!=":
                breached = observed_value != thresh
            elif op == "pct_drop_exceeds":
                # Expects observed_value to represent the actual drop percentage
                breached = observed_value >= thresh
            else:
                breached = False
        except (TypeError, InvalidOperation):
            breached = False

        if breached:
            f_key = falsifier.falsifier_id or falsifier.metric_name
            d_key = detected_at_utc[:10] if detected_at_utc else datetime.now(timezone.utc).strftime("%Y-%m-%d")
            hash_key = f"{claim.ticker}:{claim.claim_id}:{f_key}:{d_key}:{observed_value}"
            ex_hash = hashlib.sha256(hash_key.encode("utf-8")).hexdigest()[:8].upper()
            return ThesisExceptionRow(
                exception_id=f"EXC-{ex_hash}",
                claim_id=claim.claim_id,
                ticker=claim.ticker,
                detected_at_utc=detected_at_utc,
                exception_type="FALSIFIER_BREACHED",
                metric_name=falsifier.metric_name,
                expected_value=f"{op} {thresh} {falsifier.unit}",
                observed_value=f"{observed_value} {falsifier.unit}",
                source_citation=source_citation or "Observed market data / filing",
                delta_impact=falsifier.invalidation_impact,
                human_action="REVIEW_REQUIRED",
                status="PENDING",
            )
        return None

    @classmethod
    def evaluate_catalyst(
        cls,
        claim: ThesisClaimRecord,
        catalyst: CatalystEvent,
        as_of_date_str: str,
        detected_at_utc: str,
        observed_scheduled_date: Optional[str] = None,
        observed_source_locator: Optional[str] = None,
    ) -> Optional[ThesisExceptionRow]:
        """Evaluates calendar drift, delay, or stale status of a catalyst event."""
        # 1. Date changed / Postponed / Rescheduled
        if observed_scheduled_date and observed_scheduled_date.strip() != catalyst.expected_date_str.strip():
            c_key = catalyst.catalyst_id or catalyst.event_name
            d_key = as_of_date_str[:10] if as_of_date_str else datetime.now(timezone.utc).strftime("%Y-%m-%d")
            hash_key = f"{claim.ticker}:{claim.claim_id}:{c_key}:DATE_CHANGED:{observed_scheduled_date.strip()}:{d_key}"
            ex_hash = hashlib.sha256(hash_key.encode("utf-8")).hexdigest()[:8].upper()
            return ThesisExceptionRow(
                exception_id=f"EXC-{ex_hash}",
                claim_id=claim.claim_id,
                ticker=claim.ticker,
                detected_at_utc=detected_at_utc,
                exception_type="DATE_CHANGED",
                metric_name=catalyst.event_name,
                expected_value=catalyst.expected_date_str,
                observed_value=observed_scheduled_date.strip(),
                source_citation=observed_source_locator or catalyst.source_locator,
                delta_impact="TIMELINE_DRIFT",
                human_action="REVIEW_REQUIRED",
                status="PENDING",
            )

        # 2. Stale detection (expected date passed by more than confidence window with status still PENDING)
        if catalyst.status.upper() == "PENDING":
            exp_dt = cls._parse_date(catalyst.expected_date_str)
            as_of_dt = cls._parse_date(as_of_date_str)
            if exp_dt and as_of_dt:
                deadline = exp_dt + timedelta(days=catalyst.confidence_window_days)
                if as_of_dt > deadline:
                    days_overdue = (as_of_dt - exp_dt).days
                    c_key = catalyst.catalyst_id or catalyst.event_name
                    d_key = as_of_date_str[:10] if as_of_date_str else datetime.now(timezone.utc).strftime("%Y-%m-%d")
                    hash_key = f"{claim.ticker}:{claim.claim_id}:{c_key}:STALE:{d_key}"
                    ex_hash = hashlib.sha256(hash_key.encode("utf-8")).hexdigest()[:8].upper()
                    return ThesisExceptionRow(
                        exception_id=f"EXC-{ex_hash}",
                        claim_id=claim.claim_id,
                        ticker=claim.ticker,
                        detected_at_utc=detected_at_utc,
                        exception_type="STALE",
                        metric_name=catalyst.event_name,
                        expected_value=f"{catalyst.expected_date_str} (Window: +{catalyst.confidence_window_days}d)",
                        observed_value=f"Overdue by {days_overdue} days without filing update",
                        source_citation=catalyst.source_locator,
                        delta_impact="STALE_CATALYST_RISK",
                        human_action="REVIEW_REQUIRED",
                        status="PENDING",
                    )

        # 3. New source detected
        if observed_source_locator and catalyst.source_locator and observed_source_locator != catalyst.source_locator:
            c_key = catalyst.catalyst_id or catalyst.event_name
            d_key = as_of_date_str[:10] if as_of_date_str else datetime.now(timezone.utc).strftime("%Y-%m-%d")
            hash_key = f"{claim.ticker}:{claim.claim_id}:{c_key}:NEW_SOURCE:{observed_source_locator}:{d_key}"
            ex_hash = hashlib.sha256(hash_key.encode("utf-8")).hexdigest()[:8].upper()
            return ThesisExceptionRow(
                exception_id=f"EXC-{ex_hash}",
                claim_id=claim.claim_id,
                ticker=claim.ticker,
                detected_at_utc=detected_at_utc,
                exception_type="NEW_SOURCE",
                metric_name=catalyst.event_name,
                expected_value=catalyst.source_locator,
                observed_value=observed_source_locator,
                source_citation=observed_source_locator,
                delta_impact="NEW_EVIDENCE_SOURCE",
                human_action="REVIEW_REQUIRED",
                status="PENDING",
            )

        return None

    @classmethod
    def evaluate_claim(
        cls,
        claim: ThesisClaimRecord,
        observed_metrics: Dict[str, Decimal],
        as_of_date_str: str,
        detected_at_utc: Optional[str] = None,
        observed_events: Optional[Dict[str, Dict[str, Any]]] = None,
        source_citations: Optional[Dict[str, str]] = None,
    ) -> List[ThesisExceptionRow]:
        """Runs comprehensive scan across all falsifiers and catalysts for an active claim."""
        now_utc = detected_at_utc or datetime.now(timezone.utc).isoformat()
        citations = source_citations or {}
        events_map = observed_events or {}
        exceptions: List[ThesisExceptionRow] = []

        # 1. Evaluate Falsifiers
        for falsifier in claim.falsifiers:
            m_name = falsifier.metric_name
            if m_name in observed_metrics:
                obs_val = observed_metrics[m_name]
                src = citations.get(m_name)
                exc = cls.evaluate_falsifier(
                    claim=claim,
                    falsifier=falsifier,
                    observed_value=obs_val,
                    detected_at_utc=now_utc,
                    source_citation=src,
                )
                if exc:
                    exceptions.append(exc)
            else:
                # Missing required evidence is an explicit review condition, never an implicit pass.
                f_key = falsifier.falsifier_id or m_name
                d_key = as_of_date_str[:10] if as_of_date_str else now_utc[:10]
                hash_key = f"{claim.ticker}:{claim.claim_id}:{f_key}:NOT_EVALUABLE:{d_key}"
                ex_hash = hashlib.sha256(hash_key.encode("utf-8")).hexdigest()[:8].upper()
                exceptions.append(
                    ThesisExceptionRow(
                        exception_id=f"EXC-{ex_hash}",
                        claim_id=claim.claim_id,
                        ticker=claim.ticker,
                        detected_at_utc=now_utc,
                        exception_type="NOT_EVALUABLE",
                        metric_name=m_name,
                        expected_value=f"{falsifier.operator} {falsifier.threshold_value} {falsifier.unit}",
                        observed_value="Required metric unavailable; rule was not evaluated.",
                        source_citation=citations.get(m_name),
                        delta_impact="DATA_UNAVAILABLE",
                        human_action="REVIEW_REQUIRED",
                        status="PENDING",
                    )
                )

        # 2. Evaluate Catalysts
        for catalyst in claim.linked_catalysts:
            c_info = events_map.get(catalyst.catalyst_id) or events_map.get(catalyst.event_name) or {}
            obs_date = c_info.get("observed_date") or c_info.get("scheduled_date")
            obs_src = c_info.get("source_locator")
            exc = cls.evaluate_catalyst(
                claim=claim,
                catalyst=catalyst,
                as_of_date_str=as_of_date_str,
                detected_at_utc=now_utc,
                observed_scheduled_date=obs_date,
                observed_source_locator=obs_src,
            )
            if exc:
                exceptions.append(exc)

        return exceptions

