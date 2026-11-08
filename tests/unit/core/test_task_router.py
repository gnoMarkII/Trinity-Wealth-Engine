"""Unit tests for Six-Condition Task Routing Engine (Chapter 1 & Lab 1)."""
import pytest
from core.governance.task_router import (
    ExecutionSurface,
    TaskRouteRecord,
    TaskRoutingContext,
    WorkShape,
    evaluate_task_route,
)


def test_chapter1_routing_expert_call_with_mnpi_risk_quarantined():
    """Book Chapter 1 (p. 23 item 8):
    'Store an expert-call note that may contain material nonpublic information.'
    Route to compliance/quarantine review before any AI surface.
    """
    ctx = TaskRoutingContext(contains_mnpi_risk=True)
    res = evaluate_task_route(
        task_id="TASK-008",
        task_description="Store expert interview call on unreleased chip roadmap",
        work_shape=WorkShape.MNPI_OR_INCIDENT_REVIEW,
        context=ctx,
    )
    assert res.selected_surface == ExecutionSurface.QUARANTINE_OR_POLICY_REJECT
    assert res.status == "STOP"
    assert "Compliance" in res.authoritative_system


def test_chapter1_routing_portfolio_rebalance_to_pm():
    """Book Chapter 1 (p. 22 item 7):
    'Calculate exposure in the live risk system and approve a hedge.'
    Touches capital and hedging -> Route to HUMAN_PM_DECISION.
    """
    ctx = TaskRoutingContext(touches_capital_or_policy=True)
    res = evaluate_task_route(
        task_id="TASK-007",
        task_description="Calculate factor beta and rebalance tech portfolio sleeve",
        work_shape=WorkShape.PORTFOLIO_EXPOSURE_AND_HEDGING,
        context=ctx,
    )
    assert res.selected_surface == ExecutionSurface.HUMAN_PM_DECISION
    assert res.status == "REVIEW_REQUIRED"
    assert "Human PM" in res.authoritative_system


def test_chapter1_routing_deterministic_macro_series():
    """Book Chapter 1 (p. 22 item 6):
    'Show payroll growth as it appeared before later revisions.'
    Point-in-time calculation -> Route to DETERMINISTIC_TOOL_OR_SCRIPT.
    """
    ctx = TaskRoutingContext(has_authoritative_calculator=True, requires_reconstructability=True)
    res = evaluate_task_route(
        task_id="TASK-006",
        task_description="Build macro vintage walk for Nonfarm Payrolls at June 10 cutoff",
        work_shape=WorkShape.PIT_MACRO_SERIES,
        context=ctx,
    )
    assert res.selected_surface == ExecutionSurface.DETERMINISTIC_TOOL_OR_SCRIPT
    assert "Python Core" in res.authoritative_system


def test_chapter1_routing_unstructured_document_synthesis():
    """Book Chapter 1 (p. 21 item 1):
    'Summarize the risk factors added in this quarter's 10-Q.'
    Document reading -> Route to LLM_RESEARCH_SYNTHESIS.
    """
    ctx = TaskRoutingContext(is_data_approved_and_sanitized=True)
    res = evaluate_task_route(
        task_id="TASK-001",
        task_description="Extract and summarize newly added risk factors from Q2 10-Q",
        work_shape=WorkShape.ROUTINE_READING_OR_INGEST,
        context=ctx,
    )
    assert res.selected_surface == ExecutionSurface.LLM_RESEARCH_SYNTHESIS
    assert "Hermes" in res.authoritative_system

