"""Unit tests for FinancialContext validation and readiness auditing."""
from decimal import Decimal
import pytest

from core.investor_essence.models import FinancialContextSnapshot
from core.investor_essence.financial_context import (
    evaluate_financial_readiness,
    update_readiness,
)


class TestFinancialContext:
    def test_empty_snapshot_detects_blocking_issues(self):
        snapshot = FinancialContextSnapshot(
            snapshot_id="fc_1",
            portfolio_id="port_default",
            unknown_fields=["emergency_reserves", "obligations", "withdrawal_plan"],
        )
        issues = evaluate_financial_readiness(snapshot)
        codes = [i.code for i in issues]
        assert "MISSING_HORIZON" in codes
        assert "UNKNOWN_RESERVES" in codes
        assert "UNKNOWN_OBLIGATIONS" in codes
        assert "MISSING_WITHDRAWAL_PLAN" in codes

        updated = update_readiness(snapshot)
        assert updated.is_ready_for_numeric_policy is False
        assert len(updated.readiness_issues) == 4

    def test_complete_snapshot_is_ready(self):
        snapshot = FinancialContextSnapshot(
            snapshot_id="fc_2",
            portfolio_id="port_default",
            horizon_years=Decimal("10"),
            emergency_reserves_months=Decimal("6"),
            obligations_monthly=Decimal("30000.00"),
            target_use_amount=Decimal("500000.00"),
            target_use_timeline="5 ปีข้างหน้าสำหรับดาวน์บ้าน",
        )
        issues = evaluate_financial_readiness(snapshot)
        assert len(issues) == 0

        updated = update_readiness(snapshot)
        assert updated.is_ready_for_numeric_policy is True
        assert len(updated.readiness_issues) == 0

    def test_warning_issue_only_allows_readiness(self):
        # Missing withdrawal plan is warning only, but horizon and reserves and obligations are present
        snapshot = FinancialContextSnapshot(
            snapshot_id="fc_3",
            portfolio_id="port_default",
            horizon_years=Decimal("5"),
            emergency_reserves_amount=Decimal("200000.00"),
            obligations_monthly=Decimal("0.00"),  # Explicit zero obligations
            unknown_fields=["withdrawal_plan"],
        )
        issues = evaluate_financial_readiness(snapshot)
        assert len(issues) == 1
        assert issues[0].code == "MISSING_WITHDRAWAL_PLAN"
        assert issues[0].severity == "warning"

        updated = update_readiness(snapshot)
        # Warnings do not block numeric readiness
        assert updated.is_ready_for_numeric_policy is True
