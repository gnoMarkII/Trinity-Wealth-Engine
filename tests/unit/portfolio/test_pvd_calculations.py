from datetime import date
from decimal import Decimal

from tools.portfolio.domain.pvd_calculations import allocate_cost_basis, vested_amount
from tools.portfolio.domain.pvd_models import PvdContributionSummary, PvdStatement, PvdSubFundHolding
from tools.portfolio.domain.pvd_validation import validate_statement


def statement(*, total_nav="3.00", total_contribution="1.00"):
    return PvdStatement(
        member_id="1234567890",
        member_hash="a" * 64,
        as_of_date=date(2026, 9, 30),
        summary=PvdContributionSummary(
            member_principal=Decimal(total_contribution),
            member_benefit=Decimal(total_nav) - Decimal(total_contribution),
            employer_principal=Decimal("0"),
            employer_benefit=Decimal("0"),
            total_contribution=Decimal(total_contribution),
            total_nav=Decimal(total_nav),
            employer_vesting_percent=Decimal("100"),
            member_vested_amount=Decimal(total_nav),
            employer_vested_amount=Decimal("0"),
        ),
        holdings=[
            PvdSubFundHolding(code=code, name=code, units=Decimal("1"), statement_nav=Decimal("1"), statement_market_value=Decimal("1"), weight_percent=Decimal("33.33"))
            for code in ("SINSA_C", "SINSA_A", "SINSA_B")
        ],
    )


def test_rounding_residual_goes_to_largest_valuation_and_tie_breaks_by_code():
    result = validate_statement(statement())
    allocated = {row.code: row.allocated_cost for row in result.holdings}
    assert allocated == {"SINSA_A": Decimal("0.34"), "SINSA_B": Decimal("0.33"), "SINSA_C": Decimal("0.33")}
    assert sum(allocated.values()) == Decimal("1.00")


def test_reconciliation_adjustment_is_applied_before_cost_allocation():
    source = statement(total_nav="3.04", total_contribution="1.00")
    result = validate_statement(source)
    assert sum(row.reconciliation_adjustment for row in result.holdings) == Decimal("0.04")
    assert sum(row.allocated_cost for row in result.holdings) == Decimal("1.00")


def test_employer_vesting_uses_integer_safe_decimal_calculation():
    assert vested_amount(Decimal("213712.95"), Decimal("175160.50"), Decimal("50")) == Decimal("301293.20")


def test_reconciliation_outside_statement_tolerance_is_rejected():
    source = statement(total_nav="3.06", total_contribution="1.00")
    try:
        validate_statement(source)
    except ValueError as exc:
        assert "0.05" in str(exc)
    else:
        raise AssertionError("expected statement reconciliation rejection")
