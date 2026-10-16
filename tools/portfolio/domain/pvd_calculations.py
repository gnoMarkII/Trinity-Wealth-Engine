"""Exact Decimal calculations used by PVD statement previews and commits."""
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json

from .pvd_models import PvdStatement

MONEY = Decimal("0.01")
ZERO = Decimal("0")


def money(value: Decimal) -> Decimal:
    return value.quantize(MONEY, rounding=ROUND_HALF_UP)


def semantic_fingerprint(statement: PvdStatement) -> str:
    payload = {
        "fund_code": statement.fund_code,
        "member_hash": statement.member_hash,
        "as_of_date": statement.as_of_date.isoformat(),
        "summary": {k: str(v) for k, v in statement.summary.model_dump().items()},
        "holdings": [
            {
                "code": h.code,
                "units": str(h.units),
                "nav": str(h.statement_nav),
                "market_value": str(h.statement_market_value),
            }
            for h in sorted(statement.holdings, key=lambda row: row.code)
        ],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def allocate_cost_basis(statement: PvdStatement) -> PvdStatement:
    """Allocate total contributions by reconciled market value, including residual cents."""
    total_nav = statement.summary.total_nav
    contribution = statement.summary.total_contribution
    holdings = sorted(statement.holdings, key=lambda row: row.code)
    if not holdings:
        raise ValueError("Statement ไม่มีรายการกองทุนย่อย")

    raw_values = [h.statement_market_value + h.reconciliation_adjustment for h in holdings]
    if any(value < ZERO for value in raw_values):
        raise ValueError("การปรับยอดทำให้มูลค่ากองทุนย่อยติดลบ")
    valuation_total = sum(raw_values, ZERO)

    allocated: list[Decimal]
    if total_nav == ZERO:
        if contribution != ZERO or any(h.units != ZERO for h in holdings):
            raise ValueError("ยอด NAV เป็นศูนย์แต่มีเงินต้นหรือหน่วยลงทุน")
        allocated = [money(ZERO) for _ in holdings]
    else:
        if valuation_total == ZERO:
            raise ValueError("ยอด NAV เป็นบวกแต่ไม่มีมูลค่ากองทุนย่อย")
        allocated = [money(value * contribution / valuation_total) for value in raw_values]
        residual = money(contribution - sum(allocated, ZERO))
        positive = [i for i, value in enumerate(raw_values) if value > ZERO]
        if residual and not positive:
            raise ValueError("ไม่พบกองทุนย่อยที่รับส่วนต่างการปัดเศษได้")
        if positive:
            # Codes are already sorted; max(value) therefore uses the first code on a tie.
            largest_value = max(raw_values[i] for i in positive)
            recipient = next(i for i in positive if raw_values[i] == largest_value)
            allocated[recipient] = money(allocated[recipient] + residual)

    for i, holding in enumerate(holdings):
        holding.allocated_cost = allocated[i]
    statement.holdings = holdings
    return statement


def vested_amount(member_total: Decimal, employer_total: Decimal, vesting_percent: Decimal) -> Decimal:
    if not Decimal("0") <= vesting_percent <= Decimal("100"):
        raise ValueError("Vesting ต้องอยู่ระหว่าง 0 ถึง 100 เปอร์เซ็นต์")
    # Member contributions are always vested; the slider applies to employer benefits.
    return money(member_total + employer_total * vesting_percent / Decimal("100"))

