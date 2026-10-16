"""PVD statement reconciliation rules, kept independent of PDF parsing."""
from decimal import Decimal

from .pvd_calculations import MONEY, allocate_cost_basis, money
from .pvd_models import PvdStatement


def validate_statement(statement: PvdStatement) -> PvdStatement:
    summary = statement.summary
    if not statement.member_id.strip() or not statement.member_hash:
        raise ValueError("ไม่พบรหัสสมาชิกกองทุนสำรองเลี้ยงชีพ")
    if not statement.fund_code.strip():
        raise ValueError("ไม่พบรหัสกองทุน")
    if not statement.holdings:
        raise ValueError("ไม่พบตารางกองทุนย่อย")

    codes = [holding.code.strip().upper() for holding in statement.holdings]
    if any(not code for code in codes) or len(codes) != len(set(codes)):
        raise ValueError("รหัสกองทุนย่อยว่างหรือซ้ำกัน")

    for label, value in (
        ("member_principal", summary.member_principal),
        ("member_benefit", summary.member_benefit),
        ("employer_principal", summary.employer_principal),
        ("employer_benefit", summary.employer_benefit),
        ("total_contribution", summary.total_contribution),
        ("total_nav", summary.total_nav),
    ):
        if not value.is_finite():
            raise ValueError(f"{label} ต้องเป็นจำนวนเงินที่ finite")
    if money(summary.member_principal + summary.employer_principal) != money(summary.total_contribution):
        raise ValueError("เงินต้นสมาชิกและนายจ้างรวมไม่เท่ากับเงินสะสมทั้งหมด")
    if money(summary.member_principal + summary.member_benefit + summary.employer_principal + summary.employer_benefit) != money(summary.total_nav):
        raise ValueError("เงินต้นและผลประโยชน์รวมไม่เท่ากับ NAV")
    if money(summary.member_vested_amount) != money(summary.member_principal + summary.member_benefit):
        raise ValueError("เงินสะสมฝั่งสมาชิกต้องได้รับสิทธิเต็มจำนวน")
    if summary.employer_vested_amount < 0 or summary.employer_vested_amount > summary.employer_principal + summary.employer_benefit:
        raise ValueError("ยอดสิทธินายจ้างเกินยอดเงินนายจ้าง")

    market_total = Decimal("0")
    for row in statement.holdings:
        if any(not value.is_finite() for value in (row.units, row.statement_nav, row.statement_market_value, row.weight_percent)):
            raise ValueError(f"ข้อมูลกองทุนย่อย {row.code} มีค่า NaN/Infinity")
        if min(row.units, row.statement_nav, row.statement_market_value) < 0:
            raise ValueError(f"ข้อมูลกองทุนย่อย {row.code} ติดลบ")
        if row.units > 0 and row.statement_nav <= 0:
            raise ValueError(f"กองทุนย่อย {row.code} มีหน่วยลงทุนแต่ NAV ไม่เป็นบวก")
        if row.units == 0 and (row.statement_market_value != 0 or row.allocated_cost != 0):
            raise ValueError(f"กองทุนย่อย {row.code} ไม่มีหน่วยแต่ยังมียอดมูลค่าหรือต้นทุน")
        market_total += row.statement_market_value

    difference = money(summary.total_nav - market_total)
    if abs(difference) > Decimal("0.05"):
        raise ValueError(f"มูลค่ากองทุนย่อยต่างจาก NAV {difference:.2f} บาท เกินช่วงยอมรับ 0.05 บาท")
    if difference:
        candidates = sorted(
            (row for row in statement.holdings if row.statement_market_value > 0),
            key=lambda row: (-row.statement_market_value, row.code),
        )
        if not candidates:
            raise ValueError("มียอด reconciliation แต่ไม่มีแถวกองทุนย่อยที่รับ adjustment ได้")
        candidates[0].reconciliation_adjustment = difference

    validated = allocate_cost_basis(statement)
    if sum((row.allocated_cost for row in validated.holdings), Decimal("0")) != summary.total_contribution:
        raise ValueError("ต้นทุนจัดสรรรวมไม่เท่ากับเงินต้นสะสม")
    return validated

