"""Text-based BBLAM PVD PDF parser using pypdf.

The parser intentionally fails closed when required statement labels or rows are
missing. Its accepted line layout is covered by synthetic fixtures; production
rollout still needs a real statement fixture to confirm the BBLAM text layout.
"""
from __future__ import annotations

import hashlib
import io
import re
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Optional

import pypdf

from core.datetime_utils import parse_thai_or_iso_date, extract_date_by_labels
from core.financial_parsers import extract_financial_decimal, extract_labeled_decimal
from core.text_utils import extract_labeled_value
from tools.portfolio.domain.pvd_models import PvdContributionSummary, PvdStatement, PvdSubFundHolding
from tools.portfolio.domain.pvd_validation import validate_statement
from tools.portfolio.ports.pvd_port import PvdParserPort


_NUMBER = re.compile(r"(?<![A-Za-z])[-+]?\d[\d,]*(?:\.\d+)?")
_ACCOUNTING_NUMBER = re.compile(
    r"\(\s*(?P<negative>\d[\d,]*(?:\.\d+)?)\s*\)|(?P<plain>(?<![A-Za-z])[-+]?\d[\d,]*(?:\.\d+)?)"
)
_DATE = re.compile(r"(?<!\d)(\d{1,2}[/-]\d{1,2}[/-]\d{4}|\d{4}-\d{2}-\d{2})(?!\d)")
_CODE = re.compile(r"(?<![A-Za-z0-9])(SINSA(?:_[A-Z0-9_]+|[0-9]+))(?![A-Za-z0-9])", re.IGNORECASE)


def _decimal(raw: str) -> Decimal:
    val = extract_financial_decimal(raw)
    if val is None or not val.is_finite():
        raise ValueError(f"Invalid PVD numeric value: {raw}")
    return val


def _number_values(text: str) -> list[Decimal]:
    """Read whitespace-separated statement figures, including accounting negatives."""
    values: list[Decimal] = []
    for match in _ACCOUNTING_NUMBER.finditer(text):
        if match.group("negative") is not None:
            values.append(-_decimal(match.group("negative")))
        else:
            values.append(_decimal(match.group("plain")))
    return values


def _normal_member_id(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def _parse_date(raw: str) -> date:
    raw = raw.strip()
    parsed = parse_thai_or_iso_date(raw)
    if not parsed:
        raise ValueError(f"ไม่สามารถแปลงวันที่ใน Statement ได้: {raw}")
    if parsed.year < 1900 or parsed > date.today():
        raise ValueError("วันที่ใน Statement อยู่นอกช่วงที่รองรับ")
    return parsed


def _label_value(text: str, labels: tuple[str, ...], *, required: bool = True) -> Optional[Decimal]:
    val = extract_labeled_decimal(text, labels)
    if val is None:
        if required:
            raise ValueError(f"ไม่พบยอด '{labels[0]}' ใน Statement")
        return None
    return val


class BblamPvdPdfParserAdapter(PvdParserPort):
    def parse_statement(self, pdf_bytes: bytes, password: Optional[str] = None) -> PvdStatement:
        if not pdf_bytes or not pdf_bytes.startswith(b"%PDF-"):
            raise ValueError("ไฟล์ที่อัปโหลดไม่ใช่ PDF")
        try:
            reader = pypdf.PdfReader(io.BytesIO(pdf_bytes), strict=True)
            if reader.is_encrypted:
                if password is None:
                    raise ValueError("PDF นี้มีรหัสผ่าน กรุณาระบุรหัสผ่าน")
                if reader.decrypt(password) == 0:
                    raise ValueError("รหัสผ่าน PDF ไม่ถูกต้อง")
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"อ่าน PDF ไม่สำเร็จ: {exc}") from exc
        if not text.strip():
            raise ValueError("ไม่พบข้อความใน PDF; รองรับเฉพาะ Statement แบบ text-based")

        as_of_d = extract_date_by_labels(
            text,
            ["As of Date", "As of", "Statement Date", "วันที่ข้อมูล", "วันที่สรุป", "วันที่ณ"],
        )
        if not as_of_d:
            raise ValueError("ไม่พบวันที่ของ Statement")
        if as_of_d.year < 1900 or as_of_d > date.today():
            raise ValueError("วันที่ใน Statement อยู่นอกช่วงที่รองรับ")
        as_of_date = as_of_d

        member_id = extract_labeled_value(
            text,
            [
                "Member ID",
                "Member No.",
                "Member Number",
                "Member Code",
                "Unit Holder ID",
                "Unit Holder No.",
                "Unit Holder Number",
                "รหัสสมาชิก",
                "เลขที่สมาชิก",
            ],
            value_pattern=r"((?:EM\s+)?[A-Z0-9-]+)",
        )
        if not member_id:
            raise ValueError("ไม่พบรหัสสมาชิกใน Statement")
        member_id = member_id.strip()
        normalized_id = _normal_member_id(member_id)
        member_hash = hashlib.sha256(normalized_id.encode("utf-8")).hexdigest()
        masked_member_id = f"••••{normalized_id[-4:]}" if len(normalized_id) > 4 else "••••"

        fund_code_raw = extract_labeled_value(text, ["Fund Code"], value_pattern=r"([A-Z0-9-]+)")
        fund_code = f"BBLAM_{fund_code_raw.upper()}" if fund_code_raw else "BBLAM"

        # BBLAM's member certificate layout stores the contribution/balance
        # summary in one table and each sub-fund's NAV in a later return table.
        # Parse those columns explicitly instead of treating every SINSA row as
        # a synthetic `code units NAV value` line.
        lines = text.splitlines()
        total_index = next((i for i, line in enumerate(lines) if re.search(r"(?<![A-Za-z0-9])Total\s*\(\s*Baht\s*\)", line, re.IGNORECASE)), None)
        accumulated_index = next((i for i, line in enumerate(lines) if re.search(r"Accumulated\s+Return\s+of\s+each\s+Sub\s+Fund", line, re.IGNORECASE)), None)
        if total_index is not None and accumulated_index is not None:
            total_line = re.sub(r"(?<=\s)\(\d+\)(?=\s+[-+]?\d[\d,]*(?:\.\d+)?\s*$)", " ", lines[total_index])
            totals = _number_values(total_line)
            if len(totals) != 5:
                raise ValueError("รูปแบบตารางยอดรวม BBLAM ไม่ถูกต้อง")

            values_by_code: dict[str, tuple[Decimal, Decimal, Decimal]] = {}
            for line in lines[:accumulated_index]:
                code_match = _CODE.search(line)
                if not code_match:
                    continue
                code = code_match.group(1).upper()
                values = _number_values(line[code_match.end():])
                # Designated proportion, current proportion, member principal,
                # member benefit, employer principal, employer benefit, units,
                # market value.
                if len(values) < 8:
                    continue
                values_by_code[code] = (values[1], values[6], values[7])

            nav_by_code: dict[str, Decimal] = {}
            for line in lines[accumulated_index + 1:]:
                code_match = _CODE.search(line)
                if not code_match:
                    continue
                code = code_match.group(1).upper()
                values = _number_values(line[code_match.end():])
                if values:
                    nav_by_code.setdefault(code, values[0])

            if not values_by_code or set(values_by_code) - set(nav_by_code):
                raise ValueError("ตาราง BBLAM ขาดแถวกองทุนย่อยหรือ NAV ต่อหน่วย")

            holdings = [
                PvdSubFundHolding(
                    code=code,
                    name=code,
                    units=values_by_code[code][1],
                    statement_nav=nav_by_code[code],
                    statement_market_value=values_by_code[code][2],
                    weight_percent=values_by_code[code][0],
                )
                for code in sorted(values_by_code)
            ]
            member_principal, member_benefit, employer_principal, employer_benefit, total_nav = totals
            total_contribution = member_principal + employer_principal
            summary = PvdContributionSummary(
                member_principal=member_principal,
                member_benefit=member_benefit,
                employer_principal=employer_principal,
                employer_benefit=employer_benefit,
                total_contribution=total_contribution,
                total_nav=total_nav,
                employer_vesting_percent=Decimal("100"),
                member_vested_amount=member_principal + member_benefit,
                employer_vested_amount=employer_principal + employer_benefit,
            )
            statement = PvdStatement(
                fund_code=fund_code,
                member_id=member_id,
                member_hash=member_hash,
                masked_member_id=masked_member_id,
                as_of_date=as_of_date,
                summary=summary,
                holdings=holdings,
                source_pdf_sha256=hashlib.sha256(pdf_bytes).hexdigest(),
            )
            return validate_statement(statement)

        member_principal = _label_value(text, ("Member Principal", "Member Contribution", "Employee Principal", "Employee Contribution", "เงินสะสมสมาชิก", "เงินสะสมของสมาชิก"))
        member_benefit = _label_value(text, ("Member Benefit", "Member Return", "Employee Benefit", "ผลประโยชน์เงินสะสมสมาชิก", "ผลประโยชน์ของสมาชิก"))
        employer_principal = _label_value(text, ("Employer Principal", "Employer Contribution", "Company Principal", "เงินสมทบนายจ้าง", "เงินสมทบของนายจ้าง"))
        employer_benefit = _label_value(text, ("Employer Benefit", "Employer Return", "Company Benefit", "ผลประโยชน์เงินสมทบนายจ้าง", "ผลประโยชน์ของนายจ้าง"))
        total_contribution = _label_value(text, ("Total Contribution", "Total Principal", "เงินสะสมรวม", "เงินต้นรวม"))
        total_nav = _label_value(text, ("Total NAV", "Total Fund Value", "Total Account Value", "มูลค่ารวม", "ยอดเงินรวม", "มูลค่าทรัพย์สินสุทธิรวม"))

        holding_rows: list[PvdSubFundHolding] = []
        seen: set[str] = set()
        for line in text.splitlines():
            code_match = _CODE.search(line)
            if not code_match:
                continue
            code = code_match.group(1).upper()
            if code in seen:
                continue
            nums = [_decimal(m.group(0)) for m in _NUMBER.finditer(line[code_match.end():])]
            if len(nums) < 3:
                continue
            # BBLAM statement row: units, NAV, market value, optional weight.
            units, nav, market_value = nums[:3]
            weight = nums[3] if len(nums) >= 4 else Decimal("0")
            name = line[code_match.end():].strip()
            name = _NUMBER.sub("", name).strip(" |\t:-") or code
            holding_rows.append(PvdSubFundHolding(
                code=code,
                name=name,
                units=units,
                statement_nav=nav,
                statement_market_value=market_value,
                weight_percent=weight,
            ))
            seen.add(code)
        if not holding_rows:
            raise ValueError("ไม่พบแถวกองทุนย่อย SINSA ในตาราง Statement")

        employer_total = employer_principal + employer_benefit
        summary = PvdContributionSummary(
            member_principal=member_principal,
            member_benefit=member_benefit,
            employer_principal=employer_principal,
            employer_benefit=employer_benefit,
            total_contribution=total_contribution,
            total_nav=total_nav,
            employer_vesting_percent=Decimal("100"),
            member_vested_amount=member_principal + member_benefit,
            employer_vested_amount=employer_total,
        )
        statement = PvdStatement(
            fund_code=fund_code,
            member_id=member_id,
            member_hash=member_hash,
            masked_member_id=masked_member_id,
            as_of_date=as_of_date,
            summary=summary,
            holdings=holding_rows,
            source_pdf_sha256=hashlib.sha256(pdf_bytes).hexdigest(),
        )
        return validate_statement(statement)
