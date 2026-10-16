from datetime import date
from decimal import Decimal

import pytest

from tools.portfolio.adapters.bblam_pvd import bblam_pvd_pdf_parser_adapter as parser_module
from tools.portfolio.adapters.bblam_pvd.bblam_pvd_pdf_parser_adapter import BblamPvdPdfParserAdapter


TEXT = """BBLAM Provident Fund Statement
As of Date: 30/09/2026
Member ID: 1234567890
Member Principal: 1,500.00
Member Benefit: 300.00
Employer Principal: 1,000.00
Employer Benefit: 200.00
Total Contribution: 2,500.00
Total NAV: 3,000.00
SINSA_FI Fixed Income 10.0000 100.0000 1,000.00 33.33%
SINSA_GLOB Global Equity 20.0000 100.0000 2,000.00 66.67%
"""

OFFICIAL_LAYOUT_TEXT = """Certificate/Statement of Provident Fund Member
As of 30/06/2568 Fund Code 01272541
Member Code EM 1234567890 Member Name TEST MEMBER
Total (Baht) 150.00 20.00 100.00 10.00 (7) 280.00
Outstanding Balance of each Fund / Sub Fund
SINSA_FI 0.00 48.21 80.00 10.00 40.00 5.00 10.0000 135.00
SINSA_ASIA 0.00 51.79 70.00 10.00 60.00 5.00 15.0000 145.00
Accumulated Return of each Sub Fund
SINSA_FI 13.4985 01/01/2568 - 30/06/2568= 2.29%
SINSA_ASIA 17.0778 01/01/2568 - 30/06/2568= 8.68%
"""


class FakePage:
    def extract_text(self):
        return TEXT


class FakeReader:
    def __init__(self, _stream, strict=True):
        self.is_encrypted = False
        self.pages = [FakePage()]


def test_parses_statement_labels_and_subfund_rows(monkeypatch):
    monkeypatch.setattr(parser_module.pypdf, "PdfReader", FakeReader)
    result = BblamPvdPdfParserAdapter().parse_statement(b"%PDF-synthetic")
    assert result.as_of_date == date(2026, 9, 30)
    assert result.member_hash and result.masked_member_id.endswith("7890")
    assert result.summary.total_nav == 3000
    assert [row.code for row in result.holdings] == ["SINSA_FI", "SINSA_GLOB"]
    assert sum(row.allocated_cost for row in result.holdings) == result.summary.total_contribution


def test_parses_bblam_member_certificate_layout_with_separate_nav_table(monkeypatch):
    class OfficialReader(FakeReader):
        def __init__(self, _stream, strict=True):
            self.is_encrypted = False
            self.pages = [type("Page", (), {"extract_text": lambda self: OFFICIAL_LAYOUT_TEXT})()]

    monkeypatch.setattr(parser_module.pypdf, "PdfReader", OfficialReader)
    result = BblamPvdPdfParserAdapter().parse_statement(b"%PDF-official-layout")

    assert result.as_of_date == date(2025, 6, 30)
    assert result.fund_code == "BBLAM_01272541"
    assert result.member_id == "EM 1234567890"
    assert result.masked_member_id.endswith("7890")
    assert result.summary.member_principal == 150
    assert result.summary.member_benefit == 20
    assert result.summary.employer_principal == 100
    assert result.summary.employer_benefit == 10
    assert result.summary.total_contribution == 250
    assert result.summary.total_nav == 280
    rows = {row.code: row for row in result.holdings}
    assert rows["SINSA_FI"].statement_nav == Decimal("13.4985")
    assert rows["SINSA_FI"].units == Decimal("10")
    assert rows["SINSA_FI"].statement_market_value == Decimal("135")
    assert rows["SINSA_ASIA"].statement_nav == Decimal("17.0778")


def test_rejects_missing_member_or_non_pdf(monkeypatch):
    parser = BblamPvdPdfParserAdapter()
    with pytest.raises(ValueError, match="ไม่ใช่ PDF"):
        parser.parse_statement(b"not a pdf")

    class MissingMemberReader(FakeReader):
        def __init__(self, stream, strict=True):
            super().__init__(stream, strict)
            self.pages = [type("Page", (), {"extract_text": lambda self: TEXT.replace("Member ID: 1234567890", "")})()]

    monkeypatch.setattr(parser_module.pypdf, "PdfReader", MissingMemberReader)
    with pytest.raises(ValueError, match="รหัสสมาชิก"):
        parser.parse_statement(b"%PDF-synthetic")


def test_password_is_passed_unchanged_to_encrypted_pdf_reader(monkeypatch):
    received = []

    class EncryptedReader(FakeReader):
        def __init__(self, stream, strict=True):
            super().__init__(stream, strict)
            self.is_encrypted = True

        def decrypt(self, password):
            received.append(password)
            return 1 if password == " 1234 " else 0

    monkeypatch.setattr(parser_module.pypdf, "PdfReader", EncryptedReader)
    BblamPvdPdfParserAdapter().parse_statement(b"%PDF-synthetic", password=" 1234 ")
    assert received == [" 1234 "]
