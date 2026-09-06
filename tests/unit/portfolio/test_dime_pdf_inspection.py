import io
from pathlib import Path
import pytest
import pypdf
from fastapi.testclient import TestClient

from tools.portfolio.ports.trade_ingestion_port import (
    TradeEmailSourcePort,
    TradeDocumentMetadata,
    TradeDocumentParserPort,
    TradeStagingPort,
)
from tools.portfolio.services.dime_sync_service import DimeSyncService
from tools.portfolio.services.batch_trade_import_service import BatchTradeImportService
from tools.portfolio.domain.models import TradeImportItem
from api.main import app
from api.auth import require_session
from api.dependencies import get_portfolio_service


def _create_test_pdf_bytes(text: str = "Test Confirmation Note Content", password: str = None) -> bytes:
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    # pypdf blank page might not have text stream, so let's add text annotation or metadata
    writer.add_metadata({"/Title": "Dime Confirmation Note", "/Subject": text})
    if password:
        writer.encrypt(password)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


class MockEmailSource(TradeEmailSourcePort):
    def __init__(self, pdf_bytes: bytes):
        self._pdf_bytes = pdf_bytes

    def search_dime_emails(self, query: str = "", limit: int = None):
        return [
            TradeDocumentMetadata(
                message_id="msg_123",
                attachment_id="att_456",
                subject="[Dime!] Confirmation Note",
                sender="no-reply@dime.co.th",
                received_at="2026-09-01",
                filename="confirm_123.pdf",
                size_bytes=len(self._pdf_bytes),
            )
        ]

    def search_wealthx_emails(self, query: str = "", limit: int = None):
        return []

    def search_scbam_emails(self, query: str = "", limit: int = None):
        return []

    def fetch_pdf_attachment(self, message_id: str, attachment_id: str) -> bytes:
        if message_id == "msg_123" and attachment_id == "att_456":
            return self._pdf_bytes
        raise ValueError(f"Attachment not found: {message_id}, {attachment_id}")

    def fetch_email_html_body(self, message_id: str) -> str:
        return "<html><body>Mock HTML Body</body></html>"


class MockParser(TradeDocumentParserPort):
    def parse_confirmation_pdf(self, pdf_bytes: bytes, password: str = None):
        return []


class MockStaging(TradeStagingPort):
    def stage_items(self, items, ttl_seconds=1800, session_id=None):
        return "scan_1"

    def get_staged_items(self, scan_id, session_id=None):
        return []

    def delete_staged(self, scan_id, session_id=None):
        pass


def test_dime_sync_service_get_email_pdf_unencrypted():
    raw_pdf = _create_test_pdf_bytes("Hello Dime")
    email_source = MockEmailSource(raw_pdf)
    service = DimeSyncService(
        email_source=email_source,
        parser=MockParser(),
        staging=MockStaging(),
        batch_import_service=None,
    )

    pdf_bytes, filename = service.get_email_pdf("msg_123", "att_456", decrypt=True)
    assert len(pdf_bytes) > 0
    assert filename == "dime_msg_123.pdf"

    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    assert not reader.is_encrypted


def test_dime_sync_service_get_email_pdf_encrypted():
    encrypted_pdf = _create_test_pdf_bytes("Secret Dime Trade", password="secret_pass_123")
    email_source = MockEmailSource(encrypted_pdf)
    service = DimeSyncService(
        email_source=email_source,
        parser=MockParser(),
        staging=MockStaging(),
        batch_import_service=None,
    )

    # Decrypt with correct password
    decrypted_bytes, filename = service.get_email_pdf("msg_123", "att_456", password="secret_pass_123", decrypt=True)
    reader = pypdf.PdfReader(io.BytesIO(decrypted_bytes))
    assert not reader.is_encrypted

    # Raw fetch without decrypt
    raw_bytes, _ = service.get_email_pdf("msg_123", "att_456", decrypt=False)
    raw_reader = pypdf.PdfReader(io.BytesIO(raw_bytes))
    assert raw_reader.is_encrypted


def test_dime_sync_service_get_email_pdf_text():
    raw_pdf = _create_test_pdf_bytes("Dime Page Text")
    email_source = MockEmailSource(raw_pdf)
    service = DimeSyncService(
        email_source=email_source,
        parser=MockParser(),
        staging=MockStaging(),
        batch_import_service=None,
    )

    result = service.get_email_pdf_text("msg_123", "att_456")
    assert result["message_id"] == "msg_123"
    assert result["attachment_id"] == "att_456"
    assert result["page_count"] == 1
    assert len(result["pages"]) == 1
    assert result["pages"][0]["page_number"] == 1


def test_api_dime_pdf_endpoints():
    raw_pdf = _create_test_pdf_bytes("API Test PDF", password="password123")
    email_source = MockEmailSource(raw_pdf)
    dime_service = DimeSyncService(
        email_source=email_source,
        parser=MockParser(),
        staging=MockStaging(),
        batch_import_service=None,
    )

    class MockPortfolioService:
        _dime_sync_service = dime_service

    app.dependency_overrides[require_session] = lambda: None
    app.dependency_overrides[get_portfolio_service] = lambda: MockPortfolioService()

    try:
        client = TestClient(app)

        # Test GET /api/portfolio/dime/pdf
        res_pdf = client.get("/api/portfolio/dime/pdf?message_id=msg_123&attachment_id=att_456&password=password123")
        assert res_pdf.status_code == 200
        assert res_pdf.headers["content-type"] == "application/pdf"
        assert "inline" in res_pdf.headers["content-disposition"]
        assert res_pdf.headers.get("x-frame-options") == "SAMEORIGIN"
        assert "frame-ancestors" in res_pdf.headers.get("content-security-policy", "")
        # Confirm returned PDF is decrypted
        reader = pypdf.PdfReader(io.BytesIO(res_pdf.content))
        assert not reader.is_encrypted

        # Test GET /api/portfolio/dime/pdf-text
        res_text = client.get("/api/portfolio/dime/pdf-text?message_id=msg_123&attachment_id=att_456&password=password123")
        assert res_text.status_code == 200
        data = res_text.json()
        assert data["message_id"] == "msg_123"
        assert data["attachment_id"] == "att_456"
        assert data["page_count"] == 1
        assert len(data["pages"]) == 1
        assert data["pages"][0]["page_number"] == 1
    finally:
        app.dependency_overrides.clear()


def test_reconciliation_fractional_shares_tolerance():
    from decimal import Decimal
    from tools.portfolio.domain.calculations import validate_reconciliation_invariant

    # 1. Real NOK trade from Dime confirmation: 36.662240 * 7.2300 = 265.07 != 265.16 (diff=0.09)
    # Statement Gross (265.16) - Fee (0.43) == Net (264.73) exactly.
    ok, msg = validate_reconciliation_invariant(
        units=Decimal("36.662240"),
        price=Decimal("7.2300"),
        gross_amount=Decimal("265.16"),
        fees=Decimal("0.43"),
        net_amount=Decimal("264.73"),
        action="SELL",
    )
    assert ok is True, f"NOK reconciliation failed: {msg}"

    # 2. Real INTC trade from Dime confirmation: 7.035904 * 19.8700 = 139.80 != 139.78 (diff=0.02)
    # Statement Gross (139.78) + Fee (0.22) == Net (140.00) exactly.
    ok, msg = validate_reconciliation_invariant(
        units=Decimal("7.035904"),
        price=Decimal("19.8700"),
        gross_amount=Decimal("139.78"),
        fees=Decimal("0.22"),
        net_amount=Decimal("140.00"),
        action="BUY",
    )
    assert ok is True, f"INTC reconciliation failed: {msg}"

    # 3. Whole shares mismatch must still fail strictly if diff > 0.01
    ok_whole, _ = validate_reconciliation_invariant(
        units=Decimal("100"),
        price=Decimal("10.00"),
        gross_amount=Decimal("1005.00"),
        fees=Decimal("5.00"),
        net_amount=Decimal("1010.00"),
        action="BUY",
    )
    assert ok_whole is False

    # 4. Gross corruption on fractional shares must still fail if diff exceeds theoretical rounding bound
    ok_corrupt, _ = validate_reconciliation_invariant(
        units=Decimal("7.035904"),
        price=Decimal("19.8700"),
        gross_amount=Decimal("200.00"),
        fees=Decimal("0.22"),
        net_amount=Decimal("200.22"),
        action="BUY",
    )
    assert ok_corrupt is False


def test_parse_mutual_fund_with_spaces_in_name():
    from tools.portfolio.adapters.dime.isolated_parser_adapter import _parse_dime_text_to_dicts

    # Simulated layout text from Dime Mutual Fund confirmation note
    sample_text = """
Confirmation Note / Receipt / Tax Invoice
Tax Invoice No. : DIMEMF20251015000440
Effective Date : 15/10/2025

Order ID     Transaction Type     Fund Name              Units          NAV/Unit     Total Amount     Fee Include Vat
2202510150007450   SUB            PRINCIPAL VNEQ-A       670.4839       14.9146      10,000.00        147.78
"""
    items = _parse_dime_text_to_dicts(sample_text)
    assert len(items) == 1
    assert items[0]["symbol"] == "PRINCIPAL VNEQ-A"
    assert items[0]["order_id"] == "2202510150007450"
    assert items[0]["action"] == "BUY"
    assert items[0]["units"] == "670.4839"
    assert items[0]["price"] == "14.9146"
    assert items[0]["gross_amount"] == "10000.00"
    assert items[0]["net_amount"] == "10000.00"

