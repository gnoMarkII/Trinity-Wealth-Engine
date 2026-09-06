import os
import tempfile
from decimal import Decimal
from pathlib import Path
import pytest

from tools.portfolio.domain.models import (
    TradeImportItem,
    TradeFeeBreakdown,
    PortfolioState,
    Holding,
)
from tools.portfolio.domain.errors import (
    TradeReconciliationError,
)
from tools.portfolio.ports.trade_ingestion_port import (
    TradeDocumentMetadata,
    TradeEmailSourcePort,
    TradeDocumentParserPort,
    TradeStagingPort,
)
from tools.portfolio.adapters.wealthx.wealthx_parser_adapter import (
    _parse_wealthx_text_to_dicts,
    WealthXPdfParserAdapter,
)
from tools.portfolio.adapters.dime.inmemory_staging_adapter import InMemoryStagingAdapter
from tools.portfolio.adapters.markdown.repository_adapter import MarkdownVaultRepositoryAdapter
from tools.portfolio.services.batch_trade_import_service import BatchTradeImportService
from tools.portfolio.services.wealthx_sync_service import (
    WealthXSyncService,
    _load_sync_history,
    _save_sync_history,
)

SAMPLE_WEALTHX_TEXT = """
บริษัทหลักทรัพย์ เวลธ์ เอกซ์ จำกัด (สำนักงานใหญ่)
ใบยืนยันการซื้อขายหน่วยลงทุน (Confirmation Note)
เลขที่สัญญา (Account No.) : 000100418-2
ชื่อลูกค้า (Customer Name) : นาย กวินภพ อ่อนนิ่ม
วันที่ส่งคำสั่ง / Trade Date : 12/06/2026
วันที่ครบกำหนดชำระ / Due Date : 15/06/2026
เลขที่ใบยืนยัน (Settlement No.) : DN 202606150569
บลจ. / AMC : บริษัทหลักทรัพย์จัดการกองทุน ทาลิส จำกัด
ชื่อกองทุน / Fund Name : กองทุนเปิด ทาลิส หุ้น ยูเอส ออล โกรท (TLWORLD-X)
ประเภทรายการ / Transaction Type : ซื้อหน่วยลงทุน (BUY SUB)
เลขอ้างอิง / Reference No. : 2392606120006840
จำนวนเงิน / Amount (Baht) : 10,000.00
ค่าธรรมเนียมการซื้อ / Front-end Fee (Baht) : 9.32
ภาษีมูลค่าเพิ่ม / VAT (Baht) : 0.65
ค่าธรรมเนียมรวม / Total Fee (Baht) : 9.97
จำนวนเงินสุทธิ / Net Amount (Baht) : 10,000.00
ราคาต่อหน่วย / Price per Unit (Baht) : 11.1311
จำนวนหน่วย / Allocated Units : 898.3838
"""

SAMPLE_WEALTHX_TEXT_2 = """
บริษัทหลักทรัพย์ เวลธ์ เอกซ์ จำกัด (สำนักงานใหญ่)
ใบยืนยันการซื้อขายหน่วยลงทุน (Confirmation Note)
เลขที่สัญญา (Account No.) : 000100418-2
ชื่อลูกค้า (Customer Name) : นาย กวินภพ อ่อนนิ่ม
วันที่ส่งคำสั่ง / Trade Date : 20/07/2026
วันที่ครบกำหนดชำระ / Due Date : 22/07/2026
เลขที่ใบยืนยัน (Settlement No.) : DN 202607220112
บลจ. / AMC : บริษัทหลักทรัพย์จัดการกองทุน ทาลิส จำกัด
ชื่อกองทุน / Fund Name : กองทุนเปิด ทาลิส แนสแด็ก 100 อินคัม โพรเทคชั่น ห้ามขายผู้ลงทุนรายย่อย (TLNDQINCOME-UH-X)
ประเภทรายการ / Transaction Type : ซื้อหน่วยลงทุน (BUY SUB)
เลขอ้างอิง / Reference No. : 2392607200009999
จำนวนเงิน / Amount (Baht) : 50,000.00
ค่าธรรมเนียมการซื้อ / Front-end Fee (Baht) : 0.00
ภาษีมูลค่าเพิ่ม / VAT (Baht) : 0.00
ค่าธรรมเนียมรวม / Total Fee (Baht) : 0.00
จำนวนเงินสุทธิ / Net Amount (Baht) : 50,000.00
ราคาต่อหน่วย / Price per Unit (Baht) : 10.2540
จำนวนหน่วย / Allocated Units : 4876.1459
"""


SAMPLE_WEALTHX_MULTI_ROW_TEXT = """
ใบยืนยันการซื้อขายหน่วยลงทุน (Confirmation Note)
เลขที่สัญญา (Account No.) : 000100418-2
วันที่ส่งคำสั่ง / Trade Date : 25/03/2026
วันที่ครบกำหนดชำระ / Due Date : 27/03/2026
เลขที่ใบยืนยัน (Settlement No.) : DN 202603270119
   ชื่อกองทุน                 บลจ.             ประเภทคำาสั่ง          เลขที่อ้างอิง         จำานวนหน่วย             ราคา/หน่วย            ค่าธรรมเนียมรวม            จำานวนเงิน (บาท)
   Fund Code                 AMC                  Type of               รายการ             No. of Units             Unit Price         ภาษีมูลค่าเพิ่ม (บาท)        Amount (Baht)
   TLWORLD-                TALISAM                  BUY                   SUB                 504.0322                  9.9200                          4.99                5,000.00
   X                                                                 239260325
                                                                       0022908
   TLNDQINC                TALISAM                  BUY                   SUB               1,000.7105                  9.9929                         78.56              10,000.00
   OME-UH-X                                                          239260325
                                                                       0022909
                                                                                          รวมมูลค่าซื้อ (บาท) / Total Buy (Baht)                                          15,000.00
"""


def test_parse_wealthx_text_to_dicts_tlworld():
    items = _parse_wealthx_text_to_dicts(SAMPLE_WEALTHX_TEXT)
    assert len(items) == 1
    row = items[0]
    assert row["confirmation_no"] == "DN 202606150569"
    assert row["order_id"] == "2392606120006840"
    assert row["trade_date"] == "2026-06-12"
    assert row["settlement_date"] == "2026-06-15"
    assert row["symbol"] == "TLWORLD-X"
    assert row["action"] == "BUY"
    assert row["units"] == "898.3838"
    assert row["price"] == "11.1311"
    assert row["net_amount"] == "10000.00"
    assert row["gross_amount"] == "10000.00"
    assert row["source"] == "WEALTHX"
    assert row["asset_type"] == "Fund"
    assert row["cash_adjusted"] is True


def test_parse_wealthx_text_to_dicts_complex_symbol():
    items = _parse_wealthx_text_to_dicts(SAMPLE_WEALTHX_TEXT_2)
    assert len(items) == 1
    row = items[0]
    assert row["symbol"] == "TLNDQINCOME-UH-X"
    assert row["confirmation_no"] == "DN 202607220112"
    assert row["order_id"] == "2392607200009999"
    assert row["trade_date"] == "2026-07-20"
    assert row["net_amount"] == "50000.00"
    assert row["source"] == "WEALTHX"


def test_parse_wealthx_text_to_dicts_multi_row():
    items = _parse_wealthx_text_to_dicts(SAMPLE_WEALTHX_MULTI_ROW_TEXT)
    assert len(items) == 2

    # Row 1: TLWORLD-X
    r1 = items[0]
    assert r1["symbol"] == "TLWORLD-X"
    assert r1["units"] == "504.0322"
    assert r1["price"] == "9.9200"
    assert r1["net_amount"] == "5000.00"
    assert r1["order_id"] == "2392603250022908"
    assert r1["confirmation_no"] == "DN 202603270119"

    # Row 2: TLNDQINCOME-UH-X
    r2 = items[1]
    assert r2["symbol"] == "TLNDQINCOME-UH-X"
    assert r2["units"] == "1000.7105"
    assert r2["price"] == "9.9929"
    assert r2["net_amount"] == "10000.00"
    assert r2["order_id"] == "2392603250022909"
    assert r2["confirmation_no"] == "DN 202603270119"



class MockWealthXEmailSource(TradeEmailSourcePort):
    def __init__(self, emails, pdf_map):
        self._emails = emails
        self._pdf_map = pdf_map

    def search_dime_emails(self, query: str = "", limit: int = 20):
        return []

    def search_wealthx_emails(self, query: str = "", limit=None):
        return self._emails

    def search_scbam_emails(self, query: str = "", limit=None):
        return []

    def fetch_pdf_attachment(self, message_id: str, attachment_id: str) -> bytes:
        return self._pdf_map[message_id]

    def fetch_email_html_body(self, message_id: str) -> str:
        return ""


class MockWealthXParser(TradeDocumentParserPort):
    def __init__(self, text_map):
        self._text_map = text_map

    def parse_confirmation_pdf(self, pdf_bytes: bytes, password=None):
        key = pdf_bytes.decode("utf-8", errors="ignore")
        text = self._text_map.get(key, SAMPLE_WEALTHX_TEXT)
        dicts = _parse_wealthx_text_to_dicts(text)
        results = []
        for r in dicts:
            results.append(
                TradeImportItem(
                    item_id=r["item_id"],
                    trade_date=r["trade_date"],
                    settlement_date=r["settlement_date"],
                    symbol=r["symbol"],
                    action=r["action"],
                    units=Decimal(r["units"]),
                    price=Decimal(r["price"]),
                    gross_amount=Decimal(r["gross_amount"]),
                    fees=TradeFeeBreakdown(
                        commission=Decimal("0.00"),
                        vat=Decimal("0.00"),
                        other_fees=Decimal("0.00"),
                        fee_currency="THB",
                    ),
                    net_amount=Decimal(r["net_amount"]),
                    currency=r["currency"],
                    confirmation_no=r["confirmation_no"],
                    order_id=r["order_id"],
                    source=r["source"],
                    fingerprint=r["fingerprint"],
                    cash_adjusted=r["cash_adjusted"],
                    asset_type=r["asset_type"],
                )
            )
        return results


def test_wealthx_sync_service_stream_and_deduplication(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setenv("OBSIDIAN_VAULT_PATH", tmpdir)

        # 2 emails with identical Confirmation & Order ID (real case from live test: UID 45020 & 45022)
        email1 = TradeDocumentMetadata(
            message_id="msg1",
            attachment_id="msg1_att0",
            subject="ใบยืนยันการซื้อขาย Confirmation Note 1",
            sender="noreply@wealthx.co",
            received_at="Fri, 12 Jun 2026 10:00:00 +0700",
            filename="wealthx_confirmation_msg1.pdf",
            size_bytes=1024,
            x_gm_msgid="msg1",
            uid="45020",
        )
        email2 = TradeDocumentMetadata(
            message_id="msg2",
            attachment_id="msg2_att0",
            subject="ใบยืนยันการซื้อขาย Confirmation Note 1 (Copy)",
            sender="noreply@wealthx.co",
            received_at="Fri, 12 Jun 2026 10:05:00 +0700",
            filename="wealthx_confirmation_msg2.pdf",
            size_bytes=1024,
            x_gm_msgid="msg2",
            uid="45022",
        )

        email_source = MockWealthXEmailSource(
            emails=[email1, email2],
            pdf_map={"msg1": b"pdf1", "msg2": b"pdf1"},  # Both return SAMPLE_WEALTHX_TEXT
        )
        parser = MockWealthXParser({"pdf1": SAMPLE_WEALTHX_TEXT})
        staging = InMemoryStagingAdapter()

        repo = MarkdownVaultRepositoryAdapter()
        batch_service = BatchTradeImportService(repo=repo)

        service = WealthXSyncService(
            email_source=email_source,
            parser=parser,
            staging=staging,
            batch_import_service=batch_service,
        )

        events = list(service.stream_batch_sync(portfolio_id="default", session_id="test_sess"))
        complete_event = next(e for e in events if e["event"] == "complete")

        # 2 duplicate emails must consolidate to exactly 1 unique item!
        assert complete_event["data"]["item_count"] == 1
        scan_id = complete_event["data"]["scan_id"]
        assert scan_id != ""

        # Test commit
        state = service.commit_staged(scan_id=scan_id, session_id="test_sess", portfolio_id="default")
        
        # Verify holding TLWORLD-X and CASH_THB reduction
        tlworld_holding = next((h for h in state.holdings if h.symbol == "TLWORLD-X"), None)
        assert tlworld_holding is not None
        assert tlworld_holding.units == 898.3838
        assert tlworld_holding.asset_type == "Fund"

        cash_holding = next((h for h in state.holdings if h.symbol == "CASH_THB"), None)
        assert cash_holding is not None
        # CASH_THB should have decreased by 10,000.00
        assert cash_holding.units == -10000.00
