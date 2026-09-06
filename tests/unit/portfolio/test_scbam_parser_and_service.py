import json
from decimal import Decimal
from typing import List, Optional
import pytest

from tools.portfolio.adapters.scb.scbam_parser_adapter import (
    parse_scbam_fundclick_html,
    SCBAMRawOrder,
)
from tools.portfolio.domain.models import (
    PortfolioState,
    TradeImportItem,
    TradeFeeBreakdown,
)
from tools.portfolio.ports.trade_ingestion_port import (
    TradeDocumentMetadata,
    TradeEmailSourcePort,
    TradeStagingPort,
)
from tools.portfolio.ports.thai_fund_port import ThaiFundPricePort, FundNavData
from tools.portfolio.adapters.dime.inmemory_staging_adapter import InMemoryStagingAdapter
from tools.portfolio.services.batch_trade_import_service import BatchTradeImportService
from tools.portfolio.services.scbam_sync_service import SCBAMSyncService


SAMPLE_SCBAM_HTML_SP500 = """
<!DOCTYPE html>
<html>
<body>
<div style="font-family: sans-serif;">
  <h2>ยืนยันการทำรายการซื้อกองทุน</h2>
  <p>เรียน ท่านผู้ถือหน่วยลงทุน</p>
  <p>บริษัทหลักทรัพย์จัดการกองทุน ไทยพาณิชย์ จำกัด ขอแจ้งรายละเอียดการทำรายการ ดังนี้</p>
  <table>
    <tr><td>วันที่ทำรายการ:</td><td>26/12/2025 เวลา 15:58:25 น.</td></tr>
    <tr><td>วันที่คำสั่งมีผล:</td><td>29/12/2025</td></tr>
    <tr><td>เลขที่รายการ:</td><td>ADV-2025-12-26-15.58.25.068380</td></tr>
    <tr><td>เลขที่บัญชีกองทุน:</td><td>000027893865</td></tr>
    <tr><td>กองทุน:</td><td>SCBS&P500E (กองทุนเปิดไทยพาณิชย์หุ้นยูเอส (ชนิดช่องทางอิเล็กทรอนิกส์))</td></tr>
    <tr><td>จำนวนเงิน:</td><td>35,000.00 บาท</td></tr>
  </table>
</div>
</body>
</html>
"""

SAMPLE_SCBAM_HTML_GVALUE = """
<!DOCTYPE html>
<html>
<body>
<div>
  <p>วันที่ทำรายการ: 15/01/2026 เวลา 10:30:12 น.</p>
  <p>วันที่คำสั่งมีผล: 16/01/2026</p>
  <p>เลขที่รายการ: ADV-2026-01-15-10.30.12.112233</p>
  <p>เลขที่บัญชีกองทุน: 000027893865</p>
  <p>กองทุน: SCBGVALUE(E) (กองทุนเปิดไทยพาณิชย์ โกลบอล แวลู (ชนิดช่องทางอิเล็กทรอนิกส์))</p>
  <p>จำนวนเงิน: 5,000.00 บาท</p>
</div>
</body>
</html>
"""

SAMPLE_SCBAM_HTML_SP500_STD = """
<!DOCTYPE html>
<html>
<body>
<div style="font-family: sans-serif;">
  <h2>ยืนยันการทำรายการซื้อกองทุน</h2>
  <p>เรียน ท่านผู้ถือหน่วยลงทุน</p>
  <p>บริษัทหลักทรัพย์จัดการกองทุน ไทยพาณิชย์ จำกัด ขอแจ้งรายละเอียดการทำรายการ ดังนี้</p>
  <table>
    <tr><td>วันที่ทำรายการ:</td><td>29/12/2025 เวลา 08:11:44 น.</td></tr>
    <tr><td>วันที่คำสั่งมีผล:</td><td>29/12/2025</td></tr>
    <tr><td>เลขที่รายการ:</td><td>2025-12-29-08.11.44.680007</td></tr>
    <tr><td>เลขที่บัญชีกองทุน:</td><td>000027893865</td></tr>
    <tr><td>กองทุน:</td><td>SCBS&P500E (กองทุนเปิดไทยพาณิชย์หุ้นยูเอส (ชนิดช่องทางอิเล็กทรอนิกส์))</td></tr>
    <tr><td>จำนวนเงิน:</td><td>35,000.00 บาท</td></tr>
  </table>
</div>
</body>
</html>
"""

SAMPLE_SCBAM_HTML_GVALUE_STD = """
<!DOCTYPE html>
<html>
<body>
<div>
  <p>วันที่ทำรายการ: 15/01/2026 เวลา 10:30:12 น.</p>
  <p>วันที่คำสั่งมีผล: 16/01/2026</p>
  <p>เลขที่รายการ: 2026-01-15-10.30.12.112233</p>
  <p>เลขที่บัญชีกองทุน: 000027893865</p>
  <p>กองทุน: SCBGVALUE(E) (กองทุนเปิดไทยพาณิชย์ โกลบอล แวลู (ชนิดช่องทางอิเล็กทรอนิกส์))</p>
  <p>จำนวนเงิน: 5,000.00 บาท</p>
</div>
</body>
</html>
"""


class MockSCBAMEmailSource(TradeEmailSourcePort):
    def __init__(self, email_bodies: dict):
        self.email_bodies = email_bodies

    def search_dime_emails(self, query: str = "", limit: int = 20) -> List[TradeDocumentMetadata]:
        return []

    def fetch_pdf_attachment(self, message_id: str, attachment_id: str) -> bytes:
        return b""

    def search_wealthx_emails(self, query: str = "", limit: Optional[int] = None) -> List[TradeDocumentMetadata]:
        return []

    def search_scbam_emails(self, since_date=None, query=None, limit=100) -> List[TradeDocumentMetadata]:
        metas = []
        for msg_id in self.email_bodies.keys():
            metas.append(TradeDocumentMetadata(
                message_id=msg_id,
                attachment_id="",
                subject="ยืนยันการทำรายการซื้อกองทุน",
                sender="fundclick.scbam@scb.co.th",
                received_at="2025-12-26 16:00:00",
                filename="SCBAM Order HTML",
                size_bytes=len(self.email_bodies[msg_id]),
                uid=msg_id,
            ))
        return metas

    def fetch_email_html_body(self, message_id_or_uid: str) -> str:
        return self.email_bodies.get(message_id_or_uid, "")


class MockThaiFundPriceAdapter(ThaiFundPricePort):
    def __init__(self, nav_map: dict):
        self.nav_map = nav_map

    def has_fund(self, symbol: str) -> bool:
        return True

    def refresh_catalog(self, force: bool = False) -> int:
        return 1

    def fetch_nav(self, symbol: str) -> Optional[FundNavData]:
        val = self.nav_map.get((symbol, "latest"), 0.0)
        return FundNavData(symbol=symbol, nav=val, nav_date="2026-01-01") if val > 0 else None

    def fetch_historical_nav(self, symbol: str, target_date: str) -> Optional[float]:
        return self.nav_map.get((symbol, target_date))


class MockBatchTradeImporter:
    def __init__(self):
        self.imported_items = []

    def execute_batch_import(self, items: List[TradeImportItem], portfolio_id: str = "default") -> PortfolioState:
        self.imported_items.extend(items)
        return PortfolioState(
            cash_balances={"THB": Decimal("100000.00")},
            holdings=[],
            total_portfolio_value_thb=Decimal("100000.00"),
            last_replayed_at="2026-09-06T00:00:00Z",
            last_updated="2026-09-06T00:00:00Z",
        )


def test_parse_scbam_fundclick_html_sp500():
    order = parse_scbam_fundclick_html(SAMPLE_SCBAM_HTML_SP500)
    assert order is not None
    assert order.tx_date == "2025-12-26"
    assert order.tx_time == "15:58:25"
    assert order.effective_date == "2025-12-29"
    assert order.transaction_no == "ADV-2025-12-26-15.58.25.068380"
    assert order.account_no == "000027893865"
    assert order.fund_code == "SCBS&P500E"
    assert order.amount == Decimal("35000.00")
    assert order.action == "BUY"


def test_parse_scbam_fundclick_html_gvalue():
    order = parse_scbam_fundclick_html(SAMPLE_SCBAM_HTML_GVALUE)
    assert order is not None
    assert order.tx_date == "2026-01-15"
    assert order.effective_date == "2026-01-16"
    assert order.transaction_no == "ADV-2026-01-15-10.30.12.112233"
    assert order.account_no == "000027893865"
    assert order.fund_code == "SCBGVALUE(E)"
    assert order.amount == Decimal("5000.00")


def test_parse_scbam_fundclick_html_invalid():
    assert parse_scbam_fundclick_html("") is None
    assert parse_scbam_fundclick_html("<html><body>No transaction here</body></html>") is None


def test_scbam_sync_service_stream_and_commit():
    email_source = MockSCBAMEmailSource({
        "msg_1": SAMPLE_SCBAM_HTML_SP500_STD,
        "msg_2": SAMPLE_SCBAM_HTML_GVALUE_STD,
    })
    price_port = MockThaiFundPriceAdapter({
        ("SCBS&P500E", "2025-12-29"): 40.3609,
        ("SCBGVALUE(E)", "2026-01-16"): 10.5000,
    })
    staging = InMemoryStagingAdapter()
    batch_importer = MockBatchTradeImporter()

    service = SCBAMSyncService(
        email_source=email_source,
        price_port=price_port,
        staging=staging,
        batch_importer=batch_importer,
    )

    events_raw = "".join(list(service.stream_scbam_sync(portfolio_id="test_port")))
    event_types = []
    complete_data = None

    for block in events_raw.split("\n\n"):
        if not block.strip():
            continue
        for line in block.split("\n"):
            if line.startswith("event:"):
                event_types.append(line.split(":", 1)[1].strip())
            elif line.startswith("data:") and "event: complete" in block:
                complete_data = json.loads(line[5:].strip())

    assert "progress" in event_types
    assert "order" in event_types
    assert "complete" in event_types
    assert "done" in event_types

    assert complete_data is not None
    assert complete_data["item_count"] == 2
    items = complete_data["items"]
    assert items[0]["symbol"] == "SCBS&P500E"
    assert items[0]["order_id"] == "SCB-2025-12-29-08.11.44.680007"
    assert items[0]["gross_amount"] == "35000.00"
    assert items[0]["price"] == "40.3609"
    assert items[0]["asset_type"] == "Fund"
    # Units = 35000 / 40.3609 = 867.175905... (UNITS_QUANTUM is 6 decimals)
    assert items[0]["units"] == "867.175905"

    assert items[1]["symbol"] == "SCBGVALUE(E)"
    assert items[1]["asset_type"] == "Fund"
    # Units = 5000 / 10.5 = 476.190476...
    assert items[1]["units"] == "476.190476"

    # Test commit
    scan_id = complete_data["scan_id"]
    state, count = service.commit_scbam_sync(portfolio_id="test_port", scan_session_id=scan_id)
    assert count == 2
    assert len(batch_importer.imported_items) == 2


def test_scbam_sync_service_deduplication():
    # Both emails have same transaction_no
    email_source = MockSCBAMEmailSource({
        "msg_1": SAMPLE_SCBAM_HTML_SP500_STD,
        "msg_dup": SAMPLE_SCBAM_HTML_SP500_STD,
    })
    price_port = MockThaiFundPriceAdapter({
        ("SCBS&P500E", "2025-12-29"): 40.3609,
    })
    staging = InMemoryStagingAdapter()
    batch_importer = MockBatchTradeImporter()

    service = SCBAMSyncService(
        email_source=email_source,
        price_port=price_port,
        staging=staging,
        batch_importer=batch_importer,
    )

    events = "".join(list(service.stream_scbam_sync(portfolio_id="test_port")))
    complete_data = None
    for block in events.split("\n\n"):
        if "event: complete" in block:
            for line in block.split("\n"):
                if line.startswith("data:"):
                    complete_data = json.loads(line[5:].strip())

    assert complete_data is not None
    # Duplicate was consolidated
    assert complete_data["item_count"] == 1


def test_scbam_single_email_scan():
    email_source = MockSCBAMEmailSource({
        "msg_1": SAMPLE_SCBAM_HTML_SP500_STD,
    })
    price_port = MockThaiFundPriceAdapter({
        ("SCBS&P500E", "2025-12-29"): 40.3609,
    })
    staging = InMemoryStagingAdapter()
    batch_importer = MockBatchTradeImporter()

    service = SCBAMSyncService(
        email_source=email_source,
        price_port=price_port,
        staging=staging,
        batch_importer=batch_importer,
    )

    scan_id, items = service.scan_scbam_email("msg_1")
    assert len(items) == 1
    assert items[0].symbol == "SCBS&P500E"
    assert items[0].asset_type == "Fund"
    assert items[0].price == Decimal("40.3609")
    assert items[0].gross_amount == Decimal("35000.00")
    assert items[0].units == Decimal("867.175905")
    assert items[0].order_id == "SCB-2025-12-29-08.11.44.680007"


def test_scbam_quarantine_advance_orders():
    """Verify that orders starting with ADV- are quarantined with warning and not staged."""
    email_source = MockSCBAMEmailSource({
        "msg_adv": SAMPLE_SCBAM_HTML_SP500,  # Contains ADV-2025-12-26...
    })
    price_port = MockThaiFundPriceAdapter({
        ("SCBS&P500E", "2025-12-29"): 40.3609,
    })
    staging = InMemoryStagingAdapter()
    batch_importer = MockBatchTradeImporter()

    service = SCBAMSyncService(
        email_source=email_source,
        price_port=price_port,
        staging=staging,
        batch_importer=batch_importer,
    )

    events_raw = "".join(list(service.stream_scbam_sync(portfolio_id="test_port")))
    event_types = []
    complete_data = None

    for block in events_raw.split("\n\n"):
        if not block.strip():
            continue
        for line in block.split("\n"):
            if line.startswith("event:"):
                event_types.append(line.split(":", 1)[1].strip())
            elif line.startswith("data:") and "event: complete" in block:
                complete_data = json.loads(line[5:].strip())

    assert "warning" in event_types
    assert "order" not in event_types
    assert complete_data is not None
    assert complete_data["item_count"] == 0
    assert len(complete_data["warnings"]) == 1
    assert "Advance Order" in complete_data["warnings"][0]["reason"]

    # Single scan raises ValueError
    with pytest.raises(ValueError, match="Advance Order"):
        service.scan_scbam_email("msg_adv")

