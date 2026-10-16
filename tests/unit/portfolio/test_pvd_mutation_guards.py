from decimal import Decimal

import pytest

from tools.portfolio.domain.models import TradeFeeBreakdown, TradeImportItem
from tools.portfolio.services.batch_trade_import_service import BatchTradeImportService


def test_generic_trade_batch_rejects_pvd_snapshot_before_repository_access():
    item = TradeImportItem(
        item_id="pvd-1", trade_date="2026-09-30", symbol="PVD_SINSA_FI", action="PVD_SNAPSHOT",
        units=Decimal("1"), price=Decimal("10"), gross_amount=Decimal("10"), fees=TradeFeeBreakdown(),
        net_amount=Decimal("10"), currency="THB", confirmation_no="statement", source="BBLAM_PVD",
        fingerprint="f" * 64, cash_adjusted=False, asset_type="PVD", bucket_id="retirement_pvd", pvd_metadata_json="{}",
    )
    service = BatchTradeImportService(repo=None)
    with pytest.raises(ValueError, match="PvdSyncService"):
        service.execute_batch_import([item])
