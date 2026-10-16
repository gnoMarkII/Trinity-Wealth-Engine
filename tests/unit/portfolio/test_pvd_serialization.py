from datetime import date
from decimal import Decimal

from api.schemas.portfolio import ActualPortfolioStateDTO
from tools.portfolio.domain.models import Holding, PortfolioState, _now_iso
from tools.portfolio.domain.pvd_models import PvdHoldingValuation, PvdStatementMetadata


def test_pvd_metadata_round_trips_through_state_and_api_dto():
    valuation = PvdHoldingValuation(
        fund_code="BBLAM", sub_fund_code="SINSA_FI", units=Decimal("123.4567"), statement_nav=Decimal("10.1234"),
        statement_market_value=Decimal("1250.00"), cost_basis=Decimal("1000.00"), as_of_date=date(2026, 9, 30),
        statement_fingerprint="f" * 64, archive_id="a" * 64,
    )
    metadata = PvdStatementMetadata(
        fund_code="BBLAM", member_hash="b" * 64, masked_member_id="••••7890", as_of_date=date(2026, 9, 30),
        total_contribution=Decimal("1000"), total_nav=Decimal("1250"), member_principal=Decimal("600"), member_benefit=Decimal("200"),
        employer_principal=Decimal("300"), employer_benefit=Decimal("150"), employer_vesting_percent=Decimal("50"),
        member_vested_amount=Decimal("800"), employer_vested_amount=Decimal("225"), source_pdf_sha256="c" * 64,
        semantic_fingerprint="f" * 64, archive_id="a" * 64,
    )
    state = PortfolioState(last_updated=_now_iso(), holdings=[Holding(symbol="PVD_SINSA_FI", asset_type="PVD", units=123.4567, pvd_valuation=valuation)], pvd_statements=[metadata], pvd_import_epoch="epoch")
    reloaded = PortfolioState.model_validate(state.model_dump(mode="json"))
    dto = ActualPortfolioStateDTO.model_validate(reloaded.model_dump(exclude_none=True))
    wire = dto.model_dump(mode="json")
    assert wire["holdings"][0]["pvd_valuation"]["statement_nav"] == "10.1234"
    assert wire["pvd_statements"][0]["as_of_date"] == "2026-09-30"
    assert "member_hash" not in wire["pvd_statements"][0]
