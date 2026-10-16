from datetime import date
from decimal import Decimal

from tools.portfolio.domain.calculations import _replay_symbol_trades, recalc_all
from tools.portfolio.domain.models import Holding, PortfolioState, _now_iso
from tools.portfolio.domain.pvd_models import PvdHoldingValuation


def test_snapshot_replay_replaces_prior_target_instead_of_accumulating():
    rows = [
        {"Transaction_ID": "one", "Timestamp": "2026-09-30 00:00:00", "Action": "PVD_SNAPSHOT", "Units": "100", "Cost_THB": "900.00"},
        {"Transaction_ID": "two", "Timestamp": "2026-10-31 00:00:00", "Action": "PVD_SNAPSHOT", "Units": "110", "Cost_THB": "1000.00"},
    ]
    _updated, units, avg_cost, realized = _replay_symbol_trades(rows, "PVD_SINSA_FI", "THB")
    assert units == 110
    assert avg_cost == 9.0909
    assert realized == 0


def test_recalc_uses_exact_statement_value_and_cost_basis_after_reload():
    holding = Holding(
        symbol="PVD_SINSA_FI", asset_type="PVD", units=2, bucket_id="retirement_pvd",
        pvd_valuation=PvdHoldingValuation(
            fund_code="BBLAM", sub_fund_code="SINSA_FI", units=Decimal("2"), statement_nav=Decimal("10.1234"),
            statement_market_value=Decimal("20.25"), reconciliation_adjustment=Decimal("0.04"),
            cost_basis=Decimal("17.11"), as_of_date=date(2026, 9, 30),
            statement_fingerprint="f" * 64, archive_id="a" * 64,
        ),
    )
    state = PortfolioState.model_validate(PortfolioState(last_updated=_now_iso(), holdings=[holding]).model_dump(mode="json"))
    recalc_all(state)
    assert state.holdings[0].market_value_thb == 20.29
    assert state.summary.total_cost_basis_thb == 17.11
    assert state.summary.total_unrealized_profit == 3.18
    assert state.holdings[0].unrealized_pnl_value == 3.18


def test_recalc_preserves_custom_assigned_bucket_for_pvd():
    holding = Holding(
        symbol="PVD_SINSA_FI", asset_type="PVD", units=2, bucket_id="defensive",
        pvd_valuation=PvdHoldingValuation(
            fund_code="BBLAM", sub_fund_code="SINSA_FI", units=Decimal("2"), statement_nav=Decimal("10.0"),
            statement_market_value=Decimal("20.0"), reconciliation_adjustment=Decimal("0.0"),
            cost_basis=Decimal("18.0"), as_of_date=date(2026, 9, 30),
            statement_fingerprint="f" * 64, archive_id="a" * 64,
        ),
    )
    state = PortfolioState(last_updated=_now_iso(), holdings=[holding])
    recalc_all(state)
    assert state.holdings[0].bucket_id == "defensive"


def test_apply_pvd_snapshot_rows_preserves_existing_holding_bucket():
    import json
    from tools.portfolio.domain.pvd_models import PvdStatementMetadata
    from tools.portfolio.domain.pvd_replay import apply_pvd_snapshot_rows

    metadata = PvdStatementMetadata(
        fund_code="BBLAM", member_hash="b" * 64, masked_member_id="••••7890", as_of_date=date(2026, 9, 30),
        total_contribution=Decimal("1000"), total_nav=Decimal("1250"), member_principal=Decimal("600"), member_benefit=Decimal("200"),
        employer_principal=Decimal("300"), employer_benefit=Decimal("150"), employer_vesting_percent=Decimal("50"),
        member_vested_amount=Decimal("800"), employer_vested_amount=Decimal("225"), source_pdf_sha256="c" * 64,
        semantic_fingerprint="f" * 64, archive_id="a" * 64,
    )
    metadata_json = json.dumps(metadata.model_dump(mode="json"))
    holding = Holding(
        symbol="PVD_SINSA_GLOB", asset_type="PVD", units=10, bucket_id="core_equities"
    )
    rows = [{
        "Transaction_ID": "row-1", "Timestamp": "2026-09-30 00:00:00", "Action": "PVD_SNAPSHOT",
        "Symbol": "PVD_SINSA_GLOB", "Units": "15", "Price": "20.0", "Cost_THB": "250.0",
        "Gross_Amount": "300.0", "Confirmation_No": "statement-1", "PVD_Metadata_JSON": metadata_json,
        "Reconciliation_Adjustment": "0.0", "Bucket_ID": "retirement_pvd",
    }]
    holdings = [holding]
    apply_pvd_snapshot_rows(holdings, rows)
    assert holdings[0].bucket_id == "core_equities"
    assert holdings[0].units == 15.0

