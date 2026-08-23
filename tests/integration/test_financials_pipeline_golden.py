"""Golden Invariant and Pipeline Integration Tests for Pragmatic Hexagonal Architecture."""
import os
from unittest.mock import MagicMock
import pandas as pd
import pytest

from tools.market.financials.adapters.composite_us_provider import CompositeUsFinancialProvider
from tools.market.financials.adapters.edgar_subclient import EdgarSubclient
from tools.market.financials.adapters.in_memory_cache_adapter import InMemoryCacheAdapter
from tools.market.financials.adapters.sec_8k_subclient import Sec8KSubclient
from tools.market.financials.domain.models import FinancialStatementsDTO
from tools.market.financials.service import FinancialsService


def _build_mock_statement(records: list[dict], statement_type: str):
    df = pd.DataFrame(records)
    mock_stmt = MagicMock()
    mock_stmt.to_dataframe.return_value = df
    return mock_stmt


def test_hexagonal_ftnt_pipeline_golden_invariants():
    """Golden test for FTNT with SEC EDGAR + 8-K reconciliation under Hexagonal architecture."""
    # 1. Prepare Mock 10-Q & 10-K DataFrames
    q2_2026_inc = [
        {"standard_concept": "RevenueFromContractWithCustomerExcludingAssessedTax", "2026-06-30 (3M)": 1_634_000_000.0},
        {"standard_concept": "CostOfGoodsAndServicesSold", "2026-06-30 (3M)": 300_000_000.0},
        {"standard_concept": "GrossProfit", "2026-06-30 (3M)": 1_334_000_000.0},
        {"standard_concept": "SellingAndMarketingExpense", "2026-06-30 (3M)": 500_000_000.0},
        {"standard_concept": "GeneralAndAdministrativeExpense", "2026-06-30 (3M)": 100_000_000.0},
        {"standard_concept": "ResearchAndDevelopmentExpense", "2026-06-30 (3M)": 200_000_000.0},
        {"standard_concept": "OperatingExpenses", "2026-06-30 (3M)": 800_000_000.0},
        {"standard_concept": "OperatingIncomeLoss", "2026-06-30 (3M)": 534_000_000.0},
        {"standard_concept": "NetIncomeLoss", "2026-06-30 (3M)": 400_000_000.0},
        {"standard_concept": "EarningsPerShareDiluted", "2026-06-30 (3M)": 0.52},
    ]
    q2_2026_bs = [
        {"standard_concept": "CashAndCashEquivalentsAtCarryingValue", "2026-06-30": 2_500_000_000.0},
        {"standard_concept": "DeferredContractCostCurrent", "2026-06-30": 435_100_000.0},
        {"standard_concept": "DeferredContractCostNoncurrent", "2026-06-30": 350_000_000.0},
        {"standard_concept": "AssetsCurrent", "2026-06-30": 4_000_000_000.0},
        {"standard_concept": "Assets", "2026-06-30": 8_000_000_000.0},
        {"standard_concept": "LiabilitiesCurrent", "2026-06-30": 3_500_000_000.0},
        {"standard_concept": "Liabilities", "2026-06-30": 5_000_000_000.0},
        {"standard_concept": "StockholdersEquity", "2026-06-30": 3_000_000_000.0},
    ]
    q2_2026_cf = [
        {"standard_concept": "NetCashProvidedByUsedInOperatingActivities", "2026-06-30 (6M YTD)": 1_043_600_000.0},
        {"standard_concept": "PaymentsToAcquirePropertyPlantAndEquipment", "2026-06-30 (6M YTD)": 78_000_000.0},
    ]

    q1_2026_inc = [
        {"standard_concept": "RevenueFromContractWithCustomerExcludingAssessedTax", "2026-03-31 (3M)": 1_500_000_000.0},
    ]
    q1_2026_bs = [
        {"standard_concept": "Assets", "2026-03-31": 7_500_000_000.0},
    ]
    q1_2026_cf = [
        {"standard_concept": "NetCashProvidedByUsedInOperatingActivities", "2026-03-31 (3M)": 0.0},
        {"standard_concept": "PaymentsToAcquirePropertyPlantAndEquipment", "2026-03-31 (3M)": 0.0},
    ]
    mock_xb_q1 = MagicMock()
    mock_xb_q1.statements.income_statement.return_value = _build_mock_statement(q1_2026_inc, "income")
    mock_xb_q1.statements.balance_sheet.return_value = _build_mock_statement(q1_2026_bs, "balance_sheet")
    mock_xb_q1.statements.cash_flow_statement.return_value = _build_mock_statement(q1_2026_cf, "cash_flow")
    mock_filing_q1 = MagicMock()
    mock_filing_q1.period_of_report = "2026-03-31"
    mock_filing_q1.form = "10-Q"
    mock_filing_q1.url = "https://www.sec.gov/Archives/edgar/data/ftnt-10q1.htm"
    mock_filing_q1.xbrl.return_value = mock_xb_q1

    # Mock XBRL object Q2
    mock_xb_q2 = MagicMock()
    mock_xb_q2.statements.income_statement.return_value = _build_mock_statement(q2_2026_inc, "income")
    mock_xb_q2.statements.balance_sheet.return_value = _build_mock_statement(q2_2026_bs, "balance_sheet")
    mock_xb_q2.statements.cash_flow_statement.return_value = _build_mock_statement(q2_2026_cf, "cash_flow")

    mock_filing_q2 = MagicMock()
    mock_filing_q2.period_of_report = "2026-06-30"
    mock_filing_q2.form = "10-Q"
    mock_filing_q2.url = "https://www.sec.gov/Archives/edgar/data/ftnt-10q.htm"
    mock_filing_q2.xbrl.return_value = mock_xb_q2

    # Mock 8-K Attachment
    table_8k_html = """
    <html>
    <body>
        <table>
            <tr><th colspan="3">Condensed Consolidated Statements of Cash Flows and Non-GAAP Reconciliations</th></tr>
            <tr><th>(in millions)</th><th>Three Months Ended June 30, 2026</th><th>Three Months Ended June 30, 2025</th></tr>
            <tr><td>Net cash provided by operating activities</td><td>1043.6</td><td>800.0</td></tr>
            <tr><td>Purchases of property and equipment</td><td>(78.0)</td><td>(50.0)</td></tr>
            <tr><td>Free cash flow</td><td>965.6</td><td>750.0</td></tr>
            <tr><td>Real estate related purchases</td><td>30.3</td><td>10.0</td></tr>
            <tr><td>Adjusted free cash flow</td><td>995.9</td><td>760.0</td></tr>
            <tr><td>Free cash flow margin (% of revenue)</td><td>48.6 %</td><td>40.0 %</td></tr>
        </table>
    </body>
    </html>
    """
    mock_att = MagicMock()
    mock_att.document = "ex-99.1.htm"
    mock_att.description = "Press Release"
    mock_att.url = "https://www.sec.gov/Archives/edgar/data/ftnt-ex99.htm"
    mock_att.download.return_value = table_8k_html.encode("utf-8")

    mock_filing_8k = MagicMock()
    mock_filing_8k.filing_date = "2026-08-05"
    mock_filing_8k.attachments = [mock_att]

    # Inject mock subclient
    fake_edgar = EdgarSubclient()
    fake_edgar.get_company_filings = MagicMock(return_value=([], [mock_filing_q1, mock_filing_q2], [mock_filing_8k]))

    composite_provider = CompositeUsFinancialProvider(
        edgar_subclient=fake_edgar,
        sec_8k_subclient=Sec8KSubclient(),
    )

    result = composite_provider.fetch_statements("FTNT", "FTNT")
    assert result is not None
    assert result.ticker == "FTNT"
    assert result.schema_version == 6

    # 2. Check Golden Invariants
    q_cat_cf = next((c for c in result.quarterly if c.statement_type == "cash_flow"), None)
    assert q_cat_cf is not None
    p_q2 = next((p for p in q_cat_cf.periods if p.period_key == "2026-Q2"), None)
    assert p_q2 is not None

    # Invariant 1: Calculated FCF == $965.6M
    calc_fcf_cell = p_q2.items["calculated_free_cash_flow"]
    assert calc_fcf_cell.value == 965_600_000.0

    # Invariant 2: Reported Non-GAAP FCF == $995.9M (Adjusted FCF reported by FTNT, NOT $48.6M margin %)
    rep_fcf_cell = p_q2.items["reported_free_cash_flow"]
    assert rep_fcf_cell.source_type == "reported"
    assert rep_fcf_cell.value == 995_900_000.0

    # Invariant 3: Disclosed Adjustments == +$30.3M
    adj_fcf_cell = p_q2.items["free_cash_flow_adjustments"]
    assert adj_fcf_cell.value == 30_300_000.0

    # Invariant 4: Deferred Contract Costs (Current $435.1M + Non-Current $350.0M) == $785.1M
    q_cat_bs = next((c for c in result.quarterly if c.statement_type == "balance_sheet"), None)
    assert q_cat_bs is not None
    p_q2_bs = next((p for p in q_cat_bs.periods if p.period_key == "2026-Q2"), None)
    assert p_q2_bs is not None
    dcc_cell = p_q2_bs.items["deferred_contract_costs"]
    assert dcc_cell.value == 785_100_000.0
    assert dcc_cell.is_derived is True

    # Invariant 5: Period Uniqueness (no duplicate period keys)
    keys = [p.period_key for p in q_cat_cf.periods]
    assert len(keys) == len(set(keys))
