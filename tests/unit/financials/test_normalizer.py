"""Unit tests for HTML/XBRL Normalizer, Table to Grid, and Sub-item Derivations."""
from bs4 import BeautifulSoup
import pandas as pd
from tools.market.financials.domain.models import FinancialCellDTO
from tools.market.financials.domain.normalizer import (
    extract_canonical_cell_from_df,
    resolve_sub_line_items,
    select_xbrl_column,
    table_to_grid,
    validate_sec_url,
)


def test_validate_sec_url():
    assert validate_sec_url("https://www.sec.gov/Archives/edgar/data/123.htm") == "https://www.sec.gov/Archives/edgar/data/123.htm"
    assert validate_sec_url("http://insecure.com/doc.htm") is None
    assert validate_sec_url("https://malicious-sec.gov.phishing.com/doc.htm") is None
    assert validate_sec_url(None) is None


def test_table_to_grid_with_colspan_rowspan():
    html = """
    <table>
        <tr>
            <th rowspan="2">Line Item</th>
            <th colspan="2">Three Months Ended</th>
        </tr>
        <tr>
            <th>June 30, 2024</th>
            <th>June 30, 2023</th>
        </tr>
        <tr>
            <td>Revenue</td>
            <td>$1,500.0</td>
            <td>$1,200.0</td>
        </tr>
    </table>
    """
    soup = BeautifulSoup(html, "html.parser")
    grid = table_to_grid(soup.find("table"))

    assert len(grid) == 3
    assert grid[0][0] == "Line Item"
    assert grid[0][1] == "Three Months Ended"
    assert grid[0][2] == "Three Months Ended"
    assert grid[1][0] == "Line Item"
    assert grid[1][1] == "June 30, 2024"
    assert grid[1][2] == "June 30, 2023"
    assert grid[2][0] == "Revenue"
    assert grid[2][1] == "$1,500.0"
    assert grid[2][2] == "$1,200.0"


def test_select_xbrl_column():
    cols = ["2024-06-30 (3M)", "2024-06-30 (6M YTD)", "2023-06-30 (3M)"]
    target, dur, lbl = select_xbrl_column(cols, "2024-06-30", "income", is_quarterly=True, fiscal_quarter=2)
    assert target == "2024-06-30 (3M)"
    assert dur == 91

    target_cf, dur_cf, lbl_cf = select_xbrl_column(cols, "2024-06-30", "cash_flow", is_quarterly=True, fiscal_quarter=2)
    assert target_cf == "2024-06-30 (6M YTD)"
    assert dur_cf == 182


def test_resolve_sub_line_items_dcc():
    df = pd.DataFrame({
        "standard_concept": ["DeferredContractCostCurrent", "DeferredContractCostNoncurrent"],
        "concept": ["us-gaap_DeferredContractCostCurrent", "us-gaap_DeferredContractCostNoncurrent"],
        "col_2024": [300_000_000.0, 485_100_000.0],
    })
    items: dict[str, FinancialCellDTO] = {}
    resolve_sub_line_items("balance_sheet", items, df=df, target_col="col_2024", filing_url="https://sec.gov/test")

    dcc = items.get("deferred_contract_costs")
    assert dcc is not None
    assert dcc.value == 785_100_000.0
    assert dcc.is_derived is True
    assert dcc.source_type == "derived"
