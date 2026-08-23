"""Financial Domain Normalizer: Parsing Helpers, XBRL/HTML Column Resolution, and Sub-item Derivations."""
import logging
import urllib.parse
from typing import Any, Literal, Optional
import pandas as pd
from tools.market.financials.domain.calculations import finite_or_none
from tools.market.financials.domain.constants import EDGAR_CONCEPT_SYNONYMS
from tools.market.financials.domain.models import FinancialCellDTO

log = logging.getLogger(__name__)


def validate_sec_url(url: Optional[str]) -> Optional[str]:
    """ตรวจสอบว่า URL ของเอกสารเป็น HTTPS บน sec.gov อย่างเข้มงวด"""
    if not url:
        return None
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme == "https" and parsed.netloc in ["www.sec.gov", "sec.gov"]:
            return url
    except Exception:
        pass
    return None


def collect_distinct_authoritative_filings(filings_iterable: Any, target_distinct_count: int) -> list[Any]:
    """คัดเลือก filings โดยจัดกลุ่มตาม period_of_report และเลือกฉบับ authoritative ล่าสุด (Tie-break ด้วย accession_no)"""
    sorted_filings = sorted(
        filings_iterable,
        key=lambda f: (
            str(getattr(f, "filing_date", "")),
            str(getattr(f, "acceptance_datetime", "")),
            str(getattr(f, "accession_no", None) or getattr(f, "accession_number", "")),
        ),
        reverse=True,
    )

    periods_map: dict[str, Any] = {}
    for filing in sorted_filings:
        period_raw = getattr(filing, "period_of_report", None)
        if not period_raw:
            log.warning(
                "Skipping filing %s: missing period_of_report",
                getattr(filing, "accession_no", None) or getattr(filing, "accession_number", "N/A"),
            )
            continue
        period_str = str(period_raw)[:10]
        if period_str not in periods_map:
            periods_map[period_str] = filing
        if len(periods_map) >= target_distinct_count:
            break

    authoritative_list = list(periods_map.values())
    authoritative_list.sort(key=lambda f: str(getattr(f, "period_of_report", "")))
    return authoritative_list


def table_to_grid(table_elem: Any) -> list[list[str]]:
    """แปลง HTML table เป็น 2D matrix/grid ที่รองรับ rowspan และ colspan อย่างสมบูรณ์"""
    rows = table_elem.find_all("tr")
    if not rows:
        return []

    max_data_cols = 0
    for r in rows:
        cells = r.find_all(["td", "th"])
        total_span = sum(int(c.get("colspan", 1)) for c in cells)
        max_data_cols = max(max_data_cols, total_span)

    matrix: list[list[Optional[str]]] = []
    for row_idx, r in enumerate(rows):
        while row_idx >= len(matrix):
            matrix.append([])
        cells = r.find_all(["td", "th"])
        if not cells:
            continue

        row_span_sum = sum(int(c.get("colspan", 1)) for c in cells)
        col_offset = 0
        if row_idx == 0 and row_span_sum == max_data_cols - 1 and all(c.name == "th" for c in cells):
            col_offset = 1

        col_idx = col_offset
        for cell in cells:
            while col_idx < len(matrix[row_idx]) and matrix[row_idx][col_idx] is not None:
                col_idx += 1

            text = cell.get_text(" ").strip().replace("\xa0", " ")

            try:
                rowspan = int(cell.get("rowspan", 1))
            except (ValueError, TypeError):
                rowspan = 1
            try:
                colspan = int(cell.get("colspan", 1))
            except (ValueError, TypeError):
                colspan = 1

            for r_offset in range(rowspan):
                target_r = row_idx + r_offset
                while target_r >= len(matrix):
                    matrix.append([])
                for c_offset in range(colspan):
                    target_c = col_idx + c_offset
                    while target_c >= len(matrix[target_r]):
                        matrix[target_r].append(None)
                    matrix[target_r][target_c] = text
            col_idx += colspan

    grid: list[list[str]] = []
    for r in matrix:
        grid.append([c if c is not None else "" for c in r])
    return grid


def select_xbrl_column(
    columns: list[str],
    period_end_date: str,
    statement_type: Literal["income", "balance_sheet", "cash_flow"],
    is_quarterly: bool,
    fiscal_quarter: Optional[int] = None,
) -> tuple[Optional[str], Optional[int], Optional[str]]:
    """เลือกคอลัมน์ XBRL ที่ตรงกับประเภทงบการเงินและงวดเวลาอย่างเข้มงวด

    Returns:
        tuple[Optional[str], Optional[int], Optional[str]]: (target_column_name, source_duration_days, source_column_label)
    """
    matching_cols = [c for c in columns if period_end_date in c]
    if not matching_cols:
        return None, None, None

    if statement_type == "balance_sheet":
        instant_cols = [
            c for c in matching_cols
            if "3m" not in c.lower() and "6m" not in c.lower() and "9m" not in c.lower() and "12m" not in c.lower() and "ytd" not in c.lower()
        ]
        target = instant_cols[0] if instant_cols else matching_cols[0]
        return target, None, target

    if not is_quarterly:
        return matching_cols[0], 365, matching_cols[0]

    # Quarterly Duration (10-Q)
    if statement_type == "income":
        col_3m = [c for c in matching_cols if "(3m)" in c.lower() or "3 months" in c.lower() or ("3m" in c.lower() and "ytd" not in c.lower())]
        if col_3m:
            return col_3m[0], 91, col_3m[0]
        non_ytd = [c for c in matching_cols if "ytd" not in c.lower() and "6m" not in c.lower() and "9m" not in c.lower()]
        if non_ytd:
            return non_ytd[0], 91, non_ytd[0]
        return matching_cols[0], 91, matching_cols[0]

    elif statement_type == "cash_flow":
        if fiscal_quarter == 1:
            col_q1 = [c for c in matching_cols if "(3m)" in c.lower() or "3m" in c.lower() or "ytd" not in c.lower()]
            target = col_q1[0] if col_q1 else matching_cols[0]
            return target, 91, target
        elif fiscal_quarter == 2:
            col_q2 = [c for c in matching_cols if "6m" in c.lower() or "ytd" in c.lower() or "6 months" in c.lower()]
            target = col_q2[0] if col_q2 else matching_cols[0]
            return target, 182, target
        elif fiscal_quarter == 3:
            col_q3 = [c for c in matching_cols if "9m" in c.lower() or "ytd" in c.lower() or "9 months" in c.lower()]
            target = col_q3[0] if col_q3 else matching_cols[0]
            return target, 273, target
        else:
            return matching_cols[0], 91, matching_cols[0]

    return matching_cols[0], 91, matching_cols[0]


def extract_canonical_cell_from_df(df: pd.DataFrame, canonical_key: str, col_name: str) -> tuple[Optional[float], Optional[str]]:
    """ค้นหาค่าตัวเลขของ canonical key ใน DataFrame โดยอ้างอิงจาก standard_concept หรือ concept synonyms อย่างเข้มงวด"""
    if df.empty or col_name not in df.columns:
        return None, None

    synonyms = EDGAR_CONCEPT_SYNONYMS.get(canonical_key, [])

    # 1. ค้นหาจาก standard_concept column (Exact case-insensitive match)
    if "standard_concept" in df.columns:
        for syn in synonyms:
            matches = df[df["standard_concept"].astype(str).str.strip().str.lower() == syn.lower()]
            if not matches.empty:
                val = matches[col_name].dropna().values
                if len(val) > 0:
                    res = finite_or_none(val[0])
                    if res is not None:
                        matched_concept = str(matches["concept"].values[0]) if "concept" in matches.columns else syn
                        return res, matched_concept

    # 2. ค้นหาจาก concept column (raw tag ending with exact syn e.g. us-gaap_AssetsCurrent or AssetsCurrent)
    if "concept" in df.columns:
        for syn in synonyms:
            for idx, row in df.iterrows():
                c = str(row.get("concept", "")).strip()
                tag_name = c.split("_")[-1].split(":")[-1]
                if tag_name.lower() == syn.lower():
                    v = finite_or_none(row.get(col_name))
                    if v is not None:
                        return v, c

    return None, None


def resolve_sub_line_items(
    statement_type: Literal["income", "balance_sheet"],
    items: dict[str, FinancialCellDTO],
    df: Optional[pd.DataFrame] = None,
    target_col: Optional[str] = None,
    filing_url: Optional[str] = None,
) -> None:
    """ช่วยคำนวณและเติมเต็มรายการ Sub-items / Derivations เช่น SG&A, Total Debt, Total Equity, DCC"""
    if statement_type == "income":
        sm_val = items.get("sales_and_marketing", FinancialCellDTO()).value
        ga_val = items.get("general_and_administrative", FinancialCellDTO()).value
        sga_val = items.get("selling_general_admin", FinancialCellDTO()).value
        if sga_val is None and sm_val is not None and ga_val is not None:
            items["selling_general_admin"] = FinancialCellDTO(
                value=round(sm_val + ga_val, 2),
                source_type="derived",
                source_concept=None,
                source_filing_url=filing_url,
                derivation="S&M + G&A",
                is_derived=True,
            )

    elif statement_type == "balance_sheet":
        # 1. Deferred Contract Costs resolution
        dcc_cell = items.get("deferred_contract_costs", FinancialCellDTO())
        if (dcc_cell.value is None or dcc_cell.source_type == "unavailable") and df is not None and target_col is not None:
            dcc_curr, dcc_curr_tag = extract_canonical_cell_from_df(df, "deferred_contract_costs_current", target_col)
            dcc_noncurr, dcc_noncurr_tag = extract_canonical_cell_from_df(df, "deferred_contract_costs_noncurrent", target_col)
            if dcc_curr is not None or dcc_noncurr is not None:
                combined_dcc = round((dcc_curr or 0.0) + (dcc_noncurr or 0.0), 2)
                items["deferred_contract_costs"] = FinancialCellDTO(
                    value=combined_dcc,
                    source_type="derived" if (dcc_curr is not None and dcc_noncurr is not None) else "reported",
                    source_concept=f"{dcc_curr_tag or ''} + {dcc_noncurr_tag or ''}".strip(" +"),
                    source_filing_url=filing_url,
                    derivation="Deferred Contract Costs (Current) + Deferred Contract Costs (Non-Current)" if (dcc_curr is not None and dcc_noncurr is not None) else None,
                    formula="deferred_contract_costs_current + deferred_contract_costs_noncurrent" if (dcc_curr is not None and dcc_noncurr is not None) else None,
                    input_items=["deferred_contract_costs_current", "deferred_contract_costs_noncurrent"] if (dcc_curr is not None and dcc_noncurr is not None) else None,
                    is_derived=(dcc_curr is not None and dcc_noncurr is not None),
                )

        # 2. Total Debt derivation
        tot_debt = items.get("total_debt", FinancialCellDTO()).value
        st_debt = items.get("short_term_debt", FinancialCellDTO()).value
        lt_debt = items.get("long_term_debt", FinancialCellDTO()).value
        if tot_debt is None and (st_debt is not None or lt_debt is not None):
            items["total_debt"] = FinancialCellDTO(
                value=round((st_debt or 0.0) + (lt_debt or 0.0), 2),
                source_type="derived",
                source_concept=None,
                source_filing_url=filing_url,
                derivation="Short-Term Debt + Long-Term Debt",
                is_derived=True,
            )

        # 3. Total Equity & Non-controlling Interests resolution
        tot_eq = items.get("total_equity", FinancialCellDTO()).value
        stk_eq = items.get("stockholders_equity", FinancialCellDTO()).value
        nci_cell = items.get("noncontrolling_interests", FinancialCellDTO())
        nci = nci_cell.value if (nci_cell.source_type == "reported" and nci_cell.value is not None) else None

        if nci is None:
            items["noncontrolling_interests"] = FinancialCellDTO(
                value=0.0,
                source_type="not_applicable",
                derivation="No Non-controlling Interests disclosed",
                source_filing_url=filing_url,
                is_derived=False,
            )
            if tot_eq is None and stk_eq is not None:
                items["total_equity"] = FinancialCellDTO(
                    value=stk_eq,
                    source_type="derived",
                    source_filing_url=filing_url,
                    derivation="Total Equity = Stockholders' Equity (NCI not applicable)",
                    formula="stockholders_equity",
                    input_items=["stockholders_equity"],
                    is_derived=True,
                )
            elif stk_eq is None and tot_eq is not None:
                items["stockholders_equity"] = FinancialCellDTO(
                    value=tot_eq,
                    source_type="derived",
                    source_filing_url=filing_url,
                    derivation="Stockholders' Equity = Total Equity (NCI not applicable)",
                    formula="total_equity",
                    input_items=["total_equity"],
                    is_derived=True,
                )
        else:
            if tot_eq is None and stk_eq is not None:
                items["total_equity"] = FinancialCellDTO(
                    value=round(stk_eq + nci, 2),
                    source_type="derived",
                    source_filing_url=filing_url,
                    derivation="Stockholders' Equity + Non-controlling Interests",
                    formula="stockholders_equity + noncontrolling_interests",
                    input_items=["stockholders_equity", "noncontrolling_interests"],
                    is_derived=True,
                )
            elif stk_eq is None and tot_eq is not None:
                items["stockholders_equity"] = FinancialCellDTO(
                    value=round(tot_eq - nci, 2),
                    source_type="derived",
                    source_filing_url=filing_url,
                    derivation="Total Equity - Non-controlling Interests",
                    formula="total_equity - noncontrolling_interests",
                    input_items=["total_equity", "noncontrolling_interests"],
                    is_derived=True,
                )
