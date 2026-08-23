"""Sub-client for parsing 8-K Exhibit 99.1 Press Releases (Non-GAAP FCF and Q4 EPS)."""
import logging
from typing import Any, Optional
import pandas as pd
from tools.market.financials.domain.normalizer import table_to_grid, validate_sec_url

log = logging.getLogger(__name__)


class Sec8KSubclient:
    """Sub-client สำหรับการสกัดข้อมูลเฉพาะจาก 8-K Press Release (Q4 EPS และ Non-GAAP FCF Reconciliation)"""

    def extract_q4_eps(
        self, filings_8k: list[Any], fiscal_year: int, fy_period_end_date: str
    ) -> tuple[Optional[float], Optional[float], Optional[str]]:
        """ค้นหา Basic EPS และ Diluted EPS ของ Q4 จาก 8-K Filing (Item 2.02 / Exhibit 99.1 Press Release)"""
        if not filings_8k:
            return None, None, None

        try:
            from bs4 import BeautifulSoup
            target_dt = pd.to_datetime(fy_period_end_date)

            for filing in filings_8k:
                f_date_raw = getattr(filing, "filing_date", None)
                if not f_date_raw:
                    continue
                f_dt = pd.to_datetime(str(f_date_raw)[:10])
                days_diff = (f_dt - target_dt).days

                # 8-K earnings release is usually published 15 to 100 days after FY end
                if not (15 <= days_diff <= 100):
                    continue

                attachments = getattr(filing, "attachments", []) or []
                exhibit_att = None
                for att in attachments:
                    doc_name = str(getattr(att, "document", "")).lower()
                    desc_name = str(getattr(att, "description", "")).lower()
                    if "99" in doc_name or "ex-99" in desc_name or "99.1" in doc_name or "press release" in desc_name:
                        exhibit_att = att
                        break

                if not exhibit_att:
                    continue

                content = exhibit_att.download()
                if not content:
                    continue

                exhibit_url = validate_sec_url(getattr(exhibit_att, "url", None) or getattr(filing, "url", None))
                soup = BeautifulSoup(content, "html.parser")
                tables = soup.find_all("table")

                for t in tables:
                    text_t = t.get_text(" ")
                    if ("Three Months Ended" not in text_t and "Quarter Ended" not in text_t) or (
                        "per share" not in text_t.lower() and "diluted" not in text_t.lower()
                    ):
                        continue

                    rows = t.find_all("tr")

                    # 1. ระบุ Header row ที่มี date tokens
                    header_dates: list[str] = []
                    for r in rows:
                        cells = [c.get_text(" ").strip().replace("\xa0", " ") for c in r.find_all(["td", "th"]) if c.get_text(" ").strip()]
                        date_matches = []
                        for cell in cells:
                            for m in ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]:
                                if m in cell.lower() and str(target_dt.year) in cell:
                                    date_matches.append(cell)
                        if len(date_matches) >= 1 and not header_dates:
                            header_dates = cells
                            break

                    target_col_idx: Optional[int] = None
                    for c_idx, h_text in enumerate(header_dates):
                        if str(target_dt.year) in h_text and ("12" in h_text or "december" in h_text.lower() or "31" in h_text):
                            target_col_idx = c_idx
                            break

                    if target_col_idx is None:
                        target_col_idx = 0

                    found_basic = None
                    found_diluted = None

                    for r in rows:
                        cells = [c.get_text(" ").strip().replace("\xa0", " ") for c in r.find_all(["td", "th"]) if c.get_text(" ").strip()]
                        if not cells:
                            continue

                        row_label = cells[0].lower()
                        numeric_tokens = []
                        for cell_text in cells[1:]:
                            cleaned = cell_text.replace("$", "").replace(",", "").strip()
                            try:
                                f_val = float(cleaned)
                                numeric_tokens.append(f_val)
                            except ValueError:
                                pass

                        if not numeric_tokens or len(numeric_tokens) <= target_col_idx:
                            continue

                        if "diluted" in row_label and found_diluted is None:
                            found_diluted = numeric_tokens[target_col_idx]
                        elif "basic" in row_label and found_basic is None:
                            found_basic = numeric_tokens[target_col_idx]

                    if found_diluted is not None or found_basic is not None:
                        log.info(
                            "Found Q4 EPS in 8-K %s (%s): Basic=$%s, Diluted=$%s",
                            getattr(filing, "accession_number", ""),
                            exhibit_url,
                            found_basic,
                            found_diluted,
                        )
                        return found_basic, found_diluted, exhibit_url

        except Exception as e:
            log.warning("Error parsing 8-K Q4 EPS for FY %s: %s", fiscal_year, e)

        return None, None, None

    def extract_fcf_reconciliation(
        self,
        filings_8k: list[Any],
        fiscal_year: int,
        period_end_date: str,
        is_quarterly: bool = False,
        fiscal_quarter: Optional[int] = None,
    ) -> tuple[Optional[float], Optional[float], Optional[float], Optional[str], Optional[str], Optional[str]]:
        """Table-aware 5-step Non-GAAP Free Cash Flow Reconciliation Extractor from 8-K Press Releases (Item 2.02 / Exhibit 99.1)."""
        if not filings_8k:
            return None, None, None, None, None, "FCF_RECONCILIATION_NOT_FOUND"

        try:
            from bs4 import BeautifulSoup
            target_dt = pd.to_datetime(period_end_date)
            month_names = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
            month_abbrs = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
            target_m_name = month_names[target_dt.month - 1]
            target_m_abbr = month_abbrs[target_dt.month - 1]
            target_yr_str = str(target_dt.year)

            for filing in filings_8k:
                f_date_raw = getattr(filing, "filing_date", None)
                if not f_date_raw:
                    continue
                f_dt = pd.to_datetime(str(f_date_raw)[:10])
                days_diff = (f_dt - target_dt).days

                # 8-K earnings releases are typically filed 10 to 120 days after period end
                if not (10 <= days_diff <= 120):
                    continue

                attachments = getattr(filing, "attachments", []) or []
                exhibit_att = None
                for att in attachments:
                    doc_name = str(getattr(att, "document", "")).lower()
                    desc_name = str(getattr(att, "description", "")).lower()
                    if "99" in doc_name or "ex-99" in desc_name or "99.1" in doc_name or "press release" in desc_name or "earnings" in desc_name:
                        exhibit_att = att
                        break

                if not exhibit_att:
                    continue

                content = exhibit_att.download()
                if not content:
                    continue

                exhibit_url = validate_sec_url(getattr(exhibit_att, "url", None) or getattr(filing, "url", None))
                soup = BeautifulSoup(content, "html.parser")
                tables = soup.find_all("table")

                for t in tables:
                    text_t = t.get_text(" ").lower()
                    # Step 1: Heading/Table Filter
                    if ("operating activities" not in text_t and "operating cash flow" not in text_t) or "free cash flow" not in text_t:
                        continue

                    # Detect unit multiplier
                    unit_multiplier = 1_000_000.0
                    if "in thousands" in text_t or "(in thousands)" in text_t or "$ in thousands" in text_t:
                        unit_multiplier = 1_000.0
                    elif "in billions" in text_t or "(in billions)" in text_t or "$ in billions" in text_t:
                        unit_multiplier = 1_000_000_000.0
                    elif "in millions" in text_t or "(in millions)" in text_t or "$ in millions" in text_t:
                        unit_multiplier = 1_000_000.0

                    # Step 2: 2D Grid with Rowspan/Colspan
                    grid = table_to_grid(t)
                    if len(grid) < 3 or not grid[0]:
                        continue

                    num_cols = max(len(r) for r in grid)
                    header_depth = min(6, len(grid))
                    col_headers: dict[int, str] = {}
                    for c_idx in range(num_cols):
                        hdr_texts = []
                        for r_idx in range(header_depth):
                            if c_idx < len(grid[r_idx]):
                                hdr_texts.append(grid[r_idx][c_idx])
                        col_headers[c_idx] = " ".join(hdr_texts).lower()

                    # Step 3: Deterministic Period Column Identification
                    target_col_idx: Optional[int] = None
                    for c_idx in range(1, num_cols):
                        hdr = col_headers[c_idx]
                        has_yr = target_yr_str in hdr
                        has_mo = (
                            target_m_name in hdr
                            or target_m_abbr in hdr
                            or f" {target_dt.month}/" in hdr
                            or f"/{target_dt.month}/" in hdr
                            or f"-{target_dt.month:02d}-" in hdr
                            or f"-{target_dt.month:02d}" in hdr
                            or "30" in hdr
                            or "31" in hdr
                        )

                        if is_quarterly:
                            is_q_dur = "three months" in hdr or "quarter ended" in hdr or "3m" in hdr or ("month ended" in hdr and "twelve" not in hdr and "year" not in hdr)
                            if is_q_dur and has_yr and (has_mo or num_cols <= 4):
                                target_col_idx = c_idx
                                break
                            elif has_yr and has_mo and "twelve" not in hdr and "year ended" not in hdr:
                                target_col_idx = c_idx
                                break
                        else:
                            is_a_dur = "year ended" in hdr or "twelve months" in hdr or "12m" in hdr or "full year" in hdr or "fy" in hdr
                            if is_a_dur and has_yr:
                                target_col_idx = c_idx
                                break
                            elif has_yr and ("three months" not in hdr and "quarter ended" not in hdr):
                                target_col_idx = c_idx
                                break

                    # Fail-closed on period column
                    if target_col_idx is None:
                        continue

                    calc_fcf: Optional[float] = None
                    rep_fcf: Optional[float] = None
                    adj_fcf: Optional[float] = None
                    adj_val: Optional[float] = None
                    adj_label = ""
                    ocf_val: Optional[float] = None
                    capex_val: Optional[float] = None
                    base_fcf: Optional[float] = None
                    adjustment_items: list[tuple[str, float]] = []

                    # Step 4 & 5: Exact Semantic Row Matching & Strict Unit Parsing
                    for r_idx in range(len(grid)):
                        row = grid[r_idx]
                        if len(row) <= target_col_idx:
                            continue
                        lbl = row[0].strip().lower()
                        if not lbl and len(row) > 1:
                            lbl = row[1].strip().lower()

                        # REJECT ROWS with percentage / margin / share / rate
                        if any(term in lbl for term in ["margin", "%", "percent", "per share", "rate", "ratio", "as a percentage", "percentage of"]):
                            continue

                        cell_text = row[target_col_idx].strip()
                        if not cell_text or "%" in cell_text or "percent" in cell_text.lower():
                            continue

                        is_neg = False
                        if ("(" in cell_text and ")" in cell_text) or cell_text.startswith("-"):
                            is_neg = True
                        cleaned = cell_text.replace("$", "").replace(",", "").replace("(", "").replace(")", "").strip()
                        try:
                            f_raw = float(cleaned)
                        except ValueError:
                            continue

                        f_val = -abs(f_raw) if is_neg else f_raw

                        # Unit scaling
                        if 0 < abs(f_val) < 50_000:
                            scaled_val = f_val * unit_multiplier
                        elif 50_000 <= abs(f_val) < 50_000_000:
                            scaled_val = f_val * 1_000.0
                        else:
                            scaled_val = f_val

                        # Semantic row match
                        if "operating activities" in lbl or "operating cash flow" in lbl:
                            ocf_val = scaled_val
                        elif "purchases of property" in lbl or "capital expenditures" in lbl or "additions to property" in lbl or "payments to acquire property" in lbl:
                            capex_val = -abs(scaled_val)
                        elif "adjusted free cash flow" in lbl or "non-gaap free cash flow" in lbl or "non gaap free cash flow" in lbl:
                            adj_fcf = scaled_val
                        elif "calculated free cash flow" in lbl or "free cash flow (gaap)" in lbl:
                            calc_fcf = scaled_val
                        elif "real estate" in lbl:
                            re_val = abs(scaled_val) if (lbl.startswith("add") or lbl.startswith("plus") or not is_neg) else -abs(scaled_val)
                            adjustment_items.append((row[0].strip(), re_val))
                        elif "proceeds from" in lbl or "intellectual property" in lbl or "patent" in lbl or "settlement" in lbl or "litigation" in lbl:
                            ip_val = -abs(scaled_val) if (lbl.startswith("less") or lbl.startswith("minus") or is_neg) else abs(scaled_val)
                            adjustment_items.append((row[0].strip(), ip_val))
                        elif lbl.startswith("add:") or lbl.startswith("plus:") or lbl.startswith("less:") or lbl.startswith("minus:") or "adjustment" in lbl:
                            gen_val = -abs(scaled_val) if (lbl.startswith("less") or lbl.startswith("minus") or is_neg) else abs(scaled_val)
                            adjustment_items.append((row[0].strip(), gen_val))
                        elif "free cash flow" in lbl and "margin" not in lbl:
                            base_fcf = scaled_val

                    # Calculated FCF derivation
                    if calc_fcf is None and ocf_val is not None and capex_val is not None:
                        calc_fcf = round(ocf_val - abs(capex_val), 2)
                    elif calc_fcf is None and base_fcf is not None:
                        calc_fcf = base_fcf

                    # Reported FCF and adjustments resolution
                    if adj_fcf is not None:
                        rep_fcf = adj_fcf
                        if calc_fcf is not None and not adjustment_items:
                            adj_val = round(adj_fcf - calc_fcf, 2)
                    elif base_fcf is not None:
                        rep_fcf = base_fcf
                        if adj_fcf is not None and not adjustment_items:
                            adj_val = round(adj_fcf - base_fcf, 2)
                    elif calc_fcf is not None:
                        rep_fcf = calc_fcf

                    if adjustment_items:
                        adj_val = round(sum(v for _, v in adjustment_items), 2)
                        adj_label = ", ".join(f"{name} (${val/1e6:+.1f}M)" for name, val in adjustment_items)
                    elif adj_val is not None and adj_val != 0.0:
                        adj_label = "8-K Disclosed Non-GAAP adjustments"
                    else:
                        adj_val = 0.0

                    # Fail-closed validation check
                    if rep_fcf is not None:
                        if abs(rep_fcf) < 1_000_000:
                            log.warning("Extracted FCF %s was identified as percentage margin instead of currency", rep_fcf)
                            return None, None, None, exhibit_url, None, "FCF_RECONCILIATION_FAILED"

                        log.info(
                            "Extracted 8-K FCF for %s (Q=%s): Calc=$%s, Rep=$%s, Adj=$%s (%s)",
                            period_end_date,
                            fiscal_quarter if is_quarterly else "FY",
                            calc_fcf,
                            rep_fcf,
                            adj_val,
                            adj_label or "Non-GAAP adjustments",
                        )
                        return calc_fcf, adj_val, rep_fcf, exhibit_url, adj_label or "8-K Disclosed Non-GAAP adjustments", None

        except Exception as e:
            log.warning("Error parsing 8-K Non-GAAP FCF for %s: %s", period_end_date, e)

        return None, None, None, None, None, "FCF_RECONCILIATION_NOT_FOUND"
