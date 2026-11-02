"""SCBAM Fund Click HTML Email Parser (Hexagonal Architecture).

Parses order confirmation emails from SCBAM Fund Click (fundclick.scbam@scb.co.th).
Extracts transaction date, effective date, transaction number, fund symbol,
account number, and amount.
"""
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional
from bs4 import BeautifulSoup

from core.logger import get_logger

log = get_logger(__name__)


@dataclass
class SCBAMRawOrder:
    tx_date: str          # YYYY-MM-DD
    tx_time: str          # HH:MM:SS
    effective_date: str   # YYYY-MM-DD (NAV allocation date)
    transaction_no: str   # e.g. ADV-2025-12-26-15.58.25.068380
    account_no: str       # e.g. 000027893865
    fund_code: str        # e.g. SCBS&P500E, SCBGVALUE(E)
    amount: Decimal       # e.g. 35000.00
    action: str = "BUY"
    raw_html: str = ""


from core.datetime_utils import format_iso_date, parse_thai_or_iso_date, extract_date_by_labels
from core.financial_parsers import extract_financial_decimal, extract_labeled_decimal
from core.text_utils import extract_labeled_value


def _parse_date_to_iso(date_str: str) -> str:
    """Convert DD/MM/YYYY (AD or BE) to YYYY-MM-DD."""
    d = parse_thai_or_iso_date(date_str)
    return format_iso_date(d) if d else date_str.strip()


def parse_scbam_fundclick_html(html_content: str) -> Optional[SCBAMRawOrder]:
    """Parse HTML body of SCBAM Fund Click purchase confirmation email.
    
    Returns SCBAMRawOrder if successfully extracted, None otherwise.
    """
    if not html_content or not html_content.strip():
        return None

    soup = BeautifulSoup(html_content, "html.parser")
    text = soup.get_text(separator="\n")

    # 1. Transaction Date and Time
    # Thai: วันที่ทำรายการ: 26/12/2025 เวลา 15:58:51 น.
    # English: Transaction Date: 26/12/2025 15:58:51 hrs.
    tx_d = extract_date_by_labels(text, ["Transaction Date", "วันที่ทำรายการ"])
    if not tx_d:
        log.warning("SCBAM parser: Transaction Date not found in email.")
        return None

    tx_date = format_iso_date(tx_d)
    tx_time_raw = extract_labeled_value(
        text,
        ["เวลา", "Transaction Date", "วันที่ทำรายการ"],
        value_pattern=r"(\d{2}:\d{2}:\d{2})",
    ) or "00:00:00"

    # 2. Effective Date
    # Thai: วันที่คำสั่งมีผล: 29/12/2025
    # English: Effective Date: 29/12/2025
    eff_d = extract_date_by_labels(text, ["Effective Date", "วันที่คำสั่งมีผล"])
    if not eff_d:
        log.warning("SCBAM parser: Effective Date not found in email.")
        return None
    effective_date = format_iso_date(eff_d)

    # 3. Transaction Number
    # Thai: เลขที่รายการ: ADV-2025-12-26-15.58.25.068380
    # English: Transaction Number: ADV-2025-12-26-15.58.25.068380
    transaction_no = extract_labeled_value(
        text,
        ["Transaction Number", "เลขที่รายการ"],
        value_pattern=r"([A-Za-z0-9\-\.]+)",
    )
    if not transaction_no:
        log.warning("SCBAM parser: Transaction Number not found in email.")
        return None

    # 4. Account Number
    # Thai: เลขที่บัญชีกองทุน: 000027893865
    # English: Unit Holder Number: 000027893865
    account_no = extract_labeled_value(
        text,
        ["Unit Holder Number", "เลขที่บัญชีกองทุน"],
        value_pattern=r"(\d+)",
    ) or ""

    # 5. Fund Code
    # Thai: กองทุน: SCBS&P500E (กองทุนเปิดไทยพาณิชย์หุ้นยูเอส (ชนิดช่องทางอิเล็กทรอนิกส์))
    # English: Fund Code: SCBS&P500E
    fund_code = extract_labeled_value(
        text,
        ["Fund Code", "(?<!บัญชี)กองทุน"],
        value_pattern=r"([A-Za-z0-9\(\)\-\&_]+)",
    )
    if not fund_code:
        log.warning("SCBAM parser: Fund Code not found in email.")
        return None

    # 6. Amount
    # Thai: จำนวนเงิน: 35,000.00 บาท
    # English: Amount (THB): 35,000.00
    amount = extract_labeled_decimal(
        text,
        ["Amount (THB)", "Amount", "จำนวนเงิน"],
    )
    if amount is None:
        log.warning("SCBAM parser: Amount not found in email.")
        return None

    # 7. Action
    action = "BUY"
    if "ขาย" in text or "Redemption" in text:
        action = "SELL"

    return SCBAMRawOrder(
        tx_date=tx_date,
        tx_time=tx_time_raw,
        effective_date=effective_date,
        transaction_no=transaction_no,
        account_no=account_no,
        fund_code=fund_code,
        amount=amount,
        action=action,
        raw_html=html_content,
    )
