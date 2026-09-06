"""SCBAM Fund Click HTML Email Parser (Hexagonal Architecture).

Parses order confirmation emails from SCBAM Fund Click (fundclick.scbam@scb.co.th).
Extracts transaction date, effective date, transaction number, fund symbol,
account number, and amount.
"""
from dataclasses import dataclass
from decimal import Decimal
import re
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


def _parse_date_to_iso(date_str: str) -> str:
    """Convert DD/MM/YYYY to YYYY-MM-DD."""
    parts = date_str.strip().split('/')
    if len(parts) == 3:
        d, m, y = parts[0], parts[1], parts[2]
        return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
    return date_str.strip()


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
    m_tx_date = re.search(r"(?:วันที่ทำรายการ|Transaction Date):[\s\n]*(\d{2}/\d{2}/\d{4})(?:[\s\n]*(?:เวลา\s*)?(\d{2}:\d{2}:\d{2}))?", text)
    if not m_tx_date:
        log.warning("SCBAM parser: Transaction Date not found in email.")
        return None

    tx_date_raw = m_tx_date.group(1)
    tx_time_raw = m_tx_date.group(2) or "00:00:00"
    tx_date = _parse_date_to_iso(tx_date_raw)

    # 2. Effective Date
    # Thai: วันที่คำสั่งมีผล: 29/12/2025
    # English: Effective Date: 29/12/2025
    m_eff_date = re.search(r"(?:วันที่คำสั่งมีผล|Effective Date):\s*(\d{2}/\d{2}/\d{4})", text)
    if not m_eff_date:
        log.warning("SCBAM parser: Effective Date not found in email.")
        return None
    effective_date = _parse_date_to_iso(m_eff_date.group(1))

    # 3. Transaction Number
    # Thai: เลขที่รายการ: ADV-2025-12-26-15.58.25.068380
    # English: Transaction Number: ADV-2025-12-26-15.58.25.068380
    m_tx_no = re.search(r"(?:เลขที่รายการ|Transaction Number):\s*([A-Za-z0-9\-\.]+)", text)
    if not m_tx_no:
        log.warning("SCBAM parser: Transaction Number not found in email.")
        return None
    transaction_no = m_tx_no.group(1).strip()

    # 4. Account Number
    # Thai: เลขที่บัญชีกองทุน: 000027893865
    # English: Unit Holder Number: 000027893865
    m_acc = re.search(r"(?:เลขที่บัญชีกองทุน|Unit Holder Number):\s*(\d+)", text)
    account_no = m_acc.group(1).strip() if m_acc else ""

    # 5. Fund Code
    # Thai: กองทุน: SCBS&P500E (กองทุนเปิดไทยพาณิชย์หุ้นยูเอส (ชนิดช่องทางอิเล็กทรอนิกส์))
    # English: Fund Code: SCBS&P500E
    m_fund = re.search(r"(?:(?<!บัญชี)กองทุน|Fund Code):\s*([A-Za-z0-9\(\)\-\&_]+)", text)
    if not m_fund:
        log.warning("SCBAM parser: Fund Code not found in email.")
        return None
    fund_code = m_fund.group(1).strip()

    # 6. Amount
    # Thai: จำนวนเงิน: 35,000.00 บาท
    # English: Amount (THB): 35,000.00
    m_amt = re.search(r"(?:จำนวนเงิน|Amount\s*(?:\(THB\))?):\s*([\d,]+\.\d{2})", text)
    if not m_amt:
        log.warning("SCBAM parser: Amount not found in email.")
        return None
    amt_str = m_amt.group(1).replace(",", "").strip()
    amount = Decimal(amt_str)

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
