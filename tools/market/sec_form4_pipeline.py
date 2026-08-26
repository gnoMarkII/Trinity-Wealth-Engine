"""SEC Form 4 Ingestion Pipeline & Raw Filing Ledger Parser

สกัดข้อมูล Form 4 / Form 4A (XML) ของ SEC EDGAR เข้าสู่ Two-Tier Storage:
1. Raw Filing Ledger (`sec_form4_raw_ledger`) — Immutable append-only audit trail
2. Normalized Insider Transactions (`sec_insider_transactions`) — Canonical transaction records
"""
import sqlite3
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

# Weight mapping by transaction code
TRANSACTION_CODE_WEIGHTS = {
    "P": 1.0,   # Open market purchase (highest conviction)
    "S": 0.8,   # Open market sale
    "M": 0.3,   # Option exercise
    "A": 0.1,   # Grant/Award
    "F": 0.05,  # Tax withholding
    "G": 0.05,  # Gift
}


def parse_form4_xml(xml_content: str, accession_number: str, filing_url: str = "") -> dict:
    """แยกและสกัดข้อมูลจาก SEC Form 4 XML payload"""
    root = ET.fromstring(xml_content)

    # Issuer
    issuer_cik = root.findtext(".//issuer/issuerCik", default="").strip()
    ticker = root.findtext(".//issuer/issuerTradingSymbol", default="").strip().upper()

    # Period of report
    period_of_report = root.findtext(".//periodOfReport", default="").strip()
    filed_at = period_of_report or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Reporting Owner
    owner_node = root.find(".//reportingOwner")
    reporting_owner_cik = None
    reporting_owner_name = None
    is_director = False
    is_officer = False
    is_ten_percent_owner = False
    officer_title = None

    if owner_node is not None:
        reporting_owner_cik = owner_node.findtext(".//rptOwnerCik", default="").strip() or None
        reporting_owner_name = owner_node.findtext(".//rptOwnerName", default="").strip() or None
        
        rel = owner_node.find(".//reportingOwnerRelationship")
        if rel is not None:
            is_director = rel.findtext("isDirector", "0").strip() in ("1", "true", "TRUE")
            is_officer = rel.findtext("isOfficer", "0").strip() in ("1", "true", "TRUE")
            is_ten_percent_owner = rel.findtext("isTenPercentOwner", "0").strip() in ("1", "true", "TRUE")
            officer_title = rel.findtext("officerTitle", "").strip() or None

    # Check amendment
    doc_type = root.findtext(".//documentType", default="4").strip()
    is_amendment = doc_type == "4/A"
    amends_accession = None
    if is_amendment:
        amends_accession = root.findtext(".//amendment/amendedAccessionNumber", default="").strip() or None

    transactions = []
    # Table I: Non-Derivative Transactions
    for i, tx_node in enumerate(root.findall(".//nonDerivativeTransaction")):
        tx_date = tx_node.findtext(".//transactionDate/value", default="").strip()
        tx_code = tx_node.findtext(".//transactionCoding/transactionCode", default="P").strip().upper()
        shares_str = tx_node.findtext(".//transactionAmounts/transactionShares/value", default="0").strip()
        price_str = tx_node.findtext(".//transactionAmounts/transactionPricePerShare/value", default="0").strip()
        acq_disp = tx_node.findtext(".//transactionAmounts/transactionAcquiredDisposedCode/value", default="A").strip().upper()
        shares_following_str = tx_node.findtext(".//postTransactionAmounts/sharesOwnedFollowingTransaction/value", default="0").strip()
        ownership_nature = tx_node.findtext(".//ownershipNature/directOrIndirectOwnership/value", default="D").strip().upper()

        try:
            shares = float(shares_str)
        except ValueError:
            shares = 0.0
        try:
            price = float(price_str)
        except ValueError:
            price = 0.0
        try:
            shares_following = float(shares_following_str)
        except ValueError:
            shares_following = None

        tx_id = f"{accession_number}_nd_{i}"
        weight = TRANSACTION_CODE_WEIGHTS.get(tx_code, 0.2)

        transactions.append({
            "transaction_id": tx_id,
            "transaction_date": tx_date,
            "transaction_code": tx_code,
            "shares": shares,
            "price_per_share": price,
            "acquired_or_disposed": acq_disp,
            "shares_owned_following": shares_following,
            "ownership_nature": ownership_nature,
            "is_derivative": False,
            "normalized_weight": weight,
        })

    return {
        "accession_number": accession_number,
        "issuer_cik": issuer_cik,
        "ticker": ticker,
        "filing_url": filing_url or f"https://www.sec.gov/edgar/data/{issuer_cik}/{accession_number}",
        "filed_at": filed_at,
        "reporting_owner_cik": reporting_owner_cik,
        "reporting_owner_name": reporting_owner_name,
        "is_director": is_director,
        "is_officer": is_officer,
        "is_ten_percent_owner": is_ten_percent_owner,
        "officer_title": officer_title,
        "raw_xml_payload": xml_content,
        "is_amendment": is_amendment,
        "amends_accession_number": amends_accession,
        "transactions": transactions,
    }


def ingest_form4_data(conn: sqlite3.Connection, parsed_data: dict) -> None:
    """บันทึก parsed form 4 เข้า raw ledger และ transactions"""
    now = time.time()
    is_amendment = parsed_data.get("is_amendment", False)
    amends_accession = parsed_data.get("amends_accession_number")
    if is_amendment and amends_accession:
        conn.execute("DELETE FROM sec_insider_transactions WHERE accession_number = ?", (amends_accession,))

    conn.execute(
        """
        INSERT OR REPLACE INTO sec_form4_raw_ledger (
            accession_number, issuer_cik, ticker, filing_url, filed_at,
            reporting_owner_cik, reporting_owner_name, is_director,
            is_officer, is_ten_percent_owner, officer_title, raw_xml_payload,
            is_amendment, amends_accession_number, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            parsed_data["accession_number"],
            parsed_data["issuer_cik"],
            parsed_data["ticker"].upper(),
            parsed_data["filing_url"],
            parsed_data["filed_at"],
            parsed_data.get("reporting_owner_cik"),
            parsed_data.get("reporting_owner_name"),
            1 if parsed_data.get("is_director", False) else 0,
            1 if parsed_data.get("is_officer", False) else 0,
            1 if parsed_data.get("is_ten_percent_owner", False) else 0,
            parsed_data.get("officer_title"),
            parsed_data.get("raw_xml_payload"),
            1 if is_amendment else 0,
            amends_accession,
            now,
        ),
    )

    for tx in parsed_data.get("transactions", []):
        conn.execute(
            """
            INSERT OR REPLACE INTO sec_insider_transactions (
                transaction_id, accession_number, ticker, transaction_date,
                transaction_code, shares, price_per_share, acquired_or_disposed,
                shares_owned_following, ownership_nature, is_derivative,
                normalized_weight, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                tx["transaction_id"],
                parsed_data["accession_number"],
                parsed_data["ticker"].upper(),
                tx["transaction_date"],
                tx["transaction_code"].upper(),
                tx["shares"],
                tx["price_per_share"],
                tx["acquired_or_disposed"].upper(),
                tx.get("shares_owned_following"),
                tx.get("ownership_nature"),
                1 if tx.get("is_derivative", False) else 0,
                tx.get("normalized_weight", 1.0),
                now,
            ),
        )
    conn.commit()


def sync_insider_filings_from_yfinance(conn: sqlite3.Connection, ticker: str) -> None:
    """Deprecated compatibility wrapper; production uses SqliteInsiderSyncAdapter."""
    from tools.market.adapters.insider_provider import YFinanceInsiderHistoryAdapter

    for parsed in YFinanceInsiderHistoryAdapter().fetch(ticker):
        ingest_form4_data(conn, parsed)
