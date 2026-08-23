"""SEC Form 4 Filings and Insider Transactions SQLite DAO repository."""
import sqlite3
import time
from typing import Optional, List, Dict


def record_sec_form4_filing(
    conn: sqlite3.Connection,
    accession_number: str,
    issuer_cik: str,
    ticker: str,
    filing_url: str,
    filed_at: str,
    reporting_owner_cik: str | None = None,
    reporting_owner_name: str | None = None,
    is_director: bool = False,
    is_officer: bool = False,
    is_ten_percent_owner: bool = False,
    officer_title: str | None = None,
    raw_xml_payload: str | None = None,
    is_amendment: bool = False,
    amends_accession_number: str | None = None,
) -> None:
    """บันทึก Raw SEC Form 4 Filing ลง Ledger พร้อมจัดการ Form 4/A Amendment Invalidation"""
    now = time.time()
    if is_amendment and amends_accession_number:
        conn.execute("DELETE FROM sec_insider_transactions WHERE accession_number = ?", (amends_accession_number,))

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
            accession_number,
            issuer_cik,
            ticker.upper(),
            filing_url,
            filed_at,
            reporting_owner_cik,
            reporting_owner_name,
            1 if is_director else 0,
            1 if is_officer else 0,
            1 if is_ten_percent_owner else 0,
            officer_title,
            raw_xml_payload,
            1 if is_amendment else 0,
            amends_accession_number,
            now,
        ),
    )
    conn.commit()


def record_sec_insider_transaction(
    conn: sqlite3.Connection,
    transaction_id: str,
    accession_number: str,
    ticker: str,
    transaction_date: str,
    transaction_code: str,
    shares: float,
    price_per_share: float,
    acquired_or_disposed: str,
    shares_owned_following: float | None = None,
    ownership_nature: str | None = None,
    is_derivative: bool = False,
    normalized_weight: float = 1.0,
) -> None:
    """บันทึก Normalized Transaction ที่สกัดจาก Form 4"""
    now = time.time()
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
            transaction_id,
            accession_number,
            ticker.upper(),
            transaction_date,
            transaction_code.upper(),
            shares,
            price_per_share,
            acquired_or_disposed.upper(),
            shares_owned_following,
            ownership_nature,
            1 if is_derivative else 0,
            normalized_weight,
            now,
        ),
    )
    conn.commit()


def get_sec_insider_filings_and_transactions(
    conn: sqlite3.Connection, ticker: str, since_date: str | None = None
) -> list[dict]:
    """ดึงประวัติ Insider Filings และ Transactions ของ ticker จาก Ledger"""
    query = """
        SELECT 
            t.transaction_id, t.accession_number, t.ticker, t.transaction_date,
            t.transaction_code, t.shares, t.price_per_share, t.acquired_or_disposed,
            t.shares_owned_following, t.ownership_nature, t.is_derivative, t.normalized_weight,
            f.issuer_cik, f.filing_url, f.filed_at, f.reporting_owner_cik,
            f.reporting_owner_name, f.is_director, f.is_officer, f.is_ten_percent_owner,
            f.officer_title, f.is_amendment, f.amends_accession_number
        FROM sec_insider_transactions t
        JOIN sec_form4_raw_ledger f ON t.accession_number = f.accession_number
        WHERE t.ticker = ?
    """
    params = [ticker.upper()]
    if since_date:
        query += " AND t.transaction_date >= ?"
        params.append(since_date)
    query += " ORDER BY t.transaction_date DESC, f.filed_at DESC"

    cur = conn.execute(query, params)
    rows = cur.fetchall()
    return [dict(r) for r in rows]
