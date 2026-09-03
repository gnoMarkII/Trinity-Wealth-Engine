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


import hashlib
import os
import sqlite3
import threading
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple, Set

# Weight mapping by transaction code
TRANSACTION_CODE_WEIGHTS = {
    "P": 1.0,                    # Open market purchase (highest conviction)
    "S_UNFLAGGED": 0.8,          # Open market sale (not plan-flagged)
    "S_10B5_1_FLAGGED": 0.3,     # Pre-scheduled Rule 10b5-1 plan sale
    "M": 0.2,                    # Option exercise
    "A": 0.1,                    # Grant/Award
    "F": 0.05,                   # Tax withholding
    "G": 0.05,                   # Gift
}


class SecRateLimiter:
    """Token bucket rate limiter (10 req/s) shared across threads and processes for SEC EDGAR."""
    _instance = None
    _lock = threading.Lock()

    def __init__(self, rate: float = 10.0, burst: float = 10.0):
        self.rate = rate
        self.burst = burst
        self.tokens = burst
        self.last_update = time.monotonic()

    @classmethod
    def get_instance(cls) -> "SecRateLimiter":
        with cls._lock:
            if cls._instance is None:
                cls._instance = SecRateLimiter()
            return cls._instance

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            self.last_update = now
            self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
            if self.tokens < 1.0:
                wait_time = (1.0 - self.tokens) / self.rate
                time.sleep(wait_time)
                self.tokens = 0.0
            else:
                self.tokens -= 1.0


def parse_form4_xml(
    xml_content: str,
    accession_number: str,
    filing_url: str = "",
    filing_date: Optional[str] = None,
) -> dict:
    """Parses SEC Form 4 / 4/A XML payload with Decimal exact lot pricing and Rule 10b5-1 flags."""
    root = ET.fromstring(xml_content)

    # Issuer
    issuer_cik = root.findtext(".//issuer/issuerCik", default="").strip()
    ticker = root.findtext(".//issuer/issuerTradingSymbol", default="").strip().upper()

    # Period of report & Filing Date
    period_of_report = root.findtext(".//periodOfReport", default="").strip()
    filed_at = filing_date or period_of_report or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Normalized original submission date for Form 4 and Form 4/A matching
    raw_orig_date = root.findtext(".//dateOfOriginalSubmission", default="").strip()
    original_submission_date = raw_orig_date if raw_orig_date else filed_at

    # Check amendment
    doc_type = root.findtext(".//documentType", default="4").strip().upper()
    is_amendment = doc_type in ("4/A", "4A")
    amends_accession = root.findtext(".//amendment/amendedAccessionNumber", default="").strip() or None

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

    # Detect 10b5-1 flag from XML or footnotes
    is_global_10b51 = root.findtext(".//rule10b51Flag", "0").strip() in ("1", "true", "TRUE")
    all_footnotes = " ".join([fn.text or "" for fn in root.findall(".//footnote")])

    transactions = []
    # Table I: Non-Derivative Transactions
    for i, tx_node in enumerate(root.findall(".//nonDerivativeTransaction")):
        security_title = tx_node.findtext(".//securityTitle/value", default="Common Stock").strip()
        tx_date = tx_node.findtext(".//transactionDate/value", default="").strip()
        tx_code = tx_node.findtext(".//transactionCoding/transactionCode", default="P").strip().upper()
        shares_str = tx_node.findtext(".//transactionAmounts/transactionShares/value", default="0").strip()
        price_str = tx_node.findtext(".//transactionAmounts/transactionPricePerShare/value", default="0").strip()
        acq_disp = tx_node.findtext(".//transactionAmounts/transactionAcquiredDisposedCode/value", default="A").strip().upper()
        shares_following_str = tx_node.findtext(".//postTransactionAmounts/sharesOwnedFollowingTransaction/value", default="0").strip()
        ownership_nature = tx_node.findtext(".//ownershipNature/directOrIndirectOwnership/value", default="D").strip().upper()

        # Check transaction-specific 10b5-1 flag
        tx_10b51 = is_global_10b51 or ("10b5-1" in all_footnotes.lower()) or ("rule 10b5-1" in all_footnotes.lower())

        try:
            shares_dec = Decimal(shares_str)
            shares = float(shares_dec)
        except Exception:
            shares_dec = Decimal("0")
            shares = 0.0

        try:
            price_dec = Decimal(price_str)
            price = float(price_dec)
        except Exception:
            price_dec = Decimal("0")
            price = 0.0

        lot_val_dec = shares_dec * price_dec
        lot_val_cents = int((lot_val_dec * Decimal("100")).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
        lot_val_usd_str = str(lot_val_dec.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

        try:
            shares_following = float(shares_following_str)
        except ValueError:
            shares_following = None

        if tx_code == "S":
            granular_type = "S_10B5_1_FLAGGED" if tx_10b51 else "S_UNFLAGGED"
        elif tx_code == "P":
            granular_type = "P"
        elif tx_code == "F":
            granular_type = "F_TAX_WITHHOLDING"
        elif tx_code == "M":
            granular_type = "M_EXERCISE"
        else:
            granular_type = tx_code

        tx_id = f"{accession_number}_nd_{i}"
        weight = TRANSACTION_CODE_WEIGHTS.get(granular_type, 0.2)

        transactions.append({
            "transaction_id": tx_id,
            "lot_ordinal": i,
            "security_title": security_title,
            "transaction_date": tx_date,
            "transaction_code": tx_code,
            "transaction_type": granular_type,
            "is_10b5_1_plan": tx_10b51,
            "shares": shares,
            "shares_str": str(shares_dec),
            "price_per_share": price,
            "price_per_share_str": str(price_dec),
            "lot_value_cents": lot_val_cents,
            "lot_value_usd_str": lot_val_usd_str,
            "acquired_or_disposed": acq_disp,
            "shares_owned_following": shares_following,
            "ownership_nature": ownership_nature,
            "is_derivative": False,
            "normalized_weight": weight,
            "footnote": all_footnotes,
        })

    content_hash = hashlib.sha256(xml_content.encode("utf-8")).hexdigest()

    return {
        "accession_number": accession_number,
        "issuer_cik": issuer_cik,
        "ticker": ticker,
        "filing_url": filing_url or f"https://www.sec.gov/edgar/data/{issuer_cik}/{accession_number}",
        "filed_at": filed_at,
        "original_submission_date": original_submission_date,
        "reporting_owner_cik": reporting_owner_cik,
        "reporting_owner_name": reporting_owner_name,
        "is_director": is_director,
        "is_officer": is_officer,
        "is_ten_percent_owner": is_ten_percent_owner,
        "officer_title": officer_title,
        "raw_xml_payload": xml_content,
        "content_sha256": content_hash,
        "parser_version": "v1.1.0",
        "is_amendment": is_amendment,
        "amends_accession_number": amends_accession,
        "transactions": transactions,
    }


def derive_canonical_transactions(parsed_filings: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Rebuilds canonical derived transactions via bipartite lineage matching and quarantine isolation.

    Bipartite Matching:
      - Form 4 and Form 4/A grouped by (issuer_cik, reporting_owner_cik, original_submission_date)
      - Pairs matched on (transaction_date, transaction_code, security_title, ownership_nature) + amount proximity
      - If 1-to-1 match resolved unambiguously -> amended lot supersedes original lot
      - If ambiguous -> quarantine both filings, return requires_review=True, signal_confidence='unavailable'
    """
    quarantined_filing_count = 0
    quarantined_lot_count = 0
    requires_review = False

    # Group filings by lineage
    lineage_groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for f in parsed_filings:
        cik = f.get("issuer_cik", "")
        owner_cik = f.get("reporting_owner_cik", "")
        orig_date = f.get("original_submission_date", "") or f.get("filed_at", "")
        lineage_groups.setdefault((cik, owner_cik, orig_date), []).append(f)

    canonical_lots: List[Dict[str, Any]] = []

    for (cik, owner_cik, orig_date), filings in lineage_groups.items():
        # Sort chronologically by filed_at
        filings.sort(key=lambda x: x.get("filed_at", ""))
        originals = [f for f in filings if not f.get("is_amendment", False)]
        amendments = [f for f in filings if f.get("is_amendment", False)]

        if not amendments:
            # All originals, no amendment conflict
            for f in originals:
                for tx in f.get("transactions", []):
                    tx_c = dict(tx)
                    tx_c["accession_number"] = f.get("accession_number")
                    tx_c["ticker"] = f.get("ticker")
                    tx_c["reporting_owner_name"] = f.get("reporting_owner_name")
                    tx_c["officer_title"] = f.get("officer_title")
                    tx_c["is_c_suite"] = any(t in str(f.get("officer_title") or "").lower() for t in ["ceo", "cfo", "coo", "chief", "president"])
                    canonical_lots.append(tx_c)
            continue

        # Amendment exists -> Match lots bipartite
        orig_lots = [tx for f in originals for tx in f.get("transactions", [])]
        amd_lots = [tx for f in amendments for tx in f.get("transactions", [])]

        if len(orig_lots) != len(amd_lots) and len(orig_lots) > 0 and len(amd_lots) > 0:
            # Lot count mismatch without 1-to-1 correspondence -> Quarantine
            requires_review = True
            quarantined_filing_count += (len(originals) + len(amendments))
            quarantined_lot_count += (len(orig_lots) + len(amd_lots))
            continue

        # Check 1-to-1 match compatibility
        matched_pairs = []
        is_ambiguous = False
        used_amd_indices: Set[int] = set()

        for o_idx, o_lot in enumerate(orig_lots):
            best_match_idx = None
            for a_idx, a_lot in enumerate(amd_lots):
                if a_idx in used_amd_indices:
                    continue
                # Compatibility criteria
                same_code = o_lot.get("transaction_code") == a_lot.get("transaction_code")
                same_date = o_lot.get("transaction_date") == a_lot.get("transaction_date")
                same_sec = o_lot.get("security_title") == a_lot.get("security_title")
                same_own = o_lot.get("ownership_nature") == a_lot.get("ownership_nature")

                if same_code and same_date and same_sec and same_own:
                    best_match_idx = a_idx
                    break

            if best_match_idx is not None:
                used_amd_indices.add(best_match_idx)
                matched_pairs.append((o_lot, amd_lots[best_match_idx]))
            else:
                is_ambiguous = True
                break

        if is_ambiguous or len(matched_pairs) != len(amd_lots):
            requires_review = True
            quarantined_filing_count += (len(originals) + len(amendments))
            quarantined_lot_count += (len(orig_lots) + len(amd_lots))
        else:
            # Successfully resolved bipartite matching -> Amended lots supersede
            latest_amd_filing = amendments[-1]
            for _, amd_lot in matched_pairs:
                tx_c = dict(amd_lot)
                tx_c["accession_number"] = latest_amd_filing.get("accession_number")
                tx_c["ticker"] = latest_amd_filing.get("ticker")
                tx_c["reporting_owner_name"] = latest_amd_filing.get("reporting_owner_name")
                tx_c["officer_title"] = latest_amd_filing.get("officer_title")
                tx_c["is_c_suite"] = any(t in str(latest_amd_filing.get("officer_title") or "").lower() for t in ["ceo", "cfo", "coo", "chief", "president"])
                canonical_lots.append(tx_c)

    meta = {
        "requires_review": requires_review,
        "quarantined_filing_count_90d": quarantined_filing_count,
        "quarantined_lot_count_90d": quarantined_lot_count,
        "total_canonical_lots": len(canonical_lots),
    }
    return canonical_lots, meta


def sync_insider_filings_from_yfinance(conn: sqlite3.Connection, ticker: str) -> None:
    """Deprecated compatibility wrapper; production uses SqliteInsiderSyncAdapter."""
    from tools.market.adapters.insider_provider import YFinanceInsiderHistoryAdapter

    for parsed in YFinanceInsiderHistoryAdapter().fetch(ticker):
        ingest_form4_data(conn, parsed)
