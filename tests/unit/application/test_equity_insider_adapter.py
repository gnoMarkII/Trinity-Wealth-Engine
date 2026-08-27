import pytest

from api.db.adapters import SqliteInsiderSyncAdapter
from api.db.connection import get_connection


def _filing(accession: str) -> dict:
    return {
        "accession_number": accession,
        "issuer_cik": "0000000000",
        "ticker": "MSFT",
        "filing_url": "https://example.test/filing",
        "filed_at": "2026-08-25",
        "reporting_owner_name": "Example Officer",
        "is_officer": True,
        "transactions": [
            {
                "transaction_id": f"{accession}_tx",
                "transaction_date": "2026-08-25",
                "transaction_code": "P",
                "shares": 10.0,
                "price_per_share": 100.0,
                "acquired_or_disposed": "A",
                "normalized_weight": 1.0,
            }
        ],
    }


class _Provider:
    def __init__(self, records):
        self.records = records

    def fetch(self, ticker):
        assert ticker == "MSFT"
        return self.records


def test_insider_sync_requires_an_injected_history_provider():
    """Provider construction belongs to the composition root, never the DB adapter."""
    with pytest.raises(TypeError):
        SqliteInsiderSyncAdapter(db_path="unused.sqlite")


def test_insider_sync_persists_provider_records(tmp_path):
    db_path = str(tmp_path / "insider.sqlite")
    SqliteInsiderSyncAdapter(
        db_path=db_path,
        provider=_Provider([_filing("yf_one")]),
    ).sync("MSFT")

    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT accession_number, ticker FROM sec_form4_raw_ledger"
        ).fetchall()
        transactions = conn.execute(
            "SELECT transaction_id FROM sec_insider_transactions"
        ).fetchall()
    finally:
        conn.close()

    assert [dict(row) for row in rows] == [
        {"accession_number": "yf_one", "ticker": "MSFT"}
    ]
    assert [dict(row) for row in transactions] == [{"transaction_id": "yf_one_tx"}]


def test_insider_sync_rolls_back_all_records_on_invalid_payload(tmp_path):
    db_path = str(tmp_path / "insider-rollback.sqlite")
    provider = _Provider([_filing("yf_one"), {"ticker": "MSFT"}])

    with pytest.raises(KeyError):
        SqliteInsiderSyncAdapter(db_path=db_path, provider=provider).sync("MSFT")

    conn = get_connection(db_path)
    try:
        count = conn.execute("SELECT COUNT(*) AS count FROM sec_form4_raw_ledger").fetchone()["count"]
        tx_count = conn.execute("SELECT COUNT(*) AS count FROM sec_insider_transactions").fetchone()["count"]
    finally:
        conn.close()

    assert count == 0
    assert tx_count == 0
