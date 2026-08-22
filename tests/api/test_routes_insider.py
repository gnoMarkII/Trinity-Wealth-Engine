from datetime import datetime, timezone, timedelta
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.state_db import get_connection, record_sec_form4_filing, record_sec_insider_transaction
from tools.market.sec_form4_pipeline import parse_form4_xml, ingest_form4_data

client = TestClient(app)


@pytest.fixture(autouse=True)
def override_require_session():
    from api.auth import require_session
    app.dependency_overrides[require_session] = lambda: {"user_id": "mock_user"}
    yield
    app.dependency_overrides = {}


SAMPLE_FORM4_XML = """<?xml version="1.0"?>
<ownershipDocument>
    <documentType>4</documentType>
    <periodOfReport>2026-08-15</periodOfReport>
    <issuer>
        <issuerCik>0000320193</issuerCik>
        <issuerTradingSymbol>AAPL</issuerTradingSymbol>
    </issuer>
    <reportingOwner>
        <reportingOwnerId>
            <rptOwnerCik>0001234567</rptOwnerCik>
            <rptOwnerName>Cook Tim</rptOwnerName>
        </reportingOwnerId>
        <reportingOwnerRelationship>
            <isDirector>1</isDirector>
            <isOfficer>1</isOfficer>
            <officerTitle>Chief Executive Officer</officerTitle>
        </reportingOwnerRelationship>
    </reportingOwner>
    <nonDerivativeTable>
        <nonDerivativeTransaction>
            <transactionDate><value>2026-08-15</value></transactionDate>
            <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
            <transactionAmounts>
                <transactionShares><value>50000</value></transactionShares>
                <transactionPricePerShare><value>220.50</value></transactionPricePerShare>
                <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
            </transactionAmounts>
            <postTransactionAmounts>
                <sharesOwnedFollowingTransaction><value>3500000</value></sharesOwnedFollowingTransaction>
            </postTransactionAmounts>
            <ownershipNature>
                <directOrIndirectOwnership><value>D</value></directOrIndirectOwnership>
            </ownershipNature>
        </nonDerivativeTransaction>
    </nonDerivativeTable>
</ownershipDocument>
"""


SAMPLE_FORM4A_AMENDMENT_XML = """<?xml version="1.0"?>
<ownershipDocument>
    <documentType>4/A</documentType>
    <periodOfReport>2026-08-15</periodOfReport>
    <issuer>
        <issuerCik>0000320193</issuerCik>
        <issuerTradingSymbol>AAPL</issuerTradingSymbol>
    </issuer>
    <reportingOwner>
        <reportingOwnerId>
            <rptOwnerCik>0001234567</rptOwnerCik>
            <rptOwnerName>Cook Tim</rptOwnerName>
        </reportingOwnerId>
        <reportingOwnerRelationship>
            <isDirector>1</isDirector>
            <isOfficer>1</isOfficer>
            <officerTitle>Chief Executive Officer</officerTitle>
        </reportingOwnerRelationship>
    </reportingOwner>
    <amendment>
        <amendedAccessionNumber>0000320193-26-000100</amendedAccessionNumber>
    </amendment>
    <nonDerivativeTable>
        <nonDerivativeTransaction>
            <transactionDate><value>2026-08-15</value></transactionDate>
            <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
            <transactionAmounts>
                <transactionShares><value>60000</value></transactionShares>
                <transactionPricePerShare><value>220.50</value></transactionPricePerShare>
                <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
            </transactionAmounts>
            <postTransactionAmounts>
                <sharesOwnedFollowingTransaction><value>3510000</value></sharesOwnedFollowingTransaction>
            </postTransactionAmounts>
            <ownershipNature>
                <directOrIndirectOwnership><value>D</value></directOrIndirectOwnership>
            </ownershipNature>
        </nonDerivativeTransaction>
    </nonDerivativeTable>
</ownershipDocument>
"""


def test_parse_form4_xml_standard_purchase():
    parsed = parse_form4_xml(SAMPLE_FORM4_XML, accession_number="0000320193-26-000100")
    assert parsed["ticker"] == "AAPL"
    assert parsed["reporting_owner_name"] == "Cook Tim"
    assert parsed["is_officer"] is True
    assert parsed["is_director"] is True
    assert parsed["officer_title"] == "Chief Executive Officer"
    assert parsed["is_amendment"] is False
    assert len(parsed["transactions"]) == 1

    tx = parsed["transactions"][0]
    assert tx["transaction_code"] == "P"
    assert tx["shares"] == 50000.0
    assert tx["price_per_share"] == 220.50
    assert tx["acquired_or_disposed"] == "A"
    assert tx["normalized_weight"] == 1.0


def test_form4_amendment_handling_replaces_prior_transactions():
    conn = get_connection()
    # 1. Ingest original
    parsed_orig = parse_form4_xml(SAMPLE_FORM4_XML, accession_number="0000320193-26-000100")
    ingest_form4_data(conn, parsed_orig)

    # 2. Ingest amendment
    parsed_amend = parse_form4_xml(SAMPLE_FORM4A_AMENDMENT_XML, accession_number="0000320193-26-000101")
    ingest_form4_data(conn, parsed_amend)

    # Check that original transactions for 0000320193-26-000100 were purged
    cur = conn.execute("SELECT * FROM sec_insider_transactions WHERE accession_number = ?", ("0000320193-26-000100",))
    assert len(cur.fetchall()) == 0

    # Check new transaction exists with updated shares (60000)
    cur = conn.execute("SELECT * FROM sec_insider_transactions WHERE accession_number = ?", ("0000320193-26-000101",))
    rows = cur.fetchall()
    assert len(rows) == 1
    assert rows[0]["shares"] == 60000.0


def test_get_equity_insider_filings_cluster_buy_and_net_shares():
    conn = get_connection()
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Ingest 3 distinct buyers for MSFT within 30 days to trigger cluster buy
    insiders = ["Satya Nadella", "Amy Hood", "Brad Smith"]
    for i, name in enumerate(insiders):
        acc = f"0000789019-26-000{i+1}0"
        record_sec_form4_filing(
            conn=conn,
            accession_number=acc,
            issuer_cik="0000789019",
            ticker="MSFT",
            filing_url=f"https://www.sec.gov/edgar/data/789019/{acc}",
            filed_at=today_str,
            reporting_owner_name=name,
            is_officer=True,
            officer_title="Executive",
        )
        record_sec_insider_transaction(
            conn=conn,
            transaction_id=f"{acc}_tx_0",
            accession_number=acc,
            ticker="MSFT",
            transaction_date=today_str,
            transaction_code="P",
            shares=10000.0,
            price_per_share=420.0,
            acquired_or_disposed="A",
            normalized_weight=1.0,
        )

    res = client.get("/api/equity/MSFT/insider-filings?range=1mo&interval=1d")
    assert res.status_code == 200
    data = res.json()
    assert data["ticker"] == "MSFT"
    assert data["net_shares_30d"] == 30000.0
    assert data["cluster_buy_count"] == 3
    assert len(data["filings"]) == 3
    assert all(f["is_cluster_buy"] is True for f in data["filings"])
