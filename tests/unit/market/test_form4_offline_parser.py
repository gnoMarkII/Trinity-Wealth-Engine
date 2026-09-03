"""Unit tests for Two-Tier SEC Form 4 Parser, Rule 10b5-1 Flagging, and Multi-Key Amendment Deduplication (Phase 4 / P0.4)."""
import pytest
from tools.market.sec_form4_pipeline import derive_canonical_transactions, parse_form4_xml
from tools.market.ownership import compute_canonical_insider_conviction


MOCK_FORM4_ORIGINAL_XML = """<?xml version="1.0"?>
<ownershipDocument>
    <documentType>4</documentType>
    <periodOfReport>2026-07-01</periodOfReport>
    <issuer>
        <issuerCik>0001262039</issuerCik>
        <issuerTradingSymbol>FTNT</issuerTradingSymbol>
    </issuer>
    <reportingOwner>
        <rptOwnerCik>0001234567</rptOwnerCik>
        <rptOwnerName>Xie Ken</rptOwnerName>
        <reportingOwnerRelationship>
            <isDirector>1</isDirector>
            <isOfficer>1</isOfficer>
            <officerTitle>Chief Executive Officer</officerTitle>
        </reportingOwnerRelationship>
    </reportingOwner>
    <nonDerivativeTable>
        <nonDerivativeTransaction>
            <securityTitle><value>Common Stock</value></securityTitle>
            <transactionDate><value>2026-07-01</value></transactionDate>
            <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
            <transactionAmounts>
                <transactionShares><value>10000</value></transactionShares>
                <transactionPricePerShare><value>160.00</value></transactionPricePerShare>
                <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
            </transactionAmounts>
            <postTransactionAmounts><sharesOwnedFollowingTransaction><value>500000</value></sharesOwnedFollowingTransaction></postTransactionAmounts>
            <ownershipNature><directOrIndirectOwnership><value>D</value></directOrIndirectOwnership></ownershipNature>
        </nonDerivativeTransaction>
    </nonDerivativeTable>
    <footnotes>
        <footnote id="F1">The sales reported were effected pursuant to a Rule 10b5-1 trading plan.</footnote>
    </footnotes>
</ownershipDocument>
"""

MOCK_FORM4_AMENDMENT_XML = """<?xml version="1.0"?>
<ownershipDocument>
    <documentType>4/A</documentType>
    <periodOfReport>2026-07-01</periodOfReport>
    <dateOfOriginalSubmission>2026-07-01</dateOfOriginalSubmission>
    <issuer>
        <issuerCik>0001262039</issuerCik>
        <issuerTradingSymbol>FTNT</issuerTradingSymbol>
    </issuer>
    <reportingOwner>
        <rptOwnerCik>0001234567</rptOwnerCik>
        <rptOwnerName>Xie Ken</rptOwnerName>
        <reportingOwnerRelationship>
            <isDirector>1</isDirector>
            <isOfficer>1</isOfficer>
            <officerTitle>Chief Executive Officer</officerTitle>
        </reportingOwnerRelationship>
    </reportingOwner>
    <nonDerivativeTable>
        <nonDerivativeTransaction>
            <securityTitle><value>Common Stock</value></securityTitle>
            <transactionDate><value>2026-07-01</value></transactionDate>
            <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
            <transactionAmounts>
                <transactionShares><value>8000</value></transactionShares>
                <transactionPricePerShare><value>162.00</value></transactionPricePerShare>
                <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
            </transactionAmounts>
            <postTransactionAmounts><sharesOwnedFollowingTransaction><value>502000</value></sharesOwnedFollowingTransaction></postTransactionAmounts>
            <ownershipNature><directOrIndirectOwnership><value>D</value></directOrIndirectOwnership></ownershipNature>
        </nonDerivativeTransaction>
    </nonDerivativeTable>
    <footnotes>
        <footnote id="F1">This amendment corrects the number of shares sold under the Rule 10b5-1 plan to 8,000.</footnote>
    </footnotes>
</ownershipDocument>
"""


def test_parse_form4_and_form4a_xml():
    """Verify parsing of Form 4 and Form 4/A with normalized original submission date and 10b5-1 flag."""
    parsed_orig = parse_form4_xml(MOCK_FORM4_ORIGINAL_XML, accession_number="0001262039-26-000001", filing_date="2026-07-01")
    assert parsed_orig["is_amendment"] is False
    assert parsed_orig["original_submission_date"] == "2026-07-01"
    assert len(parsed_orig["transactions"]) == 1
    assert parsed_orig["transactions"][0]["transaction_type"] == "S_10B5_1_FLAGGED"
    assert parsed_orig["transactions"][0]["shares"] == 10000.0

    parsed_amd = parse_form4_xml(MOCK_FORM4_AMENDMENT_XML, accession_number="0001262039-26-000005", filing_date="2026-07-05")
    assert parsed_amd["is_amendment"] is True
    assert parsed_amd["original_submission_date"] == "2026-07-01"
    assert len(parsed_amd["transactions"]) == 1
    assert parsed_amd["transactions"][0]["transaction_type"] == "S_10B5_1_FLAGGED"
    assert parsed_amd["transactions"][0]["shares"] == 8000.0


def test_derive_canonical_transactions_deduplication():
    """Verify that Form 4/A supersedes original Form 4 without double counting shares or dollar amounts."""
    parsed_orig = parse_form4_xml(MOCK_FORM4_ORIGINAL_XML, accession_number="0001262039-26-000001", filing_date="2026-07-01")
    parsed_amd = parse_form4_xml(MOCK_FORM4_AMENDMENT_XML, accession_number="0001262039-26-000005", filing_date="2026-07-05")

    # Ingest both original and amendment
    canonical_txs, quarantine_meta = derive_canonical_transactions([parsed_orig, parsed_amd])

    # MUST have exactly 1 deduplicated transaction
    assert len(canonical_txs) == 1
    assert quarantine_meta["requires_review"] is False

    tx = canonical_txs[0]
    # Must reflect the amended values (8,000 shares @ $162.00)
    assert tx["shares"] == 8000.0
    assert tx["price_per_share"] == 162.00
    assert tx["accession_number"] == "0001262039-26-000005"

    # Compute insider conviction
    conviction, flags = compute_canonical_insider_conviction(
        ticker="FTNT",
        market="US",
        canonical_transactions=canonical_txs,
        quarantine_meta=quarantine_meta,
    )
    assert conviction.rule_10b5_1_filing_count_90d == 1
    assert conviction.rule_10b5_1_lot_count_90d == 1
    assert conviction.rule_10b5_1_s_value_cents == int(8000 * 162.00 * 100)
    assert conviction.rule_10b5_1_s_value_usd_str == f"{8000 * 162.00:.2f}"
    assert conviction.signal_confidence == "high"
