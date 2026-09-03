"""Unit tests for Form 4 bipartite lineage matching, cent precision lot pricing, rate limiter, and quarantine state."""
import pytest
from decimal import Decimal
from tools.market.sec_form4_pipeline import parse_form4_xml, derive_canonical_transactions, SecRateLimiter
from tools.market.ownership import compute_canonical_insider_conviction


def test_sec_rate_limiter_singleton_and_acquire():
    limiter1 = SecRateLimiter.get_instance()
    limiter2 = SecRateLimiter.get_instance()
    assert limiter1 is limiter2
    # Acquiring tokens should complete without error
    limiter1.acquire()


def test_form4_exact_cent_lot_pricing():
    # XML with 4-decimal prices e.g. $160.1376, $161.8201, $163.0821
    xml = """<?xml version="1.0"?>
    <ownershipDocument>
        <issuer>
            <issuerCik>0001262039</issuerCik>
            <issuerTradingSymbol>FTNT</issuerTradingSymbol>
        </issuer>
        <periodOfReport>2026-08-15</periodOfReport>
        <reportingOwner>
            <reportingOwnerRelationship>
                <isOfficer>1</isOfficer>
                <officerTitle>Chief Executive Officer</officerTitle>
            </reportingOwnerRelationship>
        </reportingOwner>
        <nonDerivativeTransaction>
            <securityTitle><value>Common Stock</value></securityTitle>
            <transactionDate><value>2026-08-15</value></transactionDate>
            <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
            <transactionAmounts>
                <transactionShares><value>50000</value></transactionShares>
                <transactionPricePerShare><value>160.1376</value></transactionPricePerShare>
                <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
            </transactionAmounts>
            <rule10b51Flag>1</rule10b51Flag>
        </nonDerivativeTransaction>
        <nonDerivativeTransaction>
            <securityTitle><value>Common Stock</value></securityTitle>
            <transactionDate><value>2026-08-15</value></transactionDate>
            <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
            <transactionAmounts>
                <transactionShares><value>60000</value></transactionShares>
                <transactionPricePerShare><value>161.8201</value></transactionPricePerShare>
                <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
            </transactionAmounts>
            <rule10b51Flag>1</rule10b51Flag>
        </nonDerivativeTransaction>
        <nonDerivativeTransaction>
            <securityTitle><value>Common Stock</value></securityTitle>
            <transactionDate><value>2026-08-15</value></transactionDate>
            <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
            <transactionAmounts>
                <transactionShares><value>52400</value></transactionShares>
                <transactionPricePerShare><value>163.0821</value></transactionPricePerShare>
                <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
            </transactionAmounts>
            <rule10b51Flag>1</rule10b51Flag>
        </nonDerivativeTransaction>
    </ownershipDocument>
    """
    parsed = parse_form4_xml(xml, accession_number="0001262039-26-000001", filing_date="2026-08-16")
    txs = parsed["transactions"]
    assert len(txs) == 3

    # Lot 1: 50,000 * 160.1376 = $8,006,880.00
    assert txs[0]["lot_value_cents"] == 800688000
    assert txs[0]["lot_value_usd_str"] == "8006880.00"

    # Lot 2: 60,000 * 161.8201 = $9,709,206.00
    assert txs[1]["lot_value_cents"] == 970920600
    assert txs[1]["lot_value_usd_str"] == "9709206.00"

    # Lot 3: 52,400 * 163.0821 = $8,545,502.04
    assert txs[2]["lot_value_cents"] == 854550204
    assert txs[2]["lot_value_usd_str"] == "8545502.04"

    # Total Sum = 8,006,880.00 + 9,709,206.00 + 8,545,502.04 = $26,261,588.04
    total_cents = sum(t["lot_value_cents"] for t in txs)
    assert total_cents == 2626158804


def test_form4_bipartite_amendment_matching():
    # Original filing F0 with 1 lot
    xml_orig = """<?xml version="1.0"?>
    <ownershipDocument>
        <issuer><issuerCik>0001262039</issuerCik><issuerTradingSymbol>FTNT</issuerTradingSymbol></issuer>
        <periodOfReport>2026-08-15</periodOfReport>
        <dateOfOriginalSubmission>2026-08-16</dateOfOriginalSubmission>
        <reportingOwner><rptOwnerCik>0001234567</rptOwnerCik></reportingOwner>
        <nonDerivativeTransaction>
            <securityTitle><value>Common Stock</value></securityTitle>
            <transactionDate><value>2026-08-15</value></transactionDate>
            <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
            <transactionAmounts>
                <transactionShares><value>10000</value></transactionShares>
                <transactionPricePerShare><value>160.0000</value></transactionPricePerShare>
                <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
            </transactionAmounts>
        </nonDerivativeTransaction>
    </ownershipDocument>
    """
    f0 = parse_form4_xml(xml_orig, accession_number="0001-orig", filing_date="2026-08-16")

    # Amendment filing FA with corrected price
    xml_amd = """<?xml version="1.0"?>
    <ownershipDocument>
        <issuer><issuerCik>0001262039</issuerCik><issuerTradingSymbol>FTNT</issuerTradingSymbol></issuer>
        <periodOfReport>2026-08-15</periodOfReport>
        <dateOfOriginalSubmission>2026-08-16</dateOfOriginalSubmission>
        <documentType>4/A</documentType>
        <reportingOwner><rptOwnerCik>0001234567</rptOwnerCik></reportingOwner>
        <nonDerivativeTransaction>
            <securityTitle><value>Common Stock</value></securityTitle>
            <transactionDate><value>2026-08-15</value></transactionDate>
            <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
            <transactionAmounts>
                <transactionShares><value>10000</value></transactionShares>
                <transactionPricePerShare><value>161.5000</value></transactionPricePerShare>
                <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
            </transactionAmounts>
        </nonDerivativeTransaction>
    </ownershipDocument>
    """
    fa = parse_form4_xml(xml_amd, accession_number="0001-amd", filing_date="2026-08-18")

    canonical, meta = derive_canonical_transactions([f0, fa])
    assert meta["requires_review"] is False
    assert len(canonical) == 1
    assert canonical[0]["price_per_share"] == 161.5000
    assert canonical[0]["accession_number"] == "0001-amd"


def test_form4_ambiguous_amendment_quarantined_not_zero():
    # Original filing with 1 lot, amendment with 3 lots -> ambiguous mismatch -> quarantine
    xml_orig = """<?xml version="1.0"?>
    <ownershipDocument>
        <issuer><issuerCik>0001262039</issuerCik><issuerTradingSymbol>FTNT</issuerTradingSymbol></issuer>
        <periodOfReport>2026-08-15</periodOfReport>
        <dateOfOriginalSubmission>2026-08-16</dateOfOriginalSubmission>
        <reportingOwner><rptOwnerCik>0001234567</rptOwnerCik></reportingOwner>
        <nonDerivativeTransaction>
            <securityTitle><value>Common Stock</value></securityTitle>
            <transactionDate><value>2026-08-15</value></transactionDate>
            <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
            <transactionAmounts><transactionShares><value>10000</value></transactionShares><transactionPricePerShare><value>160.00</value></transactionPricePerShare><transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode></transactionAmounts>
        </nonDerivativeTransaction>
    </ownershipDocument>
    """
    f0 = parse_form4_xml(xml_orig, accession_number="0001-orig", filing_date="2026-08-16")

    xml_amd = """<?xml version="1.0"?>
    <ownershipDocument>
        <issuer><issuerCik>0001262039</issuerCik><issuerTradingSymbol>FTNT</issuerTradingSymbol></issuer>
        <periodOfReport>2026-08-15</periodOfReport>
        <dateOfOriginalSubmission>2026-08-16</dateOfOriginalSubmission>
        <documentType>4/A</documentType>
        <reportingOwner><rptOwnerCik>0001234567</rptOwnerCik></reportingOwner>
        <nonDerivativeTransaction>
            <securityTitle><value>Common Stock</value></securityTitle>
            <transactionDate><value>2026-08-15</value></transactionDate>
            <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
            <transactionAmounts><transactionShares><value>3000</value></transactionShares><transactionPricePerShare><value>161.00</value></transactionPricePerShare><transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode></transactionAmounts>
        </nonDerivativeTransaction>
        <nonDerivativeTransaction>
            <securityTitle><value>Common Stock</value></securityTitle>
            <transactionDate><value>2026-08-15</value></transactionDate>
            <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
            <transactionAmounts><transactionShares><value>7000</value></transactionShares><transactionPricePerShare><value>162.00</value></transactionPricePerShare><transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode></transactionAmounts>
        </nonDerivativeTransaction>
    </ownershipDocument>
    """
    fa = parse_form4_xml(xml_amd, accession_number="0001-amd", filing_date="2026-08-18")

    canonical, meta = derive_canonical_transactions([f0, fa])
    assert meta["requires_review"] is True
    assert meta["quarantined_filing_count_90d"] == 2
    assert meta["quarantined_lot_count_90d"] == 3

    # Pass into conviction aggregation
    conviction, flags = compute_canonical_insider_conviction(
        ticker="FTNT",
        market="US",
        canonical_transactions=canonical,
        quarantine_meta=meta,
    )

    assert conviction.status == "requires_review"
    assert conviction.signal_confidence == "unavailable"
    assert conviction.quarantined_filing_count_90d == 2
    assert conviction.quarantined_lot_count_90d == 3
