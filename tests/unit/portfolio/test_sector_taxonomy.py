"""Unit & Regression Tests for Bilingual Sector Taxonomy.

Verifies:
- F02 fix: Short acronym 'it' does NOT match as a substring inside 'utilities', 'reit', etc.
- Standard 11 GICS sector mapping for both Thai and English.
- Exact match vs token match precedence.
"""
import pytest
from core.portfolio.sector_taxonomy import normalize_sector_name, GICS_SECTORS_EN


def test_utilities_does_not_map_to_information_technology():
    """F02 Regression: 'utilities' contains 'it', but must map to Utilities, not IT."""
    sec_en, sec_th = normalize_sector_name("utilities")
    assert sec_en == "Utilities"
    assert sec_th == "สาธารณูปโภคพื้นฐาน"

    sec_en2, _ = normalize_sector_name("สาธารณูปโภค")
    assert sec_en2 == "Utilities"

    sec_en3, _ = normalize_sector_name("อรรถประโยชน์")
    assert sec_en3 == "Utilities"


def test_reit_does_not_map_to_information_technology():
    """F02 Regression: 'reit' contains 'it', but must map to Real Estate, not IT."""
    sec_en, sec_th = normalize_sector_name("reit")
    assert sec_en == "Real Estate"
    assert sec_th == "อสังหาริมทรัพย์"

    sec_en2, _ = normalize_sector_name("reits")
    assert sec_en2 == "Real Estate"

    sec_en3, _ = normalize_sector_name("Real Estate")
    assert sec_en3 == "Real Estate"


def test_information_technology_exact_and_standalone():
    """IT and Technology map to Information Technology."""
    assert normalize_sector_name("IT")[0] == "Information Technology"
    assert normalize_sector_name("it")[0] == "Information Technology"
    assert normalize_sector_name("technology")[0] == "Information Technology"
    assert normalize_sector_name("tech")[0] == "Information Technology"
    assert normalize_sector_name("เทคโนโลยี")[0] == "Information Technology"
    assert normalize_sector_name("กลุ่มเทคโนโลยีสารสนเทศและการสื่อสาร")[0] == "Information Technology"


def test_other_gics_sectors_normalized():
    """All 11 GICS sectors resolve correctly in English and Thai."""
    assert normalize_sector_name("Health Care")[0] == "Health Care"
    assert normalize_sector_name("healthcare")[0] == "Health Care"
    assert normalize_sector_name("การแพทย์")[0] == "Health Care"

    assert normalize_sector_name("Financials")[0] == "Financials"
    assert normalize_sector_name("ธนาคาร")[0] == "Financials"

    assert normalize_sector_name("Consumer Discretionary")[0] == "Consumer Discretionary"
    assert normalize_sector_name("สินค้าฟุ่มเฟือย")[0] == "Consumer Discretionary"

    assert normalize_sector_name("Consumer Staples")[0] == "Consumer Staples"
    assert normalize_sector_name("อาหารและเครื่องดื่ม")[0] == "Consumer Staples"

    assert normalize_sector_name("Communication Services")[0] == "Communication Services"
    assert normalize_sector_name("telecom")[0] == "Communication Services"

    assert normalize_sector_name("Industrials")[0] == "Industrials"
    assert normalize_sector_name("การขนส่งและโลจิสติกส์")[0] == "Industrials"

    assert normalize_sector_name("Energy")[0] == "Energy"
    assert normalize_sector_name("พลังงาน")[0] == "Energy"

    assert normalize_sector_name("Materials")[0] == "Materials"
    assert normalize_sector_name("ปิโตรเคมีและเคมีภัณฑ์")[0] == "Materials"

    assert normalize_sector_name("Cash")[0] == "Cash / Liquidity"
    assert normalize_sector_name("เงินสด")[0] == "Cash / Liquidity"


def test_unmapped_and_empty():
    assert normalize_sector_name("")[0] == "Other"
    assert normalize_sector_name(None)[0] == "Other"
    assert normalize_sector_name("UnmappedCryptoSector")[0] == "UnmappedCryptoSector"
