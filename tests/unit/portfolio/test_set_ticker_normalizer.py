import pytest
from tools.portfolio.adapters.thai_fund.set_ticker_normalizer import (
    normalize_set_company_name,
    clean_company_name,
)


def test_clean_company_name():
    raw = "หุ้นสามัญ บริษัทเดลต้า อีเลคโทรนิคส์ (ประเทศไทย) จำกัด (มหาชน)"
    cleaned = clean_company_name(raw)
    assert "หุ้นสามัญ" not in cleaned
    assert "จำกัด" not in cleaned
    assert "เดลต้า" in cleaned


def test_normalize_set_heavyweights():
    assert normalize_set_company_name("หุ้นสามัญ บริษัทเดลต้า อีเลคโทรนิคส์ (ประเทศไทย) จำกัด (มหาชน)") == "DELTA"
    assert normalize_set_company_name("บริษัท ปตท. จำกัด (มหาชน)") == "PTT"
    assert normalize_set_company_name("บมจ. แอดวานซ์ อินโฟร์ เซอร์วิส") == "ADVANC"
    assert normalize_set_company_name("บริษัท ซีพี ออลล์ จำกัด (มหาชน)") == "CPALL"
    assert normalize_set_company_name("บริษัท กรุงเทพดุสิตเวชการ จำกัด (มหาชน)") == "BDMS"
    assert normalize_set_company_name("ธนาคารกรุงไทย จำกัด (มหาชน)") == "KTB"
    assert normalize_set_company_name("ธนาคารกสิกรไทย จำกัด (มหาชน)") == "KBANK"
    assert normalize_set_company_name("บริษัท เอสซีบี เอกซ์ จำกัด (มหาชน)") == "SCB"
    assert normalize_set_company_name("บริษัท ท่าอากาศยานไทย จำกัด (มหาชน)") == "AOT"
    assert normalize_set_company_name("บริษัท กัลฟ์ เอ็นเนอร์จี ดีเวลลอปเมนท์ จำกัด (มหาชน)") == "GULF"


def test_normalize_parentheses_ticker():
    assert normalize_set_company_name("บริษัท เดลต้า อีเลคโทรนิคส์ (DELTA)") == "DELTA"
    assert normalize_set_company_name("ปูนซิเมนต์ไทย (SCC)") == "SCC"


def test_normalize_already_valid_ticker():
    assert normalize_set_company_name("DELTA") == "DELTA"
    assert normalize_set_company_name("PTT") == "PTT"


def test_normalize_unknown():
    assert normalize_set_company_name("") is None
    assert normalize_set_company_name("บริษัท ไม่ทราบชื่อ จำกัด (มหาชน)") is None
