"""SET Ticker Normalizer.

Maps Thai legal company names and descriptions to standard SET stock symbols.
E.g., "หุ้นสามัญ บริษัทเดลต้า อีเลคโทรนิคส์ (ประเทศไทย) จำกัด (มหาชน)" -> "DELTA"
"""
from __future__ import annotations

import re
from typing import Optional

# Prefixes and suffixes to strip from company names
_PREFIX_PATTERNS = [
    r"^หุ้นสามัญ\s*(?:บริษัท)?\s*",
    r"^หุ้นบุริมสิทธิ\s*(?:บริษัท)?\s*",
    r"^หน่วยลงทุน\s*",
    r"^บริษัท\s*",
    r"^บมจ\.\s*",
    r"^กองทุนรวมโครงสร้างพื้นฐาน\s*",
    r"^กองทุนรวมสิทธิการเช่าอสังหาริมทรัพย์\s*",
    r"^ทรัสต์เพื่อการลงทุนในสิทธิการเช่าอสังหาริมทรัพย์\s*",
    r"^ทรัสต์เพื่อการลงทุนในอสังหาริมทรัพย์\s*",
]

_SUFFIX_PATTERNS = [
    r"\s*จำกัด\s*\(\s*มหาชน\s*\)$",
    r"\s*จำกัด\s*\(มหาชน\)$",
    r"\s*\(มหาชน\)$",
    r"\s*จำกัด$",
    r"\s*\(ประเทศไทย\)$",
]

# Canonical Thai company name keywords to SET ticker
_THAI_NAME_TO_TICKER: dict[str, str] = {
    # SET50 / SET100 Heavyweights
    "เดลต้า": "DELTA",
    "เดลต้า อีเลคโทรนิคส์": "DELTA",
    "ปตท.": "PTT",
    "ปตท": "PTT",
    "ปตท. สำรวจและผลิตปิโตรเลียม": "PTTEP",
    "ปตท.สผ.": "PTTEP",
    "แอดวานซ์ อินโฟร์ เซอร์วิส": "ADVANC",
    "แอดวานซ์": "ADVANC",
    "ซีพี ออลล์": "CPALL",
    "ซีพีออลล์": "CPALL",
    "กรุงเทพดุสิตเวชการ": "BDMS",
    "ธนาคารกรุงไทย": "KTB",
    "ธนาคารกสิกรไทย": "KBANK",
    "กสิกรไทย": "KBANK",
    "เอสซีบี เอกซ์": "SCB",
    "ไทยพาณิชย์": "SCB",
    "ธนาคารไทยพาณิชย์": "SCB",
    "ท่าอากาศยานไทย": "AOT",
    "กัลฟ์": "GULF",
    "กัลฟ์ เอ็นเนอร์จี": "GULF",
    "กัลฟ์ เอ็นเนอร์จี ดีเวลลอปเมนท์": "GULF",
    "ทรู คอร์ปอเรชั่น": "TRUE",
    "ทรู": "TRUE",
    "เซ็นทรัลพัฒนา": "CPN",
    "ปูนซิเมนต์ไทย": "SCC",
    "ธนาคารกรุงเทพ": "BBL",
    "อินทัช": "INTUCH",
    "อินทัช โฮลดิ้งส์": "INTUCH",
    "เซ็นทรัล รีเทล": "CRC",
    "เซ็นทรัล รีเทล คอร์ปอเรชั่น": "CRC",
    "ไมเนอร์ อินเตอร์เนชั่นแนล": "MINT",
    "ไมเนอร์": "MINT",
    "โรงพยาบาลบำรุงราษฎร์": "BH",
    "บำรุงราษฎร์": "BH",
    "ธนาคารทหารไทยธนชาต": "TTB",
    "ทหารไทยธนชาต": "TTB",
    "เจริญโภคภัณฑ์อาหาร": "CPF",
    "ดับบลิวเอชเอ": "WHA",
    "ดับบลิวเอชเอ คอร์ปอเรชั่น": "WHA",
    "อินโดรามา เวนเจอร์ส": "IVL",
    "ไทยออยล์": "TOP",
    "แลนด์แอนด์เฮ้าส์": "LH",
    "โฮม โปรดักส์ เซ็นเตอร์": "HMPRO",
    "โฮมโปร": "HMPRO",
    "โอสถสภา": "OSP",
    "คาราบาวกรุ๊ป": "CBG",
    "คอมเซเว่น": "COM7",
    "พลังงานบริสุทธิ์": "EA",
    "โกลบอล เพาเวอร์ ซินเนอร์ยี่": "GPSC",
    "บี.กริม เพาเวอร์": "BGRIM",
    "บีกริม": "BGRIM",
    "บัตรกรุงไทย": "KTC",
    "เมืองไทย แคปปิตอล": "MTC",
    "ศรีสวัสดิ์ คอร์ปอเรชั่น": "SAWAD",
    "ศรีสวัสดิ์": "SAWAD",
    "เงินติดล้อ": "TIDLOR",
    "ทิสโก้ไฟแนนเชียลกรุ๊ป": "TISCO",
    "ทิสโก้": "TISCO",
    "บีทีเอส กรุ๊ป โฮลดิ้งส์": "BTS",
    "บีทีเอส": "BTS",
    "ทางด่วนและรถไฟฟ้ากรุงเทพ": "BEM",
    "บ้านปู": "BANPU",
    "ไทยยูเนี่ยน กรุ๊ป": "TU",
    "ไทยยูเนี่ยน": "TU",
    "ผลิตไฟฟ้า": "EGCO",
    "ราช กรุ๊ป": "RATCH",
    "เคซีอี อิเลคโทรนิคส์": "KCE",
    "เคซีอี": "KCE",
    "ฮานา ไมโครอิเล็คโทรนิคส": "HANA",
    "ฮานา": "HANA",
    "ไอ-เทล คอร์ปอเรชั่น": "ITC",
    "อมตะ คอร์ปอเรชัน": "AMATA",
    "อมตะ": "AMATA",
    "ทุนธนชาต": "TCAP",
    "อิชิตัน กรุ๊ป": "ICHI",
    "อิชิตัน": "ICHI",
    "เซ็ปเป้": "SAPPE",
    "เอสไอเอสบี": "SISB",
    "เมก้า ไลฟ์ไซแอ็นซ์": "MEGA",
    "โรงพยาบาลจุฬารัตน์": "CHG",
    "แพลน บี มีเดีย": "PLANB",
    "แพลนบี": "PLANB",
    # Infrastructure & REITs
    "โทรคมนาคม ดิจิทัล": "DIF",
    "บรอดแบนด์อินเทอร์เน็ต จัสมิน": "JASIF",
    "ซีพีเอ็น รีเทล โกรท": "CPNREIT",
    "CPN รีเทล โกรท": "CPNREIT",
    "ดับบลิวเอชเอ พรีเมี่ยม โกรท": "WHART",
    "อิมแพ็ค โกรท": "IMPACT",
}

# English common names to SET ticker
_ENG_NAME_TO_TICKER: dict[str, str] = {
    "DELTA ELECTRONICS": "DELTA",
    "PTT PCL": "PTT",
    "PTT EXPLORATION AND PRODUCTION": "PTTEP",
    "ADVANCED INFO SERVICE": "ADVANC",
    "CP ALL": "CPALL",
    "BANGKOK DUSIT MEDICAL SERVICES": "BDMS",
    "KRUNG THAI BANK": "KTB",
    "KASIKORNBANK": "KBANK",
    "SCB X": "SCB",
    "SIAM COMMERCIAL BANK": "SCB",
    "AIRPORTS OF THAILAND": "AOT",
    "GULF ENERGY DEVELOPMENT": "GULF",
    "TRUE CORPORATION": "TRUE",
    "CENTRAL PATTANA": "CPN",
    "SIAM CEMENT": "SCC",
    "BANGKOK BANK": "BBL",
    "INTOUCH HOLDINGS": "INTUCH",
    "CENTRAL RETAIL": "CRC",
    "MINOR INTERNATIONAL": "MINT",
    "BUMRUNGRAD HOSPITAL": "BH",
    "TMBTHANACHART BANK": "TTB",
    "CHAROEN POKPHAND FOODS": "CPF",
    "WHA CORPORATION": "WHA",
    "INDORAMA VENTURES": "IVL",
    "THAI OIL": "TOP",
    "LAND AND HOUSES": "LH",
    "HOME PRODUCT CENTER": "HMPRO",
    "OSOTSPA": "OSP",
    "CARABAO GROUP": "CBG",
    "COM7": "COM7",
    "ENERGY ABSOLUTE": "EA",
    "GLOBAL POWER SYNERGY": "GPSC",
    "B.GRIMM POWER": "BGRIM",
    "KRUNGTHAI CARD": "KTC",
    "MUANGTHAI CAPITAL": "MTC",
    "SRISAWAD CORPORATION": "SAWAD",
    "NGERN TID LOR": "TIDLOR",
    "TISCO FINANCIAL GROUP": "TISCO",
    "BTS GROUP HOLDINGS": "BTS",
    "BANGKOK EXPRESSWAY AND METRO": "BEM",
    "BANPU": "BANPU",
    "THAI UNION GROUP": "TU",
}


def clean_company_name(raw_name: str) -> str:
    """Strip common legal prefixes and suffixes from Thai company name."""
    cleaned = raw_name.strip()
    for pat in _PREFIX_PATTERNS:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE).strip()
    for pat in _SUFFIX_PATTERNS:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned


def normalize_set_company_name(name: str) -> Optional[str]:
    """Map Thai or English company name string to SET stock symbol.
    
    Returns uppercase SET ticker if resolved, else None.
    """
    return normalize_set_ticker(name)

_GENERIC_NON_TICKERS = {
    "CASH", "BOND", "BONDS", "GOLD", "OTHER", "OTHERS", "NAV", "TOTAL", "DEPOSIT", "THB", "USD", "CURRENCY"
}


def normalize_set_ticker(name: str) -> Optional[str]:
    """Resolves Thai or English holding name from Finnomena API to canonical SET ticker.

    Examples:
        "บมจ.เดลต้า อีเลคโทรนิคส์ (ประเทศไทย)" -> "DELTA"
        "PTT EXPLORATION & PROD" -> "PTTEP"
        "ADVANC" -> "ADVANC"
        "CASH" -> None (not a SET ticker)
    """
    if not name:
        return None

    cleaned = clean_company_name(name)
    if not cleaned:
        return None

    upper_cleaned = cleaned.upper()
    if upper_cleaned in _GENERIC_NON_TICKERS:
        return None

    # 1. Exact match against known Thai keywords
    if cleaned in _THAI_NAME_TO_TICKER:
        return _THAI_NAME_TO_TICKER[cleaned]

    # 2. Substring match for longer Thai descriptions (sorted by length descending to match PTTEP before PTT)
    for th_key, ticker in sorted(_THAI_NAME_TO_TICKER.items(), key=lambda x: len(x[0]), reverse=True):
        if th_key in cleaned or th_key in name:
            return ticker

    # 3. English name exact / substring match (sorted by length descending)
    upper_name = name.upper()
    for eng_key, ticker in sorted(_ENG_NAME_TO_TICKER.items(), key=lambda x: len(x[0]), reverse=True):
        if eng_key in upper_name:
            return ticker

    # 4. Check if cleaned name is already a valid SET ticker format (2-8 uppercase letters/numbers)
    if re.fullmatch(r"^[A-Z0-9]{2,8}$", upper_cleaned) and upper_cleaned not in _GENERIC_NON_TICKERS:
        return upper_cleaned

    # 5. Extract ticker if in parentheses: e.g. "บริษัท เดลต้า (DELTA)"
    paren_match = re.search(r"\(([A-Z0-9]{2,8})\)", name)
    if paren_match:
        cand = paren_match.group(1).upper()
        if cand not in _GENERIC_NON_TICKERS:
            return cand

    return None
