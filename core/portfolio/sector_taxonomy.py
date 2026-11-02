"""Bilingual Sector Taxonomy.

Normalizes Thai mutual fund sector naming conventions (from SEC/Finnomena)
into standard 11 GICS English sector names aligned with Macro Sector Rotation.
"""
from __future__ import annotations

from typing import Tuple

# Standard 11 GICS Sectors
GICS_SECTORS_EN = [
    "Information Technology",
    "Health Care",
    "Financials",
    "Consumer Discretionary",
    "Communication Services",
    "Industrials",
    "Consumer Staples",
    "Energy",
    "Utilities",
    "Real Estate",
    "Materials",
]

# Mapping table from raw sector string (Thai or English variations) to (GICS_EN, Thai_Display)
_SECTOR_TAXONOMY_MAP: list[tuple[list[str], str, str]] = [
    (
        [
            "เทคโนโลยี",
            "กลุ่มเทคโนโลยีสารสนเทศและการสื่อสาร",
            "เทคโนโลยีสารสนเทศ",
            "information technology",
            "technology",
            "tech",
            "it",
            "xlk",
        ],
        "Information Technology",
        "เทคโนโลยี",
    ),
    (
        [
            "การแพทย์",
            "สุขภาพและการแพทย์",
            "สุขภาพ",
            "กลุ่มบริการ/การแพทย์",
            "บริการทางการแพทย์",
            "health care",
            "healthcare",
            "xlv",
        ],
        "Health Care",
        "การแพทย์และสุขภาพ",
    ),
    (
        [
            "บริการด้านการเงิน",
            "กลุ่มธุรกิจการเงิน",
            "การเงิน",
            "ธนาคาร",
            "การเงินและธนาคาร",
            "financials",
            "financial services",
            "banking",
            "finance",
            "xlf",
        ],
        "Financials",
        "บริการด้านการเงิน",
    ),
    (
        [
            "สินค้าฟุ่มเฟือย/ตามวัฏจักร",
            "สินค้าฟุ่มเฟือย",
            "สินค้าอุปโภคบริโภคตามวัฏจักร",
            "กลุ่มสินค้าอุปโภคบริโภค",
            "consumer discretionary",
            "consumer cyclical",
            "cyclical",
            "xly",
        ],
        "Consumer Discretionary",
        "สินค้าฟุ่มเฟือย/ตามวัฏจักร",
    ),
    (
        [
            "สินค้าจำเป็นต่อการดำรงชีวิต",
            "สินค้าจำเป็น",
            "สินค้าอุปโภคบริโภคไม่ตามวัฏจักร",
            "อาหารและเครื่องดื่ม",
            "consumer staples",
            "consumer defensive",
            "staples",
            "xlp",
        ],
        "Consumer Staples",
        "สินค้าจำเป็นต่อการบริโภค",
    ),
    (
        [
            "บริการด้านการสื่อสาร",
            "การสื่อสาร",
            "กลุ่มเทคโนโลยีและสื่อสาร",
            "สื่อและสิ่งพิมพ์",
            "communication services",
            "communication",
            "telecommunications",
            "telecom",
            "xlc",
        ],
        "Communication Services",
        "บริการด้านการสื่อสาร",
    ),
    (
        [
            "อุตสาหกรรม",
            "กลุ่มอุตสาหกรรม",
            "บริการรับเหมาก่อสร้าง",
            "การขนส่งและโลจิสติกส์",
            "ขนส่ง",
            "industrials",
            "industrial",
            "capital goods",
            "xli",
        ],
        "Industrials",
        "อุตสาหกรรมและโลจิสติกส์",
    ),
    (
        [
            "พลังงาน",
            "กลุ่มทรัพยากร/พลังงานและสาธารณูปโภค",
            "พลังงานและสาธารณูปโภค",
            "energy",
            "oil & gas",
            "xle",
        ],
        "Energy",
        "พลังงาน",
    ),
    (
        [
            "สาธารณูปโภคพื้นฐาน",
            "อรรถประโยชน์",
            "สาธารณูปโภค",
            "utilities",
            "infrastructure",
            "xlu",
        ],
        "Utilities",
        "สาธารณูปโภคพื้นฐาน",
    ),
    (
        [
            "อสังหาริมทรัพย์",
            "กลุ่มอสังหาริมทรัพย์และก่อสร้าง",
            "พัฒนาอสังหาริมทรัพย์",
            "กองทุนรวมอสังหาริมทรัพย์",
            "real estate",
            "reit",
            "reits",
            "xlre",
        ],
        "Real Estate",
        "อสังหาริมทรัพย์",
    ),
    (
        [
            "วัสดุทั่วไป",
            "กลุ่มสินค้าอุตสาหกรรม/วัสดุอุตสาหกรรม",
            "ปิโตรเคมีและเคมีภัณฑ์",
            "วัสดุก่อสร้าง",
            "เหมืองแร่",
            "materials",
            "basic materials",
            "xlb",
        ],
        "Materials",
        "วัสดุพื้นฐานและเคมีภัณฑ์",
    ),
    (
        [
            "เงินสด",
            "เงินฝากและเงินลงทุนระยะสั้น",
            "ตราสารหนี้ระยะสั้น",
            "เงินสดและรายการเทียบเท่าเงินสด",
            "cash",
            "money market",
            "liquidity",
        ],
        "Cash / Liquidity",
        "เงินสดและสภาพคล่อง",
    ),
]


import re


def normalize_sector_name(raw_name: str) -> Tuple[str, str]:
    """Normalize raw sector name to (GICS_English, Thai_Display).

    If unmapped, returns (raw_name.strip(), raw_name.strip()).
    Uses strict token and exact phrase matching to prevent short abbreviations
    (e.g., 'it') from substring matching unrelated sectors (e.g., 'utilities', 'reit').
    """
    if not raw_name:
        return "Other", "อื่นๆ"

    cleaned = raw_name.strip().lower()

    # Pass 1: Direct match against standard GICS English
    for en_sec in GICS_SECTORS_EN:
        if en_sec.lower() == cleaned:
            for _, gics, th in _SECTOR_TAXONOMY_MAP:
                if gics == en_sec:
                    return en_sec, th
            return en_sec, en_sec

    # Pass 2: Exact match against all registered keywords
    for keywords, gics_en, thai_name in _SECTOR_TAXONOMY_MAP:
        for kw in keywords:
            if kw.strip().lower() == cleaned:
                return gics_en, thai_name

    # Pass 3: Token / whole-word match or Thai substring
    tokens = set(re.findall(r"[a-z0-9]+", cleaned))

    for keywords, gics_en, thai_name in _SECTOR_TAXONOMY_MAP:
        for kw in keywords:
            kw_l = kw.strip().lower()
            # Standalone English word token
            if re.match(r"^[a-z0-9]+$", kw_l):
                if kw_l in tokens:
                    return gics_en, thai_name
            elif any(ord(c) < 128 for c in kw_l):
                # Multi-word English phrase with word boundary
                if re.search(rf"(?<![a-z0-9]){re.escape(kw_l)}(?![a-z0-9])", cleaned):
                    return gics_en, thai_name
            else:
                # Thai keyword substring match
                if kw_l in cleaned:
                    return gics_en, thai_name

    return raw_name.strip(), raw_name.strip()


# Curated Large-Cap / Benchmark constituent ticker to GICS sector mapping
_KNOWN_TICKER_SECTOR_MAP: dict[str, str] = {
    # Information Technology
    "MSFT": "Information Technology",
    "AAPL": "Information Technology",
    "NVDA": "Information Technology",
    "AVGO": "Information Technology",
    "ORCL": "Information Technology",
    "CRM": "Information Technology",
    "ADBE": "Information Technology",
    "AMD": "Information Technology",
    "INTC": "Information Technology",
    "CSCO": "Information Technology",
    "QCOM": "Information Technology",
    "TXN": "Information Technology",
    "NOW": "Information Technology",
    "AMAT": "Information Technology",
    "MU": "Information Technology",
    "LRCX": "Information Technology",
    "PANW": "Information Technology",
    "FTNT": "Information Technology",
    "PLTR": "Information Technology",
    "ASML": "Information Technology",
    "SNPS": "Information Technology",
    "ARM": "Information Technology",
    "TSM": "Information Technology",
    "005930.KS": "Information Technology",
    "DELTA": "Information Technology",
    "HANA": "Information Technology",
    "KCE": "Information Technology",
    # Communication Services
    "GOOGL": "Communication Services",
    "GOOG": "Communication Services",
    "META": "Communication Services",
    "NFLX": "Communication Services",
    "DIS": "Communication Services",
    "CMCSA": "Communication Services",
    "TMUS": "Communication Services",
    "VZ": "Communication Services",
    "T": "Communication Services",
    "0700.HK": "Communication Services",
    "ADVANC": "Communication Services",
    "TRUE": "Communication Services",
    "INTUCH": "Communication Services",
    # Consumer Discretionary
    "AMZN": "Consumer Discretionary",
    "TSLA": "Consumer Discretionary",
    "HD": "Consumer Discretionary",
    "MCD": "Consumer Discretionary",
    "NKE": "Consumer Discretionary",
    "SBUX": "Consumer Discretionary",
    "LOW": "Consumer Discretionary",
    "BKNG": "Consumer Discretionary",
    "TJX": "Consumer Discretionary",
    "9988.HK": "Consumer Discretionary",
    "CRC": "Consumer Discretionary",
    "HMPRO": "Consumer Discretionary",
    "MINT": "Consumer Discretionary",
    # Health Care
    "UNH": "Health Care",
    "JNJ": "Health Care",
    "LLY": "Health Care",
    "ABBV": "Health Care",
    "MRK": "Health Care",
    "PFE": "Health Care",
    "TMO": "Health Care",
    "ABT": "Health Care",
    "DHR": "Health Care",
    "BMY": "Health Care",
    "AMGN": "Health Care",
    "AZN": "Health Care",
    "CSL": "Health Care",
    "BDMS": "Health Care",
    "BH": "Health Care",
    "BCH": "Health Care",
    "CHG": "Health Care",
    # Financials
    "BRK.B": "Financials",
    "BRK-B": "Financials",
    "JPM": "Financials",
    "V": "Financials",
    "MA": "Financials",
    "BAC": "Financials",
    "WFC": "Financials",
    "MS": "Financials",
    "GS": "Financials",
    "BLK": "Financials",
    "AXP": "Financials",
    "C": "Financials",
    "PYPL": "Financials",
    "KBANK": "Financials",
    "SCB": "Financials",
    "BBL": "Financials",
    "KTB": "Financials",
    "TTB": "Financials",
    # Consumer Staples
    "PG": "Consumer Staples",
    "COST": "Consumer Staples",
    "WMT": "Consumer Staples",
    "KO": "Consumer Staples",
    "PEP": "Consumer Staples",
    "PM": "Consumer Staples",
    "MDLZ": "Consumer Staples",
    "MO": "Consumer Staples",
    "CL": "Consumer Staples",
    "CPALL": "Consumer Staples",
    "CBG": "Consumer Staples",
    "OSP": "Consumer Staples",
    "CPF": "Consumer Staples",
    "TU": "Consumer Staples",
    # Industrials
    "GE": "Industrials",
    "CAT": "Industrials",
    "UNP": "Industrials",
    "HON": "Industrials",
    "RTX": "Industrials",
    "BA": "Industrials",
    "DE": "Industrials",
    "LMT": "Industrials",
    "UPS": "Industrials",
    "ROP": "Industrials",
    "FDS": "Industrials",
    "AOT": "Industrials",
    "BEM": "Industrials",
    "BTS": "Industrials",
    # Energy
    "XOM": "Energy",
    "CVX": "Energy",
    "COP": "Energy",
    "SLB": "Energy",
    "EOG": "Energy",
    "MPC": "Energy",
    "PTT": "Energy",
    "PTTEP": "Energy",
    "TOP": "Energy",
    "BANPU": "Energy",
    # Utilities
    "NEE": "Utilities",
    "SO": "Utilities",
    "DUK": "Utilities",
    "CEG": "Utilities",
    "SRE": "Utilities",
    "AEP": "Utilities",
    "GULF": "Utilities",
    "GPSC": "Utilities",
    "BGRIM": "Utilities",
    "RATCH": "Utilities",
    "EGCO": "Utilities",
    # Real Estate
    "PLD": "Real Estate",
    "AMT": "Real Estate",
    "EQIX": "Real Estate",
    "PSA": "Real Estate",
    "WELL": "Real Estate",
    "SPG": "Real Estate",
    "LHGRID": "Real Estate",
    "CPN": "Real Estate",
    "AP": "Real Estate",
    "LH": "Real Estate",
    "SPALI": "Real Estate",
    # Materials
    "LIN": "Materials",
    "SHW": "Materials",
    "FCX": "Materials",
    "APD": "Materials",
    "NEM": "Materials",
    "ECL": "Materials",
    "SCC": "Materials",
    "PTTGC": "Materials",
    "IVL": "Materials",
}


def lookup_ticker_sector(ticker: str) -> Tuple[str, str]:
    """Look up GICS sector for a known ticker, returning (GICS_EN, Thai_Display).
    
    If ticker is not in curated mapping, returns ('Other', 'อื่นๆ').
    """
    if not ticker:
        return "Other", "อื่นๆ"

    clean = ticker.strip().upper().replace(".BK", "")
    sec_en = _KNOWN_TICKER_SECTOR_MAP.get(clean) or _KNOWN_TICKER_SECTOR_MAP.get(ticker.strip().upper())
    if sec_en:
        return normalize_sector_name(sec_en)
    return "Other", "อื่นๆ"

