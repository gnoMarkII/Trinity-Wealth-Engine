"""Security & Issuer Identity Value Objects and Asset Classification.

Architecture Layer: core/portfolio/ (Pure Domain)
Standardizes multi-market security identification, deduplication, and asset classification
without arbitrary ticker manipulation or cross-market alias contamination.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal, Optional

AssetClass = Literal["EQUITY", "REIT", "FUND", "ETF", "GOLD", "BOND", "CASH", "UNKNOWN"]
MarketType = Literal["US", "TH", "GLOBAL"]

# Known Thai REITs / Property Funds
_THAI_REIT_SYMBOLS = {
    "CPNREIT", "WHART", "FTREIT", "LHHOTEL", "AIMIRT", "ALLY", "DIF",
    "JASIF", "POPF", "FUTUREPF", "INETREIT", "BOFFICE", "CPNCG",
    "GVREIT", "SPRIME", "B-WORK", "PROSPECT", "GROREIT", "IMPACT",
    "BAREIT", "HYDRO", "TLHPF", "URBNPF", "QHHR", "QHPF", "TIF1",
}

# Known Gold indicators
_GOLD_SYMBOLS = {"GOLD", "GLD", "IAU", "XAU", "ARR_GOLD", "GC=F"}

# Known Thai SET stock symbols without .BK
_CURATED_SET_SYMBOLS = {
    "DELTA", "PTT", "PTTEP", "ADVANC", "CPALL", "BDMS", "KTB", "KBANK",
    "SCB", "AOT", "GULF", "TRUE", "CPN", "SCC", "BBL", "INTUCH", "MINT",
    "OR", "BH", "CRC", "BEM", "TOP", "GPSC", "IVL", "HMPRO", "WHA",
    "BGRIM", "LH", "CBG", "OSP", "BANPU", "TU", "COM7", "TIDLOR", "MTC",
    "SAWAD", "KCE", "HANA", "CENTEL", "AMATA", "AP", "SPALI", "CHG",
    "BCH", "BTS", "RATCH", "EGCO", "TTB", "TLI", "GLOBAL", "ITC",
}


@dataclass(frozen=True)
class SecurityIdentity:
    """Canonical value object for an investable security / holding identity."""
    security_id: str      # e.g., "US:MSFT", "TH:CPALL", "GLOBAL:9988.HK"
    issuer_id: str        # e.g., "MSFT", "CPALL"
    symbol: str           # e.g., "MSFT", "CPALL"
    market: MarketType
    currency: str         # "USD", "THB", etc.
    asset_type: AssetClass
    company_name: str = ""

    def __hash__(self) -> int:
        return hash((self.security_id, self.asset_type))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SecurityIdentity):
            return False
        return (self.security_id, self.asset_type) == (other.security_id, other.asset_type)


def classify_asset_type(
    symbol: str,
    asset_type_raw: Optional[str] = None,
    name: Optional[str] = None,
) -> AssetClass:
    """Classify holding or constituent into one of 8 canonical asset classes."""
    sym = (symbol or "").strip().upper()
    raw_type = (asset_type_raw or "").strip().upper()
    nm = (name or "").strip().lower()

    # 1. Cash / Liquidity
    if any(k in raw_type for k in ("CASH", "CURRENCY", "DEPOSIT", "MONEY MARKET", "LIQUIDITY")) or sym in ("CASH", "THB", "USD"):
        return "CASH"

    # 2. Funds / ETFs / PVD (Checked before Gold/Bonds to avoid misidentifying gold mutual funds like SCBGOLD or PVD_SINSA_GOLD as direct physical assets)
    if any(k in raw_type for k in ("MUTUAL_FUND", "FUND", "PVD", "RMF", "SSF", "THAIESG")) or sym.startswith("PVD_") or "กองทุน" in nm:
        if bool(re.search(r"\betf\b", raw_type, re.IGNORECASE)) or bool(re.search(r"\betf\b", nm, re.IGNORECASE)):
            return "ETF"
        return "FUND"

    if raw_type == "ETF" or bool(re.search(r"\betf\b", nm, re.IGNORECASE)):
        return "ETF"

    # 3. Gold / Precious Metals
    if sym in _GOLD_SYMBOLS or raw_type == "GOLD" or "ทองคำ" in nm or ("GOLD" in sym and not any(k in raw_type for k in ("FUND", "PVD", "RMF", "SSF"))):
        return "GOLD"

    # 4. Fixed Income / Bonds
    if any(k in raw_type for k in ("BOND", "DEBENTURE", "TREASURY", "FIXED INCOME", "ตราสารหนี้")) or "พันธบัตร" in nm or "หุ้นกู้" in nm:
        return "BOND"

    # 5. REITs & Property Infrastructure Funds
    clean_sym = sym.replace(".BK", "")
    if (
        clean_sym in _THAI_REIT_SYMBOLS
        or raw_type in ("REIT", "PROPERTY FUND")
        or "ทรัสต์เพื่อการลงทุนในอสังหาริมทรัพย์" in nm
        or "กองทุนรวมอสังหาริมทรัพย์" in nm
        or "กองทุนรวมโครงสร้างพื้นฐาน" in nm
        or bool(re.search(r"\breit\b", nm, re.IGNORECASE))
    ):
        return "REIT"

    # 6. Equities
    if raw_type in ("EQUITY", "STOCK", "COMMON_STOCK") or "หุ้นสามัญ" in nm:
        return "EQUITY"

    # Default heuristic if raw_type not provided
    if clean_sym:
        return "EQUITY"

    return "UNKNOWN"


def resolve_security_identity(
    symbol: str,
    market: Optional[str] = None,
    company_name: Optional[str] = None,
    asset_type: Optional[str] = None,
) -> SecurityIdentity:
    """Resolve raw ticker, market hint, and company name into canonical SecurityIdentity."""
    raw_sym = (symbol or "").strip()
    upper_sym = raw_sym.upper()
    c_name = (company_name or "").strip()

    # Determine market
    market_resolved: MarketType = "US"
    clean_ticker = upper_sym

    if upper_sym.endswith(".BK"):
        market_resolved = "TH"
        clean_ticker = upper_sym[:-3].strip()
    elif market and market.upper() in ("US", "GLOBAL"):
        market_resolved = market.upper()  # type: ignore
        clean_ticker = upper_sym.strip()
    elif market and market.upper() == "TH":
        market_resolved = "TH"
        clean_ticker = upper_sym.replace(".BK", "").strip()
    elif clean_ticker in _CURATED_SET_SYMBOLS or clean_ticker in _THAI_REIT_SYMBOLS:
        market_resolved = "TH"
    elif "." in upper_sym:
        # Check non-TH dotted symbol (e.g., 9988.HK, BRK.B)
        suffix = upper_sym.split(".")[-1]
        if suffix in ("HK", "T", "L", "DE", "SS"):
            market_resolved = "GLOBAL"
        else:
            # E.g. BRK.B in US
            market_resolved = "US"

    # Currency
    if market_resolved == "TH":
        currency = "THB"
    elif market_resolved == "US":
        currency = "USD"
    else:
        currency = "USD"  # fallback default for global

    classified_type = classify_asset_type(clean_ticker, asset_type_raw=asset_type, name=c_name)

    security_id = f"{market_resolved}:{clean_ticker}"
    issuer_id = clean_ticker

    return SecurityIdentity(
        security_id=security_id,
        issuer_id=issuer_id,
        symbol=clean_ticker,
        market=market_resolved,
        currency=currency,
        asset_type=classified_type,
        company_name=c_name or clean_ticker,
    )
