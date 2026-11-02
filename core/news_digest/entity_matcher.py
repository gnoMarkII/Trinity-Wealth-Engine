"""Financial Entity and Ticker Matcher.

Deterministic, zero-AI-cost contextual entity matcher for Thai and global assets.
Replaces fragile blind word-boundary regexes with alias mapping and ambiguity suppression.
"""
import re
from typing import Dict, List, Optional, Set, Tuple

# Acronyms and short English words that collide with tickers unless qualified with $ or aliases
COMMON_STOPWORDS_AND_ACRONYMS: Set[str] = {
    "A", "AN", "ALL", "AM", "ARE", "AS", "AT", "BE", "BY", "CAN", "DO", "FOR",
    "GO", "HAS", "HE", "IN", "IS", "IT", "ITS", "NOW", "OF", "ON", "ONE", "OR",
    "OUT", "SEE", "SO", "THE", "TO", "UP", "US", "WE", "WHO", "YOU",
    "NAV", "USD", "THB", "GDP", "CPI", "FED", "BOT", "SET", "SET50", "SET100",
    "SPY", "QQQ", "DIA", "IWM", "VTI", "VOO", "XLK", "XLF", "XLV", "XLE", "XLI",
    "XLY", "XLP", "XLU", "XLRE", "XLB", "XLC", "PVD", "RMF", "SSF", "ESG",
    "ROIC", "WACC", "DCF", "RDCF", "EBITDA", "EBIT", "GAAP", "OCF", "FCF",
    "ADTV", "SMA", "EMA", "RSI", "MACD", "PE", "PBV", "AI", "SEC", "CEO", "CFO",
    "HHI", "ETF", "ETFS", "REIT", "REITS", "GOLD", "BOND", "BONDS", "CASH",
    "BUY", "SELL", "HOLD", "TRIM", "DIP", "YTD", "QTD", "MTD", "CAGR", "EPS",
    "DPS", "ROE", "ROA", "EV", "FX", "BPS", "IRR", "NPV", "TTM", "FY",
    "CORE", "RISK", "GROWTH", "SPEC", "SAFE", "ALPHA", "BETA", "TECH", "SECTOR", "DIV",
}

# Known Thai & Global stock aliases for high-precision entity resolution
DEFAULT_COMPANY_ALIASES: Dict[str, List[str]] = {
    "DELTA": ["เดลต้า", "เดลต้า อีเลคโทรนิคส์", "DELTA.BK", "Delta Electronics"],
    "PTT": ["ปตท", "ปตท.", "PTT.BK", "ปิโตรเลียมแห่งประเทศไทย"],
    "CPALL": ["ซีพี ออลล์", "ซีพีออลล์", "CPALL.BK", "CP All"],
    "ADVANC": ["แอดวานซ์", "เอไอเอส", "AIS", "ADVANC.BK"],
    "AOT": ["การท่าอากาศยาน", "ทอท", "ทอท.", "AOT.BK"],
    "BDMS": ["กรุงเทพดุสิตเวชการ", "รพ.กรุงเทพ", "BDMS.BK"],
    "GULF": ["กัลฟ์", "กัลฟ์ เอ็นเนอร์จี", "GULF.BK"],
    "KBANK": ["กสิกรไทย", "ธนาคารกสิกร", "เคแบงก์", "KBANK.BK"],
    "SCB": ["ไทยพาณิชย์", "ธนาคารไทยพาณิชย์", "เอสซีบี", "SCB.BK"],
    "TRUE": ["ทรู", "ทรู คอร์ปอเรชั่น", "TRUE.BK"],
    "ON": ["$ON", "onsemi", "ON Semiconductor", "ออน เซมิคอนดักเตอร์"],
    "NVDA": ["$NVDA", "Nvidia", "เอ็นวิเดีย"],
    "AAPL": ["$AAPL", "Apple", "แอปเปิ้ล"],
    "MSFT": ["$MSFT", "Microsoft", "ไมโครซอฟท์"],
    "GOOGL": ["$GOOGL", "Google", "Alphabet", "กูเกิล"],
    "AMZN": ["$AMZN", "Amazon", "แอมะซอน"],
    "TSLA": ["$TSLA", "Tesla", "เทสลา"],
}

_CASHTAG_RE = re.compile(r"\$([A-Z]{1,5})")
_BOUNDARY_TICKER_RE = re.compile(r"(?<![A-Za-z0-9])([A-Z]{1,5}(?:\.[A-Z]{2})?)(?![A-Za-z0-9])")


def is_ambiguous_ticker(symbol: str) -> bool:
    """Return True if symbol is a stopword or 1-2 char word that requires explicit cashtag or alias."""
    clean = symbol.strip().upper().replace(".BK", "")
    return clean in COMMON_STOPWORDS_AND_ACRONYMS or len(clean) <= 2


def contains_ticker_boundary(text: str, symbol: str) -> bool:
    """Thai-friendly boundary search that correctly handles Thai-adjacent tickers (e.g. 'หุ้นDELTA')."""
    clean_sym = symbol.strip().replace(".BK", "")
    if not clean_sym:
        return False
    # Uses lookarounds rather than \b to avoid Thai characters being treated as \w
    pattern = rf"(?<![A-Za-z0-9]){re.escape(clean_sym)}(?![A-Za-z0-9])"
    return bool(re.search(pattern, text, re.IGNORECASE))


def find_matching_tickers(
    text: str,
    target_symbols: Set[str],
    explicit_tickers: Optional[List[str]] = None,
    custom_aliases: Optional[Dict[str, List[str]]] = None,
) -> Set[str]:
    """Extract matched ticker symbols from text with alias and cashtag support."""
    matched: Set[str] = set()
    explicit_tickers = explicit_tickers or []
    explicit_set = {t.strip().upper().lstrip("$").replace(".BK", "") for t in explicit_tickers if t}
    for clean in explicit_set:
        if clean:
            matched.add(clean)

    if not text:
        return matched

    # 1. Match explicit cashtags ($NVDA, $ON)
    for found in _CASHTAG_RE.findall(text):
        clean = found.strip().upper().replace(".BK", "")
        if clean in target_symbols or not target_symbols:
            matched.add(clean)

    # 2. Match general boundary tickers (skipping ambiguous stopwords unless in explicit list)
    for found in _BOUNDARY_TICKER_RE.findall(text):
        clean = found.strip().upper().replace(".BK", "")
        if is_ambiguous_ticker(clean) and clean not in explicit_set:
            continue
        if clean in target_symbols:
            matched.add(clean)

    # 3. Match known company aliases (Thai names, full corporate names)
    combined_aliases = dict(DEFAULT_COMPANY_ALIASES)
    if custom_aliases:
        combined_aliases.update(custom_aliases)

    for sym, aliases in combined_aliases.items():
        clean_sym = sym.strip().upper().replace(".BK", "")
        if target_symbols and clean_sym not in target_symbols:
            continue
        if clean_sym in matched:
            continue

        for alias in aliases:
            if alias.startswith("$"):
                continue
            # If alias is all ASCII letters/numbers, use boundary
            if re.match(r"^[A-Za-z0-9\s.-]+$", alias):
                pattern = rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])"
                if re.search(pattern, text, re.IGNORECASE):
                    matched.add(clean_sym)
                    break
            else:
                # Thai script or non-ASCII: substring match is natural
                if alias.lower() in text.lower():
                    matched.add(clean_sym)
                    break

    # 4. Direct scan for known target symbols with Thai-friendly boundary
    for sym in target_symbols:
        clean_sym = sym.strip().upper().replace(".BK", "")
        if clean_sym not in matched and not is_ambiguous_ticker(clean_sym):
            if contains_ticker_boundary(text, clean_sym):
                matched.add(clean_sym)

    return matched
