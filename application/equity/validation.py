"""Pure Equity input validation used by inbound adapters."""
import re


def validate_ticker(ticker: str) -> str:
    clean = ticker.strip().upper()
    # Preserve the legacy API's accepted symbol length; path-safety is the
    # actual boundary here and provider adapters decide whether a symbol
    # exists.
    if not clean or not re.match(r"^[A-Z0-9.\-_]+$", clean):
        raise ValueError("Invalid ticker format")
    if ".." in clean or "/" in clean or "\\" in clean:
        raise ValueError("Path traversal not allowed")
    return clean
