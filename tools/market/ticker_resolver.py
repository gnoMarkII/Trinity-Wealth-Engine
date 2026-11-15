"""Deterministic company-name and alias resolver for equity tickers."""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

_LEGAL_SUFFIXES = frozenset({
    "INC",
    "INCORPORATED",
    "CORP",
    "CORPORATION",
    "CO",
    "COMPANY",
    "LTD",
    "LIMITED",
    "PLC",
    "PCL",
    "HOLDING",
    "HOLDINGS",
    "GROUP",
    "PUBLIC",
    "THE",
})

_COMMON_COMPANY_ALIASES: dict[str, str] = {
    # Mega-cap / Semiconductor / Software / Vault coverage
    "INTEL": "INTC",
    "INTEL CORP": "INTC",
    "INTEL CORPORATION": "INTC",
    "APPLE": "AAPL",
    "APPLE INC": "AAPL",
    "NVIDIA": "NVDA",
    "MICROSOFT": "MSFT",
    "ALPHABET": "GOOGL",
    "GOOGLE": "GOOGL",
    "AMAZON": "AMZN",
    "FACEBOOK": "META",
    "META PLATFORMS": "META",
    "TESLA": "TSLA",
    "NETFLIX": "NFLX",
    "ORACLE": "ORCL",
    "PALANTIR": "PLTR",
    "PALANTIR TECHNOLOGIES": "PLTR",
    "FORTINET": "FTNT",
    "BROADCOM": "AVGO",
    "QUALCOMM": "QCOM",
    "MICRON": "MU",
    "TSMC": "TSM",
    "TAIWAN SEMICONDUCTOR": "TSM",
    "BERKSHIRE": "BRK-B",
    "BERKSHIRE HATHAWAY": "BRK-B",
    "BRK.B": "BRK-B",
    "COSTCO": "COST",
    "WALMART": "WMT",
    "SALESFORCE": "CRM",
    "ADOBE": "ADBE",
    "PAYPAL": "PYPL",
    "SHOPIFY": "SHOP",
    "CLOUDFLARE": "NET",
    "CROWDSTRIKE": "CRWD",
    "SNOWFLAKE": "SNOW",
    "DATADOG": "DDOG",
    "ELI LILLY": "LLY",
    "JPMORGAN": "JPM",
    "JP MORGAN": "JPM",
    "GOLDMAN SACHS": "GS",
    "EXXON": "XOM",
    "EXXON MOBIL": "XOM",
    "CHEVRON": "CVX",
    "VISA": "V",
    "MASTERCARD": "MA",
    # Thai Equities (common colloquial names & bare SET tickers)
    "ปตท": "PTT.BK",
    "PTT": "PTT.BK",
    "กสิกร": "KBANK.BK",
    "กสิกรไทย": "KBANK.BK",
    "KBANK": "KBANK.BK",
    "ซีพีออลล์": "CPALL.BK",
    "CPALL": "CPALL.BK",
    "แอดวานซ์": "ADVANC.BK",
    "ADVANC": "ADVANC.BK",
    "ทอท": "AOT.BK",
    "AOT": "AOT.BK",
    "ปูนใหญ่": "SCC.BK",
    "SCC": "SCC.BK",
    "เดลต้า": "DELTA.BK",
}


def _normalize_label(text: str) -> str:
    cleaned = re.sub(r"[.,()'\"/&]+", " ", (text or "").strip().upper())
    tokens = [tok for tok in cleaned.split() if tok and tok not in _LEGAL_SUFFIXES]
    return " ".join(tokens)


def resolve_equity_ticker(
    query_or_ticker: str,
    known_records: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    """Resolve a stock ticker, company name, or alias (e.g. INTEL -> INTC) to its canonical ticker."""
    raw = (query_or_ticker or "").strip()
    if raw.startswith("$"):
        raw = raw[1:].strip()
    clean = raw.upper()
    if not clean:
        return ""

    records = list(known_records or [])

    # 1. Exact ticker match against existing vault/portfolio records wins immediately
    for rec in records:
        rec_ticker = str(rec.get("ticker") or rec.get("symbol") or "").strip().upper()
        if rec_ticker and rec_ticker == clean:
            return rec_ticker

    # 2. Check normalized alias map (exact raw or normalized without legal suffix)
    if clean in _COMMON_COMPANY_ALIASES:
        return _COMMON_COMPANY_ALIASES[clean]
    norm_query = _normalize_label(clean)
    if norm_query and norm_query in _COMMON_COMPANY_ALIASES:
        return _COMMON_COMPANY_ALIASES[norm_query]

    # 3. Dynamic lookup against known company names in Vault / Portfolio / Watchlist
    if norm_query and records:
        # 3a. Exact normalized company name match
        for rec in records:
            rec_ticker = str(rec.get("ticker") or rec.get("symbol") or "").strip().upper()
            rec_name = str(rec.get("company_name") or rec.get("label") or "").strip()
            if not rec_ticker or not rec_name:
                continue
            if _normalize_label(rec_name) == norm_query:
                return rec_ticker

        # 3b. Prefix / primary word token match when query >= 4 chars (e.g. "INTEL" -> "Intel Corporation")
        if len(norm_query) >= 4:
            for rec in records:
                rec_ticker = str(rec.get("ticker") or rec.get("symbol") or "").strip().upper()
                rec_name = str(rec.get("company_name") or rec.get("label") or "").strip()
                if not rec_ticker or not rec_name:
                    continue
                name_tokens = _normalize_label(rec_name).split()
                if name_tokens and (
                    name_tokens[0] == norm_query
                    or norm_query in name_tokens
                ):
                    return rec_ticker

    return clean


def find_company_alias_matches(
    text: str,
    known_records: Sequence[Mapping[str, Any]] | None = None,
) -> list[str]:
    """Find canonical tickers mentioned via company name or alias inside free-form user text."""
    clean_text = re.sub(r"https?://\S+", " ", text or "")
    matches: list[str] = []
    seen: set[str] = set()

    def _add(ticker: str) -> None:
        tk = ticker.strip().upper()
        if tk and tk not in seen:
            seen.add(tk)
            matches.append(tk)

    # Check known Vault/Portfolio company names first (multi-word and primary word)
    for rec in (known_records or []):
        rec_ticker = str(rec.get("ticker") or rec.get("symbol") or "").strip().upper()
        rec_name = str(rec.get("company_name") or rec.get("label") or "").strip()
        if not rec_ticker or not rec_name:
            continue
        norm_name = _normalize_label(rec_name)
        if not norm_name:
            continue
        primary_word = norm_name.split()[0]
        for candidate_phrase in (norm_name, primary_word):
            if len(candidate_phrase) < 4:
                continue
            pattern = rf"(?<![A-Za-z0-9]){re.escape(candidate_phrase)}(?![A-Za-z0-9])"
            if re.search(pattern, clean_text, re.IGNORECASE):
                _add(rec_ticker)
                break

    # Check curated company aliases
    for alias, ticker in _COMMON_COMPANY_ALIASES.items():
        if len(alias) < 3:
            continue
        if alias.isascii():
            pattern = rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])"
            if re.search(pattern, clean_text, re.IGNORECASE):
                _add(ticker)
        elif alias in clean_text:
            _add(ticker)

    return matches

