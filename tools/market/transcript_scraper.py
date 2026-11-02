"""Transcript scraper for earnings call transcripts.

Primary Source:
- StockAnalysis.com (direct HTTP with desktop headers)
Secondary Fallbacks:
- Direct WebFetch / Article paragraph extraction
- SEC EDGAR / Manual ingestion

Extracts full text with speaker attribution (Name, Title, Prepared Remarks vs Q&A).
"""
from datetime import datetime, timezone
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from bs4 import BeautifulSoup
import httpx

from core.datetime_utils import format_iso_date, parse_fiscal_quarter, parse_thai_or_iso_date
from schemas.promise_schemas import TranscriptCallDTO

log = logging.getLogger(__name__)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

DATE_PATTERNS = [
    "%b %d, %Y",  # Aug 26, 2026
    "%B %d, %Y",  # August 26, 2026
    "%Y-%m-%d",   # 2026-08-26
]


def parse_date_str(date_raw: str) -> str:
    """Parse raw date string to standard YYYY-MM-DD."""
    clean = date_raw.strip()
    d = parse_thai_or_iso_date(clean)
    if d:
        return format_iso_date(d)
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def parse_quarter_label(text: str) -> Optional[str]:
    """Extract standard quarter format FYYYYYQX from text like 'Earnings Call: Q4 2025'."""
    return parse_fiscal_quarter(text)


def discover_stockanalysis_calls(ticker: str, limit: int = 8) -> List[TranscriptCallDTO]:
    """Discover available earnings calls on StockAnalysis for a given ticker."""
    clean_ticker = ticker.strip().lower()
    url = f"https://stockanalysis.com/stocks/{clean_ticker}/transcripts/"
    
    try:
        with httpx.Client(headers=DEFAULT_HEADERS, timeout=12.0, follow_redirects=True) as client:
            resp = client.get(url)
            if resp.status_code != 200:
                log.warning("StockAnalysis transcripts index returned %d for %s", resp.status_code, ticker)
                return []
            
            soup = BeautifulSoup(resp.text, "html.parser")
            calls: List[TranscriptCallDTO] = []
            seen_hrefs = set()
            
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if "/transcripts/" not in href or href == f"/stocks/{clean_ticker}/transcripts/":
                    continue
                if href in seen_hrefs:
                    continue
                seen_hrefs.add(href)
                
                parent = a.find_parent("div")
                parent_text = parent.get_text(" | ", strip=True) if parent else a.get_text(strip=True)
                
                # Filter specifically for Earnings Call entries
                if "Earnings Call" not in parent_text and not re.search(r"Q[1-4]\s+\d{4}", parent_text):
                    continue
                
                quarter = parse_quarter_label(parent_text)
                if not quarter:
                    continue
                
                # Parse date from parent text (e.g. 'Earnings Call: Q2 2027 | Aug 26, 2026')
                date_str = ""
                parts = [p.strip() for p in parent_text.split("|")]
                if len(parts) >= 2:
                    date_str = parse_date_str(parts[1])
                else:
                    date_match = re.search(r"([A-Za-z]{3,9}\.?\s+\d{1,2},?\s+\d{4})", parent_text)
                    if date_match:
                        date_str = parse_date_str(date_match.group(1))
                    else:
                        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                
                full_url = f"https://stockanalysis.com{href}" if href.startswith("/") else href
                
                calls.append(
                    TranscriptCallDTO(
                        quarter=quarter,
                        date=date_str,
                        status="ok",
                        source="stockanalysis",
                        source_url=full_url,
                        fetch_method="direct_http",
                    )
                )
                if len(calls) >= limit:
                    break
            
            return calls
    except Exception as exc:
        log.warning("Failed to discover StockAnalysis calls for %s: %s", ticker, exc)
        return []


def parse_stockanalysis_html(html_content: str) -> Tuple[str, int]:
    """Parse StockAnalysis transcript HTML into structured text with speaker attribution."""
    soup = BeautifulSoup(html_content, "html.parser")
    article = soup.find("article") or soup.find("main") or soup
    
    blocks: List[str] = []
    current_speaker = ""
    current_title = ""
    
    for el in article.descendants:
        if not hasattr(el, "name") or not el.name:
            continue
        
        classes = el.get("class") or []
        # Speaker heading detection
        if el.name == "div" and "font-bold" in classes and ("text-lg" in classes or "md:text-xl" in classes):
            current_speaker = el.get_text(strip=True)
            sibling = el.find_next_sibling()
            if sibling and "italic" in (sibling.get("class") or []):
                current_title = sibling.get_text(strip=True)
            else:
                current_title = ""
        elif el.name == "p" and "text-default" in classes:
            p_text = el.get_text(" ", strip=True)
            if p_text:
                header = f"\n\n### {current_speaker}"
                if current_title:
                    header += f" ({current_title})"
                header += "\n"
                
                # Check if this speaker was already the last header to avoid repeating
                if blocks and blocks[-1].startswith(header.strip()):
                    blocks.append(p_text)
                else:
                    blocks.append(f"{header}{p_text}")
    
    if not blocks:
        # Fallback to .transcript-sentence elements
        sentences = [el.get_text(strip=True) for el in soup.find_all(class_=lambda c: c and "transcript-sentence" in c)]
        if sentences:
            full_text = " ".join(sentences)
            return full_text, len(full_text.split())
        
        # Generic paragraph fallback
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p") if len(p.get_text(strip=True)) > 20]
        full_text = "\n\n".join(paragraphs)
        return full_text, len(full_text.split())
    
    full_text = "\n\n".join(blocks).strip()
    words = len(full_text.split())
    return full_text, words


def fetch_transcript_text(call: TranscriptCallDTO) -> Tuple[str, int]:
    """Fetch and parse transcript text for an individual call."""
    if not call.source_url:
        raise ValueError(f"No source_url specified for call {call.quarter}")
    
    with httpx.Client(headers=DEFAULT_HEADERS, timeout=15.0, follow_redirects=True) as client:
        resp = client.get(call.source_url)
        resp.raise_for_status()
        text, words = parse_stockanalysis_html(resp.text)
        return text, words
