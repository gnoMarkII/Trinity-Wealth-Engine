import concurrent.futures

import os

from datetime import datetime

import yfinance as yf

from fredapi import Fred

from langchain_core.tools import tool

from core.logger import get_logger

from core.retry import with_retry as _with_retry


from core.logger import get_logger
log = get_logger(__name__)


def _parse_float_from_str(s: str) -> float | None:
    import re
    if not s:
        return None
    try:
        clean = s.replace(',', '')
        match = re.search(r'[-+]?\d*\.\d+|[-+]?\d+', clean)
        if match:
            return float(match.group())
        return None
    except ValueError:
        return None

def _parse_markdown_table_rows(content: str) -> list[dict]:
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    table_lines = []
    for ln in lines:
        if ln.startswith("|"):
            table_lines.append(ln)
    if len(table_lines) < 3:
        return []
    
    headers = [h.strip() for h in table_lines[0].split("|")[1:-1]]
    rows = []
    for ln in table_lines[2:]:
        cells = [c.strip() for c in ln.split("|")[1:-1]]
        if len(cells) == len(headers):
            rows.append(dict(zip(headers, cells)))
    return rows
