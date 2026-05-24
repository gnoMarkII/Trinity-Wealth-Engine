"""Shared retry helper สำหรับ external API calls (yfinance, FRED ฯลฯ)

ใช้ pattern เดียวกับ main.py loop:
  - ตรวจ transient HTTP codes (429/500/502/503/504) + network errors
  - exponential backoff 2^attempt seconds
  - retry budget 3 attempts (re-raise after exhausted)

Exception coverage:
  - built-in TimeoutError / ConnectionError
  - requests.exceptions.{Timeout, ConnectionError, HTTPError}
  - curl_cffi.requests.exceptions.{Timeout, ConnectionError, HTTPError}  (yfinance ≥0.2.50)
  - yfinance.exceptions.YFRateLimitError
  - fallback: regex บน str(exc) สำหรับ exception ที่ถูก wrap/chain
"""
import re
import time

# Optional imports — แต่ละ HTTP stack อาจไม่ติดตั้งครบทุกตัว
_NETWORK_CLASSES: list[type] = [TimeoutError, ConnectionError]
_HTTP_ERROR_CLASSES: list[type] = []

try:
    import requests.exceptions as _req_exc
    _NETWORK_CLASSES += [_req_exc.Timeout, _req_exc.ConnectionError]
    _HTTP_ERROR_CLASSES.append(_req_exc.HTTPError)
except ImportError:
    pass

try:
    import curl_cffi.requests.exceptions as _ccf_exc
    _NETWORK_CLASSES += [_ccf_exc.Timeout, _ccf_exc.ConnectionError]
    _HTTP_ERROR_CLASSES.append(_ccf_exc.HTTPError)
except ImportError:
    pass

try:
    from yfinance.exceptions import YFRateLimitError
    _RATE_LIMIT_CLASSES: tuple[type, ...] = (YFRateLimitError,)
except ImportError:
    _RATE_LIMIT_CLASSES = ()

_NETWORK_TUPLE = tuple(_NETWORK_CLASSES)
_HTTP_ERROR_TUPLE = tuple(_HTTP_ERROR_CLASSES)

_MAX_RETRIES = 3
_TRANSIENT_HTTP_CODES = {429, 500, 502, 503, 504}
_TRANSIENT_PATTERN = re.compile(
    r'(\b429\b|\b500\b|\b502\b|\b503\b|\b504\b'
    r'|rate.?limit|too.?many.?requests|timeout|temporarily|service.?unavailable)',
    re.IGNORECASE,
)


def _http_status(exc: Exception) -> int | None:
    """ดึง HTTP status code จาก exception ที่ผูกกับ Response (ถ้ามี)"""
    resp = getattr(exc, "response", None)
    if resp is None:
        return None
    return getattr(resp, "status_code", None)


def is_transient_error(exc: Exception) -> bool:
    """ตรวจว่า exception ควร retry หรือไม่ — ครอบทั้ง class-based check + regex fallback"""
    if _RATE_LIMIT_CLASSES and isinstance(exc, _RATE_LIMIT_CLASSES):
        return True
    if isinstance(exc, _NETWORK_TUPLE):
        return True
    if _HTTP_ERROR_TUPLE and isinstance(exc, _HTTP_ERROR_TUPLE):
        code = _http_status(exc)
        if code in _TRANSIENT_HTTP_CODES:
            return True
        # บาง HTTPError ไม่มี response ผูก — fallback ไป regex บน message
    return bool(_TRANSIENT_PATTERN.search(str(exc)))


def with_retry(fn, *args, **kwargs):
    """รัน fn พร้อม exponential backoff retry — re-raise ถ้า exhaust budget หรือ non-transient

    Args:
        fn: callable
        *args, **kwargs: forwarded ให้ fn
    """
    for attempt in range(_MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if attempt < _MAX_RETRIES - 1 and is_transient_error(e):
                time.sleep(2 ** attempt)
                continue
            raise
