import pytest
from datetime import datetime, date
from tools.portfolio.models import _coerce_iso_string

def test_coerce_iso_string():
    # Test datetime (has hour)
    dt = datetime(2026, 6, 21, 10, 30, 45)
    assert _coerce_iso_string(dt) == "2026-06-21T10:30:45"
    
    # Test date (no hour)
    d = date(2026, 6, 21)
    assert _coerce_iso_string(d) == "2026-06-21"
    
    # Test string (no isoformat)
    s = "2026-06-21T10:30:45"
    assert _coerce_iso_string(s) == s
