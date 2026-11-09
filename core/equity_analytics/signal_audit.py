"""Point-in-Time Maturity Guard, Rank IC & Newey-West HAC t-Statistic.

Architecture Layer: core/equity_analytics/ (Pure Domain)
Strict Pure Python — no numpy, pandas, pydantic, sqlite3, or API imports.
Enforces:
- Purged Maturity Guard: filing_date + embargo_days <= as_of_date (calendar-day delay buffer;
  strictly eliminates lookahead bias when historical filing dates are tracked)
- Cross-sectional Rank IC (Spearman Rank Correlation) using pure Decimal arithmetic
- Newey-West HAC (Heteroskedasticity and Autocorrelation Consistent) t-statistic for factor significance
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, localcontext
from typing import List, Optional, Tuple, Union

NumberLike = Union[Decimal, int, float, str]


def _parse_iso_date(raw: str) -> Optional[date]:
    if not isinstance(raw, str):
        return None
    cleaned = raw.strip()
    if not cleaned:
        return None
    try:
        if "T" in cleaned or " " in cleaned or len(cleaned) > 10:
            iso_val = cleaned[:-1] + "+00:00" if cleaned.endswith("Z") else cleaned
            return datetime.fromisoformat(iso_val).date()
        return date.fromisoformat(cleaned)
    except ValueError:
        return None


def _to_finite_decimal_or_none(val: NumberLike) -> Optional[Decimal]:
    if isinstance(val, bool):
        raise ValueError("Sample values must be numeric, not bool")
    if val is None:
        return None
    try:
        dec = val if isinstance(val, Decimal) else Decimal(str(val))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return dec if dec.is_finite() else None


def audit_signal_maturity(
    as_of_date: str,
    filing_date: str,
    embargo_days: int = 2,
) -> bool:
    """Verifies that public financial disclosures are mature and tradeable as of the analysis date.

    Prevents lookahead bias by requiring that the official filing disclosure occurred at least
    `embargo_days` calendar days before the trading/analysis date.

    Args:
        as_of_date: Analysis or trading decision date (ISO 8601 date or timestamp).
        filing_date: Official SEC/SET disclosure or earnings release date (ISO 8601 date or timestamp).
        embargo_days: Non-negative calendar-day ingestion delay buffer (default 2 days).

    Returns:
        True if the data was fully available without lookahead bias, False if prematurely leaking
        or if either date is malformed/missing.
    """
    if isinstance(embargo_days, bool) or not isinstance(embargo_days, int) or embargo_days < 0:
        raise ValueError(f"embargo_days must be a non-negative integer, got {embargo_days!r}")

    d_asof = _parse_iso_date(as_of_date)
    d_filing = _parse_iso_date(filing_date)
    if d_asof is None or d_filing is None:
        return False

    mature_date = d_filing + timedelta(days=embargo_days)
    return mature_date <= d_asof


def _fractional_rank(values: List[Decimal]) -> List[Decimal]:
    """Computes fractional ranks for a list of Decimal values (handles ties by averaging)."""
    n = len(values)
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [Decimal("0")] * n

    i = 0
    while i < n:
        j = i
        while j < n - 1 and indexed[j][1] == indexed[j + 1][1]:
            j += 1
        avg_rank = Decimal(i + 1 + j + 1) / Decimal("2")
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1

    return ranks


def compute_rank_ic(
    factor_values: List[NumberLike],
    forward_returns: List[NumberLike],
) -> Optional[float]:
    """Computes Spearman Rank Information Coefficient (Rank IC) across an asset cross-section.

    Inputs must be aligned one-for-one by date/security; mismatched lengths raise ValueError.

    Args:
        factor_values: Raw factor scores or values.
        forward_returns: Realized forward return values.

    Returns:
        Rank correlation in [-1.0, 1.0], or None if fewer than 3 valid aligned pairs or zero variance
        (e.g., constant factor or constant forward returns).
    """
    if len(factor_values) != len(forward_returns):
        raise ValueError(
            f"Aligned factor_values and forward_returns length mismatch: "
            f"{len(factor_values)} vs {len(forward_returns)}"
        )

    n = len(factor_values)
    if n < 3:
        return None

    valid_pairs: List[Tuple[Decimal, Decimal]] = []
    for x_raw, y_raw in zip(factor_values, forward_returns):
        dx = _to_finite_decimal_or_none(x_raw)
        dy = _to_finite_decimal_or_none(y_raw)
        if dx is not None and dy is not None:
            valid_pairs.append((dx, dy))

    if len(valid_pairs) < 3:
        return None

    fx = [p[0] for p in valid_pairs]
    fy = [p[1] for p in valid_pairs]

    rank_x = _fractional_rank(fx)
    rank_y = _fractional_rank(fy)

    with localcontext() as ctx:
        ctx.prec = 34
        count_dec = Decimal(len(rank_x))
        m_x = sum(rank_x, Decimal("0")) / count_dec
        m_y = sum(rank_y, Decimal("0")) / count_dec

        cov = sum(((rx - m_x) * (ry - m_y) for rx, ry in zip(rank_x, rank_y)), Decimal("0"))
        var_x = sum(((rx - m_x) ** 2 for rx in rank_x), Decimal("0"))
        var_y = sum(((ry - m_y) ** 2 for ry in rank_y), Decimal("0"))

        if var_x <= Decimal("1e-12") or var_y <= Decimal("1e-12"):
            return None

        denom = (var_x * var_y).sqrt()
        if denom <= Decimal("1e-12"):
            return None

        ic = cov / denom
        if ic > Decimal("1"):
            ic = Decimal("1")
        elif ic < Decimal("-1"):
            ic = Decimal("-1")

        return float(ic.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))


def compute_newey_west_hac_tstat(
    ic_series: List[NumberLike],
    max_lags: int = 2,
) -> Optional[float]:
    """Computes Newey-West HAC t-statistic for a time-series of factor IC values using Decimal.

    Accounts for autocorrelation and heteroskedasticity over sequential evaluation periods.

    Args:
        ic_series: Historical series of Information Coefficients.
        max_lags: Non-negative maximum autocorrelation lag truncation (default 2).

    Returns:
        Newey-West t-statistic, or None if insufficient observations (< 5) or zero/non-positive HAC variance.
    """
    if isinstance(max_lags, bool) or not isinstance(max_lags, int) or max_lags < 0:
        raise ValueError(f"max_lags must be a non-negative integer, got {max_lags!r}")

    valid_ics: List[Decimal] = []
    for raw in ic_series:
        dec = _to_finite_decimal_or_none(raw)
        if dec is not None:
            valid_ics.append(dec)

    t = len(valid_ics)
    if t < 5:
        return None

    with localcontext() as ctx:
        ctx.prec = 34
        t_dec = Decimal(t)
        mean_ic = sum(valid_ics, Decimal("0")) / t_dec
        devs = [x - mean_ic for x in valid_ics]

        gamma_0 = sum((d ** 2 for d in devs), Decimal("0")) / t_dec
        if gamma_0 <= Decimal("1e-12"):
            return None

        nw_var = gamma_0
        lags = min(max_lags, t - 2)

        for j in range(1, lags + 1):
            weight = Decimal("1") - (Decimal(j) / Decimal(lags + 1))  # Bartlett kernel
            cov_j = sum((devs[k] * devs[k - j] for k in range(j, t)), Decimal("0")) / t_dec
            nw_var += Decimal("2") * weight * cov_j

        if nw_var <= Decimal("1e-12"):
            return None

        se = (nw_var / t_dec).sqrt()
        if se <= Decimal("1e-12"):
            return None

        t_stat = mean_ic / se
        return float(t_stat.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
