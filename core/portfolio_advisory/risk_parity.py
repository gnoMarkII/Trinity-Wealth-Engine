"""Portfolio-Level Risk Contribution & Active Share (Risk Parity & Radar).

Architecture Layer: core/portfolio_advisory/ (Pure Domain)
Strict Pure Python — no numpy, pandas, pydantic, sqlite3, or API imports.
Enforces:
- Active Share between actual weights and target policy weights: (1/2) * sum(|w_i - b_i|)
- Euler's theorem risk decomposition using exact Decimal matrix algebra
- Symmetric pairwise correlation validation (explicit zero preservation, no `or` falsy fallback)
- Positive-semidefinite (PSD) covariance check distinguishing valid zero-variance hedges
  from invalid/indefinite correlation structures
- Explicit UNAVAILABLE_ZERO_VARIANCE status when portfolio variance is zero
- Permutation-invariant largest-remainder rounding closure (100.00% for nonzero variance)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, ROUND_FLOOR, ROUND_HALF_UP
from typing import Any, Dict, List, Literal, Optional, Tuple

_ZERO = Decimal("0")
_ONE = Decimal("1")
_HUNDRED = Decimal("100")
_CENT = Decimal("0.01")
_TWO_DP = Decimal("0.01")
_FOUR_DP = Decimal("0.0001")
_PSD_TOL = Decimal("1e-12")
_ZERO_VAR_EPS = Decimal("1e-12")


def _to_finite_decimal(val: Any, name: str) -> Decimal:
    if isinstance(val, bool) or val is None:
        raise ValueError(f"{name} must be a finite numeric value, got {val!r}")
    try:
        dec = val if isinstance(val, Decimal) else Decimal(str(val).strip())
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(f"{name} is not a valid decimal: {val!r}") from exc
    if not dec.is_finite():
        raise ValueError(f"{name} must be finite, got {val!r}")
    return dec


def _close_to_hundred_percent(raw_pcts: Dict[str, Decimal], sorted_keys: List[str]) -> Dict[str, Decimal]:
    """Rounds exact percentages to 0.01% with exact 100.00% closure via deterministic largest remainder."""
    if not sorted_keys:
        return {}

    floored_cents: Dict[str, int] = {}
    remainders: Dict[str, Decimal] = {}
    for k in sorted_keys:
        scaled = raw_pcts[k] * _HUNDRED  # in units of 0.01%
        floored = int(scaled.to_integral_value(rounding=ROUND_FLOOR))
        floored_cents[k] = floored
        remainders[k] = scaled - Decimal(floored)

    target_total_cents = 10000
    deficit_cents = target_total_cents - sum(floored_cents.values())

    if deficit_cents > 0:
        ranked = sorted(sorted_keys, key=lambda k: (-remainders[k], k))
        for idx in range(deficit_cents):
            k = ranked[idx % len(ranked)]
            floored_cents[k] += 1
    elif deficit_cents < 0:
        ranked = sorted(sorted_keys, key=lambda k: (remainders[k], k))
        for idx in range(-deficit_cents):
            k = ranked[idx % len(ranked)]
            floored_cents[k] -= 1

    return {
        k: (Decimal(floored_cents[k]) / _HUNDRED).quantize(_CENT)
        for k in sorted_keys
    }


def _assert_correlation_matrix_psd(matrix: List[List[Decimal]]) -> None:
    """Checks positive-semidefiniteness via complete-pivoted Cholesky decomposition.

    Accepts rank-deficient valid PSD matrices (such as a zero-variance perfect hedge)
    while rejecting any indefinite / non-PSD correlation structure.
    """
    n = len(matrix)
    if n <= 1:
        return

    work = [row[:] for row in matrix]
    for k in range(n):
        pivot = k
        max_diag = work[k][k]
        for p in range(k + 1, n):
            if work[p][p] > max_diag:
                max_diag = work[p][p]
                pivot = p

        if max_diag < -_PSD_TOL:
            raise ValueError("Correlation/covariance matrix is not positive semidefinite")

        if pivot != k:
            work[k], work[pivot] = work[pivot], work[k]
            for row in work:
                row[k], row[pivot] = row[pivot], row[k]

        diag = work[k][k]
        if diag <= _PSD_TOL:
            for i in range(k, n):
                for j in range(k, n):
                    if abs(work[i][j]) > _PSD_TOL:
                        raise ValueError("Correlation/covariance matrix is not positive semidefinite")
            return

        for i in range(k + 1, n):
            factor = work[i][k] / diag
            if factor == _ZERO:
                continue
            for j in range(i, n):
                updated = work[i][j] - factor * work[k][j]
                work[i][j] = updated
                work[j][i] = updated


@dataclass(frozen=True)
class RiskContributionSummary:
    """Summary of portfolio risk decomposition and active share."""
    total_portfolio_variance: Decimal
    total_portfolio_volatility_pct: Decimal
    risk_contributions_pct: Dict[str, Decimal] = field(default_factory=dict)
    inverse_vol_weights_pct: Dict[str, Decimal] = field(default_factory=dict)
    active_share_pct: Decimal = Decimal("0.00")
    risk_contribution_status: Literal["AVAILABLE", "UNAVAILABLE_ZERO_VARIANCE", "NO_ACTIVE_ASSETS"] = "AVAILABLE"
    risk_contribution_reason: Optional[str] = None


def compute_bucket_active_share_pct(
    actual_weights: Dict[str, Decimal],
    target_weights: Dict[str, Decimal],
) -> Decimal:
    """Compute Active Share (%) across portfolio buckets or assets:
    Active Share = 0.5 * sum(|w_actual_i - w_target_i|)

    Expects weights in percent scale (e.g. 30.0 for 30%).
    Returns Active Share in percent (0.00% to 100.00%).
    """
    all_keys = sorted(set(actual_weights.keys()).union(set(target_weights.keys())))
    if not all_keys:
        return Decimal("0.00")

    total_abs_diff = _ZERO
    for key in all_keys:
        act = _to_finite_decimal(actual_weights.get(key, _ZERO), f"actual_weights[{key}]")
        tgt = _to_finite_decimal(target_weights.get(key, _ZERO), f"target_weights[{key}]")
        total_abs_diff += abs(act - tgt)

    active_share = total_abs_diff / Decimal("2")
    return active_share.quantize(_TWO_DP, rounding=ROUND_HALF_UP)


def compute_risk_contributions_and_parity(
    weights: Dict[str, Decimal],
    volatilities: Dict[str, Decimal],
    pairwise_correlations: Optional[Dict[Tuple[str, str], Decimal]] = None,
    default_correlation: Decimal = Decimal("0.30"),
    shrinkage_factor: Decimal = Decimal("0.10"),
) -> RiskContributionSummary:
    """Compute Euler risk decomposition and inverse-volatility weights for a portfolio.

    Args:
        weights: Asset weights (proportional or percent, e.g. {"PTT": Decimal("30"), "AAPL": Decimal("70")}).
        volatilities: Annualized volatility in percent (e.g. {"PTT": Decimal("20.0"), "AAPL": Decimal("25.0")}).
        pairwise_correlations: Optional map of ((asset_a, asset_b), corr) in [-1.0, 1.0].
        default_correlation: Fallback correlation between different assets if not in pairwise_correlations.
        shrinkage_factor: Linear shrinkage towards target constant correlation (0.0 to 1.0).

    Returns:
        RiskContributionSummary with total portfolio volatility, TRC% per asset, and inverse-vol weights.
    """
    const_corr = _to_finite_decimal(default_correlation, "default_correlation")
    if const_corr < Decimal("-1") or const_corr > _ONE:
        raise ValueError(f"default_correlation must be in [-1, 1], got {const_corr}")

    lamb = _to_finite_decimal(shrinkage_factor, "shrinkage_factor")
    if lamb < _ZERO or lamb > _ONE:
        raise ValueError(f"shrinkage_factor must be in [0, 1], got {lamb}")

    # Validate pairwise correlation map once and build symmetric unordered lookup
    symmetric_corr: Dict[Tuple[str, str], Decimal] = {}
    if pairwise_correlations:
        for pair, raw_val in pairwise_correlations.items():
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError(f"Invalid correlation key {pair!r}; expected 2-tuple of tickers")
            a, b = pair
            c_dec = _to_finite_decimal(raw_val, f"pairwise_correlations[{a},{b}]")
            if c_dec < Decimal("-1") or c_dec > _ONE:
                raise ValueError(f"Correlation for ({a}, {b}) must be in [-1, 1], got {c_dec}")
            if a == b:
                if abs(c_dec - _ONE) > Decimal("1e-9"):
                    raise ValueError(f"Self-correlation for {a} must be 1.0, got {c_dec}")
                continue
            canon = (a, b) if a < b else (b, a)
            if canon in symmetric_corr and symmetric_corr[canon] != c_dec:
                raise ValueError(
                    f"Asymmetric correlation entries for ({a}, {b}): {symmetric_corr[canon]} vs {c_dec}"
                )
            symmetric_corr[canon] = c_dec

    # Sort assets canonically for permutation invariance
    assets: List[str] = []
    parsed_weights: Dict[str, Decimal] = {}
    parsed_vols: Dict[str, Decimal] = {}
    for k in sorted(weights.keys()):
        w_dec = _to_finite_decimal(weights[k], f"weights[{k}]")
        if w_dec < _ZERO:
            raise ValueError(f"weights[{k}] must be non-negative, got {w_dec}")
        v_raw = volatilities.get(k, _ZERO)
        v_dec = _to_finite_decimal(v_raw, f"volatilities[{k}]")
        if v_dec < _ZERO:
            raise ValueError(f"volatilities[{k}] must be non-negative, got {v_dec}")
        if w_dec > _ZERO and v_dec > _ZERO:
            assets.append(k)
            parsed_weights[k] = w_dec
            parsed_vols[k] = v_dec

    if not assets:
        return RiskContributionSummary(
            total_portfolio_variance=Decimal("0.0000"),
            total_portfolio_volatility_pct=Decimal("0.00"),
            risk_contributions_pct={},
            inverse_vol_weights_pct={},
            active_share_pct=Decimal("0.00"),
            risk_contribution_status="NO_ACTIVE_ASSETS",
            risk_contribution_reason="No active assets with positive weight and positive volatility.",
        )

    total_w = sum(parsed_weights[a] for a in assets)
    if total_w <= _ZERO:
        return RiskContributionSummary(
            total_portfolio_variance=Decimal("0.0000"),
            total_portfolio_volatility_pct=Decimal("0.00"),
            risk_contributions_pct={},
            inverse_vol_weights_pct={},
            active_share_pct=Decimal("0.00"),
            risk_contribution_status="NO_ACTIVE_ASSETS",
            risk_contribution_reason="Total portfolio weight is non-positive.",
        )

    norm_w: Dict[str, Decimal] = {a: parsed_weights[a] / total_w for a in assets}

    # 1. Build symmetric correlation matrix with explicit zero preservation
    n = len(assets)
    corr_rows: List[List[Decimal]] = [[_ZERO for _ in range(n)] for _ in range(n)]
    for i in range(n):
        corr_rows[i][i] = _ONE
        for j in range(i + 1, n):
            a_i = assets[i]
            a_j = assets[j]
            canon = (a_i, a_j) if a_i < a_j else (a_j, a_i)
            c_val = symmetric_corr[canon] if canon in symmetric_corr else const_corr
            shrunk_c = (_ONE - lamb) * c_val + lamb * const_corr
            if shrunk_c > _ONE:
                shrunk_c = _ONE
            elif shrunk_c < Decimal("-1"):
                shrunk_c = Decimal("-1")
            corr_rows[i][j] = shrunk_c
            corr_rows[j][i] = shrunk_c

    # 2. Verify positive-semidefiniteness (allows valid zero-variance hedges, rejects invalid matrices)
    _assert_correlation_matrix_psd(corr_rows)

    # 3. Covariance matrix & Marginal contribution vector Cov * w
    cov_w: Dict[str, Decimal] = {a: _ZERO for a in assets}
    for i, a_i in enumerate(assets):
        row_sum = _ZERO
        for j, a_j in enumerate(assets):
            cov_ij = parsed_vols[a_i] * parsed_vols[a_j] * corr_rows[i][j]
            row_sum += cov_ij * norm_w[a_j]
        cov_w[a_i] = row_sum

    port_var = sum(norm_w[a] * cov_w[a] for a in assets)
    if port_var < _ZERO and port_var >= -_PSD_TOL:
        port_var = _ZERO
    elif port_var < -_PSD_TOL:
        raise ValueError(f"Negative portfolio variance {port_var} encountered from invalid covariance")

    port_vol = port_var.sqrt() if port_var > _ZERO else _ZERO

    # 4. Inverse-Volatility Weights (always defined when positive volatilities exist)
    inv_vols = {a: (_ONE / parsed_vols[a]) for a in assets}
    sum_inv_vol = sum(inv_vols.values())
    raw_iv_pct = {a: (inv_vols[a] / sum_inv_vol) * _HUNDRED for a in assets}
    inverse_vol_weights_pct = _close_to_hundred_percent(raw_iv_pct, assets)

    dec_port_var = port_var.quantize(_FOUR_DP, rounding=ROUND_HALF_UP)
    dec_port_vol = port_vol.quantize(_TWO_DP, rounding=ROUND_HALF_UP)

    # 5. Euler Total Risk Contribution (TRC%) — undefined when portfolio variance is zero
    if port_var <= _ZERO_VAR_EPS:
        return RiskContributionSummary(
            total_portfolio_variance=Decimal("0.0000"),
            total_portfolio_volatility_pct=Decimal("0.00"),
            risk_contributions_pct={},
            inverse_vol_weights_pct=inverse_vol_weights_pct,
            active_share_pct=Decimal("0.00"),
            risk_contribution_status="UNAVAILABLE_ZERO_VARIANCE",
            risk_contribution_reason="Euler percentage risk contributions are undefined for a zero-variance hedged portfolio.",
        )

    raw_trc_pct = {a: (norm_w[a] * cov_w[a] / port_var) * _HUNDRED for a in assets}
    risk_contributions_pct = _close_to_hundred_percent(raw_trc_pct, assets)

    return RiskContributionSummary(
        total_portfolio_variance=dec_port_var,
        total_portfolio_volatility_pct=dec_port_vol,
        risk_contributions_pct=risk_contributions_pct,
        inverse_vol_weights_pct=inverse_vol_weights_pct,
        active_share_pct=Decimal("0.00"),
        risk_contribution_status="AVAILABLE",
        risk_contribution_reason=None,
    )
