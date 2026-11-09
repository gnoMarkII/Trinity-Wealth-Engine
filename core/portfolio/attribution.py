"""Brinson-Fachler Performance Attribution & Carino Multi-Period Linking.

Architecture Layer: core/portfolio/ (Pure Domain)
Strictly adheres to Hexagonal Architecture:
- Pure Python standard library only (dataclasses, decimal, math, typing).
- Zero dependencies on frameworks, databases, or API layers.
- Deterministic math with exact closure reconciliation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, localcontext
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class BucketAttributionInput:
    """Input for a single allocation bucket or asset category in one period."""
    bucket_id: str
    bucket_name: str
    portfolio_weight: Decimal      # 0.0 .. 1.0 (Sum of all buckets in period must equal 1.0)
    benchmark_weight: Decimal      # 0.0 .. 1.0 (Sum of all buckets in period must equal 1.0)
    portfolio_return: Decimal      # Decimal return, e.g. 0.05 for +5.0%
    benchmark_return: Decimal      # Decimal return, e.g. 0.03 for +3.0%


@dataclass(frozen=True)
class BucketAttributionRow:
    """Attribution breakdown for a single bucket in one period or linked across periods."""
    bucket_id: str
    bucket_name: str
    portfolio_weight: Decimal
    benchmark_weight: Decimal
    active_weight: Decimal
    portfolio_return: Decimal
    benchmark_return: Decimal
    allocation_effect: Decimal
    selection_effect: Decimal
    interaction_effect: Decimal
    total_active_contribution: Decimal


@dataclass(frozen=True)
class BrinsonAttributionReport:
    """Consolidated Brinson-Fachler Performance Attribution Report."""
    portfolio_gross_return: Decimal
    benchmark_gross_return: Decimal
    portfolio_cost_rate: Decimal
    benchmark_cost_rate: Decimal
    net_active_return: Decimal
    allocation_effect_total: Decimal
    selection_effect_total: Decimal
    interaction_effect_total: Decimal
    cost_effect: Decimal
    cash_drag_effect: Decimal
    reconciliation_error: Decimal
    is_reconciled: bool
    rows: list[BucketAttributionRow] = field(default_factory=list)
    portfolio_net_return: Optional[Decimal] = None
    benchmark_net_return: Optional[Decimal] = None
    nominal_portfolio_cost_rate_sum: Optional[Decimal] = None
    nominal_benchmark_cost_rate_sum: Optional[Decimal] = None
    period_count: int = 1


def _require_finite_dec(val: float | int | str | Decimal, label: str = "value") -> Decimal:
    if isinstance(val, bool):
        raise ValueError(f"{label} must be numeric, not bool")
    if isinstance(val, Decimal):
        d = val
    else:
        try:
            d = Decimal(str(val))
        except (InvalidOperation, TypeError, ValueError) as exc:
            raise ValueError(f"invalid decimal {label}: {val!r}") from exc
    if not d.is_finite():
        raise ValueError(f"{label} must be finite, got {val!r}")
    return d


def _safe_dec(val: float | int | str | Decimal, quantum: str = "0.00000001") -> Decimal:
    d = _require_finite_dec(val)
    return d.quantize(Decimal(quantum), rounding=ROUND_HALF_UP)


def compute_single_period_brinson(
    buckets: list[BucketAttributionInput],
    portfolio_cost_rate: Decimal = Decimal("0"),
    benchmark_cost_rate: Decimal = Decimal("0"),
    tolerance: Decimal = Decimal("0.001"),
) -> BrinsonAttributionReport:
    """Calculates Single-Period Brinson-Fachler Performance Attribution.

    Formula per bucket i:
        A_i = (w_p,i - w_b,i) * (r_b,i - R_b_gross)
        S_i = w_b,i * (r_p,i - r_b,i)
        I_i = (w_p,i - w_b,i) * (r_p,i - r_b,i)
        T_i = A_i + S_i + I_i

    Closure invariant:
        R_a_gross = R_p_gross - R_b_gross = sum(A_i) + sum(S_i) + sum(I_i)
        R_a_net   = R_a_gross - (portfolio_cost_rate - benchmark_cost_rate)
    """
    if not buckets:
        raise ValueError("At least one bucket input is required for attribution analysis.")

    p_cost = _require_finite_dec(portfolio_cost_rate, "portfolio_cost_rate")
    b_cost = _require_finite_dec(benchmark_cost_rate, "benchmark_cost_rate")
    tol = _require_finite_dec(tolerance, "tolerance")
    if tol < Decimal("0"):
        raise ValueError(f"tolerance must be non-negative, got {tol}")

    validated_buckets: list[tuple[BucketAttributionInput, Decimal, Decimal, Decimal, Decimal]] = []
    for b in buckets:
        wp = _require_finite_dec(b.portfolio_weight, f"portfolio_weight[{b.bucket_id}]")
        wb = _require_finite_dec(b.benchmark_weight, f"benchmark_weight[{b.bucket_id}]")
        rp = _require_finite_dec(b.portfolio_return, f"portfolio_return[{b.bucket_id}]")
        rb = _require_finite_dec(b.benchmark_return, f"benchmark_return[{b.bucket_id}]")
        if wp < Decimal("0") or wb < Decimal("0"):
            raise ValueError(f"Bucket weights must be non-negative for {b.bucket_id}")
        validated_buckets.append((b, wp, wb, rp, rb))

    sum_wp = sum((item[1] for item in validated_buckets), Decimal("0"))
    sum_wb = sum((item[2] for item in validated_buckets), Decimal("0"))

    if abs(sum_wp - Decimal("1.0")) > tol:
        raise ValueError(f"Portfolio weights must sum to 1.0 +/- {tol}, got {sum_wp}")
    if abs(sum_wb - Decimal("1.0")) > tol:
        raise ValueError(f"Benchmark weights must sum to 1.0 +/- {tol}, got {sum_wb}")
    if sum_wp <= Decimal("0") or sum_wb <= Decimal("0"):
        raise ValueError("Total portfolio and benchmark weights must be positive.")

    # Normalize weights permitted within tolerance consistently so sum(w_p) == sum(w_b) == 1.0
    # and the Brinson-Fachler closure identity holds exactly.
    norm_buckets: list[tuple[BucketAttributionInput, Decimal, Decimal, Decimal, Decimal]] = [
        (
            b,
            wp if sum_wp == Decimal("1") else (wp / sum_wp),
            wb if sum_wb == Decimal("1") else (wb / sum_wb),
            rp,
            rb,
        )
        for b, wp, wb, rp, rb in validated_buckets
    ]

    # Gross portfolio and benchmark returns
    r_p_gross = sum((wp * rp for _, wp, _, rp, _ in norm_buckets), Decimal("0"))
    r_b_gross = sum((wb * rb for _, _, wb, _, rb in norm_buckets), Decimal("0"))

    rows: list[BucketAttributionRow] = []
    alloc_total = Decimal("0")
    select_total = Decimal("0")
    inter_total = Decimal("0")
    cash_drag = Decimal("0")

    for b, wp, wb, rp, rb in norm_buckets:
        active_w = wp - wb
        # Allocation effect: active weight * (benchmark bucket return - total benchmark return)
        alloc = active_w * (rb - r_b_gross)
        # Selection effect: benchmark weight * (portfolio bucket return - benchmark bucket return)
        select = wb * (rp - rb)
        # Interaction effect: active weight * active return
        inter = active_w * (rp - rb)
        total_contrib = alloc + select + inter

        row = BucketAttributionRow(
            bucket_id=b.bucket_id,
            bucket_name=b.bucket_name,
            portfolio_weight=wp,
            benchmark_weight=wb,
            active_weight=active_w,
            portfolio_return=rp,
            benchmark_return=rb,
            allocation_effect=alloc,
            selection_effect=select,
            interaction_effect=inter,
            total_active_contribution=total_contrib,
        )
        rows.append(row)

        alloc_total += alloc
        select_total += select
        inter_total += inter

        # Identify cash drag if bucket is cash
        bid_upper = b.bucket_id.strip().upper()
        bname_upper = b.bucket_name.strip().upper()
        if "__CASH__" in bid_upper or "CASH" in bid_upper or "เงินสด" in bname_upper:
            cash_drag += total_contrib

    cost_effect = b_cost - p_cost
    r_p_net = r_p_gross - p_cost
    r_b_net = r_b_gross - b_cost
    r_a_net = r_p_net - r_b_net

    sum_explained = alloc_total + select_total + inter_total + cost_effect
    reconciliation_error = r_a_net - sum_explained
    is_reconciled = abs(reconciliation_error) <= Decimal("0.00000001")

    return BrinsonAttributionReport(
        portfolio_gross_return=r_p_gross,
        benchmark_gross_return=r_b_gross,
        portfolio_cost_rate=p_cost,
        benchmark_cost_rate=b_cost,
        net_active_return=r_a_net,
        allocation_effect_total=alloc_total,
        selection_effect_total=select_total,
        interaction_effect_total=inter_total,
        cost_effect=cost_effect,
        cash_drag_effect=cash_drag,
        reconciliation_error=reconciliation_error,
        is_reconciled=is_reconciled,
        rows=rows,
        portfolio_net_return=r_p_net,
        benchmark_net_return=r_b_net,
        nominal_portfolio_cost_rate_sum=p_cost,
        nominal_benchmark_cost_rate_sum=b_cost,
        period_count=1,
    )


def link_multi_period_brinson_carino(
    periods: list[BrinsonAttributionReport],
) -> BrinsonAttributionReport:
    """Links multiple single-period (or previously linked) Brinson attribution reports.

    Uses pure-Decimal Carino (1999) Logarithmic Linking Algorithm to ensure geometric closure:
        1 + R_p = prod(1 + R_p,t)
        1 + R_b = prod(1 + R_b,t)
        R_a = R_p - R_b

    Preserves effective cumulative cost drag (`portfolio_gross_return - portfolio_net_return`)
    in `portfolio_cost_rate` / `benchmark_cost_rate` so linked reports obey the same
    gross-minus-cost = net invariant as single-period reports and can be nested idempotently,
    while recording nominal summed costs separately.
    """
    if not periods:
        raise ValueError("At least one period report is required for Carino linking.")

    for idx, p in enumerate(periods):
        p_gross = _require_finite_dec(p.portfolio_gross_return, f"periods[{idx}].portfolio_gross_return")
        b_gross = _require_finite_dec(p.benchmark_gross_return, f"periods[{idx}].benchmark_gross_return")
        p_cost = _require_finite_dec(p.portfolio_cost_rate, f"periods[{idx}].portfolio_cost_rate")
        b_cost = _require_finite_dec(p.benchmark_cost_rate, f"periods[{idx}].benchmark_cost_rate")
        p_net = p.portfolio_net_return if p.portfolio_net_return is not None else (p_gross - p_cost)
        b_net = p.benchmark_net_return if p.benchmark_net_return is not None else (b_gross - b_cost)
        if (Decimal("1") + p_net) <= Decimal("0") or (Decimal("1") + b_net) <= Decimal("0"):
            raise ValueError(
                f"Carino logarithmic linking undefined for period {idx} return <= -100% "
                f"(portfolio_net={p_net}, benchmark_net={b_net})"
            )

    if len(periods) == 1:
        return periods[0]

    with localcontext() as ctx:
        ctx.prec = 34

        r_p_net_compound = Decimal("1")
        r_b_net_compound = Decimal("1")
        r_p_gross_compound = Decimal("1")
        r_b_gross_compound = Decimal("1")
        nominal_p_cost_sum = Decimal("0")
        nominal_b_cost_sum = Decimal("0")
        total_period_count = 0

        for p in periods:
            p_net = p.portfolio_net_return if p.portfolio_net_return is not None else (
                p.portfolio_gross_return - p.portfolio_cost_rate
            )
            b_net = p.benchmark_net_return if p.benchmark_net_return is not None else (
                p.benchmark_gross_return - p.benchmark_cost_rate
            )
            r_p_net_compound *= (Decimal("1") + p_net)
            r_b_net_compound *= (Decimal("1") + b_net)
            r_p_gross_compound *= (Decimal("1") + p.portfolio_gross_return)
            r_b_gross_compound *= (Decimal("1") + p.benchmark_gross_return)
            nominal_p_cost_sum += (
                p.nominal_portfolio_cost_rate_sum
                if p.nominal_portfolio_cost_rate_sum is not None
                else p.portfolio_cost_rate
            )
            nominal_b_cost_sum += (
                p.nominal_benchmark_cost_rate_sum
                if p.nominal_benchmark_cost_rate_sum is not None
                else p.benchmark_cost_rate
            )
            total_period_count += max(1, int(getattr(p, "period_count", 1)))

        r_p_net = r_p_net_compound - Decimal("1")
        r_b_net = r_b_net_compound - Decimal("1")
        r_p_gross = r_p_gross_compound - Decimal("1")
        r_b_gross = r_b_gross_compound - Decimal("1")
        total_active_net = r_p_net - r_b_net

        if (Decimal("1") + r_p_net) <= Decimal("0") or (Decimal("1") + r_b_net) <= Decimal("0"):
            raise ValueError("Carino logarithmic linking undefined when compounded return <= -100%.")

        def _excess_log_ratio(rp: Decimal, rb: Decimal) -> Decimal:
            one_plus_rp = Decimal("1") + rp
            one_plus_rb = Decimal("1") + rb
            if one_plus_rp <= Decimal("0") or one_plus_rb <= Decimal("0"):
                raise ValueError("Logarithmic linking undefined for returns <= -100%.")
            diff = rp - rb
            if abs(diff) < Decimal("1e-18"):
                return Decimal("1") / one_plus_rp
            return (one_plus_rp.ln() - one_plus_rb.ln()) / diff

        K = _excess_log_ratio(r_p_net, r_b_net)
        if K == Decimal("0"):
            raise ValueError("Carino normalization constant K evaluated to zero.")

        c_factors: list[Decimal] = []
        for p in periods:
            p_net_t = p.portfolio_net_return if p.portfolio_net_return is not None else (
                p.portfolio_gross_return - p.portfolio_cost_rate
            )
            b_net_t = p.benchmark_net_return if p.benchmark_net_return is not None else (
                p.benchmark_gross_return - p.benchmark_cost_rate
            )
            k_t = _excess_log_ratio(p_net_t, b_net_t)
            c_factors.append(k_t / K)

        # Effective compounded cumulative cost drag satisfies: gross - effective_cost == net
        effective_p_cost_rate = r_p_gross - r_p_net
        effective_b_cost_rate = r_b_gross - r_b_net

        bucket_map: dict[str, dict[str, Any]] = {}
        total_alloc = Decimal("0")
        total_select = Decimal("0")
        total_inter = Decimal("0")
        total_cost = Decimal("0")
        total_cash_drag = Decimal("0")

        for p, c_t in zip(periods, c_factors):
            p_span = Decimal(str(max(1, int(getattr(p, "period_count", 1)))))
            total_alloc += c_t * p.allocation_effect_total
            total_select += c_t * p.selection_effect_total
            total_inter += c_t * p.interaction_effect_total
            total_cost += c_t * p.cost_effect
            total_cash_drag += c_t * p.cash_drag_effect

            for row in p.rows:
                if row.bucket_id not in bucket_map:
                    bucket_map[row.bucket_id] = {
                        "bucket_name": row.bucket_name,
                        "alloc": Decimal("0"),
                        "select": Decimal("0"),
                        "inter": Decimal("0"),
                        "contrib": Decimal("0"),
                        "w_p_sum": Decimal("0"),
                        "w_b_sum": Decimal("0"),
                    }
                bm = bucket_map[row.bucket_id]
                bm["alloc"] += c_t * row.allocation_effect
                bm["select"] += c_t * row.selection_effect
                bm["inter"] += c_t * row.interaction_effect
                bm["contrib"] += c_t * row.total_active_contribution
                bm["w_p_sum"] += row.portfolio_weight * p_span
                bm["w_b_sum"] += row.benchmark_weight * p_span

        linked_rows: list[BucketAttributionRow] = []
        n_span = Decimal(str(total_period_count))
        for bid, bm in bucket_map.items():
            avg_wp = bm["w_p_sum"] / n_span
            avg_wb = bm["w_b_sum"] / n_span
            linked_rows.append(BucketAttributionRow(
                bucket_id=bid,
                bucket_name=bm["bucket_name"],
                portfolio_weight=avg_wp,
                benchmark_weight=avg_wb,
                active_weight=avg_wp - avg_wb,
                portfolio_return=Decimal("0"),
                benchmark_return=Decimal("0"),
                allocation_effect=bm["alloc"],
                selection_effect=bm["select"],
                interaction_effect=bm["inter"],
                total_active_contribution=bm["contrib"],
            ))

        sum_explained = total_alloc + total_select + total_inter + total_cost
        reconciliation_error = total_active_net - sum_explained
        is_reconciled = abs(reconciliation_error) <= Decimal("0.00000001")

    return BrinsonAttributionReport(
        portfolio_gross_return=r_p_gross,
        benchmark_gross_return=r_b_gross,
        portfolio_cost_rate=effective_p_cost_rate,
        benchmark_cost_rate=effective_b_cost_rate,
        net_active_return=total_active_net,
        allocation_effect_total=total_alloc,
        selection_effect_total=total_select,
        interaction_effect_total=total_inter,
        cost_effect=total_cost,
        cash_drag_effect=total_cash_drag,
        reconciliation_error=reconciliation_error,
        is_reconciled=is_reconciled,
        rows=linked_rows,
        portfolio_net_return=r_p_net,
        benchmark_net_return=r_b_net,
        nominal_portfolio_cost_rate_sum=nominal_p_cost_sum,
        nominal_benchmark_cost_rate_sum=nominal_b_cost_sum,
        period_count=total_period_count,
    )
