"""Tier 1 Deterministic Portfolio Advisory Engine (< 50ms).

Architecture Layer: core/portfolio_advisory/ (Pure Domain)
Performs deterministic calculations, 11-point guardrails, market lot sizing,
dual-currency cash ledger, and conviction-adjusted advice using Python Decimal.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP
import hashlib
import json
from typing import Any, Dict, List, Literal, Optional, Tuple

from core.portfolio_advisory.guardrails import (
    check_future_date_anomaly,
    check_look_through_concentration_cap,
    check_macro_cross_scale_drift,
    check_missing_price_policy,
    check_price_scale_anomaly,
    check_stale_portfolio_price,
    check_unassigned_holdings_drift,
    compute_live_upside,
    resolve_holding_currency_and_market,
    safe_decimal,
    validate_bucket_target_sum,
    validate_fx_rate,
    verify_ticker_market_identity,
)
from core.portfolio_advisory.lot_rounding import (
    calculate_fee_inclusive_lot_sizing,
    calculate_market_lot_sizing,
)
from core.portfolio_advisory.friction import estimate_order_friction
from core.portfolio_advisory.circuit_breaker import (
    DrawdownPostureTier,
    evaluate_drawdown_circuit_breaker,
)
from core.portfolio_advisory.risk_parity import (
    compute_bucket_active_share_pct,
    compute_risk_contributions_and_parity,
)
from core.portfolio_advisory.models import (
    AdvisoryActionCode,
    AdvisoryDataMode,
    AssetEvaluationTrack,
    DataAnomalyWarning,
    DualCurrencyCashPlan,
    HoldingAdvisoryItem,
    PortfolioAdvisoryQuantReport,
    PortfolioRiskRadar,
    QuarantinedHoldingItem,
)
from core.portfolio_advisory.sector_mapping import (
    is_sector_match,
    normalize_to_canonical_sector,
)


def compute_portfolio_state_hash(holdings: List[Any], cash_thb: Decimal, cash_usd: Decimal) -> str:
    """Generates deterministic SHA-256 hash representing current portfolio composition."""
    tokens = []
    for h in sorted(holdings, key=lambda x: str(getattr(x, "symbol", ""))):
        sym = str(getattr(h, "symbol", "")).strip().upper()
        units = str(safe_decimal(getattr(h, "units", None), quantum="0.0001"))
        val = str(safe_decimal(getattr(h, "market_value_thb", None), quantum="0.01"))
        tokens.append(f"{sym}:{units}:{val}")
    tokens.append(f"CASH_THB:{cash_thb}:CASH_USD:{cash_usd}")
    raw_str = "|".join(tokens)
    return hashlib.sha256(raw_str.encode("utf-8")).hexdigest()[:16]


_SYNTHESIS_METHODOLOGY_VERSION = "v2"


def _extract(obj: Any, key: str, default: Any = None) -> Any:
    """Safely extracts a field from a dataclass, pydantic model, dict, or mock."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    val = getattr(obj, key, default)
    if val is None or val.__class__.__name__ == "MagicMock":
        return default
    return val


def _extract_positive_decimal(obj: Any, key: str, quantum: str = "0.0001") -> Optional[Decimal]:
    raw = _extract(obj, key)
    if isinstance(raw, bool) or raw is None:
        return None
    dec = safe_decimal(raw, default=None, quantum=quantum)
    if dec is None or dec <= Decimal("0"):
        return None
    return dec


def _extract_unrounded_positive_decimal(val: Any) -> Optional[Decimal]:
    if isinstance(val, bool) or val is None or val.__class__.__name__ == "MagicMock":
        return None
    if isinstance(val, (int, float, str, Decimal)):
        dec = safe_decimal(val, default=None, quantum=None)
        if dec is not None and dec > Decimal("0"):
            return dec
    return None


def _resolve_validated_sidecar(
    sym: str,
    clean_sym: str,
    holding_market: str,
    equity_sidecars: Optional[Dict[str, Any]],
) -> Tuple[Optional[Any], Optional[Any]]:
    """Finds market-validated sidecar and nested quant_signals supporting dict and model shapes."""
    if not equity_sidecars:
        return None, None

    candidates: List[Any] = []
    seen_ids: set[int] = set()
    preferred_keys = [sym, f"{clean_sym}.BK" if holding_market == "TH" else "", clean_sym]
    for k in preferred_keys:
        if k and k in equity_sidecars:
            sc = equity_sidecars[k]
            if id(sc) not in seen_ids:
                candidates.append(sc)
                seen_ids.add(id(sc))

    for k, sc in equity_sidecars.items():
        if id(sc) in seen_ids:
            continue
        k_norm = str(k).strip().upper()
        if k_norm == sym or k_norm.replace(".BK", "") == clean_sym:
            candidates.append(sc)
            seen_ids.add(id(sc))

    for sc in candidates:
        sig = _extract(sc, "quant_signals")
        sc_market = _extract(sc, "market") or _extract(sig, "market")
        sc_ticker = str(_extract(sc, "ticker") or _extract(sig, "ticker") or clean_sym).strip().upper()
        if sc_market:
            if not verify_ticker_market_identity(sym, holding_market, sc_ticker, str(sc_market)):
                continue
        else:
            sc_clean = sc_ticker.replace(".BK", "")
            if sc_clean != clean_sym:
                continue
        return sc, sig

    return None, None


def _compute_synthesis_input_hash(
    *,
    portfolio_state_hash: str,
    data_mode: AdvisoryDataMode,
    macro_direction: Optional[Any],
    sector_rotation_snapshot: Optional[Any],
    risk_radar: PortfolioRiskRadar,
    holdings_advisory: List[HoldingAdvisoryItem],
    quarantined_holdings: List[QuarantinedHoldingItem],
    anomaly_warnings: List[DataAnomalyWarning],
    look_through_summary: Optional[Any],
) -> str:
    """Computes complete deterministic cache key covering all Tier 2 synthesis inputs."""
    macro_digest = None
    if macro_direction is not None:
        sa = _extract(macro_direction, "sector_analysis") or {}
        macro_digest = {
            "overall_regime": _extract(macro_direction, "overall_regime") or _extract(macro_direction, "regime_name"),
            "evaluated_at": _extract(macro_direction, "evaluated_at"),
            "conviction_level": _extract(macro_direction, "conviction_level"),
            "focus_themes": list(_extract(macro_direction, "focus_themes", []) or []),
            "leading_sectors": list(_extract(sa, "leading_sectors", []) or []),
            "lagging_sectors": list(_extract(sa, "lagging_sectors", []) or []),
            "asset_allocation": _extract(macro_direction, "asset_allocation", []) or [],
        }

    sector_digest = None
    if sector_rotation_snapshot is not None:
        rows = _extract(sector_rotation_snapshot, "rows", []) or []
        row_pairs = []
        for r in rows:
            t = str(_extract(r, "ticker", "") or "").upper()
            q = _extract(r, "quadrant") or _extract(_extract(r, "current"), "quadrant")
            if not q:
                history = _extract(r, "history")
                pts = history if isinstance(history, list) else (
                    (history.get("weekly") or history.get("daily") or []) if isinstance(history, dict) else []
                )
                if pts:
                    q = _extract(pts[-1], "quadrant")
            row_pairs.append([t, q])
        row_pairs.sort()
        sector_digest = {
            "as_of_date": _extract(sector_rotation_snapshot, "as_of_date"),
            "snapshot_id": _extract(sector_rotation_snapshot, "snapshot_id"),
            "rows": row_pairs,
        }

    lt_digest = None
    if look_through_summary is not None:
        top_stocks = _extract(look_through_summary, "top_overlapped_stocks", []) or []
        secs = _extract(look_through_summary, "sector_exposure", []) or []
        lt_digest = {
            "top": [
                [
                    str(_extract(s, "ticker", "")),
                    str(_extract(s, "total_weight_pct", "")),
                    bool(_extract(s, "is_cap_breached", False)),
                ]
                for s in top_stocks[:10]
            ],
            "sectors": [
                [str(_extract(sec, "sector_name", "")), str(_extract(sec, "weight_pct", ""))]
                for sec in secs
            ],
            "warnings": list(_extract(look_through_summary, "hidden_concentration_warnings", []) or []),
        }

    canonical_payload = {
        "methodology_version": _SYNTHESIS_METHODOLOGY_VERSION,
        "portfolio_state_hash": portfolio_state_hash,
        "data_mode": data_mode.value,
        "macro": macro_digest,
        "sector_rotation": sector_digest,
        "risk_radar": {
            "weighted_beta": str(risk_radar.weighted_beta) if risk_radar.weighted_beta is not None else None,
            "hhi": str(risk_radar.hhi) if risk_radar.hhi is not None else None,
            "bucket_active_share_pct": str(risk_radar.bucket_active_share_pct) if risk_radar.bucket_active_share_pct is not None else None,
            "bucket_active_share_status": risk_radar.bucket_active_share_status,
            "risk_contribution_status": risk_radar.risk_contribution_status,
            "risk_contributions_pct": {k: str(v) for k, v in sorted(risk_radar.risk_contributions_pct.items())},
            "inverse_vol_weights_pct": {k: str(v) for k, v in sorted(risk_radar.inverse_vol_weights_pct.items())},
            "total_portfolio_volatility_pct": str(risk_radar.total_portfolio_volatility_pct) if risk_radar.total_portfolio_volatility_pct is not None else None,
            "volatility_estimated_symbols": sorted(risk_radar.volatility_estimated_symbols),
            "drawdown_pct": str(risk_radar.drawdown_pct) if risk_radar.drawdown_pct is not None else None,
            "drawdown_posture_tier": risk_radar.drawdown_posture_tier,
            "drawdown_status": risk_radar.drawdown_status,
            "drawdown_hurdle_adjustment": str(risk_radar.drawdown_hurdle_adjustment),
            "purpose_bucket_drift": {k: str(v) for k, v in sorted(risk_radar.purpose_bucket_drift.items())},
        },
        "holdings": [
            {
                "symbol": h.symbol,
                "action_code": h.action_code.value,
                "target_weight_pct": str(h.target_weight_pct),
                "weight_drift_pct": str(h.weight_drift_pct),
                "is_immediate_execution": h.is_immediate_execution,
                "risk_contribution_pct": str(h.risk_contribution_pct) if h.risk_contribution_pct is not None else None,
                "estimated_friction_bps": str(h.estimated_friction_bps) if h.estimated_friction_bps is not None else None,
                "volatility_source": h.volatility_source,
                "core_conviction_score": str(h.core_conviction_score) if h.core_conviction_score is not None else None,
                "live_upside_12m_pct": str(h.live_upside_12m_pct) if h.live_upside_12m_pct is not None else None,
                "macro_alignment": h.macro_alignment,
                "sector_quadrant": h.sector_quadrant,
                "lot_amount_native": str(h.lot_sizing.amount_native) if h.lot_sizing else None,
                "breached_falsifiers": list(h.breached_falsifiers),
            }
            for h in holdings_advisory
        ],
        "quarantined": [[q.symbol, q.guardrail_code] for q in quarantined_holdings],
        "warnings": [[w.guardrail_code, w.ticker, w.severity] for w in anomaly_warnings],
        "look_through": lt_digest,
    }
    encoded = json.dumps(canonical_payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]


def evaluate_portfolio_advisory(
    portfolio_state: Any,
    macro_direction: Optional[Any] = None,
    sector_rotation_snapshot: Optional[Any] = None,
    equity_sidecars: Optional[Dict[str, Any]] = None,
    watchlist_sidecars: Optional[Dict[str, Any]] = None,
    cached_narrative: Optional[Dict[str, Any]] = None,
    look_through_summary: Optional[Any] = None,
    peak_nav_thb: Optional[Decimal] = None,
    flow_adjusted_current_nav_thb: Optional[Decimal] = None,
    enable_drawdown_defense: bool = True,
) -> PortfolioAdvisoryQuantReport:
    """Deterministic Tier 1 Advisory Engine executing Stages A through D."""
    evaluated_at = datetime.now(timezone.utc).isoformat()
    warnings: list[DataAnomalyWarning] = []
    quarantined: list[QuarantinedHoldingItem] = []

    # =================================================================
    # Stage A: Pre-flight Guardrails, Currency Resolution & Filtering
    # =================================================================
    # 1. Determine Graceful Degradation Mode (S4)
    has_macro = macro_direction is not None
    has_micro = bool(equity_sidecars)
    if has_macro and has_micro:
        data_mode = AdvisoryDataMode.FULL_CROSS_SCALE
    elif has_micro:
        data_mode = AdvisoryDataMode.PORTFOLIO_PLUS_MICRO
    else:
        data_mode = AdvisoryDataMode.PORTFOLIO_BUCKET_ONLY

    # 2. Macro Temporal Checks (T1, T3)
    macro_evaluated_at = None
    if macro_direction:
        macro_evaluated_at = _extract(macro_direction, "evaluated_at")
        t1_warn = check_future_date_anomaly(macro_evaluated_at)
        if t1_warn:
            warnings.append(t1_warn)

        sector_as_of = None
        if sector_rotation_snapshot:
            sector_as_of = _extract(sector_rotation_snapshot, "as_of_date")
        macro_freshness, t3_warns = check_macro_cross_scale_drift(macro_evaluated_at, sector_as_of)
        warnings.extend(t3_warns)

    # 3. FX Rate Validation (I2)
    raw_fx = getattr(portfolio_state.summary, "usd_thb_rate", None) if hasattr(portfolio_state, "summary") else None
    if not isinstance(raw_fx, (int, float, str, Decimal)):
        fx_rates_dict = getattr(portfolio_state, "fx_rates", None)
        if isinstance(fx_rates_dict, dict):
            raw_fx = fx_rates_dict.get("USDTHB") or fx_rates_dict.get("USD_THB")
    fx_rate, fx_warn = validate_fx_rate(safe_decimal(raw_fx, default=Decimal("35.50")))
    if fx_warn:
        warnings.append(fx_warn)

    # 4. Filter Active Holdings & Separate Cash Ledger (D2)
    raw_holdings = getattr(portfolio_state, "holdings", []) or []
    active_holdings = []
    cash_thb = Decimal("0")
    cash_usd = Decimal("0")
    cash_bucket_val_thb: dict[str, Decimal] = {}

    for h in raw_holdings:
        status = _extract(h, "status")
        if isinstance(status, str) and status.lower() in ("archived", "inactive"):
            continue
        sym = str(getattr(h, "symbol", "")).strip().upper()
        units = safe_decimal(getattr(h, "units", None), quantum="0.0001")
        if units <= Decimal("0"):
            continue

        asset_type = str(getattr(h, "asset_type", "")).strip().lower()
        h_bid = _extract(h, "bucket_id")
        if sym in ("CASH_THB", "THB") or asset_type == "cash_thb":
            thb_val = safe_decimal(_extract(h, "market_value_thb") or _extract(h, "units"))
            cash_thb += thb_val
            if isinstance(h_bid, str) and h_bid:
                cash_bucket_val_thb[h_bid] = cash_bucket_val_thb.get(h_bid, Decimal("0")) + thb_val
        elif sym in ("CASH_USD", "USD") or asset_type == "cash_usd":
            usd_val = safe_decimal(_extract(h, "market_value_usd") or _extract(h, "units"))
            cash_usd += usd_val
            if isinstance(h_bid, str) and h_bid:
                cash_bucket_val_thb[h_bid] = cash_bucket_val_thb.get(h_bid, Decimal("0")) + (usd_val * fx_rate)
        elif asset_type == "cash":
            # Determine cash currency
            curr = str(getattr(h, "currency", "THB")).strip().upper()
            if curr == "USD":
                usd_val = safe_decimal(_extract(h, "market_value_usd") or _extract(h, "units"))
                cash_usd += usd_val
                if isinstance(h_bid, str) and h_bid:
                    cash_bucket_val_thb[h_bid] = cash_bucket_val_thb.get(h_bid, Decimal("0")) + (usd_val * fx_rate)
            else:
                thb_val = safe_decimal(_extract(h, "market_value_thb") or _extract(h, "units"))
                cash_thb += thb_val
                if isinstance(h_bid, str) and h_bid:
                    cash_bucket_val_thb[h_bid] = cash_bucket_val_thb.get(h_bid, Decimal("0")) + thb_val
        else:
            active_holdings.append(h)

    # If portfolio summary has cash_balance, incorporate if not already accounted
    if hasattr(portfolio_state, "summary") and portfolio_state.summary:
        summary_cash = safe_decimal(getattr(portfolio_state.summary, "cash_balance_thb", None))
        if cash_thb == Decimal("0") and summary_cash > Decimal("0"):
            cash_thb = summary_cash

    # 5. Check Bucket Target Sum (I3) — support both `allocation_targets` (PortfolioState) and `buckets` (mocks)
    raw_alloc_targets = getattr(portfolio_state, "allocation_targets", None)
    raw_buckets = getattr(portfolio_state, "buckets", None)
    if isinstance(raw_alloc_targets, (list, tuple)) and len(raw_alloc_targets) > 0:
        buckets = list(raw_alloc_targets)
    elif isinstance(raw_buckets, (list, tuple)):
        buckets = list(raw_buckets)
    else:
        buckets = []

    _, b_warn = validate_bucket_target_sum(buckets)
    if b_warn:
        warnings.append(b_warn)

    bucket_target_map: dict[str, Decimal] = {}
    bucket_name_map: dict[str, str] = {}
    for b in buckets:
        bid = str(_extract(b, "bucket_id") or _extract(b, "id") or "")
        bname = str(_extract(b, "name") or bid)
        bpct = safe_decimal(_extract(b, "target_percent"), quantum="0.01")
        if bid:
            bucket_target_map[bid] = bpct
            bucket_name_map[bid] = bname

    bucket_holding_counts: dict[str, int] = {}
    bucket_actual_val_thb: dict[str, Decimal] = {bid: Decimal("0") for bid in bucket_target_map}
    total_cash_thb = cash_thb + (cash_usd * fx_rate)
    bucketed_cash_total = Decimal("0")
    for cb_id, cb_val in cash_bucket_val_thb.items():
        if cb_val > Decimal("0"):
            bucket_actual_val_thb[cb_id] = bucket_actual_val_thb.get(cb_id, Decimal("0")) + cb_val
            bucketed_cash_total += cb_val

    unbucketed_cash_thb = max(Decimal("0"), total_cash_thb - bucketed_cash_total)
    if unbucketed_cash_thb > Decimal("0"):
        cash_bucket_key = "cash" if "cash" in bucket_target_map else "__unbucketed_cash__"
        bucket_actual_val_thb[cash_bucket_key] = (
            bucket_actual_val_thb.get(cash_bucket_key, Decimal("0")) + unbucketed_cash_thb
        )

    unassigned_val_thb = Decimal("0")
    for h in active_holdings:
        h_mkt_val = safe_decimal(getattr(h, "market_value_thb", None))
        raw_bid = _extract(h, "bucket_id")
        bid = raw_bid.strip() if isinstance(raw_bid, str) else ""
        if bid:
            bucket_holding_counts[bid] = bucket_holding_counts.get(bid, 0) + 1
            bucket_actual_val_thb[bid] = bucket_actual_val_thb.get(bid, Decimal("0")) + h_mkt_val
        else:
            unassigned_val_thb += h_mkt_val
            if h_mkt_val > Decimal("0"):
                bucket_actual_val_thb["__unassigned__"] = (
                    bucket_actual_val_thb.get("__unassigned__", Decimal("0")) + h_mkt_val
                )

    # =================================================================
    # Stage B: Risk Radar, Concentration & Sector Alignment
    # =================================================================
    total_holdings_val_thb = sum(
        (safe_decimal(getattr(h, "market_value_thb", None)) for h in active_holdings),
        Decimal("0"),
    )
    total_nav_thb = total_holdings_val_thb + total_cash_thb

    # Drawdown Circuit Breaker Evaluation (R16 / W03)
    summary_obj = getattr(portfolio_state, "summary", None)
    effective_peak_nav = (
        _extract_unrounded_positive_decimal(peak_nav_thb)
        if peak_nav_thb is not None
        else _extract_unrounded_positive_decimal(_extract(summary_obj, "flow_adjusted_peak_nav_thb"))
    )
    effective_dd_current_nav = (
        _extract_unrounded_positive_decimal(flow_adjusted_current_nav_thb)
        if flow_adjusted_current_nav_thb is not None
        else (
            _extract_unrounded_positive_decimal(_extract(summary_obj, "flow_adjusted_nav_thb"))
            or total_nav_thb
        )
    )
    drawdown_posture = evaluate_drawdown_circuit_breaker(
        current_nav_thb=effective_dd_current_nav,
        peak_nav_thb=effective_peak_nav if enable_drawdown_defense else None,
    )
    if drawdown_posture.is_peak_available and drawdown_posture.posture_tier != DrawdownPostureTier.NORMAL:
        warnings.append(
            DataAnomalyWarning(
                guardrail_code="DRAWDOWN_CIRCUIT_BREAKER_ACTIVE",
                ticker="PORTFOLIO",
                message=(
                    f"Drawdown circuit breaker {drawdown_posture.posture_tier.value} active "
                    f"(DD={drawdown_posture.drawdown_pct}%, hurdle +{drawdown_posture.hurdle_adjustment})."
                ),
                severity="critical" if drawdown_posture.posture_tier == DrawdownPostureTier.TIER_3_DEFENSIVE else "warning",
                remedy_hint="; ".join(drawdown_posture.action_restrictions),
            )
        )

    bucket_actual_pct_map: dict[str, Decimal] = {}
    active_share_actual_pct_map: dict[str, Decimal] = {}
    purpose_bucket_drift: dict[str, Decimal] = {}
    if total_nav_thb > Decimal("0"):
        for bid, actual_val in bucket_actual_val_thb.items():
            if bid in bucket_target_map or actual_val > Decimal("0"):
                raw_pct = (actual_val / total_nav_thb) * Decimal("100")
                active_share_actual_pct_map[bid] = raw_pct
                bucket_actual_pct_map[bid] = safe_decimal(raw_pct, quantum="0.1")
        for bid, target_pct in bucket_target_map.items():
            actual_pct = bucket_actual_pct_map.get(bid, Decimal("0.0"))
            purpose_bucket_drift[bid] = safe_decimal(actual_pct - target_pct, quantum="0.1")

    unassigned_nav_pct = (
        (unassigned_val_thb / total_nav_thb * Decimal("100")) if total_nav_thb > Decimal("0") else Decimal("0")
    )
    unassigned_drift_active, u_warn = check_unassigned_holdings_drift(unassigned_val_thb, total_nav_thb)
    if u_warn:
        warnings.append(u_warn)

    # Sector Crowding & FX Split
    sector_exposure_thb: dict[str, Decimal] = {}
    thb_exposure = cash_thb
    usd_exposure = cash_usd * fx_rate

    # Macro leading/lagging sectors and quadrants
    leading_sectors: list[str] = []
    lagging_sectors: list[str] = []
    if macro_direction:
        sector_analysis = getattr(macro_direction, "sector_analysis", None) or (
            macro_direction.get("sector_analysis") if isinstance(macro_direction, dict) else {}
        )
        if sector_analysis:
            leading_sectors = [str(s).strip() for s in sector_analysis.get("leading_sectors", [])]
            lagging_sectors = [str(s).strip() for s in sector_analysis.get("lagging_sectors", [])]

    sector_quadrant_members: dict[str, list[str]] = {
        "Leading": [], "Improving": [], "Weakening": [], "Lagging": []
    }
    if sector_rotation_snapshot:
        # Extract quadrants from rows if available (support both list and dict history)
        rows = _extract(sector_rotation_snapshot, "rows", []) or []
        for r in rows:
            ticker = str(_extract(r, "ticker", "") or "")
            q = _extract(r, "quadrant")
            if not q:
                curr_obj = _extract(r, "current")
                q = _extract(curr_obj, "quadrant")
            if not q:
                history = _extract(r, "history")
                points = []
                if isinstance(history, list):
                    points = history
                elif isinstance(history, dict):
                    points = history.get("weekly") or history.get("daily") or []
                if points:
                    last_p = points[-1]
                    q = _extract(last_p, "quadrant")
            if q in sector_quadrant_members and ticker:
                if ticker not in sector_quadrant_members[q]:
                    sector_quadrant_members[q].append(ticker)
                can_sec = normalize_to_canonical_sector(ticker)
                if can_sec and can_sec not in sector_quadrant_members[q]:
                    sector_quadrant_members[q].append(can_sec)

    # Beta and holding tracking
    weighted_beta_sum = Decimal("0")
    beta_weight_sum = Decimal("0")

    # =================================================================
    # Stage C & D: Decision Matrix, Lot Rounding & Dual-Currency Budget
    # =================================================================
    holdings_advisory: list[HoldingAdvisoryItem] = []
    validated_signals_by_idx: dict[int, Any] = {}
    execution_market_by_idx: dict[int, Literal["TH", "US", "GLOBAL"]] = {}
    immediate_deploy_thb = Decimal("0")
    immediate_deploy_usd = Decimal("0")
    immediate_friction_thb = Decimal("0")
    immediate_friction_usd = Decimal("0")
    conditional_limit_thb = Decimal("0")
    conditional_limit_usd = Decimal("0")

    for h in active_holdings:
        sym = str(getattr(h, "symbol", "")).strip().upper()
        clean_sym = sym.replace(".BK", "").strip().upper()
        name = str(_extract(h, "company_name") or _extract(h, "name") or sym)
        asset_type = str(getattr(h, "asset_type", "Stock")).strip()
        current_units = safe_decimal(getattr(h, "units", None), quantum="0.0001")
        h_val_thb = safe_decimal(getattr(h, "market_value_thb", None))
        current_weight_pct = (
            safe_decimal((h_val_thb / total_nav_thb * Decimal("100")), quantum="0.1")
            if total_nav_thb > Decimal("0")
            else Decimal("0")
        )

        market, native_curr = resolve_holding_currency_and_market(h)

        if native_curr == "USD":
            usd_exposure += h_val_thb
            current_price_native = safe_decimal(
                getattr(h, "current_price_usd", None) or getattr(h, "current_price", None),
                quantum="0.0001"
            )
        else:
            thb_exposure += h_val_thb
            current_price_native = safe_decimal(
                getattr(h, "current_price_thb", None) or getattr(h, "current_price", None),
                quantum="0.0001"
            )

        # Classify 3 Tracks
        lower_type = asset_type.lower()
        if any(k in lower_type for k in ("rmf", "ssf", "thaiesg", "pvd")):
            eval_track = AssetEvaluationTrack.TRACK_3_LOCKED_VEHICLE
        elif any(k in lower_type for k in ("fund", "mutual_fund", "gold", "bond", "etf")):
            eval_track = AssetEvaluationTrack.TRACK_2_MACRO_LIQUID
        else:
            eval_track = AssetEvaluationTrack.TRACK_1_DIRECT_EQUITIES

        # Find market-validated sidecar and nested quant_signals
        sidecar, sidecar_signals = _resolve_validated_sidecar(sym, clean_sym, market, equity_sidecars)

        # Sector tracking
        stock_sector = _extract(sidecar_signals, "peer_sector") or _extract(h, "sector", "Other")
        can_sector = normalize_to_canonical_sector(stock_sector) or "Other"
        sector_exposure_thb[can_sector] = sector_exposure_thb.get(can_sector, Decimal("0")) + h_val_thb

        # Temporal and Price Sanity Guardrails (T2, I1)
        h_price_date = _extract(h, "price_as_of") or _extract(h, "as_of_date")
        sc_date = _extract(sidecar, "analysis_date")
        is_price_stale, stale_warn = check_stale_portfolio_price(h_price_date, sc_date)
        if stale_warn:
            warnings.append(replace(stale_warn, ticker=sym))

        sidecar_price = safe_decimal(_extract(sidecar_signals, "raw_analysis_price"), default=None, quantum="0.0001")

        # Hard-Fail Missing Price Policy (Ch. 8, p. 57)
        missing_price_warn = check_missing_price_policy(
            current_price_native=current_price_native,
            holding_units=current_units,
            asset_type=asset_type,
            ticker=sym,
        )
        if missing_price_warn:
            warnings.append(missing_price_warn)
            quarantined.append(QuarantinedHoldingItem(
                symbol=sym,
                reason=missing_price_warn.message,
                guardrail_code=missing_price_warn.guardrail_code,
                current_value_thb=h_val_thb,
                details=f"Holding units={current_units}, Current price={current_price_native}",
            ))
            validated_signals_by_idx[len(holdings_advisory)] = sidecar_signals
            holdings_advisory.append(HoldingAdvisoryItem(
                symbol=sym,
                asset_name=name,
                asset_type=asset_type,
                market=market,
                native_currency=native_curr,
                evaluation_track=eval_track,
                bucket_id=_extract(h, "bucket_id"),
                bucket_name=bucket_name_map.get(str(_extract(h, "bucket_id", ""))),
                current_units=current_units,
                current_price_native=current_price_native,
                current_value_thb=h_val_thb,
                current_weight_pct=current_weight_pct,
                target_weight_pct=current_weight_pct,
                weight_drift_pct=Decimal("0.0"),
                action_code=AdvisoryActionCode.QUARANTINED_DATA_ANOMALY,
                action_rationale=f"กักกันสินทรัพย์ชั่วคราว: {missing_price_warn.message}",
                data_warnings=[missing_price_warn.message],
            ))
            continue

        price_anomaly_warn = check_price_scale_anomaly(current_price_native, sidecar_price, ticker=sym)
        if price_anomaly_warn:
            warnings.append(price_anomaly_warn)
            quarantined.append(QuarantinedHoldingItem(
                symbol=sym,
                reason=price_anomaly_warn.message,
                guardrail_code=price_anomaly_warn.guardrail_code,
                current_value_thb=h_val_thb,
                details=f"Portfolio price={current_price_native}, Sidecar price={sidecar_price}",
            ))
            # Quarantined: Skip rebalancing advice
            validated_signals_by_idx[len(holdings_advisory)] = sidecar_signals
            holdings_advisory.append(HoldingAdvisoryItem(
                symbol=sym,
                asset_name=name,
                asset_type=asset_type,
                market=market,
                native_currency=native_curr,
                evaluation_track=eval_track,
                bucket_id=_extract(h, "bucket_id"),
                bucket_name=bucket_name_map.get(str(_extract(h, "bucket_id", ""))),
                current_units=current_units,
                current_price_native=current_price_native,
                current_value_thb=h_val_thb,
                current_weight_pct=current_weight_pct,
                target_weight_pct=current_weight_pct,
                weight_drift_pct=Decimal("0.0"),
                action_code=AdvisoryActionCode.QUARANTINED_DATA_ANOMALY,
                action_rationale=f"กักกันสินทรัพย์ชั่วคราว: {price_anomaly_warn.message}",
                data_warnings=[price_anomaly_warn.message],
            ))
            continue

        # Extract Micro Signals
        core_conv = None
        biz_conv = None
        price_stage = None
        action_stance = None
        stop_loss = None
        buy_zone_min = None
        buy_zone_max = None
        target_price = None
        breached_falsifiers: list[str] = []

        if sidecar_signals:
            scorecard = _extract(sidecar_signals, "deterministic_scorecard")
            if scorecard:
                core_conv = safe_decimal(_extract(scorecard, "core_conviction_score"), default=None, quantum="0.1")
                biz_conv = safe_decimal(_extract(scorecard, "business_conviction_score"), default=None, quantum="0.1")
                action_stance = _extract(scorecard, "action_stance")

            tactical = _extract(sidecar_signals, "tactical_setup")
            if tactical:
                price_stage = _extract(tactical, "price_stage")
                stop_loss = safe_decimal(_extract(tactical, "invalidation_stop_loss"), default=None, quantum="0.0001")
                buy_zone_min = safe_decimal(_extract(tactical, "buy_zone_min"), default=None, quantum="0.0001")
                buy_zone_max = safe_decimal(_extract(tactical, "buy_zone_max"), default=None, quantum="0.0001")

            rdcf = _extract(sidecar_signals, "reverse_dcf_result")
            if rdcf:
                raw_target = _extract(rdcf, "target_price_12m") or _extract(rdcf, "alternative_target_price_12m")
                target_price = safe_decimal(raw_target, default=None, quantum="0.0001")

            raw_beta = _extract(sidecar_signals, "beta")
            if raw_beta is None:
                raw_beta = _extract(sidecar_signals, "beta_5y")
            beta_val = safe_decimal(raw_beta, default=None, quantum="0.01")
            if beta_val and beta_val > Decimal("0"):
                weighted_beta_sum += beta_val * h_val_thb
                beta_weight_sum += h_val_thb

            raw_falsifiers = _extract(sidecar_signals, "thesis_falsifiers") or _extract(sidecar, "thesis_falsifiers", [])
            for f in raw_falsifiers or []:
                thresh = safe_decimal(_extract(f, "threshold_value"), default=None, quantum="0.0001")
                fid = str(_extract(f, "falsifier_id", ""))
                m_name = str(_extract(f, "metric_name", ""))
                if thresh and thresh > Decimal("0") and ("STOP_LOSS" in fid or "Support" in m_name):
                    if current_price_native < thresh:
                        breached_falsifiers.append(f"Breached Stop Level ({current_price_native} < {thresh})")

        # T5: Live Upside Re-anchoring
        live_upside = compute_live_upside(current_price_native, target_price)

        # Macro Alignment & Quadrant
        quadrant = None
        for q_name, q_list in sector_quadrant_members.items():
            if any(is_sector_match(can_sector, m) for m in q_list):
                quadrant = q_name
                break

        macro_align = "unavailable"
        if has_macro:
            if quadrant == "Leading" or any(is_sector_match(can_sector, s) for s in leading_sectors):
                macro_align = "tailwind"
            elif quadrant == "Lagging" or any(is_sector_match(can_sector, s) for s in lagging_sectors):
                macro_align = "headwind"
            else:
                macro_align = "neutral"

        # Bucket Target & Drift (per-holding equal share of bucket target)
        bid = _extract(h, "bucket_id")
        bname = bucket_name_map.get(str(bid), str(bid or "Unassigned"))
        bucket_target_pct = bucket_target_map.get(str(bid), Decimal("0"))
        actual_bucket_pct = bucket_actual_pct_map.get(str(bid), Decimal("0"))
        bucket_drift_pct = purpose_bucket_drift.get(str(bid), Decimal("0"))

        target_weight_pct = current_weight_pct
        if bid and bid in bucket_target_map:
            b_count = max(1, bucket_holding_counts.get(bid, 1))
            target_weight_pct = safe_decimal(
                bucket_target_map[bid] / Decimal(str(b_count)),
                quantum="0.1",
            )

        weight_drift_pct = safe_decimal(current_weight_pct - target_weight_pct, quantum="0.1")
        pnl_pct = safe_decimal(_extract(h, "unrealized_pnl_percent"), default=None, quantum="0.1")

        # -------------------------------------------------------------
        # Decision Matrix & Action Determination
        # -------------------------------------------------------------
        action_code = AdvisoryActionCode.HOLD_WITHIN_BAND
        rationale = f"สัดส่วน ({current_weight_pct}%) อยู่ในกรอบสมดุลของถัง {bname} (เป้าหมาย {target_weight_pct}%)"
        is_immediate = False
        target_rebalance_val_thb = Decimal("0")

        # Track 3: Tax Locked Preservation
        if eval_track == AssetEvaluationTrack.TRACK_3_LOCKED_VEHICLE:
            action_code = AdvisoryActionCode.HOLD_LOCKED_VEHICLE
            rationale = "สินทรัพย์ลดหย่อนภาษี/เงินสำรองเกษียณ — ห้ามสั่งขายเพื่อปรับพอร์ต"

        # Anti-Falling-Knife & Urgent Reduce (D5)
        elif (
            breached_falsifiers
            or (stop_loss and stop_loss > Decimal("0") and current_price_native < stop_loss)
            or action_stance in ("REDUCE", "AVOID", "SELL")
            or (
                price_stage == "STAGE_4_MARKDOWN"
                and (
                    (core_conv is not None and core_conv < Decimal("5.0"))
                    or (biz_conv is not None and biz_conv < Decimal("5.0"))
                )
            )
        ):
            action_code = AdvisoryActionCode.URGENT_REDUCE
            if breached_falsifiers:
                rationale = f"ตรวจพบสัญญาณ Thesis Falsifier ({', '.join(breached_falsifiers[:2])}) — ตัดวงจรความเสียหาย (Stop Loss / Falsifier Breached)"
            elif stop_loss and stop_loss > Decimal("0") and current_price_native < stop_loss:
                rationale = f"ราคาปัจจุบัน ({current_price_native}) หลุดระดับ Stop Loss ({stop_loss}) — ตัดวงจรความเสียหาย (Stop Loss Breached)"
            else:
                rationale = "หุ้นอยู่ใน Stage 4 ขาลงรุนแรงและปัจจัยพื้นฐานอ่อนแอ — แนะนำลดสัดส่วนเพื่อตัดวงจรความเสียหาย (Stop Loss / Stage 4 Breakdown)"
            # Recommend trimming 50% or full position
            target_rebalance_val_thb = -(h_val_thb * Decimal("0.5"))
            is_immediate = True

        # Track 1 Direct Equities: Dynamic Conviction Cap & Winner Let Run (15% / 20% single-stock cap)
        elif eval_track == AssetEvaluationTrack.TRACK_1_DIRECT_EQUITIES and current_weight_pct > Decimal("15.0"):
            is_winner = bool(biz_conv and biz_conv >= Decimal("8.0") and (pnl_pct is not None and pnl_pct > Decimal("0")))
            if is_winner and current_weight_pct <= Decimal("20.0"):
                action_code = AdvisoryActionCode.LET_WINNER_RUN
                rationale = f"หุ้นผู้ชนะคุณภาพสูง (Conviction {biz_conv}/10) — ปรับเพดานสัดส่วนเป็น 20% เพื่อให้กำไรเติบโตเต็มที่"
            else:
                action_code = AdvisoryActionCode.TRIM_OVERWEIGHT
                cap = Decimal("20.0") if is_winner else Decimal("15.0")
                rationale = f"สัดส่วนหุ้นรายตัว ({current_weight_pct}%) เกินเพดานความเสี่ยงสูงสุด ({cap}%) — ทยอยขายทำกำไรลดความกระจุกตัว"
                excess_pct = current_weight_pct - cap
                target_rebalance_val_thb = -((excess_pct / Decimal("100")) * total_nav_thb)
                is_immediate = True

        # Track 2 Macro / Mutual Funds / Gold: Rebalance based on Strategy Bucket Drift (not 15% single-stock cap)
        elif eval_track == AssetEvaluationTrack.TRACK_2_MACRO_LIQUID and (
            (bucket_drift_pct > Decimal("5.0") and weight_drift_pct > Decimal("3.5") and current_weight_pct > Decimal("10.0"))
            or current_weight_pct > Decimal("30.0")
        ):
            action_code = AdvisoryActionCode.TRIM_OVERWEIGHT
            trim_pct = min(weight_drift_pct, bucket_drift_pct) if bucket_drift_pct > Decimal("0") else (current_weight_pct - Decimal("25.0"))
            trim_pct = max(Decimal("2.0"), safe_decimal(trim_pct, quantum="0.1"))
            rationale = (
                f"สัดส่วน ({current_weight_pct}%) ทำให้ถัง '{bname}' เกินเป้าหมาย ({actual_bucket_pct}% เทียบเป้า {bucket_target_pct}%) "
                f"— แนะนำทยอยปรับลดสัดส่วนลง {trim_pct}% เพื่อสมดุลพอร์ต"
            )
            target_rebalance_val_thb = -((trim_pct / Decimal("100")) * total_nav_thb)
            is_immediate = True

        # Accumulation / Buy Actions (Anti-Falling-Knife blocks buys during STAGE_4_MARKDOWN)
        elif (
            action_stance in ("ACCUMULATE_NOW", "BREAKOUT_BUY")
            and price_stage != "STAGE_4_MARKDOWN"
            and not is_price_stale
        ):
            in_buy_zone = (
                buy_zone_max is None or current_price_native <= buy_zone_max
            )
            if in_buy_zone and not (unassigned_drift_active and weight_drift_pct < Decimal("0")):
                action_code = (
                    AdvisoryActionCode.BREAKOUT_BUY
                    if action_stance == "BREAKOUT_BUY"
                    else AdvisoryActionCode.ACCUMULATE_NOW
                )
                add_pct = (
                    min(max(abs(weight_drift_pct), Decimal("2.0")), Decimal("4.0"))
                    if weight_drift_pct < Decimal("0")
                    else Decimal("2.5")
                )
                rationale = f"สัญญาณ Conviction สูง ({core_conv}/10) และราคาอยู่ในโซนสะสม — แนะนำเคาะซื้อวันนี้ (Immediate Execution)"
                is_immediate = True
                target_rebalance_val_thb = (add_pct / Decimal("100")) * total_nav_thb

        elif price_stage != "STAGE_4_MARKDOWN" and (
            action_stance == "ACCUMULATE_ON_DIP"
            or (
                action_stance in ("ACCUMULATE_NOW", "BREAKOUT_BUY")
                and buy_zone_max
                and current_price_native > buy_zone_max
            )
        ):
            action_code = AdvisoryActionCode.ACCUMULATE_ON_DIP
            add_pct = (
                min(max(abs(weight_drift_pct), Decimal("1.5")), Decimal("3.5"))
                if weight_drift_pct < Decimal("0")
                else Decimal("2.0")
            )
            rationale = f"ปัจจัยพื้นฐานดีแต่ราคาปัจจุบันสูงกว่าโซนซื้อ — ตั้ง Limit Order หรือรอรับในกรอบ ({buy_zone_min} – {buy_zone_max})"
            is_immediate = False
            target_rebalance_val_thb = (add_pct / Decimal("100")) * total_nav_thb

        elif is_price_stale:
            action_code = AdvisoryActionCode.WAIT_PRICE_REFRESH
            rationale = "ราคาค้างเกิน 3 วันทำการ — ระงับคำสั่งซื้อขายจนกว่าจะรีเฟรชราคาตลาดล่าสุด"

        else:
            # Portfolio-aware contextual rationale for holdings within band or awaiting micro sidecar
            if price_stage == "STAGE_4_MARKDOWN":
                if stop_loss and stop_loss > Decimal("0"):
                    rationale = (
                        f"แนวโน้มราคายังอยู่ใน Stage 4 ขาลง (กฎ Anti-Falling-Knife: งดซื้อถัวเฉลี่ย) "
                        f"แต่ราคาปัจจุบัน ({current_price_native}) ยังยืนเหนือจุด Stop Loss ({stop_loss}) — ถือรอดูความชัดเจนและระวังหากหลุด {stop_loss}"
                    )
                else:
                    rationale = (
                        "แนวโน้มราคายังอยู่ใน Stage 4 ขาลง (กฎ Anti-Falling-Knife: งดซื้อถัวเฉลี่ย) "
                        "— ถือรอดูสัญญาณสร้างฐานก่อนพิจารณาเพิ่มสัดส่วน"
                    )
            elif eval_track == AssetEvaluationTrack.TRACK_2_MACRO_LIQUID:
                if bucket_drift_pct < Decimal("-5.0"):
                    rationale = (
                        f"สินทรัพย์หลักในถัง '{bname}' (สัดส่วน {current_weight_pct}%, ถังรวม {actual_bucket_pct}% "
                        f"ต่ำกว่าเป้าถัง {bucket_target_pct}% อยู่ {bucket_drift_pct}%) — ถือครองเป็นแกนหลักของถัง"
                    )
                else:
                    rationale = (
                        f"สินทรัพย์กองทุน/มหภาคในถัง '{bname}' (สัดส่วน {current_weight_pct}%, "
                        f"ถังรวม {actual_bucket_pct}% เทียบเป้า {bucket_target_pct}%) — ถือครองตามกรอบกลยุทธ์"
                    )
            elif sidecar_signals is None:
                if pnl_pct is not None and pnl_pct <= Decimal("-15.0"):
                    rationale = (
                        f"ขาดทุนสะสม {pnl_pct:+.1f}% (สัดส่วน {current_weight_pct}% ในถัง '{bname}') "
                        f"— ยังไม่มีผลวิเคราะห์หุ้นรายตัว แนะนำรัน Equity Analysis เพื่อประเมินพื้นฐานและจุด Stop Loss"
                    )
                elif weight_drift_pct < Decimal("-2.5"):
                    rationale = (
                        f"สัดส่วนปัจจุบัน ({current_weight_pct}%) ต่ำกว่าเป้าเฉลี่ยในถัง '{bname}' ({target_weight_pct}%) "
                        f"อยู่ {weight_drift_pct}% — แนะนำรัน Equity Analysis เพื่อประเมินจุดเข้าสะสมเพิ่ม"
                    )
                elif pnl_pct is not None:
                    rationale = (
                        f"สัดส่วน {current_weight_pct}% ในถัง '{bname}' (เป้าหมาย {target_weight_pct}%, "
                        f"กำไร/ขาดทุนสะสม {pnl_pct:+.1f}%) — ถือครองตามกรอบสมดุล"
                    )

        # Enforce Drawdown Circuit Breaker posture restrictions & raised conviction hurdles (R16 / W03)
        holding_warnings: list[str] = []
        if (
            drawdown_posture.is_peak_available
            and drawdown_posture.posture_tier != DrawdownPostureTier.NORMAL
            and action_code in (
                AdvisoryActionCode.ACCUMULATE_NOW,
                AdvisoryActionCode.BREAKOUT_BUY,
                AdvisoryActionCode.ACCUMULATE_ON_DIP,
            )
        ):
            base_hurdle = (
                Decimal("6.5")
                if action_code == AdvisoryActionCode.ACCUMULATE_ON_DIP
                else Decimal("7.5")
            )
            effective_hurdle = base_hurdle + drawdown_posture.hurdle_adjustment
            bid_lower = str(bid or "").strip().lower()
            bname_lower = bname.lower()
            is_core_bucket = ("core" in bid_lower) or ("core" in bname_lower)

            block_reason: Optional[str] = None
            if drawdown_posture.posture_tier == DrawdownPostureTier.TIER_3_DEFENSIVE:
                block_reason = (
                    f"Drawdown Circuit Breaker (TIER_3_DEFENSIVE, DD {drawdown_posture.drawdown_pct}%): "
                    "ระงับคำสั่งซื้อใหม่ทั้งหมดเพื่อรักษาเงินต้น"
                )
            elif drawdown_posture.posture_tier == DrawdownPostureTier.TIER_2_THROTTLE:
                if not is_core_bucket:
                    block_reason = (
                        f"Drawdown Circuit Breaker (TIER_2_THROTTLE, DD {drawdown_posture.drawdown_pct}%): "
                        f"จำกัดคำสั่งซื้อเฉพาะสินทรัพย์ถัง Core เท่านั้น (ถัง '{bname}' ถูกชะลอ)"
                    )
                elif core_conv is None or core_conv < effective_hurdle:
                    block_reason = (
                        f"Drawdown Circuit Breaker (TIER_2_THROTTLE, +1.0 hurdle): "
                        f"Core Conviction ({core_conv}) ต่ำกว่าเกณฑ์ขั้นต่ำที่ปรับเพิ่ม ({effective_hurdle}/10)"
                    )
            elif drawdown_posture.posture_tier == DrawdownPostureTier.TIER_1_CAUTION:
                if core_conv is None or core_conv < effective_hurdle:
                    block_reason = (
                        f"Drawdown Circuit Breaker (TIER_1_CAUTION, +0.5 hurdle): "
                        f"Core Conviction ({core_conv}) ต่ำกว่าเกณฑ์ขั้นต่ำที่ปรับเพิ่ม ({effective_hurdle}/10)"
                    )

            if block_reason:
                action_code = AdvisoryActionCode.HOLD_WITHIN_BAND
                target_rebalance_val_thb = Decimal("0")
                is_immediate = False
                rationale = block_reason
                holding_warnings.append(block_reason)

        # Lot Sizing Calculation (fee-inclusive for BUY budgets, exchange-aware for Track 2 funds/bonds/gold)
        execution_market: Literal["TH", "US", "GLOBAL"] = (
            "GLOBAL"
            if (eval_track == AssetEvaluationTrack.TRACK_2_MACRO_LIQUID and "etf" not in lower_type)
            else ("US" if market == "US" else "TH")
        )
        adtv_obs = _extract_positive_decimal(sidecar_signals, "adtv_local_currency", quantum="0.01")
        vol_obs = _extract_positive_decimal(sidecar_signals, "volatility_pct", quantum="0.0001")

        lot_sizing = None
        if target_rebalance_val_thb != Decimal("0") and current_price_native > Decimal("0"):
            execution_price_native = (
                buy_zone_max
                if (
                    action_code == AdvisoryActionCode.ACCUMULATE_ON_DIP
                    and buy_zone_max is not None
                    and buy_zone_max > Decimal("0")
                )
                else current_price_native
            )
            native_amt = (
                target_rebalance_val_thb / fx_rate if native_curr == "USD" else target_rebalance_val_thb
            )
            if target_rebalance_val_thb > Decimal("0"):
                budget_native = safe_decimal(abs(native_amt), quantum="0.01") or Decimal("0.00")
                lot_sizing = calculate_fee_inclusive_lot_sizing(
                    market=execution_market,
                    native_currency=native_curr,
                    max_cash_budget_native=budget_native,
                    current_price_native=execution_price_native,
                    fx_rate=fx_rate,
                    adtv_native=adtv_obs,
                    annualized_volatility_pct=vol_obs,
                    is_tax_locked=(eval_track == AssetEvaluationTrack.TRACK_3_LOCKED_VEHICLE),
                )
                if lot_sizing.amount_native > Decimal("0"):
                    f_buy = estimate_order_friction(
                        market=execution_market,
                        action_side="BUY",
                        order_amount_native=lot_sizing.amount_native,
                        adtv_native=adtv_obs,
                        annualized_volatility_pct=vol_obs,
                    )
                    if is_immediate:
                        if native_curr == "USD":
                            immediate_deploy_usd += lot_sizing.amount_native
                            immediate_friction_usd += f_buy.total_friction_native
                        else:
                            immediate_deploy_thb += lot_sizing.amount_thb
                            immediate_friction_thb += f_buy.total_friction_native
                    else:
                        if native_curr == "USD":
                            conditional_limit_usd += lot_sizing.amount_native
                        else:
                            conditional_limit_thb += lot_sizing.amount_thb
            else:
                lot_sizing = calculate_market_lot_sizing(
                    market=execution_market,
                    native_currency=native_curr,
                    target_amount_native=abs(native_amt),
                    current_price_native=execution_price_native,
                    fx_rate=fx_rate,
                    is_tax_locked=(eval_track == AssetEvaluationTrack.TRACK_3_LOCKED_VEHICLE),
                )

        item = HoldingAdvisoryItem(
            symbol=sym,
            asset_name=name,
            asset_type=asset_type,
            market=market,
            native_currency=native_curr,
            evaluation_track=eval_track,
            bucket_id=bid,
            bucket_name=bucket_name_map.get(str(bid)),
            current_units=current_units,
            current_price_native=current_price_native,
            current_value_thb=h_val_thb,
            current_weight_pct=current_weight_pct,
            target_weight_pct=target_weight_pct,
            weight_drift_pct=weight_drift_pct,
            action_code=action_code,
            action_rationale=rationale,
            lot_sizing=lot_sizing,
            is_immediate_execution=is_immediate,
            conditional_buy_zone_min=buy_zone_min,
            conditional_buy_zone_max=buy_zone_max,
            core_conviction_score=core_conv,
            business_conviction_score=biz_conv,
            price_stage=price_stage,
            live_upside_12m_pct=live_upside,
            macro_alignment=macro_align,
            sector_quadrant=quadrant,
            breached_falsifiers=breached_falsifiers,
            data_warnings=holding_warnings,
        )
        validated_signals_by_idx[len(holdings_advisory)] = sidecar_signals
        execution_market_by_idx[len(holdings_advisory)] = execution_market
        holdings_advisory.append(item)

    # Cash Overdraft & FX Pool Guard (I5) — includes estimated transaction friction
    immediate_cash_req_usd = immediate_deploy_usd + immediate_friction_usd
    immediate_cash_req_thb = immediate_deploy_thb + immediate_friction_thb
    fx_conversion_needed_thb = Decimal("0")
    if immediate_cash_req_usd > cash_usd:
        usd_deficit = immediate_cash_req_usd - cash_usd
        fx_conversion_needed_thb = safe_decimal(usd_deficit * fx_rate, quantum="0.01")
        warnings.append(DataAnomalyWarning(
            guardrail_code="CASH_OVERDRAFT_AND_FX_POOL_GUARD",
            ticker="CASH_USD",
            message=f"คำสั่งซื้อหุ้น US วันนี้รวมค่าธรรมเนียม (${immediate_cash_req_usd}) เกินเงินสดดอลลาร์ในพอร์ต (${cash_usd}). ต้องแลกเงินบาทเพิ่ม {fx_conversion_needed_thb:,.2f} THB",
            severity="warning",
            remedy_hint="Convert THB to USD in Dime/broker before executing US buy orders.",
        ))

    thb_surplus = cash_thb - immediate_cash_req_thb - fx_conversion_needed_thb
    if thb_surplus < Decimal("0"):
        warnings.append(DataAnomalyWarning(
            guardrail_code="CASH_OVERDRAFT_AND_FX_POOL_GUARD",
            ticker="CASH_THB",
            message=f"ยอดซื้อวันนี้รวมค่าธรรมเนียม ({immediate_cash_req_thb + fx_conversion_needed_thb:,.2f} THB) เกินเงินสดบาทที่มีอยู่ ({cash_thb:,.2f} THB)",
            severity="warning",
            remedy_hint="Reduce buy order size or sell overweight positions before executing.",
        ))
    usd_surplus = cash_usd - immediate_cash_req_usd if cash_usd >= immediate_cash_req_usd else Decimal("0")

    cash_plan = DualCurrencyCashPlan(
        cash_thb_available=cash_thb,
        cash_usd_available=cash_usd,
        usd_thb_fx_rate=fx_rate,
        immediate_deploy_thb=immediate_deploy_thb,
        immediate_deploy_usd=immediate_deploy_usd,
        conditional_limit_order_thb=conditional_limit_thb,
        conditional_limit_order_usd=conditional_limit_usd,
        thb_surplus_after_immediate=thb_surplus,
        usd_surplus_after_immediate=usd_surplus,
        fx_conversion_needed_thb=fx_conversion_needed_thb,
    )

    # Risk Radar Summary
    weighted_beta = (
        safe_decimal(weighted_beta_sum / beta_weight_sum, quantum="0.01")
        if beta_weight_sum > Decimal("0")
        else None
    )
    crowded_sectors = {
        k: safe_decimal(v / total_nav_thb * Decimal("100"), quantum="0.1")
        for k, v in sector_exposure_thb.items()
        if total_nav_thb > Decimal("0") and (v / total_nav_thb * Decimal("100")) > Decimal("30.0")
    }
    fx_split = {
        "THB": safe_decimal(thb_exposure / total_nav_thb * Decimal("100"), quantum="0.1") if total_nav_thb > Decimal("0") else Decimal("100"),
        "USD": safe_decimal(usd_exposure / total_nav_thb * Decimal("100"), quantum="0.1") if total_nav_thb > Decimal("0") else Decimal("0"),
    }

    # Concentration & Diversification Math (HHI & Effective Holdings Count)
    invested_holdings = [
        h for h in holdings_advisory
        if not h.asset_type.lower().startswith("cash") and h.current_value_thb > Decimal("0")
    ]
    invested_total_val = sum((h.current_value_thb for h in invested_holdings), Decimal("0"))

    hhi: Optional[Decimal] = None
    effective_holdings_count: Optional[Decimal] = None
    if invested_total_val > Decimal("0"):
        w_sq_sum = Decimal("0")
        for h in invested_holdings:
            w = h.current_value_thb / invested_total_val
            w_sq_sum += (w * w)
        hhi = safe_decimal(w_sq_sum, quantum="0.0001")
        if hhi and hhi > Decimal("0"):
            effective_holdings_count = safe_decimal(Decimal("1") / hhi, quantum="0.1")

    # Top-1, Top-3, Top-5 weights relative to total_nav_thb
    sorted_weights = sorted([h.current_weight_pct for h in invested_holdings], reverse=True)
    top_1_weight_pct = sorted_weights[0] if len(sorted_weights) > 0 else Decimal("0")
    top_3_weight_pct = sum(sorted_weights[:3], Decimal("0"))
    top_5_weight_pct = sum(sorted_weights[:5], Decimal("0"))

    # Concentration Risk Level Classification
    concentration_level: Literal["LOW", "MODERATE", "HIGH"] = "LOW"
    if hhi is not None:
        if (
            hhi >= Decimal("0.2500")
            or top_1_weight_pct >= Decimal("20.0")
            or (effective_holdings_count is not None and effective_holdings_count < Decimal("4.0"))
        ):
            concentration_level = "HIGH"
        elif (
            hhi >= Decimal("0.1500")
            or (effective_holdings_count is not None and effective_holdings_count < Decimal("8.0"))
            or top_3_weight_pct >= Decimal("50.0")
        ):
            concentration_level = "MODERATE"
        else:
            concentration_level = "LOW"

    # Bucket Active Share (%)
    bucket_active_share: Optional[Decimal] = None
    bucket_active_share_status: Literal["AVAILABLE", "UNAVAILABLE"] = "UNAVAILABLE"
    if total_nav_thb > Decimal("0") and bucket_target_map:
        bucket_active_share = compute_bucket_active_share_pct(active_share_actual_pct_map, bucket_target_map)
        bucket_active_share_status = "AVAILABLE"

    # Risk Contributions & Risk Parity Weights (via validated nested quant_signals)
    holdings_weights_map: dict[str, Decimal] = {}
    holdings_vols_map: dict[str, Decimal] = {}
    estimated_vol_syms: set[str] = set()
    for idx, h in enumerate(holdings_advisory):
        sym = h.symbol
        if sym and h.current_value_thb > Decimal("0") and not h.asset_type.lower().startswith("cash"):
            holdings_weights_map[sym] = holdings_weights_map.get(sym, Decimal("0")) + h.current_value_thb
            sig = validated_signals_by_idx.get(idx)
            vol_obs = _extract_positive_decimal(sig, "volatility_pct", quantum="0.0001")
            if vol_obs is not None:
                holdings_vols_map[sym] = vol_obs
                h.volatility_pct = vol_obs.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                h.volatility_source = "observed"
                h.is_volatility_estimated = False
            else:
                holdings_vols_map.setdefault(sym, Decimal("20.0"))
                estimated_vol_syms.add(sym)
                h.volatility_pct = None
                h.volatility_source = "estimated_default"
                h.is_volatility_estimated = True
                est_msg = "Observed volatility unavailable; using 20.0% estimated default for risk parity weighting."
                if est_msg not in h.data_warnings:
                    h.data_warnings.append(est_msg)

    risk_contrib_summary = compute_risk_contributions_and_parity(
        weights=holdings_weights_map,
        volatilities=holdings_vols_map,
    )

    # Attach risk contribution & estimated order friction to each HoldingAdvisoryItem
    for idx, h in enumerate(holdings_advisory):
        if (
            risk_contrib_summary.risk_contribution_status == "AVAILABLE"
            and h.symbol in risk_contrib_summary.risk_contributions_pct
        ):
            h.risk_contribution_pct = risk_contrib_summary.risk_contributions_pct[h.symbol]
        else:
            h.risk_contribution_pct = None

        if h.lot_sizing and h.lot_sizing.amount_native > Decimal("0"):
            is_buy = h.action_code in (
                AdvisoryActionCode.ACCUMULATE_NOW,
                AdvisoryActionCode.ACCUMULATE_ON_DIP,
                AdvisoryActionCode.BREAKOUT_BUY,
                AdvisoryActionCode.INITIATE_WATCHLIST,
            )
            side = "BUY" if is_buy else "SELL"
            sig = validated_signals_by_idx.get(idx)
            adtv = _extract_positive_decimal(sig, "adtv_local_currency", quantum="0.01")
            vol_obs = _extract_positive_decimal(sig, "volatility_pct", quantum="0.0001")
            exec_market = execution_market_by_idx.get(idx, "US" if h.market == "US" else "TH")

            f_est = estimate_order_friction(
                market=exec_market,
                action_side=side,
                order_amount_native=h.lot_sizing.amount_native,
                adtv_native=adtv,
                annualized_volatility_pct=vol_obs,
            )
            h.estimated_friction_bps = f_est.total_friction_bps
            if f_est.liquidity_warning and f_est.liquidity_warning not in h.data_warnings:
                h.data_warnings.append(f_est.liquidity_warning)

    total_port_vol = (
        risk_contrib_summary.total_portfolio_volatility_pct
        if risk_contrib_summary.risk_contribution_status != "NO_ACTIVE_ASSETS"
        else None
    )

    risk_radar = PortfolioRiskRadar(
        weighted_beta=weighted_beta,
        sector_crowding=crowded_sectors,
        fx_exposure_pct=fx_split,
        purpose_bucket_drift=purpose_bucket_drift,
        unassigned_holdings_value_thb=unassigned_val_thb,
        unassigned_holdings_nav_pct=unassigned_nav_pct,
        unassigned_drift_guard_active=unassigned_drift_active,
        hhi=hhi,
        effective_holdings_count=effective_holdings_count,
        top_1_weight_pct=top_1_weight_pct,
        top_3_weight_pct=top_3_weight_pct,
        top_5_weight_pct=top_5_weight_pct,
        concentration_risk_level=concentration_level,
        bucket_active_share_pct=bucket_active_share,
        bucket_active_share_status=bucket_active_share_status,
        risk_contributions_pct=risk_contrib_summary.risk_contributions_pct,
        inverse_vol_weights_pct=risk_contrib_summary.inverse_vol_weights_pct,
        total_portfolio_volatility_pct=total_port_vol,
        risk_contribution_status=risk_contrib_summary.risk_contribution_status,
        volatility_estimated_symbols=sorted(estimated_vol_syms),
        drawdown_pct=drawdown_posture.drawdown_pct if drawdown_posture.is_peak_available else None,
        drawdown_posture_tier=drawdown_posture.posture_tier.value,
        drawdown_status=drawdown_posture.status,
        drawdown_hurdle_adjustment=drawdown_posture.hurdle_adjustment,
        drawdown_action_restrictions=list(drawdown_posture.action_restrictions),
    )

    state_hash = compute_portfolio_state_hash(active_holdings, cash_thb, cash_usd)

    if look_through_summary:
        top_exposures = (
            getattr(look_through_summary, "holdings_exposure", None)
            or getattr(look_through_summary, "top_overlapped_stocks", None)
            or getattr(look_through_summary, "top_holdings", None)
            or (
                look_through_summary.get("holdings_exposure")
                or look_through_summary.get("top_overlapped_stocks")
                or look_through_summary.get("top_holdings", [])
                if isinstance(look_through_summary, dict)
                else []
            )
        )
        for exp in (top_exposures or []):
            is_breach, cap_warn = check_look_through_concentration_cap(exp)
            if is_breach and cap_warn:
                warnings.append(cap_warn)

    synthesis_input_hash = _compute_synthesis_input_hash(
        portfolio_state_hash=state_hash,
        data_mode=data_mode,
        macro_direction=macro_direction,
        sector_rotation_snapshot=sector_rotation_snapshot,
        risk_radar=risk_radar,
        holdings_advisory=holdings_advisory,
        quarantined_holdings=quarantined,
        anomaly_warnings=warnings,
        look_through_summary=look_through_summary,
    )

    return PortfolioAdvisoryQuantReport(
        evaluated_at=evaluated_at,
        data_mode=data_mode,
        portfolio_state_hash=state_hash,
        total_nav_thb=total_nav_thb,
        cash_plan=cash_plan,
        risk_radar=risk_radar,
        holdings_advisory=holdings_advisory,
        quarantined_holdings=quarantined,
        anomaly_warnings=warnings,
        macro_regime_summary=(
            macro_direction.model_dump() if hasattr(macro_direction, "model_dump") else (
                macro_direction if isinstance(macro_direction, dict) else None
            )
        ),
        cached_narrative=cached_narrative,
        look_through_summary=look_through_summary,
        synthesis_input_hash=synthesis_input_hash,
    )
