"""LLM Payload Builder for Tier 2 Strategic Playbook Synthesis.

Architecture Layer: core/portfolio_advisory/ (Pure Domain)
Formats deterministic Tier 1 calculation results into a structured, sanitized payload
with strict source timestamps for the LLM Portfolio Synthesizer.
"""
from __future__ import annotations

from typing import Any, Dict

from core.portfolio_advisory.models import PortfolioAdvisoryQuantReport


def build_advisory_llm_payload(report: PortfolioAdvisoryQuantReport) -> Dict[str, Any]:
    """Builds token-efficient, sanitized JSON payload for the Tier 2 LLM synthesizer."""
    cash_plan = report.cash_plan
    radar = report.risk_radar

    holdings_summary = []
    for h in report.holdings_advisory:
        item = {
            "symbol": h.symbol,
            "market": h.market,
            "asset_type": h.asset_type,
            "evaluation_track": h.evaluation_track.value,
            "action_code": h.action_code.value,
            "action_rationale": h.action_rationale,
            "current_weight_pct": float(h.current_weight_pct),
            "target_weight_pct": float(h.target_weight_pct),
            "weight_drift_pct": float(h.weight_drift_pct),
            "is_immediate_execution": h.is_immediate_execution,
        }
        if h.core_conviction_score is not None:
            item["core_conviction_score"] = float(h.core_conviction_score)
        if h.business_conviction_score is not None:
            item["business_conviction_score"] = float(h.business_conviction_score)
        if h.live_upside_12m_pct is not None:
            item["live_upside_12m_pct"] = float(h.live_upside_12m_pct)
        if h.conditional_buy_zone_min is not None and h.conditional_buy_zone_max is not None:
            item["buy_zone"] = f"{h.conditional_buy_zone_min} - {h.conditional_buy_zone_max}"
        if h.lot_sizing:
            item["recommended_lots"] = {
                "board_lot_shares": h.lot_sizing.board_lot_shares,
                "whole_shares": h.lot_sizing.whole_shares,
                "fractional_shares": float(h.lot_sizing.fractional_shares) if h.lot_sizing.fractional_shares else None,
                "amount_native": float(h.lot_sizing.amount_native),
                "native_currency": h.lot_sizing.native_currency,
                "amount_thb": float(h.lot_sizing.amount_thb),
            }
        if h.breached_falsifiers:
            item["breached_falsifiers"] = h.breached_falsifiers
        holdings_summary.append(item)

    quarantined_summary = [
        {"symbol": q.symbol, "reason": q.reason, "guardrail_code": q.guardrail_code}
        for q in report.quarantined_holdings
    ]

    warnings_summary = [
        {"guardrail_code": w.guardrail_code, "ticker": w.ticker, "message": w.message, "severity": w.severity}
        for w in report.anomaly_warnings
    ]

    payload: dict[str, Any] = {
        "evaluated_at": report.evaluated_at,
        "data_mode": report.data_mode.value,
        "portfolio_state_hash": report.portfolio_state_hash,
        "total_nav_thb": float(report.total_nav_thb),
        "cash_ledger": {
            "cash_thb_available": float(cash_plan.cash_thb_available),
            "cash_usd_available": float(cash_plan.cash_usd_available),
            "usd_thb_fx_rate": float(cash_plan.usd_thb_fx_rate),
            "immediate_deploy_thb": float(cash_plan.immediate_deploy_thb),
            "immediate_deploy_usd": float(cash_plan.immediate_deploy_usd),
            "conditional_limit_order_thb": float(cash_plan.conditional_limit_order_thb),
            "conditional_limit_order_usd": float(cash_plan.conditional_limit_order_usd),
            "fx_conversion_needed_thb": float(cash_plan.fx_conversion_needed_thb),
        },
        "risk_radar": {
            "weighted_beta": float(radar.weighted_beta) if radar.weighted_beta is not None else None,
            "sector_crowding": {k: float(v) for k, v in radar.sector_crowding.items()},
            "fx_exposure_pct": {k: float(v) for k, v in radar.fx_exposure_pct.items()},
            "unassigned_holdings_nav_pct": float(radar.unassigned_holdings_nav_pct),
            "unassigned_drift_guard_active": radar.unassigned_drift_guard_active,
            "hhi": float(radar.hhi) if radar.hhi is not None else None,
            "effective_holdings_count": float(radar.effective_holdings_count) if radar.effective_holdings_count is not None else None,
            "top_1_weight_pct": float(radar.top_1_weight_pct) if getattr(radar, "top_1_weight_pct", None) is not None else 0.0,
            "top_3_weight_pct": float(radar.top_3_weight_pct) if getattr(radar, "top_3_weight_pct", None) is not None else 0.0,
            "top_5_weight_pct": float(radar.top_5_weight_pct) if getattr(radar, "top_5_weight_pct", None) is not None else 0.0,
            "concentration_risk_level": getattr(radar, "concentration_risk_level", "LOW") or "LOW",
        },
        "holdings_advisory": holdings_summary,
        "quarantined_holdings": quarantined_summary,
        "anomaly_warnings": warnings_summary,
    }

    if report.macro_regime_summary:
        macro = report.macro_regime_summary
        payload["macro_regime"] = {
            "overall_regime": macro.get("overall_regime") or macro.get("regime_name"),
            "evaluated_at": macro.get("evaluated_at"),
            "leading_sectors": (macro.get("sector_analysis") or {}).get("leading_sectors", []),
            "lagging_sectors": (macro.get("sector_analysis") or {}).get("lagging_sectors", []),
            "asset_allocation": macro.get("asset_allocation", []),
        }

    if hasattr(report, "look_through_summary") and getattr(report, "look_through_summary", None):
        lt = getattr(report, "look_through_summary")
        payload["look_through_overlap_summary"] = {
            "top_overlapped_stocks": [
                {
                    "ticker": getattr(s, "ticker", ""),
                    "company_name": getattr(s, "company_name", ""),
                    "direct_weight_pct": float(getattr(s, "direct_weight_pct", 0)),
                    "indirect_weight_pct": float(getattr(s, "indirect_weight_pct", 0)),
                    "total_weight_pct": float(getattr(s, "total_weight_pct", 0)),
                    "is_cap_breached": getattr(s, "is_cap_breached", False),
                    "is_hidden_concentration": getattr(s, "is_hidden_concentration", False),
                    "source_funds": [f"{f[0]} ({float(f[1]):,.0f} THB)" for f in getattr(s, "source_funds", [])],
                }
                for s in (getattr(lt, "top_overlapped_stocks", []) or [])[:10]
            ],
            "hidden_concentration_warnings": getattr(lt, "hidden_concentration_warnings", []) or [],
            "true_sectors": [
                {
                    "sector_name": getattr(sec, "sector_name", ""),
                    "weight_pct": float(getattr(sec, "weight_pct", 0)),
                    "is_crowded": getattr(sec, "is_crowded", False),
                }
                for sec in (getattr(lt, "sector_exposure", []) or [])
            ],
        }

    return payload
