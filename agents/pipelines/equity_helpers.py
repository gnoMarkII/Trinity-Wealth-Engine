"""Equity intel pipeline output construction and fallback helpers.

Architecture Layer: agents/pipelines/
Extracted from manager_agent.py to encapsulate MicroQuantOutput provisional sidecars,
narrative fallbacks, and synthesizer failure recovery.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

from schemas.micro_quant_schemas import (
    EquitySentimentContext,
    MicroQuantOutput,
    PortfolioHoldingContext,
    QuantSignals,
    ThesisDeltaSummary,
)

log = logging.getLogger(__name__)


def compute_thesis_delta_from_vault(
    ticker: str,
    current_signals: QuantSignals,
    current_date: Optional[str] = None,
    vault_path: Optional[Path] = None,
) -> Optional[ThesisDeltaSummary]:
    """Computes Thesis Evolution & Drift compared to previous Equity Sidecar in Vault."""
    try:
        base = vault_path or Path(os.getenv("OBSIDIAN_VAULT_PATH", "memories"))
        sanitized_ticker = ticker.replace(".BK", "").strip().upper()
        raw_ticker = ticker.strip().upper()
        candidate_paths = [
            base / "30_Knowledge_Base" / "Stocks" / sanitized_ticker / "Analysis" / f"{sanitized_ticker} Equity Analysis latest.json",
            base / "30_Knowledge_Base" / "Stocks" / raw_ticker / "Analysis" / f"{raw_ticker} Equity Analysis latest.json",
            base / "30_Knowledge_Base" / "Stocks" / sanitized_ticker / f"{sanitized_ticker} Equity Analysis latest.json",
            base / "30_Knowledge_Base" / "Stocks" / raw_ticker / f"{raw_ticker} Equity Analysis latest.json",
            base / "30_Knowledge_Base" / "Equities" / sanitized_ticker / "latest.json",
            base / "30_Knowledge_Base" / "Equities" / raw_ticker / "latest.json",
        ]
        sidecar_path = next((p for p in candidate_paths if p.exists()), None)
        if not sidecar_path:
            return None

        prev_data = json.loads(sidecar_path.read_text(encoding="utf-8"))
        prev_quant = prev_data.get("quant_signals") or {}
        prev_date = prev_data.get("analysis_date", "")

        prev_price = prev_quant.get("raw_analysis_price") or prev_quant.get("current_price")
        curr_price = getattr(current_signals, "raw_analysis_price", None) or getattr(current_signals, "current_price", None)
        price_change_pct = (
            round(((curr_price - prev_price) / prev_price * 100.0), 2)
            if (curr_price and prev_price and prev_price > 0)
            else None
        )

        prev_sc = prev_quant.get("deterministic_scorecard") or {}
        curr_sc = current_signals.deterministic_scorecard
        prev_conv = prev_sc.get("core_conviction_score")
        curr_conv = curr_sc.core_conviction_score if curr_sc else None
        conv_delta = (
            round(curr_conv - prev_conv, 1)
            if (curr_conv is not None and prev_conv is not None)
            else None
        )

        prev_stance = prev_sc.get("action_stance")
        curr_stance = curr_sc.action_stance if curr_sc else None

        prev_rdcf = prev_quant.get("reverse_dcf_result") or {}
        prev_target = prev_rdcf.get("target_price_12m") or prev_rdcf.get("alternative_target_price_12m")
        curr_target = (
            (current_signals.reverse_dcf_result.target_price_12m or current_signals.reverse_dcf_result.alternative_target_price_12m)
            if current_signals.reverse_dcf_result
            else None
        )
        target_delta = (
            round(((curr_target - prev_target) / prev_target * 100.0), 2)
            if (curr_target and prev_target and prev_target > 0)
            else None
        )

        prev_f = (prev_quant.get("piotroski_breakdown") or {}).get("f_score")
        curr_f = (
            current_signals.piotroski_breakdown.f_score
            if current_signals.piotroski_breakdown
            else None
        )
        f_delta = (curr_f - prev_f) if (curr_f is not None and prev_f is not None) else None

        breached: list[str] = []
        prev_falsifiers = prev_data.get("thesis_falsifiers") or prev_quant.get("thesis_falsifiers") or []
        for pf in prev_falsifiers:
            m_name = pf.get("metric_name", "")
            thresh = pf.get("threshold_value")
            if thresh is not None and "Margin" in m_name:
                curr_margin = (
                    current_signals.gaap_operating_margin.value_pct
                    if current_signals.gaap_operating_margin
                    else None
                )
                if curr_margin is not None and curr_margin < thresh:
                    breached.append(f"{m_name} ({curr_margin}%) breached floor ({thresh}%)")
            elif thresh is not None and ("Support" in m_name or "STOP_LOSS" in pf.get("falsifier_id", "")):
                if curr_price is not None and curr_price < thresh:
                    breached.append(f"{m_name} ({curr_price}) breached stop level ({thresh})")

        return ThesisDeltaSummary(
            previous_analysis_date=prev_date or "previous",
            price_change_pct=price_change_pct,
            conviction_score_delta=conv_delta,
            previous_action_stance=prev_stance,
            current_action_stance=curr_stance,
            target_price_change_pct=target_delta,
            f_score_delta=f_delta,
            breached_falsifiers=breached,
        )
    except Exception as e:
        log.warning("compute_thesis_delta_from_vault failed: %s", e)
        return None


def load_portfolio_holding_context(
    ticker: str,
    market: str,
    current_signals: QuantSignals,
) -> Optional[PortfolioHoldingContext]:
    """Retrieves portfolio position state for the analyzed ticker (fail-open)."""
    try:
        from tools.portfolio.core import get_structured_portfolio_state

        state = get_structured_portfolio_state(refresh_prices=False)
        stance = (
            current_signals.deterministic_scorecard.action_stance
            if current_signals.deterministic_scorecard
            else "HOLD_WAIT"
        )
        default_hint = (
            "INITIATE_NEW"
            if stance in ("ACCUMULATE_NOW", "ACCUMULATE_ON_DIP", "BREAKOUT_BUY")
            else "WAIT_DO_NOT_ENTER"
        )

        if not state or not state.holdings:
            return PortfolioHoldingContext(
                is_held=False,
                portfolio_id="default",
                units=0.0,
                avg_cost=None,
                unrealized_pnl_pct=None,
                current_weight_pct=0.0,
                bucket_id=None,
                position_action_hint=default_hint,
            )

        sanitized_ticker = ticker.replace(".BK", "").strip().upper()
        holding = None
        for h in state.holdings:
            sym = h.symbol.replace(".BK", "").strip().upper()
            if sym == sanitized_ticker or h.symbol.strip().upper() == ticker.strip().upper():
                holding = h
                break

        if not holding or holding.units <= 0:
            return PortfolioHoldingContext(
                is_held=False,
                portfolio_id="default",
                units=0.0,
                avg_cost=None,
                unrealized_pnl_pct=None,
                current_weight_pct=0.0,
                bucket_id=None,
                position_action_hint=default_hint,
            )

        total_val = state.summary.total_value_thb if state.summary else 0.0
        weight_pct = (
            round((holding.market_value_thb / total_val * 100.0), 2) if total_val > 0 else 0.0
        )

        avg_cost = holding.avg_cost_usd if market == "US" else (holding.avg_cost_thb or holding.avg_cost_usd)
        pnl_pct = holding.unrealized_pnl_percent

        if weight_pct >= 15.0:
            if stance == "REDUCE" or weight_pct >= 20.0:
                hint = "TRIM_OVERWEIGHT"
            else:
                hint = "HOLD_FULL_SIZED"
        elif pnl_pct is not None and pnl_pct > 0:
            if stance in ("ACCUMULATE_NOW", "ACCUMULATE_ON_DIP", "BREAKOUT_BUY"):
                hint = "ADD_TO_WINNER"
            else:
                hint = "HOLD_FULL_SIZED"
        elif pnl_pct is not None and pnl_pct < 0:
            if stance in ("ACCUMULATE_NOW", "ACCUMULATE_ON_DIP", "BREAKOUT_BUY"):
                hint = "AVERAGING_DOWN_CAUTION"
            else:
                hint = "WAIT_DO_NOT_ENTER"
        else:
            hint = "HOLD_FULL_SIZED" if stance not in ("REDUCE", "INSUFFICIENT_DATA") else "WAIT_DO_NOT_ENTER"

        return PortfolioHoldingContext(
            is_held=True,
            portfolio_id="default",
            units=float(holding.units),
            avg_cost=avg_cost,
            unrealized_pnl_pct=pnl_pct,
            current_weight_pct=weight_pct,
            bucket_id=holding.bucket_id,
            position_action_hint=hint,
        )
    except Exception as e:
        log.warning("load_portfolio_holding_context failed: %s", e)
        return None


def build_provisional_equity_output(
    validated_signals: QuantSignals,
    *,
    analysis_date: str | None = None,
    thesis_delta: Optional[ThesisDeltaSummary] = None,
    portfolio_context: Optional[PortfolioHoldingContext] = None,
) -> MicroQuantOutput:
    """Builds a provisional MicroQuantOutput before Narrative LLM generation."""
    from schemas.micro_quant_schemas import build_unavailable_sentiment_context

    return MicroQuantOutput(
        ticker=validated_signals.ticker,
        market=validated_signals.market,
        analysis_date=analysis_date or datetime.now().strftime("%Y-%m-%d"),
        quant_signals=validated_signals,
        sentiment_context=build_unavailable_sentiment_context(),
        narrative_analysis="รอการประมวลผล Narrative จาก LLM",
        base_case_summary="ตัวเลขรีเฟรชแล้ว กำลังสร้าง narrative",
        narrative_status="pending",
        piotroski_breakdown=validated_signals.piotroski_breakdown,
        reverse_dcf_result=validated_signals.reverse_dcf_result,
        tactical_setup=validated_signals.tactical_setup,
        insider_conviction=validated_signals.insider_conviction,
        earnings_guidance_context=validated_signals.earnings_guidance_context,
        deterministic_scorecard=validated_signals.deterministic_scorecard,
        thesis_falsifiers=validated_signals.thesis_falsifiers,
        evidence_snapshot=validated_signals.evidence_snapshot,
        thesis_delta=thesis_delta,
        portfolio_context=portfolio_context,
    )


def build_equity_narrative_fallback_dict(error: Exception | str) -> dict[str, Any]:
    """Builds a fallback dictionary for EquitySentimentContext when narrative parsing fails."""
    return {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "market_sentiment": "neutral",
        "key_themes": [],
        "tail_risks": [],
        "sources_summary": f"Data fetch/parse failed: {str(error)}",
    }


def build_synthesized_equity_output(
    quant_data: dict[str, Any],
    narrative_data: dict[str, Any],
    narrative_result: Any,
    *,
    analysis_date: str | None = None,
    thesis_delta: Optional[ThesisDeltaSummary] = None,
    portfolio_context: Optional[PortfolioHoldingContext] = None,
) -> MicroQuantOutput:
    """Builds a complete MicroQuantOutput from successful synthesis."""
    signals = QuantSignals.model_validate(quant_data)
    return MicroQuantOutput(
        ticker=quant_data["ticker"],
        market=quant_data["market"],
        analysis_date=analysis_date or datetime.now().strftime("%Y-%m-%d"),
        quant_signals=signals,
        sentiment_context=EquitySentimentContext.model_validate(narrative_data),
        narrative_analysis=getattr(narrative_result, "narrative_analysis", ""),
        base_case_summary=getattr(narrative_result, "base_case_summary", ""),
        piotroski_breakdown=signals.piotroski_breakdown,
        reverse_dcf_result=signals.reverse_dcf_result,
        tactical_setup=signals.tactical_setup,
        insider_conviction=signals.insider_conviction,
        earnings_guidance_context=signals.earnings_guidance_context,
        deterministic_scorecard=signals.deterministic_scorecard,
        thesis_falsifiers=signals.thesis_falsifiers,
        evidence_snapshot=signals.evidence_snapshot,
        thesis_delta=thesis_delta,
        portfolio_context=portfolio_context,
    )


def build_fallback_equity_output(
    quant_data: dict[str, Any],
    narrative_data: dict[str, Any],
    error: Exception | str,
    *,
    analysis_date: str | None = None,
    thesis_delta: Optional[ThesisDeltaSummary] = None,
    portfolio_context: Optional[PortfolioHoldingContext] = None,
) -> Optional[MicroQuantOutput]:
    """Preserves quantitative numbers when LLM synthesis fails, returning LLM_UNAVAILABLE output."""
    if not quant_data or "ticker" not in quant_data:
        return None

    signals = QuantSignals.model_validate(quant_data)
    fallback_sentiment = EquitySentimentContext(
        evaluated_at=datetime.now(timezone.utc).isoformat(),
        market_sentiment="neutral",
        key_themes=[],
        tail_risks=[],
        sources_summary=f"LLM synthesis pending/unavailable: {str(error)}",
    )
    return MicroQuantOutput(
        ticker=quant_data["ticker"],
        market=quant_data.get("market", "US"),
        analysis_date=analysis_date or datetime.now().strftime("%Y-%m-%d"),
        quant_signals=signals,
        sentiment_context=(
            EquitySentimentContext.model_validate(narrative_data)
            if narrative_data
            else fallback_sentiment
        ),
        narrative_analysis=f"ตัวเลขรีเฟรชแล้ว แต่ narrative ยังสร้างไม่ได้ (LLM_UNAVAILABLE: {str(error)})",
        base_case_summary="ตัวเลขรีเฟรชแล้ว แต่ narrative ยังสร้างไม่ได้",
        piotroski_breakdown=signals.piotroski_breakdown,
        reverse_dcf_result=signals.reverse_dcf_result,
        tactical_setup=signals.tactical_setup,
        insider_conviction=signals.insider_conviction,
        earnings_guidance_context=signals.earnings_guidance_context,
        deterministic_scorecard=signals.deterministic_scorecard,
        thesis_falsifiers=signals.thesis_falsifiers,
        evidence_snapshot=signals.evidence_snapshot,
        thesis_delta=thesis_delta,
        portfolio_context=portfolio_context,
    )
