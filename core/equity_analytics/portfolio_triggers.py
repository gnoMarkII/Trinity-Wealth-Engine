"""Pure domain engine for Portfolio-Aware Decision Triggers.

Architecture Rule (Layer 1 - Core):
Pure Python standard library only — strictly zero external dependencies (no FastAPI, Pydantic,
SQLite, or yfinance) per Hexagonal Architecture AST Domain Layer Isolation rules.
"""
from typing import Any, Dict, List, Optional
from core.equity_analytics.models import PortfolioAwareTrigger


class PortfolioTriggerResolver:
    """Synthesizes fundamental falsifiers, upcoming calendar events, and portfolio risk limits into checkable triggers."""

    @staticmethod
    def resolve_triggers(
        falsifiers: Optional[List[Dict[str, Any]]] = None,
        holding_context: Optional[Dict[str, Any]] = None,
        technical_stage: Optional[int] = None,
        upcoming_earnings_date: Optional[str] = None,
        current_pe: Optional[float] = None,
        pe_5y_high: Optional[float] = None,
    ) -> List[PortfolioAwareTrigger]:
        """Resolves active and pending decision triggers with case attribution."""
        triggers: List[PortfolioAwareTrigger] = []

        # 1. Upcoming Event Catalyst (e.g. Earnings date)
        if upcoming_earnings_date:
            triggers.append(
                PortfolioAwareTrigger(
                    trigger_id="EVT_UPCOMING_EARNINGS",
                    title=f"Upcoming Earnings Release ({upcoming_earnings_date})",
                    condition_text="Whether quarterly revenue and margin guidance meet consensus targets",
                    category="event_catalyst",
                    case_impact="SETTLES_BUY_VS_SELL",
                    impact_description="Directly settles whether current valuation is justified or whether guidance faces further cuts.",
                    is_currently_triggered=False,
                )
            )

        # 2. Fundamental Floors (from thesis_falsifiers)
        if falsifiers:
            for f in falsifiers[:3]:  # Top 3 fundamental falsifiers
                f_id = f.get("falsifier_id", "FALSIFIER")
                m_name = f.get("metric_name", "Fundamental Metric")
                thresh = f.get("threshold_value")
                cond = f.get("condition", "breaches threshold")
                expl = f.get("narrative_explanation", "Fundamental thesis invalidation point.")
                triggers.append(
                    PortfolioAwareTrigger(
                        trigger_id=f"FUND_{f_id}",
                        title=f"{m_name} Baseline Floor",
                        condition_text=f"{m_name} {cond} {thresh if thresh is not None else ''}".strip(),
                        category="fundamental_floor",
                        case_impact="BREAKS_BUY_CASE",
                        impact_description=f"Invalidates the core accumulation thesis: {expl}",
                        is_currently_triggered=False,
                        metric_threshold_value=float(thresh) if thresh is not None else None,
                    )
                )

        # 3. Valuation Ceiling
        if current_pe and pe_5y_high and current_pe >= pe_5y_high * 0.95:
            triggers.append(
                PortfolioAwareTrigger(
                    trigger_id="VAL_CEILING_STRETCHED",
                    title=f"Valuation Near 5-Year High ({pe_5y_high:.1f}x)",
                    condition_text=f"Current P/E ({current_pe:.1f}x) is within 5% of its 5-year peak",
                    category="valuation_ceiling",
                    case_impact="STRENGTHENS_SELL_CASE",
                    impact_description="Multiple expansion leaves zero room for forward earnings misses, skewing risk/reward to downside.",
                    is_currently_triggered=True,
                    metric_current_value=current_pe,
                    metric_threshold_value=pe_5y_high,
                )
            )

        # 4. Portfolio Concentration Ceiling
        if holding_context and holding_context.get("is_held"):
            weight_pct = float(holding_context.get("current_weight_pct") or 0.0)
            max_limit = float(holding_context.get("max_concentration_limit") or 20.0)
            is_triggered = weight_pct >= max_limit
            triggers.append(
                PortfolioAwareTrigger(
                    trigger_id="PORT_CONCENTRATION_LIMIT",
                    title=f"Portfolio Concentration Ceiling ({max_limit:.0f}%)",
                    condition_text=f"Position weight in portfolio reaches or exceeds {max_limit:.0f}% (current: {weight_pct:.1f}%)",
                    category="portfolio_concentration",
                    case_impact="FORCES_PORTFOLIO_TRIM",
                    impact_description="Independent of company fundamentals, disciplined portfolio risk limits dictate trimming to manage total wealth exposure.",
                    is_currently_triggered=is_triggered,
                    metric_current_value=weight_pct,
                    metric_threshold_value=max_limit,
                )
            )

            # 5. Anti-Falling-Knife Rule (Stage 4 Downtrend + Negative P&L)
            pnl_pct = float(holding_context.get("unrealized_pnl_pct") or 0.0)
            if pnl_pct < -10.0 and technical_stage == 4:
                triggers.append(
                    PortfolioAwareTrigger(
                        trigger_id="ANTI_AVERAGING_DOWN_RULE",
                        title="Anti-Falling-Knife Rule (Stage 4 Downtrend)",
                        condition_text=f"Position is down {pnl_pct:.1f}% while price action remains in Stage 4 distribution",
                        category="anti_averaging_down",
                        case_impact="BLOCKS_DIP_BUYING",
                        impact_description="Strictly blocks emotional averaging down into a deteriorating business until base formation is confirmed.",
                        is_currently_triggered=True,
                        metric_current_value=pnl_pct,
                        metric_threshold_value=-10.0,
                    )
                )

        return triggers
