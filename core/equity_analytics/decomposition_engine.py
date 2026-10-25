"""Pure domain engine for Price Move Decomposition.

Architecture Rule (Layer 1 - Core):
Pure Python standard library only — strictly zero external dependencies (no FastAPI, Pydantic,
SQLite, or yfinance) per Hexagonal Architecture AST Domain Layer Isolation rules.
"""
from typing import Optional
from core.equity_analytics.models import PriceMoveDecomposition


class PriceMoveDecompositionEngine:
    """Calculates deterministic attribution of stock price returns between EPS growth and Multiple Expansion."""

    @staticmethod
    def evaluate(
        horizon_label: str,
        start_date: str,
        end_date: str,
        start_price: float,
        end_price: float,
        start_eps: float,
        end_eps: float,
    ) -> PriceMoveDecomposition:
        """Decomposes price return into EPS growth and P/E multiple expansion components.

        Mathematical identity:
            (1 + TotalReturn) = (1 + EPSGrowth) * (1 + MultipleExpansion)
        """
        # Guard against invalid price values
        s_price = max(float(start_price), 0.0001)
        e_price = max(float(end_price), 0.0001)
        total_return_pct = round(((e_price - s_price) / s_price) * 100.0, 2)

        # Non-positive EPS boundary condition (loss-making company or turnaround)
        if start_eps <= 0 or end_eps <= 0:
            eps_growth_pct = round(((end_eps - start_eps) / max(abs(start_eps), 0.0001)) * 100.0, 2)
            tier = "SPECULATIVE_EXPANSION" if total_return_pct >= 0 else "TWIN_DETERIORATION"
            summary = (
                f"Negative/Zero EPS in period (start: {start_eps:.2f}, end: {end_eps:.2f}). "
                "Multiple decomposition not meaningful; driven by turn-around speculation or cash burn."
            )
            return PriceMoveDecomposition(
                horizon_label=horizon_label,
                start_date=start_date,
                end_date=end_date,
                start_price=round(s_price, 2),
                end_price=round(e_price, 2),
                total_return_pct=total_return_pct,
                start_eps=round(start_eps, 2),
                end_eps=round(end_eps, 2),
                eps_growth_pct=eps_growth_pct,
                start_pe=0.0,
                end_pe=0.0,
                pe_expansion_pct=0.0,
                primary_driver="earnings_growth" if abs(eps_growth_pct) > 0 else "multiple_expansion",
                diagnosis_tier=tier,
                diagnosis_summary=summary,
            )

        start_pe = round(s_price / start_eps, 2)
        end_pe = round(e_price / end_eps, 2)
        eps_growth_pct = round(((end_eps - start_eps) / start_eps) * 100.0, 2)
        pe_expansion_pct = round(((end_pe - start_pe) / start_pe) * 100.0, 2)

        # Classification & diagnosis
        if total_return_pct >= 0:
            if eps_growth_pct > 0 and pe_expansion_pct <= 0:
                tier = "VALUATION_COMPRESSION_CUSHION"
                driver = "earnings_growth"
                summary = (
                    f"Earnings grew strongly (+{eps_growth_pct}%) while P/E contracted by {abs(pe_expansion_pct)}%. "
                    "Gains are fundamentally backed with strong valuation margin of safety."
                )
            elif eps_growth_pct >= pe_expansion_pct:
                tier = "FUNDAMENTAL_DRIVEN"
                driver = "earnings_growth"
                summary = (
                    f"Price gain (+{total_return_pct}%) was primarily supported by solid earnings growth (+{eps_growth_pct}%), "
                    f"outpacing multiple re-rating (+{pe_expansion_pct}%)."
                )
            else:
                tier = "SPECULATIVE_EXPANSION"
                driver = "multiple_expansion"
                summary = (
                    f"Multiple expansion (+{pe_expansion_pct}%) dominated earnings growth (+{eps_growth_pct}%). "
                    "Fragile re-rating: price is vulnerable to multiple contraction if upcoming guidance misses."
                )
        else:
            if eps_growth_pct < 0 and pe_expansion_pct < 0:
                tier = "TWIN_DETERIORATION"
                driver = "twin_headwinds"
                summary = (
                    f"Both earnings contraction ({eps_growth_pct}%) and valuation de-rating ({pe_expansion_pct}%) "
                    "compounded the stock decline."
                )
            elif pe_expansion_pct < 0:
                tier = "VALUATION_COMPRESSION_CUSHION"
                driver = "multiple_expansion"
                summary = (
                    f"Decline was driven by market multiple contraction ({pe_expansion_pct}%) despite earnings holding "
                    f"({eps_growth_pct}%). Potential value stabilization candidate."
                )
            else:
                tier = "TWIN_DETERIORATION"
                driver = "earnings_growth"
                summary = (
                    f"Fundamental deterioration in earnings ({eps_growth_pct}%) caused the decline, "
                    f"even while P/E multiple adjusted by {pe_expansion_pct}%."
                )

        return PriceMoveDecomposition(
            horizon_label=horizon_label,
            start_date=start_date,
            end_date=end_date,
            start_price=round(s_price, 2),
            end_price=round(e_price, 2),
            total_return_pct=total_return_pct,
            start_eps=round(start_eps, 2),
            end_eps=round(end_eps, 2),
            eps_growth_pct=eps_growth_pct,
            start_pe=start_pe,
            end_pe=end_pe,
            pe_expansion_pct=pe_expansion_pct,
            primary_driver=driver,
            diagnosis_tier=tier,
            diagnosis_summary=summary,
        )
