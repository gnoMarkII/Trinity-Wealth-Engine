"""Pure domain engine for Historical Thesis Reconstruction (Then vs Now).

Architecture Rule (Layer 1 - Core):
Pure Python standard library only — strictly zero external dependencies (no FastAPI, Pydantic,
SQLite, or yfinance) per Hexagonal Architecture AST Domain Layer Isolation rules.
"""
from typing import Optional
from core.equity_analytics.models import ThenVsNowSnapshot


class ThenVsNowReconstructionEngine:
    """Contrasts historical market expectations and valuations at entry versus today."""

    @staticmethod
    def evaluate(
        is_held: bool,
        entry_date: Optional[str],
        entry_price: Optional[float],
        current_price: float,
        then_pe: Optional[float] = None,
        now_pe: Optional[float] = None,
        then_growth_pct: Optional[float] = None,
        now_growth_pct: Optional[float] = None,
        what_was_true_then: Optional[str] = None,
        what_is_true_now: Optional[str] = None,
    ) -> ThenVsNowSnapshot:
        """Constructs an objective Then vs Now comparison to mitigate anchoring bias."""
        c_price = max(float(current_price), 0.01)
        t_price = max(float(entry_price), 0.01) if entry_price else c_price

        if is_held and entry_date:
            period_label = f"Around Entry ({entry_date})"
            then_date = entry_date
        else:
            period_label = "Baseline (1 Year Ago)"
            then_date = "1 Year Ago"

        now_date = "Today"

        # Determine narrative descriptions if not explicitly supplied
        if what_was_true_then:
            then_desc = what_was_true_then
        else:
            parts = [f"Trading at ${t_price:.2f}"]
            if then_pe is not None:
                parts.append(f"at {then_pe:.1f}x P/E")
            if then_growth_pct is not None:
                parts.append(f"with annual growth around {then_growth_pct:.1f}%")
            then_desc = " ".join(parts) + ". Market pricing reflected earlier cycle expectations."

        if what_is_true_now:
            now_desc = what_is_true_now
        else:
            parts = [f"Current price is ${c_price:.2f}"]
            if now_pe is not None:
                parts.append(f"at {now_pe:.1f}x P/E")
            if now_growth_pct is not None:
                parts.append(f"with expected growth of {now_growth_pct:.1f}%")
            now_desc = " ".join(parts) + ". Multiple and growth expectations have reset to current market reality."

        # Objective reflection question (Anti-Anchoring)
        if then_pe and now_pe and now_pe > then_pe * 1.25 and (now_growth_pct or 0) <= (then_growth_pct or 0):
            reflection = (
                f"Stocks bought around this level in {period_label} were generally bought for growth acceleration. "
                f"Today, the multiple expanded ({then_pe:.1f}x ➔ {now_pe:.1f}x) while growth moderated. "
                "Does your holding thesis rely on further multiple expansion, or on business re-acceleration?"
            )
        elif then_pe and now_pe and now_pe < then_pe * 0.8:
            reflection = (
                f"Valuation has compressed significantly from {then_pe:.1f}x to {now_pe:.1f}x since {period_label}. "
                "Does the current price offer an attractive margin of safety, or has the core business thesis permanently impaired?"
            )
        else:
            reflection = (
                f"Given what is true today at ${c_price:.2f}, is this position worth holding at its current size, "
                f"independent of what it cost in {period_label}?"
            )

        return ThenVsNowSnapshot(
            is_held=is_held,
            period_label=period_label,
            then_date=then_date,
            now_date=now_date,
            then_price=round(t_price, 2),
            now_price=round(c_price, 2),
            then_pe=round(then_pe, 2) if then_pe is not None else None,
            now_pe=round(now_pe, 2) if now_pe is not None else None,
            then_growth_pct=round(then_growth_pct, 2) if then_growth_pct is not None else None,
            now_growth_pct=round(now_growth_pct, 2) if now_growth_pct is not None else None,
            what_was_true_then=then_desc,
            what_is_true_now=now_desc,
            thesis_reflection_question=reflection,
        )
