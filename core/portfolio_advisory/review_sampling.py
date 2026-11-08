"""Core Domain: Risk-Based Review Sampling Policy.

Reference:
- Claude Code for Hedge Funds: Chapter 9 (pp. 65-66) & Appendix F (p. 108: Review Sampling Policy)
- Invariant: Eliminates Ceremonial Approval by bounding inspection size K in [3, 5]
  with tiered priority (100% Breaches -> Top Mover -> Deterministic Fair Rotation).

Architecture Rule (Layer 1 - Core Domain):
- Pure Python standard library only (no external frameworks, no sqlite, no api imports).
- Deterministic, zero side-effects.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Sequence


class SampleType(str, Enum):
    """Categorization of review sample items based on risk hierarchy."""
    BREACH = "BREACH"          # Critical thesis exception, stop-loss, or quarantine
    TOP_MOVER = "TOP_MOVER"    # Largest daily price or PnL displacement
    ROTATION = "ROTATION"      # Calendar-rotated holding for uniform monthly audit


@dataclass(frozen=True)
class ReviewSampleItem:
    """A bounded inspection item that the Portfolio Manager affirmatively verifies."""
    sample_id: str
    symbol: str
    sample_type: SampleType
    headline: str
    target_value: str
    urgency: str = "NORMAL"     # "CRITICAL", "HIGH", "NORMAL"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "symbol": self.symbol,
            "sample_type": self.sample_type.value,
            "headline": self.headline,
            "target_value": self.target_value,
            "urgency": self.urgency,
        }


def _extract_symbol(item: Any) -> str:
    """Deterministically extracts a normalized ticker symbol from an item or dict."""
    if isinstance(item, dict):
        sym = item.get("symbol") or item.get("ticker") or ""
    else:
        sym = getattr(item, "symbol", "") or getattr(item, "ticker", "")
    return str(sym).strip().upper()


def select_review_sample(
    holdings: Sequence[Any],
    thesis_exceptions: Sequence[Dict[str, Any]] = (),
    quarantined_holdings: Sequence[Dict[str, Any]] = (),
    pending_decisions: Sequence[Dict[str, Any]] = (),
    top_mover: Optional[Dict[str, Any]] = None,
    review_date: Optional[str] = None,
    min_sample_size: int = 3,
    max_sample_size: int = 5,
) -> List[ReviewSampleItem]:
    """Selects 3 to 5 high-priority review items using deterministic hierarchical sampling.

    Priority Hierarchy:
    1. 100% of Critical Breaches (quarantine, thesis exceptions, urgent reductions, up to 3).
    2. Extreme Mover (top absolute daily displacement, 1 item).
    3. Calendar Fair Rotation (day_of_month % N, guarantees monthly human review for every asset).
    4. Deterministic padding up to min_sample_size if holdings are available.
    """
    sampled: List[ReviewSampleItem] = []
    seen_symbols: set[str] = set()

    # --- Priority 1: Critical Breaches (up to 3 items) ---
    # 1.1 Quarantined Holdings
    for q in quarantined_holdings:
        sym = _extract_symbol(q)
        if not sym or sym in seen_symbols:
            continue
        reason = q.get("reason", "Asset quarantined by guardrail")
        code = q.get("guardrail_code", "QUARANTINE")
        sampled.append(
            ReviewSampleItem(
                sample_id=f"SMP-BREACH-{sym}",
                symbol=sym,
                sample_type=SampleType.BREACH,
                headline=f"Quarantine Breach ({code})",
                target_value=str(reason),
                urgency="CRITICAL",
            )
        )
        seen_symbols.add(sym)
        if len(sampled) >= 3:
            break

    # 1.2 Thesis Exceptions (if quota remains)
    if len(sampled) < 3:
        for exc in thesis_exceptions:
            sym = _extract_symbol(exc)
            if not sym or sym in seen_symbols:
                continue
            exc_type = exc.get("exception_type", "Thesis Falsifier Triggered")
            obs_val = exc.get("observed_value")
            exp_val = exc.get("expected_value")
            target_str = f"Observed {obs_val} (Exp: {exp_val})" if obs_val and exp_val else str(obs_val or exc_type)
            sampled.append(
                ReviewSampleItem(
                    sample_id=f"SMP-BREACH-{sym}",
                    symbol=sym,
                    sample_type=SampleType.BREACH,
                    headline=f"Thesis Exception ({exc_type})",
                    target_value=target_str,
                    urgency="CRITICAL",
                )
            )
            seen_symbols.add(sym)
            if len(sampled) >= 3:
                break

    # 1.3 High Urgency Decisions (if quota remains)
    if len(sampled) < 3:
        for dec in pending_decisions:
            sym = _extract_symbol(dec)
            urgency = str(dec.get("urgency", "")).upper()
            action_code = str(dec.get("action_code", "")).upper()
            if not sym or sym in seen_symbols:
                continue
            if urgency in ("HIGH", "CRITICAL") or action_code == "URGENT_REDUCE":
                action = dec.get("action", "Urgent PM action required")
                sampled.append(
                    ReviewSampleItem(
                        sample_id=f"SMP-BREACH-{sym}",
                        symbol=sym,
                        sample_type=SampleType.BREACH,
                        headline=f"Urgent Decision ({action_code or urgency})",
                        target_value=str(action),
                        urgency="HIGH",
                    )
                )
                seen_symbols.add(sym)
                if len(sampled) >= 3:
                    break

    # --- Priority 2: Extreme Daily Mover (1 item) ---
    mover_sym = _extract_symbol(top_mover) if top_mover else ""
    if mover_sym and mover_sym not in seen_symbols and len(sampled) < max_sample_size:
        change_pct = top_mover.get("change_pct", 0.0) if top_mover else 0.0
        change_str = f"{change_pct:+.2f}%" if isinstance(change_pct, (int, float)) else str(change_pct)
        sampled.append(
            ReviewSampleItem(
                sample_id=f"SMP-MOVER-{mover_sym}",
                symbol=mover_sym,
                sample_type=SampleType.TOP_MOVER,
                headline="Top Daily PnL Mover",
                target_value=f"Daily movement: {change_str}",
                urgency="NORMAL",
            )
        )
        seen_symbols.add(mover_sym)

    # --- Priority 3: Calendar Fair Rotation (1 item) ---
    all_holding_symbols: List[str] = []
    for h in holdings:
        s = _extract_symbol(h)
        if s and s not in all_holding_symbols:
            all_holding_symbols.append(s)

    # Deterministic calendar seed
    day_seed = 1
    if review_date:
        try:
            parts = review_date.split("-")
            day_seed = int(parts[-1])
        except (ValueError, IndexError):
            day_seed = 1

    uninspected_symbols = [s for s in all_holding_symbols if s not in seen_symbols]

    if uninspected_symbols and len(sampled) < max_sample_size:
        rot_idx = day_seed % len(uninspected_symbols)
        rot_sym = uninspected_symbols[rot_idx]
        sampled.append(
            ReviewSampleItem(
                sample_id=f"SMP-ROTATION-{rot_sym}",
                symbol=rot_sym,
                sample_type=SampleType.ROTATION,
                headline="Deterministic Calendar Rotation",
                target_value="Uniform monthly holding verification",
                urgency="NORMAL",
            )
        )
        seen_symbols.add(rot_sym)

    # --- Priority 4: Pad up to min_sample_size if holdings are available ---
    remaining_symbols = [s for s in all_holding_symbols if s not in seen_symbols]
    pad_idx = 0
    while len(sampled) < min_sample_size and pad_idx < len(remaining_symbols):
        pad_sym = remaining_symbols[pad_idx]
        sampled.append(
            ReviewSampleItem(
                sample_id=f"SMP-ROTATION-{pad_sym}",
                symbol=pad_sym,
                sample_type=SampleType.ROTATION,
                headline="Portfolio Baseline Verification",
                target_value="Periodic holding review",
                urgency="NORMAL",
            )
        )
        seen_symbols.add(pad_sym)
        pad_idx += 1

    # Final safety clamp: never exceed max_sample_size
    return sampled[:max_sample_size]

