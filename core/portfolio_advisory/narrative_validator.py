"""Post-LLM Narrative Consistency Validator (Guardrail I6).

Architecture Layer: core/portfolio_advisory/ (Pure Domain)
Enforces:
- Detects hallucinated tickers not present in the portfolio or valid universe.
- Detects directional contradictions (e.g. recommending to buy a stock flagged for URGENT_REDUCE).
"""
from __future__ import annotations

import re
from typing import Any, List, Set, Tuple

from core.news_digest.entity_matcher import (
    COMMON_STOPWORDS_AND_ACRONYMS,
    contains_ticker_boundary,
)
from core.portfolio_advisory.models import (
    AdvisoryActionCode,
    PortfolioAdvisoryNarrative,
    PortfolioAdvisoryQuantReport,
)

# Common financial acronyms and macro tickers allowed in narratives
ALLOWED_ACRONYMS: Set[str] = COMMON_STOPWORDS_AND_ACRONYMS

BUY_KEYWORDS = ("ซื้อ", "สะสม", "เพิ่มน้ำหนัก", "buy", "accumulate", "add")
SELL_KEYWORDS = ("ขาย", "ลดน้ำหนัก", "ตัดขาดทุน", "sell", "trim", "cut loss", "reduce")

_TICKER_CANDIDATE_RE = re.compile(r'(?<![A-Za-z0-9])[A-Z]{2,5}(?![A-Za-z0-9])')


def validate_advisory_narrative_consistency(
    narrative: Any,
    quant_report: PortfolioAdvisoryQuantReport,
) -> Tuple[bool, List[str]]:
    """Validates strategic narrative output against deterministic quant report."""
    errors: list[str] = []

    # 1. Collect all narrative text blocks
    text_blocks: list[str] = []
    if isinstance(narrative, dict):
        text_blocks.append(str(narrative.get("macro_executive_summary") or ""))
        text_blocks.extend(str(s) for s in narrative.get("immediate_action_playbook", []))
        text_blocks.extend(str(s) for s in narrative.get("conditional_dip_plan", []))
        text_blocks.extend(str(s) for s in narrative.get("risk_mitigation_and_hedging", []))
        text_blocks.extend(str(s) for s in narrative.get("whats_working", []))
        text_blocks.extend(str(s) for s in narrative.get("what_concerns_me", []))
        text_blocks.extend(str(s) for s in narrative.get("strategic_discussion_points", []))
    else:
        text_blocks.append(str(getattr(narrative, "macro_executive_summary", "") or ""))
        text_blocks.extend(str(s) for s in getattr(narrative, "immediate_action_playbook", []) or [])
        text_blocks.extend(str(s) for s in getattr(narrative, "conditional_dip_plan", []) or [])
        text_blocks.extend(str(s) for s in getattr(narrative, "risk_mitigation_and_hedging", []) or [])
        text_blocks.extend(str(s) for s in getattr(narrative, "whats_working", []) or [])
        text_blocks.extend(str(s) for s in getattr(narrative, "what_concerns_me", []) or [])
        text_blocks.extend(str(s) for s in getattr(narrative, "strategic_discussion_points", []) or [])

    full_narrative_text = " ".join(text_blocks)

    # 2. Build holding symbol and action map
    valid_symbols: dict[str, AdvisoryActionCode] = {}
    for h in quant_report.holdings_advisory:
        clean = h.symbol.replace(".BK", "").strip().upper()
        valid_symbols[clean] = h.action_code
        valid_symbols[h.symbol.strip().upper()] = h.action_code
        if getattr(h, "asset_name", None):
            for token in _TICKER_CANDIDATE_RE.findall(h.asset_name):
                valid_symbols[token] = h.action_code

    for q in quant_report.quarantined_holdings:
        clean = q.symbol.replace(".BK", "").strip().upper()
        valid_symbols[clean] = AdvisoryActionCode.QUARANTINED_DATA_ANOMALY
        valid_symbols[q.symbol.strip().upper()] = AdvisoryActionCode.QUARANTINED_DATA_ANOMALY

    if quant_report.macro_regime_summary:
        macro_str = str(quant_report.macro_regime_summary)
        for token in _TICKER_CANDIDATE_RE.findall(macro_str):
            valid_symbols.setdefault(token, AdvisoryActionCode.HOLD_WITHIN_BAND)

    lt_summary = getattr(quant_report, "look_through_summary", None)
    if lt_summary:
        raw_items = []
        for attr in ("top_overlapped_stocks", "holdings_exposure", "top_holdings"):
            val = getattr(lt_summary, attr, None) or (lt_summary.get(attr) if isinstance(lt_summary, dict) else None)
            if val and isinstance(val, (list, tuple)):
                raw_items.extend(val)

        for item in raw_items:
            raw_sym = (
                getattr(item, "ticker", None)
                or getattr(item, "stock_symbol", None)
                or getattr(item, "symbol", None)
                or (
                    item.get("ticker")
                    or item.get("stock_symbol")
                    or item.get("symbol")
                    if isinstance(item, dict)
                    else None
                )
            )
            if raw_sym:
                sym_clean = str(raw_sym).replace(".BK", "").strip().upper()
                valid_symbols.setdefault(sym_clean, AdvisoryActionCode.HOLD_WITHIN_BAND)
                valid_symbols.setdefault(str(raw_sym).strip().upper(), AdvisoryActionCode.HOLD_WITHIN_BAND)

    # 3. Directional Contradiction Check
    immediate_text = " ".join(
        narrative.get("immediate_action_playbook", [])
        if isinstance(narrative, dict)
        else getattr(narrative, "immediate_action_playbook", [])
    )

    # Split sentences by newline, bullet point, semicolon, or period with trailing space (Thai-friendly)
    sentence_split_regex = re.compile(r'[\n\r]+|[;•\t]+|\.\s+')
    immediate_sentences = [
        s.strip() for s in sentence_split_regex.split(immediate_text) if s.strip()
    ]

    for sym, action in valid_symbols.items():
        if contains_ticker_boundary(immediate_text, sym):
            # If stock is URGENT_REDUCE, TRIM_OVERWEIGHT, or QUARANTINED, verify it's not advised to buy
            if action in (
                AdvisoryActionCode.URGENT_REDUCE,
                AdvisoryActionCode.TRIM_OVERWEIGHT,
                AdvisoryActionCode.QUARANTINED_DATA_ANOMALY,
            ):
                sentences = [s for s in immediate_sentences if contains_ticker_boundary(s, sym)]
                for s in sentences:
                    s_lower = s.lower()
                    if any(bk in s_lower for bk in BUY_KEYWORDS) and not any(sk in s_lower for sk in SELL_KEYWORDS):
                        errors.append(
                            f"Directional Contradiction on {sym}: Action code is {action.value}, but narrative advises buying ('{s.strip()}')"
                        )

            # If stock is ACCUMULATE_NOW or BREAKOUT_BUY, verify it's not advised to panic sell
            elif action in (AdvisoryActionCode.ACCUMULATE_NOW, AdvisoryActionCode.BREAKOUT_BUY):
                sentences = [s for s in immediate_sentences if contains_ticker_boundary(s, sym)]
                for s in sentences:
                    s_lower = s.lower()
                    if any(sk in s_lower for sk in SELL_KEYWORDS) and not any(bk in s_lower for bk in BUY_KEYWORDS):
                        errors.append(
                            f"Directional Contradiction on {sym}: Action code is {action.value}, but narrative advises selling ('{s.strip()}')"
                        )

    # 4. Ticker Hallucination Check
    # Match standalone uppercase ticker candidates (2 to 5 uppercase characters)
    raw_candidates = set(_TICKER_CANDIDATE_RE.findall(full_narrative_text))
    for cand in raw_candidates:
        if cand in ALLOWED_ACRONYMS:
            continue
        if cand not in valid_symbols and f"{cand}.BK" not in valid_symbols:
            # Check if mentioned in macro regime or focus themes
            if quant_report.macro_regime_summary:
                macro_str = str(quant_report.macro_regime_summary)
                if cand in macro_str:
                    continue
            errors.append(f"Hallucinated Ticker Detected: '{cand}' is not an active portfolio holding or valid reference.")

    is_valid = len(errors) == 0
    return is_valid, errors
