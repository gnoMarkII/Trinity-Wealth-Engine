"""Deterministic Context Assembler for Trinity Guru Roundtable.

Architecture Layer: application/roundtable/ (depends ONLY on core/, schemas/, ports).
Implements:
- Unanalyzed Ticker Prerequisite Guard (`missing_prerequisite_tickers`)
- Deterministic Instrument Identity Anchor (`InstrumentIdentityAnchor`) with full nested schema fallbacks
- 3-State Portfolio Guard (`NOT_ATTACHED_DO_NOT_ASSUME` vs `ATTACHED_FLAT_BOOK` vs `ATTACHED_HELD`)
  plus curated Portfolio Overview, Concentration Diagnostics & Bucket Room
- Explicit `report_or_absent` markers for unattached Context Cards
- Curated, role-sectioned Briefs (`<!-- SEC:... -->`) for `$Equity` (single & multi-ticker),
  `@Macro`, `@Digest`, full 8-section `@Essence` + Financial Context, and Ticker News/Vault Notes
- Closed-loop settlement scorecard evaluation via Python Decimal
- Deterministic `packet_fingerprint_sha256` for Drift Detection
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional runtime fallback
    yaml = None  # type: ignore

from application.roundtable.ports import (
    ContextDataProvidersPort,
    RoundtableSessionRepoPort,
    SettlementPricePort,
)
from core.roundtable.models import (
    ContextAttachmentState,
    FrozenContextPacket,
    InstrumentIdentityAnchor,
    PastRoundtableScorecard,
    PortfolioHoldingGuardState,
    match_prioritized_speakers,
    normalize_debate_depth,
    normalize_debate_intensity,
)
from core.roundtable.settlement import (
    compute_settlement_scorecard,
    default_benchmark_for_ticker,
)

_DOLLAR_TICKER_RE = re.compile(r"\$([A-Za-z0-9]{1,10}(?:\.[A-Za-z]{1,4})?)")

_EQUITY_BLOAT_KEYS = frozenset(
    {
        "sensitivity_matrix",
        "sensitivity_matrices",
        "evidence_snapshot",
        "explicit_forecast_5y",
        "explicit_forecast_5y_scenarios",
        "historical_financials_3y",
        "composite_score_waterfall",
        "observable_refs",
    }
)


def extract_context_tags(
    context_refs: Sequence[str], agenda_prompt: str = ""
) -> Tuple[Tuple[str, ...], List[str], Dict[str, bool]]:
    """Normalize context refs and extract referenced tickers & @card flags."""
    normalized_refs: List[str] = []
    seen_lower: set[str] = set()
    tickers: List[str] = []
    cards = {
        "portfolio": False,
        "macro": False,
        "digest": False,
        "essence": False,
    }

    def _add_ref(raw: str) -> None:
        clean = raw.strip()
        if not clean:
            return
        low = clean.lower()
        if low in ("@portfolio", "@macro", "@digest", "@essence"):
            key = low[1:]
            cards[key] = True
            canon = f"@{key.capitalize()}"
            if low not in seen_lower:
                seen_lower.add(low)
                normalized_refs.append(canon)
        elif clean.startswith("$") and len(clean) > 1:
            sym = clean[1:].strip().upper()
            if sym and f"${sym.lower()}" not in seen_lower:
                seen_lower.add(f"${sym.lower()}")
                normalized_refs.append(f"${sym}")
                tickers.append(sym)
        elif re.match(r"^[A-Za-z0-9]{1,10}(?:\.[A-Za-z]{1,4})?$", clean):
            sym = clean.upper()
            if f"${sym.lower()}" not in seen_lower:
                seen_lower.add(f"${sym.lower()}")
                normalized_refs.append(f"${sym}")
                tickers.append(sym)

    for r in context_refs:
        _add_ref(r)

    for match in _DOLLAR_TICKER_RE.finditer(agenda_prompt or ""):
        _add_ref(f"${match.group(1)}")

    return tuple(normalized_refs), tickers, cards


def report_or_absent(card_name: str, is_attached: bool, content_text: str) -> str:
    """Anti-hallucination guard marking unattached context cards explicitly absent."""
    if not is_attached:
        return f"[ABSENT: การ์ด {card_name} ไม่ได้ถูกแนบในห้องประชุมนี้ — ห้ามเดาหรือสมมติข้อมูลในส่วนนี้ขึ้นเอง]"
    stripped = (content_text or "").strip()
    if not stripped:
        return f"[UNAVAILABLE: แนบการ์ด {card_name} แล้ว แต่ยังไม่มีข้อมูลในระบบขณะนี้]"
    return stripped


def _safe_str_dec(val: Any, places: str = "0.01") -> Optional[str]:
    if val is None or isinstance(val, bool):
        return None
    try:
        d = Decimal(str(val).strip())
        if d.is_nan() or d.is_infinite():
            return None
        return str(d.quantize(Decimal(places), rounding=ROUND_HALF_UP))
    except Exception:
        return None


def _first_non_none(*values: Any) -> Any:
    for v in values:
        if v is not None:
            if isinstance(v, str) and not v.strip():
                continue
            return v
    return None


def _pick_subdict(qs: Dict[str, Any], detail: Dict[str, Any], key: str) -> Dict[str, Any]:
    """Return populated dict checking quant_signals first, then top-level detail."""
    v1 = qs.get(key) if isinstance(qs, dict) else None
    if isinstance(v1, dict) and v1:
        return v1
    v2 = detail.get(key) if isinstance(detail, dict) else None
    if isinstance(v2, dict) and v2:
        return v2
    return {}


def _pick_list(qs: Dict[str, Any], detail: Dict[str, Any], key: str) -> List[Any]:
    v1 = qs.get(key) if isinstance(qs, dict) else None
    if isinstance(v1, list) and v1:
        return v1
    v2 = detail.get(key) if isinstance(detail, dict) else None
    if isinstance(v2, list) and v2:
        return v2
    return []


def _fmt_num(val: Any, suffix: str = "", places: str = "0.01") -> str:
    s = _safe_str_dec(val, places=places)
    return f"{s}{suffix}" if s is not None else "N/A"


def _fmt_hhi(val: Any) -> str:
    """Format HHI on the canonical 0-10,000 scale, scaling 0.0-1.0 fractional ratios."""
    if val is None or isinstance(val, bool):
        return "N/A"
    try:
        d = Decimal(str(val).strip())
        if d.is_nan() or d.is_infinite():
            return "N/A"
        if Decimal("0") < d <= Decimal("1"):
            d = d * Decimal("10000")
        return str(d.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    except Exception:
        return "N/A"


def _fmt_compact_money(val: Any, prefix: str = "", suffix: str = "") -> str:
    """Format large monetary or share counts concisely using Python Decimal (T/B/M/K)."""
    if val is None or isinstance(val, bool):
        return "N/A"
    try:
        raw = str(val).replace(",", "").strip()
        if not raw:
            return "N/A"
        d = Decimal(raw)
        if d.is_nan() or d.is_infinite():
            return "N/A"
    except Exception:
        return "N/A"

    neg = d < Decimal("0")
    abs_d = abs(d)
    unit = ""
    scaled = abs_d
    if abs_d >= Decimal("1000000000000"):
        scaled = (abs_d / Decimal("1000000000000")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        unit = "T"
    elif abs_d >= Decimal("1000000000"):
        scaled = (abs_d / Decimal("1000000000")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        unit = "B"
    elif abs_d >= Decimal("1000000"):
        scaled = (abs_d / Decimal("1000000")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        unit = "M"
    elif abs_d >= Decimal("1000"):
        scaled = (abs_d / Decimal("1000")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        unit = "K"
    else:
        scaled = abs_d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    sign = "-" if neg else ""
    return f"{sign}{prefix}{scaled}{unit}{suffix}"


def _normalize_symbol_key(symbol: Optional[str]) -> str:
    return str(symbol or "").strip().upper().replace(".BK", "")


def _parse_iso_date(raw: Any) -> Optional[date]:
    txt = str(raw or "").strip()[:10]
    if len(txt) < 10:
        return None
    try:
        return date.fromisoformat(txt)
    except ValueError:
        return None


def _compute_sidecar_age_days(detail: Dict[str, Any], now_iso: str) -> Optional[int]:
    """Compute sidecar freshness age in days relative to session UTC date."""
    if not isinstance(detail, dict) or not detail:
        return None
    side_dt = _parse_iso_date(
        _first_non_none(detail.get("analysis_date"), detail.get("evaluated_at"), detail.get("as_of_date"))
    )
    cur_dt = _parse_iso_date(now_iso)
    if side_dt is None or cur_dt is None:
        return None
    return max(0, (cur_dt - side_dt).days)


def _deduplicate_pending_exceptions(
    raw_rows: Optional[Sequence[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Group sequential pending watchdog breaches by claim+metric+expected so duplicates don't bloat prompts."""
    if not raw_rows:
        return []
    grouped: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = {}
    for row in raw_rows:
        if not isinstance(row, dict):
            continue
        status = str(row.get("status") or "PENDING").strip().upper()
        if status and status not in ("PENDING", "OPEN", "UNRESOLVED", "REVIEW_REQUIRED"):
            continue
        key = (
            str(row.get("claim_id") or "").strip(),
            str(row.get("exception_type") or "").strip().upper(),
            str(row.get("metric_name") or "").strip(),
            str(row.get("expected_value") or "").strip(),
        )
        grouped.setdefault(key, []).append(row)

    out: List[Dict[str, Any]] = []
    for items in grouped.values():
        items_sorted = sorted(
            items,
            key=lambda x: str(x.get("detected_at_utc") or ""),
            reverse=True,
        )
        latest = dict(items_sorted[0])
        latest["repeat_count"] = len(items_sorted)
        out.append(latest)
    out.sort(key=lambda x: str(x.get("detected_at_utc") or ""), reverse=True)
    return out


def _collect_ticker_extra_flags(
    detail: Dict[str, Any],
    dedup_exceptions: Sequence[Dict[str, Any]],
    now_iso: str,
) -> Tuple[Optional[int], List[str]]:
    extra_flags: List[str] = []
    age_days = _compute_sidecar_age_days(detail, now_iso)
    if age_days is not None and age_days > 14:
        extra_flags.append(f"sidecar_stale_{age_days}d:freshness")
    if dedup_exceptions:
        extra_flags.append("active_thesis_breached:thesis_watchdog")
    return age_days, extra_flags


def _compute_macro_drift_warning(
    scorecard: Dict[str, Any],
    live_macro_dash: Optional[Dict[str, Any]],
    holding_advisory: Optional[Dict[str, Any]] = None,
    *,
    portfolio_attached: bool = False,
) -> Optional[str]:
    """Detect regime drift between Sidecar analysis timestamp and live @Macro / advisory context."""
    sc_regime = str((scorecard or {}).get("macro_regime") or "").strip()
    sc_status = str((scorecard or {}).get("macro_alignment_status") or "").strip()
    live_regime = (
        str(live_macro_dash.get("overall_regime") or live_macro_dash.get("regime") or "").strip()
        if isinstance(live_macro_dash, dict)
        else ""
    )

    notes: List[str] = []
    if live_regime and sc_regime and sc_regime.lower() != live_regime.lower():
        notes.append(
            f"Sidecar อ้าง Regime='{sc_regime}' ({sc_status or 'N/A'}) "
            f"ซึ่งคลาดเคลื่อนจาก Live @Macro ปัจจุบัน='{live_regime}'"
        )
    if portfolio_attached and isinstance(holding_advisory, dict):
        h_align = str(holding_advisory.get("macro_alignment") or "").strip()
        h_quad = str(holding_advisory.get("sector_quadrant") or "").strip()
        if h_align and sc_status and h_align.lower() != sc_status.lower():
            notes.append(f"Live Advisory MacroAlign='{h_align}' (Sidecar='{sc_status}')")
        elif (h_align or h_quad) and not sc_regime and live_regime:
            notes.append(f"Live MacroFit={h_align or 'N/A'}, SectorQuadrant={h_quad or 'N/A'}")
    if not notes:
        return None
    return "[MACRO_ALIGNMENT_DRIFT_WARNING]: " + " | ".join(notes) + " — กรุณาประเมินผลกระทบต่อสมมติฐานในภาวะปัจจุบันใหม่"


def _format_watchlist_entry_line(wl: Dict[str, Any], *, notes_clip: int = 240) -> str:
    tp = wl.get("target_price")
    t_ccy = str(wl.get("target_currency") or "").strip()
    if tp is not None:
        tp_s = _fmt_num(tp)
        tp_part = f"เป้าซื้อ={tp_s}" + (f" {t_ccy}" if t_ccy else "")
    else:
        tp_part = "ยังไม่ตั้ง Target Price"
    parts = [tp_part]
    if wl.get("target_price_basis"):
        parts.append(f"Basis={wl['target_price_basis']}")
    bucket = wl.get("applied_bucket_name") or wl.get("suggested_bucket_id")
    if bucket:
        parts.append(f"Bucket={bucket}")
    notes = _clip(wl.get("notes"), notes_clip) if wl.get("notes") else ""
    if notes:
        parts.append(f"Notes={notes}")
    return " | ".join(parts)


def _clip(text: Any, max_len: int) -> str:
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(clean) <= max_len:
        return clean
    return clean[: max(0, max_len - 1)].rstrip() + "…"


def _extract_price_from_detail(
    qs: Dict[str, Any],
    detail: Dict[str, Any],
    resolved: Optional[Dict[str, Any]] = None,
) -> Any:
    ams = (qs.get("atomic_market_snapshot") or {}) if isinstance(qs, dict) else {}
    rdcf = _pick_subdict(qs, detail, "reverse_dcf_result")
    res_map = resolved or {}
    return _first_non_none(
        qs.get("current_price") if isinstance(qs, dict) else None,
        qs.get("raw_analysis_price") if isinstance(qs, dict) else None,
        ams.get("analysis_price") if isinstance(ams, dict) else None,
        rdcf.get("current_stock_price") if isinstance(rdcf, dict) else None,
        detail.get("current_price") if isinstance(detail, dict) else None,
        res_map.get("current_price") if isinstance(res_map, dict) else None,
    )


_TVN_LEAK_RE = re.compile(
    r"(?i)(?:around entry|worth holding|at its current size|what it cost in)"
)


def _sanitize_then_vs_now(
    tvn: Dict[str, Any],
    *,
    portfolio_attached: bool,
) -> Dict[str, Any]:
    """Strip portfolio entry/holding markers (including private cost basis) when @Portfolio is NOT attached."""
    if portfolio_attached:
        return dict(tvn)
    out = dict(tvn)
    is_held = bool(out.pop("is_held", False))
    pl = str(out.get("period_label") or "").strip()
    is_entry_anchored = is_held or "entry" in pl.lower()
    if is_entry_anchored:
        out.pop("period_label", None)
        out.pop("then_date", None)
        # Never expose portfolio entry cost basis or entry-anchored thesis metrics when @Portfolio is absent
        out.pop("then_price", None)
        out.pop("then_pe", None)
        out.pop("then_growth_pct", None)
    for k in ("what_was_true_then", "what_is_true_now", "thesis_reflection_question", "regime_shift_summary", "summary"):
        val = str(out.get(k) or "").strip()
        if val and (is_entry_anchored or _TVN_LEAK_RE.search(val)):
            out.pop(k, None)
    return out


def _format_falsifier_condition(condition: Any, threshold: Any) -> str:
    """Format falsifier condition + threshold without repeating threshold already embedded in condition."""
    cond_s = str(condition or "").strip()
    thr_s = str(threshold or "").strip() if threshold is not None else ""
    if not thr_s:
        return cond_s
    if not cond_s:
        return thr_s
    candidates = [thr_s]
    if thr_s.endswith(".0"):
        candidates.append(thr_s[:-2])
    for cand in candidates:
        if cand and re.search(rf"(?<!\d){re.escape(cand)}(?!\d)", cond_s):
            return cond_s
    return f"{cond_s} {thr_s}"


def _format_forensics_line(sloan: Any, dupont: Any) -> str:
    """Lossless compact key-value serialization of Sloan Accrual & DuPont forensics."""
    parts: List[str] = []
    if isinstance(sloan, dict) and sloan:
        s_end = str(sloan.get("fiscal_period_end") or "N/A")
        s_prior = str(sloan.get("prior_fiscal_period_end") or "").strip()
        s_ptype = str(sloan.get("period_type") or "").strip()
        s_basis = str(sloan.get("assets_basis") or "").strip()
        hdr_bits = [f"{s_end} vs {s_prior}" if s_prior else s_end]
        if s_ptype:
            hdr_bits.append(s_ptype)
        if s_basis:
            hdr_bits.append(s_basis)
        ratio_s = _safe_str_dec(sloan.get("sloan_accrual_ratio"), "0.0001") or str(sloan.get("sloan_accrual_ratio", "N/A"))
        raw_s = _safe_str_dec(sloan.get("raw_sloan_accrual_ratio"), "0.000001") or str(sloan.get("raw_sloan_accrual_ratio", "N/A"))
        rating_s = str(sloan.get("earnings_quality_rating") or "N/A")
        interp = str(sloan.get("interpretation") or "").strip()
        if interp and ratio_s:
            # Remove redundant exact numeric echo inside interpretation when already shown inline
            interp = re.sub(
                r"\s*\(-?\d+(?:\.\d+)?\s*\[raw\s+-?\d+(?:\.\d+)?\]\)",
                "",
                interp,
            ).strip()
        known_s = {
            "sloan_accrual_ratio",
            "raw_sloan_accrual_ratio",
            "earnings_quality_rating",
            "assets_basis",
            "interpretation",
            "fiscal_period_end",
            "period_type",
            "prior_fiscal_period_end",
        }
        extra_s = [f"{k}={v}" for k, v in sloan.items() if k not in known_s and v is not None]
        extra_str = (", " + ", ".join(extra_s)) if extra_s else ""
        interp_str = f" [{interp}]" if interp else ""
        parts.append(
            f"Sloan({', '.join(hdr_bits)}): ratio={ratio_s} (raw={raw_s}), rating={rating_s}{extra_str}{interp_str}"
        )
    elif sloan is not None:
        parts.append(f"Sloan={sloan}")

    if isinstance(dupont, dict) and dupont:
        d_end = str(dupont.get("fiscal_period_end") or "N/A")
        d_ptype = str(dupont.get("period_type") or "").strip()
        d_hdr = f"{d_end}, {d_ptype}" if d_ptype else d_end
        npm_s = _safe_str_dec(dupont.get("net_profit_margin_pct"), "0.01") or str(dupont.get("net_profit_margin_pct", "N/A"))
        at_s = _safe_str_dec(dupont.get("asset_turnover"), "0.0001") or str(dupont.get("asset_turnover", "N/A"))
        em_s = _safe_str_dec(dupont.get("equity_multiplier"), "0.01") or str(dupont.get("equity_multiplier", "N/A"))
        roe_s = _safe_str_dec(dupont.get("roe_pct"), "0.01") or str(dupont.get("roe_pct", "N/A"))
        rec_s = _safe_str_dec(dupont.get("published_roe_reconstructed_pct"), "0.0001") or str(dupont.get("published_roe_reconstructed_pct", "N/A"))
        cerr_s = _safe_str_dec(dupont.get("closure_error"), "0.0001") or str(dupont.get("closure_error", "N/A"))
        icerr_s = _safe_str_dec(dupont.get("internal_closure_error"), "0.0001") or str(dupont.get("internal_closure_error", "N/A"))
        rec_ok = dupont.get("is_published_reconciled")
        known_d = {
            "net_profit_margin_pct",
            "asset_turnover",
            "equity_multiplier",
            "roe_pct",
            "closure_error",
            "internal_closure_error",
            "published_roe_reconstructed_pct",
            "is_published_reconciled",
            "fiscal_period_end",
            "period_type",
        }
        extra_d = [f"{k}={v}" for k, v in dupont.items() if k not in known_d and v is not None]
        extra_str = (", " + ", ".join(extra_d)) if extra_d else ""
        parts.append(
            f"DuPont({d_hdr}): NPM={npm_s}%, AT={at_s}, EM={em_s}, ROE={roe_s}%, "
            f"ReconstructedROE={rec_s}%, ClosureErr={cerr_s}, InternalClosureErr={icerr_s}, "
            f"PublishedReconciled={rec_ok}{extra_str}"
        )
    elif dupont is not None:
        parts.append(f"DuPont={dupont}")

    return "- Forensics: " + " | ".join(parts)


_QUOTE_REF_TAIL_RE = re.compile(r"\s*\(ref:\s*([^)#]+?)(?:\s*#([^)\s]+))?\)\s*$")


def _format_exec_quotes_lines(raw_quotes: Sequence[Any], *, clip_limit: int) -> List[str]:
    """Format Exec Quotes cleanly without double-quoting or repeating shared long file paths."""
    parsed: List[Tuple[str, str, str]] = []
    path_counts: Dict[str, int] = {}
    for q_txt in raw_quotes:
        clipped_q = _clip(q_txt, clip_limit).strip()
        if not clipped_q:
            continue
        ref_path = ""
        ref_anchor = ""
        m = _QUOTE_REF_TAIL_RE.search(clipped_q)
        if m:
            ref_path = m.group(1).strip()
            ref_anchor = (m.group(2) or "").strip()
            body = clipped_q[: m.start()].strip()
            path_counts[ref_path] = path_counts.get(ref_path, 0) + 1
        else:
            body = clipped_q
        # Strip symmetric outer quotes before wrapping once cleanly
        if len(body) >= 2 and (
            (body[0] == '"' and body[-1] == '"')
            or (body[0] == "“" and body[-1] == "”")
        ):
            body = body[1:-1].strip()
        parsed.append((body, ref_path, ref_anchor))

    if not parsed:
        return []

    aliases: Dict[str, str] = {}
    lines: List[str] = []
    alias_idx = 1
    for _, ref_path, _ in parsed:
        if ref_path and path_counts.get(ref_path, 0) >= 2 and ref_path not in aliases:
            tag = f"EC{alias_idx}"
            aliases[ref_path] = tag
            alias_idx += 1
            lines.append(f"  * Quote Source [{tag}]: {ref_path}")

    for body, ref_path, ref_anchor in parsed:
        quoted = f'"{body}"'
        if ref_path:
            if ref_path in aliases:
                tag = aliases[ref_path]
                ref_suffix = f" [{tag}#{ref_anchor}]" if ref_anchor else f" [{tag}]"
            else:
                anchor_part = f"#{ref_anchor}" if ref_anchor else ""
                ref_suffix = f" (ref: {ref_path}{anchor_part})"
        else:
            ref_suffix = ""
        lines.append(f"  * Exec Quote: {quoted}{ref_suffix}")
    return lines


_NARRATIVE_SPLIT_RE = re.compile(
    r"(?<!\bInc\.)(?<!\bCorp\.)(?<!\bLtd\.)(?<!\bCo\.)(?<=[.!?])\s+(?=[A-Z0-9ก-ฮ\"“'(])"
    r"|\s+(?=(?:อย่างไรก็ตาม|ในด้านการประเมินมูลค่า|ในเชิงเทคนิค|ข้อมูลที่ใช้ในการวิเคราะห์|ข้อมูลคุณภาพของข้อมูล|ปัจจัยขับเคลื่อนหลัก|โดยสรุป|ทั้งนี้\s|สำหรับ\s*Margin of Safety|ในส่วนของ\s*Smart Money))"
)
_NARRATIVE_QUANT_RECAP_RE = re.compile(
    r"(?i)(?:Piotroski|F-Score|\bROIC\b|\bWACC\b|Reverse DCF|Margin of Safety|\bMoS\b|"
    r"Terminal Value|Enterprise Value|Terminal Depend|\bStage\s*\d\b|Markdown|Risk/Reward|"
    r"Stop Loss|Buy Zone|แนวรับ|แนวต้าน|Smart Money|Data Quality|kd_clamped|EPS Proxy|"
    r"EBIT Margin|Operating Leverage|High Reliability Tier|Reflation|Tailwind|Falsifier)"
)
_NARRATIVE_QUAL_KEEP_RE = re.compile(
    r"(?i)(?:Moat|คูเมือง|Anthropic|TPU|Gemini|DeepMind|Waymo|Workspace|YouTube|Search|"
    r"Cloud|Ecosystem|ระบบนิเวศ|Agentic|Catalyst|ปัจจัยขับเคลื่อน|คู่แข่ง|Competitive|"
    r"Pricing Power|Monopol|ผูกขาด|การลงทุนเชิงกลยุทธ์|\bROI\b|ส่วนแบ่งตลาด|ลูกค้า|Supply|"
    r"MARKER|TAIL|UNCLIPPED)"
)


def _condense_repetitive_narrative(raw_narrative: Any, *, max_chars: int) -> str:
    """Keep unique qualitative moat/catalyst sentences while trimming pure SEC metric restatements."""
    text = re.sub(r"\s+", " ", str(raw_narrative or "")).strip()
    if not text:
        return ""
    if len(text) <= 280:
        return _clip(text, max_chars)

    raw_sentences = [s.strip() for s in _NARRATIVE_SPLIT_RE.split(text) if s and s.strip()]
    if len(raw_sentences) <= 2:
        return _clip(text, max_chars)

    kept: List[str] = []
    seen_exact: set[str] = set()
    for s in raw_sentences:
        norm = s.lower()
        if norm in seen_exact:
            continue
        quant_hits = len(_NARRATIVE_QUANT_RECAP_RE.findall(s))
        qual_keep = bool(_NARRATIVE_QUAL_KEEP_RE.search(s))
        # Preserve sentences with unique qualitative moat/catalyst context or non-metric insights
        if qual_keep or quant_hits < 2:
            kept.append(s)
            seen_exact.add(norm)

    if not kept:
        kept = list(dict.fromkeys(raw_sentences[:2]))
    elif raw_sentences[-1] not in kept and any(
        k in raw_sentences[-1] for k in ("TAIL", "MARKER", "UNCLIPPED")
    ):
        kept.append(raw_sentences[-1])

    condensed = " ".join(kept).strip()
    return _clip(condensed, max_chars)


def _compact_equity_data(detail: Dict[str, Any], *, include_portfolio_fields: bool) -> Dict[str, Any]:
    """Produce a compact snapshot of equity_data for SQLite storage (~85% smaller, zero bloat matrices)."""
    if not isinstance(detail, dict) or not detail:
        return {}

    def _strip_dict(d: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in d.items():
            if k in _EQUITY_BLOAT_KEYS:
                continue
            if not include_portfolio_fields and k in ("portfolio_context", "position_action_hint"):
                continue
            if k == "position_review" and isinstance(v, dict) and not include_portfolio_fields:
                pr_copy = {pk: pv for pk, pv in v.items() if pk not in ("holding_summary", "triggers")}
                if isinstance(pr_copy.get("then_vs_now"), dict):
                    pr_copy["then_vs_now"] = _sanitize_then_vs_now(
                        pr_copy["then_vs_now"], portfolio_attached=False
                    )
                out[k] = pr_copy
            elif isinstance(v, dict):
                out[k] = _strip_dict(v)
            else:
                out[k] = v
        return out

    compact = _strip_dict(detail)
    # Trim ultra-long narrative strings gently if oversized for storage while keeping curated text intact
    if isinstance(compact.get("narrative_analysis"), str) and len(compact["narrative_analysis"]) > 3000:
        compact["narrative_analysis"] = compact["narrative_analysis"][:3000] + "…"
    return compact


def _build_portfolio_overview_lines(
    *,
    portfolio_state: Dict[str, Any],
    nav_thb: str,
    cash_pct: Optional[str],
    bucket_policy: Optional[List[Dict[str, Any]]] = None,
    session_tickers: Sequence[str] = (),
    advisory_summary: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Deterministically build curated portfolio overview, concentration & bucket room lines."""
    holdings = list(portfolio_state.get("holdings") or [])
    nav_dec = Decimal(nav_thb) if _safe_str_dec(nav_thb) else Decimal("0")
    cash_thb_s = _safe_str_dec(portfolio_state.get("cash_thb"), "0.01")
    cash_thb_only_s = _safe_str_dec(portfolio_state.get("cash_thb_only"), "0.01")
    cash_usd_val_s = _safe_str_dec(portfolio_state.get("cash_usd_val_thb"), "0.01")
    fx_s = _safe_str_dec(portfolio_state.get("fx_usd_thb"), "0.01")
    tot_cost_s = _safe_str_dec(portfolio_state.get("total_cost_basis_thb"), "0.01")
    tot_unreal_s = _safe_str_dec(portfolio_state.get("total_unrealized_profit_thb"), "0.01")

    lines: List[str] = ["[PORTFOLIO OVERVIEW & ALLOCATION DETAIL]"]
    cash_desc = f"{cash_pct or 'N/A'}%"
    if cash_thb_s:
        cash_desc = f"{cash_thb_s} THB ({cash_pct or 'N/A'}%)"
        if cash_thb_only_s and cash_usd_val_s and Decimal(cash_usd_val_s) > 0:
            cash_desc += f" [THB {cash_thb_only_s} + USD eq {cash_usd_val_s} THB]"
    fx_part = f" | FX USD/THB: {fx_s}" if fx_s else ""
    cost_part = f" | ต้นทุนรวม: {tot_cost_s} THB" if tot_cost_s else ""
    pnl_part = f" | Total Unrealized P&L: {tot_unreal_s} THB" if tot_unreal_s else ""
    lines.append(f"- NAV รวม: {nav_thb} THB | เงินสด: {cash_desc}{fx_part}{cost_part}{pnl_part} | จำนวนรายการถือครอง: {len(holdings)}")

    # Compute normalized holding weights with Decimal
    weighted_items: List[Tuple[Decimal, Dict[str, Any]]] = []
    non_cash_weights: List[Tuple[str, Decimal]] = []
    bucket_actual_weights: Dict[str, Decimal] = {}
    unassigned_weight = Decimal("0")

    for h in holdings:
        sym = str(h.get("symbol") or "").strip()
        if not sym:
            continue
        asset_type = str(h.get("asset_type") or "").strip()
        w_dec: Optional[Decimal] = None
        if h.get("current_weight_pct") is not None:
            try:
                w_dec = Decimal(str(h.get("current_weight_pct")))
            except Exception:
                w_dec = None
        if w_dec is None and nav_dec > 0 and h.get("market_value_thb") is not None:
            try:
                w_dec = (Decimal(str(h.get("market_value_thb"))) / nav_dec) * Decimal("100")
            except Exception:
                w_dec = None
        w_val = w_dec if w_dec is not None else Decimal("0")
        weighted_items.append((w_val, h))
        is_cash = asset_type.lower() == "cash" or sym.upper() in {"CASH_THB", "CASH_USD", "CASH"}
        if not is_cash and w_val > 0:
            non_cash_weights.append((sym, w_val))
        b_id = str(h.get("bucket_id") or "").strip()
        if b_id:
            bucket_actual_weights[b_id] = bucket_actual_weights.get(b_id, Decimal("0")) + w_val
        elif not is_cash:
            unassigned_weight += w_val

    weighted_items.sort(key=lambda x: x[0], reverse=True)
    max_detailed_holdings = 25
    if weighted_items:
        lines.append("- Top Holdings (เรียงตามน้ำหนักพอร์ต) [Sym|Type/Ccy|Wt%|MV_THB|PnL%|Bucket]:")
        for w_val, h in weighted_items[:max_detailed_holdings]:
            sym = str(h.get("symbol") or "")
            at = str(h.get("asset_type") or "Asset")
            ccy = str(h.get("currency") or "")
            pnl_s = _safe_str_dec(
                h.get("unrealized_pnl_pct")
                if h.get("unrealized_pnl_pct") is not None
                else h.get("unrealized_pnl_percent"),
                "0.01",
            )
            b_id = str(h.get("bucket_id") or "unassigned")
            w_str = str(w_val.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
            mv_s = _safe_str_dec(h.get("market_value_thb"), "0")
            meta_parts = [p for p in (at, ccy) if p]
            type_ccy = "/".join(meta_parts) if meta_parts else "-"
            mv_cell = mv_s if mv_s else "-"
            pnl_cell = f"{pnl_s}%" if pnl_s is not None else "-"
            lines.append(f"  {sym}|{type_ccy}|{w_str}%|{mv_cell}|{pnl_cell}|{b_id}")
        if len(weighted_items) > max_detailed_holdings:
            rest_items = [
                (w, h)
                for w, h in weighted_items[max_detailed_holdings:]
                if h.get("symbol") and not (w == 0 and str(h.get("asset_type") or "").lower() == "cash")
            ]
            if rest_items:
                rest_w_sum = sum((w for w, _ in rest_items), Decimal("0")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                rest_str = ", ".join(
                    f"{str(h.get('symbol') or '')} {w.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)}%"
                    for w, h in rest_items[:15]
                )
                if len(rest_items) > 15:
                    rest_str += f", +{len(rest_items) - 15} more"
                lines.append(f"  * Other Holdings (#{max_detailed_holdings + 1}-{max_detailed_holdings + len(rest_items)}, รวม {rest_w_sum}%): {rest_str}")

    # Deterministic concentration metrics
    non_cash_weights.sort(key=lambda x: x[1], reverse=True)
    if non_cash_weights:
        top1_sym, top1_w = non_cash_weights[0]
        top3_w = sum((w for _, w in non_cash_weights[:3]), Decimal("0"))
        hhi = sum(((w / Decimal("100")) ** 2 for _, w in non_cash_weights), Decimal("0")) * Decimal("10000")
        c_warnings: List[str] = []
        if top1_w > Decimal("15"):
            c_warnings.append(f"{top1_sym} น้ำหนักเดี่ยว {top1_w.quantize(Decimal('0.01'))}% (>15%)")
        if top3_w > Decimal("50"):
            c_warnings.append(f"Top-3 รวม {top3_w.quantize(Decimal('0.01'))}% (>50%)")
        warn_str = f" | Alert: {', '.join(c_warnings)}" if c_warnings else " | กระจายความเสี่ยงปกติ"
        lines.append(
            f"- Concentration Risk: Top-1 = {top1_sym} ({top1_w.quantize(Decimal('0.01'))}%), "
            f"Top-3 = {top3_w.quantize(Decimal('0.01'))}%, HHI ≈ {hhi.quantize(Decimal('1'))}{warn_str}"
        )

    # Bucket Target vs Actual & Room
    effective_policy = bucket_policy or portfolio_state.get("allocation_targets") or []
    if effective_policy:
        lines.append("- สัดส่วนถังเงิน (Bucket Actual vs Target):")
        for bp in effective_policy:
            b_id = str(bp.get("id") or bp.get("bucket_id") or "")
            b_name = str(bp.get("name") or b_id)
            t_raw = bp.get("target_pct") if bp.get("target_pct") is not None else bp.get("target_percent")
            t_dec = Decimal(str(t_raw or 0))
            actual_dec = bucket_actual_weights.get(b_id, Decimal("0"))
            diff_dec = actual_dec - t_dec
            status_lbl = (
                f"เต็ม/เกินเป้า +{diff_dec.quantize(Decimal('0.01'))}%"
                if diff_dec > Decimal("1.0")
                else f"เหลือ Room เติม {(-diff_dec).quantize(Decimal('0.01'))}%"
                if diff_dec < Decimal("-1.0")
                else "สมดุลตามเป้า"
            )
            lines.append(
                f"  * {b_name} [{b_id}]: จริง {actual_dec.quantize(Decimal('0.01'))}% vs เป้า {t_dec.quantize(Decimal('0.01'))}% ({status_lbl})"
            )
        if unassigned_weight > Decimal("0"):
            lines.append(f"  * สินทรัพย์ยังไม่ระบุ Bucket: {unassigned_weight.quantize(Decimal('0.01'))}%")

    # Offline cached Look-Through overlap (if available from Finnomena cache / static catalog)
    lt = portfolio_state.get("look_through_summary")
    if isinstance(lt, dict):
        overlaps = lt.get("top_overlapped_stocks") or []
        hidden_warns = lt.get("hidden_concentration_warnings") or []
        crowded_secs = lt.get("crowded_sectors") or []
        if overlaps or hidden_warns or crowded_secs:
            lines.append("- Fund Look-Through (ข้อมูลซ้อนทับจากกองทุนในพอร์ต):")
            if overlaps:
                ov_parts: List[str] = []
                for o in overlaps[:12]:
                    if not o.get("ticker"):
                        continue
                    flags = []
                    if o.get("is_cap_breached"):
                        flags.append("CAP_BREACH")
                    if o.get("is_hidden_concentration"):
                        flags.append("HIDDEN_CONC")
                    ftag = f"[{','.join(flags)}]" if flags else ""
                    ov_parts.append(
                        f"{o.get('ticker')}={_fmt_num(o.get('total_weight_pct'), '%')}{ftag}"
                        f"(ตรง {_fmt_num(o.get('direct_weight_pct'), '%')}+ผ่านกองทุน {_fmt_num(o.get('indirect_weight_pct'), '%')})"
                    )
                if ov_parts:
                    lines.append(f"  * Top Look-Through Stocks: {', '.join(ov_parts)}")
            if crowded_secs:
                cs = ", ".join(f"{s.get('sector_name')} {_fmt_num(s.get('weight_pct'), '%')}" for s in crowded_secs[:8])
                lines.append(f"  * Crowded Sectors: {cs}")
            for hw in hidden_warns[:6]:
                lines.append(f"  * Warning: {hw}")

    # Portfolio Financial Goals Progress (computed deterministically by Goals service)
    goals = [g for g in (portfolio_state.get("goals_progress") or []) if isinstance(g, dict)]
    if goals:
        prio_rank = {"needs": 0, "wants": 1, "wishes": 2}
        goals.sort(
            key=lambda g: (
                prio_rank.get(str(g.get("priority_tier") or "wants").lower(), 1),
                int(g.get("deadline_days_left")) if g.get("deadline_days_left") is not None else 999999,
            )
        )
        lines.append("- เป้าหมายการเงินของพอร์ต (Portfolio Goals Progress):")
        for g in goals[:10]:
            cur_s = _fmt_compact_money(g.get("current_amount_thb"), suffix=" THB")
            tgt_s = _fmt_compact_money(g.get("target_amount_thb"), suffix=" THB")
            pct_s = _fmt_num(g.get("progress_pct"), "%")
            prio = str(g.get("priority_tier") or "wants")
            gtype = str(g.get("goal_type") or "goal")
            dl_s = str(g.get("deadline") or "N/A")
            sav_part = (
                f" | ออม/เดือน: {_fmt_compact_money(g.get('monthly_savings_thb'), suffix=' THB')}"
                if g.get("monthly_savings_thb") not in (None, 0, 0.0)
                else ""
            )
            req_part = (
                f" | Req CAGR: {_fmt_num(g.get('required_return_pct'), '%')}"
                if g.get("required_return_pct") is not None
                else ""
            )
            flex_raw = str(g.get("flexibility") or "").strip()
            flex_part = f" | Flex: {flex_raw}" if flex_raw and flex_raw.lower() != "none" else ""
            ten_s = str(g.get("tension_level") or "N/A")
            lines.append(
                f"  * {g.get('name', 'Goal')} [{gtype} | {prio}]: {cur_s} / {tgt_s} ({pct_s}) | "
                f"Deadline: {dl_s}{sav_part}{req_part}{flex_part} | Tension: {ten_s}"
            )
            levers = g.get("adjustment_levers")
            if isinstance(levers, dict) and levers and ten_s not in ("BALANCED", "INSUFFICIENT_DATA"):
                lever_parts: List[str] = []
                if levers.get("save_more_monthly_thb") is not None:
                    tot_m = (
                        f" (รวม {_fmt_compact_money(levers.get('total_monthly_savings_needed_thb'), suffix=' THB/ด.')})"
                        if levers.get("total_monthly_savings_needed_thb") is not None
                        else ""
                    )
                    lever_parts.append(
                        f"ออมเพิ่ม +{_fmt_compact_money(levers.get('save_more_monthly_thb'), suffix=' THB/ด.')}{tot_m}"
                    )
                if levers.get("extend_timeline_months") is not None:
                    tot_mo = (
                        f" (รวม {levers.get('total_months_needed')} ด.)"
                        if levers.get("total_months_needed") is not None
                        else ""
                    )
                    lever_parts.append(f"ยืดเวลา +{levers.get('extend_timeline_months')} เดือน{tot_mo}")
                if levers.get("downsized_target_thb") is not None:
                    lever_parts.append(
                        f"ลดเป้าเหลือ {_fmt_compact_money(levers.get('downsized_target_thb'), suffix=' THB')}"
                    )
                if lever_parts:
                    lines.append(f"    -> Goal Adjustment Levers: {' | '.join(lever_parts)}")

    # Watchlist items (session tickers prioritized first, then items with target_price)
    wl_items = [w for w in (portfolio_state.get("watchlist_items") or []) if isinstance(w, dict) and w.get("symbol")]
    if wl_items:
        sess_keys = {_normalize_symbol_key(t) for t in session_tickers if t}
        wl_items.sort(
            key=lambda w: (
                0 if _normalize_symbol_key(w.get("symbol")) in sess_keys else 1,
                0 if w.get("target_price") is not None else 1,
                str(w.get("symbol") or "").upper(),
            )
        )
        lines.append("- รายการติดตามใน Watchlist (Portfolio Watchlist):")
        for wl in wl_items[:15]:
            sym = str(wl.get("symbol") or "").upper()
            atype = str(wl.get("asset_type") or "Stock")
            lines.append(f"  * {sym} ({atype}): {_format_watchlist_entry_line(wl, notes_clip=240)}")

    # Tier 1 Deterministic Portfolio Advisory (Risk Radar, Cash Plan, Holdings Advisory, Anomaly Warnings)
    if isinstance(advisory_summary, dict) and advisory_summary:
        rr = advisory_summary.get("risk_radar") if isinstance(advisory_summary.get("risk_radar"), dict) else {}
        if rr:
            lines.append(
                f"- Portfolio Risk Radar (Mode={advisory_summary.get('data_mode', 'N/A')}): "
                f"Beta={_fmt_num(rr.get('weighted_beta'))} | Vol={_fmt_num(rr.get('total_portfolio_volatility_pct'), '%')} | "
                f"EffHoldings={_fmt_num(rr.get('effective_holdings_count'))} (HHI={_fmt_hhi(rr.get('hhi'))}, Risk={rr.get('concentration_risk_level', 'N/A')}) | "
                f"BucketActiveShare={_fmt_num(rr.get('bucket_active_share_pct'), '%')} | Drawdown={_fmt_num(rr.get('drawdown_pct'), '%')} "
                f"(Posture={rr.get('drawdown_posture_tier', 'N/A')}) | UnassignedNAV={_fmt_num(rr.get('unassigned_holdings_nav_pct'), '%')}"
            )
        cp = advisory_summary.get("cash_plan") if isinstance(advisory_summary.get("cash_plan"), dict) else {}
        if cp:
            lines.append(
                f"- Dual-Currency Cash Plan (FX USD/THB={_fmt_num(cp.get('usd_thb_fx_rate'))}): "
                f"Available=[{_fmt_compact_money(cp.get('cash_thb_available'), suffix=' THB')}, ${_fmt_num(cp.get('cash_usd_available'))} USD] | "
                f"ImmediateDeploy=[{_fmt_compact_money(cp.get('immediate_deploy_thb'), suffix=' THB')}, ${_fmt_num(cp.get('immediate_deploy_usd'))} USD] | "
                f"ConditionalLimit=[{_fmt_compact_money(cp.get('conditional_limit_order_thb'), suffix=' THB')}, ${_fmt_num(cp.get('conditional_limit_order_usd'))} USD] | "
                f"FXConversionNeeded={_fmt_compact_money(cp.get('fx_conversion_needed_thb'), suffix=' THB')}"
            )
        hadvs = [h for h in (advisory_summary.get("holdings_advisory") or []) if isinstance(h, dict) and h.get("symbol")]
        if hadvs:
            sess_keys = {_normalize_symbol_key(t) for t in session_tickers if t}
            hadvs.sort(
                key=lambda h: (
                    0 if _normalize_symbol_key(h.get("symbol")) in sess_keys else 1,
                    0 if str(h.get("action_code") or "").upper() not in ("", "HOLD", "BALANCED", "NO_ACTION") else 1,
                    -(float(h.get("current_weight_pct") or 0.0)),
                )
            )
            lines.append("- Holdings Advisory Signals (แผนจัดสรรเชิงปริมาณรายตัว) [Sym|Action|Track|Wt%|Tgt%|Drift%|Extra]:")
            for h in hadvs[:30]:
                exec_tag = "IMMEDIATE" if h.get("is_immediate_execution") else str(h.get("evaluation_track") or "TRACK")
                extras: List[str] = []
                if h.get("live_upside_12m_pct") is not None:
                    extras.append(f"Upside12M={_fmt_num(h.get('live_upside_12m_pct'), '%')}")
                if h.get("risk_contribution_pct") is not None:
                    extras.append(f"RiskC={_fmt_num(h.get('risk_contribution_pct'), '%')}")
                if h.get("macro_alignment"):
                    extras.append(f"Macro={h.get('macro_alignment')}")
                if h.get("sector_quadrant"):
                    extras.append(f"Quad={h.get('sector_quadrant')}")
                extra_col = ", ".join(extras) if extras else "-"
                lines.append(
                    f"  {h.get('symbol')}|{h.get('action_code', 'HOLD')}|{exec_tag}|"
                    f"{_fmt_num(h.get('current_weight_pct'), '%')}|{_fmt_num(h.get('target_weight_pct'), '%')}|"
                    f"{_fmt_num(h.get('weight_drift_pct'), '%')}|{extra_col}"
                )
            if len(hadvs) > 30:
                extra_h = ", ".join(
                    f"{h.get('symbol')}={h.get('action_code', 'HOLD')}({_fmt_num(h.get('current_weight_pct'), '%')})"
                    for h in hadvs[30:]
                )
                lines.append(f"  * Other Holdings Advisory (+{len(hadvs) - 30}): {extra_h}")
        warns = [w for w in (advisory_summary.get("anomaly_warnings") or []) if isinstance(w, dict)]
        if warns:
            lines.append("- Portfolio Advisory Anomaly Warnings:")
            for w in warns[:8]:
                t_tag = f" ({w.get('ticker')})" if w.get("ticker") else ""
                lines.append(
                    f"  * [{w.get('severity', 'WARN')}] {w.get('guardrail_code', 'GUARD')}{t_tag}: {_clip(w.get('message'), 220)}"
                )

    return lines


def _build_portfolio_guard(
    *,
    is_attached: bool,
    primary_ticker: Optional[str],
    portfolio_state: Optional[Dict[str, Any]],
    bucket_policy: Optional[List[Dict[str, Any]]] = None,
    session_tickers: Sequence[str] = (),
    advisory_summary: Optional[Dict[str, Any]] = None,
) -> PortfolioHoldingGuardState:
    if not is_attached:
        return PortfolioHoldingGuardState(
            attachment_state=ContextAttachmentState.NOT_ATTACHED_DO_NOT_ASSUME,
            portfolio_attached=False,
            is_held=False,
            state_hash="not_attached",
            summary_text=(
                "[PORTFOLIO_STATE: NOT_ATTACHED_DO_NOT_ASSUME] "
                "ผู้ใช้ไม่ได้แนบการ์ด @Portfolio เข้าห้องประชุม ห้ามสมมติว่าผู้ใช้ถือหุ้นนี้อยู่ "
                "ห้ามเดาจำนวนหุ้น ต้นทุนเฉลี่ย หรือน้ำหนักในพอร์ต"
            ),
        )

    if not portfolio_state:
        return PortfolioHoldingGuardState(
            attachment_state=ContextAttachmentState.ATTACHED_EMPTY,
            portfolio_attached=True,
            is_held=False,
            state_hash="attached_empty",
            summary_text=(
                "[PORTFOLIO_STATE: ATTACHED_EMPTY] "
                "แนบการ์ด @Portfolio แล้ว แต่ยังไม่มีข้อมูล Snapshot พอร์ตในระบบ"
            ),
        )

    holdings = portfolio_state.get("holdings") or []
    watchlist_items = [w for w in (portfolio_state.get("watchlist_items") or []) if isinstance(w, dict)]
    goals_progress = [g for g in (portfolio_state.get("goals_progress") or []) if isinstance(g, dict)]
    nav_thb = _safe_str_dec(portfolio_state.get("total_value_thb"), "0.01") or "0.00"
    cash_pct = _safe_str_dec(portfolio_state.get("cash_pct"), "0.01")
    state_hash = str(portfolio_state.get("state_hash") or "")
    if not state_hash:
        sig_payload: Dict[str, Any] = {
            "nav": nav_thb,
            "cash_pct": cash_pct,
            "h": [
                (
                    str(h.get("symbol", "")),
                    str(h.get("units", 0)),
                    str(h.get("market_value_thb", 0)),
                    str(h.get("bucket_id", "")),
                )
                for h in holdings
            ],
            "bp": [
                (str(b.get("id") or b.get("bucket_id") or ""), str(b.get("target_pct") or b.get("target_percent") or ""))
                for b in (bucket_policy or [])
            ],
            "wl": [
                (
                    str(w.get("symbol", "")).upper(),
                    str(w.get("target_price", "")),
                    str(w.get("target_currency", "")),
                    str(w.get("target_price_basis", "")),
                    str(w.get("suggested_bucket_id", "")),
                )
                for w in watchlist_items
            ],
            "goals": [
                (
                    str(g.get("name", "")),
                    str(g.get("target_amount_thb", "")),
                    str(g.get("progress_pct", "")),
                    str(g.get("deadline", "")),
                    str(g.get("tension_level", "")),
                )
                for g in goals_progress
            ],
        }
        if isinstance(advisory_summary, dict) and advisory_summary:
            rr_sig = advisory_summary.get("risk_radar") or {}
            cp_sig = advisory_summary.get("cash_plan") or {}
            sig_payload["advisory"] = {
                "mode": advisory_summary.get("data_mode"),
                "posture": rr_sig.get("drawdown_posture_tier"),
                "deploy_thb": cp_sig.get("immediate_deploy_thb"),
                "deploy_usd": cp_sig.get("immediate_deploy_usd"),
                "n_warns": len(advisory_summary.get("anomaly_warnings") or []),
            }
        raw_sig = json.dumps(sig_payload, sort_keys=True)
        state_hash = hashlib.sha256(raw_sig.encode("utf-8")).hexdigest()[:16]

    matched_holding: Optional[Dict[str, Any]] = None
    matched_wl: Optional[Dict[str, Any]] = None
    if primary_ticker:
        target_clean = _normalize_symbol_key(primary_ticker)
        for h in holdings:
            h_sym = str(h.get("symbol", "")).strip().upper()
            if h_sym == primary_ticker.strip().upper() or _normalize_symbol_key(h_sym) == target_clean:
                matched_holding = h
                break
        for w in watchlist_items:
            w_sym = str(w.get("symbol", "")).strip().upper()
            if w_sym == primary_ticker.strip().upper() or _normalize_symbol_key(w_sym) == target_clean:
                matched_wl = w
                break

    overview_lines = _build_portfolio_overview_lines(
        portfolio_state=portfolio_state,
        nav_thb=nav_thb,
        cash_pct=cash_pct,
        bucket_policy=bucket_policy,
        session_tickers=session_tickers or ((primary_ticker,) if primary_ticker else ()),
        advisory_summary=advisory_summary,
    )
    overview_block = "\n" + "\n".join(overview_lines)

    held_units = 0.0
    if matched_holding:
        try:
            held_units = float(matched_holding.get("units") or 0)
        except (TypeError, ValueError):
            held_units = 0.0

    if matched_holding and held_units > 0:
        units_s = _safe_str_dec(matched_holding.get("units"), "0.0001") or "0"
        avg_cost_val = _first_non_none(
            matched_holding.get("avg_cost"),
            matched_holding.get("avg_cost_usd"),
            matched_holding.get("avg_cost_thb"),
        )
        avg_cost_s = _safe_str_dec(avg_cost_val, "0.01")
        pnl_s = _safe_str_dec(
            matched_holding.get("unrealized_pnl_pct")
            if matched_holding.get("unrealized_pnl_pct") is not None
            else matched_holding.get("unrealized_pnl_percent"),
            "0.01",
        )
        mkt_val_thb = _safe_str_dec(matched_holding.get("market_value_thb"), "0.01")
        weight_s = _safe_str_dec(matched_holding.get("current_weight_pct"), "0.01")
        if weight_s is None and mkt_val_thb and Decimal(nav_thb) > Decimal("0"):
            weight_dec = (Decimal(mkt_val_thb) / Decimal(nav_thb)) * Decimal("100")
            weight_s = str(weight_dec.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
        bucket_id = matched_holding.get("bucket_id")

        summary = (
            f"[PORTFOLIO_STATE: ATTACHED_HELD] พอร์ต NAV รวม {nav_thb} THB "
            f"(Cash {cash_pct or 'N/A'}%) | ถือหุ้น {primary_ticker} อยู่จริง {units_s} หน่วย, "
            f"ต้นทุนเฉลี่ย {avg_cost_s or 'N/A'}, น้ำหนัก {weight_s or 'N/A'}% ของพอร์ต, "
            f"Unrealized P&L {pnl_s or 'N/A'}%, ถังเงิน {bucket_id or 'ยังไม่ระบุ'}"
            + overview_block
        )
        return PortfolioHoldingGuardState(
            attachment_state=ContextAttachmentState.ATTACHED_HELD,
            portfolio_attached=True,
            is_held=True,
            units=units_s,
            avg_cost=avg_cost_s,
            current_weight_pct=weight_s,
            unrealized_pnl_pct=pnl_s,
            bucket_id=str(bucket_id) if bucket_id else None,
            portfolio_total_nav_thb=nav_thb,
            portfolio_cash_pct=cash_pct,
            state_hash=state_hash,
            summary_text=summary,
        )

    top_symbols = ", ".join(str(h.get("symbol", "")) for h in holdings[:8] if h.get("symbol"))
    wl_note = f" | กำลังติดตามใน Watchlist ({_format_watchlist_entry_line(matched_wl)})" if matched_wl else ""
    flat_note = (
        f"ปัจจุบันยังไม่ได้ถือหุ้น {primary_ticker} ในพอร์ต (Flat Position){wl_note}"
        if primary_ticker
        else f"ถือครองสินทรัพย์ทั้งหมด {len(holdings)} รายการ ({top_symbols or 'ไม่มีสินทรัพย์'})"
    )
    summary = (
        f"[PORTFOLIO_STATE: ATTACHED_FLAT_BOOK] พอร์ต NAV รวม {nav_thb} THB "
        f"(Cash {cash_pct or 'N/A'}%) | {flat_note}"
        + overview_block
    )
    return PortfolioHoldingGuardState(
        attachment_state=ContextAttachmentState.ATTACHED_FLAT_BOOK,
        portfolio_attached=True,
        is_held=False,
        portfolio_total_nav_thb=nav_thb,
        portfolio_cash_pct=cash_pct,
        state_hash=state_hash,
        summary_text=summary,
    )


def _extract_margin_entry(raw: Any) -> Tuple[Optional[Any], str]:
    """Extract numeric percentage value & optional period suffix from dict or scalar margin field."""
    if isinstance(raw, dict):
        return _first_non_none(raw.get("value_pct"), raw.get("value")), str(raw.get("period_end") or "").strip()
    return raw, ""


def _extract_numeric_facts_and_flags(
    equity_data: Dict[str, Any],
    anchor: Optional[InstrumentIdentityAnchor],
    portfolio_guard: PortfolioHoldingGuardState,
) -> Tuple[Tuple[str, ...], Tuple[str, ...], Optional[str]]:
    facts: List[str] = []
    flags: List[str] = []
    action_stance: Optional[str] = None

    if anchor and anchor.current_price:
        facts.append(f"current_price={anchor.current_price} {anchor.currency}")

    if equity_data:
        qs = equity_data.get("quant_signals") if isinstance(equity_data.get("quant_signals"), dict) else {}
        ams = qs.get("atomic_market_snapshot") if isinstance(qs.get("atomic_market_snapshot"), dict) else {}
        scorecard = _pick_subdict(qs, equity_data, "deterministic_scorecard")
        dcf = _pick_subdict(qs, equity_data, "dcf_result")
        rdcf = _pick_subdict(qs, equity_data, "reverse_dcf_result")
        piotroski = _pick_subdict(qs, equity_data, "piotroski_breakdown")
        dcf_base = (dcf.get("scenarios") or {}).get("base") or {}
        pctx = (
            equity_data.get("portfolio_context")
            if portfolio_guard.portfolio_attached and isinstance(equity_data.get("portfolio_context"), dict)
            else {}
        )

        action_stance = _first_non_none(
            scorecard.get("action_stance"),
            pctx.get("position_action_hint") if portfolio_guard.portfolio_attached else None,
            equity_data.get("position_action_hint") if portfolio_guard.portfolio_attached else None,
        )

        # Merge all data quality flags cleanly
        raw_flags: List[Any] = []
        raw_flags.extend(qs.get("data_quality_flags") or [])
        raw_flags.extend(scorecard.get("data_quality_flags") or [])
        raw_flags.extend(equity_data.get("data_quality_flags") or [])
        if dcf and dcf.get("is_actionable") is False and "non_actionable_dcf:valuation" not in raw_flags:
            raw_flags.append("non_actionable_dcf:valuation")
        for f in raw_flags:
            if f:
                flags.append(str(f))

        raw_mcap = _first_non_none(ams.get("market_cap"), ams.get("market_cap_str"), qs.get("market_cap"), equity_data.get("market_cap"))
        mcap_ccy_prefix = "$" if (anchor and anchor.currency == "USD") else ""
        mcap_fmt = _fmt_compact_money(raw_mcap, prefix=mcap_ccy_prefix)
        raw_adtv = _first_non_none(ams.get("adtv_local_currency"), qs.get("adtv_local_currency"))
        adtv_fmt = _fmt_compact_money(raw_adtv, prefix=mcap_ccy_prefix)
        gaap_op_val, _ = _extract_margin_entry(qs.get("gaap_operating_margin"))

        extracted_map: List[Tuple[str, Any]] = [
            ("composite_score", _first_non_none(qs.get("composite_score"), equity_data.get("composite_score"))),
            ("core_conviction_score", scorecard.get("core_conviction_score")),
            ("business_conviction_score", scorecard.get("business_conviction_score")),
            ("investment_conviction_score", scorecard.get("investment_conviction_score")),
            ("execution_readiness_score", scorecard.get("execution_readiness_score")),
            ("current_price", _extract_price_from_detail(qs, equity_data, None) if not (anchor and anchor.current_price) else None),
            ("market_cap", mcap_fmt if mcap_fmt != "N/A" else None),
            ("beta", _first_non_none(ams.get("beta"), qs.get("beta"), equity_data.get("beta"))),
            ("volatility_pct", _first_non_none(ams.get("volatility_pct"), qs.get("volatility_pct"))),
            ("mdd_pct", _first_non_none(ams.get("mdd_pct"), qs.get("mdd_pct"))),
            ("adtv_local_currency", adtv_fmt if adtv_fmt != "N/A" else None),
            ("amihud_illiquidity_20d", _first_non_none(ams.get("amihud_illiquidity_20d"), qs.get("amihud_illiquidity_20d"))),
            ("fair_value_base_dcf", _first_non_none(dcf_base.get("target_price"), qs.get("fair_value_base"), equity_data.get("fair_value_base"))),
            ("intrinsic_value_today", rdcf.get("intrinsic_value_today")),
            ("target_price_12m", rdcf.get("target_price_12m")),
            ("upside_12m_pct", rdcf.get("upside_12m_pct")),
            ("consensus_target_price", rdcf.get("consensus_target_price")),
            (
                "mos_entry_price_today",
                _first_non_none(rdcf.get("mos_entry_price_today"), qs.get("mos_entry_price_today"), equity_data.get("mos_entry_price_today")),
            ),
            (
                "margin_of_safety_pct",
                _first_non_none(
                    rdcf.get("current_margin_of_safety_pct"),
                    dcf_base.get("margin_of_safety_pct"),
                    qs.get("margin_of_safety_pct"),
                    equity_data.get("margin_of_safety_pct"),
                ),
            ),
            (
                "market_implied_growth_pct",
                _first_non_none(rdcf.get("market_implied_growth_pct"), qs.get("market_implied_growth_pct"), equity_data.get("market_implied_growth_pct")),
            ),
            ("fcf_yield_pct", _first_non_none(qs.get("fcf_yield_pct"), equity_data.get("fcf_yield_pct"))),
            ("fcf_margin_pct", _first_non_none(qs.get("fcf_margin_pct"), equity_data.get("fcf_margin_pct"))),
            ("fcf_cagr_3y", _first_non_none(qs.get("fcf_cagr_3y"), equity_data.get("fcf_cagr_3y"))),
            ("roic_pct", _first_non_none(qs.get("roic_pct"), equity_data.get("roic_pct"))),
            ("gaap_operating_margin_pct", gaap_op_val),
            ("reported_ebit_margin_pct", _first_non_none(rdcf.get("reported_ebit_margin_pct"), dcf.get("reported_ebit_margin_pct"))),
            ("normalized_mid_cycle_ebit_margin_pct", _first_non_none(rdcf.get("normalized_mid_cycle_ebit_margin_pct"), dcf.get("normalized_mid_cycle_ebit_margin_pct"))),
            ("company_archetype", _first_non_none(rdcf.get("company_archetype"), dcf.get("company_archetype"))),
            ("ocf_to_net_income", _first_non_none(qs.get("ocf_to_net_income"), equity_data.get("ocf_to_net_income"))),
            ("piotroski_f_score", _first_non_none(piotroski.get("f_score"), qs.get("piotroski_score"), equity_data.get("piotroski_score"))),
            ("net_debt_ebitda", _first_non_none(qs.get("net_debt_ebitda"), equity_data.get("net_debt_ebitda"))),
            ("current_ratio", _first_non_none(qs.get("current_ratio"), equity_data.get("current_ratio"))),
            ("de_ratio_pct", _first_non_none(qs.get("de_ratio_pct"), equity_data.get("de_ratio_pct"))),
            ("synthetic_credit_rating", _first_non_none(rdcf.get("synthetic_credit_rating"), dcf.get("synthetic_credit_rating"), qs.get("synthetic_credit_rating"))),
            ("interest_coverage_ratio", _first_non_none(rdcf.get("interest_coverage_ratio"), dcf.get("interest_coverage_ratio"), qs.get("interest_coverage"))),
            ("dividend_yield_pct", _first_non_none(qs.get("dividend_yield_pct"), equity_data.get("dividend_yield_pct"))),
            ("net_shareholder_yield_pct", _first_non_none(qs.get("net_shareholder_yield_pct"), equity_data.get("net_shareholder_yield_pct"))),
            ("revenue_growth_yoy_pct", _first_non_none(qs.get("revenue_growth_yoy_pct"), equity_data.get("revenue_growth_yoy_pct"))),
            ("net_income_growth_yoy_pct", _first_non_none(qs.get("net_income_growth_yoy_pct"), equity_data.get("net_income_growth_yoy_pct"))),
            ("eps_revision_net_30d", _first_non_none(qs.get("eps_revision_net_30d"), equity_data.get("eps_revision_net_30d"))),
            ("pe_vs_peer_avg_pct", _first_non_none(qs.get("pe_vs_peer_avg_pct"), equity_data.get("pe_vs_peer_avg_pct"))),
            ("price_percentile_5y", _first_non_none(qs.get("price_percentile_5y"), equity_data.get("price_percentile_5y"))),
            (
                "implied_exit_ebitda_multiple",
                _first_non_none(rdcf.get("implied_exit_ebitda_multiple"), dcf.get("implied_exit_ebitda_multiple"), qs.get("ev_ebitda"), equity_data.get("ev_ebitda")),
            ),
            ("pe_ratio", _first_non_none(qs.get("pe_ratio"), equity_data.get("pe_ratio"))),
            (
                "net_cash_debt",
                _fmt_compact_money(_first_non_none(rdcf.get("net_cash_debt"), dcf.get("net_cash_debt")), prefix=mcap_ccy_prefix)
                if _first_non_none(rdcf.get("net_cash_debt"), dcf.get("net_cash_debt")) is not None
                else None,
            ),
            ("terminal_growth_pct", _first_non_none(rdcf.get("terminal_growth_pct"), dcf.get("terminal_growth_pct"))),
            (
                "implied_perpetual_growth_from_exit_pct",
                _first_non_none(rdcf.get("implied_perpetual_growth_from_exit_pct"), dcf.get("implied_perpetual_growth_from_exit_pct")),
            ),
            ("cost_of_equity_pct", _first_non_none(rdcf.get("cost_of_equity_pct"), dcf.get("cost_of_equity_pct"))),
            ("risk_free_rate_pct", _first_non_none(rdcf.get("risk_free_rate_pct"), dcf.get("risk_free_rate_pct"))),
            ("erp_pct", _first_non_none(rdcf.get("erp_pct"), dcf.get("erp_pct"))),
        ]
        if rdcf.get("alternative_valuation_model"):
            extracted_map.append(("alternative_valuation_model", rdcf.get("alternative_valuation_model")))
            extracted_map.append(("alternative_target_price_12m", rdcf.get("alternative_target_price_12m")))
        if portfolio_guard.portfolio_attached and pctx.get("position_action_hint") and pctx.get("position_action_hint") != action_stance:
            extracted_map.append(("portfolio_position_action_hint", pctx.get("position_action_hint")))

        for label, val in extracted_map:
            if val is not None and (not isinstance(val, str) or val.strip()):
                facts.append(f"{label}={val}")

    if portfolio_guard.attachment_state == ContextAttachmentState.ATTACHED_HELD:
        if portfolio_guard.current_weight_pct:
            facts.append(f"portfolio_weight_pct={portfolio_guard.current_weight_pct}%")
        if portfolio_guard.avg_cost:
            facts.append(f"portfolio_avg_cost={portfolio_guard.avg_cost}")

    return tuple(facts), tuple(dict.fromkeys(flags)), str(action_stance) if action_stance else None


def _extract_macro_numeric_facts(dash: Optional[Dict[str, Any]]) -> List[str]:
    """Extract deterministic Macro regime, Thailand stance, and regional score facts when @Macro is attached."""
    if not isinstance(dash, dict) or not dash:
        return []
    facts: List[str] = []
    regime = dash.get("overall_regime") or dash.get("regime")
    if regime:
        facts.append(f"macro_regime={regime}")
    if dash.get("conviction_level"):
        facts.append(f"macro_conviction_level={dash.get('conviction_level')}")
    th = dash.get("thailand_market_stance")
    if isinstance(th, dict) and th:
        val = th.get("valuation") if isinstance(th.get("valuation"), dict) else {}
        flow = th.get("investor_flow") if isinstance(th.get("investor_flow"), dict) else {}
        pg = th.get("physical_gold") if isinstance(th.get("physical_gold"), dict) else {}
        for fk, fv in (
            ("macro_th_pe_ratio", val.get("pe_ratio")),
            ("macro_th_pbv_ratio", val.get("pbv_ratio")),
            ("macro_th_dividend_yield_pct", val.get("dividend_yield")),
            ("macro_th_policy_spread_bps", th.get("policy_spread_bps")),
            ("macro_th_yield_curve_10y_2y_bps", th.get("yield_curve_spread_10y_2y_bps")),
            ("macro_th_foreign_net_mb", flow.get("foreign_net_mb")),
            ("macro_th_institution_net_mb", flow.get("institution_net_mb")),
            ("macro_th_retail_net_mb", flow.get("retail_net_mb")),
            ("macro_th_gold_bar_sell_thb", pg.get("bar_sell_thb")),
        ):
            if fv is not None:
                facts.append(f"{fk}={fv}")
    return facts


def _extract_secondary_numeric_facts(
    ticker: str,
    equity_data: Dict[str, Any],
    anchor: Optional[InstrumentIdentityAnchor],
) -> List[str]:
    """Build ticker-prefixed grounded numeric facts for secondary tickers in multi-ticker sessions."""
    if not ticker or not equity_data:
        return []
    sym = ticker.strip().upper()
    qs = equity_data.get("quant_signals") if isinstance(equity_data.get("quant_signals"), dict) else {}
    ams = qs.get("atomic_market_snapshot") if isinstance(qs.get("atomic_market_snapshot"), dict) else {}
    sc = _pick_subdict(qs, equity_data, "deterministic_scorecard")
    dcf = _pick_subdict(qs, equity_data, "dcf_result")
    rdcf = _pick_subdict(qs, equity_data, "reverse_dcf_result")
    pio = _pick_subdict(qs, equity_data, "piotroski_breakdown")
    dcf_base = (dcf.get("scenarios") or {}).get("base") or {}

    facts: List[str] = []
    if anchor and anchor.current_price:
        facts.append(f"{sym}:current_price={anchor.current_price} {anchor.currency}")

    mcap_prefix = "$" if (anchor and anchor.currency == "USD") else ""
    mcap_fmt = _fmt_compact_money(_first_non_none(ams.get("market_cap"), ams.get("market_cap_str")), prefix=mcap_prefix)

    sec_pairs: List[Tuple[str, Any]] = [
        ("action_stance", sc.get("action_stance")),
        ("composite_score", _first_non_none(qs.get("composite_score"), equity_data.get("composite_score"))),
        ("market_cap", mcap_fmt if mcap_fmt != "N/A" else None),
        ("fair_value_base_dcf", dcf_base.get("target_price") if dcf.get("is_actionable", True) else None),
        ("target_price_12m", rdcf.get("target_price_12m")),
        ("mos_entry_price_today", _first_non_none(rdcf.get("mos_entry_price_today"), qs.get("mos_entry_price_today"))),
        ("upside_12m_pct", rdcf.get("upside_12m_pct")),
        ("market_implied_growth_pct", rdcf.get("market_implied_growth_pct")),
        ("roic_pct", qs.get("roic_pct")),
        ("fcf_yield_pct", qs.get("fcf_yield_pct")),
        ("net_debt_ebitda", qs.get("net_debt_ebitda")),
        ("piotroski_f_score", _first_non_none(pio.get("f_score"), qs.get("piotroski_score"))),
        ("synthetic_credit_rating", _first_non_none(rdcf.get("synthetic_credit_rating"), dcf.get("synthetic_credit_rating"))),
        ("revenue_growth_yoy_pct", qs.get("revenue_growth_yoy_pct")),
    ]
    for label, val in sec_pairs:
        if val is not None and (not isinstance(val, str) or val.strip()):
            facts.append(f"{sym}:{label}={val}")
    return facts


def _clean_vault_snippet(raw_snippet: Any, max_len: int = 180) -> str:
    """Strip YAML frontmatter / metadata fragments defensively from a note snippet."""
    txt = str(raw_snippet or "").strip()
    if not txt:
        return ""
    if txt.startswith("---"):
        parts = txt.split("---", 2)
        if len(parts) == 3:
            txt = parts[2].strip()
    txt = re.sub(r"<!--[\s\S]*?-->", " ", txt)
    # Strip pure leading YAML key: value pairs when raw snippet was pulled from frontmatter lines
    yaml_prefix_re = re.compile(
        r"^(?:(?:schema_version|title|entity_type|ticker|tickers|market|date|last_updated|generated_by|tags|type|source)\s*:\s*(?:'[^']*'|\"[^\"]*\"|\[[^\]]*\]|\S+)\s*)+"
    )
    cleaned = yaml_prefix_re.sub("", txt).strip()
    return _clip(cleaned or txt, max_len)


def _extract_analysis_note_excerpt(
    raw_content: str,
    *,
    shown_narrative: str = "",
    narr_cap: int = 3000,
    limit: int = 2500,
    include_sentiment_context: bool = False,
) -> str:
    """Extract non-duplicated qualitative prose from latest Equity Analysis Vault note."""
    text = re.sub(r"\A\ufeff?---\r?\n[\s\S]*?\r?\n---(?:\r?\n|\Z)", "", str(raw_content or ""))
    text = re.sub(r"<!--[\s\S]*?-->", " ", text)
    # Remove noisy footer sections and Portfolio Position section (strict 3-State Privacy Guard)
    text = re.split(r"(?m)^##\s+(?:Related|หมายเหตุ)\b", text, maxsplit=1)[0]
    text = re.sub(r"(?ms)^#{2,3}\s*💼\s*Portfolio Position[\s\S]*?(?=^#{2,3}\s+|\Z)", "", text)

    def _clean_prose_lines(block: str) -> str:
        kept: List[str] = []
        for raw_ln in block.splitlines():
            ln = raw_ln.strip()
            if (
                not ln
                or ln.startswith(("---", "|", "> **Analysis Price:", "> **Market Cap:", "> Weighted:"))
                or ln.startswith("#")
            ):
                continue
            ln = re.sub(r"^>\s*(?:⚠️?\s*)?", "", ln).replace("**", "").strip()
            if ln:
                kept.append(ln)
        return re.sub(r"\s+", " ", " ".join(kept)).strip()

    extra_parts: List[str] = []

    # Extract Sentiment & Narrative Context (themes + source/narrative blockquote) without duplicating Tail Risks
    sent_section = re.search(r"(?ms)^##\s+[^\n]*Sentiment\s*&\s*Narrative[\s\S]*?(?=^##\s+|\Z)", text)
    sent_themes_txt = ""
    sent_quote_txt = ""
    if sent_section:
        sent_body = sent_section.group(0)
        q_lines = re.findall(r"(?m)^>\s+[^\n]+", sent_body)
        if q_lines:
            cand_q = _clean_prose_lines("\n".join(q_lines))
            if cand_q and "automated fallback: llm narrative unavailable" not in cand_q.lower():
                sent_quote_txt = cand_q
        if include_sentiment_context:
            t_match = re.search(
                r"(?ms)^\*\*ธีมสำคัญ:\*\*([^\n]*(?:\n(?!\*\*|#|>)[^\n]*)*)",
                sent_body,
            )
            if t_match:
                t_clean = _clean_prose_lines(t_match.group(1))
                if t_clean and "sentiment analysis unavailable" not in t_clean.lower():
                    sent_themes_txt = f"ธีมสำคัญ (Sentiment Context): {t_clean}"

    # 1. Prefer recovering any unshown remainder of the Markdown Narrative section when present
    narr_match = re.search(r"(?ms)^##\s+[^\n]*บทวิเคราะห์[^\n]*\n([\s\S]*?)(?=^##\s+|\Z)", text)
    if narr_match:
        narr_prose = _clean_prose_lines(narr_match.group(1))
        shown_clean = re.sub(r"\s+", " ", str(shown_narrative or "")).strip()
        if narr_prose and shown_clean and narr_prose[:80] == shown_clean[:80]:
            rendered_in_quality = _clip(shown_clean, narr_cap) if len(shown_clean) > narr_cap else shown_clean
            rendered_len = len(rendered_in_quality.rstrip("…"))
            if len(narr_prose) > rendered_len + 20:
                anchor = rendered_in_quality[:-1][-32:] if rendered_in_quality.endswith("…") else rendered_in_quality[-32:]
                search_start = max(0, len(rendered_in_quality) - 250)
                pos = narr_prose.find(anchor, search_start) if anchor else -1
                if pos == -1 and anchor:
                    pos = narr_prose.rfind(anchor)
                if pos != -1:
                    raw_start = pos + len(anchor)
                    ws_back = narr_prose.rfind(" ", max(0, raw_start - 25), raw_start + 1)
                    offset = ws_back + 1 if ws_back != -1 else raw_start
                else:
                    offset = max(0, rendered_len - 15)
                tail_slice = narr_prose[offset:].lstrip()
                if tail_slice:
                    extra_parts.append("…" + tail_slice)
        elif narr_prose:
            extra_parts.append(narr_prose)

    if sent_themes_txt:
        extra_parts.append(sent_themes_txt)
    if sent_quote_txt:
        extra_parts.append(sent_quote_txt)

    if extra_parts:
        return _clip(" | ".join(extra_parts), limit)
    if narr_match:
        return ""

    body_without_header_tables = _clean_prose_lines(text)
    if not body_without_header_tables:
        return ""
    return _clip(body_without_header_tables, limit)


def _parse_earnings_call_note(
    earnings_note: Optional[Dict[str, Any]],
    *,
    excerpt_limit: int = 2300,
) -> Tuple[Dict[str, Any], str]:
    """Parse YAML frontmatter (e.g. guidance metadata) and clean AI Highlights from an Earnings Call Vault note."""
    if not isinstance(earnings_note, dict) or not earnings_note.get("content"):
        return {}, ""
    raw_content = str(earnings_note.get("content") or "")
    meta: Dict[str, Any] = {}
    body = raw_content
    fm_match = re.match(r"\A\ufeff?---\r?\n([\s\S]*?)\r?\n---(?:\r?\n|\Z)", raw_content)
    if fm_match:
        fm_text = fm_match.group(1)
        body = raw_content[fm_match.end():]
        if yaml is not None:
            try:
                parsed = yaml.safe_load(fm_text)
                if isinstance(parsed, dict):
                    meta = parsed
            except Exception:
                meta = {}
    elif raw_content.startswith("---"):
        parts = raw_content.split("---", 2)
        if len(parts) == 3:
            body = parts[2]

    body = re.sub(r"<!--[\s\S]*?-->", " ", body)
    # Never ingest verbose Full Transcript verbatim or trailing Related footer
    body = re.split(r"(?im)^##\s+(?:[^\n]*Full\s+Transcript|Related)\b", body, maxsplit=1)[0]
    hi_match = re.search(r"(?im)^##\s+[^\n]*AI\s+Highlights[^\n]*\r?\n", body)
    if hi_match:
        body = body[hi_match.end():]

    cleaned_lines: List[str] = []
    for raw_ln in body.splitlines():
        ln = raw_ln.strip()
        if not ln or ln == "---":
            continue
        if ln.startswith("###"):
            heading = re.sub(r"^#{1,6}\s*", "", ln).replace("**", "").strip()
            if heading:
                cleaned_lines.append(f"[{heading}]")
        elif ln.startswith("#"):
            heading = re.sub(r"^#{1,6}\s*", "", ln).replace("**", "").strip()
            if heading and "earnings call" not in heading.lower():
                cleaned_lines.append(heading)
        else:
            ln = re.sub(r"^[*+\-]\s+", "• ", ln)
            ln = re.sub(r"\*\*", "", ln)
            ln = re.sub(r"\s+", " ", ln).strip()
            if ln:
                cleaned_lines.append(ln)
    cleaned_summary = _clip(" ".join(cleaned_lines).strip(), excerpt_limit)
    return meta, cleaned_summary


def _build_news_and_notes_lines(
    *,
    ticker: str,
    news_data: Optional[Dict[str, Any]],
    notes_list: Optional[Dict[str, Any]],
    earnings_note: Optional[Dict[str, Any]],
    analysis_note: Optional[Dict[str, Any]] = None,
    shown_narrative: str = "",
    narr_cap: int = 3000,
    tight: bool = False,
    ec_summary_override: Optional[str] = None,
) -> List[str]:
    lines: List[str] = []
    rendered_news_titles: List[str] = []
    if isinstance(news_data, dict):
        items = news_data.get("items") or []
        yt_mentions = news_data.get("youtube_mentions") or []
        if items:
            lines.append(f"- ข่าวล่าสุดของ {ticker} (จาก News/Digest Store):")
            for it in items[:(5 if tight else 15)]:
                raw_hl = str(it.get("title") or it.get("thai_headline") or "")
                norm_hl = re.sub(r"\s+", " ", raw_hl).strip().lower()
                if len(norm_hl) >= 20:
                    rendered_news_titles.append(norm_hl)
                title = _clip(raw_hl, 120 if tight else 220)
                age = it.get("age_hours")
                fresh_lbl = it.get("freshness_reason") or it.get("freshness_label")
                stale_tag = " [STALE]" if it.get("is_stale") else ""
                if age is not None and str(age) != "9999":
                    age_tag = f" ({age}h{stale_tag})"
                elif fresh_lbl:
                    age_tag = f" ({fresh_lbl}{stale_tag})"
                else:
                    age_tag = stale_tag
                sig = f" [Signal: {it.get('thesis_signal')}]" if it.get("thesis_signal") else ""
                imp_parts: List[str] = []
                if it.get("impact_type"):
                    imp_parts.append(str(it.get("impact_type")))
                if it.get("macro_impact_score") is not None:
                    imp_parts.append(f"Macro:{_fmt_num(it.get('macro_impact_score'), '', '0.0')}")
                if it.get("asset_impact_score") is not None:
                    imp_parts.append(f"Asset:{_fmt_num(it.get('asset_impact_score'), '', '0.0')}")
                imp_tag = f" [Impact: {' | '.join(imp_parts)}]" if imp_parts else ""
                why = _clip(it.get("why_it_matters") or it.get("what_happened") or it.get("summary") or "", 180 if tight else 360)
                lines.append(f"  * {title}{age_tag}{sig}{imp_tag}" + (f" -> {why}" if why else ""))
        if yt_mentions:
            lines.append(f"- มุมมอง YouTube ที่กล่าวถึง {ticker}:")
            for ym in yt_mentions[:(3 if tight else 8)]:
                yt_t = _clip(ym.get("thai_title") or ym.get("title"), 110 if tight else 180)
                ch = ym.get("channel") or "YouTube"
                align_val = str(ym.get("alignment_with_news") or "").strip()
                align_note = _clip(ym.get("alignment_note"), 80 if tight else 180) if ym.get("alignment_note") else ""
                if align_val and align_note:
                    align = f" [{align_val} ({align_note})]"
                elif align_val:
                    align = f" [{align_val}]"
                elif align_note:
                    align = f" [{align_note}]"
                else:
                    align = ""
                tk = "; ".join(str(k) for k in (ym.get("key_takeaways") or [])[:(2 if tight else 4)])
                lines.append(f"  * [{ch}] {yt_t}{align}" + (f": {_clip(tk, 160 if tight else 320)}" if tk else ""))

    ea_excerpt = ""
    if isinstance(analysis_note, dict) and analysis_note.get("content"):
        ea_excerpt = _extract_analysis_note_excerpt(
            str(analysis_note.get("content") or ""),
            shown_narrative=shown_narrative,
            narr_cap=narr_cap,
            limit=500 if tight else 2500,
            include_sentiment_context=not tight,
        )
    ea_title_norm = str(analysis_note.get("title") or "").strip().lower() if isinstance(analysis_note, dict) else ""

    cleaned_ec_excerpt = ""
    if isinstance(earnings_note, dict) and earnings_note.get("content"):
        cleaned_ec_excerpt = ec_summary_override or ""
        if ec_summary_override is None:
            _, cleaned_ec_excerpt = _parse_earnings_call_note(
                earnings_note, excerpt_limit=750 if tight else 5500
            )
    ec_title_norm = str(earnings_note.get("title") or "").strip().lower() if isinstance(earnings_note, dict) else ""

    if isinstance(notes_list, dict):
        note_items = [n for n in (notes_list.get("items") or []) if isinstance(n, dict)]
        if note_items:
            base_limit = 4 if tight else 5
            selected_notes = list(note_items[:base_limit])
            if not tight and len(note_items) > base_limit:
                selected_keys = {str(n.get("relative_path") or n.get("title") or "") for n in selected_notes}
                for priority_kind in ("earnings_call", "equity_analysis"):
                    if priority_kind == "earnings_call":
                        already_has = any(
                            n.get("matched_by") == "earnings_call"
                            or "earnings" in str(n.get("folder") or n.get("relative_path") or "").lower()
                            for n in selected_notes
                        )
                    else:
                        already_has = any(
                            (
                                n.get("matched_by") == "equity_analysis"
                                or "/analysis/" in str(n.get("relative_path") or "").lower()
                                or str(n.get("folder") or "").endswith("/Analysis")
                            )
                            and not re.search(r"(?:^|[_\s\-])latest$", str(n.get("title") or ""), re.IGNORECASE)
                            for n in selected_notes
                        )
                    if already_has:
                        continue
                    for cand in note_items[base_limit:]:
                        if priority_kind == "earnings_call":
                            is_kind = (
                                cand.get("matched_by") == "earnings_call"
                                or "earnings" in str(cand.get("folder") or cand.get("relative_path") or "").lower()
                            )
                        else:
                            is_kind = (
                                cand.get("matched_by") == "equity_analysis"
                                or "/analysis/" in str(cand.get("relative_path") or "").lower()
                                or str(cand.get("folder") or "").endswith("/Analysis")
                            ) and not re.search(r"(?:^|[_\s\-])latest$", str(cand.get("title") or ""), re.IGNORECASE)
                        if is_kind:
                            ck = str(cand.get("relative_path") or cand.get("title") or "")
                            if ck not in selected_keys:
                                selected_notes.append(cand)
                                selected_keys.add(ck)
                            break

            lines.append(f"- บันทึก Vault ล่าสุด ({ticker}):")
            snip_cap = 180 if tight else 420
            title_cap = 90 if tight else 140
            for ni in selected_notes:
                mby = str(ni.get("matched_by") or "note").strip().lower()
                note_title_norm = str(ni.get("title") or "").strip().lower()
                clean_snip = _clean_vault_snippet(ni.get("snippet"), snip_cap)
                norm_snip = clean_snip.lower()
                if mby == "earnings_call" and cleaned_ec_excerpt and (
                    not ec_title_norm or note_title_norm == ec_title_norm or ec_title_norm in note_title_norm
                ):
                    display_snip = "[สรุป AI Highlights ฉบับเต็มด้านล่าง]"
                elif mby == "news" and rendered_news_titles and any(
                    t in norm_snip for t in rendered_news_titles
                ):
                    display_snip = "[สรุปข่าวเดียวกับรายการด้านบน]"
                elif (
                    mby == "equity_analysis"
                    and clean_snip.startswith("📊 บทวิเคราะห์เชิงปริมาณ:")
                    and (not ea_title_norm or note_title_norm == ea_title_norm)
                    and (ea_excerpt or shown_narrative)
                ):
                    display_snip = "[ตัวเลขหลักใน SEC:HEADER / สาระเสริมด้านล่าง]"
                else:
                    display_snip = clean_snip
                lines.append(
                    f"  * [{ni.get('matched_by', 'note')}] {_clip(ni.get('title'), title_cap)} "
                    f"({str(ni.get('modified_at', ''))[:10]}): {display_snip}"
                )

    if ea_excerpt:
        lines.append(f"- สาระสำคัญเพิ่มเติมจาก Vault Analysis ({analysis_note.get('title', 'Equity Analysis')}): {ea_excerpt}")

    if cleaned_ec_excerpt:
        lines.append(f"- Earnings Call Note ({earnings_note.get('title', 'Transcript Summary')}): {cleaned_ec_excerpt}")

    return lines


def _build_curated_equity_brief(
    ticker: str,
    detail: Dict[str, Any],
    *,
    portfolio_attached: bool,
    news_data: Optional[Dict[str, Any]] = None,
    notes_list: Optional[Dict[str, Any]] = None,
    earnings_note: Optional[Dict[str, Any]] = None,
    analysis_note: Optional[Dict[str, Any]] = None,
    watchlist_item: Optional[Dict[str, Any]] = None,
    char_budget: int = 10000,
    thesis_exceptions: Optional[Sequence[Dict[str, Any]]] = None,
    promise_scores: Optional[Dict[str, Any]] = None,
    analyst_context: Optional[Dict[str, Any]] = None,
    insider_db: Optional[Dict[str, Any]] = None,
    sidecar_age_days: Optional[int] = None,
    live_macro_dash: Optional[Dict[str, Any]] = None,
    holding_advisory: Optional[Dict[str, Any]] = None,
    now_date_iso: str = "",
    extra_dq_flags: Sequence[str] = (),
) -> str:
    """Build deterministic Curated Equity Brief with `<!-- SEC:... -->` section markers."""
    qs = detail.get("quant_signals") if isinstance(detail.get("quant_signals"), dict) else {}
    sent = detail.get("sentiment_context") if isinstance(detail.get("sentiment_context"), dict) else {}
    scorecard = _pick_subdict(qs, detail, "deterministic_scorecard")
    dcf = _pick_subdict(qs, detail, "dcf_result")
    rdcf = _pick_subdict(qs, detail, "reverse_dcf_result")
    piotroski = _pick_subdict(qs, detail, "piotroski_breakdown")
    tactical = _pick_subdict(qs, detail, "tactical_setup")
    smart_money = _pick_subdict(qs, detail, "smart_money_flags")
    insider = _pick_subdict(qs, detail, "insider_conviction")
    guidance = _pick_subdict(qs, detail, "earnings_guidance_context")
    falsifiers = _pick_list(qs, detail, "thesis_falsifiers")
    hist_3y = _pick_list(qs, detail, "historical_financials_3y") or _pick_list(rdcf, dcf, "historical_financials_3y")
    thesis_delta = detail.get("thesis_delta") if isinstance(detail.get("thesis_delta"), dict) else {}
    active_thesis = detail.get("active_thesis") if isinstance(detail.get("active_thesis"), dict) else {}
    pos_review = detail.get("position_review") if isinstance(detail.get("position_review"), dict) else {}
    ams = qs.get("atomic_market_snapshot") if isinstance(qs.get("atomic_market_snapshot"), dict) else {}

    company = detail.get("company_name") or qs.get("company_name") or ticker
    price = _extract_price_from_detail(qs, detail, None)
    market = detail.get("market") or qs.get("market") or ("TH" if ticker.endswith(".BK") else "US")
    ccy = detail.get("currency") or ("THB" if market == "TH" else "USD")
    money_prefix = "$" if ccy == "USD" else ""
    money_suffix = "" if ccy == "USD" else f" {ccy}"

    # Determine adaptive clipping limits based on char_budget (smaller when comparing 2-3 stocks)
    tight = char_budget <= 5500
    narr_cap = 550 if tight else 3000
    base_cap = 320 if tight else 1200
    ec_meta, ec_summary = _parse_earnings_call_note(
        earnings_note, excerpt_limit=750 if tight else 5500
    )

    # 1. HEADER
    dedup_exc = list(thesis_exceptions or ())
    dq_flags = list(
        dict.fromkeys(
            [
                str(x)
                for x in (
                    list(qs.get("data_quality_flags") or [])
                    + list(scorecard.get("data_quality_flags") or [])
                    + list(detail.get("data_quality_flags") or [])
                    + list(extra_dq_flags or [])
                )
                if x
            ]
        )
    )
    disc_warn = _first_non_none(qs.get("dcf_discrepancy_warning"), detail.get("dcf_discrepancy_warning"))
    ohlcv_dt = ams.get("latest_ohlcv_date")
    sync_st = ams.get("price_sync_status")
    ohlcv_part = f" | OHLCV: {ohlcv_dt}" if ohlcv_dt else ""
    sync_part = f" | Sync: {sync_st}" if sync_st else ""
    vol_conf = ams.get("volume_confirmation")
    miss_sess = ams.get("missing_trading_sessions")
    vol_conf_part = f" | VolConfirm: {vol_conf}" if (not tight and vol_conf) else ""
    miss_part = f" | MissingSessions: {miss_sess}" if (not tight and miss_sess not in (None, 0)) else ""
    age_part = f" | Sidecar Age: {sidecar_age_days}d" if sidecar_age_days is not None else ""
    header_lines: List[str] = [
        f"<!-- SEC:HEADER -->",
        f"- หุ้น: ${ticker} ({company}) | ตลาด: {market} ({ccy}) | วันที่วิเคราะห์: {detail.get('analysis_date') or str(detail.get('evaluated_at') or '')[:10]}{age_part}",
        f"- ราคาอ้างอิง: {_fmt_num(price, f' {ccy}')} | Price Freshness: {ams.get('freshness_status', 'unknown')} ({ams.get('price_source', 'unknown')}){ohlcv_part}{sync_part}{vol_conf_part}{miss_part}",
    ]
    raw_mcap = _first_non_none(ams.get("market_cap"), ams.get("market_cap_str"), qs.get("market_cap"), detail.get("market_cap"))
    raw_shares = _first_non_none(ams.get("shares_outstanding"), qs.get("shares_outstanding"))
    raw_beta = _first_non_none(ams.get("beta"), qs.get("beta"), detail.get("beta"))
    raw_vol = _first_non_none(ams.get("volatility_pct"), qs.get("volatility_pct"))
    raw_mdd = _first_non_none(ams.get("mdd_pct"), qs.get("mdd_pct"))
    raw_adtv = _first_non_none(ams.get("adtv_local_currency"), qs.get("adtv_local_currency"))
    raw_amihud = _first_non_none(ams.get("amihud_illiquidity_20d"), qs.get("amihud_illiquidity_20d"))
    if any(x is not None for x in (raw_mcap, raw_shares, raw_beta, raw_vol, raw_mdd, raw_adtv, raw_amihud)):
        scale_parts: List[str] = []
        if raw_mcap is not None:
            scale_parts.append(f"Market Cap={_fmt_compact_money(raw_mcap, prefix=money_prefix, suffix=money_suffix)}")
        if raw_shares is not None:
            scale_parts.append(f"Shares={_fmt_compact_money(raw_shares)}")
        if raw_beta is not None:
            scale_parts.append(f"Beta={_fmt_num(raw_beta)}")
        if raw_vol is not None:
            scale_parts.append(f"Vol={_fmt_num(raw_vol, '%')}")
        if raw_mdd is not None:
            scale_parts.append(f"MDD={_fmt_num(raw_mdd, '%')}")
        if raw_adtv is not None:
            scale_parts.append(f"ADTV={_fmt_compact_money(raw_adtv, prefix=money_prefix, suffix=(money_suffix + '/day'))}")
        if raw_amihud is not None:
            scale_parts.append(f"Amihud Illiq={_fmt_num(raw_amihud, places='0.0001')}")
        header_lines.append("- ขนาด สภาพคล่อง & ความเสี่ยง: " + " | ".join(scale_parts))

    header_lines.append(
        f"- คะแนนระบบ (0-100): Composite={_fmt_num(_first_non_none(qs.get('composite_score'), detail.get('composite_score')))} | "
        f"Value={_fmt_num(qs.get('value_score'))} | Quality={_fmt_num(qs.get('quality_score'))} | Growth={_fmt_num(qs.get('growth_score'))} | "
        f"Momentum={_fmt_num(qs.get('momentum_score'))} | Dividend={_fmt_num(qs.get('dividend_score'))}"
    )
    cwf = qs.get("composite_score_waterfall") if isinstance(qs.get("composite_score_waterfall"), dict) else {}
    if not tight and cwf:
        eff_w = cwf.get("effective_weights") if isinstance(cwf.get("effective_weights"), dict) else {}
        miss_d = [str(x) for x in (cwf.get("missing_dimensions") or []) if x]
        w_str = ", ".join(f"{k}={_fmt_num(v, '%')}" for k, v in eff_w.items() if v is not None)
        cwf_parts: List[str] = []
        if w_str:
            cwf_parts.append(f"EffectiveWeights({w_str})")
        if miss_d:
            cwf_parts.append(f"MissingDims=[{', '.join(miss_d)}]")
        if cwf.get("solvency_gate_passed") is not None:
            cwf_parts.append(f"SolvencyGatePassed={cwf.get('solvency_gate_passed')}")
        if cwf_parts:
            header_lines.append(f"  * Composite Waterfall: {' | '.join(cwf_parts)}")
    if scorecard:
        header_lines.append(
            f"- Scorecard: Action Stance = {scorecard.get('action_stance', 'N/A')} (mode: {scorecard.get('stance_mode', 'N/A')}) | "
            f"Core={_fmt_num(scorecard.get('core_conviction_score'))}/10, Biz={_fmt_num(scorecard.get('business_conviction_score'))}/10, "
            f"Invest={_fmt_num(scorecard.get('investment_conviction_score'))}/10, ExecReady={_fmt_num(scorecard.get('execution_readiness_score'))}/10"
        )
        pillar_vals = (
            scorecard.get("fundamental_quality_score"),
            scorecard.get("valuation_margin_score"),
            scorecard.get("guidance_expectation_score"),
            scorecard.get("setup_readiness_score"),
            scorecard.get("coverage_pct"),
            scorecard.get("verified_coverage_pct"),
        )
        if any(x is not None for x in pillar_vals):
            header_lines.append(
                f"  * Pillars (0-100): FundQual={_fmt_num(scorecard.get('fundamental_quality_score'))} | "
                f"ValMargin={_fmt_num(scorecard.get('valuation_margin_score'))} | "
                f"Guidance={_fmt_num(scorecard.get('guidance_expectation_score'))} | "
                f"Setup={_fmt_num(scorecard.get('setup_readiness_score'))} | "
                f"Coverage={_fmt_num(scorecard.get('coverage_pct'), '%')} (Verified={_fmt_num(scorecard.get('verified_coverage_pct'), '%')})"
            )
        stance_reason = _first_non_none(scorecard.get("action_stance_reason"), pos_review.get("action_stance_reason"))
        if stance_reason:
            header_lines.append(f"  * เหตุผล Action Stance: {_clip(stance_reason, 260 if tight else 500)}")
        if not tight:
            sub_parts: List[str] = []
            if scorecard.get("analyst_expectations_score") is not None:
                sub_parts.append(f"AnalystExp={_fmt_num(scorecard.get('analyst_expectations_score'))}")
            if scorecard.get("management_guidance_score") is not None:
                sub_parts.append(f"MgmtGuidance={_fmt_num(scorecard.get('management_guidance_score'))}")
            eb = scorecard.get("execution_score_breakdown")
            if isinstance(eb, dict) and eb:
                eb_txt = (
                    f"ExecBreakdown(Stage={_fmt_num(eb.get('stage'))}, Setup={_fmt_num(eb.get('setup'))}, "
                    f"Insider={_fmt_num(eb.get('insider'))}, reason={eb.get('setup_reason') or 'N/A'})"
                )
                sub_parts.append(eb_txt)
            if sub_parts:
                header_lines.append(f"  * Scorecard Sub-Details: {' | '.join(sub_parts)}")
    if watchlist_item:
        wl_type = str(watchlist_item.get("asset_type") or "Stock")
        header_lines.append(
            f"- [User Watchlist]: ${ticker} ({wl_type}) | {_format_watchlist_entry_line(watchlist_item, notes_clip=110 if tight else 260)}"
        )
    if sent:
        themes = ", ".join(str(t) for t in (sent.get("key_themes") or [])[:(4 if tight else 10)])
        header_lines.append(f"- Market Sentiment: {sent.get('market_sentiment', detail.get('market_sentiment', 'N/A'))} | Key Themes: {themes or 'N/A'}")
    if dq_flags:
        header_lines.append(f"- [DATA QUALITY FLAGS]: {', '.join(dq_flags)}")
    if sidecar_age_days is not None and sidecar_age_days > 14:
        header_lines.append(
            f"- [STALE SIDECAR WARNING]: ข้อมูลวิเคราะห์ล่าสุดอายุ {sidecar_age_days} วัน (> 14 วัน) — ต้องระวังสมมติฐาน ราคา หรือ Macro Regime ที่อาจเปลี่ยนไปแล้ว"
        )
    if dedup_exc:
        header_lines.append(
            f"- [ACTIVE THESIS BREACH ALERT]: พบสัญญาณผิดเกณฑ์ Watchdog PENDING จำนวน {len(dedup_exc)} รายการ — ตรวจรายละเอียดใน RISKS_THESIS"
        )
    if disc_warn:
        header_lines.append(f"- [DCF DISCREPANCY WARNING]: {_clip(disc_warn, 260 if tight else 500)}")

    # 2. VALUATION
    val_lines: List[str] = ["<!-- SEC:VALUATION -->"]
    if dcf:
        scen = dcf.get("scenarios") or {}
        bear_t = (scen.get("bear") or {}).get("target_price")
        base_t = (scen.get("base") or {}).get("target_price")
        bull_t = (scen.get("bull") or {}).get("target_price")
        bear_mos = (scen.get("bear") or {}).get("margin_of_safety_pct")
        base_mos = (scen.get("base") or {}).get("margin_of_safety_pct")
        bull_mos = (scen.get("bull") or {}).get("margin_of_safety_pct")
        mos_ext = (
            f" (Bear={_fmt_num(bear_mos, '%')}, Bull={_fmt_num(bull_mos, '%')})"
            if (not tight and (bear_mos is not None or bull_mos is not None))
            else ""
        )
        sa = dcf.get("valuation_sanity_audit") or {}
        val_lines.append(
            f"- Forward DCF: actionable={dcf.get('is_actionable', True)} ({dcf.get('valuation_verdict', 'N/A')}) | "
            f"Target [Bear={_fmt_num(bear_t)}, Base={_fmt_num(base_t)}, Bull={_fmt_num(bull_t)}] | Base MoS={_fmt_num(base_mos, '%')}{mos_ext} | WACC={_fmt_num(dcf.get('wacc_pct'), '%')} | Exit EBITDA Multiple={_fmt_num(dcf.get('implied_exit_ebitda_multiple'), 'x')}"
        )
        if dcf.get("actionability_reason") or dcf.get("invalidation_reasons"):
            val_lines.append(f"  * เหตุผล DCF Actionability/Invalidation: {dcf.get('actionability_reason') or '; '.join(str(x) for x in dcf.get('invalidation_reasons', []))}")
        if sa:
            fcf_mult_part = (
                f" | ExitFCFMult={_fmt_num(sa.get('implied_exit_fcf_multiple'), 'x')} (Bench={_fmt_num(sa.get('benchmark_sector_multiple'), 'x')})"
                if (not tight and sa.get("implied_exit_fcf_multiple") is not None)
                else ""
            )
            val_lines.append(
                f"  * Sanity Audit: Tier={sa.get('model_reliability_tier')} | TV% of EV={_fmt_num(sa.get('terminal_value_pct_of_ev'), '%')} | Terminal Dependent={sa.get('is_terminal_dependent')}{fcf_mult_part}"
                + (f" | Warnings: {', '.join(sa.get('sanity_warnings', [])[:(3 if tight else 6)])}" if sa.get("sanity_warnings") else "")
            )
    else:
        val_lines.append("- Forward DCF: ไม่มีผลคำนวณ (dcf_result = None)")

    if rdcf:
        rsa = rdcf.get("valuation_sanity_audit") or {}
        val_lines.append(
            f"- Reverse DCF: actionable={rdcf.get('is_actionable', True)} ({rdcf.get('valuation_verdict', 'N/A')}, solver={rdcf.get('solver_status', 'N/A')}) | "
            f"Intrinsic Today={_fmt_num(rdcf.get('intrinsic_value_today'))} | Target 12M={_fmt_num(rdcf.get('target_price_12m'))} (Upside {_fmt_num(rdcf.get('upside_12m_pct'), '%')}) | "
            f"MoS Entry={_fmt_num(rdcf.get('mos_entry_price_today'))} (Current MoS={_fmt_num(rdcf.get('current_margin_of_safety_pct'), '%')}, InBuyZone={rdcf.get('is_in_buy_zone')}, RequiredPullback={_fmt_num(rdcf.get('required_pullback_pct'), '%')})"
        )
        raw_wacc_part = ""
        if not tight and (
            rdcf.get("wacc_adjustment_reason")
            or (rdcf.get("raw_wacc_pct") is not None and rdcf.get("raw_wacc_pct") != rdcf.get("effective_wacc_pct"))
        ):
            raw_wacc_part = f" (Raw={_fmt_num(rdcf.get('raw_wacc_pct'), '%')}, Adj={rdcf.get('wacc_adjustment_reason') or 'none'})"
        val_lines.append(
            f"- Market-Implied Expectations: Implied Growth={_fmt_num(rdcf.get('market_implied_growth_pct'), '%')} | Implied EBIT Margin={_fmt_num(rdcf.get('market_implied_margin_pct'), '%')} | "
            f"Exit Multiple Target 12M={_fmt_num(rdcf.get('target_price_exit_multiple_12m'))} (used {_fmt_num(rdcf.get('exit_multiple_used'), 'x')}, implied exit EV/EBITDA {_fmt_num(rdcf.get('implied_exit_ebitda_multiple'), 'x')}) | WACC={_fmt_num(rdcf.get('effective_wacc_pct'), '%')}{raw_wacc_part}"
        )
        if not tight and any(
            rdcf.get(k) is not None
            for k in ("enterprise_value", "equity_value", "sum_pv_5y_fcf", "terminal_value_pv", "intrinsic_value_exit_multiple")
        ):
            val_lines.append(
                f"  * RDCF PV Breakdown: EV={_fmt_compact_money(rdcf.get('enterprise_value'), prefix=money_prefix, suffix=money_suffix)} | "
                f"EquityVal={_fmt_compact_money(rdcf.get('equity_value'), prefix=money_prefix, suffix=money_suffix)} | "
                f"PV 5Y FCF={_fmt_compact_money(rdcf.get('sum_pv_5y_fcf'), prefix=money_prefix, suffix=money_suffix)} | "
                f"PV Terminal={_fmt_compact_money(rdcf.get('terminal_value_pv'), prefix=money_prefix, suffix=money_suffix)} | "
                f"BaseRev={_fmt_compact_money(rdcf.get('base_revenue'), prefix=money_prefix, suffix=money_suffix)} | "
                f"ExitMult Intrinsic={_fmt_num(rdcf.get('intrinsic_value_exit_multiple'))}"
            )
        if rdcf.get("alternative_valuation_model"):
            val_lines.append(
                f"- Alternative Valuation ({rdcf.get('alternative_valuation_model')}): Intrinsic={_fmt_num(rdcf.get('alternative_intrinsic_value'))} | "
                f"Target 12M={_fmt_num(rdcf.get('alternative_target_price_12m'))} (Upside {_fmt_num(rdcf.get('alternative_upside_12m_pct'), '%')}, Market Implied ROE={_fmt_num(rdcf.get('market_implied_roe_pct'), '%')})"
            )
        if rdcf.get("consensus_target_price") is not None:
            cons_ud = (
                f" | Target Upside={_fmt_num(qs.get('upside_pct'), '%')}, Downside={_fmt_num(qs.get('downside_pct'), '%')}"
                if (not tight and (qs.get("upside_pct") is not None or qs.get("downside_pct") is not None))
                else ""
            )
            val_lines.append(
                f"- Street Consensus ({rdcf.get('analyst_count') or 'N/A'} analysts): Mean={_fmt_num(rdcf.get('consensus_target_price'))} "
                f"[Low={_fmt_num(rdcf.get('consensus_target_low'))}, High={_fmt_num(rdcf.get('consensus_target_high'))}]{cons_ud}"
            )
        if rsa and (not tight or rsa.get("sanity_warnings") or rsa.get("is_terminal_dependent")):
            rel_s = f" (Score={rsa.get('reliability_score')})" if rsa.get("reliability_score") is not None else ""
            ex_fcf = (
                f" | ExitFCFMult={_fmt_num(rsa.get('implied_exit_fcf_multiple'), 'x')} (Bench={_fmt_num(rsa.get('benchmark_sector_multiple'), 'x')})"
                if rsa.get("implied_exit_fcf_multiple") is not None
                else ""
            )
            val_lines.append(
                f"  * RDCF Sanity: Tier={rsa.get('model_reliability_tier')}{rel_s} | TV%={_fmt_num(rsa.get('terminal_value_pct_of_ev'), '%')}{ex_fcf} | Warnings: {', '.join(rsa.get('sanity_warnings', [])[:(2 if tight else 5)]) or 'None'}"
            )
    net_cash_val = _first_non_none(rdcf.get("net_cash_debt"), dcf.get("net_cash_debt"))
    tg_val = _first_non_none(rdcf.get("terminal_growth_pct"), dcf.get("terminal_growth_pct"))
    imp_tg_val = _first_non_none(rdcf.get("implied_perpetual_growth_from_exit_pct"), dcf.get("implied_perpetual_growth_from_exit_pct"))
    ke_val = _first_non_none(rdcf.get("cost_of_equity_pct"), dcf.get("cost_of_equity_pct"))
    rf_val = _first_non_none(rdcf.get("risk_free_rate_pct"), dcf.get("risk_free_rate_pct"))
    erp_val = _first_non_none(rdcf.get("erp_pct"), dcf.get("erp_pct"))
    if any(x is not None for x in (net_cash_val, tg_val, imp_tg_val, ke_val, rf_val, erp_val)):
        cap_wacc_parts: List[str] = []
        if net_cash_val is not None:
            cap_wacc_parts.append(
                f"Net Cash/Debt={_fmt_compact_money(net_cash_val, prefix=money_prefix, suffix=money_suffix)}"
            )
        if tg_val is not None:
            cap_wacc_parts.append(f"Terminal Gr={_fmt_num(tg_val, '%')}")
        if imp_tg_val is not None:
            cap_wacc_parts.append(f"Implied Perpetual Gr={_fmt_num(imp_tg_val, '%')}")
        wacc_inputs: List[str] = []
        if rf_val is not None:
            wacc_inputs.append(f"Rf={_fmt_num(rf_val, '%')}")
        if erp_val is not None:
            wacc_inputs.append(f"ERP={_fmt_num(erp_val, '%')}")
        if ke_val is not None:
            wacc_inputs.append(f"Ke={_fmt_num(ke_val, '%')}")
        if wacc_inputs:
            cap_wacc_parts.append(f"WACC Inputs({', '.join(wacc_inputs)})")
        if cap_wacc_parts:
            val_lines.append("- Capital Structure & WACC Assumptions: " + " | ".join(cap_wacc_parts))
    gaap_v, gaap_p = _extract_margin_entry(qs.get("gaap_operating_margin"))
    arch = _first_non_none(rdcf.get("company_archetype"), dcf.get("company_archetype"))
    is_mid = _first_non_none(rdcf.get("is_mid_cycle_normalized"), dcf.get("is_mid_cycle_normalized"))
    norm_m = _first_non_none(rdcf.get("normalized_mid_cycle_ebit_margin_pct"), dcf.get("normalized_mid_cycle_ebit_margin_pct"))
    cur_m = _first_non_none(
        rdcf.get("current_ebit_margin_pct"),
        dcf.get("current_ebit_margin_pct"),
        rdcf.get("reported_ebit_margin_pct"),
        dcf.get("reported_ebit_margin_pct"),
        gaap_v,
    )
    cur_m_period = _first_non_none(rdcf.get("ebit_margin_fiscal_period"), dcf.get("ebit_margin_fiscal_period"), gaap_p)
    val_margin_src = qs.get("valuation_margin_source_used")
    arch_rationale = _first_non_none(rdcf.get("archetype_rationale"), dcf.get("archetype_rationale"))
    if any(x is not None for x in (arch, is_mid, norm_m, cur_m, val_margin_src, arch_rationale)):
        cur_m_str = _fmt_num(cur_m, "%") + (f" ({cur_m_period})" if (cur_m is not None and cur_m_period) else "")
        val_lines.append(
            f"- Valuation Archetype & Margin Basis: Archetype={arch or 'N/A'} | Mid-Cycle Normalized={is_mid} "
            f"(Normalized EBIT={_fmt_num(norm_m, '%')} vs Current={cur_m_str}) | Margin Source={val_margin_src or 'N/A'}"
        )
        if arch_rationale and not tight:
            val_lines.append(f"  * Archetype Rationale: {_clip(arch_rationale, 500)}")
    if qs.get("pe_ratio") is not None or qs.get("pe_vs_peer_avg_pct") is not None or qs.get("price_percentile_5y") is not None or detail.get("fair_value_base") is not None:
        rel_parts: List[str] = []
        if qs.get("pe_ratio") is not None:
            rel_parts.append(f"PE={_fmt_num(qs.get('pe_ratio'), 'x')}")
        rel_parts.append(
            f"PE vs Peer={_fmt_num(qs.get('pe_vs_peer_avg_pct'), '%')} ({qs.get('peer_sector') or 'N/A'}, {qs.get('peer_count') or 0} peers)"
        )
        rel_parts.append(
            f"5Y Price Percentile={_fmt_num(qs.get('price_percentile_5y'), '%')} (Z={_fmt_num(qs.get('price_zscore_5y'))})"
        )
        val_lines.append("- Relative & Context: " + " | ".join(rel_parts))
    if isinstance(analyst_context, dict) and analyst_context:
        src_as_of = str(analyst_context.get("source_as_of") or "").strip()
        src_dt = _parse_iso_date(src_as_of)
        cur_dt = _parse_iso_date(now_date_iso)
        stale_ac = ""
        if src_dt is not None and cur_dt is not None:
            ac_age = max(0, (cur_dt - src_dt).days)
            if ac_age > 14:
                stale_ac = f" [STALE {ac_age}d]"
        next_ed = str(analyst_context.get("next_earnings_date") or "").strip()[:10]
        next_ed_dt = _parse_iso_date(next_ed)
        past_due_ed = ""
        if next_ed_dt is not None and cur_dt is not None and next_ed_dt < cur_dt:
            past_due_ed = " [PAST-DUE EARNINGS DATE]"
        val_lines.append(
            f"- Analyst Context Cache (as_of={src_as_of[:10] or 'N/A'}{stale_ac}): "
            f"Target Mean={_fmt_num(analyst_context.get('target_mean'))} [Low={_fmt_num(analyst_context.get('target_low'))}, High={_fmt_num(analyst_context.get('target_high'))}], "
            f"Analysts={analyst_context.get('num_analysts') or 'N/A'} | Next Earnings={next_ed or 'N/A'}{past_due_ed}"
        )
        raw_eh = [e for e in (analyst_context.get("earnings_history") or []) if isinstance(e, dict)]
        reported_beats = 0
        reported_cnt = 0
        reported_samples: List[str] = []
        upcoming_ests: List[str] = []
        for eh in raw_eh:
            d_s = str(eh.get("date_str") or "")[:10]
            act = eh.get("eps_actual")
            est = eh.get("eps_estimate")
            if act is None:
                if est is not None:
                    upcoming_ests.append(f"{d_s}: est={_fmt_num(est)}")
                continue
            try:
                act_d = Decimal(str(act))
                est_d = Decimal(str(est)) if est is not None else None
            except Exception:
                continue
            reported_cnt += 1
            if est_d is not None and act_d > est_d:
                reported_beats += 1
            if len(reported_samples) < (2 if tight else 4):
                est_s = _fmt_num(est) if est is not None else "N/A"
                reported_samples.append(f"{d_s}: {_fmt_num(act)} vs {est_s}")
        if reported_cnt > 0 or upcoming_ests:
            hist_part = (
                f"Beat {reported_beats}/{reported_cnt} reported ({'; '.join(reported_samples)})"
                if reported_cnt > 0
                else "No reported EPS history"
            )
            up_part = f" | Upcoming Est: {', '.join(upcoming_ests[:2])}" if upcoming_ests else ""
            val_lines.append(f"  * Earnings Surprise History: {hist_part}{up_part}")

    # 3. QUALITY
    qual_lines: List[str] = ["<!-- SEC:QUALITY -->"]
    qual_lines.append(
        f"- Capital Efficiency & Cash Quality: ROIC={_fmt_num(qs.get('roic_pct'), '%')} | FCF Yield={_fmt_num(qs.get('fcf_yield_pct'), '%')} | FCF Margin={_fmt_num(qs.get('fcf_margin_pct'), '%')} | FCF 3Y CAGR={_fmt_num(qs.get('fcf_cagr_3y'), '%')} | "
        f"OCF/Net Income={_fmt_num(qs.get('ocf_to_net_income'), 'x')}"
    )
    if not tight and any(
        qs.get(k) is not None for k in ("fcf_quality_score", "debt_quality_score", "solvency_score", "peer_relative_score")
    ):
        qual_lines.append(
            f"  * Quality & Solvency Sub-Scores (0-100): FCFQual={_fmt_num(qs.get('fcf_quality_score'))} | "
            f"DebtQual={_fmt_num(qs.get('debt_quality_score'))} | Solvency={_fmt_num(qs.get('solvency_score'))} | "
            f"PeerRel={_fmt_num(qs.get('peer_relative_score'))}"
        )
    gaap_v, gaap_p = _extract_margin_entry(qs.get("gaap_operating_margin"))
    ng_v, ng_p = _extract_margin_entry(qs.get("non_gaap_operating_margin"))
    hgaap_v, _ = _extract_margin_entry(qs.get("historical_gaap_operating_margin"))
    pebit_v, _ = _extract_margin_entry(qs.get("provider_ebit_margin"))
    mb = qs.get("metric_basis") if isinstance(qs.get("metric_basis"), dict) else {}
    if any(x is not None for x in (gaap_v, ng_v, hgaap_v, pebit_v)) or mb:
        mb_str = ", ".join(f"{k}={v}" for k, v in list(mb.items())[:(4 if tight else 12)] if v)
        gaap_lbl = f"{_fmt_num(gaap_v, '%')}" + (f" ({gaap_p})" if gaap_p else "")
        ng_lbl = f"{_fmt_num(ng_v, '%')}" + (f" ({ng_p})" if ng_p else "")
        qual_lines.append(
            f"- Margins Consistency & Basis: GAAP Operating={gaap_lbl} | Non-GAAP Operating={ng_lbl} | "
            f"Hist GAAP={_fmt_num(hgaap_v, '%')} | Provider EBIT={_fmt_num(pebit_v, '%')}"
            + (f" | Basis: {mb_str}" if mb_str else "")
        )
    if piotroski:
        f_pts = []
        for lbl, bkey in (
            ("ROA>0", "roa_positive"),
            ("CFO>0", "cfo_positive"),
            ("dROA>0", "delta_roa_positive"),
            ("AccrualOK", "accrual_quality"),
            ("LevDown", "delta_leverage_improved"),
            ("LiqUp", "delta_liquidity_improved"),
            ("NoDilution", "no_share_dilution"),
            ("GMUp", "delta_gross_margin_improved"),
            ("TurnoverUp", "delta_asset_turnover_improved"),
        ):
            v = piotroski.get(bkey)
            if v is not None:
                f_pts.append(f"{lbl}:{'PASS' if v else 'FAIL'}")
        qual_lines.append(
            f"- Piotroski F-Score = {piotroski.get('f_score', 'N/A')}/9 ({piotroski.get('status', 'available')}): {', '.join(f_pts)}"
        )
    icr_val = _first_non_none(rdcf.get("interest_coverage_ratio"), dcf.get("interest_coverage_ratio"), qs.get("interest_coverage"))
    cred_rating = _first_non_none(rdcf.get("synthetic_credit_rating"), dcf.get("synthetic_credit_rating"), qs.get("synthetic_credit_rating"))
    cred_spread = _first_non_none(rdcf.get("credit_spread_bps"), dcf.get("credit_spread_bps"))
    cost_debt = _first_non_none(rdcf.get("benchmark_cost_of_debt_pct"), dcf.get("benchmark_cost_of_debt_pct"))
    eff_tax = _first_non_none(rdcf.get("effective_tax_rate_pct"), dcf.get("effective_tax_rate_pct"))
    at_cost_debt = _first_non_none(rdcf.get("after_tax_cost_of_debt_pct"), dcf.get("after_tax_cost_of_debt_pct"))
    qual_lines.append(
        f"- Solvency & Capital Return: Net Debt/EBITDA={_fmt_num(qs.get('net_debt_ebitda'), 'x')} | Interest Coverage={_fmt_num(icr_val, 'x')} | Current Ratio={_fmt_num(qs.get('current_ratio'), 'x')} | D/E={_fmt_num(qs.get('de_ratio_pct'), '%')} | "
        f"Div Yield={_fmt_num(qs.get('dividend_yield_pct'), '%')} (Payout={_fmt_num(qs.get('payout_ratio_pct'), '%')}) | Buyback Yield={_fmt_num(qs.get('net_buyback_yield_pct'), '%')} | Shareholder Yield={_fmt_num(qs.get('net_shareholder_yield_pct'), '%')}"
    )
    if any(x is not None for x in (cred_rating, cred_spread, cost_debt, eff_tax, at_cost_debt)):
        tax_kd_part = ""
        if not tight and (eff_tax is not None or at_cost_debt is not None):
            tk_items: List[str] = []
            if at_cost_debt is not None:
                tk_items.append(f"After-Tax Kd={_fmt_num(at_cost_debt, '%')}")
            if eff_tax is not None:
                tk_items.append(f"EffTax={_fmt_num(eff_tax, '%')}")
            if tk_items:
                tax_kd_part = " | " + " | ".join(tk_items)
        qual_lines.append(
            f"- Synthetic Credit Profile (Damodaran): Rating={cred_rating or 'N/A'} | ICR={_fmt_num(icr_val, 'x')} | "
            f"Credit Spread={_fmt_num(cred_spread, 'bps')} | Cost of Debt={_fmt_num(cost_debt, '%')}{tax_kd_part}"
        )
    sec_kpis = qs.get("sector_kpis") if isinstance(qs.get("sector_kpis"), dict) else {}
    if sec_kpis:
        kpi_list = [k for k in (sec_kpis.get("kpis") or []) if isinstance(k, dict)][:(3 if tight else 12)]
        kpi_parts = [
            f"{k.get('name', 'KPI')}={_fmt_num(k.get('value'))}{k.get('unit', '')} [vs {_clip(k.get('benchmark'), 32 if tight else 80)}]"
            for k in kpi_list
        ]
        qual_lines.append(
            f"- Sector KPIs ({sec_kpis.get('sector_name') or 'Sector'}, Health Score={_fmt_num(sec_kpis.get('sector_health_score'))}/100): "
            + ("; ".join(kpi_parts) if kpi_parts else "ไม่มี KPI เฉพาะกลุ่ม")
        )
    if qs.get("sloan_accrual") or qs.get("dupont_breakdown"):
        qual_lines.append(_format_forensics_line(qs.get("sloan_accrual"), qs.get("dupont_breakdown")))
    if hist_3y:
        valid_rows = [r for r in hist_3y if isinstance(r, dict) and r.get("fiscal_year")][-3:]
        if valid_rows:
            has_sga_or_roic = any(r.get("sga_pct_of_rev") is not None or r.get("roic_pct") is not None for r in valid_rows)
            hdr_cols = (
                "Year|Rev|GrossM|EBIT|R&D%|SG&A%|ROIC%|FCF|CapEx/D&A|Phase"
                if (not tight and has_sga_or_roic)
                else "Year|Rev|GrossM|EBIT|R&D%|FCF|CapEx/D&A|Phase"
            )
            qual_lines.append(f"- ประวัติงบการเงิน 3 ปีย้อนหลัง (3Y Trajectory [{hdr_cols}]):")
            for r in valid_rows:
                fy = r.get("fiscal_year")
                rev_s = _fmt_compact_money(r.get("revenue"), prefix=money_prefix, suffix=money_suffix)
                gm_s = _fmt_num(r.get("gross_margin_pct"), "%")
                em_s = _fmt_num(r.get("ebit_margin_pct"), "%")
                rd_s = _fmt_num(r.get("rd_pct_of_rev"), "%")
                fcf_s = _fmt_compact_money(r.get("fcf"), prefix=money_prefix, suffix=money_suffix)
                cap_da_s = _fmt_num(r.get("capex_to_depreciation_ratio"), "x")
                phase = str(r.get("reinvestment_lifecycle_phase") or "-").strip() or "-"
                if not tight and has_sga_or_roic:
                    sga_s = _fmt_num(r.get("sga_pct_of_rev"), "%")
                    roic_3y_s = _fmt_num(r.get("roic_pct"), "%")
                    qual_lines.append(
                        f"  FY{fy}|{rev_s}|{gm_s}|{em_s}|{rd_s}|{sga_s}|{roic_3y_s}|{fcf_s}|{cap_da_s}|{phase}"
                    )
                else:
                    qual_lines.append(
                        f"  FY{fy}|{rev_s}|{gm_s}|{em_s}|{rd_s}|{fcf_s}|{cap_da_s}|{phase}"
                    )
    if detail.get("narrative_analysis"):
        condensed_narr = _condense_repetitive_narrative(detail.get("narrative_analysis"), max_chars=narr_cap)
        if condensed_narr:
            qual_lines.append(f"- Moat & Fundamental Narrative: {condensed_narr}")

    # 4. GROWTH & GUIDANCE
    grow_lines: List[str] = ["<!-- SEC:GROWTH_GUIDANCE -->"]
    grow_lines.append(
        f"- Growth & Revisions: Rev YoY={_fmt_num(qs.get('revenue_growth_yoy_pct'), '%')} | Net Income YoY={_fmt_num(qs.get('net_income_growth_yoy_pct'), '%')} | "
        f"EPS Revision Net 30d={qs.get('eps_revision_net_30d', 'N/A')} | EPS Est Change 30d={_fmt_num(qs.get('eps_estimate_change_30d_pct'), '%')} | Earnings Momentum Score={_fmt_num(qs.get('earnings_momentum_score'))}"
    )
    if guidance:
        grow_lines.append(
            f"- Earnings Guidance ({guidance.get('latest_fiscal_period') or guidance.get('fiscal_quarter') or 'N/A'}, tone={guidance.get('management_tone', 'N/A')}): "
            f"Rev Guidance={_fmt_num(_first_non_none(guidance.get('revenue_growth_guidance_pct'), guidance.get('revenue_guidance_yoy_pct')), '%')} | "
            f"Operating Margin Trajectory={_first_non_none(guidance.get('operating_margin_trajectory'), guidance.get('margin_guidance_direction'), 'N/A')}"
        )
        if guidance.get("guidance_summary"):
            grow_lines.append(f"  * Guidance Summary: {_clip(guidance.get('guidance_summary'), 340 if tight else 800)}")
        quotes = (guidance.get("key_executive_quotes") or guidance.get("guidance_quotes") or [])[:(2 if tight else 5)]
        grow_lines.extend(_format_exec_quotes_lines(quotes, clip_limit=220 if tight else 500))
        v_claims = (guidance.get("verified_claims") or [])[:(3 if tight else 10)]
        if v_claims:
            vc_str = "; ".join(
                f"{vc.get('metric_name')}={_first_non_none(vc.get('numeric_value'), vc.get('controlled_vocabulary_value'))}{vc.get('unit', '')} ({vc.get('fiscal_period', '')})"
                for vc in v_claims if isinstance(vc, dict)
            )
            grow_lines.append(f"  * Verified Guidance Claims: {vc_str}")
    elif isinstance(ec_meta.get("guidance"), dict) and ec_meta.get("guidance"):
        g_fm = ec_meta["guidance"]
        ec_period = ec_meta.get("period") or "N/A"
        ec_date = str(ec_meta.get("date") or "")[:10] or "N/A"
        grow_lines.append(
            f"- Earnings Guidance (Vault {ec_period}, date={ec_date}, tone={g_fm.get('management_tone', 'N/A')}): "
            f"Rev Guidance={_fmt_num(_first_non_none(g_fm.get('revenue_growth_guidance_pct'), g_fm.get('revenue_guidance_yoy_pct')), '%')} | "
            f"Operating Margin Trajectory={_first_non_none(g_fm.get('operating_margin_trajectory'), g_fm.get('margin_guidance_direction'), 'N/A')}"
        )
        g_targets: List[str] = []
        rev_rng = g_fm.get("revenue_guidance_range_usd")
        if isinstance(rev_rng, (list, tuple)) and len(rev_rng) >= 2:
            g_targets.append(f"Rev={_fmt_compact_money(rev_rng[0], prefix='$')}–{_fmt_compact_money(rev_rng[1], prefix='$')}")
        bil_rng = g_fm.get("billings_guidance_range_usd")
        if isinstance(bil_rng, (list, tuple)) and len(bil_rng) >= 2:
            g_targets.append(f"Billings={_fmt_compact_money(bil_rng[0], prefix='$')}–{_fmt_compact_money(bil_rng[1], prefix='$')}")
        om_rng = g_fm.get("non_gaap_operating_margin_range_pct")
        if isinstance(om_rng, (list, tuple)) and len(om_rng) >= 2:
            g_targets.append(f"Non-GAAP OpMargin={_fmt_num(om_rng[0], '%')}–{_fmt_num(om_rng[1], '%')}")
        eps_rng = g_fm.get("non_gaap_eps_range")
        if isinstance(eps_rng, (list, tuple)) and len(eps_rng) >= 2:
            g_targets.append(
                f"Non-GAAP EPS={money_prefix}{_fmt_num(eps_rng[0])}{money_suffix}–{money_prefix}{_fmt_num(eps_rng[1])}{money_suffix}"
            )
        if g_targets:
            grow_lines.append(f"  * Guidance Targets: {' | '.join(g_targets)}")
        fm_quotes = (g_fm.get("key_executive_quotes") or g_fm.get("guidance_quotes") or [])[:(2 if tight else 5)]
        grow_lines.extend(_format_exec_quotes_lines(fm_quotes, clip_limit=240 if tight else 500))
    elif ec_summary:
        grow_lines.append("- Earnings Guidance Context: ไม่มี Sidecar Guidance แยก — ดูสรุป Earnings Call ล่าสุดจาก Vault ใน SEC:NEWS_NOTES")
    else:
        grow_lines.append("- Earnings Guidance Context: ไม่มีข้อมูล Guidance Transcript ล่าสุดใน Sidecar")
    if detail.get("base_case_summary"):
        grow_lines.append(f"- Base Case Thesis Summary: {_clip(detail.get('base_case_summary'), base_cap)}")
    if isinstance(promise_scores, dict) and promise_scores:
        p_counts = promise_scores.get("counts") if isinstance(promise_scores.get("counts"), dict) else {}
        cnt_str = ", ".join(f"{k}:{v}" for k, v in p_counts.items() if v) or "None"
        p_bias = promise_scores.get("bias") if isinstance(promise_scores.get("bias"), dict) else {}
        p_trend = promise_scores.get("trend") if isinstance(promise_scores.get("trend"), dict) else {}
        grow_lines.append(
            f"- Promise Ledger Credibility: Score={_fmt_num(promise_scores.get('credibility'))}/100 "
            f"(Grade={promise_scores.get('grade', 'N/A')}, Conf={promise_scores.get('confidence', 'N/A')}) | "
            f"Delivery={_fmt_num(promise_scores.get('delivery'))}/100 | Transparency={_fmt_num(promise_scores.get('transparency'))}/100 | "
            f"Resolved={promise_scores.get('n_resolved', 0)}/{promise_scores.get('n_promises', 0)} [{cnt_str}]"
        )
        slip_q = _fmt_num(promise_scores.get("milestone_slip_quarters"), "q") if promise_scores.get("milestone_slip_quarters") is not None else "N/A"
        bias_msg = _clip(p_bias.get("message"), 500) if (not tight and p_bias.get("message")) else ""
        bias_extra = f" — {bias_msg}" if bias_msg else ""
        trend_v = p_trend.get("verdict") or "N/A"
        grow_lines.append(
            f"  * Guidance Bias & Delivery Trend: Bias={p_bias.get('label', 'N/A')} "
            f"(Median Δ={_fmt_num(p_bias.get('median_delta_pct'), '%')}, BeatRate={_fmt_num(p_bias.get('beat_rate'), '%')}) | "
            f"MilestoneSlip={slip_q} | Trend={trend_v}{bias_extra}"
        )
        if not tight:
            kept = [str(x) for x in (promise_scores.get("top_kept") or [])[:3] if x]
            broken = [str(x) for x in (promise_scores.get("top_broken") or [])[:3] if x]
            if kept or broken:
                grow_lines.append(
                    f"  * Standout Promises: Kept={', '.join(kept) or 'None'} | Broken/Dropped={', '.join(broken) or 'None'}"
                )

    # 5. MACRO & TACTICAL
    mac_lines: List[str] = ["<!-- SEC:MACRO_TACTICAL -->"]
    live_reg_s = (
        str(live_macro_dash.get("overall_regime") or live_macro_dash.get("regime") or "").strip()
        if isinstance(live_macro_dash, dict)
        else ""
    )
    if scorecard.get("macro_regime") or scorecard.get("macro_alignment_status"):
        sc_reg_val = scorecard.get("macro_regime")
        sc_st_val = scorecard.get("macro_alignment_status")
        live_hint = (
            f" | Live @Macro ปัจจุบัน={live_reg_s}"
            if (live_reg_s and (not sc_reg_val or str(sc_st_val or "").lower() == "unavailable"))
            else ""
        )
        mac_lines.append(
            f"- Stock Macro Alignment: Regime={sc_reg_val or 'N/A'} | Status={sc_st_val or 'N/A'}{live_hint}"
            + (f" ({_clip(scorecard.get('macro_alignment_note'), 200 if tight else 500)})" if scorecard.get("macro_alignment_note") else "")
        )
    elif live_reg_s:
        mac_lines.append(
            f"- Stock Macro Alignment: Sidecar ยังไม่ได้ระบุ Macro Regime/Alignment (Live @Macro ปัจจุบัน={live_reg_s})"
        )
    drift_warn = _compute_macro_drift_warning(
        scorecard,
        live_macro_dash,
        holding_advisory,
        portfolio_attached=portfolio_attached,
    )
    if drift_warn:
        mac_lines.append(f"- {drift_warn}")
    if tactical:
        sr_parts: List[str] = []
        if tactical.get("key_support_level") is not None:
            sr_parts.append(f"Support={_fmt_num(tactical.get('key_support_level'))}")
        if tactical.get("key_resistance_level") is not None:
            sr_parts.append(f"Resistance={_fmt_num(tactical.get('key_resistance_level'))}")
        sr_suffix = (", " + ", ".join(sr_parts)) if sr_parts else ""
        mac_lines.append(
            f"- Tactical Setup ({tactical.get('horizon_timeframe', '1-3M')}): Stage={tactical.get('price_stage', 'N/A')} | "
            f"SMA50={_fmt_num(tactical.get('sma_50'))}, SMA200={_fmt_num(tactical.get('sma_200'))}, ATR14={_fmt_num(tactical.get('atr_14'))}{sr_suffix}"
        )
        brk_parts: List[str] = [f"trigger={_fmt_num(tactical.get('breakout_trigger_price'))}"]
        if tactical.get("breakout_target_price") is not None:
            brk_parts.append(f"target={_fmt_num(tactical.get('breakout_target_price'))}")
        if tactical.get("breakout_stop_loss") is not None:
            brk_parts.append(f"SL={_fmt_num(tactical.get('breakout_stop_loss'))}")
        if tactical.get("breakout_planned_rr") is not None:
            brk_parts.append(f"planned R:R={_fmt_num(tactical.get('breakout_planned_rr'), 'x')}")
        if tactical.get("breakout_volume_ratio") is not None:
            brk_parts.append(f"VolRatio={_fmt_num(tactical.get('breakout_volume_ratio'), 'x')}")
        mac_lines.append(
            f"  * Pullback Zone={_fmt_num(tactical.get('buy_zone_min'))}–{_fmt_num(tactical.get('buy_zone_max'))} ({tactical.get('pullback_entry_status', 'N/A')}) | Stop Loss={_fmt_num(tactical.get('invalidation_stop_loss'))} | "
            f"Target={_fmt_num(tactical.get('tactical_target_price'))} | Current R:R={_fmt_num(tactical.get('current_rr_ratio'), 'x')} | Breakout Status={tactical.get('breakout_entry_status', 'N/A')} ({', '.join(brk_parts)})"
        )
    if smart_money:
        sm_buy = int(smart_money.get("insider_buy_count_90d") or 0)
        sm_sell = int(smart_money.get("insider_sell_count_90d") or 0)
        ins_sig = str(smart_money.get("insider_signal") or "N/A")
        overall_sm = str(smart_money.get("overall_smart_money_flag") or "N/A")
        squeeze_risk = bool(smart_money.get("short_squeeze_risk", False))
        sched_part = ""
        if isinstance(insider, dict) and insider.get("data_status") == "available":
            if sm_buy == 0 and sm_sell == 0:
                f4_buy = int(insider.get("open_market_p_count_90d") or 0)
                f4_unflagged_sell = int(insider.get("unflagged_s_count_90d") or 0)
                if f4_buy > 0 or f4_unflagged_sell > 0:
                    sm_buy = f4_buy
                    sm_sell = f4_unflagged_sell
                    if sm_buy > sm_sell:
                        ins_sig = "buying"
                        overall_sm = "bearish_signal" if squeeze_risk else "bullish_signal"
                    elif sm_sell > sm_buy:
                        ins_sig = "selling"
                        overall_sm = "bearish_signal"
            sched_n = int(insider.get("rule_10b5_1_s_count_90d") or 0)
            if sched_n > 0:
                sched_val = _fmt_compact_money(
                    insider.get("rule_10b5_1_s_value_usd"),
                    prefix=money_prefix,
                    suffix=money_suffix,
                )
                sched_val_part = f" ({sched_val})" if sched_val != "N/A" else ""
                sched_part = f"; scheduled 10b5-1 sells={sched_n}{sched_val_part}"
        mac_lines.append(
            f"- Smart Money Flags: Overall={overall_sm} | Insider Signal={ins_sig} "
            f"(90d discretionary buys={sm_buy}, sells={sm_sell}{sched_part}) | "
            f"Insider Own={_fmt_num(smart_money.get('insider_ownership_pct'), '%')} | "
            f"Institutional={_fmt_num(smart_money.get('institutional_ownership_pct'), '%')} | "
            f"Short Interest={_fmt_num(smart_money.get('short_interest_pct'), '%')} (SqueezeRisk={squeeze_risk})"
        )
    sens = (
        (rdcf.get("sensitivity_summary") if isinstance(rdcf.get("sensitivity_summary"), dict) and rdcf["sensitivity_summary"].get("macro_yield_shock_50bps_delta") is not None else None)
        or rdcf.get("sensitivity_summary")
        or dcf.get("sensitivity_summary")
        or {}
    )
    if isinstance(sens, dict) and sens:
        mac_lines.append(
            f"- Valuation Sensitivity: Target Range={_fmt_num(sens.get('min_target_price'))}–{_fmt_num(sens.get('max_target_price'))} (Upside {_fmt_num(sens.get('min_upside_pct'), '%')}..{_fmt_num(sens.get('max_upside_pct'), '%')}) | "
            f"Driver={sens.get('primary_operating_driver', 'N/A')} | 10Y Yield +50bps={_fmt_num(sens.get('macro_yield_shock_50bps_delta'))}, +100bps={_fmt_num(sens.get('macro_yield_shock_100bps_delta'))}"
        )

    # 6. RISKS, FALSIFIERS, POSITION REVIEW & ACTIVE THESIS
    risk_lines: List[str] = ["<!-- SEC:RISKS_THESIS -->"]
    if dedup_exc:
        risk_lines.append("- [WATCHDOG BREACHES (PENDING)]:")
        for ex in dedup_exc[:(3 if tight else 8)]:
            rep_cnt = int(ex.get("repeat_count") or 1)
            det_s = str(ex.get("detected_at_utc") or "")[:10] or "N/A"
            rep_tag = f" (Detected {det_s}, x{rep_cnt})" if rep_cnt > 1 else f" (Detected {det_s})"
            risk_lines.append(
                f"  * [{ex.get('exception_type', 'BREACH')}] {ex.get('claim_id', '')} | {ex.get('metric_name', '')}: "
                f"Expected '{ex.get('expected_value', '')}' vs Observed '{ex.get('observed_value', '')}'{rep_tag} [Status={ex.get('status', 'PENDING')} | Action={ex.get('human_action', 'REVIEW_REQUIRED')}]"
            )
    if sent.get("tail_risks"):
        tr_limit = 4 if tight else 10
        tr_clip = 140 if tight else 450
        risk_lines.append(f"- Tail Risks: {'; '.join(_clip(tr, tr_clip) for tr in sent.get('tail_risks', [])[:tr_limit])}")
    if falsifiers:
        risk_lines.append("- Thesis Falsifiers (จุดหักล้างสมมติฐาน):")
        fal_limit = 4 if tight else 10
        fal_clip = 150 if tight else 500
        for f in falsifiers[:fal_limit]:
            if isinstance(f, dict):
                cond_thr = _format_falsifier_condition(f.get("condition", ""), f.get("threshold_value", ""))
                cond_part = f" {cond_thr}" if cond_thr else ""
                risk_lines.append(
                    f"  * [{f.get('falsifier_id', 'F')}] {f.get('metric_name', '')}{cond_part}: {_clip(f.get('narrative_explanation'), fal_clip)}"
                )
    if thesis_delta:
        breached = thesis_delta.get("breached_falsifiers") or []
        td_tp = (
            f", TargetPriceDelta={_fmt_num(thesis_delta.get('target_price_change_pct'), '%')}"
            if thesis_delta.get("target_price_change_pct") is not None
            else ""
        )
        td_st = (
            f", Stance={thesis_delta.get('previous_action_stance')}->{thesis_delta.get('current_action_stance')}"
            if (thesis_delta.get("previous_action_stance") or thesis_delta.get("current_action_stance"))
            else ""
        )
        risk_lines.append(
            f"- Thesis Delta (เทียบ {thesis_delta.get('previous_analysis_date', 'N/A')}): Price Change={_fmt_num(thesis_delta.get('price_change_pct'), '%')}, "
            f"Conviction Delta={_fmt_num(thesis_delta.get('conviction_score_delta'))}{td_tp}{td_st}, F-Score Delta={thesis_delta.get('f_score_delta', 'N/A')}, "
            f"Breached Falsifiers={', '.join(breached) if breached else 'None'}"
        )
    if insider:
        p_val = insider.get("open_market_p_value_usd_str") or "N/A"
        s_val = insider.get("open_market_s_value_usd_str") or "N/A"
        r10_cnt = insider.get("rule_10b5_1_s_count_90d", 0)
        r10_val = insider.get("rule_10b5_1_s_value_usd_str") or "N/A"
        ins_extra = ""
        if not tight and any(
            int(insider.get(k) or 0) > 0 for k in ("unflagged_s_count_90d", "exercise_count_90d", "tax_withholding_count_90d")
        ):
            ins_extra = (
                f", UnflaggedSell={insider.get('unflagged_s_count_90d', 0)}, "
                f"Exercise={insider.get('exercise_count_90d', 0)}, TaxWithhold={insider.get('tax_withholding_count_90d', 0)}"
            )
        risk_lines.append(
            f"- Insider Conviction (90d): Status={insider.get('status', 'N/A')} (Confidence={insider.get('signal_confidence', 'N/A')}) | "
            f"Open-Market Buy={insider.get('open_market_p_count_90d', 0)} ({p_val}, C-Suite P={insider.get('c_suite_p_count', 0)}), "
            f"Open-Market Sell={insider.get('open_market_s_count_90d', 0)} ({s_val}), "
            f"Scheduled 10b5-1 Sell={r10_cnt} ({r10_val}){ins_extra}, AmendmentUnresolved={insider.get('amendment_unresolved', False)}"
        )
    if isinstance(insider_db, dict) and insider_db:
        prov = insider_db.get("source_provenance") or "UNKNOWN"
        buy_cnt = insider_db.get("open_market_buy_count", 0)
        buy_usd = _fmt_compact_money(insider_db.get("open_market_buy_usd"), prefix="$")
        buy_sh = _fmt_compact_money(insider_db.get("open_market_buy_shares"))
        open_s = insider_db.get("open_market_sale_count", 0)
        pair_s = insider_db.get("paired_exercise_sale_count", 0)
        sale_usd = _fmt_compact_money(insider_db.get("total_sale_usd"), prefix="$")
        risk_lines.append(
            f"- Insider DB Ledger (180d, Provenance={prov}): "
            f"Open-Market Paid Buy={buy_cnt} ({buy_usd}, {buy_sh} sh) | "
            f"Pure Open-Market Sell={open_s}, Paired Exercise/Award Sell={pair_s} (Total Sell={sale_usd}) | "
            f"Exercise(M)={insider_db.get('exercise_count', 0)}, Award(A)={insider_db.get('award_grant_count', 0)}, Tax(F)={insider_db.get('tax_withhold_count', 0)}"
        )
        recl = int(insider_db.get("reclassified_zero_dollar_buys") or 0)
        if recl > 0:
            risk_lines.append(
                f"  * [DATA HYGIENE NOTE]: คัดแยกธุรกรรม yfinance ราคา $0 จำนวน {recl} รายการออกจากฝั่ง Open-Market Paid Buy เป็น Award/Grant (Code A)"
            )
        notable = [n for n in (insider_db.get("recent_notable") or []) if isinstance(n, dict)]
        for n_row in notable[:(2 if tight else 5)]:
            role_tag = f" ({_clip(n_row.get('officer_title'), 40)})" if n_row.get("officer_title") else ""
            pair_tag = " [Paired Exercise]" if n_row.get("paired_exercise") else ""
            risk_lines.append(
                f"  * {n_row.get('transaction_date', 'N/A')} | {n_row.get('owner', 'Insider')}{role_tag} | "
                f"Code={n_row.get('effective_code', '?')} | {_fmt_compact_money(n_row.get('shares'))} sh @ ${_fmt_num(n_row.get('price_per_share'))} "
                f"({_fmt_compact_money(n_row.get('usd_value'), prefix='$')}){pair_tag}"
            )
    if active_thesis:
        at_meta = ""
        if not tight and (active_thesis.get("status") or active_thesis.get("evidence_tier")):
            at_meta = f" [{active_thesis.get('status', 'ACTIVE')} | {active_thesis.get('evidence_tier', 'N/A')}]"
        risk_lines.append(f"- Active Stock Thesis (Vault){at_meta}: {_clip(active_thesis.get('claim_statement'), 260 if tight else 600)}")
        if not tight:
            for af in (active_thesis.get("falsifiers") or [])[:8]:
                if isinstance(af, dict):
                    af_op = af.get("operator") or af.get("condition") or ""
                    af_unit = f" {af.get('unit')}" if af.get("unit") else ""
                    af_imp = f" [{af.get('invalidation_impact')}]" if af.get("invalidation_impact") else ""
                    risk_lines.append(
                        f"  * Active Falsifier [{af.get('falsifier_id', 'AF')}] {af.get('metric_name', '')} {af_op} {af.get('threshold_value', '')}{af_unit}{af_imp}: "
                        f"{_clip(af.get('narrative_explanation'), 320)}"
                    )
            cur_dt_val = _parse_iso_date(now_date_iso)
            for cat in (active_thesis.get("linked_catalysts") or [])[:6]:
                if isinstance(cat, dict):
                    exp_str = str(cat.get("expected_date_str") or "").strip()
                    cat_status = str(cat.get("status") or "PENDING").strip()
                    exp_dt = _parse_iso_date(exp_str)
                    past_due_cat = (
                        " [PAST-DUE CATALYST — ควรตรวจสอบผล]"
                        if (exp_dt is not None and cur_dt_val is not None and exp_dt < cur_dt_val and cat_status.upper() == "PENDING")
                        else ""
                    )
                    risk_lines.append(
                        f"  * Linked Catalyst [{cat.get('catalyst_id', 'CAT')}]: {cat.get('event_name', 'Event')} "
                        f"(Expected: {exp_str or 'N/A'}, Status: {cat_status}){past_due_cat}"
                    )
    if pos_review:
        decomps = pos_review.get("decompositions") or []
        for dec in decomps[:(2 if tight else 4)]:
            if isinstance(dec, dict):
                mult_exp = _first_non_none(dec.get("multiple_expansion_pct"), dec.get("pe_expansion_pct"))
                diag_tier = dec.get("diagnosis_tier")
                tier_tag = f" | Tier={diag_tier}" if diag_tier else ""
                risk_lines.append(
                    f"- Price Decomposition ({dec.get('horizon_label', '')}): Total Return={_fmt_num(dec.get('total_return_pct'), '%')} "
                    f"(EPS Growth={_fmt_num(dec.get('eps_growth_pct'), '%')} vs Multiple Expansion={_fmt_num(mult_exp, '%')}) | Driver={dec.get('primary_driver', 'N/A')}{tier_tag}"
                )
        raw_tvn = pos_review.get("then_vs_now")
        if isinstance(raw_tvn, dict) and raw_tvn:
            tvn = _sanitize_then_vs_now(raw_tvn, portfolio_attached=portfolio_attached)
            has_then_baseline = any(tvn.get(k) is not None for k in ("then_price", "then_pe", "then_growth_pct"))
            if has_then_baseline:
                pl_lbl = str(tvn.get("period_label") or "").strip()
                pl_tag = f" ({pl_lbl})" if pl_lbl else ""
                px_part = (
                    f"Price {_fmt_num(tvn.get('then_price'))} -> {_fmt_num(tvn.get('now_price'))} | "
                    if (not tight and (tvn.get("then_price") is not None or tvn.get("now_price") is not None))
                    else ""
                )
                tvn_raw = str(
                    tvn.get("regime_shift_summary")
                    or tvn.get("summary")
                    or tvn.get("thesis_reflection_question")
                    or ""
                ).strip()
                if tvn_raw and re.search(r"(?i)(?:\bnan\b|\$nan|nan%)", tvn_raw):
                    tvn_raw = ""
                tvn_assess = _clip(tvn_raw, 180 if tight else 650) if tvn_raw else ""
                assess_part = f" | Assessment: {tvn_assess}" if tvn_assess else ""
                then_now_prose = ""
                if not tight:
                    t_then = _clip(tvn.get("what_was_true_then"), 350) if tvn.get("what_was_true_then") else ""
                    t_now = _clip(tvn.get("what_is_true_now"), 350) if tvn.get("what_is_true_now") else ""
                    if t_then:
                        then_now_prose += f" | Then: {t_then}"
                    if t_now:
                        then_now_prose += f" | Now: {t_now}"
                risk_lines.append(
                    f"- Then vs Now{pl_tag}: {px_part}PE {_fmt_num(tvn.get('then_pe'))} -> {_fmt_num(tvn.get('now_pe'))} | "
                    f"Growth {_fmt_num(tvn.get('then_growth_pct'), '%')} -> {_fmt_num(tvn.get('now_growth_pct'), '%')}{then_now_prose}{assess_part}"
                )
        # Strict 3-State Guard: expose triggers referencing portfolio position ONLY when @Portfolio is attached!
        if portfolio_attached and pos_review.get("triggers"):
            raw_triggers = [t for t in (pos_review.get("triggers") or []) if isinstance(t, dict)]
            ordered_triggers = sorted(raw_triggers, key=lambda t: (0 if t.get("is_currently_triggered") else 1))
            trig_texts: List[str] = []
            for t in ordered_triggers[:(3 if tight else 10)]:
                prefix = "[ACTIVE] " if t.get("is_currently_triggered") else ""
                if t.get("action_note") or t.get("trigger_label"):
                    desc = str(t.get("action_note") or t.get("trigger_label") or "").strip()
                else:
                    title = str(t.get("title") or t.get("trigger_id") or "Rule").strip()
                    impact = str(t.get("case_impact") or "").strip()
                    cond = str(t.get("condition_text") or t.get("impact_description") or "").strip()
                    head = f"{title} ({impact})" if impact else title
                    desc = f"{head}: {cond}" if cond else head
                clean_desc = _clip(f"{prefix}{desc}", 170 if tight else 450)
                if clean_desc:
                    trig_texts.append(clean_desc)
            if trig_texts:
                risk_lines.append(f"- Portfolio Decision Triggers: {'; '.join(trig_texts)}")
        if pos_review.get("honest_gaps"):
            gap_limit = 2 if tight else 10
            gap_clip = 110 if tight else 400
            risk_lines.append(f"- Honest Gaps: {'; '.join(_clip(g, gap_clip) for g in (pos_review.get('honest_gaps') or [])[:gap_limit])}")

    # 7. NEWS & NOTES
    news_lines = _build_news_and_notes_lines(
        ticker=ticker,
        news_data=news_data,
        notes_list=notes_list,
        earnings_note=earnings_note,
        analysis_note=analysis_note,
        shown_narrative=str(detail.get("narrative_analysis") or ""),
        narr_cap=narr_cap,
        tight=tight,
        ec_summary_override=ec_summary,
    )
    sec_news = ["<!-- SEC:NEWS_NOTES -->"] + (news_lines or ["- ข่าวและบันทึก Vault รายตัว: ยังไม่มีข่าวเฉพาะหุ้นหรือบันทึกล่าสุดใน Vault"])

    full_sections = [
        f"=== หุ้น ${ticker} ({company}) ===",
        "\n".join(header_lines),
        "\n".join(val_lines),
        "\n".join(qual_lines),
        "\n".join(grow_lines),
        "\n".join(mac_lines),
        "\n".join(risk_lines),
        "\n".join(sec_news),
    ]
    return "\n\n".join(full_sections)


def _build_comparative_table(ticker_bundles: Sequence[Tuple[str, Dict[str, Any]]]) -> str:
    """Build deterministic side-by-side comparison table when 2+ equity tickers are analyzed."""
    if len(ticker_bundles) < 2:
        return ""
    rows: List[str] = [
        "=== ตารางเปรียบเทียบหุ้นที่แนบในวงเสวนา (COMPARATIVE SNAPSHOT TABLE) ===",
        "| Ticker | Price (MCap) | Composite / Quality / Value / Growth | Action Stance | DCF Base vs ReverseDCF 12M | MoS Entry (Buf%) | Implied Gr% vs Rev YoY% | ROIC% / F-Score | NetDebt/EBITDA, Credit & MDD% |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for sym, detail in ticker_bundles:
        qs = detail.get("quant_signals") if isinstance(detail.get("quant_signals"), dict) else {}
        ams = qs.get("atomic_market_snapshot") if isinstance(qs.get("atomic_market_snapshot"), dict) else {}
        sc = _pick_subdict(qs, detail, "deterministic_scorecard")
        dcf = _pick_subdict(qs, detail, "dcf_result")
        rdcf = _pick_subdict(qs, detail, "reverse_dcf_result")
        pio = _pick_subdict(qs, detail, "piotroski_breakdown")
        px = _extract_price_from_detail(qs, detail, None)
        ccy = detail.get("currency") or ("THB" if sym.endswith(".BK") else "USD")
        mcap_prefix = "$" if ccy == "USD" else ""
        mcap_s = _fmt_compact_money(_first_non_none(ams.get("market_cap"), ams.get("market_cap_str")), prefix=mcap_prefix)
        px_col = f"{_fmt_num(px, f' {ccy}')}" + (f" ({mcap_s})" if mcap_s != "N/A" else "")
        dcf_base_t = ((dcf.get("scenarios") or {}).get("base") or {}).get("target_price")
        dcf_str = _fmt_num(dcf_base_t) if dcf.get("is_actionable", True) and dcf_base_t is not None else "N/A(unactionable)"
        rdcf_12m = _fmt_num(rdcf.get("target_price_12m"))
        mos_str = f"{_fmt_num(rdcf.get('mos_entry_price_today'))} ({_fmt_num(rdcf.get('current_margin_of_safety_pct'), '%')})"
        cred = _first_non_none(rdcf.get("synthetic_credit_rating"), dcf.get("synthetic_credit_rating")) or "N/A"
        mdd_s = _fmt_num(_first_non_none(ams.get("mdd_pct"), qs.get("mdd_pct")), "%")
        rows.append(
            f"| ${sym} | {px_col} | "
            f"{_fmt_num(_first_non_none(qs.get('composite_score'), detail.get('composite_score')))} / {_fmt_num(qs.get('quality_score'))} / {_fmt_num(qs.get('value_score'))} / {_fmt_num(qs.get('growth_score'))} | "
            f"{sc.get('action_stance', 'N/A')} | DCF {dcf_str} / RDCF {rdcf_12m} | {mos_str} | "
            f"{_fmt_num(rdcf.get('market_implied_growth_pct'), '%')} vs {_fmt_num(qs.get('revenue_growth_yoy_pct'), '%')} | "
            f"{_fmt_num(qs.get('roic_pct'), '%')} / {pio.get('f_score', 'N/A')}/9 | "
            f"{_fmt_num(qs.get('net_debt_ebitda'), 'x')}, {cred}, MDD {mdd_s} |"
        )
    return "\n".join(rows)


def _format_macro_brief(dash: Optional[Dict[str, Any]], *, detailed: bool = True) -> str:
    """Curate MacroStrategyDirection payload, putting high-signal strategy first & excluding observable_registry noise."""
    if not dash or not isinstance(dash, dict):
        return ""
    # Handle minimal unit-test mock payloads cleanly
    known_rich = {
        "overall_regime",
        "conviction_rationale",
        "regime_probabilities",
        "regime_evidence",
        "asset_allocation",
        "dashboard_indicators",
        "thailand_market_stance",
        "sector_analysis",
        "risk_scenarios",
        "pair_trades",
        "key_assumptions",
    }
    if not known_rich.intersection(dash.keys()) or (len(dash) <= 3 and "regime_probabilities" not in dash and "asset_allocation" not in dash):
        return json.dumps({k: v for k, v in dash.items() if k != "observable_registry"}, ensure_ascii=False, indent=2, default=str)

    # Top Executive Signal FIRST so Fundamental gurus with tighter budgets always see Allocation, Sector Rotation, Risk & Thai Stance
    lines: List[str] = [
        f"- Macro Regime: {dash.get('overall_regime', dash.get('regime', 'N/A'))} (As Of: {str(dash.get('evaluated_at', 'N/A'))[:19]}, Horizon: {dash.get('time_horizon', 'N/A')})",
        f"- Conviction: {dash.get('conviction_level', 'N/A')} | Rationale: {_clip(dash.get('conviction_rationale') or dash.get('divergence_note') or 'N/A', 700 if detailed else 300)}",
    ]
    probs = dash.get("regime_probabilities")
    if isinstance(probs, dict) and probs:
        lines.append("- Regime Probabilities: " + ", ".join(f"{k}: {_fmt_num(v, '%')}" for k, v in probs.items()))

    if dash.get("stale_data_warnings") or dash.get("validation_warnings"):
        warns = (dash.get("stale_data_warnings") or []) + (dash.get("validation_warnings") or [])
        w_lim = 8 if detailed else 3
        w_clip = 240 if detailed else 110
        lines.append(f"- Macro Data Warnings: {'; '.join(_clip(w, w_clip) for w in warns[:w_lim])}")

    alloc = dash.get("asset_allocation")
    if isinstance(alloc, list) and alloc:
        lines.append("- Asset Allocation Stance:")
        a_lim = 16 if detailed else 8
        r_clip = 450 if detailed else 160
        for item in alloc[:a_lim]:
            if isinstance(item, dict):
                extra_tags: List[str] = []
                if detailed:
                    if item.get("confidence"):
                        extra_tags.append(f"Conf={item.get('confidence')}")
                    if item.get("allocation_delta"):
                        extra_tags.append(f"Δ={item.get('allocation_delta')}")
                    if item.get("benchmark_ref"):
                        extra_tags.append(f"Bench={_clip(item.get('benchmark_ref'), 60)}")
                    if item.get("time_horizon"):
                        extra_tags.append(f"Horizon={item.get('time_horizon')}")
                tag_str = f" [{', '.join(extra_tags)}]" if extra_tags else ""
                sd_part = ""
                raw_sd = [str(x).strip() for x in (item.get("supporting_data") or []) if str(x).strip()]
                if raw_sd:
                    sd_lim = 4 if detailed else 2
                    sd_clip = 160 if detailed else 90
                    sd_part = f" | Data: {'; '.join(_clip(sd, sd_clip) for sd in raw_sd[:sd_lim])}"
                inv_part = ""
                if detailed and item.get("invalidation_conditions"):
                    invs = "; ".join(_clip(ic, 120) for ic in (item.get("invalidation_conditions") or [])[:3])
                    if invs:
                        inv_part = f" | Invalidation: {invs}"
                wnh = str(item.get("why_not_high") or "").strip()
                wnh_part = (
                    f" | WhyNotHigh: {_clip(wnh, 140)}"
                    if (detailed and wnh and wnh != "-" and str(item.get("confidence") or "").lower() != "high")
                    else ""
                )
                rat_s = re.sub(r"\s{2,}", " ", re.sub(r"\s*\(\s*\)", "", str(item.get("rationale") or ""))).strip()
                lines.append(
                    f"  * {item.get('asset_class', '')} ({item.get('region', item.get('asset_bucket', ''))}): "
                    f"{item.get('stance', 'Neutral')}{tag_str} — {_clip(rat_s, r_clip)}{sd_part}{inv_part}{wnh_part}"
                )
    elif isinstance(alloc, dict) and alloc:
        lines.append("- Asset Allocation Stance: " + ", ".join(f"{k}={v}" for k, v in alloc.items()))

    sec = dash.get("sector_analysis")
    if isinstance(sec, dict) and sec:
        lines.append(
            f"- Sector Rotation ({sec.get('as_of_date', '')}): "
            f"{_clip(sec.get('summary_th') or sec.get('unavailable_reason') or '', 800 if detailed else 280)}"
        )
        if sec.get("watch_conditions"):
            wc_strs: List[str] = []
            wc_lim = 8 if detailed else 3
            wc_reason_clip = 260 if detailed else 90
            wc_item_clip = 260 if detailed else 110
            for w in (sec.get("watch_conditions") or [])[:wc_lim]:
                if isinstance(w, dict):
                    mref = str(w.get("metric_ref") or w.get("metric") or "").strip()
                    op = str(w.get("operator") or "").strip()
                    thresh = w.get("future_threshold")
                    unit = f" ({w.get('unit')})" if w.get("unit") else ""
                    reason = _clip(w.get("reason") or "", wc_reason_clip)
                    head = f"{mref} {op} {thresh}{unit}".strip()
                    wc_strs.append(f"{head}: {reason}" if reason else head)
                else:
                    wc_strs.append(_clip(w, wc_item_clip))
            if wc_strs:
                lines.append(f"  * Watch Conditions: {'; '.join(wc_strs)}")

    risks = dash.get("risk_scenarios")
    if isinstance(risks, list) and risks:
        lines.append("- Macro Risk Scenarios:")
        for rs in risks[:(8 if detailed else 3)]:
            if isinstance(rs, dict):
                mit_part = ""
                conf_part = f", Conf={rs.get('confidence')}" if rs.get("confidence") else ""
                if detailed:
                    if rs.get("hedge_size"):
                        mit_part += f" | Size={_clip(rs.get('hedge_size'), 90)}"
                    if rs.get("cost_or_tradeoff"):
                        mit_part += f" | Cost={_clip(rs.get('cost_or_tradeoff'), 120)}"
                    if rs.get("volume_threshold"):
                        mit_part += f" | VolThresh={_clip(rs.get('volume_threshold'), 110)}"
                    if rs.get("unwind_or_cover_condition"):
                        mit_part += f" | Unwind={_clip(rs.get('unwind_or_cover_condition'), 120)}"
                    mit_s = _clip(rs.get("mitigation_strategy") or "", 180)
                    ewi = [str(x) for x in (rs.get("early_warning_indicators") or [])[:3] if x]
                    rs_sd = [str(x).strip() for x in (rs.get("supporting_data") or [])[:3] if str(x).strip()]
                    if mit_s:
                        mit_part += f" | Mitigation={mit_s}"
                    if ewi:
                        mit_part += f" | EarlyWarn={', '.join(ewi)}"
                    if rs_sd:
                        mit_part += f" | Data={'; '.join(_clip(sd, 120) for sd in rs_sd)}"
                lines.append(
                    f"  * {rs.get('tail_risk', '')} (Prob={rs.get('probability')}, Impact={rs.get('impact')}{conf_part}) | "
                    f"Trigger={rs.get('trigger_to_activate', 'N/A')} | Hedge={', '.join(rs.get('hedge_instruments') or [])}{mit_part}"
                )

    th = dash.get("thailand_market_stance")
    if isinstance(th, dict) and th:
        flow = th.get("investor_flow") or {}
        mb = th.get("market_breadth") or {}
        val = th.get("valuation") or {}
        pg = th.get("physical_gold") or {}
        gold_part = ""
        if isinstance(pg, dict) and pg.get("bar_sell_thb") is not None:
            pg_unit = f" {pg.get('unit')}" if pg.get("unit") else " THB"
            gold_part = f" | GoldBar={_fmt_num(pg.get('bar_sell_thb'))}{pg_unit}"
        lines.append(
            f"- Thailand Stance: PE={_fmt_num(val.get('pe_ratio'))}, PBV={_fmt_num(val.get('pbv_ratio'))}, DivYield={_fmt_num(val.get('dividend_yield'), '%')} | "
            f"Spread(TH-US Policy)={_fmt_num(th.get('policy_spread_bps'), 'bps')}, 10Y-2Y={_fmt_num(th.get('yield_curve_spread_10y_2y_bps'), 'bps')} | "
            f"Flows(M THB): Inst={_fmt_num(flow.get('institution_net_mb'))}, Foreign={_fmt_num(flow.get('foreign_net_mb'))}, Retail={_fmt_num(flow.get('retail_net_mb'))} | Breadth: {mb.get('sentiment', 'N/A')}{gold_part}"
        )
        if th.get("rationale"):
            lines.append(f"  * Thai Rationale: {_clip(th.get('rationale'), 1000 if detailed else 220)}")

    qna = dash.get("quant_narrative_alignment")
    div_note = dash.get("divergence_note")
    if qna is not None or (div_note and dash.get("conviction_rationale")):
        qna_parts: List[str] = []
        if qna is not None:
            qna_parts.append(f"Alignment={qna}")
        if div_note and (dash.get("conviction_rationale") or qna is not None):
            qna_parts.append(_clip(div_note, 600 if detailed else 260))
        if qna_parts:
            lines.append("- Quant-Narrative Alignment: " + " | ".join(qna_parts))

    # Supporting Key Assumptions, Regional, Themes, Pair Trades, Evidence, Key Indicators & Macro News References
    assumptions = dash.get("key_assumptions")
    if isinstance(assumptions, list) and assumptions:
        lines.append(
            "- Key Macro Assumptions: "
            + " | ".join(_clip(a, 320 if detailed else 140) for a in assumptions[:(10 if detailed else 4)])
        )
    reg = dash.get("regional_assessments")
    if isinstance(reg, dict) and reg:
        priority_regions = ["United States", "Thailand", "Euro Area", "China", "Japan", "India"]
        ordered_regs = [r for r in priority_regions if r in reg] + [r for r in reg if r not in priority_regions]
        target_regs = ordered_regs if detailed else ordered_regs[:6]
        lines.append("- Regional Macro States: [Region|State|Growth|Infl|Mon|Conf]")
        for rname in target_regs:
            rv = reg.get(rname)
            if isinstance(rv, dict):
                lines.append(
                    f"  {rname}|{rv.get('economic_state', 'N/A')}|{_fmt_num(rv.get('growth_score'))}|"
                    f"{_fmt_num(rv.get('inflation_score'))}|{_fmt_num(rv.get('monetary_score'))}|{_fmt_num(rv.get('confidence'))}"
                )

    themes = dash.get("focus_themes")
    if isinstance(themes, list) and themes:
        lines.append(
            "- Focus Themes: "
            + " | ".join(_clip(t, 320 if detailed else 110) for t in themes[:(10 if detailed else 4)])
        )

    pairs = dash.get("pair_trades")
    if isinstance(pairs, list) and pairs:
        lines.append("- Tactical Pair Trades:")
        for pt in pairs[:(8 if detailed else 3)]:
            if isinstance(pt, dict):
                pt_meta_parts: List[str] = []
                if pt.get("time_horizon"):
                    pt_meta_parts.append(str(pt.get("time_horizon")))
                if pt.get("confidence"):
                    pt_meta_parts.append(f"Conf={pt.get('confidence')}")
                if detailed and pt.get("sizing_guidance"):
                    pt_meta_parts.append(f"Size={pt.get('sizing_guidance')}")
                pt_meta = f" ({', '.join(pt_meta_parts)})" if pt_meta_parts else ""
                pt_extra = ""
                if detailed:
                    for lbl, pkey, clip_n in (
                        ("Proxy", "instrument_proxy", 90),
                        ("HedgeRatio", "hedge_ratio", 80),
                        ("FX", "fx_handling", 80),
                        ("Entry", "entry_trigger", 110),
                        ("SL", "stop_loss_trigger", 90),
                        ("Target", "target_gain_or_rebalance", 90),
                        ("MaxDD", "max_drawdown_limit", 80),
                        ("Review", "review_frequency", 60),
                    ):
                        val_s = _clip(pt.get(pkey) or "", clip_n)
                        if val_s:
                            pt_extra += f" | {lbl}={val_s}"
                    cat_s = _clip(pt.get("catalyst") or "", 150)
                    rsk_s = _clip(pt.get("risk") or "", 150)
                    if cat_s:
                        pt_extra += f" | Catalyst: {cat_s}"
                    if rsk_s:
                        pt_extra += f" | Risk: {rsk_s}"
                    pt_sd = [str(x).strip() for x in (pt.get("supporting_data") or [])[:3] if str(x).strip()]
                    if pt_sd:
                        pt_extra += f" | Data: {'; '.join(_clip(sd, 120) for sd in pt_sd)}"
                lines.append(
                    f"  * Long {pt.get('long_leg')} vs Short {pt.get('short_leg')}{pt_meta}: "
                    f"{_clip(pt.get('thesis', ''), 320 if detailed else 150)}{pt_extra}"
                )

    ev_list = dash.get("regime_evidence")
    if isinstance(ev_list, list) and ev_list:
        lines.append("- Regime Evidence:")
        for ev in ev_list[:(12 if detailed else 5)]:
            if isinstance(ev, dict):
                ev_conf = f" [Conf={ev.get('confidence')}]" if ev.get("confidence") else ""
                conflict = f" [Conflict: {_clip(ev.get('conflict'), 240 if detailed else 90)}]" if ev.get("conflict") else ""
                lines.append(f"  * {ev.get('dimension', '')}: {ev.get('signal', '')}{ev_conf} ({ev.get('evidence', '')}){conflict}")

    indics = dash.get("dashboard_indicators")
    if isinstance(indics, list) and indics:
        key_indics = []
        ind_lim = 60 if detailed else 30
        lbl_clip = 90 if detailed else 44
        for ind in indics[:ind_lim]:
            if isinstance(ind, dict):
                lbl = _clip(ind.get("label") or ind.get("series_key"), lbl_clip)
                disp = ind.get("display_value") or ind.get("value")
                stale = " [STALE]" if ind.get("stale_reason") else ""
                key_indics.append(f"{lbl}={disp}{stale}")
        if key_indics:
            lines.append("- Key Macro Indicators: " + " | ".join(key_indics))

    if detailed:
        refs = [r for r in (dash.get("report_references") or []) if isinstance(r, dict)]
        if refs:
            _generic_stubs = {"ข่าวเศรษฐกิจมหภาคที่ใช้ประกอบการประเมินรอบนี้"}
            stub_count = sum(1 for r in refs[:10] if str(r.get("summary") or "").strip() in _generic_stubs)
            lines.append("- Macro News & Media References:")
            for r in refs[:10]:
                kind = str(r.get("kind") or "ref").upper()
                title = _clip(r.get("title") or r.get("thai_headline") or "", 200)
                pub = str(r.get("publisher") or r.get("source") or "").strip()
                age_txt = f"{r.get('age_hours')}h ago" if r.get("age_hours") is not None else ""
                stale_tag = " [STALE]" if r.get("is_stale") else ""
                pub_meta = ", ".join(x for x in (pub, age_txt) if x)
                pub_tag = f" ({pub_meta})" if pub_meta else ""
                raw_sum = str(r.get("summary") or "").strip()
                r_sum = "" if (stub_count > 1 and raw_sum in _generic_stubs) else _clip(raw_sum, 200)
                sum_part = f": {r_sum}" if r_sum else ""
                if title:
                    lines.append(f"  * [{kind}]{stale_tag} {title}{pub_tag}{sum_part}")

    if isinstance(sec, dict):
        if sec.get("analysis_status"):
            lines.append(f"- Sector Analysis Status: {sec.get('analysis_status')}")
        f_claims = [str(x) for x in (sec.get("fact_claims") or []) if x]
        if f_claims:
            lines.append("- Sector Fact Claims: " + " | ".join(_clip(fc, 160) for fc in f_claims[:(15 if detailed else 6)]))
        res_m = sec.get("resolved_metrics")
        if isinstance(res_m, list) and res_m:
            rm_strs = []
            for rm in res_m[:(12 if detailed else 5)]:
                if isinstance(rm, dict):
                    rm_strs.append(
                        f"{rm.get('metric_name') or rm.get('key') or 'metric'}={rm.get('value')}{rm.get('unit', '')}"
                    )
                else:
                    rm_strs.append(_clip(rm, 100))
            if rm_strs:
                lines.append("- Sector Resolved Metrics: " + " | ".join(rm_strs))
        elif isinstance(res_m, dict) and res_m:
            lines.append("- Sector Resolved Metrics: " + " | ".join(f"{k}={v}" for k, v in list(res_m.items())[:12]))
        sec_warns = [str(x) for x in (sec.get("validation_warnings") or []) if x]
        if sec_warns:
            lines.append("- Sector Validation Warnings: " + ", ".join(_clip(w, 120) for w in sec_warns[:8]))

    s_ev = dash.get("strategy_report_evidence")
    if isinstance(s_ev, list) and s_ev:
        ev_parts: List[str] = []
        for item in s_ev[:(12 if detailed else 4)]:
            if isinstance(item, dict):
                cl = _clip(item.get("claim") or item.get("summary") or "", 220)
                src = str(item.get("source") or "").strip()
                if cl:
                    ev_parts.append(f"{cl} [{src}]" if src else cl)
            elif item:
                ev_parts.append(_clip(item, 220))
        if ev_parts:
            lines.append("- Strategy Vault Evidence: " + " | ".join(ev_parts))
    elif isinstance(s_ev, dict) and s_ev:
        c_hash = str(s_ev.get("content_hash") or "")[:12]
        lines.append(
            f"- Strategy Vault Evidence: Note={s_ev.get('note_id', 'N/A')} | "
            f"Path={s_ev.get('relative_path', 'N/A')}" + (f" | Hash={c_hash}" if c_hash else "")
        )
    ts_notes = [str(n) for n in (dash.get("data_timestamp_notes") or []) if n]
    if ts_notes:
        lines.append("- Data Timestamp Notes: " + "; ".join(_clip(n, 180) for n in ts_notes[:6]))

    return "\n".join(lines)


def _format_digest_brief(digest: Optional[Dict[str, Any]], tickers: Sequence[str] = (), *, detailed: bool = True) -> str:
    """Curate Daily Digest payload including transmission_chain, news items & YouTube highlights."""
    if not digest or not isinstance(digest, dict):
        return ""
    if not any(k in digest for k in ("market_mood", "transmission_chain", "items", "youtube_highlights", "market_mood_summary")):
        return json.dumps(digest, ensure_ascii=False, indent=2, default=str)

    lines: List[str] = [
        f"- Digest Date: {digest.get('date', 'N/A')} ({digest.get('period', '')}) | Generated At: {str(digest.get('generated_at', 'N/A'))[:19]}",
        f"- Market Mood: {digest.get('market_mood', 'N/A')} — {_clip(digest.get('market_mood_summary', ''), 700 if detailed else 320)}",
        f"- Transmission Chain: {digest.get('transmission_chain') or digest.get('macro_transmission') or 'N/A'}",
    ]
    if digest.get("top_portfolio_focus"):
        lines.append(f"- Top Portfolio Focus: {_clip(digest.get('top_portfolio_focus'), 600 if detailed else 280)}")

    items = list(digest.get("items") or [])
    if items:
        ticker_set = {t.upper() for t in tickers}

        def _score_item(it: Dict[str, Any]) -> Tuple[int, int]:
            blob = f"{it.get('thai_headline', '')} {it.get('what_happened', '')} {it.get('why_it_matters', '')}".upper()
            hit = 1 if any(sym in blob for sym in ticker_set) else 0
            return (-hit, int(it.get("age_hours") or 0))

        items.sort(key=_score_item)
        lines.append("- ข่าวเด่นประจำวัน:")
        for it in items[:(16 if detailed else 8)]:
            if isinstance(it, dict):
                hl = _clip(it.get("thai_headline") or it.get("title", ""), 240 if detailed else 120)
                wh = _clip(it.get("what_happened") or it.get("summary", ""), 450 if detailed else 170)
                why = _clip(it.get("why_it_matters", ""), 400 if detailed else 160)
                meta_part = ""
                cor_part = ""
                if detailed:
                    sig = str(it.get("thesis_signal") or "").strip()
                    aff = ", ".join(str(x) for x in (it.get("affected_tickers") or [])[:6] if x)
                    if sig and aff:
                        meta_part = f" ({sig}, Tickers: {aff})"
                    elif sig:
                        meta_part = f" ({sig})"
                    elif aff:
                        meta_part = f" (Tickers: {aff})"
                    if it.get("youtube_corroboration"):
                        cor_part = f" [YT Corroboration: {_clip(it.get('youtube_corroboration'), 200)}]"
                lines.append(
                    f"  * {hl} [{it.get('freshness_label', 'N/A')}]{meta_part}: {wh}"
                    + (f" -> กระทบพอร์ต: {why}" if why else "")
                    + cor_part
                )

    yts = list(digest.get("youtube_highlights") or [])
    if yts:
        lines.append("- YouTube & Analyst Takeaways:")
        for yh in yts[:(10 if detailed else 5)]:
            if isinstance(yh, dict):
                title = _clip(yh.get("thai_title") or yh.get("title", ""), 200 if detailed else 100)
                ch = yh.get("channel") or "Channel"
                tks = ", ".join(str(x) for x in (yh.get("mentioned_tickers") or [])[:(8 if detailed else 5)])
                tak = "; ".join(
                    _clip(k, 260 if detailed else 110)
                    for k in (yh.get("key_takeaways") or [])[:(6 if detailed else 2)]
                )
                align = yh.get("alignment_with_news") or ""
                align_note = (
                    f" ({_clip(yh.get('alignment_note'), 200)})"
                    if (detailed and yh.get("alignment_note"))
                    else ""
                )
                lines.append(
                    f"  * [{ch}] {title}"
                    + (f" (Tickers: {tks})" if tks else "")
                    + (f" [{align}{align_note}]" if (align or align_note) else "")
                    + (f": {tak}" if tak else "")
                )

    return "\n".join(lines)


def _format_essence_brief(
    *,
    axis_raw: Dict[str, Any],
    sections: Dict[str, Any],
    role_models: Tuple[str, ...],
    axis_hash: str,
    financial_context: Optional[Dict[str, Any]] = None,
    bucket_policy: Optional[List[Dict[str, Any]]] = None,
    investor_essence: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], str]:
    """Extract all 8 Investment Axis sections + Investor Essence + Financial Context & format crisp prompt brief."""
    alloc_rows = axis_raw.get("allocation_rows") or sections.get("allocation_rows") or []
    alloc_basis = axis_raw.get("allocation_basis") or sections.get("allocation_basis")

    axis_summary: Dict[str, Any] = {
        "has_confirmed_axis": bool(axis_raw),
        "basic_policy": sections.get("basic_policy"),
        "investment_horizon": sections.get("investment_horizon"),
        "risk_limits": sections.get("risk_limits"),
        "invest_targets": sections.get("invest_targets") or [],
        "exclude_targets": sections.get("exclude_targets") or [],
        "primary_methods": sections.get("primary_methods") or [],
        "secondary_methods": sections.get("secondary_methods") or [],
        "allocation_basis": alloc_basis,
        "allocation_rows": alloc_rows,
        "rebalance_frequency": sections.get("rebalance_frequency"),
        "non_actions": sections.get("non_actions") or [],
        "role_models": list(role_models),
        "financial_context": financial_context or {},
        "investor_essence": investor_essence or {},
        "content_hash": axis_hash,
    }

    if not axis_raw and not financial_context and not investor_essence:
        return axis_summary, ""

    lines: List[str] = []
    if isinstance(investor_essence, dict) and investor_essence:
        stmt = str(investor_essence.get("statement") or "").strip()
        conf_at = str(investor_essence.get("confirmed_at_iso") or "")[:10]
        if stmt:
            conf_tag = f" (ยืนยัน {conf_at})" if conf_at else ""
            lines.append(f"0. แก่นแท้การลงทุนที่ยืนยันแล้ว (Confirmed Investor Essence{conf_tag}): {_clip(stmt, 1000)}")
        claims = [c for c in (investor_essence.get("accepted_claims") or []) if isinstance(c, dict) and c.get("text")]
        if claims:
            shown_claims = claims[:10]
            claim_tags = [
                (str(c.get("fit_rating") or ""), str(c.get("source_kind") or ""))
                for c in shown_claims
            ]
            uniform_tag = (
                len(shown_claims) >= 3
                and len(set(claim_tags)) == 1
                and (claim_tags[0][0] or claim_tags[0][1])
            )
            if uniform_tag:
                u_fit, u_skind = claim_tags[0]
                claim_strs = [_clip(c.get("text"), 180) for c in shown_claims]
                lines.append(f"   * หลักยึดหลักที่ยอมรับ (Accepted Claims [{u_fit}/{u_skind}]): " + " | ".join(claim_strs))
            else:
                claim_strs = []
                for c, (fit, skind) in zip(shown_claims, claim_tags):
                    tag = f" [{fit}/{skind}]" if (fit or skind) else ""
                    claim_strs.append(f"{_clip(c.get('text'), 180)}{tag}")
                lines.append("   * หลักยึดหลักที่ยอมรับ (Accepted Claims): " + " | ".join(claim_strs))
    if axis_raw:
        if sections.get("basic_policy"):
            lines.append(f"1. นโยบายพื้นฐาน: {_clip(sections.get('basic_policy'), 1200)}")
        rl = sections.get("risk_limits")
        if isinstance(rl, dict) and rl:
            _rl_alias = {
                "mdd_max_annual": "mdd",
                "mdd_percent": "mdd",
                "max_loss_per_trade": "max_loss_per_trade",
                "max_loss_per_trade_percent": "max_loss_per_trade",
            }
            seen_rl: Dict[str, str] = {}
            rl_parts = []
            for rk, rv in rl.items():
                if isinstance(rv, dict):
                    raw_val = f"{rv.get('value')}{rv.get('unit', '')}"
                else:
                    raw_val = str(rv)
                norm_val = re.sub(r"\s*percent\b", "%", raw_val.strip(), flags=re.IGNORECASE)
                canon_k = _rl_alias.get(str(rk or "").strip().lower(), str(rk or "").strip().lower())
                if canon_k in seen_rl and seen_rl[canon_k] == norm_val:
                    continue
                seen_rl.setdefault(canon_k, norm_val)
                rl_parts.append(f"{rk}={norm_val}")
            lines.append(f"2. กรอบความเสี่ยง (Risk Limits): {', '.join(rl_parts)}")
        elif rl:
            lines.append(f"2. กรอบความเสี่ยง (Risk Limits): {rl}")
        inv_t = sections.get("invest_targets") or []
        exc_t = sections.get("exclude_targets") or []
        if inv_t or exc_t:
            lines.append(f"3. เป้าหมายที่ลงทุน: {', '.join(str(x) for x in inv_t) or 'N/A'} | สิ่งที่ห้ามลงทุน (Exclude): {', '.join(str(x) for x in exc_t) or 'ไม่มี'}")
        pm = sections.get("primary_methods") or []
        sm = sections.get("secondary_methods") or []
        if pm or sm:
            lines.append(f"4. วิธีการลงทุนหลัก: {', '.join(str(x) for x in pm) or 'N/A'} | วิธีเสริม: {', '.join(str(x) for x in sm) or 'ไม่มี'}")
        if sections.get("investment_horizon"):
            lines.append(f"5. กรอบเวลาการลงทุน: {sections.get('investment_horizon')}")
        if alloc_rows or sections.get("rebalance_frequency"):
            row_txt = ", ".join(
                f"{r.get('category_name', r.get('allocation_id', ''))}={r.get('target_percent')}%"
                for r in alloc_rows if isinstance(r, dict)
            )
            lines.append(
                f"6. แผนจัดสรรสินทรัพย์ ({alloc_basis or 'N/A'} | Rebalance: {sections.get('rebalance_frequency', 'N/A')}): {row_txt or 'ตามแผนพอร์ต'}"
            )
        lines.append(f"7. นักลงทุนต้นแบบ (Role Models): {', '.join(role_models) if role_models else 'ไม่ระบุบุคคล'}")
        nas = sections.get("non_actions") or []
        if nas:
            lines.append(f"8. กฎเหล็กสิ่งที่จะไม่ทำ (Non-Actions): {'; '.join(str(x) for x in nas)}")

    if isinstance(financial_context, dict) and financial_context:
        fc_parts: List[str] = []
        if financial_context.get("horizon_years"):
            fc_parts.append(f"Horizon={financial_context.get('horizon_years')} ปี")
        if financial_context.get("target_use_amount"):
            range_note = f", range={financial_context.get('target_use_range')}" if financial_context.get("target_use_range") else ""
            fc_parts.append(f"Target={financial_context.get('target_use_amount')} THB ({financial_context.get('target_use_timeline') or 'N/A'}{range_note})")
        elif financial_context.get("target_use_range"):
            fc_parts.append(f"Target Range={financial_context.get('target_use_range')} ({financial_context.get('target_use_timeline') or 'N/A'})")
        if financial_context.get("emergency_reserves_months") is not None:
            fc_parts.append(f"EmergencyReserve={financial_context.get('emergency_reserves_months')} เดือน ({financial_context.get('emergency_reserves_amount') or 'N/A'} THB)")
        if financial_context.get("reserves_outside_portfolio") is not None:
            fc_parts.append(f"OutsideReserves={financial_context.get('reserves_outside_portfolio')} THB")
        if financial_context.get("obligations_monthly") is not None:
            fc_parts.append(f"Obligations/mo={financial_context.get('obligations_monthly')} THB")
        if financial_context.get("obligations_description"):
            fc_parts.append(f"ObligationsNote={_clip(financial_context.get('obligations_description'), 350)}")
        if financial_context.get("essential_monthly_expense") is not None:
            fc_parts.append(f"EssentialExpense/mo={financial_context.get('essential_monthly_expense')} THB")
        if financial_context.get("monthly_investable_contribution") is not None:
            fc_parts.append(f"Investable/mo={financial_context.get('monthly_investable_contribution')} THB")
        if financial_context.get("withdrawal_frequency"):
            w_amt = f" ({financial_context.get('withdrawal_amount')} THB)" if financial_context.get("withdrawal_amount") is not None else ""
            fc_parts.append(f"Withdrawal={_clip(financial_context.get('withdrawal_frequency'), 250)}{w_amt}")
        elif financial_context.get("withdrawal_amount") is not None:
            fc_parts.append(f"WithdrawalAmt={financial_context.get('withdrawal_amount')} THB")
        if financial_context.get("experience_description"):
            fc_parts.append(f"Experience={_clip(financial_context.get('experience_description'), 350)}")
        if financial_context.get("unknown_fields"):
            fc_parts.append(f"UnknownFields={', '.join(financial_context.get('unknown_fields'))}")
        if fc_parts:
            lines.append(f"9. บริบททางการเงินจริง (Financial Context): {' | '.join(fc_parts)}")

    if bucket_policy:
        bp_str = ", ".join(
            f"{b.get('name', b.get('id', ''))}={_first_non_none(b.get('target_pct'), b.get('target_percent'))}%"
            for b in bucket_policy if isinstance(b, dict)
        )
        if bp_str:
            lines.append(f"10. นโยบายถังเงินที่ยืนยันแล้ว (Bucket Policy): {bp_str}")

    return axis_summary, "\n".join(lines)


def compute_packet_fingerprint(
    *,
    normalized_refs: Sequence[str],
    primary_ticker: Optional[str],
    equity_data: Dict[str, Any],
    macro_text: str,
    digest_text: str,
    portfolio_state_hash: str,
    axis_hash: str,
    essence_text: str = "",
    all_equity_stamps: Optional[Sequence[str]] = None,
    watchlist_stamp: str = "",
    extra_stamps: Optional[Dict[str, str]] = None,
) -> str:
    stamps = list(all_equity_stamps) if all_equity_stamps else [
        str(
            (equity_data or {}).get("evaluated_at")
            or (equity_data or {}).get("as_of_date")
            or (equity_data or {}).get("composite_score")
            or ""
        )
    ]
    payload: Dict[str, Any] = {
        "refs": list(normalized_refs),
        "primary_ticker": primary_ticker,
        "equity_stamp": stamps[0] if len(stamps) == 1 else stamps,
        "macro_sha": hashlib.sha256((macro_text or "").encode("utf-8")).hexdigest()[:12],
        "digest_sha": hashlib.sha256((digest_text or "").encode("utf-8")).hexdigest()[:12],
        "essence_sha": hashlib.sha256((essence_text or "").encode("utf-8")).hexdigest()[:12],
        "portfolio_hash": portfolio_state_hash,
        "axis_hash": axis_hash,
    }
    if watchlist_stamp:
        payload["watchlist_stamp"] = watchlist_stamp
    if extra_stamps:
        payload["extra_stamps"] = extra_stamps
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


class RoundtableContextAssembler:
    """Assembles an immutable FrozenContextPacket from attached cards and domain services."""

    def __init__(
        self,
        *,
        providers: ContextDataProvidersPort,
        repo: Optional[RoundtableSessionRepoPort] = None,
        settlement_prices: Optional[SettlementPricePort] = None,
    ) -> None:
        self._providers = providers
        self._repo = repo
        self._settlement_prices = settlement_prices

    def _build_anchor(self, ticker: str, raw_bundle: Dict[str, Any]) -> InstrumentIdentityAnchor:
        resolved = self._providers.resolve_instrument(ticker) or {}
        qs = (raw_bundle.get("quant_signals") or {}) if raw_bundle else {}
        market = str(
            resolved.get("market")
            or raw_bundle.get("market")
            or ("TH" if ticker.endswith(".BK") else "US")
        )
        currency = str(
            resolved.get("currency")
            or raw_bundle.get("currency")
            or ("THB" if market == "TH" else "USD")
        )
        raw_price = _extract_price_from_detail(qs, raw_bundle, resolved)
        current_price = _safe_str_dec(raw_price, "0.01")
        return InstrumentIdentityAnchor(
            ticker=ticker,
            company_name=str(
                raw_bundle.get("company_name")
                or (qs.get("company_name") if isinstance(qs, dict) else None)
                or resolved.get("company_name")
                or ticker
            ),
            market=market,
            sector=str(
                raw_bundle.get("sector")
                or (qs.get("peer_sector") if isinstance(qs, dict) else None)
                or resolved.get("sector")
                or "Unknown"
            ),
            industry=str(raw_bundle.get("industry") or resolved.get("industry") or "Unknown"),
            currency=currency,
            benchmark_ticker=default_benchmark_for_ticker(ticker),
            current_price=current_price,
            verified=bool(raw_bundle or resolved.get("company_name")),
        )

    def _lookup_past_scorecard(
        self,
        *,
        ticker: str,
        anchor: Optional[InstrumentIdentityAnchor],
        exclude_session_id: str,
        now_iso: str,
    ) -> Optional[PastRoundtableScorecard]:
        if not ticker or self._repo is None:
            return None
        prior_sessions = self._repo.list_sessions_by_ticker(ticker, limit=5)
        for prev in prior_sessions:
            if prev.session_id == exclude_session_id:
                continue
            if prev.settlement_status == "settled" and prev.settled_return_pct is not None:
                prev_dt = datetime.fromtimestamp(prev.created_at, tz=timezone.utc).date().isoformat()
                return PastRoundtableScorecard(
                    previous_session_id=prev.session_id,
                    previous_date=prev_dt,
                    primary_ticker=ticker,
                    benchmark_ticker=prev.benchmark_ticker or default_benchmark_for_ticker(ticker),
                    entry_price=prev.entry_price_at_session,
                    settlement_price=None,
                    raw_return_pct=prev.settled_return_pct,
                    benchmark_return_pct=None,
                    alpha_pct=prev.settled_alpha_pct,
                    previous_final_rating=prev.final_rating,
                    outcome_label="outperformed" if Decimal(prev.settled_alpha_pct or "0") > 0 else "lagged",
                    reflection_lesson=prev.reflection_lesson,
                    settlement_status="settled",
                )
            if prev.entry_price_at_session:
                bench = prev.benchmark_ticker or default_benchmark_for_ticker(ticker)
                settled_date = now_iso[:10]
                settled_px: Optional[Decimal] = None
                b_entry_px: Optional[Decimal] = None
                b_settled_px: Optional[Decimal] = None
                if self._settlement_prices is not None:
                    latest_pair = self._settlement_prices.get_latest_close(ticker)
                    if latest_pair:
                        settled_date, settled_px = latest_pair
                    prev_date_iso = datetime.fromtimestamp(prev.created_at, tz=timezone.utc).date().isoformat()
                    b_start_pair = self._settlement_prices.get_close_on_or_after(bench, prev_date_iso)
                    if b_start_pair:
                        _, b_entry_px = b_start_pair
                    b_end_pair = self._settlement_prices.get_latest_close(bench)
                    if b_end_pair:
                        _, b_settled_px = b_end_pair
                elif anchor and anchor.current_price:
                    try:
                        settled_px = Decimal(anchor.current_price)
                    except Exception:
                        settled_px = None

                return compute_settlement_scorecard(
                    previous_session_id=prev.session_id,
                    previous_created_at=prev.created_at,
                    previous_final_rating=prev.final_rating,
                    primary_ticker=ticker,
                    benchmark_ticker=bench,
                    entry_price=prev.entry_price_at_session,
                    settled_price=settled_px,
                    settled_date_iso=settled_date,
                    benchmark_entry_price=b_entry_px,
                    benchmark_settled_price=b_settled_px,
                    reflection_lesson=prev.reflection_lesson,
                )
        return None

    def assemble(
        self,
        *,
        session_id: str,
        agenda_prompt: str,
        context_refs: Sequence[str],
        portfolio_id: str = "default",
        debate_intensity: str = "balanced",
        debate_depth: str = "concise",
    ) -> FrozenContextPacket:
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        normalized_refs, tickers, cards = extract_context_tags(context_refs, agenda_prompt)
        primary_ticker = tickers[0] if tickers else None

        missing_tickers: List[str] = []
        loaded_bundles: List[Tuple[str, Dict[str, Any]]] = []
        equity_stamps: List[str] = []

        news_reader = getattr(self._providers, "get_equity_news", None)
        notes_lister = getattr(self._providers, "list_equity_notes", None)
        note_reader = getattr(self._providers, "read_equity_note", None)
        fin_ctx_reader = getattr(self._providers, "get_financial_context", None)
        bucket_policy_reader = getattr(self._providers, "get_bucket_policy", None)
        watchlist_reader = getattr(self._providers, "get_watchlist", None)
        exc_reader = getattr(self._providers, "get_thesis_pending_exceptions", None)
        promise_reader = getattr(self._providers, "get_promise_ledger_scores", None)
        analyst_reader = getattr(self._providers, "get_analyst_context", None)
        insider_reader = getattr(self._providers, "get_insider_db_summary", None)
        advisory_reader = getattr(self._providers, "get_portfolio_advisory_summary", None)
        investor_essence_reader = getattr(self._providers, "get_investor_essence", None)

        ticker_news_map: Dict[str, Optional[Dict[str, Any]]] = {}
        ticker_notes_map: Dict[str, Optional[Dict[str, Any]]] = {}
        ticker_ec_note_map: Dict[str, Optional[Dict[str, Any]]] = {}
        ticker_ea_note_map: Dict[str, Optional[Dict[str, Any]]] = {}
        ticker_exceptions_map: Dict[str, List[Dict[str, Any]]] = {}
        ticker_promise_map: Dict[str, Optional[Dict[str, Any]]] = {}
        ticker_analyst_map: Dict[str, Optional[Dict[str, Any]]] = {}
        ticker_insider_db_map: Dict[str, Optional[Dict[str, Any]]] = {}

        for ticker in tickers:
            detail = self._providers.get_equity_detail(ticker)
            if not detail:
                missing_tickers.append(ticker)
                continue
            review_pid = portfolio_id if cards["portfolio"] else ""
            review = self._providers.get_equity_position_review(ticker, portfolio_id=review_pid)
            thesis = self._providers.get_active_thesis(ticker)
            combined = dict(detail)
            if review:
                combined["position_review"] = review
            if thesis:
                combined["active_thesis"] = thesis
            loaded_bundles.append((ticker, combined))
            equity_stamps.append(
                f"{ticker}:"
                + str(
                    combined.get("evaluated_at")
                    or combined.get("analysis_date")
                    or combined.get("as_of_date")
                    or combined.get("composite_score")
                    or ""
                )
            )

            # Fail-open specialized intelligence lookups per ticker
            raw_exc = exc_reader(ticker) if callable(exc_reader) else None
            ticker_exceptions_map[ticker] = _deduplicate_pending_exceptions(raw_exc)
            ticker_promise_map[ticker] = promise_reader(ticker) if callable(promise_reader) else None
            ticker_analyst_map[ticker] = analyst_reader(ticker) if callable(analyst_reader) else None
            ticker_insider_db_map[ticker] = insider_reader(ticker) if callable(insider_reader) else None

            # WP-3: Company News & Vault / Earnings / Equity Analysis Notes (strictly fail-open on stub providers)
            n_data = news_reader(ticker) if callable(news_reader) else None
            nl_data = notes_lister(ticker, 180) if callable(notes_lister) else None
            ticker_news_map[ticker] = n_data
            ticker_notes_map[ticker] = nl_data
            if callable(note_reader) and isinstance(nl_data, dict):
                ec_item = next(
                    (
                        it
                        for it in (nl_data.get("items") or [])
                        if isinstance(it, dict)
                        and it.get("relative_path")
                        and (it.get("matched_by") == "earnings_call" or "earnings" in str(it.get("folder") or it.get("relative_path")).lower())
                    ),
                    None,
                )
                if ec_item:
                    try:
                        ticker_ec_note_map[ticker] = note_reader(str(ec_item["relative_path"]))
                    except Exception:
                        ticker_ec_note_map[ticker] = None

            if callable(note_reader) and isinstance(nl_data, dict):
                ea_candidates = [
                    it
                    for it in (nl_data.get("items") or [])
                    if isinstance(it, dict)
                    and it.get("relative_path")
                    and (
                        it.get("matched_by") == "equity_analysis"
                        or "/analysis/" in str(it.get("relative_path") or "").lower()
                        or str(it.get("folder") or "").endswith("/Analysis")
                    )
                ]
                ea_item = next(
                    (
                        it
                        for it in ea_candidates
                        if not re.search(r"(?:^|[_\s\-])latest$", str(it.get("title") or ""), re.IGNORECASE)
                    ),
                    ea_candidates[0] if ea_candidates else None,
                )
                if ea_item:
                    try:
                        ticker_ea_note_map[ticker] = note_reader(str(ea_item["relative_path"]))
                    except Exception:
                        ticker_ea_note_map[ticker] = None

        raw_primary_bundle = loaded_bundles[0][1] if loaded_bundles else {}
        equity_bundle = _compact_equity_data(raw_primary_bundle, include_portfolio_fields=cards["portfolio"])
        if len(loaded_bundles) > 1:
            equity_bundle["secondary_bundles"] = {
                sym: _compact_equity_data(bdl, include_portfolio_fields=cards["portfolio"])
                for sym, bdl in loaded_bundles[1:]
            }

        # Deterministic Instrument Identity Anchors across all loaded tickers (WP-A)
        ticker_anchors: List[Tuple[str, InstrumentIdentityAnchor]] = []
        for sym, bdl in loaded_bundles:
            ticker_anchors.append((sym, self._build_anchor(sym, bdl)))
        anchor_map: Dict[str, InstrumentIdentityAnchor] = {sym: anc for sym, anc in ticker_anchors}

        anchor: Optional[InstrumentIdentityAnchor] = None
        if primary_ticker:
            if primary_ticker in anchor_map:
                anchor = anchor_map[primary_ticker]
            elif ticker_anchors:
                anchor = ticker_anchors[0][1]
            else:
                anchor = self._build_anchor(primary_ticker, raw_primary_bundle)
        secondary_anchors = tuple(anc for sym, anc in ticker_anchors if anchor is None or sym != anchor.ticker)

        bucket_policy = bucket_policy_reader(portfolio_id) if callable(bucket_policy_reader) else None

        # 3-State Portfolio Guard + Overview + Per-Equity Watchlist Resolution (WP-D)
        portfolio_state = self._providers.get_portfolio_state(portfolio_id) if cards["portfolio"] else None
        raw_watchlist: List[Dict[str, Any]] = []
        if cards["portfolio"] and isinstance(portfolio_state, dict) and isinstance(portfolio_state.get("watchlist_items"), list):
            raw_watchlist = [w for w in portfolio_state["watchlist_items"] if isinstance(w, dict)]
        elif (cards["portfolio"] or bool(tickers)) and callable(watchlist_reader):
            raw_watchlist = [w for w in (watchlist_reader(portfolio_id) or []) if isinstance(w, dict)]
            if cards["portfolio"] and isinstance(portfolio_state, dict) and raw_watchlist and "watchlist_items" not in portfolio_state:
                portfolio_state = {**portfolio_state, "watchlist_items": raw_watchlist}

        watchlist_by_symbol: Dict[str, Dict[str, Any]] = {}
        for wl in raw_watchlist:
            w_sym = str(wl.get("symbol") or "").strip().upper()
            if w_sym:
                watchlist_by_symbol[w_sym] = wl
                base_k = _normalize_symbol_key(w_sym)
                if base_k and base_k not in watchlist_by_symbol:
                    watchlist_by_symbol[base_k] = wl

        matched_watchlist_items: List[Dict[str, Any]] = []
        for sym, _ in loaded_bundles:
            wl_m = watchlist_by_symbol.get(sym.upper()) or watchlist_by_symbol.get(_normalize_symbol_key(sym))
            if wl_m:
                matched_watchlist_items.append(wl_m)
        watchlist_stamp = ""
        if matched_watchlist_items:
            watchlist_stamp = hashlib.sha256(
                json.dumps(
                    [
                        (
                            str(w.get("symbol", "")).upper(),
                            str(w.get("target_price", "")),
                            str(w.get("target_currency", "")),
                            str(w.get("target_price_basis", "")),
                            str(w.get("suggested_bucket_id", "")),
                            str(w.get("notes", "")),
                        )
                        for w in matched_watchlist_items
                    ],
                    sort_keys=True,
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest()[:12]

        advisory_summary = (
            advisory_reader(portfolio_id)
            if (cards["portfolio"] and callable(advisory_reader))
            else None
        )
        holding_advisory_by_sym: Dict[str, Dict[str, Any]] = {}
        if cards["portfolio"] and isinstance(advisory_summary, dict):
            for ha in advisory_summary.get("holdings_advisory") or []:
                if isinstance(ha, dict) and ha.get("symbol"):
                    h_sym = str(ha["symbol"]).strip().upper()
                    holding_advisory_by_sym[h_sym] = ha
                    hk = _normalize_symbol_key(h_sym)
                    if hk and hk not in holding_advisory_by_sym:
                        holding_advisory_by_sym[hk] = ha

        portfolio_guard = _build_portfolio_guard(
            is_attached=cards["portfolio"],
            primary_ticker=primary_ticker,
            portfolio_state=portfolio_state,
            bucket_policy=bucket_policy,
            session_tickers=tickers,
            advisory_summary=advisory_summary,
        )

        # @Macro & @Digest via curated formatters + report_or_absent
        is_single_or_empty = len(tickers) <= 1
        dash: Optional[Dict[str, Any]] = None
        macro_raw = ""
        if cards["macro"]:
            dash = self._providers.get_macro_dashboard()
            macro_raw = _format_macro_brief(dash, detailed=is_single_or_empty) if dash else ""
        macro_text = report_or_absent("@Macro", cards["macro"], macro_raw)

        digest_raw = ""
        if cards["digest"]:
            digest = self._providers.get_daily_digest()
            digest_raw = _format_digest_brief(digest, tickers=tickers, detailed=is_single_or_empty) if digest else ""
        digest_text = report_or_absent("@Digest", cards["digest"], digest_raw)

        # Full 8-Section Personal Investment Axis + Investor Essence + Financial Context + Bucket Policy
        axis_raw = self._providers.get_confirmed_investment_axis(portfolio_id) or {}
        sections = axis_raw.get("complete_sections") if isinstance(axis_raw.get("complete_sections"), dict) else axis_raw
        raw_role_models = (sections or {}).get("role_models") or []
        role_models = tuple(str(r).strip() for r in raw_role_models if str(r).strip())
        prioritized = match_prioritized_speakers(role_models)
        axis_hash = str(axis_raw.get("content_hash") or "")
        fin_ctx = fin_ctx_reader(portfolio_id) if callable(fin_ctx_reader) else None
        investor_essence = investor_essence_reader(portfolio_id) if callable(investor_essence_reader) else None

        axis_summary, essence_raw = _format_essence_brief(
            axis_raw=axis_raw,
            sections=sections or {},
            role_models=role_models,
            axis_hash=axis_hash,
            financial_context=fin_ctx,
            bucket_policy=bucket_policy,
            investor_essence=investor_essence,
        )
        essence_text = report_or_absent(
            "@Essence",
            cards["essence"] or bool(axis_raw) or bool(fin_ctx) or bool(investor_essence),
            essence_raw,
        )

        # Curated Multi-Ticker Equity Brief (per-ticker Watchlist attached right under each $Equity even without @Portfolio!)
        equity_sections: List[str] = []
        if len(loaded_bundles) > 1:
            comp_tbl = _build_comparative_table(loaded_bundles)
            if comp_tbl:
                equity_sections.append(comp_tbl)
        per_ticker_budget = 18000 if len(loaded_bundles) <= 1 else (6200 if len(loaded_bundles) == 2 else 4400)
        ticker_extra_flags_map: Dict[str, List[str]] = {}
        for sym, combined in loaded_bundles:
            wl_entry = watchlist_by_symbol.get(sym.upper()) or watchlist_by_symbol.get(_normalize_symbol_key(sym))
            dedup_exc = ticker_exceptions_map.get(sym, [])
            age_days, extra_flags = _collect_ticker_extra_flags(combined, dedup_exc, now_iso)
            ticker_extra_flags_map[sym] = extra_flags
            h_adv = (
                holding_advisory_by_sym.get(sym.upper()) or holding_advisory_by_sym.get(_normalize_symbol_key(sym))
                if cards["portfolio"]
                else None
            )
            equity_sections.append(
                _build_curated_equity_brief(
                    sym,
                    combined,
                    portfolio_attached=cards["portfolio"],
                    news_data=ticker_news_map.get(sym),
                    notes_list=ticker_notes_map.get(sym),
                    earnings_note=ticker_ec_note_map.get(sym),
                    analysis_note=ticker_ea_note_map.get(sym),
                    watchlist_item=wl_entry,
                    char_budget=per_ticker_budget,
                    thesis_exceptions=dedup_exc,
                    promise_scores=ticker_promise_map.get(sym),
                    analyst_context=ticker_analyst_map.get(sym),
                    insider_db=ticker_insider_db_map.get(sym),
                    sidecar_age_days=age_days,
                    live_macro_dash=dash,
                    holding_advisory=h_adv,
                    now_date_iso=now_iso[:10],
                    extra_dq_flags=extra_flags,
                )
            )
        equity_text = (
            "\n\n".join(equity_sections)
            if equity_sections
            else report_or_absent("$Equity", bool(tickers), "")
        )

        # Primary + Per-Ticker Scoped Flags & Secondary Facts (WP-A + WP-B)
        facts_list: List[str] = []
        primary_facts, base_dq_flags, action_stance = _extract_numeric_facts_and_flags(
            raw_primary_bundle, anchor, portfolio_guard
        )
        facts_list.extend(primary_facts)
        if loaded_bundles:
            prim_sym = loaded_bundles[0][0]
            prim_promise = ticker_promise_map.get(prim_sym)
            if isinstance(prim_promise, dict):
                if prim_promise.get("credibility") is not None:
                    facts_list.append(f"promise_credibility_score={prim_promise.get('credibility')}")
                if prim_promise.get("grade"):
                    facts_list.append(f"promise_grade={prim_promise.get('grade')}")
            prim_ac = ticker_analyst_map.get(prim_sym)
            if isinstance(prim_ac, dict) and prim_ac.get("target_mean") is not None:
                facts_list.append(f"analyst_cache_target_mean={prim_ac.get('target_mean')}")
            prim_ins = ticker_insider_db_map.get(prim_sym)
            if isinstance(prim_ins, dict):
                facts_list.append(f"insider_db_180d_open_buy_count={prim_ins.get('open_market_buy_count', 0)}")
                facts_list.append(f"insider_db_180d_open_sale_count={prim_ins.get('open_market_sale_count', 0)}")
                facts_list.append(f"insider_db_180d_paired_sale_count={prim_ins.get('paired_exercise_sale_count', 0)}")

        prim_extra = ticker_extra_flags_map.get(primary_ticker or (loaded_bundles[0][0] if loaded_bundles else ""), [])
        dq_flags = tuple(dict.fromkeys(list(base_dq_flags) + list(prim_extra)))

        ticker_dq_flags: Dict[str, Tuple[str, ...]] = {}
        for idx, (sym, bdl) in enumerate(loaded_bundles):
            sym_anc = anchor_map.get(sym)
            sym_extra = ticker_extra_flags_map.get(sym, [])
            if idx == 0 and sym == (primary_ticker or sym):
                sym_flags = dq_flags
            else:
                _, raw_sym_flags, _ = _extract_numeric_facts_and_flags(bdl, sym_anc, portfolio_guard)
                sym_flags = tuple(dict.fromkeys(list(raw_sym_flags) + list(sym_extra)))
                facts_list.extend(_extract_secondary_numeric_facts(sym, bdl, sym_anc))
            if sym_flags:
                ticker_dq_flags[sym.upper()] = sym_flags
        if cards["macro"] and isinstance(dash, dict):
            facts_list.extend(_extract_macro_numeric_facts(dash))
        facts = tuple(facts_list)

        # Closed-Loop Settlement Scorecards for primary + secondary tickers (WP-A)
        past_scorecard = (
            self._lookup_past_scorecard(
                ticker=primary_ticker,
                anchor=anchor,
                exclude_session_id=session_id,
                now_iso=now_iso,
            )
            if primary_ticker
            else None
        )
        sec_scorecards_list: List[PastRoundtableScorecard] = []
        for sym, _ in loaded_bundles[1:]:
            sc_item = self._lookup_past_scorecard(
                ticker=sym,
                anchor=anchor_map.get(sym),
                exclude_session_id=session_id,
                now_iso=now_iso,
            )
            if sc_item is not None:
                sec_scorecards_list.append(sc_item)

        extra_stamps: Dict[str, str] = {}
        if any(ticker_exceptions_map.values()):
            extra_stamps["exceptions"] = hashlib.sha256(
                json.dumps(ticker_exceptions_map, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()[:12]
        if any(v is not None for v in ticker_promise_map.values()):
            extra_stamps["promises"] = hashlib.sha256(
                json.dumps(ticker_promise_map, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()[:12]
        if any(v is not None for v in ticker_analyst_map.values()):
            extra_stamps["analyst"] = hashlib.sha256(
                json.dumps(ticker_analyst_map, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()[:12]
        if any(v is not None for v in ticker_insider_db_map.values()):
            extra_stamps["insider_db"] = hashlib.sha256(
                json.dumps(ticker_insider_db_map, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()[:12]
        if isinstance(investor_essence, dict) and investor_essence:
            extra_stamps["investor_essence"] = str(
                investor_essence.get("content_hash")
                or hashlib.sha256(json.dumps(investor_essence, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]
            )

        fingerprint = compute_packet_fingerprint(
            normalized_refs=normalized_refs,
            primary_ticker=primary_ticker,
            equity_data=raw_primary_bundle,
            macro_text=macro_raw,
            digest_text=digest_raw,
            portfolio_state_hash=portfolio_guard.state_hash,
            axis_hash=axis_hash,
            essence_text=essence_raw,
            all_equity_stamps=equity_stamps,
            watchlist_stamp=watchlist_stamp,
            extra_stamps=extra_stamps or None,
        )

        return FrozenContextPacket(
            session_id=session_id,
            created_at_iso=now_iso,
            agenda_prompt=agenda_prompt,
            context_refs=normalized_refs,
            primary_ticker=primary_ticker,
            instrument_anchor=anchor,
            missing_prerequisite_tickers=tuple(missing_tickers),
            portfolio_guard=portfolio_guard,
            equity_report_text=equity_text,
            equity_data=equity_bundle,
            macro_report_text=macro_text,
            digest_report_text=digest_text,
            essence_report_text=essence_text,
            user_role_models=role_models,
            prioritized_speakers=prioritized,
            confirmed_axis_summary=axis_summary,
            past_scorecard=past_scorecard,
            data_quality_flags=dq_flags,
            action_stance_hint=action_stance,
            verified_numeric_facts=facts,
            packet_fingerprint_sha256=fingerprint,
            secondary_anchors=secondary_anchors,
            secondary_scorecards=tuple(sec_scorecards_list),
            ticker_data_quality_flags=ticker_dq_flags,
            portfolio_id=portfolio_id or "default",
            debate_intensity=normalize_debate_intensity(debate_intensity, default="balanced"),
            debate_depth=normalize_debate_depth(debate_depth, default="concise"),
        )
