"""Consistency & TTS-Safety Verifier for Trinity Guru Roundtable turns and playbooks.

Architecture Layer: validators/
Dependencies: Standard Library + schemas + core domain + application.roundtable.ports.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import re
from typing import Dict, List, Optional, Sequence, Tuple

from application.roundtable.ports import RoundtableVerificationResult
from core.roundtable.models import (
    ContextAttachmentState,
    FrozenContextPacket,
    RoundtableStance,
    SpeakerId,
    normalize_5tier_stance,
    resolve_personas_from_packet,
)
from schemas.roundtable_schemas import (
    DistilledDecisionPlaybook,
    RoundtableTurnPayload,
)

_CANONICAL_STANCES = {s.value for s in RoundtableStance}

_DCF_CLAIM_RE = re.compile(
    r"(?i)(?:"
    r"ราคาเป้าหมาย(?:ตาม|จาก)?\s*(?<!reverse[\s\-])dcf(?:\s+(?:ของ|สำหรับ)(?:\s*หุ้น)?\s+\$?(?P<tk1>[A-Za-z0-9]{1,10}(?:\.[A-Za-z]{1,4})?))?\s*(?:คือ|อยู่ที่|ประมาณ|=)\s*[\$฿]?\d+"
    r"|ประเมินมูลค่าด้วย\s*(?<!reverse[\s\-])dcf(?:\s+(?:ของ|สำหรับ)(?:\s*หุ้น)?\s+\$?(?P<tk2>[A-Za-z0-9]{1,10}(?:\.[A-Za-z]{1,4})?))?\s*ได้ที่\s*[\$฿]?\d+"
    r"|(?<!reverse[\s\-])dcf\s+fair\s+value(?:\s+(?:for|of)\s+\$?(?P<tk3>[A-Za-z0-9]{1,10}(?:\.[A-Za-z]{1,4})?))?\s*(?:is|at|=)\s*[\$฿]?\d+"
    r")"
)
_CLAUSE_BOUNDARY_RE = re.compile(r"(?:\n+|[!?;]+\s*|,\s+|(?<!\d)\.(?!\d)\s*|\s+ส่วน\s+|\s+ขณะที่\s+|\s+ต่างจาก\s+)")
_CURRENT_PRICE_CLAIM_RE = re.compile(
    r"(?i)(?:ราคา(?:หุ้น)?ปัจจุบัน|ราคาตอนนี้|ราคาตลาดล่าสุด|current\s+(?:share\s+)?price)"
    r"(?:\s*(?:ของ|สำหรับ)?\s*\$?(?P<tk>[A-Za-z0-9]{1,8}(?:\.[A-Za-z]{1,4})?))?"
    r"\s*(?:คือ|อยู่ที่|ประมาณ|=|is|at)\s*[\$฿]?(?P<px>\d+(?:,\d{3})*(?:\.\d+)?)"
)
_PORTFOLIO_HOLDING_CLAIM_RE = re.compile(
    r"(?:คุณถืออยู่\s*\d+|คุณมี\s*\d+(?:\.\d+)?\s*หุ้น|ถือครองอยู่\s*\d+(?:\.\d+)?\s*หุ้น|"
    r"ในพอร์ตของคุณมี(?:หุ้น\s*\$?[A-Za-z0-9.\-]{1,10}\s*)?อยู่\s*\d+|"
    r"ต้นทุนเฉลี่ยของคุณ(?:อยู่ที่)?\s*\d+|"
    r"(?:น้ำหนัก|สัดส่วน(?:ใน)?พอร์ต)ของคุณ(?:ตอนนี้)?\s*\d+%?)"
)
_CONDITIONAL_PREFIX_RE = re.compile(
    r"(?i)^\s*(?:(?:if|assum(?:e|ing)|hypothetical(?:ly)?|under\s+(?:the\s+)?assumption)\b|"
    r"\u0e16\u0e49\u0e32|\u0e2b\u0e32\u0e01|\u0e2a\u0e21\u0e21\u0e15\u0e34(?:\u0e27\u0e48\u0e32)?|"
    r"\u0e20\u0e32\u0e22\u0e43\u0e15\u0e49\u0e2a\u0e21\u0e21\u0e15\u0e34\u0e10\u0e32\u0e19|"
    r"\u0e43\u0e19\u0e01\u0e23\u0e13\u0e35\u0e17\u0e35\u0e48)"
)
_CONDITIONAL_SUFFIX_RE = re.compile(
    r"(?i)^\s*(?:,\s*)?(?:(?:if|assuming)\b|"
    r"(?:in|under)\s+(?:(?:this|that|the)\s+)?hypothetical\s+scenario\b|"
    r"under\s+(?:the\s+)?assumption\b|"
    r"\u0e16\u0e49\u0e32|\u0e2b\u0e32\u0e01|\u0e2a\u0e21\u0e21\u0e15\u0e34(?:\u0e27\u0e48\u0e32)?|"
    r"\u0e20\u0e32\u0e22\u0e43\u0e15\u0e49\u0e2a\u0e21\u0e21\u0e15\u0e34\u0e10\u0e32\u0e19|"
    r"\u0e43\u0e19\u0e01\u0e23\u0e13\u0e35\u0e17\u0e35\u0e48)"
)
_SENTENCE_BOUNDARY_RE = re.compile(r"(?:[!?;\n]+|(?<!\d)\.(?!\d)\s*)")


@dataclass(frozen=True)
class _ClaimContext:
    field_path: str = ""
    parent_text: str = ""

_SOFT_AGREEMENT_OPEN_RE = re.compile(
    r"^\s*(?:ผม\s*)?(?:เห็นด้วย(?:อย่างยิ่ง)?กับ|เห็นพ้องกับ|ใช่เลย(?:ครับ|ค่ะ)|ถูกต้อง(?:เลย)?(?:ครับ|ค่ะ))"
)
_SKEPTIC_RISK_SIGNAL_RE = re.compile(
    r"(?i)(?:kill[\s\-]?switch|falsifier|stop[\s\-]?loss|drawdown|mdd|net[\s\-]?debt|icr|coverage|"
    r"dilution|insider|flag|bear|tail[\s\-]?risk|ต่ำกว่า|หลุด|หดตัว|ขาดทุน|ความเสี่ยง|จุดตาย|ชะลอ|หนี้|เพิ่มทุน|<|>)"
)
_ECHO_STOPWORDS = frozenset(
    {
        "ครับ",
        "ค่ะ",
        "นะครับ",
        "นะค่ะ",
        "ของ",
        "ที่",
        "และ",
        "กับ",
        "ใน",
        "จาก",
        "โดย",
        "เป็น",
        "อยู่",
        "ได้",
        "ไม่",
        "ว่า",
        "คือ",
        "สำหรับ",
        "หุ้น",
        "ราคา",
        "มุมมอง",
        "ประเด็น",
        "รอบนี้",
        "ตอนนี้",
        "การ",
        "the",
        "and",
        "for",
        "with",
        "from",
        "this",
        "that",
    }
)


def _extract_semantic_tokens(text: str) -> set[str]:
    clean = re.sub(r"@[\w.\-]+", " ", (text or "").lower())
    raw_tokens = re.findall(r"[a-z0-9\u0E00-\u0E7F]{2,}", clean)
    return {tok for tok in raw_tokens if tok not in _ECHO_STOPWORDS}


def _polarity_sign(stance: str) -> int:
    norm = normalize_5tier_stance(stance)
    if norm in ("BUY", "OVERWEIGHT"):
        return 1
    if norm in ("HOLD", "UNDERWEIGHT", "SELL"):
        return -1
    return 0


_CYRILLIC_TO_LATIN: Dict[str, str] = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "yo",
    "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "h", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "shch", "ъ": "",
    "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya", "і": "i", "ї": "yi",
    "є": "ye", "ґ": "g",
}
_CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]+")
_TTS_ABBREV_PATTERNS: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bEV/EBITDA\b", re.IGNORECASE), "อีวีต่ออีบิทด้า"),
    (re.compile(r"\bP/BV\b", re.IGNORECASE), "พีบีวี"),
    (re.compile(r"\bP/E\b", re.IGNORECASE), "พีอี"),
    (re.compile(r"\bYoY\b"), "เทียบปีต่อปี"),
    (re.compile(r"\bQoQ\b"), "เทียบไตรมาสก่อน"),
    (re.compile(r">="), " มากกว่าหรือเท่ากับ "),
    (re.compile(r"<="), " น้อยกว่าหรือเท่ากับ "),
)


def _normalize_mixed_cyrillic_scripts(text: str) -> str:
    """Repair stray Cyrillic characters leaking into Thai loanwords (e.g. อгресซีฟ -> อเกรสซีฟ)."""
    if not text or not _CYRILLIC_RE.search(text):
        return text
    repaired = re.sub(r"อ[гГ][рР][еЕ][сС]", "อเกรส", text)
    return _CYRILLIC_RE.sub(
        lambda m: "".join(_CYRILLIC_TO_LATIN.get(ch.lower(), "") for ch in m.group(0)),
        repaired,
    )


def sanitize_spoken_dialogue_for_tts(text: str) -> str:
    """Strip markdown artifacts and mixed-script glitches so spoken dialogue reads naturally."""
    cleaned = text or ""
    cleaned = re.sub(r"```[\s\S]*?```", "", cleaned)
    cleaned = re.sub(r"[*_`#>|]+", "", cleaned)
    cleaned = re.sub(r"\[(?:FACT|MATH|INFERRED|DOCUMENTED|UNVERIFIED)[^\]]*\]", "", cleaned, flags=re.IGNORECASE)
    cleaned = _normalize_mixed_cyrillic_scripts(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def prepare_dialogue_for_tts(text: str) -> str:
    """Normalize symbols, percentages, and financial shorthands for natural Thai TTS delivery."""
    cleaned = sanitize_spoken_dialogue_for_tts(text)
    if not cleaned:
        return ""
    cleaned = re.sub(r"(?<!\w)@([A-Za-z\u0E00-\u0E7F_]+)", r"\1", cleaned)
    cleaned = re.sub(r"\$([A-Z]{1,6}(?:\.[A-Z]{1,4})?)", r"\1", cleaned)
    cleaned = re.sub(r"(\d+(?:,\d{3})*(?:\.\d+)?)\s*%", r"\1 เปอร์เซ็นต์", cleaned)
    for pattern, replacement in _TTS_ABBREV_PATTERNS:
        cleaned = pattern.sub(replacement, cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


class RoundtableConsistencyVerifier:
    """Validates guru turns and Hermes playbooks against FrozenContextPacket invariants."""

    _DCF_UNAVAILABLE_FLAGS = {
        "negative_fcf_dcf_unavailable:dcf",
        "beta_unavailable_dcf_unavailable:dcf",
        "non_actionable_dcf:valuation",
    }

    @staticmethod
    def _packet_tickers(packet: FrozenContextPacket) -> List[str]:
        tickers: set[str] = set()
        if packet.primary_ticker:
            tickers.add(packet.primary_ticker.upper())
        if packet.instrument_anchor and packet.instrument_anchor.ticker:
            tickers.add(packet.instrument_anchor.ticker.upper())
        for anchor in getattr(packet, "secondary_anchors", ()) or ():
            if anchor.ticker:
                tickers.add(anchor.ticker.upper())
        for symbol in (getattr(packet, "ticker_data_quality_flags", None) or {}):
            if symbol:
                tickers.add(symbol.upper())
        return sorted(tickers)

    @staticmethod
    def _ticker_mentions(text: str, tickers: Sequence[str]) -> List[Tuple[int, str]]:
        found: List[Tuple[int, str]] = []
        for ticker in tickers:
            for match in re.finditer(
                rf"(?<![A-Za-z0-9])\$?{re.escape(ticker)}(?![A-Za-z0-9])",
                text,
                re.IGNORECASE,
            ):
                found.append((match.start(), ticker))
        return sorted(found, key=lambda item: item[0])

    def _resolve_claim_ticker(
        self,
        text: str,
        start: int,
        end: int,
        packet: FrozenContextPacket,
        context: Optional[_ClaimContext] = None,
        explicit_ticker: Optional[str] = None,
    ) -> Tuple[Optional[str], bool]:
        """Resolve a claim only from its own span, its parent field, or an unambiguous packet."""
        tickers = self._packet_tickers(packet)
        if explicit_ticker:
            explicit = explicit_ticker.strip().upper()
            return (explicit, False) if explicit in tickers else (None, False)

        clause_start = 0
        clause_end = len(text)
        for boundary in _CLAUSE_BOUNDARY_RE.finditer(text):
            if boundary.end() <= start:
                clause_start = boundary.end()
            elif boundary.start() >= end:
                clause_end = boundary.start()
                break
        local_mentions = self._ticker_mentions(text[clause_start:clause_end], tickers)
        if local_mentions:
            relative_start = start - clause_start
            closest_distance = min(abs(position - relative_start) for position, _ in local_mentions)
            closest = {
                ticker for position, ticker in local_mentions
                if abs(position - relative_start) == closest_distance
            }
            if len(closest) == 1:
                return next(iter(closest)), False
            return None, True

        parent_mentions = self._ticker_mentions((context.parent_text if context else ""), tickers)
        parent_scope = {ticker for _, ticker in parent_mentions}
        if len(parent_scope) == 1:
            return next(iter(parent_scope)), False
        if len(parent_scope) > 1:
            return None, True
        if len(tickers) == 1:
            return tickers[0], False
        if len(tickers) > 1:
            return None, True
        if set(packet.data_quality_flags or ()).intersection(self._DCF_UNAVAILABLE_FLAGS):
            return "__PRIMARY__", False
        return None, False

    def _find_violated_dcf_tickers(
        self,
        combined_text: str,
        packet: FrozenContextPacket,
        context: Optional[_ClaimContext] = None,
    ) -> Tuple[List[str], List[str]]:
        blocked: set[str] = set()
        for symbol, flags in (getattr(packet, "ticker_data_quality_flags", None) or {}).items():
            if set(flags or ()).intersection(self._DCF_UNAVAILABLE_FLAGS):
                blocked.add(symbol.upper())
        if set(packet.data_quality_flags or ()).intersection(self._DCF_UNAVAILABLE_FLAGS):
            if packet.primary_ticker:
                blocked.add(packet.primary_ticker.upper())
            elif packet.instrument_anchor and packet.instrument_anchor.ticker:
                blocked.add(packet.instrument_anchor.ticker.upper())
            else:
                blocked.add("__PRIMARY__")
        if not blocked:
            return [], []

        violated: List[str] = []
        ambiguous: List[str] = []
        for match in _DCF_CLAIM_RE.finditer(combined_text):
            if self._has_explicit_conditional_context(combined_text, match.start(), match.end()):
                continue
            explicit_ticker = (match.group("tk1") or match.group("tk2") or match.group("tk3") or "").strip()
            target, is_ambiguous = self._resolve_claim_ticker(
                combined_text,
                match.start(),
                match.end(),
                packet,
                context=context,
                explicit_ticker=explicit_ticker or None,
            )
            if is_ambiguous:
                ambiguous.append("DCF")
            elif target in blocked:
                violated.append(target)
        return list(dict.fromkeys(violated)), list(dict.fromkeys(ambiguous))

    def _find_invalid_speaker_attributions(
        self,
        turn: RoundtableTurnPayload,
        packet: FrozenContextPacket,
        unspoken_speakers: Sequence[str],
        prior_turns: Sequence[RoundtableTurnPayload],
    ) -> List[str]:
        personas = resolve_personas_from_packet(packet)
        persona_map = {p.speaker_id.value: p for p in personas}
        disabled_ids = {
            p.speaker_id.value
            for p in personas
            if p.speaker_id != SpeakerId.HERMES_MODERATOR and not p.enabled
        }
        already_spoke_ids = {
            t.speaker_id for t in (prior_turns or ()) if t.speaker_id not in ("user",)
        }
        strictly_unspoken_ids = {
            sid for sid in (unspoken_speakers or ())
            if sid and sid != turn.speaker_id and sid not in already_spoke_ids
        }
        blocked_ids = disabled_ids | strictly_unspoken_ids
        if not blocked_ids:
            return []

        bad_refs: List[str] = []
        reply_to = (turn.replying_to_speaker_id or "").strip()
        if reply_to and reply_to in blocked_ids:
            spec = persona_map.get(reply_to)
            bad_refs.append(spec.short_name if spec else reply_to)

        spoken = (turn.spoken_dialogue or "").strip()
        if not spoken:
            return list(dict.fromkeys(bad_refs))

        for sid in sorted(blocked_ids):
            spec = persona_map.get(sid)
            aliases: List[str] = [sid]
            if spec and (spec.short_name or "").strip():
                aliases.append(spec.short_name.strip())
            if spec and (spec.display_name or "").strip():
                aliases.append(spec.display_name.strip())
            base_short = sid.split("_", 1)[0]
            if base_short and base_short not in aliases:
                aliases.append(base_short)

            matched = False
            for alias in aliases:
                esc = re.escape(alias)
                pat = re.compile(
                    rf"(?i)(?:"
                    rf"(?:ตามที่|อย่างที่|เห็นด้วยกับ|ไม่เห็นด้วยกับ|ขอแย้ง|ตอบกลับ)\s*(?:คุณ\s*|@){esc}(?=\s|$|[,\.!?:;])"
                    rf"|(?:คุณ\s*|@){esc}\s*(?:ได้|เพิ่ง|เคย)?\s*(?:พูด|กล่าว|บอก|มอง|ระบุ|เตือน|แย้ง|เสนอ|ย้ำ|ชี้|เห็น)"
                    rf")"
                )
                if pat.search(spoken):
                    matched = True
                    break
            if matched:
                lbl = spec.short_name if spec else sid
                if lbl not in bad_refs:
                    bad_refs.append(lbl)
        return bad_refs

    def _find_price_contradictions(
        self,
        combined_text: str,
        packet: FrozenContextPacket,
        context: Optional[_ClaimContext] = None,
    ) -> Tuple[List[str], List[str]]:
        anchor_prices: Dict[str, Tuple[Decimal, str]] = {}
        if packet.instrument_anchor and packet.instrument_anchor.current_price:
            try:
                price = Decimal(str(packet.instrument_anchor.current_price).replace(",", ""))
                if price > 0:
                    ticker = (packet.instrument_anchor.ticker or packet.primary_ticker or "").upper()
                    anchor_prices[ticker] = (price, packet.instrument_anchor.currency)
            except (InvalidOperation, ValueError):
                pass
        for anchor in getattr(packet, "secondary_anchors", ()) or ():
            if anchor.ticker and anchor.current_price:
                try:
                    price = Decimal(str(anchor.current_price).replace(",", ""))
                    if price > 0:
                        anchor_prices[anchor.ticker.upper()] = (price, anchor.currency)
                except (InvalidOperation, ValueError):
                    pass
        if not anchor_prices:
            return [], []

        mismatches: List[str] = []
        ambiguous: List[str] = []
        for match in _CURRENT_PRICE_CLAIM_RE.finditer(combined_text):
            if self._has_explicit_conditional_context(combined_text, match.start(), match.end()):
                continue
            explicit_ticker = (match.group("tk") or "").strip()
            target_ticker, is_ambiguous = self._resolve_claim_ticker(
                combined_text,
                match.start(),
                match.end(),
                packet,
                context=context,
                explicit_ticker=explicit_ticker or None,
            )
            if is_ambiguous:
                ambiguous.append("current price")
                continue
            if not target_ticker or target_ticker not in anchor_prices:
                continue
            try:
                claimed_price = Decimal(str(match.group("px")).replace(",", ""))
            except (InvalidOperation, ValueError):
                continue
            anchor_price, currency = anchor_prices[target_ticker]
            relative_difference = abs(claimed_price - anchor_price) / anchor_price
            if relative_difference > Decimal("0.20"):
                mismatches.append(
                    f"${target_ticker} (claim {claimed_price} vs anchor {anchor_price} {currency})"
                )
        return list(dict.fromkeys(mismatches)), list(dict.fromkeys(ambiguous))

    def _find_echoed_speaker(
        self,
        turn: RoundtableTurnPayload,
        prior_turns: Sequence[RoundtableTurnPayload],
        packet: FrozenContextPacket,
    ) -> Optional[str]:
        if turn.speaker_id in (SpeakerId.HERMES_MODERATOR.value, SpeakerId.USER.value):
            return None
        cur_spoken = re.sub(r"\s+", " ", (turn.spoken_dialogue or "").strip().lower())
        cur_body = re.sub(r"^(?:คุณ\s*)?@?[\w.\-]+\s*(?:ครับ|ค่ะ)?[,\s]*", "", cur_spoken).strip()
        cur_tokens = _extract_semantic_tokens(cur_body)
        if len(cur_body) < 55 and len(cur_tokens) < 12:
            return None

        candidates = [
            t
            for t in (prior_turns or ())
            if t.speaker_id not in (SpeakerId.USER.value, SpeakerId.HERMES_MODERATOR.value, turn.speaker_id)
        ]
        if turn.round_stage == "opening":
            candidates = [t for t in candidates if t.round_stage == "opening"]
        else:
            candidates = candidates[-4:]

        persona_map = {p.speaker_id.value: p for p in resolve_personas_from_packet(packet)}
        for prev in candidates:
            prev_spoken = re.sub(r"\s+", " ", (prev.spoken_dialogue or "").strip().lower())
            prev_body = re.sub(r"^(?:คุณ\s*)?@?[\w.\-]+\s*(?:ครับ|ค่ะ)?[,\s]*", "", prev_spoken).strip()
            if len(cur_body) >= 55 and len(prev_body) >= 55 and cur_body[:55] == prev_body[:55]:
                spec = persona_map.get(prev.speaker_id)
                return spec.short_name if spec else prev.speaker_id

            prev_tokens = _extract_semantic_tokens(prev_body)
            if len(cur_tokens) >= 12 and len(prev_tokens) >= 12:
                union = cur_tokens | prev_tokens
                if union and (len(cur_tokens & prev_tokens) / len(union)) >= 0.72:
                    spec = persona_map.get(prev.speaker_id)
                    return spec.short_name if spec else prev.speaker_id
        return None

    @staticmethod
    def _has_explicit_conditional_context(text: str, start: int, end: int) -> bool:
        sentence_start = 0
        sentence_end = len(text)
        for boundary in _SENTENCE_BOUNDARY_RE.finditer(text):
            if boundary.end() <= start:
                sentence_start = boundary.end()
            elif boundary.start() >= end:
                sentence_end = boundary.start()
                break
        prefix = text[sentence_start:start]
        prefix = re.sub(
            r"^\s*(?:\[(?:FACT|MATH|INFERRED|DOCUMENTED|UNVERIFIED)[^\]]*\]\s*)+",
            "",
            prefix,
            flags=re.IGNORECASE,
        )
        suffix = text[end:sentence_end]
        return bool(_CONDITIONAL_PREFIX_RE.search(prefix) or _CONDITIONAL_SUFFIX_RE.search(suffix))

    def _verify_factual_text(
        self, text: str, packet: FrozenContextPacket, context: Optional[_ClaimContext] = None
    ) -> List[str]:
        issues: List[str] = []
        clean_text = (text or "").strip()
        if not clean_text:
            return issues

        violated_dcf_tickers, ambiguous_dcf = self._find_violated_dcf_tickers(
            clean_text, packet, context=context
        )
        if violated_dcf_tickers:
            lbl = ", ".join(t for t in violated_dcf_tickers if t != "__PRIMARY__") or "primary ticker"
            issues.append(f"Data Quality Flags: DCF claims are blocked for {lbl}.")
        if ambiguous_dcf:
            issues.append("Ticker Scope: DCF claim is ambiguous across packet instruments.")

        has_unqualified_holding_claim = any(
            not self._has_explicit_conditional_context(clean_text, match.start(), match.end())
            for match in _PORTFOLIO_HOLDING_CLAIM_RE.finditer(clean_text)
        )
        state = packet.portfolio_guard.attachment_state
        if has_unqualified_holding_claim and state == ContextAttachmentState.NOT_ATTACHED_DO_NOT_ASSUME:
            issues.append("Portfolio Guard (NOT_ATTACHED_DO_NOT_ASSUME): unsupported holdings or cost-basis claim.")
        elif has_unqualified_holding_claim and state in (
            ContextAttachmentState.ATTACHED_FLAT_BOOK,
            ContextAttachmentState.ATTACHED_EMPTY,
        ):
            issues.append(f"Portfolio Guard ({state.value}): claim says the user holds a position that is absent.")

        if packet.action_stance_hint in ("WAIT_DO_NOT_ENTER", "TRIM_OVERWEIGHT", "REDUCE") and re.search(
            r"(?i)(?:all-in\s*\u0e44\u0e14\u0e49\u0e40\u0e25\u0e22|\u0e41\u0e19\u0e30\u0e19\u0e33\u0e0b\u0e37\u0e49\u0e2d\u0e40\u0e15\u0e47\u0e21\u0e1e\u0e2d\u0e23\u0e4c\u0e15|\u0e40\u0e02\u0e49\u0e32\u0e0b\u0e37\u0e49\u0e2d\u0e2d\u0e22\u0e48\u0e32\u0e07\u0e22\u0e34\u0e48\u0e07\u0e17\u0e31\u0e19\u0e17\u0e35)", clean_text
        ):
            issues.append(f"Action Stance ({packet.action_stance_hint}): unsupported all-in recommendation.")

        price_mismatches, ambiguous_prices = self._find_price_contradictions(
            clean_text, packet, context=context
        )
        if price_mismatches:
            issues.append("Instrument Identity Anchor: current-price claim conflicts with packet (" + ", ".join(price_mismatches) + ").")
        if ambiguous_prices:
            issues.append("Ticker Scope: current-price claim is ambiguous across packet instruments.")
        return issues

    def verify_turn(
        self,
        turn: RoundtableTurnPayload,
        packet: FrozenContextPacket,
        unspoken_speakers: Sequence[str] = (),
        prior_turns: Sequence[RoundtableTurnPayload] = (),
    ) -> RoundtableVerificationResult:
        issues: List[str] = []
        spoken = (turn.spoken_dialogue or "").strip()

        # 1. Minimum spoken length + clean TTS formatting
        if len(spoken) < 12:
            issues.append("spoken_dialogue สั้นเกินไป ต้องมีบทพูดอย่างน้อย 1-2 ประโยคธรรมชาติ")
        if re.search(r"(\*\*|```|^\s*#)", spoken, flags=re.MULTILINE):
            issues.append("spoken_dialogue ห้ามมีสัญลักษณ์ Markdown (**, ```, #) เพื่อให้ระบบอ่านเสียง TTS ได้ลื่นไหล")

        # 2. 5-tier stance validation
        if turn.stance_tag not in _CANONICAL_STANCES:
            issues.append(f"stance_tag '{turn.stance_tag}' ไม่อยู่ในมาตรฐาน BUY/OVERWEIGHT/HOLD/UNDERWEIGHT/SELL/REVIEW")

        combined = spoken + "\n" + "\n".join(turn.evidence_bullets)
        # Shared factual guards also validate each user-visible playbook field.
        issues.extend(self._verify_factual_text(combined, packet, context=_ClaimContext(field_path="roundtable_turn")))

        # 6. Anti-Strawman: do not attribute speech to disabled or strictly unspoken panel gurus
        bad_speakers = self._find_invalid_speaker_attributions(
            turn, packet, unspoken_speakers, prior_turns
        )
        if bad_speakers:
            issues.append(
                "Anti-Strawman Guard: ห้ามสมมติหรืออ้างอิงคำพูดแทนกูรูที่ไม่ได้เปิดใช้งานหรือยังไม่ได้พูด ("
                + ", ".join(bad_speakers)
                + ")"
            )

        # 8. Crossfire / Heated Rebuttal direct target check
        prior_panel_speakers = {
            t.speaker_id
            for t in (prior_turns or ())
            if t.speaker_id not in (SpeakerId.USER.value, SpeakerId.HERMES_MODERATOR.value, turn.speaker_id)
        }
        debate_intensity = str(getattr(packet, "debate_intensity", "balanced") or "balanced").lower()
        needs_direct_target = (
            turn.speaker_id not in (SpeakerId.HERMES_MODERATOR.value, SpeakerId.USER.value)
            and bool(prior_panel_speakers)
            and (
                turn.round_stage == "crossfire"
                or (turn.round_stage == "rebuttal" and debate_intensity in ("heated", "crossfire"))
            )
        )
        reply_target = (turn.replying_to_speaker_id or "").strip()
        if needs_direct_target and (
            not reply_target
            or reply_target == turn.speaker_id
            or reply_target in (SpeakerId.USER.value, SpeakerId.HERMES_MODERATOR.value)
        ):
            issues.append(
                f"โหมด {debate_intensity} ({turn.round_stage}) ต้องระบุ `replying_to_speaker_id` เป็นกูรูคู่กรณีในวงโดยตรง (ห้ามว่างหรืออ้างตัวเอง/พิธีกร)"
            )

        # 9. Anti-Echo Guard: reject repetitive turns copying earlier guru points/wording
        echoed_from = self._find_echoed_speaker(turn, prior_turns, packet)
        if echoed_from:
            issues.append(
                f"Anti-Echo Guard: เนื้อหาซ้ำประเด็น/ถ้อยคำกับกูรูที่พูดไปแล้ว ({echoed_from}) — กรุณาข้ามไปเปิดมุมมองและตัวเลขใหม่ในเสาหลักของสายคุณทันที"
            )

        # 10. No-Soft-Agreement Guard in Heated / Crossfire Rebuttal when opposing camp exists
        if (
            turn.round_stage == "rebuttal"
            and debate_intensity in ("heated", "crossfire")
            and turn.speaker_id not in (SpeakerId.HERMES_MODERATOR.value, SpeakerId.USER.value)
            and _SOFT_AGREEMENT_OPEN_RE.search(spoken)
        ):
            latest_by_spk = {
                t.speaker_id: t.stance_tag
                for t in (prior_turns or ())
                if t.speaker_id not in (SpeakerId.USER.value, SpeakerId.HERMES_MODERATOR.value)
            }
            target_stance = latest_by_spk.get(reply_target)
            cur_pol = _polarity_sign(turn.stance_tag)
            tgt_pol = _polarity_sign(target_stance) if target_stance else 0
            if cur_pol != 0 and tgt_pol != 0 and cur_pol != tgt_pol:
                issues.append(
                    f"No-Soft-Agreement Guard: โหมด {debate_intensity} ห้ามเกริ่นเอาใจว่าเห็นด้วยกับคู่กรณีที่จุดยืนต่างกัน กรุณาชี้ข้อโต้แย้งหลักทันที"
                )

        # 11. Skeptic Downside / Kill-Switch Guard on full-length analysis turns
        if (
            turn.speaker_id == SpeakerId.SKEPTIC_GURU.value
            and turn.round_stage in ("opening", "rebuttal", "crossfire")
            and len(combined) >= 120
            and not _SKEPTIC_RISK_SIGNAL_RE.search(combined)
        ):
            issues.append(
                "Skeptic Kill-Switch Guard: เทิร์นของสายซักค้านต้องระบุความเสี่ยงด้านลบ, Data Flag หรือ Kill-Switch ที่วัดผลได้ชัดเจน"
            )

        retry_feedback = ""
        if issues:
            retry_feedback = (
                "[ROUNDTABLE CONSISTENCY VERIFIER]\nกรุณาแก้ไขจุดต่อไปนี้ในเทิร์นของคุณ:\n"
                + "\n".join(f"- {iss}" for iss in issues)
            )
        return RoundtableVerificationResult(
            is_valid=len(issues) == 0,
            issues=tuple(issues),
            retry_feedback=retry_feedback,
        )

    def verify_playbook(
        self, playbook: DistilledDecisionPlaybook, packet: FrozenContextPacket
    ) -> RoundtableVerificationResult:
        issues: List[str] = []

        if playbook.final_rating not in _CANONICAL_STANCES:
            issues.append(f"final_rating '{playbook.final_rating}' is not a canonical stance")
        if not (playbook.executive_verdict or "").strip():
            issues.append("executive_verdict must not be empty")
        if not playbook.action_playbook.primary_action.action_steps:
            issues.append("action_playbook.primary_action.action_steps needs at least one step")
        if packet.action_stance_hint in ("WAIT_DO_NOT_ENTER", "TRIM_OVERWEIGHT", "REDUCE") and playbook.final_rating == "BUY":
            issues.append(f"Action stance '{packet.action_stance_hint}' conflicts with final_rating BUY")

        visible_texts: List[Tuple[str, str, str]] = []

        def add_text(label: str, value: Optional[str], parent_text: str = "") -> None:
            if value and value.strip():
                visible_texts.append((label, value, parent_text))

        add_text("executive_verdict", playbook.executive_verdict)
        add_text("action_stance", playbook.action_stance)
        add_text("portfolio_context_summary", playbook.portfolio_context_summary)
        for index, point in enumerate(playbook.consensus_points):
            add_text(f"consensus_points[{index}]", point)
        for index, axis in enumerate(playbook.tension_axes):
            add_text(f"tension_axes[{index}].topic", axis.topic)
            add_text(f"tension_axes[{index}].bull_argument", axis.bull_argument, axis.topic)
            add_text(f"tension_axes[{index}].bear_argument", axis.bear_argument, axis.topic)
        for index, item in enumerate(playbook.personal_axis_alignment):
            add_text(f"personal_axis_alignment[{index}].dimension", item.dimension)
            add_text(f"personal_axis_alignment[{index}].detail", item.detail, item.dimension)

        options = [("action_playbook.primary_action", playbook.action_playbook.primary_action)]
        options.extend(
            (f"action_playbook.alternatives[{index}]", option)
            for index, option in enumerate(playbook.action_playbook.alternatives)
        )
        for prefix, option in options:
            add_text(f"{prefix}.label", option.label)
            for index, step in enumerate(option.action_steps):
                add_text(f"{prefix}.action_steps[{index}]", step, option.label)
            add_text(f"{prefix}.entry_zone", option.entry_zone, option.label)
            add_text(f"{prefix}.sizing_note", option.sizing_note, option.label)

        for index, item in enumerate(playbook.kill_switches):
            add_text(f"kill_switches[{index}]", item)
        add_text("reflection_summary", playbook.reflection_summary)

        for label, text, parent_text in visible_texts:
            context = _ClaimContext(field_path=label, parent_text=parent_text)
            issues.extend(
                f"{label}: {issue}"
                for issue in self._verify_factual_text(text, packet, context=context)
            )

        retry_feedback = ""
        if issues:
            retry_feedback = (
                "[HERMES PLAYBOOK VERIFIER]\nPlease correct the playbook:\n"
                + "\n".join(f"- {issue}" for issue in issues)
            )
        return RoundtableVerificationResult(
            is_valid=len(issues) == 0,
            issues=tuple(issues),
            retry_feedback=retry_feedback,
        )

