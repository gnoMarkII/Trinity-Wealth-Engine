"""Debate Stream Engine for Trinity Guru Roundtable & Hermes Distillation.

Implements TradingAgents-inspired safeguards adapted cleanly to Invest-Agents Hexagonal architecture:
- PromptHarness skill loading from ``prompts/skills/guru_roundtable/``
- Deterministic Instrument Identity Anchor at the top of every system prompt
- Anti-Strawman Turn Guard (``opponent_argument_or_opening``)
- Two-stage debate framing: Round 1 (Thesis) -> Round 2 (Cross-Exam & Risk/Sizing Posture)
- ``NO_EXTERNAL_TOOLS`` directive + ``invoke_structured_or_freetext`` resilient parsing
- Anti-Fence-Sitting & Speaking-Order Neutrality for Hermes Distillation
- Consistency Verifier probe + single retry loop
"""
from __future__ import annotations

import asyncio
from decimal import Decimal, InvalidOperation
import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from application.roundtable.ports import RoundtableDebateEnginePort, RoundtableVerifierPort
from core.llm_factory import detect_provider, get_llm
from core.model_registry import get_model_name
from core.prompt_harness import get_harness
from core.roundtable.models import (
    GURU_PERSONA_CATALOG,
    PERSONA_BY_ID,
    FrozenContextPacket,
    GuruPersonaSpec,
    SpeakerId,
    analyze_debate_pillar_coverage,
    get_preset_analytical_rubric,
    normalize_debate_depth,
    normalize_debate_intensity,
    resolve_personas_from_packet,
)
from schemas.roundtable_schemas import (
    ActionPlaybookOption,
    ActionPlaybookSpec,
    ConsensusTensionAxis,
    DistilledDecisionPlaybook,
    PersonalAxisCheckItem,
    RoundtableTurnPayload,
)
from validators.roundtable_verifier import (
    normalize_5tier_stance,
    sanitize_spoken_dialogue_for_tts,
)

log = logging.getLogger(__name__)

_NO_EXTERNAL_TOOLS = (
    "[NO_EXTERNAL_TOOLS DIRECTIVE]\n"
    "ในโหมดห้องเสวนาและสรุปผลแบบมีโครงสร้างนี้ ห้ามเรียกใช้งาน External Tool ใดๆ ทั้งสิ้น "
    "ให้วิเคราะห์จากข้อมูลใน Frozen Context Packet และบทสนทนาของเพื่อนกูรูที่แนบมาเท่านั้น"
)

_STRICT_FACT_GUARDRAIL = (
    "[STRICT FACT-GROUNDING GUARDRAIL]\n"
    "คาแรกเตอร์ โทนการพูด กรอบวิเคราะห์ และคำสั่งเน้นย้ำของผู้ใช้กำหนดมุมมองและวิธีสื่อสารเท่านั้น "
    "ห้ามบิดเบือน แก้ไข หรือแต่งตัวเลขขัดแย้งกับ FrozenContextPacket, Portfolio Guard "
    "และ Instrument Identity Anchor เด็ดขาด"
)

TModel = TypeVar("TModel", bound=BaseModel)


def _verification_failure_message(artifact: str, result: Any) -> str:
    issues = [str(issue) for issue in (getattr(result, "issues", ()) or ())]
    details = "; ".join(issues) or "verification failed without details"
    return f"Roundtable verifier rejected {artifact}: {details}"


def _persona_lookup(packet: Optional[FrozenContextPacket] = None) -> Dict[str, GuruPersonaSpec]:
    return {spec.speaker_id.value: spec for spec in resolve_personas_from_packet(packet)}


def opponent_argument_or_opening(
    speaker_id: str,
    round_stage: str,
    prior_turns: Sequence[RoundtableTurnPayload],
    unspoken_speakers: Sequence[str],
    packet: Optional[FrozenContextPacket] = None,
    *,
    crossfire_target: Optional[str] = None,
) -> str:
    """Anti-Strawman Turn Guard + Stance Clash & 5-Pillar Coverage Ledger."""
    persona_map = _persona_lookup(packet)
    intensity = normalize_debate_intensity(
        getattr(packet, "debate_intensity", "balanced") if packet is not None else "balanced",
        default="balanced",
    )

    def _label_for(sid: str) -> str:
        spec = persona_map.get(sid)
        return f"{spec.short_name} ({sid})" if spec else sid

    unspoken_labels = ", ".join(_label_for(s) for s in unspoken_speakers)
    if round_stage == "stage_intro":
        panel_labels = ", ".join(_label_for(s) for s in unspoken_speakers)
        return (
            "[สถานะการพูดในวง: Stage 0 — เปิดรายการตั้งโจทย์ก่อนกูรูทั้งโต๊ะพูด]\n"
            f"- กูรูที่จะร่วมถกเถียงในวงนี้: {panel_labels or 'ครบทุกสาย'}\n"
            "- ห้ามสมมติหรืออ้างคำพูดแทนกูรูคนอื่นที่ยังไม่ได้พูด ให้เกริ่นปูโจทย์และแกนความขัดแย้งจากข้อมูล Fact Packet เท่านั้น"
        )

    if not prior_turns:
        return (
            "[สถานะการพูดในวง: คุณเป็นผู้เปิดประเด็นคนแรก]\n"
            f"- กูรูที่ยังไม่ได้พูดในรอบนี้: {unspoken_labels or 'ไม่มี'}\n"
            "- ห้ามสมมติหรืออ้างคำพูดแทนกูรูคนอื่นที่ยังไม่ได้พูด"
        )

    # Synthesize deterministic Clash Summary & 5-Pillar Coverage for sharper rebuttals
    latest_stance_by_guru: Dict[str, str] = {}
    challengers_to_me: List[str] = []
    for t in prior_turns:
        if t.speaker_id in ("hermes_moderator", "user"):
            continue
        latest_stance_by_guru[t.speaker_id] = normalize_5tier_stance(t.stance_tag)
        if getattr(t, "replying_to_speaker_id", None) == speaker_id and t.speaker_id != speaker_id:
            if t.speaker_id not in challengers_to_me:
                challengers_to_me.append(t.speaker_id)

    bulls = [s for s, st in latest_stance_by_guru.items() if st in ("BUY", "OVERWEIGHT") and s != speaker_id]
    cautious = [s for s, st in latest_stance_by_guru.items() if st in ("HOLD", "UNDERWEIGHT", "SELL") and s != speaker_id]
    pillar_stats = analyze_debate_pillar_coverage(prior_turns)
    unchallenged = pillar_stats.get("unchallenged_pillars") or []

    lines: List[str] = []
    if round_stage == "opening" and prior_turns:
        raised_pillars = [
            pillar for pillar, speakers in (pillar_stats.get("raised_by") or {}).items() if speakers
        ]
        covered_str = ", ".join(raised_pillars) if raised_pillars else "ประเด็นเบื้องต้น"
        lines.append("[กฎกันพูดซ้ำรอบเปิด (Anti-Echo Rule)]:")
        lines.append(
            f"- เสาหลักที่กูรูคนก่อนหน้าเปิดประเด็นไปแล้ว: {covered_str}\n"
            "- ห้ามทวนข้อเท็จจริงพื้นฐานหรือพูดซ้ำประเด็นเดิมซ้ำซาก! จงข้ามไปเปิดมุมมองและตัวเลขหลักเฉพาะสายของคุณทันที "
            "(หากเห็นต่างกับคนก่อนหน้า ให้ชี้จุดต่างอย่างสุภาพและกระชับ)"
        )
    if latest_stance_by_guru and round_stage in ("rebuttal", "crossfire"):
        lines.append("[สรุปขั้วปะทะและจุดที่ยังไม่ได้ถูกโต้ (Active Clash & Unchallenged Pillars)]:")
        lines.append(
            f"- ขั้วเชิงรุก (BUY/OVERWEIGHT): {', '.join(_label_for(s) for s in bulls) or 'ยังไม่มี'} | "
            f"ขั้วระมัดระวัง/ลบ (HOLD/UNDERWEIGHT/SELL): {', '.join(_label_for(s) for s in cautious) or 'ยังไม่มี'}"
        )
        if challengers_to_me:
            lines.append(
                f"- ผู้ที่พาดพิง/ท้าทายคุณโดยตรงล่าสุด: {', '.join(_label_for(s) for s in challengers_to_me)} (คุณควรสวนกลับหรือชี้แจงกลับโดยตรง!)"
            )
        if unchallenged:
            lines.append(
                f"- เสาหลักที่ยังไม่มีใครซักค้านอย่างจริงจัง: {', '.join(unchallenged)} (หยิบมาถกให้ครบทุกมิติ อย่าวนเรื่องเดิมซ้ำ)"
            )
        # Intra-camp friction check when most speakers share similar polarity
        if (len(bulls) >= 2 and len(cautious) == 0) or (len(cautious) >= 2 and len(bulls) == 0):
            lines.append(
                "- [กฎกันห้องสอพลอ / Intra-Camp Friction]: แม้เสียงส่วนใหญ่ไปทางเดียวกัน แต่ห้ามอวยซ้ำ! จงเปิดศึกถกมิติรองที่ขัดแย้งกัน (เช่น จังหวะราคา MOS vs ตกรถ, CAPEX กินเงินสด, หรือ Macro Yield Sensitivity)"
            )

    lines.append("[ประวัติคำพูดจริงของกูรูที่พูดไปแล้ว (อ้างอิง/โต้แย้งเฉพาะจากรายการนี้เท่านั้น)]:")
    recent = list(prior_turns[-18:])
    ledger_cutoff = max(0, len(recent) - 10)
    for idx, t in enumerate(recent, start=1):
        spec = persona_map.get(t.speaker_id)
        name = spec.short_name if spec else t.speaker_id
        sizing_part = f" [Sizing: {t.sizing_posture}]" if getattr(t, "sizing_posture", None) else ""
        reply_part = f" [↩ @{t.replying_to_speaker_id}]" if getattr(t, "replying_to_speaker_id", None) else ""
        if idx <= ledger_cutoff:
            # Compact ledger for older turns beyond the latest 10 (keeps full chronological trace)
            dlg = re.sub(r"\s+", " ", str(t.spoken_dialogue or "")).strip()
            short_dlg = dlg if len(dlg) <= 140 else dlg[:139].rstrip() + "…"
            bullets = "; ".join(t.evidence_bullets[:2])
            lines.append(
                f"{idx}. [{t.round_stage}] {name} ({t.speaker_id}) [Stance: {t.stance_tag}]{reply_part}{sizing_part}: "
                f"{short_dlg}" + (f" | {bullets}" if bullets else "")
            )
        else:
            bullets = "; ".join(t.evidence_bullets[:3])
            lines.append(
                f"{idx}. [{t.round_stage}] {name} ({t.speaker_id}) [Stance: {t.stance_tag}]{reply_part}{sizing_part}: "
                f"{t.spoken_dialogue} | หลักฐาน: {bullets}"
            )
    if unspoken_speakers:
        lines.append(
            f"[กูรูที่ยังไม่ได้พูดในรอบนี้ ({unspoken_labels})]: ห้ามมโนแต่งคำพูดของบุคคลเหล่านี้ขึ้นมาโต้แย้ง"
        )
    if round_stage == "rebuttal":
        if intensity in ("heated", "crossfire"):
            lines.append(
                "[คำสั่งรอบ Rebuttal เชิงรุก (No-Soft-Agreement)]: ห้ามเกริ่นเอาใจว่าเห็นด้วยหากจุดยืนต่างกัน! "
                "ให้เลือกกูรูที่ความเห็นขัดแย้งหรือคนที่ท้าทายคุณด้านบน 1-2 คน ระบุ `replying_to_speaker_id` "
                "แล้วหักล้างสมมติฐานที่อ่อนที่สุดของอีกฝ่ายด้วยตัวเลขจริง พร้อมระบุท่าทีวางขนาดไม้ (`sizing_posture`)"
            )
        else:
            lines.append(
                "[คำสั่งรอบ Rebuttal]: กรุณาโต้แย้งหรือต่อยอดกูรูอย่างน้อย 1 คนที่พูดไปแล้วด้านบน "
                "พร้อมระบุ `replying_to_speaker_id` และสรุปท่าทีการวางขนาดไม้ (`sizing_posture`)"
            )
    elif round_stage == "crossfire":
        target_label = _label_for(crossfire_target) if crossfire_target else "คู่กรณีที่มีจุดยืนขัดแย้งมากที่สุด"
        target_hint = crossfire_target or "<speaker_id คู่กรณี>"
        lines.append(
            f"[คำสั่งรอบ Crossfire ดวลตัวต่อตัว]: คู่ปะทะของคุณคือ {target_label}! "
            f"กรุณาตั้งค่า `replying_to_speaker_id` เป็น `{target_hint}` แล้วซัดกลับข้อโต้แย้งล่าสุดของอีกฝ่ายแบบตรงไปตรงมาด้วยตัวเลขจริง "
            "พร้อมยืนยันหรือปรับเปลี่ยน `stance_tag` และ `sizing_posture` ให้ชัดเจนหลังถูกซักค้าน"
        )
    return "\n".join(lines)


_EQUITY_BLOCK_RE = re.compile(r"(?=^=== หุ้น \$)", flags=re.MULTILINE)
_SEC_TAG_RE = re.compile(r"<!--\s*SEC:([A-Z0-9_]+)\s*-->")

_SECTION_TITLES: Dict[str, str] = {
    "HEADER": "ภาพรวม & คะแนนระบบ",
    "VALUATION": "ประเมินมูลค่า (DCF / Reverse DCF / Consensus / Relative)",
    "QUALITY": "คุณภาพกำไร, ROIC, Piotroski, งบดุล & Moat",
    "GROWTH_GUIDANCE": "การเติบโต, ประมาณการ & Guidance ผู้บริหาร",
    "MACRO_TACTICAL": "มหภาค, เทคนิค, Smart Money & ความอ่อนไหว",
    "RISKS_THESIS": "ความเสี่ยง, จุดหักล้าง, Insider & Position Review",
    "NEWS_NOTES": "ข่าวสารล่าสุด, YouTube & บันทึก Vault",
}

_ROLE_SECTION_PLAN: Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...]]] = {
    "value_guru": (
        ("HEADER", "VALUATION", "RISKS_THESIS"),
        ("QUALITY", "GROWTH_GUIDANCE", "MACRO_TACTICAL", "NEWS_NOTES"),
    ),
    "quality_guru": (
        ("HEADER", "QUALITY", "GROWTH_GUIDANCE"),
        ("VALUATION", "RISKS_THESIS", "MACRO_TACTICAL", "NEWS_NOTES"),
    ),
    "growth_guru": (
        ("HEADER", "GROWTH_GUIDANCE", "NEWS_NOTES", "VALUATION"),
        ("QUALITY", "RISKS_THESIS", "MACRO_TACTICAL"),
    ),
    "macro_tactical_guru": (
        ("HEADER", "MACRO_TACTICAL", "VALUATION", "NEWS_NOTES"),
        ("GROWTH_GUIDANCE", "QUALITY", "RISKS_THESIS"),
    ),
    "skeptic_guru": (
        ("HEADER", "RISKS_THESIS", "QUALITY", "VALUATION", "NEWS_NOTES"),
        ("GROWTH_GUIDANCE", "MACRO_TACTICAL"),
    ),
}


def _clip_preserving_marker_status(text: str, max_chars: int) -> Tuple[str, bool]:
    clean = (text or "").strip()
    if not clean:
        return "", False
    if clean.startswith(("[ABSENT:", "[UNAVAILABLE:")):
        return clean, False
    if len(clean) <= max_chars:
        return clean, False
    return clean[: max(0, max_chars - 1)].rstrip() + "…", True


def _clip_preserving_marker(text: str, max_chars: int) -> str:
    clipped_text, _ = _clip_preserving_marker_status(text, max_chars)
    return clipped_text


def _parse_curated_equity_text(
    equity_text: str,
) -> Tuple[Optional[str], List[Tuple[str, Dict[str, str]]]]:
    """Parse comparative table + per-ticker curated `<!-- SEC:... -->` blocks.

    Returns ``(None, [])`` cleanly for legacy unsectioned packets or `[ABSENT]` guards.
    """
    raw = (equity_text or "").strip()
    if not raw or "<!-- SEC:" not in raw:
        return None, []

    chunks = [c.strip() for c in _EQUITY_BLOCK_RE.split(raw) if c.strip()]
    comp_table: Optional[str] = None
    ticker_blocks: List[Tuple[str, Dict[str, str]]] = []
    for chunk in chunks:
        if chunk.startswith("=== ตารางเปรียบเทียบ"):
            comp_table = chunk
            continue
        if not chunk.startswith("=== หุ้น $"):
            if comp_table is None:
                comp_table = chunk
            continue
        lines = chunk.splitlines()
        header_line = lines[0].strip()
        body = "\n".join(lines[1:]).strip()
        matches = list(_SEC_TAG_RE.finditer(body))
        if not matches:
            continue
        sec_map: Dict[str, str] = {}
        for idx, m in enumerate(matches):
            tag = m.group(1).strip().upper()
            start = m.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(body)
            sec_content = body[start:end].strip()
            if sec_content:
                sec_map[tag] = sec_content
        ticker_blocks.append((header_line, sec_map))

    if not ticker_blocks:
        return None, []
    return comp_table, ticker_blocks


def _render_ticker_sections_with_status(
    ticker_blocks: Sequence[Tuple[str, Dict[str, str]]],
    section_keys: Sequence[str],
    *,
    char_limit: int,
) -> Tuple[str, bool]:
    rendered_stocks: List[str] = []
    for header_line, sec_map in ticker_blocks:
        sec_parts: List[str] = [header_line]
        for sk in section_keys:
            content = sec_map.get(sk)
            if content:
                title = _SECTION_TITLES.get(sk, sk)
                sec_parts.append(f"[{title}]\n{content}")
        if len(sec_parts) > 1:
            rendered_stocks.append("\n".join(sec_parts))
    if not rendered_stocks:
        return "", False
    if len(rendered_stocks) == 1:
        return _clip_preserving_marker_status(rendered_stocks[0], char_limit)

    # Fair multi-ticker water-filling quota so stock #2/#3 are never starved by stock #1
    sep_overhead = 2 * max(0, len(rendered_stocks) - 1)
    avail = max(300, char_limit - sep_overhead)
    total_len = sum(len(s) for s in rendered_stocks)
    if total_len <= avail:
        return "\n\n".join(rendered_stocks), False

    base_cap = max(250, avail // len(rendered_stocks))
    short_used = sum(len(s) for s in rendered_stocks if len(s) <= base_cap)
    long_indices = [i for i, s in enumerate(rendered_stocks) if len(s) > base_cap]
    long_cap = (
        max(base_cap, (avail - short_used) // len(long_indices))
        if long_indices
        else base_cap
    )
    allocated: List[str] = []
    any_clipped = False
    for i, s in enumerate(rendered_stocks):
        if i in long_indices:
            s_clipped, was_c = _clip_preserving_marker_status(s, long_cap)
            allocated.append(s_clipped)
            any_clipped = any_clipped or was_c
        else:
            allocated.append(s)
    joined_clipped, join_c = _clip_preserving_marker_status("\n\n".join(allocated), char_limit)
    return joined_clipped, bool(any_clipped or join_c)


def _render_ticker_sections(
    ticker_blocks: Sequence[Tuple[str, Dict[str, str]]],
    section_keys: Sequence[str],
    *,
    char_limit: int,
) -> str:
    rendered, _ = _render_ticker_sections_with_status(
        ticker_blocks, section_keys, char_limit=char_limit
    )
    return rendered


_NUM_VALUE_RE = re.compile(r"^-?\d+(?:,\d{3})*(?:\.\d+)?")
_KEY_STOPWORDS = {"pct", "ratio", "usd", "thb", "base", "val", "score", "num", "max", "min"}


def _is_fact_represented(fact: str, scope_lower: str) -> bool:
    """Return True iff the exact metric value from ``fact`` already appears inside the target scope."""
    if "=" not in fact:
        return fact.strip().lower() in scope_lower
    raw_k, raw_v = fact.split("=", 1)
    # Strip optional ticker prefix (e.g., ORCL:current_price -> current_price)
    key_clean = raw_k.split(":")[-1].strip().lower()
    val_clean = raw_v.strip().lower()
    if not val_clean:
        return True
    is_pct_metric = key_clean.endswith(("_pct", "_percent", "_cagr_3y")) or "%" in val_clean
    if is_pct_metric and val_clean in scope_lower:
        return True
    if not is_pct_metric and re.search(rf"{re.escape(val_clean)}(?!\s*%)", scope_lower):
        return True

    # Handle prefixed currency values like $1.73T or $114.77B
    val_stripped = re.sub(r"\s+(?:usd|thb|x)$", "", val_clean).rstrip("%x").strip()
    if len(val_stripped) >= 4:
        if is_pct_metric and val_stripped in scope_lower:
            return True
        if not is_pct_metric and re.search(rf"{re.escape(val_stripped)}(?![0-9%])", scope_lower):
            return True

    m = _NUM_VALUE_RE.match(val_clean.lstrip("$"))
    if not m:
        return val_clean in scope_lower

    num_str = m.group(0).replace(",", "")
    candidates = [num_str]
    try:
        dec = Decimal(num_str)
        q2 = f"{dec.quantize(Decimal('0.01'))}"
        q4 = f"{dec.quantize(Decimal('0.0001'))}".rstrip("0").rstrip(".")
        for c in (q2, q4):
            if c not in candidates:
                candidates.append(c)
    except (InvalidOperation, ValueError):
        pass

    tail_guard = r"(?![0-9])" if is_pct_metric else r"(?![0-9%])"
    matched = any(
        re.search(rf"(?<![0-9A-Za-z.]){re.escape(c)}{tail_guard}", scope_lower)
        for c in candidates
    )
    if not matched:
        return False

    # Small integers/short numbers need at least one domain token from the key to avoid accidental collision
    if len(num_str.replace("-", "").replace(".", "")) <= 2:
        key_tokens = [
            tok for tok in key_clean.split("_")
            if len(tok) >= 3 and tok not in _KEY_STOPWORDS
        ]
        if key_tokens and not any(tok in scope_lower for tok in key_tokens):
            return False
    return True


def _filter_unrepresented_numeric_facts(
    facts: Sequence[str],
    blackboard_body: str,
    *,
    has_clipped_sections: bool,
    equity_scope: Optional[str] = None,
    portfolio_scope: Optional[str] = None,
    macro_scope: Optional[str] = None,
    ticker_scopes: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Keep all facts if sections were clipped or legacy; otherwise emit only unrepresented facts."""
    if not facts:
        return []
    if has_clipped_sections:
        return list(facts)
    bb_lower = blackboard_body.lower()
    eq_lower = equity_scope.lower() if equity_scope is not None else bb_lower
    port_lower = portfolio_scope.lower() if portfolio_scope is not None else bb_lower
    mac_lower = macro_scope.lower() if macro_scope is not None else bb_lower
    tk_lowers = {k.upper(): v.lower() for k, v in (ticker_scopes or {}).items()}
    unrep: List[str] = []
    for fact in facts:
        key_part = fact.split("=", 1)[0].strip() if "=" in fact else ""
        if ":" in key_part:
            tk_pref, bare_k = key_part.split(":", 1)
            tk_key = tk_pref.strip().upper()
            target_scope = tk_lowers.get(tk_key, eq_lower)
            key_lower = bare_k.strip().lower()
        else:
            key_lower = key_part.lower()
            if key_lower.startswith("portfolio_"):
                target_scope = port_lower
            elif key_lower.startswith("macro_"):
                target_scope = mac_lower
            else:
                target_scope = eq_lower
        if not _is_fact_represented(fact, target_scope):
            unrep.append(fact)
    return unrep


def _format_shared_blackboard_context(packet: FrozenContextPacket) -> Tuple[str, bool]:
    """Build byte-identical Shared Blackboard Context across all 6 seats to maximize Gemini KV Prefix Caching."""
    comp_table, ticker_blocks = _parse_curated_equity_text(packet.equity_report_text)

    parts_head: List[str] = []
    anchor_blocks: List[str] = []
    if packet.instrument_anchor:
        anc_txt = packet.instrument_anchor.to_prompt_block()
        parts_head.append(anc_txt)
        anchor_blocks.append(anc_txt)
    if getattr(packet, "secondary_anchors", ()):
        sec_lines = ["[SECONDARY INSTRUMENT ANCHORS — VERIFIED]"]
        for sa in packet.secondary_anchors:
            px_s = f"{sa.current_price} {sa.currency}" if sa.current_price else "N/A"
            sec_lines.append(
                f"- ${sa.ticker} ({sa.company_name}) | Market: {sa.market} | Benchmark: {sa.benchmark_ticker} | "
                f"Sector/Ind: {sa.sector} / {sa.industry} | Currency: {sa.currency} | Price: {px_s}"
            )
        sec_txt = "\n".join(sec_lines)
        parts_head.append(sec_txt)
        anchor_blocks.append(sec_txt)
    parts_head.append(f"[AGENDA / คำถามจากผู้ใช้]\n{packet.agenda_prompt}")
    parts_head.append(f"[บริบทพอร์ตจริง (3-State Guard)]\n{packet.portfolio_guard.summary_text}")

    if packet.past_scorecard:
        sc = packet.past_scorecard
        parts_head.append(
            "[ย้อนรอยมติครั้งก่อน (Past Roundtable Scorecard)]\n"
            f"- ประชุมครั้งก่อนวันที่: {sc.previous_date} | มติเดิม: {sc.previous_final_rating}\n"
            f"- ราคาวันประชุมเดิม: {sc.entry_price or 'N/A'} -> ปัจจุบัน: {sc.settlement_price or 'N/A'}\n"
            f"- ผลตอบแทนหุ้น: {sc.raw_return_pct or 'N/A'}% | Benchmark ({sc.benchmark_ticker}): {sc.benchmark_return_pct or 'N/A'}% | Alpha: {sc.alpha_pct or 'N/A'}%\n"
            f"- ผลลัพธ์: {sc.outcome_label} | บทเรียนครั้งก่อน: {sc.reflection_lesson or 'ยังไม่มีบทเรียน'}"
        )
    if getattr(packet, "secondary_scorecards", ()):
        ssc_lines = ["[ย้อนรอยมติครั้งก่อนของหุ้นเปรียบเทียบ (Secondary Past Scorecards)]"]
        for ssc in packet.secondary_scorecards:
            ssc_lines.append(
                f"- ${ssc.primary_ticker} ({ssc.previous_date}, มติเดิม {ssc.previous_final_rating}): "
                f"Entry={ssc.entry_price or 'N/A'} -> Now={ssc.settlement_price or 'N/A'}, "
                f"Return={ssc.raw_return_pct or 'N/A'}%, Alpha={ssc.alpha_pct or 'N/A'}% ({ssc.outcome_label})"
            )
        parts_head.append("\n".join(ssc_lines))

    if packet.user_role_models:
        parts_head.append(
            f"[นักลงทุนต้นแบบของผู้ใช้ (Role Models)]: {', '.join(packet.user_role_models)} "
            f"(กูรูสายที่ตรงกัน: {', '.join(packet.prioritized_speakers) or 'ไม่มีเฉพาะเจาะจง'})"
        )

    if packet.data_quality_flags:
        parts_head.append(f"[DATA QUALITY FLAGS]: {', '.join(packet.data_quality_flags)}")
    ticker_dq = getattr(packet, "ticker_data_quality_flags", None) or {}
    if ticker_dq and (
        len(ticker_dq) > 1
        or any(k != (packet.primary_ticker or "").upper() for k in ticker_dq)
    ):
        per_tk_str = " | ".join(
            f"${sym}: {', '.join(flist)}"
            for sym, flist in ticker_dq.items()
            if flist
        )
        if per_tk_str:
            parts_head.append(f"[DATA QUALITY FLAGS PER TICKER]: {per_tk_str}")
    if packet.action_stance_hint:
        parts_head.append(f"[QUANT SCORECARD ACTION STANCE HINT]: {packet.action_stance_hint}")

    # Canonical body ordering shared verbatim by all 6 seats
    parts_body: List[str] = []
    essence_raw = (packet.essence_report_text or "").strip()
    essence_full, essence_clipped = _clip_preserving_marker_status(essence_raw, 10000)
    parts_body.append(f"[แก่นการลงทุนและกรอบความเสี่ยง 8 หมวด + Financial Context (@Essence)]\n{essence_full}")

    if comp_table:
        parts_body.append(comp_table)

    is_single = len(ticker_blocks) <= 1
    if ticker_blocks:
        all_secs = (
            "HEADER",
            "VALUATION",
            "QUALITY",
            "GROWTH_GUIDANCE",
            "MACRO_TACTICAL",
            "RISKS_THESIS",
            "NEWS_NOTES",
        )
        eq_limit = 42000 if is_single else 24000
        eq_full, eq_clipped = _render_ticker_sections_with_status(
            ticker_blocks, all_secs, char_limit=eq_limit
        )
    else:
        eq_limit = 24000 if is_single else 16000
        eq_raw = (packet.equity_report_text or "").strip()
        eq_full, raw_eq_clipped = _clip_preserving_marker_status(eq_raw, eq_limit)
        eq_clipped = (
            bool(eq_raw)
            and not eq_raw.startswith(("[ABSENT:", "[UNAVAILABLE:"))
            and (raw_eq_clipped or "<!-- SEC:" not in eq_raw)
        )
    parts_body.append(f"[ข้อมูลหุ้นรายตัว ($Ticker)]\n{eq_full}")

    macro_raw = (packet.macro_report_text or "").strip()
    macro_limit = 14000 if is_single else 8000
    macro_block, macro_clipped = _clip_preserving_marker_status(macro_raw, macro_limit)
    parts_body.append(f"[ข้อมูลมหภาค (@Macro)]\n{macro_block}")

    digest_raw = (packet.digest_report_text or "").strip()
    digest_limit = 8000 if is_single else 6000
    digest_block, digest_clipped = _clip_preserving_marker_status(digest_raw, digest_limit)
    parts_body.append(f"[สรุปข่าวรายวัน (@Digest)]\n{digest_block}")

    has_clipped_sections = bool(eq_clipped or macro_clipped or digest_clipped or essence_clipped)

    body_joined = "\n\n".join(parts_body)
    head_joined = "\n\n".join(parts_head)
    # Scope equity facts strictly to anchors + comp_table + eq_full (preventing cross-card false matches)
    anchor_scopes = "\n\n".join(anchor_blocks)
    equity_scope_txt = "\n\n".join(p for p in (anchor_scopes, comp_table or "", eq_full) if p)
    ticker_scopes: Dict[str, str] = {}
    if ticker_blocks:
        for blk in _EQUITY_BLOCK_RE.split(eq_full):
            m_sym = re.match(r"^=== หุ้น \$([A-Za-z0-9.\-]+)\b", blk.strip())
            if m_sym:
                sym_k = m_sym.group(1).upper()
                ticker_scopes[sym_k] = f"{anchor_scopes}\n\n{blk}"
    unrepresented_facts = _filter_unrepresented_numeric_facts(
        packet.verified_numeric_facts,
        f"{head_joined}\n\n{body_joined}",
        has_clipped_sections=has_clipped_sections,
        equity_scope=equity_scope_txt,
        portfolio_scope=head_joined,
        macro_scope=macro_block,
        ticker_scopes=ticker_scopes,
    )
    if unrepresented_facts:
        parts_head.append("[VERIFIED NUMERIC FACTS]: " + " | ".join(unrepresented_facts))

    return "\n\n".join(parts_head + parts_body), has_clipped_sections


def _format_role_focus_lens(speaker_id: str, packet: FrozenContextPacket) -> str:
    """Return compact seat-specific attention instructions appended AFTER the shared cached prefix."""
    role_plan = _ROLE_SECTION_PLAN.get(speaker_id)
    if not role_plan:
        return ""
    persona_map = _persona_lookup(packet)
    spec = persona_map.get(speaker_id)
    role_lbl = f"{spec.short_name} ({spec.archetype})" if spec else speaker_id
    primary_secs, secondary_secs = role_plan
    primary_titles = ", ".join(f"[{_SECTION_TITLES.get(sk, sk)}]" for sk in primary_secs)
    secondary_titles = ", ".join(f"[{_SECTION_TITLES.get(sk, sk)}]" for sk in secondary_secs)
    if speaker_id == "macro_tactical_guru":
        primary_str = f"@Macro, @Digest และหัวข้อหุ้น {primary_titles}"
        secondary_str = f"หัวข้อหุ้น {secondary_titles}"
    elif speaker_id == "growth_guru":
        primary_str = f"{primary_titles} และ @Digest"
        secondary_str = f"{secondary_titles} และ @Macro"
    else:
        primary_str = primary_titles
        secondary_str = f"{secondary_titles}, @Macro และ @Digest"
    return (
        f"[ROLE FOCUS LENS — กรอบเน้นสำหรับ {role_lbl}]\n"
        f"- จุดโฟกัสหลัก (Primary Evidence): {primary_str}\n"
        f"- จุดตรวจสอบข้ามสาย (Cross-Check): {secondary_str}"
    )


def _format_packet_prompt_context(
    packet: FrozenContextPacket,
    speaker_id: Optional[str] = None,
) -> str:
    """Format Shared Fixed Blackboard + optional Role Focus Lens suffix for a panel seat."""
    shared_ctx, _ = _format_shared_blackboard_context(packet)
    if speaker_id and speaker_id in _ROLE_SECTION_PLAN:
        lens = _format_role_focus_lens(speaker_id, packet)
        if lens:
            return f"{shared_ctx}\n\n{lens}"
    return shared_ctx


def _extract_json_object(raw_text: str) -> Optional[Dict[str, Any]]:
    text = (raw_text or "").strip()
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", text)
    candidate = fenced.group(1) if fenced else text
    if not fenced:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            candidate = text[start : end + 1]
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _coerce_content_text(resp: Any) -> str:
    content = getattr(resp, "content", resp)
    if isinstance(content, list):
        pieces: List[str] = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                pieces.append(str(item["text"]))
            else:
                pieces.append(str(item))
        return "".join(pieces)
    return str(content or "")


def _extract_usage_metrics(raw_msg: Any) -> Optional[Dict[str, int]]:
    """Extract normalized token & cached-read counts from LangChain AIMessage metadata."""
    if raw_msg is None:
        return None
    usage = getattr(raw_msg, "usage_metadata", None)
    resp_meta = getattr(raw_msg, "response_metadata", None)

    def _as_int(val: Any) -> int:
        try:
            return int(val or 0)
        except Exception:
            return 0

    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    cached_tokens = 0

    if isinstance(usage, dict):
        input_tokens = _as_int(usage.get("input_tokens") or usage.get("prompt_tokens"))
        output_tokens = _as_int(usage.get("output_tokens") or usage.get("completion_tokens"))
        total_tokens = _as_int(usage.get("total_tokens"))
        details = usage.get("input_token_details")
        if isinstance(details, dict):
            cached_tokens = _as_int(details.get("cache_read") or details.get("cached_tokens"))
        if not cached_tokens:
            cached_tokens = _as_int(
                usage.get("cache_read_input_tokens") or usage.get("cached_content_token_count")
            )
    elif usage is not None:
        input_tokens = _as_int(getattr(usage, "input_tokens", 0))
        output_tokens = _as_int(getattr(usage, "output_tokens", 0))
        total_tokens = _as_int(getattr(usage, "total_tokens", 0))
        cached_tokens = _as_int(getattr(usage, "cached_content_token_count", 0))

    if isinstance(resp_meta, dict):
        rm_usage = resp_meta.get("usage_metadata") or resp_meta.get("token_usage")
        if isinstance(rm_usage, dict):
            if not input_tokens:
                input_tokens = _as_int(rm_usage.get("prompt_token_count") or rm_usage.get("prompt_tokens"))
            if not output_tokens:
                output_tokens = _as_int(rm_usage.get("candidates_token_count") or rm_usage.get("completion_tokens"))
            if not total_tokens:
                total_tokens = _as_int(rm_usage.get("total_token_count") or rm_usage.get("total_tokens"))
            if not cached_tokens:
                cached_tokens = _as_int(rm_usage.get("cached_content_token_count"))
                if not cached_tokens and isinstance(rm_usage.get("prompt_tokens_details"), dict):
                    cached_tokens = _as_int(rm_usage["prompt_tokens_details"].get("cached_tokens"))

    if not total_tokens and (input_tokens or output_tokens):
        total_tokens = input_tokens + output_tokens
    if not any((input_tokens, output_tokens, total_tokens, cached_tokens)):
        return None
    return {
        "input_tokens": input_tokens,
        "cached_tokens": cached_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _log_llm_usage(schema_name: str, raw_msg: Any, *, model_name: str = "") -> Optional[Dict[str, int]]:
    metrics = _extract_usage_metrics(raw_msg)
    if metrics is None:
        return None
    inp = metrics["input_tokens"]
    cached = metrics["cached_tokens"]
    hit_pct = (cached / inp * 100.0) if inp > 0 else 0.0
    log.info(
        "[Roundtable LLM Telemetry] schema=%s model=%s input_tokens=%d cached_tokens=%d cache_hit=%.1f%% output_tokens=%d total_tokens=%d",
        schema_name,
        model_name or "-",
        inp,
        cached,
        hit_pct,
        metrics["output_tokens"],
        metrics["total_tokens"],
    )
    return metrics


class GeminiRoundtableEngine(RoundtableDebateEnginePort):
    """Orchestrates multi-persona LLM calls for the Guru Roundtable Studio."""

    def __init__(
        self,
        *,
        model_name: Optional[str] = None,
        temperature: float = 0.25,
        llm_factory: Optional[Callable[[str, float], Any]] = None,
    ) -> None:
        self._model_name = model_name or get_model_name("roundtable")
        self._temperature = temperature
        self._llm_factory = llm_factory
        self._harness = get_harness("guru_roundtable")

    def _effective_temperature(self, packet: Optional[FrozenContextPacket], *, round_stage: str) -> float:
        # Preserve explicit custom temperature override on constructor when non-default
        if abs(self._temperature - 0.25) > 1e-6:
            return self._temperature
        if round_stage in ("stage_intro", "synthesis"):
            return 0.25
        intensity = normalize_debate_intensity(
            getattr(packet, "debate_intensity", "balanced") if packet is not None else "balanced",
            default="balanced",
        )
        if intensity == "crossfire":
            return 0.65
        if intensity == "heated":
            return 0.55
        return 0.30

    def _build_llm(self, *, max_output_tokens: int = 2048, temperature_override: Optional[float] = None) -> Any:
        temp = self._temperature if temperature_override is None else temperature_override
        if self._llm_factory is not None:
            return self._llm_factory(self._model_name, temp)
        provider = detect_provider(self._model_name)
        return get_llm(
            provider=provider,
            model_name=self._model_name,
            temperature=temp,
            use_fallback=True,
            max_output_tokens=max_output_tokens,
        )

    def _load_shared_system_prompt(self) -> str:
        """Byte-identical SystemMessage across all 6 seats to maximize implicit KV prefix caching."""
        shared_skill = self._harness.get_skill_text("SKILL.md", strict=False)
        return "\n\n---\n\n".join(
            p for p in (shared_skill, _STRICT_FACT_GUARDRAIL, _NO_EXTERNAL_TOOLS) if p
        )

    def _load_persona_role_block(
        self,
        speaker_id: str,
        packet: Optional[FrozenContextPacket] = None,
        *,
        include_focus_lens: bool = True,
    ) -> str:
        """Seat-specific persona, custom override, rubric, and Role Focus Lens placed AFTER the shared prefix."""
        persona_map = _persona_lookup(packet)
        spec = persona_map.get(speaker_id)
        base_spec = PERSONA_BY_ID.get(SpeakerId(speaker_id)) if speaker_id in {s.value for s in SpeakerId} else None
        persona_file = (spec.prompt_file if spec else None) or (base_spec.prompt_file if base_spec else "HERMES_MODERATOR.md")
        persona_text = self._harness.get_skill_text(persona_file, required=False, strict=False)

        custom_block = ""
        preset_id = spec.preset_id if spec is not None else "default"
        rubric = get_preset_analytical_rubric(speaker_id, preset_id)
        rubric_section = f"\n\n[PRESET ANALYTICAL RUBRIC]\n{rubric}" if rubric else ""
        if spec is not None:
            is_customized = (
                base_spec is None
                or spec.preset_id != "default"
                or spec.display_name != base_spec.display_name
                or spec.short_name != base_spec.short_name
                or spec.archetype != base_spec.archetype
                or spec.speaking_style != base_spec.speaking_style
                or bool(spec.custom_directive)
            )
            if is_customized:
                custom_block = (
                    "[CUSTOM GURU PERSONA & SPEAKING STYLE OVERRIDE]\n"
                    f"- ชื่อในวงสนทนา: {spec.display_name} (เรียกสั้นๆ ว่า {spec.short_name})\n"
                    f"- คาแรกเตอร์ / ฉายา: {spec.archetype}\n"
                    f"- โทนการพูดที่ต้องยึด: {spec.speaking_style or '-'}\n"
                    f"- คำสั่งเน้นย้ำพิเศษจากผู้ใช้: {spec.custom_directive or 'ไม่มี'}"
                    f"{rubric_section}"
                )
            elif rubric:
                custom_block = f"[PRESET ANALYTICAL RUBRIC]\n{rubric}"
        elif rubric:
            custom_block = f"[PRESET ANALYTICAL RUBRIC]\n{rubric}"

        if packet is not None and speaker_id in ("quality_guru", "skeptic_guru"):
            q_spec = persona_map.get("quality_guru")
            s_spec = persona_map.get("skeptic_guru")
            if (
                q_spec is not None
                and s_spec is not None
                and q_spec.enabled
                and s_spec.enabled
                and q_spec.preset_id == "quality_forensic_auditor"
                and s_spec.preset_id == "skeptic_forensic_sleuth"
            ):
                pair_note = (
                    "[FORENSIC PAIR COORDINATION — ห้ามวิเคราะห์ทับซ้อน]\n"
                    "- สาย Quality Auditor เจาะจงเฉพาะคุณภาพกำไรเทียบเงินสด (OCF/Net Income, Sloan Accrual), ผลกระทบ SBC และ Piotroski F-Score\n"
                    "- สาย Skeptic Sleuth เจาะจงเฉพาะความเสี่ยงหนี้สิน (Credit Rating, ICR, Net Debt/EBITDA), Data Quality Flags และ Insider Open-Market Sell"
                )
                custom_block = "\n\n".join(p for p in (custom_block, pair_note) if p)

        lens_block = (
            _format_role_focus_lens(speaker_id, packet)
            if (include_focus_lens and packet is not None)
            else ""
        )
        return "\n\n---\n\n".join(p for p in (persona_text, custom_block, lens_block) if p)

    def _load_persona_system_prompt(
        self,
        speaker_id: str,
        packet: Optional[FrozenContextPacket] = None,
    ) -> str:
        """Combined legacy-compatible persona system prompt accessor."""
        shared_sys = self._load_shared_system_prompt()
        role_block = self._load_persona_role_block(speaker_id, packet=packet, include_focus_lens=False)
        return "\n\n---\n\n".join(p for p in (shared_sys, role_block) if p)

    async def _invoke_structured_or_freetext(
        self,
        *,
        schema: Type[TModel],
        messages: List[Any],
        fallback_builder: Callable[[str], TModel],
        max_output_tokens: int = 2048,
        temperature_override: Optional[float] = None,
    ) -> TModel:
        """Try native structured output first; cleanly fall back to JSON/text parsing on errors."""
        llm = self._build_llm(max_output_tokens=max_output_tokens, temperature_override=temperature_override)

        # 1. Try structured output (with include_raw=True telemetry when supported by provider)
        try:
            try:
                structured_llm = llm.with_structured_output(schema, include_raw=True)
            except TypeError:
                structured_llm = llm.with_structured_output(schema)
            result = await asyncio.to_thread(structured_llm.invoke, messages)
            if isinstance(result, dict) and ("raw" in result or "parsed" in result):
                _log_llm_usage(schema.__name__, result.get("raw"), model_name=self._model_name)
                parsed_obj = result.get("parsed")
                if isinstance(parsed_obj, schema):
                    return parsed_obj
                if isinstance(parsed_obj, dict):
                    return schema.model_validate(parsed_obj)
                raw_fallback_json = _extract_json_object(_coerce_content_text(result.get("raw")))
                if raw_fallback_json is not None:
                    return schema.model_validate(raw_fallback_json)
            if isinstance(result, schema):
                return result
            if isinstance(result, dict):
                return schema.model_validate(result)
        except Exception as exc:
            log.warning("Structured output failed for %s (%s); falling back to freetext JSON parse", schema.__name__, exc)

        # 2. Try free-text invocation + JSON extraction fallback
        try:
            raw_resp = await asyncio.to_thread(llm.invoke, messages)
            _log_llm_usage(f"{schema.__name__}[freetext]", raw_resp, model_name=self._model_name)
            raw_text = _coerce_content_text(raw_resp)
            parsed = _extract_json_object(raw_text)
            if parsed is not None:
                try:
                    return schema.model_validate(parsed)
                except Exception:
                    pass
            return fallback_builder(raw_text)
        except Exception as exc:
            log.error("LLM freetext invocation failed for %s: %s", schema.__name__, exc)
            return fallback_builder("")

    def _sanitize_turn(self, turn: RoundtableTurnPayload, speaker_id: str, round_stage: str) -> RoundtableTurnPayload:
        cleaned_dialogue = sanitize_spoken_dialogue_for_tts(turn.spoken_dialogue)
        if not cleaned_dialogue:
            cleaned_dialogue = "ขออนุญาตให้ข้อมูลเชิงประจักษ์เพิ่มเติมจากมุมมองหลักของผมในประเด็นนี้ครับ"
        bullets = [str(b).strip() for b in turn.evidence_bullets if str(b).strip()][:3]
        if not bullets:
            bullets = ["[INFERRED] วิเคราะห์ตามกรอบข้อมูลที่แนบในห้องประชุม"]
        return turn.model_copy(
            update={
                "speaker_id": speaker_id,
                "round_stage": round_stage,
                "spoken_dialogue": cleaned_dialogue,
                "evidence_bullets": bullets,
                "stance_tag": normalize_5tier_stance(turn.stance_tag),
            }
        )

    async def generate_hermes_stage_intro(
        self,
        *,
        packet: FrozenContextPacket,
        ordered_gurus: Sequence[str],
        verifier: Optional[RoundtableVerifierPort] = None,
    ) -> RoundtableTurnPayload:
        """Generate Stage 0 Hermes Opening Hook for deep/YouTube debates without spoiling unspoken speakers."""
        sys_prompt = self._load_shared_system_prompt()
        shared_blackboard, _ = _format_shared_blackboard_context(packet)
        role_block = self._load_persona_role_block(
            SpeakerId.HERMES_MODERATOR.value, packet=packet, include_focus_lens=False
        )
        guard_block = opponent_argument_or_opening(
            SpeakerId.HERMES_MODERATOR.value,
            "stage_intro",
            (),
            ordered_gurus,
            packet=packet,
        )
        intensity = normalize_debate_intensity(getattr(packet, "debate_intensity", "crossfire"), default="crossfire")
        depth = normalize_debate_depth(getattr(packet, "debate_depth", "deep_youtube"), default="deep_youtube")
        ticker_lbl = f"${packet.primary_ticker}" if packet.primary_ticker else "พอร์ตและมหภาค"
        ref_px = (
            f"{packet.instrument_anchor.current_price} {packet.instrument_anchor.currency}"
            if packet.instrument_anchor and packet.instrument_anchor.current_price
            else "N/A"
        )

        human_prompt = (
            f"{shared_blackboard}\n\n---\n\n{role_block}\n\n"
            f"{guard_block}\n"
            f"[DEBATE_INTENSITY: {intensity}] [DEBATE_DEPTH: {depth}]\n\n"
            f"กรุณาตอบกลับเป็น JSON ตามสคีมา `RoundtableTurnPayload` สำหรับบทบาท `Hermes Moderator` (`hermes_moderator`) "
            f"ในรอบ `stage_intro` (เปิดรายการวิเคราะห์ {ticker_lbl} ที่ราคาอ้างอิง {ref_px} ใน 3-4 ประโยคภาษาพูดธรรมชาติ ดึงดูดผู้ชมด้วยโจทย์ปะทะหลัก ไร้ Markdown)"
        )
        messages: List[Any] = [SystemMessage(content=sys_prompt), HumanMessage(content=human_prompt)]
        eff_temp = self._effective_temperature(packet, round_stage="stage_intro")

        def _fallback(raw_text: str) -> RoundtableTurnPayload:
            spoken = sanitize_spoken_dialogue_for_tts(raw_text)[:520] or (
                f"สวัสดีครับ ยินดีต้อนรับเข้าสู่ห้องประชุมสภากูรู วันนี้เรามาชำแหละ {ticker_lbl} "
                f"ที่ราคาอ้างอิง {ref_px} ว่าโอกาสการเติบโตคุ้มค่ากับความเสี่ยงและราคาในตลาดตอนนี้หรือไม่ เดี๋ยวไปฟังกูรูทั้งโต๊ะซัดกันเลยครับ"
            )
            return RoundtableTurnPayload(
                speaker_id="hermes_moderator",
                round_stage="stage_intro",
                replying_to_speaker_id=None,
                spoken_dialogue=spoken,
                evidence_bullets=[f"[FACT] เปิดเวทีเสวนา {ticker_lbl} (Ref Price: {ref_px})"],
                stance_tag="REVIEW",
            )

        turn = await self._invoke_structured_or_freetext(
            schema=RoundtableTurnPayload,
            messages=messages,
            fallback_builder=_fallback,
            temperature_override=eff_temp,
        )
        turn = self._sanitize_turn(turn, SpeakerId.HERMES_MODERATOR.value, "stage_intro").model_copy(
            update={"replying_to_speaker_id": None}
        )
        if verifier is not None:
            def _verify_intro(cand: RoundtableTurnPayload) -> Any:
                try:
                    return verifier.verify_turn(
                        cand,
                        packet,
                        unspoken_speakers=ordered_gurus,
                        prior_turns=(),
                    )
                except TypeError:
                    return verifier.verify_turn(cand, packet)

            v_res = _verify_intro(turn)
            if not v_res.is_valid:
                if v_res.retry_feedback:
                    retry_msgs = messages + [HumanMessage(content=v_res.retry_feedback)]
                    retried = await self._invoke_structured_or_freetext(
                        schema=RoundtableTurnPayload,
                        messages=retry_msgs,
                        fallback_builder=lambda _: turn,
                        temperature_override=eff_temp,
                    )
                    retried = self._sanitize_turn(retried, SpeakerId.HERMES_MODERATOR.value, "stage_intro").model_copy(
                        update={"replying_to_speaker_id": None}
                    )
                    retry_result = _verify_intro(retried)
                    if retry_result.is_valid:
                        turn = retried
                        v_res = retry_result

                if not v_res.is_valid:
                    safe_turn = self._sanitize_turn(_fallback(""), SpeakerId.HERMES_MODERATOR.value, "stage_intro").model_copy(
                        update={"replying_to_speaker_id": None}
                    )
                    fallback_result = _verify_intro(safe_turn)
                    if not fallback_result.is_valid:
                        raise ValueError(_verification_failure_message("stage intro fallback", fallback_result))
                    turn = safe_turn
        return turn

    async def generate_guru_turn(
        self,
        *,
        speaker_id: str,
        round_stage: str,
        packet: FrozenContextPacket,
        prior_turns: Sequence[RoundtableTurnPayload],
        unspoken_speakers: Sequence[str],
        verifier: Optional[RoundtableVerifierPort] = None,
        crossfire_target: Optional[str] = None,
    ) -> RoundtableTurnPayload:
        sys_prompt = self._load_shared_system_prompt()
        shared_blackboard, _ = _format_shared_blackboard_context(packet)
        role_block = self._load_persona_role_block(speaker_id, packet=packet, include_focus_lens=True)
        guard_block = opponent_argument_or_opening(
            speaker_id,
            round_stage,
            prior_turns,
            unspoken_speakers,
            packet=packet,
            crossfire_target=crossfire_target,
        )
        is_prioritized = speaker_id in packet.prioritized_speakers
        priority_note = (
            "\n[หมายเหตุพิเศษ]: แนวคิดของคุณตรงกับนักลงทุนต้นแบบที่ผู้ใช้ยึดถือ กรุณานำเสนอจุดยืนให้คมชัดเป็นพิเศษ"
            if is_prioritized
            else ""
        )
        is_multi_ticker = "COMPARATIVE SNAPSHOT TABLE" in (packet.equity_report_text or "") or (
            sum(1 for r in packet.context_refs if r.startswith("$")) > 1
        )
        multi_note = (
            "\n[โหมดเปรียบเทียบหลายหุ้น]: กรุณาวิเคราะห์และเปรียบเทียบตัวเลขของทุกหุ้นที่แนบมาในตารางเทียบเคียง ห้ามพูดถึงเฉพาะหุ้นตัวแรก"
            if is_multi_ticker
            else ""
        )

        persona_map = _persona_lookup(packet)
        spec = persona_map.get(speaker_id)
        short_name = spec.short_name if spec else speaker_id
        display_name = spec.display_name if spec else speaker_id

        intensity = normalize_debate_intensity(getattr(packet, "debate_intensity", "balanced"), default="balanced")
        depth = normalize_debate_depth(getattr(packet, "debate_depth", "concise"), default="concise")
        eff_temp = self._effective_temperature(packet, round_stage=round_stage)

        if depth == "deep_youtube":
            length_directive = (
                "3-5 ประโยคภาษาพูดที่เฉียบคม ตอบโต้หมัดต่อหมัด ไร้ Markdown"
                if round_stage == "crossfire"
                else "4-7 ประโยคภาษาพูดเจาะลึกอัดแน่น 2-3 แง่มุมพร้อมตัวเลขหลักฐานจริงแบบจังหวะ Podcast ไร้ Markdown"
            )
        else:
            length_directive = (
                "2-3 ประโยคสั้นคม ไร้ Markdown"
                if round_stage == "crossfire"
                else "2-4 ประโยคภาษาพูดที่เป็นธรรมชาติ ไร้ Markdown"
            )

        human_prompt = (
            f"{shared_blackboard}\n\n---\n\n{role_block}\n\n"
            f"{guard_block}{priority_note}{multi_note}\n"
            f"[DEBATE_INTENSITY: {intensity}] [DEBATE_DEPTH: {depth}]\n\n"
            f"กรุณาตอบกลับเป็น JSON ตามสคีมา `RoundtableTurnPayload` สำหรับบทบาท `{display_name}` (`{speaker_id}`) "
            f"ในรอบ `{round_stage}` (ให้ `spoken_dialogue` มี {length_directive})"
        )
        messages: List[Any] = [SystemMessage(content=sys_prompt), HumanMessage(content=human_prompt)]

        def _fallback(raw_text: str) -> RoundtableTurnPayload:
            spoken = sanitize_spoken_dialogue_for_tts(raw_text)[:520] or (
                f"จากมุมมองของ {short_name} ผมขอประเมินข้อมูลในห้องประชุมนี้ด้วยความระมัดระวังตามตัวเลขที่แนบมาครับ"
            )
            default_target = crossfire_target or (prior_turns[-1].speaker_id if prior_turns else None)
            return RoundtableTurnPayload(
                speaker_id=speaker_id,  # type: ignore[arg-type]
                round_stage=round_stage,  # type: ignore[arg-type]
                replying_to_speaker_id=default_target,
                spoken_dialogue=spoken,
                evidence_bullets=[f"[FACT] วิเคราะห์ภายใต้บริบท {packet.primary_ticker or 'Portfolio/Macro'}"],
                stance_tag="REVIEW",
                sizing_posture="Defensive Trim/Wait" if round_stage in ("rebuttal", "crossfire") else None,
            )

        turn = await self._invoke_structured_or_freetext(
            schema=RoundtableTurnPayload,
            messages=messages,
            fallback_builder=_fallback,
            temperature_override=eff_temp,
        )
        turn = self._sanitize_turn(turn, speaker_id, round_stage)
        if round_stage == "crossfire" and crossfire_target and (
            (turn.replying_to_speaker_id or "").strip() in ("", speaker_id)
        ):
            turn = turn.model_copy(update={"replying_to_speaker_id": crossfire_target})

        if verifier is not None:
            def _verify(cand: RoundtableTurnPayload) -> Any:
                try:
                    return verifier.verify_turn(
                        cand,
                        packet,
                        unspoken_speakers=unspoken_speakers,
                        prior_turns=prior_turns,
                    )
                except TypeError:
                    return verifier.verify_turn(cand, packet)

            v_res = _verify(turn)
            if not v_res.is_valid:
                if v_res.retry_feedback:
                    retry_msgs = messages + [HumanMessage(content=v_res.retry_feedback)]
                    retried = await self._invoke_structured_or_freetext(
                        schema=RoundtableTurnPayload,
                        messages=retry_msgs,
                        fallback_builder=lambda _: turn,
                        temperature_override=eff_temp,
                    )
                    retried = self._sanitize_turn(retried, speaker_id, round_stage)
                    if round_stage == "crossfire" and crossfire_target and (
                        (retried.replying_to_speaker_id or "").strip() in ("", speaker_id)
                    ):
                        retried = retried.model_copy(update={"replying_to_speaker_id": crossfire_target})
                    retry_result = _verify(retried)
                    if retry_result.is_valid:
                        turn = retried
                        v_res = retry_result

                if not v_res.is_valid:
                    safe_turn = self._sanitize_turn(_fallback(""), speaker_id, round_stage)
                    if round_stage == "crossfire" and crossfire_target and (
                        (safe_turn.replying_to_speaker_id or "").strip() in ("", speaker_id)
                    ):
                        safe_turn = safe_turn.model_copy(update={"replying_to_speaker_id": crossfire_target})
                    fallback_result = _verify(safe_turn)
                    if not fallback_result.is_valid:
                        raise ValueError(_verification_failure_message(f"{round_stage} fallback", fallback_result))
                    turn = safe_turn
        return turn

    async def generate_hermes_synthesis(
        self,
        *,
        packet: FrozenContextPacket,
        all_turns: Sequence[RoundtableTurnPayload],
        verifier: Optional[RoundtableVerifierPort] = None,
    ) -> Tuple[RoundtableTurnPayload, DistilledDecisionPlaybook]:
        persona_map = _persona_lookup(packet)
        sys_prompt = self._load_shared_system_prompt()
        shared_blackboard, _ = _format_shared_blackboard_context(packet)
        role_block = self._load_persona_role_block(
            SpeakerId.HERMES_MODERATOR.value, packet=packet, include_focus_lens=False
        )
        transcript_block = opponent_argument_or_opening(
            SpeakerId.HERMES_MODERATOR.value,
            "synthesis",
            all_turns,
            [],
            packet=packet,
        )

        # Build stance breakdown deterministically from latest turn per guru
        latest_stances: Dict[str, str] = {}
        for t in all_turns:
            if t.speaker_id not in ("hermes_moderator", "user"):
                latest_stances[t.speaker_id] = normalize_5tier_stance(t.stance_tag)

        skeptic_spec = persona_map.get(SpeakerId.SKEPTIC_GURU.value)
        skeptic_inactive = skeptic_spec is not None and not skeptic_spec.enabled
        devil_advocate_block = ""
        if skeptic_inactive:
            devil_advocate_block = (
                "\n\n[MANDATORY DEVIL'S ADVOCATE CHECK — SKEPTIC SEAT INACTIVE]\n"
                "หมายเหตุสำคัญ: กูรูสายซักค้าน (skeptic_guru) ถูกปิดการใช้งานในวงประชุมนี้\n"
                "คุณ (Hermes) ต้องทำหน้าที่ตรวจสอบความเสี่ยงด้านลบ (Devil's Advocate) เพิ่มเติมด้วยตนเองก่อนสรุปมติ:\n"
                "1. ตรวจทวน DATA QUALITY FLAGS, ภาระหนี้ (`net_debt_ebitda`) และความสะอาดของกระแสเงินสดใน Packet\n"
                "2. ระบุสมมติฐานที่อาจผิดพลาดได้ (`thesis_falsifiers`) และเหตุการณ์ความเสี่ยงหางอ้วน (`tail_risks`) "
                "ให้ครบถ้วนใน `kill_switches` และ `tension_axes`"
            )

        _missing_lens_duties = (
            ("value_guru", "สาย Value — ตรวจ Intrinsic Value, Margin of Safety และ FCF Yield เทียบ WACC"),
            ("quality_guru", "สาย Quality — ตรวจ ROIC vs WACC, ความแกร่ง Moat, Piotroski F-Score และคุณภาพงบ OCF/NI"),
            ("growth_guru", "สาย Growth — ตรวจ Expectation Gap, EPS Revision 30 วัน และ Guidance/Catalyst 6–18 เดือน"),
            ("macro_tactical_guru", "สาย Macro & Tactical — ตรวจ @Macro Regime, Rate Shock +100bps, Price Stage, Buy Zone และ Stop Loss"),
        )
        missing_seat_lines = [
            f"- ปิดใช้งาน {persona_map[sid].short_name} ({sid}): เติมมุมมองแทน -> {duty}"
            for sid, duty in _missing_lens_duties
            if (sid in persona_map and not persona_map[sid].enabled)
        ]
        if missing_seat_lines:
            devil_advocate_block += (
                "\n\n[MISSING SEAT COVERAGE CHECK — ปิดใช้งานบางสาย]\n"
                "เนื่องจากมีกูรูหลักบางสายปิดใช้งานในรอบนี้ Hermes ต้องประเมินมิติที่ขาดแทนให้ครบก่อนสรุป Playbook:\n"
                + "\n".join(missing_seat_lines)
            )

        active_panel_ids = [
            spec.speaker_id.value
            for spec in sorted(persona_map.values(), key=lambda s: s.debate_order)
            if spec.speaker_id != SpeakerId.HERMES_MODERATOR and spec.enabled
        ]
        bull_fallback_speaker = (
            "growth_guru"
            if "growth_guru" in active_panel_ids
            else (active_panel_ids[0] if active_panel_ids else "hermes_moderator")
        )
        bear_fallback_speaker = (
            "value_guru"
            if "value_guru" in active_panel_ids
            else (active_panel_ids[-1] if active_panel_ids else "hermes_moderator")
        )

        class _HermesCombinedOutput(BaseModel):
            closing_turn: RoundtableTurnPayload
            playbook: DistilledDecisionPlaybook

        human_prompt = (
            f"{shared_blackboard}\n\n---\n\n{role_block}\n\n"
            f"{transcript_block}\n\n"
            f"คะแนนเสียงล่าสุดของกูรูแต่ละสาย: {json.dumps(latest_stances, ensure_ascii=False)}"
            f"{devil_advocate_block}\n\n"
            "ภารกิจของคุณ (Hermes Moderator):\n"
            "1. สร้าง `closing_turn` (บทสรุปปิดวง 2-4 ประโยคพูดธรรมชาติ ไร้ Markdown พร้อมหลักฐาน 2-3 ข้อ)\n"
            "2. สร้าง `playbook` (DistilledDecisionPlaybook ฉบับสมบูรณ์: final_rating, consensus_points, "
            "tension_axes, personal_axis_alignment เทียบ Investment Axis & Portfolio Guard, "
            "action_playbook พร้อม alternatives อย่างน้อย 2 ทางเลือก, และ kill_switches ที่วัดผลได้)\n"
            "อย่าแทงกั๊กเป็น HOLD ถ้าหลักฐานฝั่งหนึ่งหนักแน่นกว่าอย่างชัดเจน และห้ามอคติตามลำดับคนพูดก่อน-หลัง"
        )
        messages: List[Any] = [SystemMessage(content=sys_prompt), HumanMessage(content=human_prompt)]

        def _fallback(raw_text: str) -> _HermesCombinedOutput:
            closing_text = sanitize_spoken_dialogue_for_tts(raw_text)[:360] or (
                "ขอบคุณกูรูทุกท่านครับ ผมขอรวบรวมความเห็นและกรอบความเสี่ยงส่วนตัวของผู้ใช้เพื่อสรุปเป็นแผนปฏิบัติการไว้ในการ์ดตัดสินใจด้านขวาแล้วครับ"
            )
            default_rating = (
                packet.action_stance_hint
                if packet.action_stance_hint in ("HOLD", "OVERWEIGHT", "UNDERWEIGHT")
                else "HOLD"
            )
            default_rating = normalize_5tier_stance(default_rating)
            closing_turn = RoundtableTurnPayload(
                speaker_id="hermes_moderator",
                round_stage="synthesis",
                spoken_dialogue=closing_text,
                evidence_bullets=[
                    f"[FACT] รวมเสียงกูรู {len(latest_stances)} สาย",
                    f"[INFERRED] ประเมินเทียบกรอบความเสี่ยงและพอร์ต ({packet.portfolio_guard.attachment_state.value})",
                ],
                stance_tag=default_rating,  # type: ignore[arg-type]
            )
            playbook = DistilledDecisionPlaybook(
                final_rating=default_rating,  # type: ignore[arg-type]
                action_stance=packet.action_stance_hint or default_rating,
                core_conviction_score=1,
                executive_verdict=closing_text,
                guru_stance_breakdown=latest_stances,  # type: ignore[arg-type]
                consensus_points=["กูรูทุกสายเน้นย้ำเรื่องวินัยบริหารความเสี่ยงและตรวจสอบตัวเลขจาก Fact Packet"],
                tension_axes=[
                    ConsensusTensionAxis(
                        topic="ความคุ้มค่าด้าน Valuation เทียบแรงส่งการเติบโต",
                        bull_side_speaker=bull_fallback_speaker,
                        bull_argument="ศักยภาพการเติบโตยังหนุนระยะกลางถึงยาว",
                        bear_side_speaker=bear_fallback_speaker,
                        bear_argument="ต้องรอส่วนเผื่อความปลอดภัย (Margin of Safety) ให้ชัดขึ้น",
                    )
                ],
                personal_axis_alignment=[
                    PersonalAxisCheckItem(
                        dimension="Portfolio Context Guard",
                        status="not_set"
                        if packet.portfolio_guard.attachment_state.value == "NOT_ATTACHED_DO_NOT_ASSUME"
                        else "aligned",
                        detail=packet.portfolio_guard.summary_text,
                    )
                ],
                portfolio_context_summary=packet.portfolio_guard.summary_text,
                action_playbook=ActionPlaybookSpec(
                    primary_action=ActionPlaybookOption(
                        option_id="primary_balanced",
                        label=f"ปฏิบัติตามแผนสมดุล ({default_rating})",
                        speaker_source="hermes_moderator",
                        conviction_level=1,
                        action_steps=[
                            "รอจังหวะราคาหรือการยืนยันตัวเลขจากงบไตรมาสถัดไปก่อนเพิ่มน้ำหนัก",
                            "ตั้งจุดเตือนความเสี่ยงตาม Kill-Switch ที่กำหนด",
                        ],
                    ),
                    alternatives=[
                        ActionPlaybookOption(
                            option_id="alt_conservative",
                            label="ทางเลือกอนุรักษ์นิยม (Defensive & MOS)",
                            speaker_source=bear_fallback_speaker,
                            conviction_level=2,
                            action_steps=["รอดูแนวรับมูลค่าพื้นฐานหรือ MOS Entry Zone ก่อนทยอยสะสม"],
                        ),
                        ActionPlaybookOption(
                            option_id="alt_tactical",
                            label="ทางเลือกเชิงรุกแบบแบ่งไม้ (Growth & Flow)",
                            speaker_source=bull_fallback_speaker,
                            conviction_level=1,
                            action_steps=["แบ่งเข้าเป็น Tranche เล็กๆ เมื่อโมเมนตัมและมหภาคสนับสนุน"],
                        ),
                    ],
                ),
                kill_switches=["งบการเงินหรือ Margin ต่ำกว่าเกณฑ์เฝ้าระวัง / เกิดสัญญาณผิดสัญญาผู้บริหาร"],
                reflection_summary=(
                    f"อ้างอิงมติครั้งก่อน ({packet.past_scorecard.previous_date}): Alpha {packet.past_scorecard.alpha_pct or 'N/A'}%"
                    if packet.past_scorecard
                    else None
                ),
            )
            return _HermesCombinedOutput(closing_turn=closing_turn, playbook=playbook)

        combined = await self._invoke_structured_or_freetext(
            schema=_HermesCombinedOutput,
            messages=messages,
            fallback_builder=_fallback,
            max_output_tokens=4096,
        )
        closing_turn = self._sanitize_turn(combined.closing_turn, SpeakerId.HERMES_MODERATOR.value, "synthesis")
        playbook = combined.playbook.model_copy(
            update={
                "final_rating": normalize_5tier_stance(combined.playbook.final_rating),
                "guru_stance_breakdown": {
                    k: normalize_5tier_stance(v) for k, v in (combined.playbook.guru_stance_breakdown or latest_stances).items()
                },
                "portfolio_context_summary": combined.playbook.portfolio_context_summary or packet.portfolio_guard.summary_text,
            }
        )

        if verifier is not None:
            def _normalize_playbook(candidate: DistilledDecisionPlaybook) -> DistilledDecisionPlaybook:
                return candidate.model_copy(
                    update={
                        "final_rating": normalize_5tier_stance(candidate.final_rating),
                        "guru_stance_breakdown": {
                            k: normalize_5tier_stance(v)
                            for k, v in (candidate.guru_stance_breakdown or latest_stances).items()
                        },
                        "portfolio_context_summary": candidate.portfolio_context_summary
                        or packet.portfolio_guard.summary_text,
                    }
                )

            def _verify_synthesis(closing, candidate_playbook):
                closing_result = verifier.verify_turn(closing, packet, prior_turns=all_turns)
                playbook_result = verifier.verify_playbook(candidate_playbook, packet)
                return closing_result, playbook_result

            closing_result, playbook_result = _verify_synthesis(closing_turn, playbook)
            if not (closing_result.is_valid and playbook_result.is_valid):
                retry_feedback = "\n\n".join(
                    result.retry_feedback
                    for result in (closing_result, playbook_result)
                    if not result.is_valid and result.retry_feedback
                )
                if retry_feedback:
                    retried = await self._invoke_structured_or_freetext(
                        schema=_HermesCombinedOutput,
                        messages=messages + [HumanMessage(content=retry_feedback)],
                        fallback_builder=lambda _: _HermesCombinedOutput(
                            closing_turn=closing_turn, playbook=playbook
                        ),
                        max_output_tokens=4096,
                    )
                    retry_closing = self._sanitize_turn(
                        retried.closing_turn, SpeakerId.HERMES_MODERATOR.value, "synthesis"
                    )
                    retry_playbook = _normalize_playbook(retried.playbook)
                    retry_closing_result, retry_playbook_result = _verify_synthesis(
                        retry_closing, retry_playbook
                    )
                    if retry_closing_result.is_valid and retry_playbook_result.is_valid:
                        closing_turn = retry_closing
                        playbook = retry_playbook
                        closing_result, playbook_result = retry_closing_result, retry_playbook_result

                if not (closing_result.is_valid and playbook_result.is_valid):
                    safe_output = _fallback("")
                    safe_closing = self._sanitize_turn(
                        safe_output.closing_turn, SpeakerId.HERMES_MODERATOR.value, "synthesis"
                    )
                    safe_playbook = _normalize_playbook(safe_output.playbook)
                    fallback_closing_result, fallback_playbook_result = _verify_synthesis(
                        safe_closing, safe_playbook
                    )
                    failures = []
                    if not fallback_closing_result.is_valid:
                        failures.append(_verification_failure_message("closing fallback", fallback_closing_result))
                    if not fallback_playbook_result.is_valid:
                        failures.append(_verification_failure_message("playbook fallback", fallback_playbook_result))
                    if failures:
                        raise ValueError("; ".join(failures))
                    closing_turn = safe_closing
                    playbook = safe_playbook

        return closing_turn, playbook

    async def generate_interjection_reply(
        self,
        *,
        user_message: str,
        target_speaker_id: str,
        packet: FrozenContextPacket,
        history_turns: Sequence[RoundtableTurnPayload],
        sandbox_calc_output: Optional[str] = None,
        verifier: Optional[RoundtableVerifierPort] = None,
    ) -> RoundtableTurnPayload:
        persona_map = _persona_lookup(packet)
        target_spec = persona_map.get(target_speaker_id)
        target_label = target_spec.display_name if target_spec else target_speaker_id
        sys_prompt = self._load_shared_system_prompt()
        shared_blackboard, _ = _format_shared_blackboard_context(packet)
        role_block = self._load_persona_role_block(target_speaker_id, packet=packet, include_focus_lens=True)
        history_block = opponent_argument_or_opening(
            target_speaker_id, "interjection_reply", history_turns, [], packet=packet
        )
        calc_block = (
            f"\n[ผลการคำนวณจริงจาก Python Sandbox (Decimal)]\n{sandbox_calc_output}\n(ให้ยึดตัวเลขจากผลลัพธ์นี้ 100%)"
            if sandbox_calc_output
            else ""
        )

        human_prompt = (
            f"{shared_blackboard}\n\n---\n\n{role_block}\n\n"
            f"{history_block}\n\n"
            f"[ผู้ใช้กดยกมือแทรกถามคุณ ({target_label} / {target_speaker_id}) โดยตรง]:\n\"{user_message}\"\n"
            f"{calc_block}\n\n"
            "กรุณาตอบคำถามแทรกของผู้ใช้ตรงจุด เป็นภาษาพูด 2-4 ประโยคธรรมชาติ ไร้ Markdown ตามสคีมา `RoundtableTurnPayload`"
        )
        messages: List[Any] = [SystemMessage(content=sys_prompt), HumanMessage(content=human_prompt)]

        def _fallback(raw_text: str) -> RoundtableTurnPayload:
            spoken = sanitize_spoken_dialogue_for_tts(raw_text)[:360] or (
                "รับทราบคำถามครับ จากตัวเลขและบริบทที่มีในห้องประชุม ผมขอตอบประเด็นนี้ตามข้อมูลเชิงประจักษ์ครับ"
            )
            bullets = ["[INFERRED] ตอบคำถามแทรกจากบริบทห้องประชุม"]
            if sandbox_calc_output:
                bullets.insert(0, f"[MATH] ผลคำนวณ Sandbox: {sandbox_calc_output[:140]}")
            return RoundtableTurnPayload(
                speaker_id=target_speaker_id,  # type: ignore[arg-type]
                round_stage="interjection_reply",
                replying_to_speaker_id="user",
                spoken_dialogue=spoken,
                evidence_bullets=bullets,
                stance_tag="REVIEW",
            )

        turn = await self._invoke_structured_or_freetext(
            schema=RoundtableTurnPayload,
            messages=messages,
            fallback_builder=_fallback,
        )
        turn = self._sanitize_turn(turn, target_speaker_id, "interjection_reply").model_copy(
            update={"replying_to_speaker_id": "user"}
        )
        if verifier is not None:
            def _verify_reply(cand: RoundtableTurnPayload) -> Any:
                try:
                    return verifier.verify_turn(cand, packet, unspoken_speakers=(), prior_turns=history_turns)
                except TypeError:
                    return verifier.verify_turn(cand, packet)

            v_res = _verify_reply(turn)
            if not v_res.is_valid:
                if v_res.retry_feedback:
                    retry_msgs = messages + [HumanMessage(content=v_res.retry_feedback)]
                    retried = await self._invoke_structured_or_freetext(
                        schema=RoundtableTurnPayload,
                        messages=retry_msgs,
                        fallback_builder=lambda _: turn,
                    )
                    retried = self._sanitize_turn(retried, target_speaker_id, "interjection_reply").model_copy(
                        update={"replying_to_speaker_id": "user"}
                    )
                    retry_result = _verify_reply(retried)
                    if retry_result.is_valid:
                        turn = retried
                        v_res = retry_result

                if not v_res.is_valid:
                    safe_turn = self._sanitize_turn(
                        _fallback(""), target_speaker_id, "interjection_reply"
                    ).model_copy(update={"replying_to_speaker_id": "user"})
                    fallback_result = _verify_reply(safe_turn)
                    if not fallback_result.is_valid:
                        raise ValueError(_verification_failure_message("interjection reply fallback", fallback_result))
                    turn = safe_turn
        return turn

    async def generate_reflection_lesson(
        self,
        *,
        previous_rating: str,
        raw_return_pct: Optional[str],
        alpha_pct: Optional[str],
        previous_verdict: str,
        ticker: str,
    ) -> str:
        prompt = (
            f"สรุปบทเรียนย้อนหลัง (Reflection Lesson) สำหรับหุ้น {ticker} ให้กระชับ 2 ประโยคภาษาไทย:\n"
            f"- มติรอบก่อน: {previous_rating}\n"
            f"- คำแนะนำเดิม: {previous_verdict[:400]}\n"
            f"- ผลตอบแทนจริงตั้งแต่วันนั้น: {raw_return_pct or 'N/A'}% | Benchmark Alpha: {alpha_pct or 'N/A'}%\n"
            "บอกให้ชัดว่ามุมมองไหนแม่นยำและจุดไหนควรปรับปรุงในรอบถัดไป (ไม่ต้องใส่ Markdown)"
        )
        try:
            llm = self._build_llm()
            resp = await asyncio.to_thread(
                llm.invoke,
                [
                    SystemMessage(content="คุณคือ Hermes Quantitative Reviewer สรุปบทเรียนสั้นคม 2 ประโยค"),
                    HumanMessage(content=prompt),
                ],
            )
            _log_llm_usage("reflection_lesson", resp, model_name=self._model_name)
            cleaned = sanitize_spoken_dialogue_for_tts(_coerce_content_text(resp))
            if cleaned:
                return cleaned[:400]
        except Exception as exc:
            log.debug("Reflection generation failed, using deterministic summary: %s", exc)
        return (
            f"บทเรียนรอบก่อน ({ticker}, มติ {previous_rating}): ผลตอบแทน {raw_return_pct or 'N/A'}% "
            f"(Alpha {alpha_pct or 'N/A'}%) ควรใช้ตัวเลขจริงเป็นตัวถ่วงน้ำหนักความเชื่อมั่นรอบนี้"
        )


def get_ordered_guru_speakers(packet: FrozenContextPacket) -> List[str]:
    """Return enabled panel guru speaker_ids ordered by customized debate_order."""
    return [
        spec.speaker_id.value
        for spec in sorted(resolve_personas_from_packet(packet), key=lambda s: s.debate_order)
        if spec.speaker_id != SpeakerId.HERMES_MODERATOR and spec.enabled
    ]

