"""Renderer for Briefing Book to transform Draft into Markdown."""
import re
from typing import List

from schemas.briefing_book_schemas import (
    InvestigativeBriefingBookDraft,
    BriefingEvidenceBundle,
    RenderedBriefing,
    RenderedVisualMarker,
)


def _collect_ids(text: str, prefix: str) -> set:
    if not text:
        return set()
    return set(re.findall(r"\[(" + prefix + r"\d+)\]", str(text)))


def render_briefing_book(
    draft: InvestigativeBriefingBookDraft,
    bundle: BriefingEvidenceBundle
) -> RenderedBriefing:
    lines = [f"# {draft.title}"]
    lines.append(draft.executive_summary)
    
    section_names = ["Main Title (H1)", "Executive Summary"]
    
    def strip_markers(text):
        if not text: return ""
        return re.sub(r'\[VISUAL_EVIDENCE[^\]]*\]', '', str(text))
        
    acts = [
        ("Act I", strip_markers(draft.act1_script)),
        ("Act II", strip_markers(draft.act2_script)),
        ("Act III", strip_markers(draft.act3_script))
    ]
    
    directives_by_act = {}
    visual_markers = []
    
    for d in draft.visual_directives:
        act = getattr(d, "act", "")
        directives_by_act.setdefault(act, []).append(d)
        
    for act_name, script in acts:
        lines.append(f"## {act_name}")
        section_names.append(act_name)
        if script.strip():
            lines.append(script.strip())
        
        for d in directives_by_act.get(act_name, []):
            ev_ids = getattr(d, "evidence_ids", [])
            vid = getattr(d, 'visual_id', '')
            ev_str = ",".join(ev_ids) if ev_ids else "UNKNOWN"
            lines.append(f"[VISUAL_EVIDENCE id={vid} evidence={ev_str}]")
            lines.append(
                f"Chart: {getattr(d, 'title', '')} ({getattr(d, 'chart_type', '')})\n"
                f"Period: {getattr(d, 'date_range', '')}\n"
                f"Series: {', '.join(getattr(d, 'series_keys', []))}\n"
                f"Sources: {', '.join(getattr(d, 'sources', []))}\n"
                f"Annotation: {getattr(d, 'annotation', '')}"
            )
            visual_markers.append(RenderedVisualMarker(id=vid, evidence_ids=ev_ids, act=act_name))
            
    if draft.causality_scenarios:
        lines.append("## Causality Scenarios")
        section_names.append("Causality Scenarios")
        for sc in draft.causality_scenarios:
            lines.append(f"### {sc.name}")
            lines.append(sc.description)
            lines.append(f"Time Horizon: {sc.time_horizon}")
            lines.append(f"Probability: {sc.probability_pct:.0f}%")
            if sc.trigger_conditions:
                lines.append("Trigger Conditions:")
                for condition in sc.trigger_conditions:
                    lines.append(f"- {condition}")
            if sc.falsification_triggers:
                lines.append("Falsification Triggers:")
                for trigger in sc.falsification_triggers:
                    lines.append(f"- {trigger}")
            if sc.evidence_ids:
                lines.append(f"Evidence: {', '.join(f'[{eid}]' for eid in sc.evidence_ids)}")
            if sc.threshold_basis:
                lines.append(f"Threshold Basis: {sc.threshold_basis}")

    if draft.asset_impacts:
        lines.append("## Asset Impacts")
        section_names.append("Asset Impacts")
        for imp in draft.asset_impacts:
            lines.append(f"### {imp.symbol_or_name}")
            lines.append(f"Direction: {imp.impact_type}")
            lines.append(f"Reasoning: {imp.reasoning}")
            if imp.risk_factors:
                lines.append("Risk Factors:")
                for factor in imp.risk_factors:
                    lines.append(f"- {factor}")
            lines.append(f"Invalidation: {', '.join(imp.invalidation_conditions)}")
            if imp.evidence_ids:
                lines.append(f"Evidence: {', '.join(f'[{eid}]' for eid in imp.evidence_ids)}")

    if draft.bull_case or draft.bear_case:
        lines.append("## Bull & Bear Cases")
        section_names.append("Bull & Bear Cases")
        lines.append(f"Bull: {draft.bull_case}")
        lines.append(f"Bear: {draft.bear_case}")

    if draft.falsification_triggers:
        lines.append("## Falsification Triggers")
        section_names.append("Falsification Triggers")
        for trigger in draft.falsification_triggers:
            lines.append(f"- {trigger}")

    if draft.notebooklm_prompts:
        lines.append("## NotebookLM Prompts")
        section_names.append("NotebookLM Prompts")
        for p in draft.notebooklm_prompts:
            ptype = getattr(p, "prompt_type", "GENERAL")
            q = getattr(p, "question_or_prompt", str(p))
            out = getattr(p, "expected_output_format", "")
            lines.append(f"- [{ptype}] {q}\n  *Output Format: {out}*")

    if getattr(draft, "audio_overview_directive", None) and draft.audio_overview_directive.strip():
        lines.append("## NotebookLM Audio Directive")
        section_names.append("NotebookLM Audio Directive")
        lines.append(draft.audio_overview_directive.strip())


    lines.append("## Evidence Ledger")
    section_names.append("Evidence Ledger")
    if bundle.sources:
        lines.append("### Sources")
        for s in bundle.sources:
            lines.append(f"- **[{s.source_id}]** {s.original_title} ({s.publisher}, {s.published_at or 'unverified'})")
    if bundle.evidence_items:
        lines.append("### Facts")
        for e in bundle.evidence_items:
            lines.append(f"- **[{e.evidence_id}]** {e.claim} (Sources: {', '.join(e.source_ids)})")

    if bundle.macro_snapshot and bundle.macro_snapshot.observations:
        lines.append("### Macro Snapshot")
        for obs in bundle.macro_snapshot.observations:
            lines.append(f"- **{obs.category}**: {obs.value} {obs.unit} (Change: {obs.change_pct}%)")

    if bundle.financial_snapshots:
        lines.append("### Financial Snapshots")
        for fin in bundle.financial_snapshots:
            lines.append(f"#### Symbol: {fin.symbol}")
            if fin.revenue: lines.append(f"- Revenue: {fin.revenue}")
            if fin.net_income: lines.append(f"- Net Income: {fin.net_income}")
            if fin.fcf: lines.append(f"- FCF: {fin.fcf}")
            if fin.total_debt: lines.append(f"- Total Debt: {fin.total_debt}")
            if fin.health_notes: lines.append(f"- Notes: {fin.health_notes}")
            for period in fin.periods:
                lines.append(f"  - **{period.fiscal_period_end}**: Rev={period.total_revenue}, NI={period.net_income}, FCF={period.free_cash_flow}")

    content = "\n\n".join(lines)
    
    # Validation: Ensure critical sections were rendered
    missing_sections = []
    if not draft.executive_summary: missing_sections.append("Executive Summary")
    if not draft.causality_scenarios: missing_sections.append("Causality Scenarios")
    if not draft.asset_impacts: missing_sections.append("Asset Impacts")
    if not draft.falsification_triggers: missing_sections.append("Falsification Triggers")
    
    if missing_sections:
        raise ValueError(f"Draft is missing critical required fields: {missing_sections}")

    cited_evidence_ids = _collect_ids(content, "E")
    cited_source_ids = _collect_ids(content, "S")
    
    return RenderedBriefing(
        content=content,
        section_names=section_names,
        visual_markers=visual_markers,
        cited_evidence_ids=cited_evidence_ids,
        cited_source_ids=cited_source_ids,
    )


def append_data_gap_notes(
    content: str,
    *,
    numeric_warnings: List[str],
    macro_unavailable_reasons: List[str],
) -> str:
    """Write known data-completeness gaps into the markdown itself.

    The quality gate can already detect these (e.g. a narrated citation whose
    number never appears nearby, or a macro snapshot that failed to fetch)
    but as non-blocking warnings that only ever reached the ``.quality.json``
    sidecar. NotebookLM only ever ingests this ``.md`` file, so a gap that
    stays sidecar-only is invisible to it — this makes the gap part of the
    text it actually reads.
    """
    if not numeric_warnings and not macro_unavailable_reasons:
        return content

    lines = ["## Data Gaps"]
    for warning in numeric_warnings:
        lines.append(f"- {warning} — ควรถือว่าตัวเลขนี้ยังไม่ได้ยืนยันในเนื้อหา เมื่อสรุปควรระบุว่าเป็นข้อมูลที่ยังไม่สมบูรณ์")
    for reason in macro_unavailable_reasons:
        lines.append(f"- ข้อมูลเศรษฐกิจมหภาคบางส่วนดึงไม่สำเร็จ: {reason}")

    return content + "\n\n" + "\n\n".join(lines)


def prepend_unverified_draft_banner(content: str, *, reason: str, issues: List[str]) -> str:
    """Mark an Unverified Draft as such inside the document body itself.

    Without this, the "unverified" trust tier only exists in the filename
    suffix and the ``.quality.json`` sidecar — a reader (or NotebookLM) that
    only opens the ``.md`` has no way to know the provenance was overridden.
    """
    banner_lines = [
        "> [!WARNING] Unverified Draft",
        f"> เอกสารนี้ถูกสร้างโดยข้ามเกณฑ์ provenance บางส่วน เหตุผล: {reason}",
    ]
    for issue in issues:
        banner_lines.append(f"> - {issue}")

    banner = "\n".join(banner_lines)
    if "\n\n" in content:
        title_line, rest = content.split("\n\n", 1)
        return f"{title_line}\n\n{banner}\n\n{rest}"
    return f"{content}\n\n{banner}"
