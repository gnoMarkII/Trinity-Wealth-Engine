"""LLM-backed adapter for extracting and resolving earnings promises."""
import json
import logging
import re
from typing import Any, Dict, List, Optional
import httpx

from core.equity_analytics.canary_guard import scan_untrusted_text
from core.financial_parsers import extract_json_block
from core.llm_factory import get_llm, detect_provider
from core.model_registry import get_model_name

log = logging.getLogger(__name__)


def _extract_json_block(text: Any) -> Optional[dict]:
    """Extract and parse first valid JSON object from LLM response."""
    if isinstance(text, list):
        text = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in text
        )
    return extract_json_block(text)


def _sanitize_transcript_for_llm(transcript_text: str) -> str:
    """Scan untrusted earnings call text before LLM synthesis and neutralize injection strings."""
    scan = scan_untrusted_text(transcript_text or "", raise_on_critical=False)
    safe_text = scan.sanitized_text
    if not scan.is_safe:
        rules = [v.get("rule", "UNKNOWN") for v in scan.violations]
        log.warning("Canary guard sanitized prompt injection patterns in transcript: %s", rules)
        for violation in scan.violations:
            matched = violation.get("matched_text")
            if matched:
                safe_text = safe_text.replace(matched, "[REDACTED_INJECTION]")
    return safe_text


class PromiseExtractorAdapter:
    """Implements PromiseExtractorPort using configured LLM with strict verbatim discipline."""

    def __init__(self, slot_key: str = "promise_extractor") -> None:
        self._slot_key = slot_key

    def _get_llm_client(self):
        model_name = get_model_name(self._slot_key)
        provider = detect_provider(model_name)
        return get_llm(provider=provider, model_name=model_name).with_retry(
            retry_if_exception_type=(
                httpx.TimeoutException,
                httpx.ConnectError,
                httpx.RemoteProtocolError,
                TimeoutError,
                ConnectionError,
            ),
            wait_exponential_jitter=True,
            stop_after_attempt=3,
        )

    def extract_commitments_from_transcript(
        self,
        company: str,
        ticker: str,
        fiscal_quarter: str,
        call_date: str,
        fy_end_month: str,
        open_promises: List[Dict[str, Any]],
        transcript_text: str,
    ) -> Dict[str, Any]:
        """Extract checkable promises, reiterations, and restatements from a single transcript."""
        open_context = ""
        if open_promises:
            open_context = "CURRENT OPEN PROMISES IN LEDGER (for reiterations/restatements matching):\n"
            for p in open_promises:
                tgt_desc = (
                    f"Point: {p.get('target_point')}" if p.get("target_point") is not None
                    else f"Range: {p.get('target_low')} - {p.get('target_high')}" if p.get("target_low") is not None
                    else p.get("target_kind") or ""
                )
                open_context += (
                    f"- ID: {p.get('id')}, Metric: {p.get('metric')}, "
                    f"Due: {p.get('due')}, Target: {tgt_desc}\n"
                )

        prompt = f"""You are extracting management commitments from the {company} ({ticker}) {fiscal_quarter} earnings call held on {call_date}. The company's fiscal year ends in {fy_end_month}.

Read prepared remarks AND the Q&A. Extract statements that are ALL of:
 1. forward-looking (a future outcome or action, not a recap),
 2. spoken by a named company executive (an analyst's framing counts only if the executive confirms it),
 3. about the company's own results or actions (not macro, industry, customers or competitors),
 4. checkable: a number or explicit direction with a period (Type A), or a deliverable with a deadline (Type B).

Do NOT extract: "over time / long term / eventually / multi-year" statements; buybacks, dividends, M&A; vague intent ("continue to invest in", "focus on"); safe-harbour boilerplate; operator/IR host remarks.

Cap: at most 12 promises. Prioritise (1) company-level next-quarter and full-year guidance, (2) segment guidance, (3) milestones named in prepared remarks, (4) commitments made under analyst pressure in Q&A, (5) the rest.

If a target here merely repeats one already in the ledger unchanged, do not create a new row — report it under "reiterations" with the existing id.
If a target here CHANGES one already in the ledger, report it under "restatements" with the existing id, the new target and direction (raised/lowered); a new row will be opened for the new target.

Timing → due quarter: explicit date/quarter → that quarter [explicit]; "this year / by year-end / full year" → Q4 of that FY [explicit]; "second half / H2" → Q4 [inferred]; "soon / shortly / coming weeks" → +1 quarter [inferred]; "coming quarters / near term / next few months" → +2 quarters [inferred]; "next year" spoken in Q1–Q3 → Q4 of following FY [inferred].

{open_context}

TRANSCRIPT TEXT:
{_sanitize_transcript_for_llm(transcript_text)[:90000]}

Return JSON only with this schema:
{{
  "promises": [
    {{
      "speaker": "Name, Title",
      "section": "prepared" | "qa",
      "type": "A_numeric" | "B_milestone",
      "category": "revenue|eps|margin|opex|capex|fcf|units|growth|product|capacity|regulatory|org|other",
      "scope": "company" | "segment" | "n/a",
      "quote": "verbatim, max 40 words",
      "metric": "e.g. Q3 FY27 revenue" | "deliverable description",
      "metric_th": "the same short description in natural Thai — headline for Thai retail reader under 12 words",
      "target_kind": "range|point|floor|ceiling|direction|milestone",
      "target_low": number|null,
      "target_high": number|null,
      "target_point": number|null,
      "direction": "up|down|flat"|null,
      "unit": "USD bn|%|pts|units|..."|null,
      "basis": "GAAP|non-GAAP|constant currency|organic|n/a",
      "due": "FYyyyyQn",
      "due_precision": "explicit" | "inferred",
      "notes": ""
    }}
  ],
  "reiterations": [ {{ "existing_id": "P-...", "quote": "..." }} ],
  "restatements": [ {{ "existing_id": "P-...", "quote": "...", "new_target_low": null, "new_target_high": null, "new_target_point": null, "direction": "raised|lowered", "new_due": "FYyyyyQn" }} ]
}}
"""
        llm = self._get_llm_client()
        resp = llm.invoke(prompt)
        content = resp.content if hasattr(resp, "content") else str(resp)
        data = _extract_json_block(content)
        if not data or not isinstance(data, dict):
            log.warning("Promise extraction returned invalid JSON for %s %s", ticker, fiscal_quarter)
            return {"promises": [], "reiterations": [], "restatements": []}
        return data

    def resolve_promise(
        self,
        promise: Dict[str, Any],
        resolving_quarter: str,
        grace_quarter: Optional[str],
        transcript_text: str,
    ) -> Dict[str, Any]:
        """Check promise outcome against later transcript text."""
        target_desc = (
            f"Point: {promise.get('target_point')}" if promise.get("target_point") is not None
            else f"Range: {promise.get('target_low')} - {promise.get('target_high')}" if promise.get("target_low") is not None
            else promise.get("target_kind") or ""
        )
        prompt = f"""Promise {promise.get('id')} was made by {promise.get('speaker')} on the {promise.get('made_in')} call:
  "{promise.get('quote')}"
  Metric: {promise.get('metric')}. Target: {target_desc} {promise.get('unit') or ''}. Due: {promise.get('due')}. Basis: {promise.get('basis')}.

Search the {resolving_quarter} transcript (and grace quarter {grace_quarter or 'n/a'}) for what actually happened. Look for the metric name, product name, synonyms, and analyst questions about it.

Decide the status using this order, first match wins:
 - Restated: management explicitly changed the target before it was due → direction raised/lowered, new target.
 - Delivered: range → actual within or above; floor → actual >= low; ceiling → actual <= high; point → within tolerance (USD metrics +-1%, margins +-0.5 pt, EPS +-3%, counts/growth +-3%) or better; direction → held; milestone → confirmed on/before due.
 - Partial: numeric shortfall up to 5% (1 pt for margins) beyond tolerance; milestone <= 1 quarter late or delivered in reduced scope.
 - Missed: worse than Partial, and management acknowledged it.
 - Quietly dropped: due + 1 quarter passed, nothing delivered, never mentioned again, never restated.
 - Unverifiable: due passed, but transcript does not provide sufficient data to judge.

TRANSCRIPT TEXT:
{_sanitize_transcript_for_llm(transcript_text)[:90000]}

Return JSON only:
{{
  "status": "Delivered" | "Partial" | "Missed" | "Restated" | "Quietly dropped" | "Unverifiable",
  "restated_direction": "raised" | "lowered" | null,
  "resolved_in": "{resolving_quarter}",
  "evidence_quote": "verbatim from the resolving transcript, max 40 words",
  "outcome_value": number | null,
  "notes": "basis mismatches, ambiguity, why this status",
  "notes_th": "the same note in natural Thai"
}}
If nothing in the transcript speaks to this promise, return status Unverifiable with an empty evidence_quote — never invent a quote.
"""
        llm = self._get_llm_client()
        resp = llm.invoke(prompt)
        content = resp.content if hasattr(resp, "content") else str(resp)
        data = _extract_json_block(content)
        if not data or not isinstance(data, dict):
            return {
                "status": "Unverifiable",
                "resolved_in": resolving_quarter,
                "evidence_quote": "",
                "notes": "LLM returned unparseable resolution",
                "notes_th": "ไม่สามารถแปลงผลลัพธ์การตรวจสอบได้",
            }
        return data
