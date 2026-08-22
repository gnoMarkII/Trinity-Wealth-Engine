"""Quality Gate for Briefing Book to validate structure and evidence integrity."""
import re
from typing import Any, Tuple, List

from schemas.briefing_book_schemas import (
    ResearchQualityReport,
    QualityIssueRecord,
    BriefingEvidenceBundle,
    InvestigativeBriefingBookDraft,
    RenderedBriefing,
)


def _years_in_text(text: str) -> set:
    if not text:
        return set()
    return set(re.findall(r"\b(19\d{2}|20\d{2})\b", str(text)))


def validate_briefing_book_structure(rendered: RenderedBriefing) -> Tuple[bool, List[str]]:
    missing = []
    
    expected_markdown_headers = [
        ("Main Title (H1)", r"^#\s+"),
        ("Act I", r"^##\s*Act I"),
        ("Act II", r"^##\s*Act II"),
        ("Act III", r"^##\s*Act III"),
        ("Causality Scenarios", r"^##\s*Causality Scenarios"),
        ("Asset Impacts", r"^##\s*Asset Impacts"),
        ("Falsification Triggers", r"^##\s*Falsification Triggers"),
        ("NotebookLM Prompts", r"^##\s*NotebookLM Prompts"),
        ("Evidence Ledger", r"^##\s*Evidence Ledger"),
    ]
    
    for name, pattern in expected_markdown_headers:
        if not re.search(pattern, rendered.content, flags=re.MULTILINE | re.IGNORECASE):
            missing.append(f"{name} Section" if not name.endswith(")") else name)
            
    if "UNKNOWN" in rendered.content:
        missing.append("Placeholder 'UNKNOWN' detected in markdown")
        
    return len(missing) == 0, missing


def _is_placeholder_provenance(value: str) -> bool:
    if not value: return True
    val = value.casefold()
    return val in {"mock", "n/a", "unknown", "placeholder"} or val.startswith("test")


def _thai_ngram_jaccard(text1: str, text2: str, n: int = 3) -> float:
    import unicodedata
    t1 = re.sub(r"\s+", "", unicodedata.normalize("NFC", text1).casefold())
    t2 = re.sub(r"\s+", "", unicodedata.normalize("NFC", text2).casefold())
    if not t1 or not t2:
        return 0.0
    if len(t1) < n or len(t2) < n:
        return 1.0 if t1 == t2 else 0.0
    ng1 = {t1[i:i+n] for i in range(len(t1) - n + 1)}
    ng2 = {t2[i:i+n] for i in range(len(t2) - n + 1)}
    return len(ng1.intersection(ng2)) / len(ng1.union(ng2))


def _check_script_loopbacks(draft: InvestigativeBriefingBookDraft, add_issue) -> None:
    def clean_text(text: Any) -> str:
        if not isinstance(text, str):
            return ""
        t = re.sub(r"\[VISUAL_EVIDENCE[^\]]*\]", "", text)
        t = re.sub(r"\[E\d+\]", "", t)
        t = re.sub(r"\[S\d+\]", "", t)
        return t.strip()

    def get_sentences(text: Any) -> List[str]:
        cleaned = clean_text(text)
        if not cleaned:
            return []
        raw_sents = re.split(r"[\n\.\?!]+", cleaned)
        return [s.strip() for s in raw_sents if len(s.strip()) >= 15]

    act1_sents = get_sentences(getattr(draft, "act1_script", ""))
    act2_sents = get_sentences(getattr(draft, "act2_script", ""))
    act3_sents = get_sentences(getattr(draft, "act3_script", ""))


    acts = [("Act I", act1_sents), ("Act II", act2_sents), ("Act III", act3_sents)]
    for i in range(len(acts)):
        for j in range(i + 1, len(acts)):
            name_a, sents_a = acts[i]
            name_b, sents_b = acts[j]
            for sa in sents_a:
                for sb in sents_b:
                    sim = _thai_ngram_jaccard(sa, sb, n=3)
                    if sim >= 0.85:
                        add_issue(
                            "SCRIPT_LOOPBACK_WARNING",
                            "other",
                            "warning",
                            f"ตรวจพบประโยคซ้ำซ้อนระหว่าง {name_a} และ {name_b} (similarity {sim:.2f}): '{sa[:40]}...'",
                            bypassable=True,
                        )
                        return


def validate_briefing_book_quality(

    evidence_bundle: BriefingEvidenceBundle,
    draft: InvestigativeBriefingBookDraft,
    rendered_briefing: RenderedBriefing,
) -> ResearchQualityReport:
    """Grade evidence integrity before prose quality, and refuse false 100s."""
    
    weights = {
        "structure": 10,
        "provenance_integrity": 15,
        "claim_evidence_traceability": 20,
        "quantitative_autopsy": 15,
        "analytical_discipline": 15,
        "asset_relevance": 10,
        "detective_script_visuals": 10,
        "notebooklm_prompts": 5,
    }
    rubric = {key: 0 for key in weights}
    issues = []
    advisories = []
    score_cap = 100

    def add_issue(code: str, category: str, severity: str, desc: str, bypassable: bool = False, ev_ids=None):
        issues.append(QualityIssueRecord(
            code=code,
            category=category,
            severity=severity,
            description=desc,
            bypassable=bypassable,
            evidence_ids=ev_ids or []
        ))

    def apply_cap(limit: int, code: str, desc: str, bypassable: bool = False):
        nonlocal score_cap
        score_cap = min(score_cap, limit)
        add_issue(code, "structure", "cap", desc, bypassable)

    sources = evidence_bundle.sources or []
    evidence = evidence_bundle.evidence_items or []
    source_ids = {source.source_id for source in sources}
    evidence_ids = {item.evidence_id for item in evidence}
    core_sources = [source for source in sources if getattr(source, "source_type", "") != "data_provider"]

    structure_ok, missing_structure = validate_briefing_book_structure(rendered_briefing)
    if structure_ok:
        rubric["structure"] = weights["structure"]
    else:
        for item in missing_structure:
            add_issue("MISSING_STRUCTURE", "structure", "blocker", f"Structural requirement missing: {item}")

    if not core_sources:
        add_issue("NO_PROVENANCE_SOURCE", "provenance", "blocker", "No core news/commentary source with provenance was supplied")
    else:
        bad_provenance = False
        if all(getattr(source, "verification_status", "unverified") == "unverified" for source in core_sources):
            add_issue("UNVERIFIED_ALL_SOURCES", "provenance", "blocker", "All core sources are unverified; a briefing cannot be publishable")
            bad_provenance = True
        groups = set()
        for source in core_sources:
            ind_key = getattr(source, "independence_key", "")
            groups.add(ind_key)
            if source.publisher.casefold() == "mock" or "mock" in ind_key.casefold():
                add_issue("MOCK_SOURCE", "provenance", "blocker", f"Mock source detected: {source.source_id}")
                bad_provenance = True
            
            ver_status = getattr(source, "verification_status", "unverified")
            if ver_status == "unverified" and source.published_at == source.ingested_at:
                add_issue("INGESTION_AS_PUBLISHED", "provenance", "blocker", f"Unverified source uses ingestion time as publication time: {source.source_id}")
                bad_provenance = True
            if ver_status == "verified" and (
                not source.published_at or getattr(source, "url", "N/A") == "N/A" or _is_placeholder_provenance(source.publisher)
            ):
                add_issue("INCOMPLETE_CANONICAL", "provenance", "blocker", f"Verified source has incomplete canonical provenance: {source.source_id}")
                bad_provenance = True
            if ver_status != "verified":
                add_issue("UNVERIFIED_SOURCE_WARNING", "provenance", "warning", f"Core source {source.source_id} is {ver_status}, not verified")
            if not source.published_at:
                apply_cap(70, "MISSING_PUBLICATION_DATE", f"Core source {source.source_id} has no publication date", bypassable=True)
        if len(groups) < 2:
            apply_cap(85, "SINGLE_INDEPENDENT_SOURCE", "Only one independent core source group is available", bypassable=True)
        if not bad_provenance:
            rubric["provenance_integrity"] = weights["provenance_integrity"]

    evidence_problem = False
    if not evidence_ids:
        add_issue("EMPTY_EVIDENCE_LEDGER", "structure", "blocker", "Evidence ledger is empty")
        evidence_problem = True
    metric_values = {}
    for item in evidence:
        if not item.source_ids or set(item.source_ids) - source_ids:
            add_issue("INVALID_SOURCE_REFERENCE", "structure", "blocker", f"Evidence {item.evidence_id} has invalid source references")
            evidence_problem = True
            
        statuses = [getattr(source, "verification_status", "unverified") for source in sources if source.source_id in item.source_ids]
        if getattr(item, "classification", "") == "verified_fact" and "unverified" in statuses:
            add_issue("UNVERIFIED_FACT", "provenance", "blocker", f"Unverified source cannot support verified fact {item.evidence_id}")
            evidence_problem = True
        if getattr(item, "classification", "") == "consensus":
            groups = {getattr(source, "independence_key", "") for source in sources if source.source_id in item.source_ids}
            if len(groups) < 2:
                add_issue("INVALID_CONSENSUS", "provenance", "blocker", f"Consensus evidence {item.evidence_id} lacks two independent sources")
                evidence_problem = True
        if getattr(item, "metric_name", None) and getattr(item, "value", None) is not None:
            metric_values.setdefault(item.metric_name.casefold(), []).append(item)
            
    for metric_name, items in metric_values.items():
        values = {round(float(item.value), 8) for item in items if item.value is not None}
        def lacks_time_semantics(item: Any) -> bool:
            semantics = getattr(item, "time_semantics", "unknown")
            if semantics == "observed":
                return not getattr(item, "observed_at", None)
            if semantics == "reported":
                return not getattr(item, "reported_at", None)
            if semantics == "fiscal_period":
                return not getattr(item, "observed_at", None)
            return True

        if len(values) > 1 and any(lacks_time_semantics(item) for item in items):
            add_issue("CONFLICTING_METRICS", "numeric", "blocker", f"Conflicting metric values for '{metric_name}' lack time semantics")
            evidence_problem = True

    def _collect_ids(text: str, prefix: str) -> set:
        if not text: return set()
        return set(re.findall(r"\[(" + prefix + r"\d+)\]", str(text)))

    invalid_references = set()
    invalid_references.update(rendered_briefing.cited_evidence_ids - evidence_ids)
    invalid_references.update(rendered_briefing.cited_source_ids - source_ids)
    
    for item in sorted(invalid_references):
        add_issue("UNKNOWN_IDENTIFIER", "structure", "blocker", f"Narrative cites unknown identifier: {item}")
        
    if evidence_ids and not _collect_ids(draft.executive_summary, "E"):
        add_issue("NO_EVIDENCE_CITATION", "structure", "blocker", "Executive summary has no evidence citation")

    # Numeric Grounding Rule
    evidence_by_id = {item.evidence_id: item for item in evidence}
    narrative_fields = [
        str(getattr(draft, "executive_summary", "")), str(getattr(draft, "bull_case", "")),
        str(getattr(draft, "bear_case", "")), str(getattr(draft, "act1_script", "")),
        str(getattr(draft, "act2_script", "")), str(getattr(draft, "act3_script", "")),
    ]
    for field in narrative_fields:
        for match in re.finditer(r"\[(E\d+)\]", field):
            e_id = match.group(1)
            if e_id in evidence_by_id:
                ev = evidence_by_id[e_id]
                if getattr(ev, "value", None) is not None:
                    start = max(0, match.start() - 500)
                    end = min(len(field), match.end() + 500)
                    surrounding = field[start:end]
                    numbers = re.findall(r"\d[\d,]*(?:\.\d+)?", surrounding)
                    found = False
                    try:
                        ev_val = float(ev.value)
                        for num_str in numbers:
                            try:
                                num_val = float(num_str.replace(",", ""))
                                diff = abs(num_val - ev_val)
                                if diff < 0.05 or (ev_val != 0 and (diff / abs(ev_val)) < 0.02):
                                    found = True
                                    break
                            except (ValueError, ZeroDivisionError):
                                pass
                    except (ValueError, TypeError):
                        found = True  # If ev.value itself is non-numeric, don't fail numeric check
                    if not found:
                        add_issue("NUMERIC_GROUNDING_WARNING", "numeric", "warning", f"Narrative cites {e_id} but its numeric value ({ev.value}) is not found nearby", ev_ids=[e_id])

    if invalid_references:
        rubric["claim_evidence_traceability"] = 0
    else:
        rubric["claim_evidence_traceability"] = weights["claim_evidence_traceability"]

    quantitative_problem = False
    mode = getattr(evidence_bundle, "investigation_mode", "mixed")
    macro = getattr(evidence_bundle, "macro_snapshot", None)
    financials = list(getattr(evidence_bundle, "financial_snapshots", []) or [])
    
    if mode in {"stock", "mixed"}:
        if not financials or not any(getattr(snapshot, "status", "") == "success" and getattr(snapshot, "periods", []) for snapshot in financials):
            add_issue("FINANCIAL_PROVIDER_UNAVAILABLE", "financial", "blocker", "No usable financial statement was returned for the selected stock (required for stock/mixed modes)", bypassable=True)
            quantitative_problem = True

    if mode in {"macro", "mixed"}:
        if not (macro and getattr(macro, "observations", [])):
            apply_cap(70, "MISSING_MACRO_SNAPSHOT", "Macro/mixed briefing has no provider macro snapshot", bypassable=True)
        else:
            sectors = {getattr(obs, "category", "") for obs in macro.observations if getattr(obs, "category", None)}
            has_inflation = any("inflation" in s.lower() or "cpi" in s.lower() for s in sectors)
            has_rates = any("rate" in s.lower() or "yield" in s.lower() for s in sectors)
            has_energy = any("oil" in s.lower() or "energy" in s.lower() for s in sectors)
            
            if not has_inflation:
                add_issue("MACRO_MISSING_INFLATION", "macro", "blocker", "Macro data missing inflation sector coverage", bypassable=True)
                quantitative_problem = True
            if not has_rates:
                add_issue("MACRO_MISSING_RATES", "macro", "blocker", "Macro data missing rates sector coverage", bypassable=True)
                quantitative_problem = True
            
            stale_count = sum(1 for observation in macro.observations if getattr(observation, "is_stale", False))
            if stale_count == len(macro.observations):
                apply_cap(85, "STALE_MACRO_SNAPSHOT", "All macro observations are stale", bypassable=True)
            elif stale_count:
                advisories.append(f"{stale_count} optional macro observations are stale and must not support current claims")

    if macro and getattr(macro, "observations", []) and not any(getattr(item, "value", None) is not None for item in evidence):
        add_issue("MACRO_NOT_NUMERIC", "macro", "blocker", "Macro observations were not represented as numeric evidence")
        quantitative_problem = True
        
    if not quantitative_problem:
        rubric["quantitative_autopsy"] = weights["quantitative_autopsy"]

    scenario_problem = False
    scenarios = list(getattr(draft, "causality_scenarios", []) or [])
    if len(scenarios) < 3:
        add_issue("INSUFFICIENT_SCENARIOS", "structure", "blocker", "At least three causality scenarios are required")
        scenario_problem = True
    probability_total = sum(float(getattr(scenario, "probability_pct", 0)) for scenario in scenarios)
    if scenarios and not 95 <= probability_total <= 105:
        add_issue("SCENARIO_PROBABILITY", "numeric", "blocker", f"Scenario probabilities must total approximately 100%, got {probability_total:.1f}%")
        scenario_problem = True
    for scenario in scenarios:
        s_ev_ids = getattr(scenario, "evidence_ids", [])
        if not s_ev_ids or set(s_ev_ids) - evidence_ids:
            add_issue("SCENARIO_INVALID_EVIDENCE", "structure", "blocker", f"Scenario {getattr(scenario, 'scenario_id', 'unknown')} has invalid evidence references")
            scenario_problem = True
        if not getattr(scenario, "trigger_conditions", []) or not getattr(scenario, "falsification_triggers", []):
            add_issue("SCENARIO_LACKS_TRIGGERS", "structure", "blocker", f"Scenario {getattr(scenario, 'scenario_id', 'unknown')} lacks triggers or falsification criteria")
            scenario_problem = True
        trigger_text = " ".join([*getattr(scenario, "trigger_conditions", []), *getattr(scenario, "falsification_triggers", [])])
        if re.search(r"\d{2,}", trigger_text) and not getattr(scenario, "threshold_basis", "").strip():
            apply_cap(85, "UNATTRIBUTED_SCENARIO_THRESHOLD", f"Scenario {getattr(scenario, 'scenario_id', 'unknown')} has numeric triggers without a threshold basis")
    if not scenario_problem:
        rubric["analytical_discipline"] = weights["analytical_discipline"]

    asset_problem = False
    for asset in getattr(draft, "asset_impacts", []) or []:
        a_ev_ids = getattr(asset, "evidence_ids", [])
        if not a_ev_ids or set(a_ev_ids) - evidence_ids:
            add_issue("ASSET_INVALID_EVIDENCE", "structure", "blocker", f"Asset impact {getattr(asset, 'symbol_or_name', 'unknown')} has invalid evidence references")
            asset_problem = True
        if not getattr(asset, "risk_factors", []) or not getattr(asset, "invalidation_conditions", []):
            add_issue("ASSET_LACKS_CRITERIA", "structure", "blocker", f"Asset impact {getattr(asset, 'symbol_or_name', 'unknown')} lacks risk or invalidation criteria")
            asset_problem = True
    if not asset_problem:
        rubric["asset_relevance"] = weights["asset_relevance"]

    visual_problem = False
    directives = list(getattr(draft, "visual_directives", []) or [])
    
    act_counts = {"Act I": 0, "Act II": 0, "Act III": 0}
    visual_ids = set()
    for directive in directives:
        act = getattr(directive, "act", "")
        act_counts[act] = act_counts.get(act, 0) + 1
        vid = getattr(directive, "visual_id", "")
        if vid in visual_ids:
            add_issue("DUPLICATE_VISUAL_ID", "structure", "blocker", f"Visual directive has duplicate ID: {vid}")
            visual_problem = True
        visual_ids.add(vid)
        
    for act, count in act_counts.items():
        if count != 1:
            add_issue("VISUAL_PER_ACT_VIOLATION", "structure", "blocker", f"Visual directives must provide exactly one directive for {act}, found {count}")
            visual_problem = True

    known_years = _years_in_text(" ".join(
        [getattr(source, "published_at", "") or "" for source in sources] + [getattr(item, "observed_at", "") or "" for item in evidence]
    ))
    
    for directive in directives:
        vid = getattr(directive, "visual_id", "")
        if not getattr(directive, "series_keys", []) or not getattr(directive, "date_range", "") or not getattr(directive, "sources", []):
            add_issue("VISUAL_NOT_REPRODUCIBLE", "structure", "blocker", f"Visual directive {vid} is not reproducible")
            visual_problem = True
        if not getattr(directive, "evidence_ids", []) or set(getattr(directive, "evidence_ids", [])) - evidence_ids:
            add_issue("VISUAL_INVALID_EVIDENCE", "structure", "blocker", f"Visual directive {vid} has invalid evidence references")
            visual_problem = True
            
        data_mode = getattr(directive, "data_mode", "provider_series")
        if data_mode == "provider_series" and any(key.strip().casefold() in ["price", "volume", "sentiment", "trend", "macro", "stock", "default", "index"] for key in getattr(directive, "series_keys", [])):
            add_issue("VISUAL_GENERIC_SERIES", "structure", "blocker", f"Visual directive {vid} uses a non-reproducible generic series name")
            visual_problem = True
            
        if data_mode == "evidence_table" and getattr(directive, "series_keys", []) != ["EVIDENCE_TABLE"]:
            add_issue("VISUAL_INVALID_TABLE", "structure", "blocker", f"Visual directive {vid} has an invalid evidence-table data contract")
            visual_problem = True
            
        directive_years = _years_in_text(getattr(directive, "date_range", ""))
        if known_years and directive_years and not getattr(directive, "historical_comparison", False) and directive_years.isdisjoint(known_years):
            add_issue("VISUAL_DETACHED_PERIOD", "structure", "blocker", f"Visual directive {vid} has a period detached from its evidence")
            visual_problem = True
            
        expected = tuple(getattr(directive, "evidence_ids", []))
        rendered_in_act = [marker for marker in rendered_briefing.visual_markers if marker.act == getattr(directive, "act", "") and marker.id == vid]
        total_rendered = sum(1 for marker in rendered_briefing.visual_markers if marker.id == vid)
        
        if total_rendered == 0:
            add_issue("VISUAL_MISSING", "structure", "blocker", f"Visual marker for {vid} is missing")
            visual_problem = True
        elif total_rendered > 1:
            add_issue("VISUAL_DUPLICATED", "structure", "blocker", f"Visual marker for {vid} is duplicated")
            visual_problem = True
        elif len(rendered_in_act) == 0:
            add_issue("VISUAL_WRONG_ACT", "structure", "blocker", f"Visual marker for {vid} is in the wrong Act")
            visual_problem = True
        elif tuple(rendered_in_act[0].evidence_ids) != expected:
            add_issue("VISUAL_EVIDENCE_MISMATCH", "structure", "blocker", f"Visual marker for {vid} has mismatched evidence IDs")
            visual_problem = True
            
    if not visual_problem:
        rubric["detective_script_visuals"] = weights["detective_script_visuals"]

    prompt_problem = False
    prompts = list(getattr(draft, "notebooklm_prompts", []) or [])
    types = {getattr(prompt, "prompt_type", "") for prompt in prompts}
    if not 5 <= len(prompts) <= 8 or not {"BLIND_SPOT", "SOCRATIC", "FEYNMAN"}.issubset(types):
        add_issue("PROMPT_INCOMPLETE", "structure", "blocker", "NotebookLM prompt pack is incomplete")
        prompt_problem = True
    if any(not getattr(prompt, "question_or_prompt", "").strip() or not getattr(prompt, "expected_output_format", "").strip() for prompt in prompts):
        add_issue("PROMPT_EMPTY", "structure", "blocker", "NotebookLM prompt pack contains an empty prompt or output format")
        prompt_problem = True
    if not prompt_problem:
        rubric["notebooklm_prompts"] = weights["notebooklm_prompts"]

    _check_script_loopbacks(draft, add_issue)


    unique_issues = []
    seen = set()
    for issue in issues:
        key = (issue.code, issue.description)
        if key not in seen:
            seen.add(key)
            unique_issues.append(issue)

    score = min(sum(rubric.values()), score_cap)
    has_blocker = any(i.severity == "blocker" for i in unique_issues)
    if has_blocker:
        status = "fail"
    elif score < 100 or any(i.severity == "warning" for i in unique_issues):
        status = "degraded"
    else:
        status = "pass"
        
    return ResearchQualityReport(
        score=score,
        issues=unique_issues,
        rubric_breakdown=rubric,
        status=status,
        publishable=not has_blocker and score >= 80,
        advisories=advisories,
    )
