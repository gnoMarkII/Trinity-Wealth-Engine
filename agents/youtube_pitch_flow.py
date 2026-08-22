"""StateGraph Workflow สำหรับ YouTube Content Pitching & Research-Grade Briefing Book"""
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional, TypedDict
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from langsmith import traceable

from core.logger import get_logger
from schemas.youtube_pitch_schemas import YouTubeContentPitchItem

from tools.content.youtube_pitcher import (
    fetch_news_for_pitching,
    generate_youtube_pitches,
    parse_date_filters_from_instruction,
    synthesize_notebooklm_source,
)
from tools.content.briefing_artifacts import save_briefing_artifact
from tools.archivist.core import VAULT_PATH
from tools.content.provenance_enrichment import (
    assess_pitch_source_readiness,
    prepare_verified_candidate_pool,
)

logger = get_logger(__name__)


class YouTubePitchState(TypedDict, total=False):
    messages: list
    instruction: str
    from_date: Optional[str]
    to_date: Optional[str]
    lookback_days: int
    news_candidates: list[dict]
    macro_baselines: str
    pitches: list[dict]
    approved_pitch_ids: list[str]
    unverified_draft_selections: list[dict]
    result_summary: str
    synthesis_status: Literal["done", "done_with_warnings", "done_with_errors", "error"]
    synthesis_failures: list[str]
    synthesis_warnings: list[str]
    synthesized_pitch_ids: list[str]
    pitch_generation_attempt: int
    approval_revision: int
    source_refresh_attempts: int
    force_provenance_refresh: bool
    provenance_summary: dict[str, Any]
    approval_block_reason: Optional[str]



def fetch_topics_node(state: YouTubePitchState) -> dict:
    instruction = state.get("instruction", "")
    filters = parse_date_filters_from_instruction(instruction)
    from_d = filters.get("from_date") or state.get("from_date")
    to_d = filters.get("to_date") or state.get("to_date")
    lookback = filters.get("lookback_days", state.get("lookback_days", 7))

    candidates, macro_str, is_fallback = fetch_news_for_pitching(
        from_date=from_d,
        to_date=to_d,
        lookback_days=lookback,
    )

    messages = []
    if is_fallback:
        messages.append(
            AIMessage(
                content="⚠️ คำเตือน: ช่วงวันที่ระบุย้อนหลังเกิน 7 วัน (News Funnel Store ตัดข้อมูลข่าวดิบทุก 7 วัน) — ระบบได้ดึงข้อมูลจากโน้ตที่สังเคราะห์ไว้แล้วในฐานความรู้ (Layer 2) ร่วมด้วย",
                name="fetch_topics",
            )
        )
    messages.append(
        AIMessage(
            content=f"พบข่าวและบทวิเคราะห์ที่เกี่ยวข้อง {len(candidates)} รายการ (ช่วงวันที่: {from_d or 'ย้อนหลัง'} ถึง {to_d or f'{lookback} วันล่าสุด'})",
            name="fetch_topics",
        )
    )

    return {
        "from_date": from_d,
        "to_date": to_d,
        "lookback_days": lookback,
        "news_candidates": candidates,
        "macro_baselines": macro_str,
        "messages": messages,
    }


from langchain_core.runnables import RunnableConfig

def generate_pitches_node(state: YouTubePitchState, config: RunnableConfig) -> dict:
    candidates = state.get("news_candidates", [])
    instruction = state.get("instruction", "")
    from_d = state.get("from_date")
    to_d = state.get("to_date")
    generation_attempt = int(state.get("pitch_generation_attempt", 0) or 0)
    approval_revision = generation_attempt + 1
    force_refresh = bool(state.get("force_provenance_refresh", False))

    thread_id = config.get("configurable", {}).get("thread_id", "")
    job_id = config.get("metadata", {}).get("job_id", thread_id)

    def _annotate_readiness(batch, pool: list[dict], *, force: bool) -> tuple[dict[str, int], list[dict]]:
        from tools.content.provenance_enrichment import evaluate_unverified_draft_eligibility, generate_eligibility_token
        refreshed_by_id = {str(item.get("event_id")): dict(item) for item in pool if item.get("event_id")}
        counts = {"ready": 0, "blocked": 0, "needs_refresh": 0, "draft_eligible": 0}
        for pitch in batch.pitches:
            readiness, issues, issue_codes, refreshed_sources = assess_pitch_source_readiness(
                pitch,
                pool,
                refresh=True,
                force_refresh=force,
            )
            pitch.source_readiness = readiness  # type: ignore[assignment]
            pitch.source_readiness_issues = issues

            is_draft_eligible = False
            if readiness != "ready":
                if readiness == "needs_refresh" and not force:
                    pass  # Must force refresh at least once
                elif "SOURCE_NOT_FOUND" not in issue_codes and "MOCK_SOURCE" not in issue_codes and "MISSING_SOURCE_URL" not in issue_codes:
                    is_draft_eligible = evaluate_unverified_draft_eligibility(issue_codes)

            pitch_id = getattr(pitch, "pitch_id", "") or str(id(pitch))

            if is_draft_eligible:
                counts["draft_eligible"] += 1
                pitch.unverified_draft_eligible = True
                pitch.unverified_draft_issue_codes = issue_codes
                pitch.unverified_draft_eligibility_token = generate_eligibility_token(
                    job_id=job_id,
                    thread_id=thread_id,
                    pitch_id=pitch_id,
                    revision=approval_revision,
                    issue_codes=issue_codes,
                )

            counts[readiness] = counts.get(readiness, 0) + 1
            for source in refreshed_sources:
                event_id = source.get("event_id")
                if event_id:
                    refreshed_by_id[str(event_id)] = source
        refreshed = [refreshed_by_id.get(str(item.get("event_id")), item) for item in pool]
        return counts, refreshed

    # A manual retry starts from the verified pool.  The normal path first
    # keeps the broad editorial pool, then performs the same safe recovery
    # automatically if every draft is unselectable.
    provenance_summary: dict[str, Any] = dict(state.get("provenance_summary") or {})
    if force_refresh:
        refreshed_candidates, verified_pool, provenance_summary = prepare_verified_candidate_pool(
            candidates,
            force_refresh=True,
        )
        candidate_pool = verified_pool
        batch = generate_youtube_pitches(
            candidates=candidate_pool,
            max_pitches=4,
            instruction=instruction,
            from_date=from_d,
            to_date=to_d,
        ) if candidate_pool else None
        readiness_counts, refreshed_candidates = (
            _annotate_readiness(batch, refreshed_candidates, force=True)
            if batch is not None else ({"ready": 0, "blocked": 0, "needs_refresh": 0}, refreshed_candidates)
        )
    else:
        batch = generate_youtube_pitches(
            candidates=candidates,
            max_pitches=4,
            instruction=instruction,
            from_date=from_d,
            to_date=to_d,
        )
        readiness_counts, refreshed_candidates = _annotate_readiness(batch, candidates, force=False)

        if readiness_counts["ready"] == 0:
            refreshed_candidates, verified_pool, provenance_summary = prepare_verified_candidate_pool(candidates)
            if verified_pool:
                batch = generate_youtube_pitches(
                    candidates=verified_pool,
                    max_pitches=4,
                    instruction=instruction,
                    from_date=from_d,
                    to_date=to_d,
                )
                readiness_counts, refreshed_candidates = _annotate_readiness(batch, refreshed_candidates, force=False)

    pitches_dict_list = [p.model_dump() for p in batch.pitches] if batch is not None else []
    ready_count = readiness_counts.get("ready", 0)
    draft_eligible_count = readiness_counts.get("draft_eligible", 0)
    refreshable_count = readiness_counts.get("needs_refresh", 0)

    no_selectable_pitch = (ready_count + draft_eligible_count + refreshable_count) == 0
    msg = (
        f"สร้างไอเดียคลิป YouTube สำเร็จ {len(pitches_dict_list)} หัวข้อ "
        f"(พร้อมสร้าง {ready_count}, draft {draft_eligible_count}, ต้องแก้ provenance {readiness_counts.get('blocked', 0) + refreshable_count})"
    )
    if no_selectable_pitch:
        status_counts = provenance_summary.get("status_counts", {})
        msg += (
            " — ไม่มีหัวข้อที่เลือกได้หลังตรวจ provenance; "
            f"ตรวจแหล่งข้อมูล {provenance_summary.get('refreshed_candidates', 0)} รายการ, "
            f"ยืนยันได้ {provenance_summary.get('verified_candidates', 0)} รายการ, สถานะ {status_counts}"
        )
    return {
        "pitches": pitches_dict_list,
        "news_candidates": refreshed_candidates,
        "pitch_generation_attempt": approval_revision,
        "approval_revision": approval_revision,
        "force_provenance_refresh": False,
        "provenance_summary": provenance_summary,
        "approval_block_reason": "no_selectable_pitch" if no_selectable_pitch else None,
        "messages": [AIMessage(content=msg, name="generate_pitches")],
    }


def gate_node(state: YouTubePitchState, config: RunnableConfig) -> Command[Literal["synthesize_notebooklm", "generate_pitches"]]:
    pitches = state.get("pitches", [])
    instruction = state.get("instruction", "")
    approval_revision = int(state.get("approval_revision", 0) or 0)
    thread_id = config.get("configurable", {}).get("thread_id", "")
    job_id = config.get("metadata", {}).get("job_id", thread_id)

    selection = interrupt({
        "type": "youtube_pitch_approval",
        "pitches": pitches,
        "instruction": instruction,
        "approval_revision": approval_revision,
        "source_refresh_attempts": int(state.get("source_refresh_attempts", 0) or 0),
    })

    action = str(selection.get("action") or "approve") if isinstance(selection, dict) else "approve"
    if action == "refresh_sources":
        refresh_attempts = int(state.get("source_refresh_attempts", 0) or 0)
        if refresh_attempts >= 1:
            raise ValueError("Provenance refresh was already attempted for this job; start a new search window to find different sources")
        return Command(
            goto="generate_pitches",
            update={
                "source_refresh_attempts": refresh_attempts + 1,
                "force_provenance_refresh": True,
                "approval_block_reason": None,
            },
        )

    requested_ids = [str(value) for value in ((selection.get("approved_pitch_ids") or []) if isinstance(selection, dict) else [])]
    requested_drafts = (selection.get("unverified_draft_selections") or []) if isinstance(selection, dict) else []

    pitch_by_id = {str(pitch.get("pitch_id")): pitch for pitch in pitches if isinstance(pitch, dict)}
    readiness_by_id = {
        k: str(v.get("source_readiness") or "unknown") for k, v in pitch_by_id.items()
    }

    # 0. Reject Unknown IDs
    for pitch_id in requested_ids:
        if pitch_id not in pitch_by_id:
            raise ValueError(f"Unknown approved pitch ID: {pitch_id}")

    # 1. Publishable approvals must be explicitly ready.  Unknown is retained
    # only for old checkpoints and must never become an implicit bypass.
    for pitch_id in requested_ids:
        if readiness_by_id.get(pitch_id) != "ready":
            raise ValueError(f"Pitch {pitch_id} is not source-ready and cannot be in approved_pitch_ids")

    # 2. Reject if overlap between groups or duplicates
    requested_ids_set = set(requested_ids)
    draft_ids_list = [str(s.get("pitch_id")) for s in requested_drafts if isinstance(s, dict)]
    draft_ids_set = set(draft_ids_list)

    if len(draft_ids_list) != len(draft_ids_set):
        raise ValueError("Duplicate draft pitch IDs in request")

    if requested_ids_set.intersection(draft_ids_set):
        raise ValueError("A pitch cannot be simultaneously requested as both publishable and an unverified draft")

    # 3. Validate Draft Selections Authoritatively
    from tools.content.provenance_enrichment import verify_eligibility_token

    for selection_dict in requested_drafts:
        pitch_id = str(selection_dict.get("pitch_id"))
        ack = selection_dict.get("ack", {})
        token = ack.get("eligibility_token", "")

        if not ack.get("acknowledged"):
            raise ValueError(f"Pitch {pitch_id} requires explicit acknowledgment for Unverified Draft")
        if ack.get("policy_version") != "unverified-draft-v1":
            raise ValueError(f"Pitch {pitch_id} uses unsupported policy version")

        pitch_data = pitch_by_id.get(pitch_id)
        if not pitch_data:
            raise ValueError(f"Pitch {pitch_id} not found in current checkpoint")

        expected_token = pitch_data.get("unverified_draft_eligibility_token")
        issue_codes = list(pitch_data.get("unverified_draft_issue_codes") or [])
        if not expected_token or not pitch_data.get("unverified_draft_eligible"):
            raise ValueError(f"Pitch {pitch_id} lacks eligibility metadata in current checkpoint")

        if token != expected_token:
            raise ValueError(f"Eligibility token for {pitch_id} does not match current checkpoint (possibly stale/replayed)")

        if approval_revision <= 0 or not verify_eligibility_token(
            token,
            expected_job_id=job_id,
            expected_thread_id=thread_id,
            expected_pitch_id=pitch_id,
            expected_revision=approval_revision,
            expected_issue_codes=issue_codes,
        ):
            raise ValueError(f"Eligibility token for {pitch_id} is malformed or invalid")

    requested_styles = (selection.get("pitch_presentation_styles") or {}) if isinstance(selection, dict) else {}
    for pitch in pitches:
        if isinstance(pitch, dict) and str(pitch.get("pitch_id")) in requested_styles:
            pitch["presentation_style"] = requested_styles[str(pitch["pitch_id"])]

    return Command(
        goto="synthesize_notebooklm",
        update={
            "approved_pitch_ids": requested_ids,
            "unverified_draft_selections": requested_drafts,
            "pitches": pitches,
        },
    )


def provenance_unavailable_node(state: YouTubePitchState) -> dict:
    """Fail before Approval when no pitch has sufficient verified provenance."""
    summary = state.get("provenance_summary") or {}
    counts = summary.get("status_counts") or {}
    raise ValueError(
        "No selectable YouTube Pitch: verified provenance was unavailable. "
        f"Refreshed={summary.get('refreshed_candidates', 0)}, "
        f"verified={summary.get('verified_candidates', 0)}, "
        f"independent_publishers={summary.get('independent_publishers', 0)}, "
        f"status_counts={counts}. Retry source verification or dispatch a new search window."
    )


def _route_after_pitch_generation(state: YouTubePitchState) -> Literal["gate", "provenance_unavailable"]:
    return "provenance_unavailable" if state.get("approval_block_reason") else "gate"


def synthesize_notebooklm_node(state: YouTubePitchState, config: RunnableConfig) -> dict:
    from tools.content.briefing_artifacts import save_briefing_artifact
    import hashlib

    approved_ids = set(state.get("approved_pitch_ids") or [])
    draft_selections = state.get("unverified_draft_selections") or []
    draft_ids = {str(d.get("pitch_id")) for d in draft_selections if isinstance(d, dict)}

    pitches = state.get("pitches") or []
    candidates = state.get("news_candidates") or []
    macro_str = state.get("macro_baselines", "")

    job_id = config.get("metadata", {}).get("job_id", "")
    thread_id = config.get("configurable", {}).get("thread_id", "")

    # Zero-file protection: ถ้าไม่อนุมัติเลยแม้แต่รายการเดียว
    if not approved_ids and not draft_ids:
        line = "ไม่มีไอเดียคลิปที่ถูกเลือก อนุมัติ 0 รายการ (ไม่สร้างไฟล์ Briefing Book)"
        return {
            "result_summary": line,
            "messages": [AIMessage(content=line, name="synthesize_notebooklm")],
        }

    messages = []
    summary_lines = []
    failures: list[str] = []
    warnings: list[str] = []
    synthesized_pitch_ids: list[str] = []
    succeeded = 0
    today_str = datetime.now().strftime("%Y-%m-%d")

    for p_dict in pitches:
        if not isinstance(p_dict, dict):
            continue
        p_id = str(p_dict.get("pitch_id", ""))
        if p_id not in approved_ids and p_id not in draft_ids:
            continue

        main_title = p_id
        try:
            pitch_item = YouTubeContentPitchItem.model_validate(p_dict)
            main_title = pitch_item.working_titles[0] if pitch_item.working_titles else "untitled"

            is_draft = p_id in draft_ids
            output_mode = "unverified_draft" if is_draft else "publishable"

            override_audit_dict = None
            if is_draft:
                draft_ack = next((d.get("ack", {}) for d in draft_selections if str(d.get("pitch_id")) == p_id), {})
                token = draft_ack.get("eligibility_token", "")
                override_audit_dict = {
                    "job_id": job_id,
                    "thread_id": thread_id,
                    "pitch_id": p_id,
                    "policy_version": draft_ack.get("policy_version", "unknown"),
                    "reason": str(draft_ack.get("reason") or "Unverified Draft fallback requested by user"),
                    "server_timestamp": datetime.now().isoformat(),
                    "token_hash": hashlib.sha256(token.encode("utf-8")).hexdigest(),
                    "source_readiness_snapshot": pitch_item.unverified_draft_issue_codes or [],
                }

            # สังเคราะห์ 7 Sections
            synthesis_result = synthesize_notebooklm_source(
                pitch_item, candidates, macro_str, output_mode=output_mode, override_audit=override_audit_dict
            )

            # บันทึกลง Vault
            saved_artifact = save_briefing_artifact(
                synthesis=synthesis_result,
                title=main_title,
                vault_root=Path(VAULT_PATH),
                date_str=today_str,
            )

            synthesized_pitch_ids.append(p_id)

            if is_draft:
                line = f"✓ บันทึก Draft สำเร็จ: {saved_artifact.path} (หัวข้อ: {main_title})"
            else:
                line = f"✓ บันทึก Briefing Book สำเร็จ: {saved_artifact.path} (หัวข้อ: {main_title})"

            summary_lines.append(line)
            messages.append(AIMessage(content=line, name="synthesize_notebooklm"))
            succeeded += 1
        except ValueError as e:
            line = f"✗ ล้มเหลวในการสังเคราะห์ไอเดีย '{main_title}' ({p_id}): {e}"
            summary_lines.append(line)
            messages.append(AIMessage(content=line, name="synthesize_notebooklm"))
            failures.append(line)
        except Exception as e:
            line = f"✗ ระบบขัดข้องขณะสังเคราะห์ไอเดีย '{main_title}' ({p_id}): {e.__class__.__name__} - {e}"
            summary_lines.append(line)
            messages.append(AIMessage(content=line, name="synthesize_notebooklm"))
            failures.append(line)

    if failures and not succeeded:
        # Do not swallow an all-failed batch: JobQueue must expose a terminal
        # ``error`` rather than incorrectly labelling AG-19-style failures done.
        raise RuntimeError("YouTube Pitch synthesis failed for every approved pitch:\n" + "\n".join(failures))

    synthesis_status = "done"
    if failures:
        synthesis_status = "done_with_errors"
    elif draft_ids:
        synthesis_status = "done_with_warnings"

    return {
        "result_summary": "\n".join(summary_lines),
        "messages": messages,
        "synthesis_status": synthesis_status,
        "synthesis_failures": failures,
        "synthesis_warnings": warnings,
        "synthesized_pitch_ids": synthesized_pitch_ids,
    }


@traceable(run_type="chain")
def persist_parking_lot_node(state: YouTubePitchState, config: RunnableConfig = None) -> dict:
    """Best-effort persistence of parking lot ideas into Vault outbox and SQLite kanban backlog."""
    from tools.content.parking_lot_outbox import write_parking_lot_outbox_atomic, mark_outbox_synced_atomic
    from api.state_db import create_parking_lot_cards_atomic

    pitches = state.get("pitches", [])
    synthesized_ids = set(state.get("synthesized_pitch_ids", []))
    current_status = state.get("synthesis_status", "done")
    warnings = list(state.get("synthesis_warnings", []))
    messages = []

    job_id = "unknown"
    if config and isinstance(config, dict) and "configurable" in config:
        job_id = config["configurable"].get("job_id", "unknown")

    for p_dict in pitches:
        if not isinstance(p_dict, dict):
            continue
        p_id = str(p_dict.get("pitch_id", ""))
        if p_id not in synthesized_ids:
            continue

        ideas = p_dict.get("parking_lot_ideas") or []
        if not ideas:
            continue

        outbox_file = None
        try:
            outbox_file = write_parking_lot_outbox_atomic(
                vault_root=Path(VAULT_PATH),
                job_id=job_id,
                pitch_id=p_id,
                ideas=ideas,
            )
        except Exception as outbox_err:
            logger.error("Failed to write parking lot outbox for pitch %s: %s", p_id, outbox_err)
            idea_hashes = [hashlib.sha256(str(i).strip().casefold().encode("utf-8")).hexdigest()[:8] for i in ideas]
            logger.warning("LAST-RESORT DIAGNOSTIC: Job %s Pitch %s ideas unwritten: %s", job_id, p_id, idea_hashes)

        try:
            created = create_parking_lot_cards_atomic(ideas=ideas, source_pitch_id=p_id)
            if outbox_file:
                mark_outbox_synced_atomic(outbox_file)
            if created > 0:
                msg = f"✓ บันทึกไอเดีย Parking Lot ลง Backlog สำเร็จ: {created} รายการ (จาก Pitch {p_id})"
                messages.append(AIMessage(content=msg, name="persist_parking_lot"))
        except Exception as db_err:
            warn_msg = f"⚠️ ไม่สามารถบันทึกไอเดีย Parking Lot ({len(ideas)} รายการ) ลง SQLite ได้ (เก็บไว้ใน Outbox Vault แล้ว): {db_err}"
            logger.warning("Parking lot SQLite persist warning for pitch %s: %s", p_id, db_err)
            warnings.append(warn_msg)
            messages.append(AIMessage(content=warn_msg, name="persist_parking_lot"))
            if current_status != "done_with_errors":
                current_status = "done_with_warnings"

    return {
        "messages": messages,
        "synthesis_status": current_status,
        "synthesis_warnings": warnings,
    }


def build_youtube_pitch_graph(checkpointer=None):
    builder = StateGraph(YouTubePitchState)
    builder.add_node("fetch_topics", fetch_topics_node)
    builder.add_node("generate_pitches", generate_pitches_node)
    builder.add_node("gate", gate_node)
    builder.add_node("provenance_unavailable", provenance_unavailable_node)
    builder.add_node("synthesize_notebooklm", synthesize_notebooklm_node)
    builder.add_node("persist_parking_lot", persist_parking_lot_node)

    builder.add_edge(START, "fetch_topics")
    builder.add_edge("fetch_topics", "generate_pitches")
    builder.add_conditional_edges(
        "generate_pitches",
        _route_after_pitch_generation,
        {"gate": "gate", "provenance_unavailable": "provenance_unavailable"},
    )
    builder.add_edge("provenance_unavailable", END)
    builder.add_edge("synthesize_notebooklm", "persist_parking_lot")
    builder.add_edge("persist_parking_lot", END)

    return builder.compile(checkpointer=checkpointer)

