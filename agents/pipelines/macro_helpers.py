"""Macro intel pipeline data assembly, enrichment, and sector context helpers.

Architecture Layer: agents/pipelines/
Extracted from manager_agent.py to encapsulate Macro Quant observables,
Strategic Allocator direction enrichment, and Sector Rotation context bindings.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import os
from typing import Any, Optional

from schemas.macro_schemas import (
    AssetAllocationView,
    AssetStance,
    MacroStrategyDirection,
    MarketObservable,
)

log = logging.getLogger(__name__)


def prepare_allocator_quant_payload(
    state: dict[str, Any],
    *,
    eval_date_override: str | None = None,
) -> tuple[dict[str, Any], dict[str, MarketObservable], set[str], str]:
    """Extracts observables, splits by validity, and prepares the quant payload for Strategic Allocator."""
    quant_data = state.get("quant_score", {}) or {}
    observable_registry: dict[str, MarketObservable] = {}
    if isinstance(quant_data, dict):
        for raw in quant_data.get("market_observables", []) or []:
            try:
                obs = raw if isinstance(raw, MarketObservable) else MarketObservable.model_validate(raw)
            except Exception:
                continue
            observable_registry[obs.observable_id] = obs

    valid_observables = []
    unverified_observables = []
    for obs in observable_registry.values():
        item = {
            "observable_id": obs.observable_id,
            "asset_bucket": obs.asset_bucket,
            "region": obs.region,
            "indicator": obs.indicator,
            "value": obs.value,
            "unit": obs.unit,
            "observed_at": obs.observed_at,
            "source_file": obs.source_file,
            "provider": obs.provider,
            "stale_reason": obs.stale_reason,
        }
        if obs.is_valid:
            valid_observables.append(item)
        else:
            unverified_observables.append(item)

    allocator_quant_data = dict(quant_data) if isinstance(quant_data, dict) else {}
    allocator_quant_data["sector_rotation_context"] = state.get("sector_rotation_context")
    allocator_quant_data["sector_snapshot_id"] = state.get("sector_rotation_snapshot_id")
    allocator_quant_data["sector_context_status"] = state.get("sector_context_status")
    allocator_quant_data["valid_observables"] = valid_observables
    allocator_quant_data["invalid_observables"] = unverified_observables
    allocator_quant_data["regional_data_gaps"] = allocator_quant_data.get("data_gaps", [])
    allocator_quant_data["market_observables_by_validity"] = {
        "VALID INSTITUTIONAL HARD DATA OBSERVABLES (USE FOR HIGH/MEDIUM CONFIDENCE)": valid_observables,
        "UNVERIFIED PROXIES & STALE INDICATORS (DO NOT USE FOR CONFIDENCE / LOW CONFIDENCE ONLY)": unverified_observables,
    }
    evaluated_source_files = {
        obs.source_file for obs in observable_registry.values() if obs.source_file
    }
    evaluated_date = (
        eval_date_override
        or os.getenv("EVAL_DATE")
        or str(allocator_quant_data.get("evaluated_at", ""))[:10]
    )
    if not evaluated_date or len(evaluated_date) != 10:
        evaluated_date = datetime.now().strftime("%Y-%m-%d")
    evaluated_source_files.add(f"Macro_Baseline_{evaluated_date}.md")
    allocator_quant_data["evaluated_source_files"] = sorted(evaluated_source_files)

    return allocator_quant_data, observable_registry, evaluated_source_files, evaluated_date


def enrich_macro_direction_post_allocation(
    direction: Any,
    allocator_quant_data: dict[str, Any],
    observable_registry: dict[str, MarketObservable],
) -> Any:
    """Enriches MacroStrategyDirection with Thai stance, guaranteed SET equities, and source citations."""
    th_quant = (
        allocator_quant_data.get("regions", {}).get("Thailand", {}).get("market_stance") or {}
    )
    th_llm = getattr(direction, "thailand_market_stance", None) or {}
    merged_stance = dict(th_quant)

    ai_rationale = None
    if isinstance(th_llm, dict):
        ai_rationale = th_llm.get("rationale")
        if th_llm.get("observable_refs"):
            merged_stance["observable_refs"] = th_llm["observable_refs"]

    if not ai_rationale:
        from tools.macro.terminal_observables import synthesize_thai_market_stance_narrative

        thai_assets_in_dir = [
            a
            for a in getattr(direction, "asset_allocation", []) or []
            if getattr(a, "region", "") == "Thailand" or "THB" in getattr(a, "asset_class", "")
        ]
        ai_rationale = synthesize_thai_market_stance_narrative(merged_stance, thai_assets_in_dir)

    merged_stance["rationale"] = ai_rationale
    if not merged_stance.get("observable_refs"):
        merged_stance["observable_refs"] = [
            oid
            for oid in [
                "obs_set_flow_foreign",
                "obs_set_advance_decline_ratio",
                "obs_set_valuation_pe",
                "obs_gta_gold_bar_sell",
                "obs_diff_us_th_policy_rate_bis",
            ]
            if oid in observable_registry
        ]
    direction.thailand_market_stance = merged_stance

    # Guarantee SET Equities (Thailand) in asset allocation if Thai quant data exists
    has_thai_equity = any(
        getattr(a, "region", None) == "Thailand" and getattr(a, "asset_bucket", None) == "equities"
        for a in getattr(direction, "asset_allocation", []) or []
    )
    if not has_thai_equity and th_quant:
        flow_mb = th_quant.get("investor_flow", {}).get("foreign_net_mb")
        pe = th_quant.get("valuation", {}).get("pe_ratio")
        ad = th_quant.get("market_breadth", {}).get("advance_decline_ratio")

        has_eligible_data = (flow_mb is not None) or (pe is not None) or (ad is not None)
        if has_eligible_data:
            is_bearish = (
                (flow_mb is not None and flow_mb < 0)
                or (pe is not None and pe > 17)
                or (ad is not None and ad < 0.8)
            )
            thai_stance = AssetStance.UNDERWEIGHT if is_bearish else AssetStance.NEUTRAL
            delta_str = (
                "-2% vs benchmark" if thai_stance == AssetStance.UNDERWEIGHT else "0% vs benchmark"
            )

            support_items = []
            obs_refs = []
            rationale_parts = []

            if flow_mb is not None and "obs_set_flow_foreign" in observable_registry:
                obs_refs.append("obs_set_flow_foreign")
                support_items.append(f"SET Foreign Investor Net Flow {flow_mb:,.2f} THB Mil")
                if flow_mb < 0:
                    rationale_parts.append(f"กระแสเงินทุนต่างชาติขายสุทธิ ({flow_mb:,.1f} ลบ.)")
                elif flow_mb > 0:
                    rationale_parts.append(f"กระแสเงินทุนต่างชาติซื้อสุทธิ (+{flow_mb:,.1f} ลบ.)")
                else:
                    rationale_parts.append("กระแสเงินทุนต่างชาติทรงตัว (0.0 ลบ.)")

            if ad is not None and "obs_set_advance_decline_ratio" in observable_registry:
                obs_refs.append("obs_set_advance_decline_ratio")
                support_items.append(f"SET Market Breadth A/D {ad:.2f}x")

            if pe is not None and "obs_set_valuation_pe" in observable_registry:
                obs_refs.append("obs_set_valuation_pe")
                support_items.append(f"SET Index P/E {pe:.2f}")
                rationale_parts.append(f"ระดับ Valuation P/E อยู่ที่ {pe:.1f}x")

            if rationale_parts:
                constructed_rationale = (
                    f"ตลาดหุ้นไทยประเมินจากหลักฐานตลาดทุน: {' และ '.join(rationale_parts)}"
                )
            else:
                constructed_rationale = (
                    "ประเมินสภาวะตลาดหุ้นไทยจากข้อมูลสภาพคล่องและโครงสร้างตลาดทุน"
                )

            thai_eq_view = AssetAllocationView(
                asset_class="SET Equities (Thailand)",
                asset_bucket="equities",
                region="Thailand",
                stance=thai_stance,
                rationale=merged_stance.get("rationale") or constructed_rationale,
                confidence="medium" if len(obs_refs) >= 2 else "low",
                supporting_data=support_items,
                observable_refs=obs_refs,
                source_refs=["Terminal_V2_Settrade"],
                why_not_high="ความมั่นใจอยู่ในระดับปานกลางเนื่องจากอยู่ระหว่างรอการเชื่อมต่อ API มหภาคทางการ (NESDC/MOC)",
                allocation_delta=delta_str,
                benchmark_ref="SET Index",
                time_horizon="1-3 months",
            )
            if not hasattr(direction, "asset_allocation") or direction.asset_allocation is None:
                direction.asset_allocation = []
            direction.asset_allocation.append(thai_eq_view)

    # Ensure Cash region is Global
    for a in getattr(direction, "asset_allocation", []) or []:
        if getattr(a, "asset_bucket", None) == "cash" and getattr(a, "region", None) == "US":
            a.region = "Global"

    # Revalidate with registry to ensure all guardrails, sources, and refs are pristine
    if hasattr(direction, "revalidate_with_registry"):
        direction = direction.revalidate_with_registry(observable_registry)

    # Preserve cited sources in direction.source_files without blindly dumping all evaluated_sources
    if not getattr(direction, "source_files", None):
        direction.source_files = [
            obs.source_file for obs in observable_registry.values() if obs.is_valid and obs.source_file
        ]

    return direction


def canonicalize_macro_report_direction(
    direction: Any,
    sidecar_result: Any,
    sector_analysis_data: Any,
) -> tuple[MacroStrategyDirection, dict[str, Any], Any]:
    """Safely extracts canonical direction model and canonical report dict from sidecar result."""
    canonical_report = (
        sidecar_result[1]
        if isinstance(sidecar_result, tuple) and len(sidecar_result) > 1 and isinstance(sidecar_result[1], dict)
        else {
            **(direction.model_dump(mode="json") if hasattr(direction, "model_dump") else {}),
            "sector_analysis": sector_analysis_data,
        }
    )

    direction_fields = MacroStrategyDirection.model_fields
    canonical_direction = {
        key: canonical_report[key] for key in direction_fields if key in canonical_report
    }
    validated_direction = (
        MacroStrategyDirection.model_validate(canonical_direction)
        if canonical_direction
        else direction
    )
    resolved_sector_analysis = canonical_report.get("sector_analysis")
    return validated_direction, canonical_report, resolved_sector_analysis


def resolve_sector_context_for_quant(
    state: dict[str, Any],
    config: Any,
    raw_instruction: str,
) -> tuple[Any, dict[str, Any] | None, dict[str, Any], str, str | None, str]:
    """Pins SectorRotationSnapshot and enriches the instruction for macro_quant."""
    config_meta = (config or {}).get("metadata", {}) if isinstance(config, dict) else {}
    macro_run_id = (
        state.get("macro_run_id") or config_meta.get("job_id") or state.get("turn_id")
    )
    task_run_id = (
        state.get("macro_task_run_id") or f"{macro_run_id}:{state.get('turn_id', 'task')}:1"
    )

    snapshot = None
    sector_context = None
    sector_context_status: dict[str, Any] = {"status": "unavailable", "reason": "snapshot_prepare_failed"}
    try:
        from tools.macro.sector_rotation.bootstrap import get_sector_rotation_service
        from tools.macro.sector_rotation.domain.claims import compact_ai_context

        rotation_service = get_sector_rotation_service()
        snapshot, sector_context_status = rotation_service.pin_for_run(
            str(task_run_id),
            job_id=str(macro_run_id) if macro_run_id else None,
            task_id=str(task_run_id),
            preferred_snapshot_id=state.get("sector_rotation_snapshot_id"),
        )
        if snapshot:
            sector_context = compact_ai_context(snapshot)
            has_sector_data = any(row.status != "unavailable" for row in snapshot.rows)
            complete = snapshot.benchmark_status == "available" and all(
                row.status == "available" for row in snapshot.rows
            )
            sector_context_status = {
                **sector_context_status,
                "status": "available" if complete else "limited" if has_sector_data else "unavailable",
                "reason": None if complete else "partial_sector_coverage" if has_sector_data else "sector_history_unavailable",
            }
    except Exception as exc:
        log.warning("Sector snapshot unavailable; continuing Macro analysis without sector facts: %s", exc)
        snapshot = None
        sector_context = None
        sector_context_status = {
            "status": "unavailable",
            "reason": f"snapshot_prepare_failed:{type(exc).__name__}",
        }

    if sector_context_status["status"] == "unavailable":
        sector_context = None

    quant_instruction = raw_instruction
    if sector_context:
        quant_instruction += (
            "\n\nDeterministic sector rotation evidence for this macro run (cite its values as market evidence; "
            "do not count ETFs as GDP/CPI hard-data or infer fund flows):\n"
            + json.dumps(sector_context, ensure_ascii=False)
        )
    else:
        quant_instruction += (
            "\n\nSector rotation context status: unavailable ("
            + str(sector_context_status.get("reason") or "no_valid_snapshot")
            + "). Continue the macro analysis using the other evidence and do not assert sector facts."
        )

    return (
        snapshot,
        sector_context,
        sector_context_status,
        quant_instruction,
        str(macro_run_id) if macro_run_id else None,
        str(task_run_id),
    )


def resolve_sector_context_for_economist(
    state: dict[str, Any],
) -> tuple[Any, dict[str, Any] | None, dict[str, Any]]:
    """Pins SectorRotationSnapshot for macro_economist."""
    sector_snapshot = None
    sector_context = None
    sector_context_status: dict[str, Any] = {"status": "unavailable", "reason": "run_binding_missing"}
    task_run_id = state.get("macro_task_run_id")
    if task_run_id:
        from tools.macro.sector_rotation.bootstrap import get_sector_rotation_service
        from tools.macro.sector_rotation.domain.claims import compact_ai_context

        try:
            sector_snapshot, binding = get_sector_rotation_service().pin_for_run(
                str(task_run_id),
                job_id=str(state.get("macro_run_id")) if state.get("macro_run_id") else None,
                task_id=str(task_run_id),
            )
            if sector_snapshot is not None:
                sector_context = compact_ai_context(sector_snapshot)
                has_sector_data = any(row.status != "unavailable" for row in sector_snapshot.rows)
                complete = sector_snapshot.benchmark_status == "available" and all(
                    row.status == "available" for row in sector_snapshot.rows
                )
                sector_context_status = {
                    "status": "available" if complete else "limited" if has_sector_data else "unavailable",
                    "reason": None if complete else "partial_sector_coverage" if has_sector_data else "sector_history_unavailable",
                    "snapshot_id": sector_snapshot.snapshot_id,
                }
            else:
                sector_context_status = {
                    "status": "unavailable",
                    "reason": binding.get("unavailable_reason") or binding.get("reason") or "pinned_snapshot_unavailable",
                }
        except Exception as exc:
            log.warning("Pinned sector evidence unavailable to Macro Economist: %s", exc)
            sector_snapshot = None
            sector_context_status = {
                "status": "unavailable",
                "reason": f"pinned_snapshot_unavailable:{type(exc).__name__}",
            }

    return sector_snapshot, sector_context, sector_context_status


def build_economist_fallback_dict(error: Exception | str) -> dict[str, Any]:
    """Generates a fallback dict for NarrativeContext on economist parsing error."""
    return {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "dominant_themes": [],
        "market_sentiment": "neutral",
        "tail_risks": [],
        "policy_signals": [],
        "key_narratives_by_region": {},
        "sources_summary": f"Data fetch failed: {str(error)}",
    }
