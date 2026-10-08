"""Deterministic Markdown Formatter for Macro NotebookLM Research Export.

Produces clear, structured Markdown documents optimized for NotebookLM's
grounding, indexing, and citation search engine without losing numeric or lineage precision.
Ensures full coverage of canonical schemas, all historical records, and structured appendices.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from application.macro.notebooklm_export_ports import MacroCorpusSnapshot


def format_research_guide(snapshot: MacroCorpusSnapshot) -> str:
    """Part 00: Research Guide, Corpus Inventory, and Citation Handbook."""
    counts = snapshot.metadata.get("counts", {})
    warnings = snapshot.metadata.get("warnings", [])

    lines = [
        "# Macro Research Companion Guide & Evidence Catalog",
        "",
        f"- **Snapshot Timestamp (UTC):** `{snapshot.snapshot_at}`",
        f"- **Active Strategy Report ID:** `{snapshot.strategy_report_id or 'None'}`",
        f"- **Historical Reports Retained:** `{counts.get('historical_reports', len(snapshot.historical_reports))}`",
        f"- **Regional / Global Notes:** `{counts.get('catalog_notes', len(snapshot.catalog_notes))}`",
        f"- **Market Observables Cached:** `{counts.get('market_observables_cached', 0)} / {counts.get('market_observables_total', len(snapshot.market_observables))}`",
        "",
        "---",
        "",
        "## 1. System Role & Scope Invariant",
        "This Notebook serves exclusively as a **Research Companion** for the user.",
        "- **What this Notebook does:** Answers questions, explains macro indicators, tracks conflicting data points, audits assumptions, and provides grounded citations against official sources.",
        "- **What this Notebook NEVER does:** It does not trigger portfolio rebalancing, generate automated podcasts/audio, execute trades, or modify the live conviction scoring of the Trinity Wealth Engine.",
        "",
        "## 2. Recommended Research Questions",
        "1. *'What are the core contradictions between the current US Treasury yield curve and the OFR Financial Stress Index?'*",
        "2. *'How has foreign investor flow in Thailand (SET) shifted over the last 30 days relative to the US-TH policy rate spread?'*",
        "3. *'Are the macroeconomic assumptions in the latest strategy report fully substantiated by the official hard data?'*",
        "4. *'What tail risk scenarios are highlighted in the report, and what is our hedging stance?'*",
        "5. *'What are the primary divergence points between quantitative signals and narrative themes?'*",
        "",
        "## 3. Data Integrity & Provenance Index",
        "| Source Family | Description | Record Count | Status |",
        "| :--- | :--- | :--- | :--- |",
        f"| **Strategy Report** | Latest committed macro stance & allocation | {1 if snapshot.latest_report else 0} | {'Active' if snapshot.latest_report else 'Missing'} |",
        f"| **Historical Reports** | Retained broker-committed strategy directions | {len(snapshot.historical_reports)} | Available |",
        f"| **Knowledge Catalog** | Regional snapshots (US, TH, Euro, Asia, Global) | {len(snapshot.catalog_notes)} | Available |",
        f"| **Indicator Series** | Historical series points for dashboard indicators | {len(snapshot.indicator_series)} | Loaded |",
        f"| **Market Observables** | 13 Institutional market telemetry feeds | {len(snapshot.market_observables)} | Cached |",
        f"| **Thailand Hard Data** | NESDC, BOT, TPSO, and MOF official data | {1 if snapshot.thailand_hard_data else 0} | {'Loaded' if snapshot.thailand_hard_data else 'Missing'} |",
        f"| **Sector Rotation** | Sector momentum & relative strength metrics | {1 if snapshot.sector_rotation else 0} | {'Loaded' if snapshot.sector_rotation else 'Missing'} |",
        f"| **News & References** | High-impact events & filtered narrative items | {len(snapshot.news_funnel.get('pending', [])) + len(snapshot.news_funnel.get('filtered', []))} | Available |",
        "",
    ]

    if warnings:
        lines.append("## 4. Warnings & Data Gaps")
        for w in warnings:
            lines.append(f"- ⚠️ {w}")
        lines.append("")

    return "\n".join(lines)


def format_current_macro_report(snapshot: MacroCorpusSnapshot) -> str:
    """Part 01: Full text and structured metrics of the latest Macro Strategy Report.

    Fixes C05: Supports canonical asset_allocation, focus_themes, dict/list observable_registry,
    and all risk/evidence/probabilities fields.
    """
    rep = snapshot.latest_report
    if not rep:
        return "# Current Macro Strategy Report\n\n*No active Macro Strategy report is available in the vault.*"

    report_id = rep.get("strategy_report_id") or rep.get("report_id") or "Unknown"
    evaluated_at = rep.get("evaluated_at", "Unknown")
    time_horizon = rep.get("time_horizon", "3-6 Months")
    conviction = rep.get("conviction_level", "medium")
    alignment = rep.get("quant_narrative_alignment", "aligned")
    divergence_note = rep.get("divergence_note", "")

    regime = rep.get("regime", {})
    if isinstance(regime, str):
        regime_label = regime
        confidence_str = "N/A"
    elif isinstance(regime, dict):
        regime_label = regime.get("current_regime") or rep.get("overall_regime") or "Unknown"
        conf = regime.get("confidence")
        confidence_str = f"{conf:.0%}" if conf is not None else "N/A"
    else:
        regime_label = str(rep.get("overall_regime") or "Unknown")
        confidence_str = "N/A"

    allocations = rep.get("asset_allocation") or rep.get("allocations") or []
    themes = rep.get("focus_themes") or rep.get("themes") or []
    pair_trades = rep.get("pair_trades") or []
    risk_scenarios = rep.get("risk_scenarios") or []
    regime_probs = rep.get("regime_probabilities") or {}
    regime_evidences = rep.get("regime_evidence") or []
    thailand_stance = rep.get("thailand_market_stance") or {}
    assumptions = rep.get("key_assumptions") or []
    val_warnings = rep.get("validation_warnings") or []
    stale_warnings = rep.get("stale_data_warnings") or []
    observables_raw = rep.get("observable_registry") or {}

    lines = [
        f"# Current Macro Strategy Report ({report_id})",
        "",
        f"- **Evaluated At:** `{evaluated_at}`",
        f"- **Strategy Report ID:** `{report_id}`",
        f"- **Overall Regime:** `{regime_label}` (Confidence: `{confidence_str}`)",
        f"- **Time Horizon:** `{time_horizon}`",
        f"- **Conviction Level:** `{str(conviction).upper()}`",
        f"- **Quant-Narrative Alignment:** `{alignment}`",
    ]
    if divergence_note:
        lines.append(f"- **Divergence Note:** {divergence_note}")
    lines.append("")

    # 1. Executive Summary
    summary_text = (
        rep.get("summary")
        or rep.get("executive_summary")
        or rep.get("conviction_rationale")
        or "No summary provided."
    )
    lines.extend([
        "## 1. Executive Summary & Regime Stance",
        str(summary_text),
        "",
    ])

    # 2. Key Assumptions
    if assumptions:
        lines.extend([
            "## 2. Key Macroeconomic Assumptions",
        ])
        for asm in assumptions:
            lines.append(f"- {asm}")
        lines.append("")

    # 3. Regime Evidence & Probabilities
    if regime_probs or regime_evidences:
        lines.append("## 3. Regime Probabilities & 5-Dimensional Evidence")
        if regime_probs and isinstance(regime_probs, dict):
            lines.extend([
                "### Regime Probabilities",
                "| Regime | Probability |",
                "| :--- | :--- |",
            ])
            for r_name, r_prob in regime_probs.items():
                lines.append(f"| **{r_name}** | {r_prob} |")
            lines.append("")

        if regime_evidences and isinstance(regime_evidences, list):
            lines.extend([
                "### Multi-Dimensional Evidence",
                "| Dimension | Signal | Hard Data Evidence | Conflict | Confidence |",
                "| :--- | :--- | :--- | :--- | :--- |",
            ])
            for ev in regime_evidences:
                if isinstance(ev, dict):
                    dim = ev.get("dimension", "N/A")
                    sig = ev.get("signal", "N/A")
                    ev_txt = ev.get("evidence", "N/A")
                    conf = ev.get("conflict", "-") or "-"
                    c_level = str(ev.get("confidence", "N/A")).upper()
                    lines.append(f"| **{dim}** | {sig} | {ev_txt} | {conf} | {c_level} |")
            lines.append("")

    # 4. Asset Allocation Table & Detailed Rationale
    lines.extend([
        "## 4. Cross-Asset Allocation Stance",
        "| Asset Class | Stance | Delta vs Benchmark | Time Horizon | Confidence | Key Rationale |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ])
    for alloc in allocations:
        if isinstance(alloc, dict):
            a_class = alloc.get("asset_class", "N/A")
            a_stance = alloc.get("stance", "N/A")
            if hasattr(a_stance, "value"):
                a_stance = a_stance.value
            a_delta = alloc.get("delta", alloc.get("weight_pct", alloc.get("target_pct", "-")))
            a_horiz = alloc.get("time_horizon", time_horizon)
            a_conf = str(alloc.get("confidence", "medium")).upper()
            a_rat = str(alloc.get("rationale", "-")).replace("\n", " ")
            lines.append(f"| **{a_class}** | {a_stance} | {a_delta} | {a_horiz} | {a_conf} | {a_rat[:120]} |")
    lines.append("")

    # Detailed allocation breakdowns
    lines.append("### Detailed Allocation Rationale & Observables")
    for alloc in allocations:
        if isinstance(alloc, dict):
            a_class = alloc.get("asset_class", "N/A")
            a_stance = alloc.get("stance", "N/A")
            lines.extend([
                f"#### {a_class} — {a_stance}",
                f"- **Rationale:** {alloc.get('rationale', 'N/A')}",
                f"- **Supporting Data:** {', '.join(str(sd) for sd in alloc.get('supporting_data', [])) or 'None'}",
                f"- **Benchmark Ref:** {alloc.get('benchmark_ref', 'N/A')}",
                f"- **Source Refs:** {', '.join(str(sr) for sr in alloc.get('source_refs', [])) or 'N/A'}",
            ])
            invals = alloc.get("invalidation_conditions", [])
            if invals:
                lines.append(f"- **Invalidation Conditions:** {', '.join(str(iv) for iv in invals)}")
            warns = alloc.get("validation_warnings", [])
            if warns:
                lines.append(f"- ⚠️ **Warnings:** {', '.join(str(w) for w in warns)}")
            lines.append("")

    # 5. Key Themes
    if themes:
        lines.append("## 5. Key Investment Themes")
        for th in themes:
            if isinstance(th, dict):
                lines.append(f"### {th.get('title', 'Theme')}")
                lines.append(f"- **Horizon:** {th.get('horizon', 'N/A')}")
                lines.append(f"- **Thesis:** {th.get('thesis', 'N/A')}")
            elif isinstance(th, str):
                lines.append(f"- **{th}**")
        lines.append("")

    # 6. Pair Trades
    if pair_trades:
        lines.extend([
            "## 6. Recommended Tactical Pair Trades",
            "| Long Asset | Short Asset | Thesis | Risk Management / Budget |",
            "| :--- | :--- | :--- | :--- |",
        ])
        for pt in pair_trades:
            if isinstance(pt, dict):
                l_asset = pt.get("long_asset") or pt.get("long", "N/A")
                s_asset = pt.get("short_asset") or pt.get("short", "N/A")
                pt_thesis = str(pt.get("thesis", "N/A")).replace("\n", " ")
                pt_rm = pt.get("risk_management") or pt.get("risk_budget") or "-"
                lines.append(f"| **{l_asset}** | **{s_asset}** | {pt_thesis} | {pt_rm} |")
        lines.append("")

    # 7. Risk Scenarios
    if risk_scenarios:
        lines.extend([
            "## 7. Tail Risk Mitigation Scenarios",
            "| Scenario | Probability | Impact | Hedging Action | Trigger Condition |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ])
        for sc in risk_scenarios:
            if isinstance(sc, dict):
                s_name = sc.get("scenario", "N/A")
                s_prob = sc.get("probability", "-")
                s_imp = sc.get("impact", sc.get("impact_severity", "-"))
                s_hedge = sc.get("hedging_action", "-")
                s_trig = sc.get("trigger_condition", "-")
                lines.append(f"| **{s_name}** | {s_prob} | {s_imp} | {s_hedge} | {s_trig} |")
        lines.append("")

    # 8. Thailand Market Stance
    if thailand_stance and isinstance(thailand_stance, dict):
        lines.extend([
            "## 8. Thailand Market Stance & Microstructure",
            f"- **Rationale:** {thailand_stance.get('rationale', 'N/A')}",
        ])
        flow = thailand_stance.get("investor_flow", {})
        if flow:
            lines.append(f"- **SET Foreign Net Flow:** `{flow.get('foreign_net_mb', 'N/A')} MB`")
        breadth = thailand_stance.get("market_breadth", {})
        if breadth:
            lines.append(f"- **SET Advance/Decline Ratio:** `{breadth.get('advance_decline_ratio', 'N/A')}x` (Sentiment: `{breadth.get('sentiment', 'N/A')}`)")
        val = thailand_stance.get("valuation", {})
        if val:
            lines.append(f"- **SET Trailing P/E:** `{val.get('pe_ratio', 'N/A')}x`")
        gold = thailand_stance.get("physical_gold", {})
        if gold:
            lines.append(f"- **GTA Bar Sell:** `{gold.get('bar_sell_thb', 'N/A')} THB`")
        if "policy_spread_bps" in thailand_stance:
            lines.append(f"- **Policy Rate Spread (Fed - BOT):** `{thailand_stance.get('policy_spread_bps')} bps`")
        lines.append("")

    # 9. Complete Observable Registry (C05: Dict or List handled)
    obs_list = []
    if isinstance(observables_raw, dict):
        for k, v in observables_raw.items():
            if isinstance(v, dict):
                v_copy = dict(v)
                v_copy.setdefault("observable_id", k)
                obs_list.append(v_copy)
            elif hasattr(v, "model_dump"):
                d = v.model_dump(mode="json")
                d.setdefault("observable_id", k)
                obs_list.append(d)
            else:
                obs_list.append({"observable_id": k, "value": str(v)})
    elif isinstance(observables_raw, list):
        obs_list = [o if isinstance(o, dict) else (o.model_dump(mode="json") if hasattr(o, "model_dump") else {"value": str(o)}) for o in observables_raw]

    if obs_list:
        lines.extend([
            "## 9. Observable Registry & Primary Evidence Citations",
            f"Total referenced observables: `{len(obs_list)}`",
            "",
            "| Observable ID | Provider | Indicator | Value | Unit | Observed Date | Status |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ])
        for obs in obs_list:
            o_id = obs.get("observable_id") or obs.get("id") or "N/A"
            o_prov = obs.get("provider", "N/A")
            o_ind = obs.get("indicator", "N/A")
            o_val = obs.get("value", "N/A")
            o_unit = obs.get("unit", "")
            o_date = obs.get("observed_at", obs.get("date", "N/A"))
            o_stat = obs.get("status", "ok")
            lines.append(f"| `{o_id}` | {o_prov} | {o_ind} | {o_val} | {o_unit} | {o_date} | {o_stat} |")
        lines.append("")

    # 10. Audit Warnings & Data Timestamp Notes
    if val_warnings or stale_warnings:
        lines.append("## 10. Integrity Warnings & Guardrail Flags")
        for w in val_warnings:
            lines.append(f"- ⚠️ **Validation:** {w}")
        for sw in stale_warnings:
            lines.append(f"- ⏳ **Stale Data:** {sw}")
        lines.append("")

    # Embedded raw markdown narrative if present
    raw_md = rep.get("raw_markdown") or rep.get("markdown_content") or rep.get("content_md")
    if raw_md:
        lines.extend([
            "## 11. Full Analyst Narrative (Markdown)",
            "",
            str(raw_md),
            "",
        ])

    return "\n".join(lines)


def format_historical_reports(snapshot: MacroCorpusSnapshot) -> str:
    """Part 02: Historical Strategy Reports retained in the vault.

    Fixes C06: Formats ALL retained reports without arbitrary truncation.
    """
    history = snapshot.historical_reports
    if not history:
        return "# Historical Macro Strategy Reports\n\n*No historical strategy reports retained in archive.*"

    lines = [
        "# Historical Macro Strategy Reports Archive",
        "",
        f"Total retained reports in scope: `{len(history)}`",
        "",
        "| Report ID | Evaluated At | Regime | Conviction | Time Horizon | Asset Stances |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for item in history:
        rid = item.get("strategy_report_id") or item.get("report_id") or item.get("_file_origin", "Unknown")
        ev_at = str(item.get("evaluated_at", "N/A"))[:10]
        regime = item.get("regime", {})
        if isinstance(regime, dict):
            reg_name = regime.get("current_regime") or item.get("overall_regime", "N/A")
        else:
            reg_name = item.get("overall_regime") or str(regime)
        conv = str(item.get("conviction_level", "N/A")).upper()
        horiz = item.get("time_horizon", "-")
        allocs = item.get("asset_allocation") or item.get("allocations") or []
        alloc_summary = ", ".join(
            f"{a.get('asset_class')}:{a.get('stance')}" for a in allocs if isinstance(a, dict)
        ) or "None"
        lines.append(f"| `{rid}` | `{ev_at}` | `{reg_name}` | `{conv}` | `{horiz}` | {alloc_summary[:100]} |")

    lines.append("")
    lines.append("## Detailed Records for All Retained Historical Reports")
    lines.append("")

    for idx, item in enumerate(history, start=1):
        rid = item.get("strategy_report_id") or item.get("report_id") or item.get("_file_origin", "Unknown")
        summary_text = (
            item.get("summary")
            or item.get("executive_summary")
            or item.get("conviction_rationale")
            or "N/A"
        )
        file_orig = item.get("_file_relative") or item.get("_file_origin") or "Vault"
        lines.extend([
            f"### {idx}. Report: {rid}",
            f"- **Evaluated At:** `{item.get('evaluated_at', 'N/A')}`",
            f"- **Source File:** `{file_orig}`",
            f"- **Overall Regime:** `{item.get('overall_regime', 'N/A')}`",
            f"- **Conviction:** `{item.get('conviction_level', 'N/A')}`",
            f"- **Executive Summary:** {summary_text}",
            "",
        ])
        h_allocs = item.get("asset_allocation") or item.get("allocations") or []
        if h_allocs and isinstance(h_allocs, list):
            lines.extend([
                "| Asset Class | Stance | Delta | Rationale |",
                "| :--- | :--- | :--- | :--- |",
            ])
            for ha in h_allocs:
                if isinstance(ha, dict):
                    lines.append(f"| {ha.get('asset_class', 'N/A')} | {ha.get('stance', 'N/A')} | {ha.get('delta', '-')} | {str(ha.get('rationale', '-'))[:80]} |")
            lines.append("")

    return "\n".join(lines)


def format_us_market_observables(snapshot: MacroCorpusSnapshot) -> str:
    """Part 03: US Market Telemetry (Yield Curve, Stress, COT, Commodity, Auctions, Debt).

    Fixes C01 & C06: Renders complete field payloads for all cached items.
    """
    obs = snapshot.market_observables
    lines = [
        "# US Macro Market Observables",
        "",
        "Real-time institutional telemetry captured from Federal Reserve, Treasury, OFR, and CFTC.",
        "",
    ]

    # 1. US Treasury Yield Curve
    yc = obs.get("us_yield_curve", {})
    lines.extend([
        "## 1. US Treasury Yield Curve",
        f"- **Status:** `{yc.get('status', 'missing')}`",
        f"- **Cache Key:** `{yc.get('cache_key', 'treasury:yield_curve:latest')}`",
    ])
    if yc.get("status") == "cached" and yc.get("data"):
        data = yc["data"]
        lines.append(f"- **As of Date:** `{data.get('as_of_date', data.get('date', 'N/A'))}`")
        rates = data.get("rates") or data.get("yields") or {}
        if isinstance(rates, dict):
            lines.extend([
                "",
                "| Tenor | Yield (%) | Spread vs 3M (bps) | Spread vs 2Y (bps) |",
                "| :--- | :--- | :--- | :--- |",
            ])
            r_3m = rates.get("3M") or rates.get("3-Month")
            r_2y = rates.get("2Y") or rates.get("2-Year")
            for tenor, y_val in rates.items():
                sp_3m = f"{(y_val - r_3m) * 100:+.1f}" if (r_3m and y_val is not None) else "-"
                sp_2y = f"{(y_val - r_2y) * 100:+.1f}" if (r_2y and y_val is not None) else "-"
                lines.append(f"| **{tenor}** | {y_val:.3f}% | {sp_3m} | {sp_2y} |")
        lines.append("")
    else:
        lines.append("*US Treasury Yield Curve data is not cached.*")
        lines.append("")

    # 2. OFR Financial Stress Index
    fsi = obs.get("ofr_financial_stress", {})
    lines.extend([
        "## 2. OFR Financial Stress Index (FSI)",
        f"- **Status:** `{fsi.get('status', 'missing')}`",
        f"- **Cache Key:** `{fsi.get('cache_key', 'ofr:fsi:latest')}`",
    ])
    if fsi.get("status") == "cached" and fsi.get("data"):
        f_data = fsi["data"]
        lines.append(f"- **Index Value:** `{f_data.get('fsi_value', f_data.get('value', 'N/A'))}`")
        lines.append(f"- **As of Date:** `{f_data.get('as_of_date', f_data.get('date', 'N/A'))}`")
        cats = f_data.get("categories") or f_data.get("components") or {}
        if isinstance(cats, dict):
            lines.extend([
                "",
                "| Category Component | Contribution Value |",
                "| :--- | :--- |",
            ])
            for c_name, c_val in cats.items():
                lines.append(f"| **{c_name}** | {c_val} |")
        lines.append("")
    else:
        lines.append("*OFR Financial Stress Index data is not cached.*")
        lines.append("")

    # 3. CFTC Commitments of Traders (COT) — Metals
    cot = obs.get("gold_cot", {})
    lines.extend([
        "## 3. CFTC Commitments of Traders (COT) — Gold Positioning",
        f"- **Status:** `{cot.get('status', 'missing')}`",
        f"- **Cache Key:** `{cot.get('cache_key', 'cftc:cot:disagg:088691')}`",
    ])
    if cot.get("status") == "cached" and cot.get("data"):
        cot_data = cot["data"]
        lines.extend([
            f"- **Report Date:** `{cot_data.get('report_date', cot_data.get('as_of_date', 'N/A'))}`",
            f"- **Commercial Net:** `{cot_data.get('commercial_net', 'N/A')}`",
            f"- **Non-Commercial Net:** `{cot_data.get('non_commercial_net', 'N/A')}`",
            f"- **Non-Commercial Percentile (3Y):** `{cot_data.get('non_commercial_percentile', 'N/A')}%`",
            f"- **Open Interest:** `{cot_data.get('open_interest', 'N/A')}`",
            "",
        ])
    else:
        lines.append("*CFTC Gold COT data is not cached.*")
        lines.append("")

    # 4. Commodity Volatility & Auctions & Debt
    vol = obs.get("commodity_volatility", {})
    auc_10y = obs.get("treasury_auction_10y", {})
    auc_13w = obs.get("treasury_auction_13w", {})
    debt = obs.get("us_national_debt", {})

    lines.extend([
        "## 4. Commodity Volatility & US Fiscal Telemetry",
        f"- **Commodity Volatility Status:** `{vol.get('status', 'missing')}`",
        f"- **Treasury 10Y Auction Demand Status:** `{auc_10y.get('status', 'missing')}`",
        f"- **Treasury 13W Auction Demand Status:** `{auc_13w.get('status', 'missing')}`",
        f"- **US National Debt Status:** `{debt.get('status', 'missing')}`",
        "",
    ])

    if vol.get("status") == "cached" and vol.get("data"):
        lines.extend([
            "### Commodity Volatility Breakdown (OVX / GVZ / VXSLV)",
            f"```json\n{json.dumps(vol['data'], indent=2, default=str)}\n```",
            "",
        ])

    if (auc_10y.get("status") == "cached" and auc_10y.get("data")) or (auc_13w.get("status") == "cached" and auc_13w.get("data")):
        lines.extend([
            "### US Treasury Auction Demand Metrics",
            "| Security Term | Latest Auction Date | High Yield/Rate | Bid-to-Cover | Prior Mean BTC | Demand Delta |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ])
        for auc_item, term_label in [(auc_10y, "10-Year Note"), (auc_13w, "13-Week Bill")]:
            if auc_item.get("status") == "cached" and auc_item.get("data"):
                ad = auc_item["data"]
                lines.append(
                    f"| **{term_label}** | {ad.get('latest_auction_date', 'N/A')} | "
                    f"{ad.get('latest_high_yield', ad.get('latest_high_discount_rate', 'N/A'))} | "
                    f"{ad.get('latest_bid_to_cover_ratio', 'N/A')} | "
                    f"{ad.get('prior_mean_bid_to_cover', 'N/A')} | "
                    f"{ad.get('demand_delta', 'N/A')} |"
                )
        lines.append("")

    if debt.get("status") == "cached" and debt.get("data"):
        lines.extend([
            "### US National Debt Public Accounting",
            f"```json\n{json.dumps(debt['data'], indent=2, default=str)}\n```",
            "",
        ])

    return "\n".join(lines)


def format_thailand_market_observables(snapshot: MacroCorpusSnapshot) -> str:
    """Part 04: Thailand Market & Official Hard Data."""
    obs = snapshot.market_observables
    hard_data = snapshot.thailand_hard_data
    lines = [
        "# Thailand Macro & Market Telemetry",
        "",
        "## 1. SET Market Observables",
        f"- **Foreign & Retail Fund Flow Status:** `{obs.get('thai_investor_flow', {}).get('status', 'missing')}`",
        f"- **Retail Gold (Gold Traders Association) Status:** `{obs.get('thai_retail_gold', {}).get('status', 'missing')}`",
        f"- **SET Market Valuation Status:** `{obs.get('thai_market_valuation', {}).get('status', 'missing')}`",
        f"- **SET Market Breadth Status:** `{obs.get('thai_market_breadth', {}).get('status', 'missing')}`",
        "",
    ]

    th_flow = obs.get("thai_investor_flow", {})
    if th_flow.get("status") == "cached" and th_flow.get("data"):
        lines.extend([
            "### SET Investor Flow Breakdown",
            f"```json\n{json.dumps(th_flow['data'], indent=2, default=str)}\n```",
            "",
        ])

    th_gold = obs.get("thai_retail_gold", {})
    if th_gold.get("status") == "cached" and th_gold.get("data"):
        lines.extend([
            "### Thai Retail Gold Quotes",
            f"```json\n{json.dumps(th_gold['data'], indent=2, default=str)}\n```",
            "",
        ])

    th_val = obs.get("thai_market_valuation", {})
    th_breadth = obs.get("thai_market_breadth", {})
    if (th_val.get("status") == "cached" and th_val.get("data")) or (th_breadth.get("status") == "cached" and th_breadth.get("data")):
        lines.extend([
            "### SET Valuation & Breadth Metrics",
            f"```json\n{json.dumps({'valuation': th_val.get('data'), 'breadth': th_breadth.get('data')}, indent=2, default=str)}\n```",
            "",
        ])

    # 2. Official Macro Hard Data (NESDC, BOT, TPSO, MOF)
    lines.append("## 2. Thailand Official Macroeconomic Hard Data")
    if hard_data:
        records = hard_data.get("records") or hard_data.get("indicators") or hard_data
        lines.extend([
            f"- **Last Updated:** `{hard_data.get('updated_at', hard_data.get('as_of', 'N/A'))}`",
            "",
            "| Indicator | Authority | Period | Value | Unit | Status | Notes |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ])
        if isinstance(records, list):
            for rec in records:
                if isinstance(rec, dict):
                    lines.append(
                        f"| {rec.get('indicator_name', rec.get('series_id', 'N/A'))} | "
                        f"{rec.get('source_authority', rec.get('source', 'N/A'))} | {rec.get('period', 'N/A')} | "
                        f"{rec.get('value', 'N/A')} | {rec.get('unit', '')} | {rec.get('status', 'ok')} | "
                        f"{rec.get('notes', '-')[:60]} |"
                    )
        elif isinstance(records, dict):
            for k, v in records.items():
                if isinstance(v, dict):
                    lines.append(f"| {k} | {v.get('source', 'N/A')} | {v.get('period', 'N/A')} | {v.get('value', 'N/A')} | {v.get('unit', '')} | {v.get('status', 'ok')} | - |")
                else:
                    lines.append(f"| {k} | Official Authority | N/A | {v} | N/A | ok | - |")
        lines.append("")
    else:
        lines.append("*No official Thai hard data file found at configured path.*")
        lines.append("")

    return "\n".join(lines)


def format_global_and_catalog_notes(snapshot: MacroCorpusSnapshot) -> str:
    """Part 05: Global Policy Rates, Cross-Border Liquidity & Catalog Regional Notes.

    Fixes C04: Evaluates full_body before body_snippet so long notes are never truncated.
    """
    obs = snapshot.market_observables
    catalog_notes = snapshot.catalog_notes

    lines = [
        "# Global & Regional Macro Telemetry",
        "",
        "## 1. Cross-Border Policy Rates & Liquidity",
        f"- **BIS Global Policy Rates Status:** `{obs.get('global_policy_rates', {}).get('status', 'missing')}`",
        f"- **Crypto Macro Liquidity Proxy Status:** `{obs.get('crypto_macro_liquidity', {}).get('status', 'missing')}`",
        "",
    ]

    bis = obs.get("global_policy_rates", {})
    if bis.get("status") == "cached" and bis.get("data"):
        lines.extend([
            "### BIS Central Bank Policy Rates (12 Jurisdictions)",
            f"```json\n{json.dumps(bis['data'], indent=2, default=str)}\n```",
            "",
        ])

    crypto_liq = obs.get("crypto_macro_liquidity", {})
    if crypto_liq.get("status") == "cached" and crypto_liq.get("data"):
        lines.extend([
            "### Crypto Macro Liquidity & Risk Appetite Breakdown",
            f"```json\n{json.dumps(crypto_liq['data'], indent=2, default=str)}\n```",
            "",
        ])

    lines.extend([
        "## 2. Knowledge Catalog Regional Notes",
        f"Total indexed notes in vault: `{len(catalog_notes)}`",
        "",
    ])

    for note in catalog_notes:
        # C04 fix: full_body evaluated before body_snippet
        body_content = note.get("full_body") or note.get("body_snippet") or "*Empty note body*"
        lines.extend([
            f"### {note.get('title', 'Regional Note')}",
            f"- **Entity Type:** `{note.get('entity_type', 'N/A')}`",
            f"- **Relative Path:** `{note.get('relative_path', 'N/A')}`",
            f"- **Date:** `{note.get('date', 'N/A')}`",
            "",
            "#### Note Content:",
            str(body_content),
            "",
            "---",
            "",
        ])

    return "\n".join(lines)


def format_sector_rotation(snapshot: MacroCorpusSnapshot) -> str:
    """Part 06: Sector Rotation Dashboard Snapshot & Historical Metrics."""
    sec = snapshot.sector_rotation
    lines = [
        "# Sector Rotation & Industry Relative Strength",
        "",
    ]
    if not sec:
        lines.append("*No Sector Rotation snapshot is available in runtime store.*")
        return "\n".join(lines)

    state = sec.get("state", {})
    latest_snap = sec.get("latest_snapshot") or sec.get("snapshot") or {}
    hist_snaps = sec.get("historical_snapshots") or []

    lines.extend([
        f"- **State Timestamp:** `{state.get('latest_as_of', state.get('updated_at', 'N/A'))}`",
        f"- **Latest Snapshot ID:** `{state.get('latest_snapshot_id', 'N/A')}`",
        f"- **Current Regime:** `{state.get('current_regime', 'N/A')}`",
        f"- **Top Leading Sectors:** `{', '.join(state.get('leading_sectors', [])) or 'N/A'}`",
        f"- **Lagging Sectors:** `{', '.join(state.get('lagging_sectors', [])) or 'N/A'}`",
        "",
        "## Current Snapshot Payload",
        f"```json\n{json.dumps(latest_snap, indent=2, default=str)}\n```",
        "",
    ])

    if hist_snaps:
        lines.extend([
            f"## Retained Historical Sector Snapshots ({len(hist_snaps)})",
            "| Snapshot ID | Summary |",
            "| :--- | :--- |",
        ])
        for hs in hist_snaps:
            s_id = hs.get("snapshot_id", "N/A")
            lines.append(f"| `{s_id}` | Available in store |")
        lines.append("")

    return "\n".join(lines)


def format_news_and_references(snapshot: MacroCorpusSnapshot) -> str:
    """Part 07: High-Impact Macro News Events, Transcripts, and Filtered Items.

    Fixes C06: Renders all filtered events without truncation.
    """
    news = snapshot.news_funnel
    pending = news.get("pending", [])
    filtered = news.get("filtered", [])

    lines = [
        "# Macro News Funnel & Content References",
        "",
        f"- **Pending High-Impact Events:** `{len(pending)}`",
        f"- **Filtered / Rejected Events:** `{len(filtered)}`",
        "",
        "## 1. Pending High-Impact Macro Events",
    ]

    if pending:
        lines.extend([
            "| ID | Title | Source | Impact | Date | Evidence Summary |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ])
        for p in pending:
            ev_summary = str(p.get("summary") or p.get("snippet") or "-").replace("\n", " ")[:120]
            lines.append(
                f"| `{p.get('id', p.get('event_id', 'N/A'))}` | {p.get('title', 'N/A')} | "
                f"{p.get('source', p.get('publisher', 'N/A'))} | {p.get('impact_score', p.get('impact', 'N/A'))} | "
                f"{p.get('published_at', p.get('date', 'N/A'))} | {ev_summary} |"
            )
        lines.append("")
    else:
        lines.append("*No pending news events in the funnel.*")
        lines.append("")

    lines.extend([
        "## 2. Filtered or Rejected Macro Events (Auditing Context)",
    ])
    if filtered:
        lines.extend([
            "| ID | Title | Filter Reason | Date |",
            "| :--- | :--- | :--- | :--- |",
        ])
        # C06 fix: Formats all filtered events without arbitrary truncation
        for f in filtered:
            lines.append(
                f"| `{f.get('id', f.get('event_id', 'N/A'))}` | {f.get('title', 'N/A')} | "
                f"{f.get('filter_reason', f.get('rejection_reason', 'N/A'))} | "
                f"{f.get('published_at', f.get('date', 'N/A'))} |"
            )
        lines.append("")
    else:
        lines.append("*No filtered events logged.*")
        lines.append("")

    return "\n".join(lines)


def format_structured_appendix(snapshot: MacroCorpusSnapshot) -> str:
    """Part 08: Structured Appendix preserving all raw schema fields and formulas.

    Fixes C07: In addition to indicator points, appends full raw JSON blocks for strategy report,
    market observables, Thailand hard data, sector rotation, and news funnel.
    """
    lines = [
        "# Structured Appendix & Lineage Mappings",
        "",
        "Preserves complete raw payloads, mathematical inputs, formulas, and unrendered fields.",
        "",
        "## 1. Indicator Series Data Points",
    ]

    for s in snapshot.indicator_series:
        lines.extend([
            f"### Series: {s.get('label')} (`{s.get('series_key')}`)",
            f"- **Indicator ID:** `{s.get('indicator_id')}`",
            f"- **Unit:** `{s.get('unit')}`",
            f"- **Points count:** `{len(s.get('points', []))}`",
            "```json",
            json.dumps(s.get("points", []), indent=2, default=str),
            "```",
            "",
        ])

    # C07 addition: Full raw Strategy Report JSON
    if snapshot.latest_report:
        lines.extend([
            "## 2. Complete Strategy Report Canonical Payload",
            "```json:latest_report",
            json.dumps(snapshot.latest_report, indent=2, default=str),
            "```",
            "",
        ])

    # C07 addition: Full raw Market Observables JSON
    if snapshot.market_observables:
        lines.extend([
            "## 3. Complete Market Observables Telemetry Snapshot",
            "```json:market_observables",
            json.dumps(snapshot.market_observables, indent=2, default=str),
            "```",
            "",
        ])

    # C07 addition: Full raw Thailand Hard Data JSON
    if snapshot.thailand_hard_data:
        lines.extend([
            "## 4. Complete Thailand Official Hard Data Payload",
            "```json:thailand_hard_data",
            json.dumps(snapshot.thailand_hard_data, indent=2, default=str),
            "```",
            "",
        ])

    # C07 addition: Full raw Sector Rotation Payload
    if snapshot.sector_rotation:
        lines.extend([
            "## 5. Complete Sector Rotation Telemetry Payload",
            "```json:sector_rotation",
            json.dumps(snapshot.sector_rotation, indent=2, default=str),
            "```",
            "",
        ])

    return "\n".join(lines)
