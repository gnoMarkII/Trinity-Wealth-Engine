"""Synthetic Test Fixtures for Obsidian Vault V2 Testing.

Generates isolated, reproducible test vaults under tmp_path with all edge cases:
- Duplicate stems across folders (FTNT Hub, Holding, Concept)
- Analysis with companion revision & latest JSON sidecars
- Quant snapshots
- Earnings Calls: URL transcript, pasted transcript without URL, FY annual periods,
  multiple transcripts in same period
- News & YouTube with custom properties & companion files
- Briefing books with .quality.json & run manifests
- Macro strategies with duplicate paths
- Notes with schema_version=3 (future schema)
- Malformed YAML frontmatter and broken/ambiguous wikilinks
- Excluded folders (.obsidian, .trash, .sync_history, backup directories)
Zero live-portfolio or private data committed into tests.
"""
from __future__ import annotations

import json
from pathlib import Path


def create_comprehensive_test_vault(root: Path) -> dict[str, Path]:
    """Populates root with comprehensive synthetic files and returns key paths."""
    root.mkdir(parents=True, exist_ok=True)
    created: dict[str, Path] = {}

    # 1. FTNT Hub, Holding, Concept (Duplicate stem test case)
    hub_dir = root / "30_Knowledge_Base" / "Stocks" / "FTNT"
    hub_dir.mkdir(parents=True, exist_ok=True)
    ftnt_hub = hub_dir / "FTNT.md"
    ftnt_hub.write_text(
        "---\n"
        "schema_version: 2\n"
        "note_id: note_ftnt_hub\n"
        "entity_type: stock_hub\n"
        "title: FTNT\n"
        "ticker: FTNT\n"
        "date: 2026-09-01\n"
        "tags: [stock, cybersecurity, ftnt]\n"
        "---\n\n"
        "# FTNT\n\n"
        "Hub for Fortinet.\n"
        "Links: [[2026-09-05 FTNT Equity Analysis]] and [[2026-Q2 FTNT Earnings Call]]\n",
        encoding="utf-8",
    )
    created["ftnt_hub"] = ftnt_hub

    holding_dir = root / "20_Portfolio_Management" / "Holdings"
    holding_dir.mkdir(parents=True, exist_ok=True)
    ftnt_holding = holding_dir / "FTNT.md"
    ftnt_holding.write_text(
        "---\n"
        "schema_version: 2\n"
        "note_id: note_ftnt_holding\n"
        "entity_type: holding\n"
        "title: FTNT Holding\n"
        "ticker: FTNT\n"
        "date: 2026-09-01\n"
        "---\n\n"
        "# FTNT Portfolio Position\n",
        encoding="utf-8",
    )
    created["ftnt_holding"] = ftnt_holding

    concept_dir = root / "30_Knowledge_Base" / "Concepts"
    concept_dir.mkdir(parents=True, exist_ok=True)
    ftnt_concept = concept_dir / "FTNT.md"
    ftnt_concept.write_text(
        "---\n"
        "title: FTNT Concept\n"
        "---\n\n"
        "# FTNT Concept Stub\n",
        encoding="utf-8",
    )
    created["ftnt_concept"] = ftnt_concept

    # 2. FTNT Analysis MD + JSON sidecar (revision & latest format)
    ana_dir = hub_dir / "Analysis"
    ana_dir.mkdir(parents=True, exist_ok=True)
    ftnt_analysis_md = ana_dir / "2026-09-05 FTNT Equity Analysis.md"
    ftnt_analysis_md.write_text(
        "---\n"
        "schema_version: 2\n"
        "note_id: note_ftnt_ana_01\n"
        "entity_type: equity_analysis\n"
        "title: 2026-09-05 FTNT Equity Analysis\n"
        "ticker: FTNT\n"
        "date: 2026-09-05\n"
        "composite_score: 78.5\n"
        "---\n\n"
        "# FTNT Equity Analysis\n\n"
        "See companion hub [[FTNT]].\n",
        encoding="utf-8",
    )
    created["ftnt_analysis_md"] = ftnt_analysis_md

    ftnt_analysis_json = ana_dir / "2026-09-05 FTNT Equity Analysis.json"
    ftnt_analysis_json.write_text(
        json.dumps({
            "ticker": "FTNT",
            "market": "US",
            "analysis_date": "2026-09-05",
            "quant_signals": {
                "ticker": "FTNT",
                "company_name": "Fortinet Inc.",
                "evaluated_at": "2026-09-05T12:00:00Z",
                "composite_score": 78.5,
            },
            "sentiment_context": {
                "evaluated_at": "2026-09-05T12:00:00Z",
                "market_sentiment": "Bullish",
            },
        }, indent=2),
        encoding="utf-8",
    )
    created["ftnt_analysis_json"] = ftnt_analysis_json

    # 3. FTNT Quant snapshot
    quant_dir = hub_dir / "Quant"
    quant_dir.mkdir(parents=True, exist_ok=True)
    ftnt_quant_md = quant_dir / "2026-09-05 FTNT Quant Snapshot.md"
    ftnt_quant_md.write_text(
        "---\n"
        "schema_version: 2\n"
        "note_id: note_ftnt_quant_01\n"
        "entity_type: quant_snapshot\n"
        "title: 2026-09-05 FTNT Quant Snapshot\n"
        "ticker: FTNT\n"
        "date: 2026-09-05\n"
        "---\n\n"
        "# Quant Snapshot\n",
        encoding="utf-8",
    )
    created["ftnt_quant_md"] = ftnt_quant_md

    # 4. Earnings Calls: URL transcript, pasted transcript without URL, FY annual, duplicate period
    earnings_dir = hub_dir / "Earnings"
    earnings_dir.mkdir(parents=True, exist_ok=True)
    
    # 4a: URL transcript Q1
    ec_q1 = earnings_dir / "2026-Q1 FTNT Earnings Call.md"
    ec_q1.write_text(
        "---\n"
        "schema_version: 2\n"
        "note_id: note_ftnt_ec_q1\n"
        "entity_type: earnings_call\n"
        "title: 2026-Q1 FTNT Earnings Call\n"
        "ticker: FTNT\n"
        "period: 2026-Q1\n"
        "fiscal_quarter: 1\n"
        "fiscal_year: 2026\n"
        "source_url: https://example.com/earnings/ftnt-q1\n"
        "source_verification_status: verified\n"
        "---\n\n"
        "# Q1 2026 Call\n",
        encoding="utf-8",
    )
    created["ec_q1"] = ec_q1

    # 4b: Pasted transcript without URL Q2
    ec_q2 = earnings_dir / "2026-Q2 FTNT Earnings Call.md"
    ec_q2.write_text(
        "---\n"
        "schema_version: 2\n"
        "note_id: note_ftnt_ec_q2\n"
        "entity_type: earnings_call\n"
        "title: 2026-Q2 FTNT Earnings Call\n"
        "ticker: FTNT\n"
        "period: 2026-Q2\n"
        "fiscal_quarter: 2\n"
        "fiscal_year: 2026\n"
        "input_kind: pasted_transcript\n"
        "source_verification_status: not_verified\n"
        "---\n\n"
        "# Q2 2026 Call (Pasted Transcript)\n",
        encoding="utf-8",
    )
    created["ec_q2"] = ec_q2

    # 4c: Fiscal Year (Annual) note without fiscal_quarter
    ec_fy = earnings_dir / "2025-FY FTNT Annual Earnings Call.md"
    ec_fy.write_text(
        "---\n"
        "schema_version: 2\n"
        "note_id: note_ftnt_ec_fy\n"
        "entity_type: earnings_call\n"
        "title: 2025-FY FTNT Annual Earnings Call\n"
        "ticker: FTNT\n"
        "period_type: annual\n"
        "fiscal_year: 2025\n"
        "---\n\n"
        "# FY 2025 Annual Call\n",
        encoding="utf-8",
    )
    created["ec_fy"] = ec_fy

    # 4d: Second transcript for same period (different source hash)
    ec_q2_alt = earnings_dir / "2026-Q2 FTNT Earnings Call (Alternative Transcript).md"
    ec_q2_alt.write_text(
        "---\n"
        "schema_version: 2\n"
        "note_id: note_ftnt_ec_q2_alt\n"
        "entity_type: earnings_call\n"
        "title: 2026-Q2 FTNT Earnings Call (Alternative Transcript)\n"
        "ticker: FTNT\n"
        "period: 2026-Q2\n"
        "---\n\n"
        "# Q2 2026 Alt Transcript\n",
        encoding="utf-8",
    )
    created["ec_q2_alt"] = ec_q2_alt

    # 5. News (Article & Company News with custom fields)
    news_dir = root / "30_Knowledge_Base" / "News" / "2026" / "09"
    news_dir.mkdir(parents=True, exist_ok=True)
    news_file = news_dir / "2026-09-06 Tech Update.md"
    news_file.write_text(
        "---\n"
        "schema_version: 2\n"
        "note_id: note_news_20260906\n"
        "entity_type: company_news\n"
        "title: 2026-09-06 Tech Update\n"
        "date: 2026-09-06\n"
        "tickers: [FTNT, CRWD]\n"
        "custom_portfolio_flag: priority\n"
        "analyst_sentiment: positive\n"
        "---\n\n"
        "# Tech Update\n\n"
        "Mentions [[FTNT]] and [[CRWD]].\n",
        encoding="utf-8",
    )
    created["news_file"] = news_file

    # 6. YouTube Summaries + Companion Canvas
    yt_dir = root / "30_Knowledge_Base" / "YouTube_Summaries" / "2026" / "09"
    yt_dir.mkdir(parents=True, exist_ok=True)
    yt_md = yt_dir / "2026-09-05 Market Outlook.md"
    yt_md.write_text(
        "---\n"
        "schema_version: 2\n"
        "note_id: note_yt_20260905\n"
        "entity_type: youtube_summary\n"
        "title: 2026-09-05 Market Outlook\n"
        "video_id: dQw4w9WgXcQ\n"
        "date: 2026-09-05\n"
        "---\n\n"
        "# Market Outlook Video\n",
        encoding="utf-8",
    )
    created["yt_md"] = yt_md

    yt_canvas = yt_dir / "2026-09-05 Market Outlook.canvas"
    yt_canvas.write_text(
        json.dumps({"nodes": [{"id": "n1", "type": "text", "text": "Video Summary"}], "edges": []}),
        encoding="utf-8",
    )
    created["yt_canvas"] = yt_canvas

    # 7. Briefing + Quality (.quality.json) + Run manifest
    briefing_dir = root / "30_Knowledge_Base" / "NotebookLM_Sources" / "2026" / "09"
    briefing_dir.mkdir(parents=True, exist_ok=True)
    briefing_md = briefing_dir / "2026-09-06 Daily Briefing.md"
    briefing_md.write_text(
        "---\n"
        "schema_version: 2\n"
        "note_id: note_briefing_20260906\n"
        "entity_type: briefing_book\n"
        "title: 2026-09-06 Daily Briefing\n"
        "date: 2026-09-06\n"
        "---\n\n"
        "# Daily Briefing\n",
        encoding="utf-8",
    )
    created["briefing_md"] = briefing_md

    briefing_quality = briefing_dir / "2026-09-06 Daily Briefing.md.quality.json"
    briefing_quality.write_text(
        json.dumps({"verified": True, "score": 95, "checked_at": "2026-09-06T08:00:00Z"}),
        encoding="utf-8",
    )
    created["briefing_quality"] = briefing_quality

    manifest_file = briefing_dir / "run_manifest.json"
    manifest_file.write_text(
        json.dumps({"run_id": "run_001", "status": "completed", "artifacts": ["2026-09-06 Daily Briefing.md"]}),
        encoding="utf-8",
    )
    created["manifest_file"] = manifest_file

    # 8. Macro Strategy duplicates (simulating legacy Dual-Path Archiving)
    macro_strat_dir = root / "30_Knowledge_Base" / "Macroeconomics" / "Strategies" / "2026" / "09"
    macro_strat_dir.mkdir(parents=True, exist_ok=True)
    macro_strat_md = macro_strat_dir / "2026-09-06 Macro Direction.md"
    macro_strat_md.write_text(
        "---\n"
        "schema_version: 2\n"
        "note_id: note_macro_strat_01\n"
        "entity_type: macro_strategy\n"
        "title: 2026-09-06 Macro Direction\n"
        "date: 2026-09-06\n"
        "---\n\n"
        "# Macro Strategy\n",
        encoding="utf-8",
    )
    created["macro_strat_md"] = macro_strat_md

    macro_snap_dir = root / "30_Knowledge_Base" / "Macroeconomics" / "Daily_Snapshots" / "2026" / "09"
    macro_snap_dir.mkdir(parents=True, exist_ok=True)
    macro_dup_md = macro_snap_dir / "2026-09-06 Macro Direction.md"
    macro_dup_md.write_text(
        "---\n"
        "schema_version: 2\n"
        "note_id: note_macro_strat_01_dup\n"
        "entity_type: macro_strategy\n"
        "title: 2026-09-06 Macro Direction\n"
        "date: 2026-09-06\n"
        "---\n\n"
        "# Duplicate Macro Strategy in Snapshots\n",
        encoding="utf-8",
    )
    created["macro_dup_md"] = macro_dup_md

    # 9. Future Schema Version (schema_version=3)
    future_dir = root / "30_Knowledge_Base" / "Concepts"
    future_note = future_dir / "Future Concept.md"
    future_note.write_text(
        "---\n"
        "schema_version: 3\n"
        "note_id: note_future_01\n"
        "entity_type: advanced_concept\n"
        "title: Future Concept\n"
        "date: 2026-09-06\n"
        "custom_matrix: [[1, 2], [3, 4]]\n"
        "---\n\n"
        "# Future Schema 3 Note\n",
        encoding="utf-8",
    )
    created["future_note"] = future_note

    # 10. Malformed YAML note (deliberate syntax error)
    bad_yaml_note = future_dir / "Malformed Note.md"
    bad_yaml_note.write_text(
        "---\n"
        "title: Bad YAML\n"
        "unclosed_key: [missing bracket\n"
        "tags: 'mismatched quote\n"
        "---\n\n"
        "# Malformed Note Body\n",
        encoding="utf-8",
    )
    created["bad_yaml_note"] = bad_yaml_note

    # 11. Broken wikilink note
    broken_link_note = future_dir / "Broken Link Note.md"
    broken_link_note.write_text(
        "---\n"
        "title: Broken Link Note\n"
        "date: 2026-09-06\n"
        "entity_type: concept\n"
        "---\n\n"
        "# Broken Link Note\n\n"
        "Link to nonexistent: [[Completely_Missing_File_12345]].\n",
        encoding="utf-8",
    )
    created["broken_link_note"] = broken_link_note

    # 12. Excluded System / Trash / Backup directories
    obsidian_dir = root / ".obsidian"
    obsidian_dir.mkdir(parents=True, exist_ok=True)
    app_json = obsidian_dir / "app.json"
    app_json.write_text('{"spellcheck": true}', encoding="utf-8")
    created["app_json"] = app_json

    trash_dir = root / ".trash"
    trash_dir.mkdir(parents=True, exist_ok=True)
    trash_file = trash_dir / "Old Note.md"
    trash_file.write_text("# Deleted file", encoding="utf-8")
    created["trash_file"] = trash_file

    backup_dir = root / ".pre_migration_backup_20_Portfolio_Management_v2"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_file = backup_dir / "Portfolio_Holdings_Old.md"
    backup_file.write_text("# Backup", encoding="utf-8")
    created["backup_file"] = backup_file

    return created
