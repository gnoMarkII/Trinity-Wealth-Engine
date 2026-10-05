import json

from langchain_core.language_models.chat_models import BaseChatModel
from langchain.agents import create_agent
from tools.macro.news_radar import generate_news_radar_daily
from tools.knowledge.youtube_monitor import generate_weekly_youtube_digest
from tools.knowledge.article import ingest_article_url
from tools.knowledge.youtube import ingest_youtube_transcript
from tools.macro.baselines import get_macro_baselines
from tools.macro.content_references import news_references_from_radar, recent_youtube_references
from core.prompt_harness import get_harness

# MACRO_ECONOMIST_SYSTEM_PROMPT ถูกย้ายไปที่ prompts/skills/economist/SKILL.md ผ่านระบบ PromptHarness

_macro_economist_tools = [
    generate_news_radar_daily,
    generate_weekly_youtube_digest,
    ingest_article_url,
    ingest_youtube_transcript,
    get_macro_baselines,
]

def create_macro_economist(model: BaseChatModel):
    from langchain_core.runnables import RunnableLambda
    from langchain_core.messages import AIMessage
    from schemas.macro_schemas import NarrativeContext
    
    def _run_economist(input_dict):
        # 1. Fetch data directly without relying on ReAct loop
        try:
            baseline_text = get_macro_baselines.invoke({})
        except Exception as e:
            baseline_text = f"Error fetching baselines: {e}"
            
        try:
            news_text = generate_news_radar_daily.invoke({})
            news_references = news_references_from_radar(news_text)
        except Exception as e:
            news_text = f"Error fetching news: {e}"
            news_references = []
            
        try:
            from tools.knowledge.youtube_monitor import load_recent_youtube_insights
            from core.logger import get_logger
            youtube_text = load_recent_youtube_insights(lookback_days=14, max_chars=15_000)
            if youtube_text:
                n_clips = youtube_text.count("[") if "[" in youtube_text else 1
                get_logger(__name__).info(f"Loaded youtube clips ({n_clips} blocks, {len(youtube_text)} chars)")
            youtube_section = f"\n\n=== YouTube Analyst Insights ===\n{youtube_text}" if youtube_text else ""
            youtube_references = recent_youtube_references(lookback_days=14)
        except Exception as e:
            from core.logger import get_logger
            get_logger(__name__).warning("Error loading youtube insights: %s", e)
            youtube_section = f"\n\n=== YouTube Analyst Insights ===\nError fetching YouTube insights: {e}"
            youtube_references = []
            
        quant_score = input_dict.get("quant_score") or {}
        sector_context = input_dict.get("sector_rotation_context")
        sector_context_status = input_dict.get("sector_context_status")
        quant_context = json.dumps(quant_score, ensure_ascii=False) if quant_score else "Unavailable"
        sector_text = json.dumps(
            {"status": sector_context_status or {"status": "unavailable", "reason": "no_sector_context"},
             "snapshot_facts": sector_context},
            ensure_ascii=False,
        )
        context = (
            f"=== Baseline ===\n{baseline_text}\n\n=== News ===\n{news_text}{youtube_section}"
            f"\n\n=== Macro Quant Findings ===\n{quant_context}"
            f"\n\n=== Deterministic Sector Rotation Snapshot ===\n{sector_text}"
        )
        
        # 2. Use structured output to force correct schema
        structured = model.with_structured_output(NarrativeContext)
        
        harness = get_harness("economist")
        res = structured.invoke([
            {"role": "system", "content": harness.get_system_prompt()},
            {"role": "user", "content": harness.get_skill_text("HUMAN.md", context=context)}
        ])
        res.report_references = news_references + youtube_references

        snapshot_raw = input_dict.get("sector_rotation_snapshot")
        if snapshot_raw:
            from schemas.macro_schemas import SectorAnalysis
            from schemas.sector_rotation_schemas import SectorRotationSnapshot
            from tools.macro.sector_rotation.domain.claims import resolve_sector_claims

            snapshot = SectorRotationSnapshot.model_validate(snapshot_raw)
            analysis = res.sector_analysis or SectorAnalysis()
            has_sector_data = snapshot.benchmark_status == "available" and any(
                row.status != "unavailable" for row in snapshot.rows
            )
            complete_sector_data = has_sector_data and all(row.status == "available" for row in snapshot.rows)
            analysis.analysis_status = "available" if complete_sector_data else "limited" if has_sector_data else "unavailable"
            analysis.unavailable_reason = None if has_sector_data else "sector_history_unavailable"
            valid_macro_refs: set[str] = set()
            for raw_observable in (quant_score.get("market_observables") or []):
                if isinstance(raw_observable, dict) and raw_observable.get("observable_id"):
                    try:
                        from schemas.macro_schemas import MarketObservable
                        observable = MarketObservable.model_validate(raw_observable)
                        if observable.is_valid:
                            valid_macro_refs.add(observable.observable_id)
                    except Exception:
                        continue
            resolved, valid_conditions, valid_claims, rejected = resolve_sector_claims(
                snapshot,
                analysis.fact_claims,
                analysis.watch_conditions,
                valid_macro_refs=valid_macro_refs,
            )
            analysis.snapshot_id = snapshot.snapshot_id
            analysis.as_of_date = snapshot.as_of_date
            analysis.resolved_metrics = resolved
            analysis.fact_claims = valid_claims
            analysis.watch_conditions = valid_conditions
            analysis.validation_warnings = [*analysis.validation_warnings, *rejected]
            if not has_sector_data:
                analysis.fact_claims = []
                analysis.watch_conditions = []
                analysis.summary_th = "ข้อมูล Sector Rotation รอบนี้ไม่เพียงพอ จึงไม่นำสัญญาณ sector มาใช้ในการสรุป Macro"
            elif not complete_sector_data:
                analysis.validation_warnings.append("partial_sector_coverage")
            res.sector_analysis = analysis
        elif sector_context_status:
            from schemas.macro_schemas import SectorAnalysis

            reason = str(sector_context_status.get("reason") or "sector_snapshot_unavailable")
            res.sector_analysis = SectorAnalysis(
                analysis_status="unavailable",
                unavailable_reason=reason,
                summary_th="ข้อมูล Sector Rotation รอบนี้ยังไม่พร้อม ระบบจึงใช้ข้อมูล macro ส่วนอื่นต่อโดยไม่อ้างข้อเท็จจริงด้าน sector",
            )
        
        return {"messages": [AIMessage(content=res.model_dump_json(), name="macro_economist")]}

    return RunnableLambda(_run_economist)
