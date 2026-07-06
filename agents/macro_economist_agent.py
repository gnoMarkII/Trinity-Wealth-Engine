from langchain_core.language_models.chat_models import BaseChatModel
from langchain.agents import create_agent
from tools.macro.news_radar import generate_news_radar_daily
from tools.knowledge.youtube_monitor import generate_weekly_youtube_digest
from tools.knowledge.article import ingest_article_url
from tools.knowledge.youtube import ingest_youtube_transcript
from tools.macro.baselines import get_macro_baselines
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
        except Exception as e:
            news_text = f"Error fetching news: {e}"
            
        try:
            from tools.knowledge.youtube_monitor import load_recent_youtube_insights
            from core.logger import get_logger
            youtube_text = load_recent_youtube_insights(lookback_days=14, max_chars=15_000)
            if youtube_text:
                n_clips = youtube_text.count("[") if "[" in youtube_text else 1
                get_logger(__name__).info(f"Loaded youtube clips ({n_clips} blocks, {len(youtube_text)} chars)")
            youtube_section = f"\n\n=== YouTube Analyst Insights ===\n{youtube_text}" if youtube_text else ""
        except Exception as e:
            from core.logger import get_logger
            get_logger(__name__).warning("Error loading youtube insights: %s", e)
            youtube_section = f"\n\n=== YouTube Analyst Insights ===\nError fetching YouTube insights: {e}"
            
        context = f"=== Baseline ===\n{baseline_text}\n\n=== News ===\n{news_text}{youtube_section}"
        
        # 2. Use structured output to force correct schema
        structured = model.with_structured_output(NarrativeContext)
        
        harness = get_harness("economist")
        res = structured.invoke([
            {"role": "system", "content": harness.get_system_prompt()},
            {"role": "user", "content": harness.get_skill_text("HUMAN.md", context=context)}
        ])
        
        return {"messages": [AIMessage(content=res.model_dump_json(), name="macro_economist")]}

    return RunnableLambda(_run_economist)
