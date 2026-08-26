"""Centralized LLM model configuration registry — single source of truth สำหรับทุก
env var ที่เลือก model ทั้งใน agents/ (orchestration) และ tools/ (extraction/generation)

ใช้ get_model_name() ได้เฉพาะ slot ที่เรียก get_llm() ตรงๆ (agent layer ทั้งหมด + extractor)
ห้ามใช้กับ slot ที่ผ่าน invoke_structured_llm (youtube_pitch, news_triage,
thai_title_translation) เพราะ invoke_structured_llm รับ model_env เป็น "ชื่อ env var"
ไม่ใช่ "ชื่อ model ที่ resolve แล้ว" — ให้ดึง REGISTRY[key].env_var / REGISTRY[key].default
ไปส่งตรงๆ แทนสำหรับ 3 slot นั้น
"""
from dataclasses import dataclass
import os


@dataclass(frozen=True)
class ModelSlot:
    env_var: str
    default: str
    purpose: str
    layer: str  # "agent" | "tool"


REGISTRY: dict[str, ModelSlot] = {
    # --- Agent Layer ---
    "manager":       ModelSlot("MANAGER_MODEL",             "gemini-3.1-flash-lite-preview", "Manager Agent — routing and orchestration", "agent"),
    "router":        ModelSlot("ROUTER_MODEL",               "gemini-3.1-flash-lite-preview", "Structured routing (RouterDecision)", "agent"),
    "archivist":     ModelSlot("ARCHIVIST_MODEL",             "gemini-3.1-flash-lite-preview", "PKM management agent", "agent"),
    "bookkeeper":    ModelSlot("BOOKKEEPER_MODEL",            "gemini-3.1-flash-lite-preview", "Portfolio & accounting agent", "agent"),
    "macro_quant":   ModelSlot("MACRO_QUANT_MODEL",           "gemini-3.1-flash-lite-preview", "Quant Macro Matrix agent", "agent"),
    "economist":     ModelSlot("MACRO_ECONOMIST_MODEL",       "gemini-3.1-flash-lite-preview", "Macroeconomic narrative synthesis agent", "agent"),
    "allocator":     ModelSlot("STRATEGIC_ALLOCATOR_MODEL",   "gemini-3.1-flash-lite-preview", "Strategic Allocator agent", "agent"),
    "equity_quant":       ModelSlot("EQUITY_QUANT_MODEL",       "gemini-3.1-flash-lite-preview", "Equity Quant Signals agent (deterministic)", "agent"),
    "equity_narrative":   ModelSlot("EQUITY_NARRATIVE_MODEL",   "gemini-3.1-flash-lite-preview", "Equity Sentiment/Narrative synthesis agent", "agent"),
    "equity_synthesizer": ModelSlot("EQUITY_SYNTHESIZER_MODEL", "gemini-3.1-flash-lite-preview", "Equity final report synthesis agent", "agent"),
    # --- Tool Layer ---
    "extractor":              ModelSlot("EXTRACTOR_MODEL",              "gemini-3.1-flash-lite-preview", "Article/PDF/YouTube content extraction", "tool"),
    "youtube_pitch":           ModelSlot("YOUTUBE_PITCH_MODEL",          "gemini-3.1-flash-lite-preview", "YouTube Pitch generation + Briefing Book", "tool"),
    "news_triage":             ModelSlot("NEWS_FUNNEL_TRIAGE_MODEL",     "gemini-3.1-flash-lite-preview", "News impact scoring (batch triage)", "tool"),
    "thai_title_translation":  ModelSlot("NEWS_FUNNEL_SYNTHESIS_MODEL",  "gemini-3.1-flash-lite-preview", "Thai title translation for news", "tool"),
    "earnings_call_summarizer": ModelSlot("EARNINGS_CALL_MODEL",          "gemini-3.1-flash-lite-preview", "Earnings Call Transcript Summarizer", "tool"),
}


def get_model_name(slot_key: str) -> str:
    """Resolve model name from env var with fallback to default."""
    slot = REGISTRY[slot_key]
    return os.getenv(slot.env_var, slot.default)


def get_registry_summary() -> list[dict]:
    """Return all slots with resolved values — สำหรับ /api/debug/models"""
    return [
        {
            "slot": key,
            "env_var": slot.env_var,
            "resolved_model": os.getenv(slot.env_var, slot.default),
            "default": slot.default,
            "is_overridden": os.getenv(slot.env_var) is not None,
            "purpose": slot.purpose,
            "layer": slot.layer,
        }
        for key, slot in REGISTRY.items()
    ]
