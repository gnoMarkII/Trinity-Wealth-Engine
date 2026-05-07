import os

import anthropic
import google.genai as genai
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableWithFallbacks
from langchain_google_genai import ChatGoogleGenerativeAI

_GOOGLE_FALLBACK_MODEL = "gemini-2.5-flash"


def get_llm(
    provider: str,
    model_name: str,
    temperature: float = 0.0,
    use_fallback: bool = False,
) -> BaseChatModel | RunnableWithFallbacks:
    """สร้าง LLM instance ตาม provider ที่กำหนด

    Args:
        provider: "anthropic" หรือ "google"
        model_name: ชื่อ model เช่น "gemini-2.5-flash", "claude-sonnet-4-6"
        temperature: ระดับความสร้างสรรค์ (0.0 = deterministic)
        use_fallback: True = ผูก fallback model ไว้ (Google: gemini-2.5-flash)
                      ใช้กับ .invoke()/.stream() โดยตรง
                      ถ้าต้องการ with_structured_output ให้สร้าง fallback chain แยก
    """
    if provider == "google":
        primary = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
        if use_fallback and model_name != _GOOGLE_FALLBACK_MODEL:
            fallback = ChatGoogleGenerativeAI(model=_GOOGLE_FALLBACK_MODEL, temperature=temperature)
            return primary.with_fallbacks([fallback])
        return primary
    if provider == "anthropic":
        return ChatAnthropic(model=model_name, temperature=temperature)
    raise ValueError(f"Unknown provider '{provider}'. Choose 'anthropic' or 'google'.")


def _fetch_google_models() -> list[str]:
    try:
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        return [
            m.name
            for m in client.models.list()
            if "gemini" in m.name.lower()
        ]
    except Exception as e:
        print(f"[llm_factory] Google fetch error: {e}")
        return []


def _fetch_anthropic_models() -> list[str]:
    try:
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        return [m.id for m in client.models.list().data]
    except Exception as e:
        print(f"[llm_factory] Anthropic fetch error: {e}")
        return []


def list_available_models(provider: str | None = None) -> list[str] | dict[str, list[str]]:
    """ดึงรายชื่อโมเดลจาก Server ของ Provider โดยตรง (Dynamic Fetch)

    Args:
        provider: "anthropic", "google" หรือ None เพื่อดึงทั้งสองค่าย

    Returns:
        list[str] ถ้าระบุ provider / dict[str, list[str]] ถ้าไม่ระบุ
    """
    if provider == "google":
        return _fetch_google_models()
    if provider == "anthropic":
        return _fetch_anthropic_models()
    if provider is None:
        return {
            "google": _fetch_google_models(),
            "anthropic": _fetch_anthropic_models(),
        }
    raise ValueError(f"Unknown provider '{provider}'. Choose 'anthropic', 'google', or None.")
