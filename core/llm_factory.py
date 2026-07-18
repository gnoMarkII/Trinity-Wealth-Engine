import os
from typing import Optional, Any

import anthropic
import google.genai as genai
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableWithFallbacks
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from core.logger import get_logger

log = get_logger(__name__)

# Cross-provider fallback — รุ่นที่ถูกเรียกเมื่อ primary fail ใน get_llm(use_fallback=True)
FALLBACK_MODEL = "openai/gpt-oss-120b:free"


def detect_provider(model_name: str) -> str:
    """Auto-detect provider จาก model name (override ได้ผ่าน LLM_PROVIDER env)

    Rules:
      - claude-*              → anthropic
      - gemini-* / models/gemini-*  → google
      - มี '/' ใน name         → openrouter
      - default               → google
    """
    override = os.getenv("LLM_PROVIDER")
    if override:
        return override
    name = model_name.lower()
    if name.startswith("claude"):
        return "anthropic"
    if name.startswith(("gemini", "models/gemini")):
        return "google"
    if "/" in model_name:
        return "openrouter"
    return "google"


def _build_primary(provider: str, model_name: str, temperature: float, max_output_tokens: Optional[int] = None) -> BaseChatModel:
    if provider == "google":
        return ChatGoogleGenerativeAI(model=model_name, temperature=temperature, max_output_tokens=max_output_tokens, api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
    if provider == "anthropic":
        return ChatAnthropic(model=model_name, temperature=temperature, max_tokens=max_output_tokens)
    if provider == "openrouter":
        return ChatOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            model=model_name,
            temperature=temperature,
            max_tokens=max_output_tokens,
        )
    raise ValueError(f"Unknown provider '{provider}'. Choose 'anthropic', 'google', or 'openrouter'.")


def get_llm(
    provider: str,
    model_name: str,
    temperature: float = 0.0,
    use_fallback: bool = False,
    max_output_tokens: Optional[int] = None,
) -> BaseChatModel | RunnableWithFallbacks:
    """สร้าง LLM instance ตาม provider — เลือก wrap ด้วย cross-provider fallback ได้

    Args:
        provider: "google", "anthropic" หรือ "openrouter"
        model_name: ชื่อ model เช่น "gemini-3.1-flash-lite-preview", "claude-sonnet-4-6",
                    "openai/gpt-oss-120b:free"
        temperature: ระดับความสร้างสรรค์ (0.0 = deterministic)
        use_fallback: True = wrap primary ด้วย FALLBACK_MODEL (ข้าม provider ได้)
                      ใช้กับ .invoke()/.stream() ตรงๆ
                      สำหรับ with_structured_output ให้สร้าง fallback chain เองภายนอก
        max_output_tokens: จำนวน token สูงสุดในการตอบกลับ (ใช้สำหรับควบคุม token length / ป้องกัน JSON truncate)
    """
    primary = _build_primary(provider, model_name, temperature, max_output_tokens=max_output_tokens)

    if use_fallback and model_name != FALLBACK_MODEL:
        fallback_provider = detect_provider(FALLBACK_MODEL)
        fallback = _build_primary(fallback_provider, FALLBACK_MODEL, temperature, max_output_tokens=max_output_tokens)
        return primary.with_fallbacks([fallback])

    return primary


def invoke_structured_llm(
    schema: Any,
    model_env: str,
    prompt_lines: list[str],
    purpose: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
    default_model: str = "gemini-2.5-flash",
    provider: str = "google",
    **kwargs: Any,
) -> Any:
    """Helper สำหรับสร้างและเรียกใช้ structured LLM ด้วย schema ที่กำหนด

    Args:
        schema: Pydantic schema class
        model_env: ชื่อ Environment variable สำหรับดึงชื่อ model
        prompt_lines: รายการบรรทัดของ Prompt ที่จะส่งให้ LLM
        purpose: คำอธิบายจุดประสงค์ของ call เพื่อใช้ใน log
        max_output_tokens: จำนวน token สูงสุดในการตอบกลับ
        default_model: ค่าเริ่มต้นของ model หากไม่ได้ตั้งใน env var
        provider: "google", "anthropic" หรือ "openrouter"
    """
    model_name = os.getenv(model_env, default_model)
    call_purpose = purpose or getattr(schema, "__name__", str(schema))
    log.info("LLM Call | purpose=%s | model=%s | max_tokens=%s", call_purpose, model_name, max_output_tokens)
    llm = get_llm(provider=provider, model_name=model_name, max_output_tokens=max_output_tokens)
    structured_llm = llm.with_structured_output(schema)
    return structured_llm.invoke("\n".join(prompt_lines))


def _fetch_google_models() -> list[str]:
    try:
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        return [
            m.name
            for m in client.models.list()
            if "gemini" in m.name.lower()
        ]
    except Exception as e:
        log.warning("Google models fetch failed: %s", e)
        return []


def _fetch_anthropic_models() -> list[str]:
    try:
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        return [m.id for m in client.models.list().data]
    except Exception as e:
        log.warning("Anthropic models fetch failed: %s", e)
        return []


def _fetch_openrouter_models() -> list[str]:
    try:
        import httpx
        api_key = os.getenv("OPENROUTER_API_KEY")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        resp = httpx.get("https://openrouter.ai/api/v1/models", headers=headers, timeout=10)
        resp.raise_for_status()
        return [m["id"] for m in resp.json().get("data", [])]
    except Exception as e:
        log.warning("OpenRouter models fetch failed: %s", e)
        return []


def list_available_models(provider: str | None = None) -> list[str] | dict[str, list[str]]:
    """ดึงรายชื่อโมเดลจาก Server ของ Provider โดยตรง (Dynamic Fetch)

    Args:
        provider: "google", "anthropic", "openrouter" หรือ None เพื่อดึงทั้งหมด

    Returns:
        list[str] ถ้าระบุ provider / dict[str, list[str]] ถ้าไม่ระบุ
    """
    if provider == "google":
        return _fetch_google_models()
    if provider == "anthropic":
        return _fetch_anthropic_models()
    if provider == "openrouter":
        return _fetch_openrouter_models()
    if provider is None:
        return {
            "google": _fetch_google_models(),
            "anthropic": _fetch_anthropic_models(),
            "openrouter": _fetch_openrouter_models(),
        }
    raise ValueError(f"Unknown provider '{provider}'. Choose 'google', 'anthropic', 'openrouter', or None.")
