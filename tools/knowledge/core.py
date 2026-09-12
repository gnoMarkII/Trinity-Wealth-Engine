import os  # ไม่ได้เรียก os.* ตรงๆ ในไฟล์นี้แล้ว (model/prompt config ย้ายไป model_registry.py/prompt_harness.py)
           # แต่ tests/tools/knowledge/test_core.py patch ผ่าน "tools.knowledge.core.os.environ.get" ไว้
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

import httpx

from core.llm_factory import get_llm, detect_provider
from core.logger import get_logger
from core.model_registry import get_model_name
from core.prompt_harness import TOOLS_PROMPTS_ROOT, get_harness
from core.security import anonymize_pii

log = get_logger(__name__)

_CONTENT_CHAR_LIMIT = 20_000

@lru_cache(maxsize=1)
def _get_extractor_llm():
    """Cache LLM + retry wrapper — สร้างครั้งเดียวต่อ process"""
    extractor_model = get_model_name("extractor")
    provider = detect_provider(extractor_model)
    return get_llm(provider=provider, model_name=extractor_model).with_retry(
        retry_if_exception_type=(
            httpx.TimeoutException,
            httpx.ConnectError,
            httpx.RemoteProtocolError,
            TimeoutError,
            ConnectionError,
        ),
        wait_exponential_jitter=True,
        stop_after_attempt=3,
    )


def _extractor_system_prompt() -> str:
    return get_harness("extractor", skills_root=TOOLS_PROMPTS_ROOT).get_system_prompt()


def _call_extractor_llm(raw_content: str, source_label: str) -> str:
    """เรียก LLM สกัดข้อมูลการลงทุน — shared logic สำหรับทั้ง URL, PDF, และ YouTube Transcript"""
    raw_content, _ = anonymize_pii(raw_content)

    if len(raw_content) > _CONTENT_CHAR_LIMIT:
        raw_content = raw_content[:_CONTENT_CHAR_LIMIT] + f"\n...[ตัดทอน — เนื้อหาเกิน {_CONTENT_CHAR_LIMIT:,} ตัวอักษร]"

    log.info("LLM Call | purpose=article_extraction | model=%s", get_model_name("extractor"))

    response = _get_extractor_llm().invoke([
        {"role": "system", "content": _extractor_system_prompt()},
        {"role": "user", "content": f"Source: {source_label}\n\nContent:\n\n{raw_content}"},
    ])
    
    content = response.content
    if isinstance(content, list):
        content = content[0].get("text", "") if len(content) > 0 and isinstance(content[0], dict) else str(content)
    
    return str(content).strip()


import yaml

def _build_article_md(
    extracted: str,
    source_url: str,
    title: str,
    today: str,
    now_time: str,
    image: str | None = None,
    *,
    event_id: str | None = None,
    extracted_tickers: list[str] | None = None,
    extracted_themes: list[str] | None = None,
    published_at: str | None = None,
    canonical_publisher: str | None = None,
    canonical_url: str | None = None,
    verification_status: str | None = None,
    related_entities: list[dict[str, str]] | None = None,
) -> str:
    safe_title = title.replace(":", " -").replace("/", "-")[:80]
    meta_dict = {
        "schema_version": 2,
        "title": safe_title,
        "entity_type": "article_note",
        "source_url": source_url,
        "publisher": urlparse(source_url).netloc.replace("www.", ""),
    }
    if image:
        meta_dict["image"] = image
    meta_dict["date"] = today
    meta_dict["last_updated"] = now_time
    meta_dict["tags"] = ["article", "investment_insight"]
    if event_id is not None:
        meta_dict["event_id"] = event_id
    if extracted_tickers is not None:
        meta_dict["extracted_tickers"] = extracted_tickers
    if extracted_themes is not None:
        meta_dict["extracted_themes"] = extracted_themes
    if published_at is not None:
        meta_dict["published_at"] = published_at
    if canonical_publisher is not None:
        meta_dict["canonical_publisher"] = canonical_publisher
    if canonical_url is not None:
        meta_dict["canonical_url"] = canonical_url
    if verification_status is not None:
        meta_dict["verification_status"] = verification_status
    if related_entities is not None:
        meta_dict["related_entities"] = related_entities

    yaml_block = yaml.safe_dump(meta_dict, allow_unicode=True, sort_keys=False).strip()
    return "\n".join([
        "---",
        yaml_block,
        "---",
        "",
        f"# {safe_title}",
        f"> แหล่งที่มา: {source_url}",
        "",
        extracted,
        "",
        "## หมายเหตุ",
        "",
        "> สกัดข้อมูลจากบทความผ่าน LLM — ตรวจสอบความถูกต้องก่อนนำไปใช้ตัดสินใจลงทุน",
        "",
    ])
