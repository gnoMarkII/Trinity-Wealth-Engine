"""Archivist task preparation and verification helpers.

Architecture Layer: agents/pipelines/
Extracted from manager_agent.py to isolate Archivist prompt engineering,
filename directive injection, and verification specification.
"""
from __future__ import annotations

from typing import Any, Callable, Optional
from langchain_core.messages import AIMessage, HumanMessage
from core.utils import normalize_content
from tools.archivist.parser import extract_yaml_frontmatter_value


def build_archivist_task_instruction(
    state: dict[str, Any],
    *,
    extract_reply_fn: Callable[[list[Any]], str],
    has_frontmatter_fn: Callable[[str], bool],
) -> str:
    """Builds a deterministic, directive-rich prompt for the Archivist worker."""
    meta = state.get("route_meta") or {}
    source = meta.get("source")

    if source == "macro_intel":
        last_msg = extract_reply_fn(state.get("messages") or [])
        title = extract_yaml_frontmatter_value(last_msg, "title")
        filename_directive = (
            f"\nต้องใช้ filename='{title}' เป๊ะๆ ตามนี้เท่านั้น ห้ามดัดแปลง เปลี่ยนช่องว่างเป็นขีดล่าง "
            f"หรือเปลี่ยนรูปแบบใดๆ ทั้งสิ้น (เพื่อให้ re-run วันเดียวกัน overwrite ไฟล์เดิมแทนสร้างไฟล์ซ้ำ)"
            if title else ""
        )
        return (
            f"คุณต้องเรียกใช้เครื่องมือ write_raw_markdown เพื่อบันทึกข้อมูลดิบต่อไปนี้ลง Vault ทันที "
            f"ห้ามตอบกลับเป็นข้อความโดยไม่เรียกใช้เครื่องมือ{filename_directive}\n\n[ข้อมูลดิบ]\n{last_msg}"
        )

    if source == "equity_intel":
        last_msg = extract_reply_fn(state.get("messages") or [])
        report_title = extract_yaml_frontmatter_value(last_msg, "title")

        news_raw = state.get("equity_news_raw")
        valid_news = (
            isinstance(news_raw, str)
            and news_raw.lstrip().startswith("---")
            and bool(extract_yaml_frontmatter_value(news_raw, "title"))
        )

        if valid_news:
            news_title = extract_yaml_frontmatter_value(news_raw, "title")
            news_directive = (
                f"filename='{news_title}' เป๊ะๆ ตามนี้เท่านั้น ห้ามดัดแปลง เปลี่ยนช่องว่างเป็นขีดล่าง หรือเปลี่ยนรูปแบบใดๆ ทั้งสิ้น"
                if news_title else ""
            )
            report_directive = (
                f"filename='{report_title}' เป๊ะๆ ตามนี้เท่านั้น ห้ามดัดแปลง เปลี่ยนช่องว่างเป็นขีดล่าง หรือเปลี่ยนรูปแบบใดๆ ทั้งสิ้น"
                if report_title else ""
            )
            directives = (
                f"\nคุณต้องเรียกใช้เครื่องมือ write_raw_markdown ทั้งหมด 2 ครั้งเพื่อบันทึกทั้งข่าวหุ้นและบทวิเคราะห์หุ้นดังนี้:\n"
                f"1. บันทึกข่าวหุ้นล่าสุดด้วย {news_directive} (เพื่อให้ re-run วันเดียวกัน overwrite ไฟล์เดิมแทนสร้างไฟล์ซ้ำ)\n"
                f"[ข้อมูลดิบ - ข่าวหุ้นล่าสุด]\n{news_raw}\n\n"
                f"2. บันทึกบทวิเคราะห์หุ้นด้วย {report_directive} (เพื่อให้ re-run วันเดียวกัน overwrite ไฟล์เดิมแทนสร้างไฟล์ซ้ำ)\n"
                f"[ข้อมูลดิบ - บทวิเคราะห์หุ้น]\n{last_msg}"
            )
            return (
                f"คุณต้องเรียกใช้เครื่องมือ write_raw_markdown ทั้งหมด 2 ครั้งเพื่อบันทึกข้อมูลดิบลง Vault ทันที "
                f"ห้ามตอบกลับเป็นข้อความโดยไม่เรียกใช้เครื่องมือ{directives}"
            )
        else:
            filename_directive = (
                f"\nต้องใช้ filename='{report_title}' เป๊ะๆ ตามนี้เท่านั้น ห้ามดัดแปลง เปลี่ยนช่องว่างเป็นขีดล่าง "
                f"หรือเปลี่ยนรูปแบบใดๆ ทั้งสิ้น (เพื่อให้ re-run วันเดียวกัน overwrite ไฟล์เดิมแทนสร้างไฟล์ซ้ำ)"
                if report_title else ""
            )
            return (
                f"คุณต้องเรียกใช้เครื่องมือ write_raw_markdown เพื่อบันทึกข้อมูลดิบต่อไปนี้ลง Vault ทันที "
                f"ห้ามตอบกลับเป็นข้อความโดยไม่เรียกใช้เครื่องมือ{filename_directive}\n\n[ข้อมูลดิบ]\n{last_msg}"
            )

    msgs = list(state.get("messages") or [])
    last_msg = normalize_content(getattr(msgs[-1], "content", "")) if msgs else ""
    last_human_idx = next(
        (
            i for i in range(len(msgs) - 1, -1, -1)
            if isinstance(msgs[i], HumanMessage) and getattr(msgs[i], "name", None) != "manager"
        ),
        -1,
    )
    raw_user = normalize_content(getattr(msgs[last_human_idx], "content", "")) if last_human_idx >= 0 else ""
    mid_drain = any(
        isinstance(m, AIMessage) and normalize_content(getattr(m, "content", "")).strip()
        for m in msgs[last_human_idx + 1:-1]
    )
    if not mid_drain and has_frontmatter_fn(raw_user) and len(raw_user) > len(last_msg) * 2:
        return f"บันทึกข้อมูลดิบต่อไปนี้ลง Vault ทันที\n\n[ข้อมูลดิบ]\n{raw_user}"

    return last_msg


def resolve_archivist_verification_spec(
    route_meta: Optional[dict[str, Any]],
    task_content: str,
) -> tuple[Optional[set[str]], int]:
    """Resolves required tools and minimum invocation count for Archivist verification."""
    meta = route_meta or {}
    req_tools = (
        {"write_raw_markdown", "save_memory"}
        if meta.get("source") in ("macro_intel", "equity_intel")
        else None
    )
    min_calls = 2 if "เรียกใช้เครื่องมือ write_raw_markdown ทั้งหมด 2 ครั้ง" in (task_content or "") else 1
    return req_tools, min_calls
