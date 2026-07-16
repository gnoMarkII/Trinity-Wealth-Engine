"""Upsert การ์ด Kanban สำหรับรอบอนุมัติ News Funnel

ชั้น API เป็นเจ้าของ state_db และ column vocabulary ของ Kanban — เดิม logic นี้อยู่ใน
tools/macro/news_funnel.py ซึ่งทำให้ชั้น tools ผูกกับ SQLite ของ Web UI
caller ที่รัน synthesize แบบ scheduled (CLI) เรียกฟังก์ชันนี้เมื่อได้ status = require_kanban_approval
"""
import uuid
from contextlib import closing
from typing import Any, Dict, List

from api import state_db
from core.logger import get_logger
from tools.macro.news_funnel import format_news_funnel_card_prompt

logger = get_logger(__name__)


def upsert_news_funnel_card(period: str, pending_events: List[Dict[str, Any]]) -> None:
    """สร้างหรืออัปเดตการ์ด Kanban ของรอบ (period) ปัจจุบันด้วยรายการข่าว pending ล่าสุด

    ความล้มเหลว (เช่นไม่มีไฟล์ DB) แค่ log warning — ไม่ทำให้ scheduled run ล้ม
    """
    try:
        with closing(state_db.get_connection()) as conn:
            card_title = f"[{period.upper()}] News Funnel High-Impact ({len(pending_events)} items)"
            formatted_prompt = format_news_funnel_card_prompt(period, pending_events)
            existing_cards = state_db.list_kanban_cards(conn)
            existing_card = next(
                (c for c in existing_cards if c["flow"] == "news_funnel" and c["column_name"] in ("backlog", "approval") and period.upper() in c["title"]),
                None
            )
            if existing_card is None:
                state_db.create_kanban_card(
                    conn,
                    card_id=str(uuid.uuid4()),
                    title=card_title,
                    column_name="backlog",
                    flow="news_funnel",
                    prompt=formatted_prompt,
                    scope="both",
                )
            else:
                state_db.update_kanban_card(
                    conn,
                    card_id=existing_card["card_id"],
                    title=card_title,
                    prompt=formatted_prompt,
                    flow="news_funnel",
                    scope=existing_card["scope"] or "both",
                )
        logger.info("Created/updated News Funnel Kanban card for %d pending items.", len(pending_events))
    except Exception as e:
        logger.warning("Could not create/update Kanban card in SQLite state_db: %s", e)
