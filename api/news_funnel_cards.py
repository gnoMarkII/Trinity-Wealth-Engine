"""Upsert การ์ด Kanban สำหรับรอบอนุมัติ News Funnel

ชั้น API เป็นเจ้าของ state_db และ column vocabulary ของ Kanban — เดิม logic นี้อยู่ใน
tools/macro/news_funnel.py ซึ่งทำให้ชั้น tools ผูกกับ SQLite ของ Web UI
caller ที่รัน synthesize แบบ scheduled (CLI) เรียกฟังก์ชันนี้เมื่อได้ status = require_kanban_approval

หมายเหตุ: Discord notification ของ News Funnel ไม่ได้ยิงจากจุดนี้อีกต่อไป — ย้ายไปส่งตอน
synthesize เสร็จ (ดู tools/macro/news_funnel.py + core/discord_notifier.send_synthesized_news_discord)
เพื่อให้ส่งได้เนื้อหาสังเคราะห์เต็มแทนที่จะเป็นแค่สรุปสั้นตอนรออนุมัติ และลด double-notification
"""
from typing import Any, Dict, List

from application.macro.card_service import NewsFunnelCardApplicationService
from api.db.legacy_adapter import LegacyNewsFunnelCardAdapter, NewsFunnelPromptAdapter
from core.logger import get_logger

logger = get_logger(__name__)


def upsert_news_funnel_card(period: str, pending_events: List[Dict[str, Any]]) -> None:
    """สร้างหรืออัปเดตการ์ด Kanban ของรอบ (period) ปัจจุบันด้วยรายการข่าว pending ล่าสุด

    ความล้มเหลว (เช่นไม่มีไฟล์ DB) แค่ log warning — ไม่ทำให้ scheduled run ล้ม
    """
    try:
        NewsFunnelCardApplicationService(
            storage=LegacyNewsFunnelCardAdapter(),
            prompt=NewsFunnelPromptAdapter(),
        ).upsert(period, pending_events)
        logger.info("Created/updated News Funnel Kanban card for %d pending items.", len(pending_events))
    except Exception as e:
        logger.warning("Could not create/update Kanban card in SQLite state_db: %s", e)
