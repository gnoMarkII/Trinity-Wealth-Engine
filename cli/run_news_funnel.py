"""Unified CLI Runner สำหรับระบบ News Funnel Architecture และ Obsidian Linking Key

รองรับทั้ง positional command และ --mode flag:
- ingest : ดึงข่าว คัดกรอง Triage และบันทึกลง Persistent JSON Store
- synthesize : สังเคราะห์ข่าวตามรอบที่ระบุ (--period morning|evening|auto)
- both / auto : ทำทั้ง ingest และ synthesize (สำหรับ mode=auto จะเช็คช่วงเวลา 6-12 และ 17-23 สำหรับการสังเคราะห์)
"""
import argparse
from datetime import datetime
import sys
import os

# เพิ่ม root directory ลงใน sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from core.logger import get_logger, setup_logging
from tools.macro.news_funnel import (
    get_synthesis_period,
    run_news_funnel_ingest,
    run_news_funnel_synthesize,
)

logger = get_logger("run_news_funnel")


def _handle_synthesize_result(res: dict) -> None:
    """แสดงผล synthesize และ upsert การ์ด Kanban เมื่อรอบนี้ต้องรออนุมัติผ่าน Web UI"""
    if res.get("status") == "require_kanban_approval":
        from api.news_funnel_cards import upsert_news_funnel_card
        upsert_news_funnel_card(res.get("period") or get_synthesis_period(), res.get("pending_events") or [])
        logger.info("🔔 [Kanban Gated] %s", res.get("message"))
    else:
        logger.info("Synthesize result: %s", res)


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="News Funnel Unified Runner")
    parser.add_argument(
        "command",
        nargs="?",
        choices=["ingest", "synthesize", "both", "auto"],
        default=None,
        help="คำสั่งหลัก: ingest, synthesize, both หรือ auto",
    )
    parser.add_argument(
        "--mode",
        choices=["ingest", "synthesize", "both", "auto"],
        default=None,
        help="โหมดการทำงาน: ingest, synthesize, both หรือ auto",
    )
    parser.add_argument(
        "--period",
        choices=["morning", "evening", "auto"],
        default="auto",
        help="รอบเวลาสำหรับการสังเคราะห์ (morning, evening หรือ auto)",
    )
    parser.add_argument(
        "--store-path",
        default=None,
        help="เส้นทางไฟล์ State Store (ถ้าไม่ระบุจะใช้ค่าเริ่มต้น)",
    )
    parser.add_argument(
        "--vault-root",
        default=None,
        help="เส้นทางโฟลเดอร์รากของ Obsidian Vault",
    )
    parser.add_argument(
        "--force-autonomous",
        action="store_true",
        help="อนุญาตให้สังเคราะห์โดยอัตโนมัติ (ข้ามการรอกดอนุมัติผ่าน Kanban สำหรับเทสต์หรือดีบักเท่านั้น)",
    )
    args = parser.parse_args()

    mode = args.command or args.mode or "both"
    period = args.period
    if period == "auto":
        period = get_synthesis_period()

    if mode == "ingest":
        logger.info("Running News Funnel Ingestion...")
        res = run_news_funnel_ingest(store_path=args.store_path)
        logger.info("Ingest result: %s", res)

    elif mode == "synthesize":
        logger.info("Running News Funnel Synthesis for period: %s...", period)
        res = run_news_funnel_synthesize(
            period=period,
            store_path=args.store_path,
            vault_root=args.vault_root,
            allow_autonomous=args.force_autonomous,
        )
        _handle_synthesize_result(res)

    elif mode in ("both", "auto"):
        logger.info("Running News Funnel Pipeline (Mode: %s)...", mode)
        ingest_res = run_news_funnel_ingest(store_path=args.store_path)
        logger.info("Ingest result: %s", ingest_res)

        hour = datetime.now().hour
        if mode == "both" or (6 <= hour < 12 or 17 <= hour <= 23):
            synth_res = run_news_funnel_synthesize(
                period=period,
                store_path=args.store_path,
                vault_root=args.vault_root,
                allow_autonomous=args.force_autonomous,
            )
            _handle_synthesize_result(synth_res)
        else:
            logger.info("Current hour %02d is outside scheduled auto windows. Only ingest executed.", hour)


if __name__ == "__main__":
    main()

