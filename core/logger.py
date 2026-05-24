import logging
import os
import sys
from datetime import datetime

from core.agent_log import _ensure_file, _lock, _today_path, _truncate


class _DailyMarkdownHandler(logging.Handler):
    """Append WARNING/ERROR จากทุก Python logger ลง daily Markdown log
    Format: ### [HH:MM:SS] LEVEL [logger.name]\nmessage
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            time_str = datetime.now().strftime("%H:%M:%S")
            entry = (
                f"### [{time_str}] {record.levelname} [{record.name}]\n"
                f"{_truncate(record.getMessage(), 500)}"
            )
            path = _today_path()
            with _lock:
                _ensure_file(path)
                with path.open("a", encoding="utf-8") as f:
                    f.write(entry + "\n\n")
        except Exception:
            self.handleError(record)


def setup_logging(level: str | None = None) -> None:
    """ตั้งค่า root logger — เรียกครั้งเดียวจาก main entry point
    Level ดึงจาก LOG_LEVEL env (default: INFO) หรือ override ผ่าน argument
    """
    log_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    # ปิด noisy loggers ของ HTTP libraries
    for noisy in ("httpx", "httpcore", "urllib3", "chromadb"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # yfinance log HTTP 404 ที่ระดับ ERROR เมื่อ fallback endpoint แม้ดึงข้อมูลสำเร็จ
    # (เช่น quoteSummary fail → fallback scrape ได้ครบ) — ปิดเสียงทั้งหมดเพราะ tool
    # มี try/except + graceful Thai error message รองรับ failure ตัวจริงอยู่แล้ว
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)

    # เพิ่ม file handler — capture WARNING+ ลง daily Markdown log
    file_handler = _DailyMarkdownHandler()
    file_handler.setLevel(logging.WARNING)
    logging.getLogger().addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
