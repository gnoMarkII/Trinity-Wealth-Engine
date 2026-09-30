import logging
import logging.handlers
import os
import sys
import time
from pathlib import Path


_SETUP_DONE = False

class WindowsSafeTimedRotatingFileHandler(logging.handlers.TimedRotatingFileHandler):
    """TimedRotatingFileHandler ที่ป้องกัน PermissionError บน Windows
    
    เมื่อมีหลาย Process เปิด system.log พร้อมกัน (เช่น uvicorn backend + CLI script)
    Windows จะล็อกไฟล์ทำให้ os.rename หรือ os.remove ล้มเหลวด้วย [WinError 32]
    คลาสนี้จะดักจับ PermissionError เพื่อให้ process อื่นยังเขียน log ต่อไปได้โดยไม่ crash
    """

    def rotate(self, source: str, dest: str) -> None:
        try:
            super().rotate(source, dest)
        except (PermissionError, OSError):
            pass

    def doRollover(self) -> None:
        try:
            super().doRollover()
        except (PermissionError, OSError):
            # บน Windows หากไฟล์ถูกล็อกโดย process อื่น ให้ข้ามการ rollover ไปก่อน
            # และเปิด stream ใหม่หากถูกปิดไปแล้ว
            if self.stream is None:
                self.stream = self._open()
            currentTime = int(time.time())
            self.rolloverAt = self.computeRollover(currentTime)


def setup_logging(level: str | None = None) -> None:
    """ตั้งค่า root logger — เรียกครั้งเดียวจาก main entry point
    Level ดึงจาก LOG_LEVEL env (default: INFO) หรือ override ผ่าน argument
    """
    global _SETUP_DONE
    if _SETUP_DONE:
        return

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
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)

    # เพิ่ม file handler — capture WARNING+ ลง system.log
    log_dir = Path(__file__).resolve().parents[1] / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "system.log"
    
    file_handler = WindowsSafeTimedRotatingFileHandler(
        log_file, when="midnight", interval=1, backupCount=7, encoding="utf-8"
    )
    file_handler.setLevel(logging.WARNING)
    formatter = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    file_handler.setFormatter(formatter)
    
    logging.getLogger().addHandler(file_handler)
    _SETUP_DONE = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
