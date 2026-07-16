"""Deprecated entrypoint สำหรับระบบ News Funnel Architecture

โปรดใช้ cli/run_news_funnel.py แทน
"""
import sys
import os
import warnings

# เพิ่ม root directory ลงใน sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.logger import get_logger
from cli.run_news_funnel import main as unified_main

logger = get_logger("run_news_funnel_auto")


def main():
    warnings.warn(
        "scripts/run_news_funnel_auto.py is deprecated and will be removed in a future release. Use cli/run_news_funnel.py instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    logger.warning("run_news_funnel_auto.py is deprecated. Please switch to cli/run_news_funnel.py")
    unified_main()


if __name__ == "__main__":
    main()
