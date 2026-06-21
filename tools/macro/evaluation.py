from langsmith import traceable
import concurrent.futures

import os

from datetime import datetime

import yfinance as yf

from fredapi import Fred

from langchain_core.tools import tool

from core.logger import get_logger

from core.retry import with_retry as _with_retry


from core.logger import get_logger
log = get_logger(__name__)

from .parsers import *
from .scoring import *

import os
from pathlib import Path
from datetime import datetime
from langchain_core.tools import tool
from core.logger import get_logger
from .parsers import _parse_markdown_table_rows
from .scoring import _generate_agentic_rationales

log = get_logger(__name__)

@tool
@traceable(run_type="tool")
def evaluate_macro_matrix() -> str:
    """ประเมินข้อมูลเศรษฐกิจมหภาคและสร้างรายงาน"""
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    # Paths based on test_macro.py expectations
    vault_path = Path(os.environ.get("OBSIDIAN_VAULT_PATH", "C:/ChinoDoc/Projects/Claude/invest-agents/tests/fixtures/vault"))
    snapshots_dir = vault_path / "30_Knowledge_Base" / "Macroeconomics" / "Daily_Snapshots" / today_str
    
    # 1. Read files
    files_to_check = [
        "Thailand_Macro_Snapshot.md",
        "Macro_Snapshot.md",
        "Regional_Pulse.md",
        "US_Economic_Fundamentals.md"
    ]
    
    contents = {}
    for f in files_to_check:
        path = snapshots_dir / f
        if not path.exists():
            return f"Error: ข้อมูลไม่ครบถ้วน ไม่พบไฟล์ {f}"
        
        content = path.read_text(encoding="utf-8")
        if "ไม่พบข้อมูล" in content or "ERROR:" in content:
            return f"Error: ข้อมูลไม่ครบถ้วนในไฟล์ {f} ระบบจะไม่สร้างรายงานเพื่อป้องกันความผิดพลาด"
            
        contents[f] = content

    # 2. Score them
    try:
        report = _generate_agentic_rationales(
            macro_md=contents.get("Macro_Snapshot.md", ""),
            us_md=contents.get("US_Economic_Fundamentals.md", ""),
            regional_md=contents.get("Regional_Pulse.md", ""),
            thai_md=contents.get("Thailand_Macro_Snapshot.md", "")
        )
        return report
    except Exception as e:
        log.error(f"Failed to generate rationales: {e}")
        return f"Error: Failed to evaluate macro matrix - {str(e)}"
