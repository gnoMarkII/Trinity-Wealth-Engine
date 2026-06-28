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
def evaluate_macro_matrix() -> str:
    """ประเมินข้อมูลเศรษฐกิจมหภาคและสร้างรายงาน (Macro Matrix Score)

    [Usage/When to use]
    ใช้เมื่อต้องการวิเคราะห์สภาวะเศรษฐกิจ (Economic State) และคำนวณ Macro Matrix Score
    - ดึงข้อมูลจากไฟล์ Daily Snapshots ล่าสุดเพื่อประเมินสถานการณ์
    - สรุปผลเป็นรายงาน Markdown พร้อมตารางคะแนน 4 มิติ (Monetary Policy, Growth, Inflation, Geopolitics)

    Returns:
        str: รายงานผลวิเคราะห์ Macro Matrix ในรูปแบบ Markdown
    """
    today_str = os.environ.get("EVAL_DATE", datetime.now().strftime("%Y-%m-%d"))
    
    # Paths based on test_macro.py expectations
    vault_path = Path(os.environ.get("OBSIDIAN_VAULT_PATH", "C:/ChinoDoc/Projects/Claude/invest-agents/memories")).resolve()
    snapshots_dir = vault_path / "30_Knowledge_Base" / "Macroeconomics" / "Daily_Snapshots" / today_str
    print(f"DEBUG: vault_path = {vault_path}")
    print(f"DEBUG: snapshots_dir = {snapshots_dir}")
    
    # 1. Read files
    files_to_check_map = {
        "Global_Macro_Snapshot": f"Global_Macro_Snapshot_{today_str}.md",
        "Regional_Macro_Snapshot": f"Regional_Macro_Snapshot_{today_str}.md",
        "Country_Macro_Snapshot": f"Country_Macro_Snapshot_{today_str}.md"
    }
    
    contents = {}
    for key, f in files_to_check_map.items():
        path = snapshots_dir / f
        if not path.exists():
            # Fallback to names without date suffix for test mocks
            fallback_path = snapshots_dir / f"{key}.md"
            if fallback_path.exists():
                path = fallback_path
                f = f"{key}.md"
            else:
                return f"Error: ข้อมูลไม่ครบถ้วน ไม่พบไฟล์ {f}"
        
        content = path.read_text(encoding="utf-8")
        if "ไม่พบข้อมูล" in content or "ERROR:" in content:
            return f"Error: ข้อมูลไม่ครบถ้วนในไฟล์ {f} ระบบจะไม่สร้างรายงานเพื่อป้องกันความผิดพลาด"
            
        contents[key] = content

    # 2. Score them
    try:
        report = _generate_agentic_rationales(
            global_md=contents.get("Global_Macro_Snapshot", ""),
            regional_md=contents.get("Regional_Macro_Snapshot", ""),
            country_md=contents.get("Country_Macro_Snapshot", "")
        )
        
        # Save to Obsidian
        report_filename = f"Macro_Economic_Evaluation_{today_str}.md"
        report_path = snapshots_dir / report_filename
        report_path.write_text(report, encoding="utf-8")
        
        return f"บันทึกไฟล์สำเร็จที่: {report_path}\n\n{report}"
    except Exception as e:
        log.error(f"Failed to generate rationales: {e}")
        return f"Error: Failed to evaluate macro matrix - {str(e)}"
