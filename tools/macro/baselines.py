import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any

from langchain_core.tools import tool

import os

log = logging.getLogger(__name__)

def _get_baselines_dir() -> Path:
    vault_path = os.getenv("OBSIDIAN_VAULT_PATH", str(Path(__file__).resolve().parents[2] / "memories"))
    return Path(vault_path) / "30_Knowledge_Base" / "Macroeconomics" / "Baselines"

@tool
def get_macro_baselines() -> str:
    """ดึงข้อมูล Baseline (ข้อมูล Snapshot ของ Macro Themes) จาก 7 วันและ 30 วันที่แล้ว

    [Usage/When to use]
    ใช้เมื่อต้องการข้อมูลอดีตมาเปรียบเทียบหาจุด Pivot หรือเช็คว่าเทรนด์มีการเปลี่ยนแปลงหรือไม่
    - Economist Agent ต้องเรียกใช้ฟังก์ชันนี้ก่อนเริ่มวิเคราะห์เสมอ

    [Caution]
    - ข้อมูลที่ได้จะเป็น JSON String ของ Theme Snapshots ที่ผ่านการสรุปมาแล้ว
    - หากไม่พบข้อมูลในอดีต (เช่น เพิ่งเริ่มระบบใหม่) จะได้รับข้อความระบุว่าไม่พบข้อมูลแทน

    Args:
        None

    Returns:
        str: JSON String ที่รวบรวม Snapshot ย้อนหลัง 7 วันและ 30 วัน
    """
    try:
        now = datetime.now(timezone.utc)
        
        baselines = {
            "short_baseline_7d": _get_snapshot(now - timedelta(days=7)),
            "medium_baseline_30d": _get_snapshot(now - timedelta(days=30))
        }
        return json.dumps(baselines, ensure_ascii=False, indent=2)
    except Exception as e:
        log.error(f"Error in get_macro_baselines: {e}")
        return f"Error: ไม่สามารถดึงข้อมูล Baseline ได้ ({str(e)})"


def _get_snapshot(target_date: datetime) -> Dict[str, Any]:
    baselines_dir = _get_baselines_dir()
    if not baselines_dir.exists():
        baselines_dir.mkdir(parents=True, exist_ok=True)
        
    files = list(baselines_dir.glob("*.md"))
    
    closest_file = None
    min_diff = timedelta(days=999)
    for f in files:
        try:
            date_str = f.stem.replace("Macro_Baseline_", "")
            file_date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            diff = abs(target_date - file_date)
            if diff < min_diff and diff <= timedelta(days=2):
                min_diff = diff
                closest_file = f
        except Exception:
            continue
            
    if closest_file:
        try:
            with open(closest_file, "r", encoding="utf-8") as file:
                content = file.read()
                
            import re
            json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
                return {
                    "date": data.get("evaluated_at", "Unknown"),
                    "dominant_themes": data.get("dominant_themes", []),
                    "market_sentiment": data.get("market_sentiment", "neutral")
                }
            else:
                log.error(f"No JSON found in {closest_file}")
        except Exception as e:
            log.error(f"Failed to read baseline file {closest_file}: {e}")
            
    return {
        "date": target_date.strftime("%Y-%m-%d"),
        "dominant_themes": [],
        "market_sentiment": "neutral",
        "note": "No historical baseline found for this period."
    }

def save_macro_baseline(data: Dict[str, Any]) -> None:
    """บันทึกข้อมูล NarrativeContext เป็น Baseline สำหรับการเปรียบเทียบในอนาคต"""
    try:
        baselines_dir = _get_baselines_dir()
        if not baselines_dir.exists():
            baselines_dir.mkdir(parents=True, exist_ok=True)
            
        evaluated_at_str = data.get("evaluated_at", datetime.now(timezone.utc).isoformat())
        # Parse it to a date string YYYY-MM-DD
        try:
            evaluated_date = datetime.fromisoformat(evaluated_at_str.replace("Z", "+00:00")).strftime("%Y-%m-%d")
        except ValueError:
            evaluated_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            
        file_path = baselines_dir / f"Macro_Baseline_{evaluated_date}.md"
        
        json_content = json.dumps(data, ensure_ascii=False, indent=2)
        
        markdown_content = f"""---
title: Macro Baseline {evaluated_date}
date: {evaluated_date}
tags: [macro, baseline]
---

# Macro Baseline ({evaluated_date})

ข้อมูลด้านล่างนี้ถูกใช้เป็นฐานอ้างอิง (Baseline) สำหรับให้ AI เช็คทิศทางการเปลี่ยนแปลง (Pivot) ในอนาคต

```json
{json_content}
```
"""
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
            
        log.info(f"Successfully saved macro baseline to {file_path}")
    except Exception as e:
        log.error(f"Failed to save macro baseline: {e}")
