"""Persistent JSON State Store สำหรับระบบ News Funnel Architecture

รองรับ FileLock เพื่อป้องกัน Multi-process Race Condition ระหว่าง CLI Scheduled Task,
FastAPI Server และ Background Workers พร้อมระบุ schema_version=1
"""
from datetime import datetime, timedelta
import json
import os
from typing import Any, Dict, List, Optional
from filelock import FileLock

from pathlib import Path
from core.logger import get_logger
from core.nlp_utils import _jaccard_similarity
from schemas.news_funnel_schemas import HIGH_IMPACT_THRESHOLD
from tools._atomic_io import _atomic_write_to

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STORE_PATH = str(PROJECT_ROOT / "data" / "news_funnel_state.json")
DEFAULT_LOCK_PATH = str(PROJECT_ROOT / "data" / "news_funnel_state.json.lock")


def _get_paths(store_path: Optional[str] = None, lock_path: Optional[str] = None):
    s_path = store_path or DEFAULT_STORE_PATH
    l_path = lock_path or (s_path + ".lock")
    return s_path, l_path


def _get_initial_store() -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "processed_urls": [],
        "processed_titles": [],
        "pending_events": [],
    }


def _load_unlocked(s_path: str) -> Dict[str, Any]:
    if not os.path.exists(s_path):
        initial = _get_initial_store()
        try:
            _save_unlocked(initial, s_path)
        except Exception:
            with open(s_path, "w", encoding="utf-8") as f:
                json.dump(initial, f, ensure_ascii=False, indent=2)
        return initial

    try:
        with open(s_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or data.get("schema_version") != 1:
            logger.warning("Invalid schema_version or store format, resetting store structure while preserving events if possible.")
            default_data = _get_initial_store()
            default_data.update({k: v for k, v in data.items() if k in default_data})
            default_data["schema_version"] = 1
            data = default_data

        now_iso = datetime.now().isoformat()
        for ev in data.get("pending_events", []):
            if isinstance(ev, dict) and not ev.get("ingested_at"):
                ev["ingested_at"] = now_iso

        return data
    except Exception as e:
        logger.error("Failed to load store %s: %s", s_path, e)
        return _get_initial_store()


def prune_old_events(state: Dict[str, Any], retention_days: int = 7) -> None:
    """ลบเหตุการณ์ทุกสถานะที่เก่าเกิน retention_days เพื่อไม่ให้ JSON Store บวมเกินความจำเป็น"""
    now = datetime.now()
    cutoff = now - timedelta(days=retention_days)
    kept_events = []
    for ev in state.get("pending_events", []):
        if not isinstance(ev, dict):
            continue
        ingested_str = ev.get("ingested_at")
        if ingested_str:
            try:
                ingested_dt = datetime.fromisoformat(ingested_str)
                if ingested_dt < cutoff:
                    continue
            except ValueError:
                pass
        kept_events.append(ev)
    state["pending_events"] = kept_events

    # จำกัดความยาวรายการ processed_urls/titles ไม่ให้เกิน 2,000 รายการล่าสุด
    if len(state.get("processed_urls", [])) > 2000:
        state["processed_urls"] = state["processed_urls"][-2000:]
    if len(state.get("processed_titles", [])) > 2000:
        state["processed_titles"] = state["processed_titles"][-2000:]


def _save_unlocked(state: Dict[str, Any], s_path: str) -> None:
    state["schema_version"] = 1
    prune_old_events(state, retention_days=7)
    _atomic_write_to(Path(s_path), json.dumps(state, ensure_ascii=False, indent=2) + "\n")


def load_store(store_path: Optional[str] = None, lock_path: Optional[str] = None) -> Dict[str, Any]:
    """โหลดข้อมูล Persistent State ภายใต้ FileLock ป้องกัน Race Condition"""
    s_path, l_path = _get_paths(store_path, lock_path)
    os.makedirs(os.path.dirname(s_path) if os.path.dirname(s_path) else ".", exist_ok=True)

    lock = FileLock(l_path, timeout=10, is_singleton=True)
    with lock:
        return _load_unlocked(s_path)


def save_store(state: Dict[str, Any], store_path: Optional[str] = None, lock_path: Optional[str] = None) -> None:
    """บันทึกข้อมูล Persistent State ลงไฟล์ภายใต้ FileLock"""
    s_path, l_path = _get_paths(store_path, lock_path)
    os.makedirs(os.path.dirname(s_path) if os.path.dirname(s_path) else ".", exist_ok=True)

    lock = FileLock(l_path, timeout=10, is_singleton=True)
    with lock:
        _save_unlocked(state, s_path)


def is_title_or_url_processed(
    title: str,
    url: str,
    store_state: Optional[Dict[str, Any]] = None,
    store_path: Optional[str] = None,
    threshold: float = 0.75,
) -> bool:
    """ตรวจสอบว่า URL หรือหัวข้อข่าวนี้เคยถูกคัดกรองหรือประมวลผลแล้วหรือไม่"""
    state = store_state if store_state is not None else load_store(store_path=store_path)
    processed_urls = set(state.get("processed_urls", []))
    if url in processed_urls:
        return True

    processed_titles = state.get("processed_titles", [])
    norm_title = title.strip().lower()
    for pt in processed_titles:
        if pt.strip().lower() == norm_title:
            return True
        if _jaccard_similarity(title, pt) >= threshold:
            return True

    return False


def save_triage_events(events: List[Dict[str, Any]], store_path: Optional[str] = None) -> None:
    """บันทึกรายการข่าวใหม่ที่ผ่านการคัดกรอง (Triage) ลง store และเพิ่ม URL/title ลงรายการที่ประมวลผลแล้ว"""
    s_path, l_path = _get_paths(store_path, None)
    lock = FileLock(l_path, timeout=10, is_singleton=True)
    with lock:
        state = _load_unlocked(s_path)
        existing_event_ids = {e.get("event_id") for e in state.get("pending_events", []) if isinstance(e, dict)}
        urls = state.get("processed_urls", [])
        seen_urls = set(urls)
        processed_titles = state.get("processed_titles", [])

        for event in events:
            ev_id = event.get("event_id")
            if ev_id and ev_id not in existing_event_ids:
                event["status"] = event.get("status", "pending_synthesis")
                event.setdefault("ingested_at", datetime.now().isoformat())
                state["pending_events"].append(event)
                existing_event_ids.add(ev_id)

            for link in event.get("links", []):
                if link not in seen_urls:
                    urls.append(link)
                    seen_urls.add(link)
            title = event.get("canonical_title") or event.get("title")
            if title and title not in processed_titles:
                processed_titles.append(title)

        state["processed_urls"] = urls
        state["processed_titles"] = processed_titles
        _save_unlocked(state, s_path)


def get_pending_high_impact_events(store_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """ดึงรายการข่าว High-Impact ที่ยังรอการสังเคราะห์ (status == 'pending_synthesis')"""
    state = load_store(store_path=store_path)
    pending = []
    for ev in state.get("pending_events", []):
        if not isinstance(ev, dict):
            continue
        if ev.get("status") == "pending_synthesis":
            macro_score = ev.get("macro_impact_score", 0)
            asset_score = ev.get("asset_impact_score", 0)
            is_high = ev.get("is_high_impact")
            if is_high or max(macro_score, asset_score) >= HIGH_IMPACT_THRESHOLD:
                pending.append(ev)
    return pending


def mark_events_synthesized(event_ids: List[str], store_path: Optional[str] = None) -> None:
    """เปลี่ยนสถานะรายการเหตุการณ์เป็น synthesized"""
    if not event_ids:
        return
    s_path, l_path = _get_paths(store_path, None)
    lock = FileLock(l_path, timeout=10, is_singleton=True)
    with lock:
        state = _load_unlocked(s_path)
        target_ids = set(event_ids)

        for ev in state.get("pending_events", []):
            if isinstance(ev, dict) and ev.get("event_id") in target_ids:
                ev["status"] = "synthesized"

        _save_unlocked(state, s_path)
