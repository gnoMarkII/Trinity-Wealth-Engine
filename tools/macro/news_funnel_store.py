"""Persistent JSON State Store สำหรับระบบ News Funnel Architecture

รองรับ FileLock เพื่อป้องกัน Multi-process Race Condition ระหว่าง CLI Scheduled Task,
FastAPI Server และ Background Workers พร้อมระบุ schema_version=1
"""
from datetime import datetime, timedelta
import json
import os
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode
from filelock import FileLock

from pathlib import Path
from core.logger import get_logger
from core.nlp_utils import _jaccard_similarity
from schemas.news_funnel_schemas import HIGH_IMPACT_THRESHOLD
from tools._atomic_io import _atomic_write_to

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STORE_PATH = str(PROJECT_ROOT / "data" / "news_funnel_state.json")


def _get_paths(store_path: Optional[str] = None):
    s_path = store_path or DEFAULT_STORE_PATH
    l_path = s_path + ".lock"
    return s_path, l_path


def _get_initial_store() -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "processed_urls": [],
        "processed_titles": [],
        "pending_events": [],
        "raw_candidates": [],
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
                if ingested_dt.tzinfo is not None:
                    ingested_dt = ingested_dt.replace(tzinfo=None)
                if ingested_dt < cutoff:
                    continue
            except (ValueError, TypeError):
                pass
        kept_events.append(ev)
    state["pending_events"] = kept_events

    # Prune raw_candidates ที่เก่าเกิน 48 ชม. และกำหนด hard cap 500 รายการล่าสุด
    cutoff_raw = now - timedelta(hours=48)
    kept_raw = []
    for item in state.setdefault("raw_candidates", []):
        if not isinstance(item, dict):
            continue
        fetched_str = item.get("fetched_at")
        if fetched_str:
            try:
                fetched_dt = datetime.fromisoformat(fetched_str)
                if fetched_dt.tzinfo is not None:
                    fetched_dt = fetched_dt.replace(tzinfo=None)
                if fetched_dt < cutoff_raw:
                    continue
            except (ValueError, TypeError):
                pass
        kept_raw.append(item)
    if len(kept_raw) > 500:
        kept_raw = kept_raw[-500:]
    state["raw_candidates"] = kept_raw

    # จำกัดความยาวรายการ processed_urls/titles ไม่ให้เกิน 2,000 รายการล่าสุด
    if len(state.get("processed_urls", [])) > 2000:
        state["processed_urls"] = state["processed_urls"][-2000:]
    if len(state.get("processed_titles", [])) > 2000:
        state["processed_titles"] = state["processed_titles"][-2000:]


def _save_unlocked(state: Dict[str, Any], s_path: str) -> None:
    state["schema_version"] = 1
    prune_old_events(state, retention_days=7)
    _atomic_write_to(Path(s_path), json.dumps(state, ensure_ascii=False, indent=2) + "\n")


def load_store(store_path: Optional[str] = None) -> Dict[str, Any]:
    """โหลดข้อมูล Persistent State ภายใต้ FileLock ป้องกัน Race Condition"""
    s_path, l_path = _get_paths(store_path)
    os.makedirs(os.path.dirname(s_path) if os.path.dirname(s_path) else ".", exist_ok=True)

    lock = FileLock(l_path, timeout=10, is_singleton=True)
    with lock:
        return _load_unlocked(s_path)


def save_store(state: Dict[str, Any], store_path: Optional[str] = None) -> None:
    """บันทึกข้อมูล Persistent State ลงไฟล์ภายใต้ FileLock"""
    s_path, l_path = _get_paths(store_path)
    os.makedirs(os.path.dirname(s_path) if os.path.dirname(s_path) else ".", exist_ok=True)

    lock = FileLock(l_path, timeout=10, is_singleton=True)
    with lock:
        _save_unlocked(state, s_path)


def _normalize_url(url: str) -> str:
    """ทำ Normalization กับ URL ตัดเฉพาะ tracking parameters, www., trailing slash เพื่อป้องกันการประเมินซ้ำ"""
    if not url or not isinstance(url, str):
        return ""
    try:
        parsed = urlsplit(url.strip())
        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        path = parsed.path
        if len(path) > 1 and path.endswith("/"):
            path = path[:-1]

        tracking_keys = {
            "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
            "ref", "fbclid", "gclid", "si"
        }
        raw_params = parse_qsl(parsed.query, keep_blank_values=True)
        kept_params = []
        for k, v in raw_params:
            lk = k.lower()
            if lk in tracking_keys or lk.startswith("utm_"):
                continue
            kept_params.append((k, v))

        new_query = urlencode(sorted(kept_params)) if kept_params else ""
        return urlunsplit((scheme, netloc, path, new_query, ""))
    except Exception:
        return url.strip().lower()


def is_title_or_url_processed(
    title: str,
    url: str,
    store_state: Optional[Dict[str, Any]] = None,
    store_path: Optional[str] = None,
    threshold: float = 0.75,
    include_raw: bool = False,
) -> bool:
    """ตรวจสอบว่า URL หรือหัวข้อข่าวนี้เคยถูกคัดกรองหรือประมวลผลแล้วหรือไม่"""
    state = store_state if store_state is not None else load_store(store_path=store_path)
    processed_urls = set(state.get("processed_urls", []))
    if url and url in processed_urls:
        return True

    norm_input_url = _normalize_url(url)
    if norm_input_url:
        norm_processed_urls = {_normalize_url(p_url) for p_url in processed_urls if p_url}
        if norm_input_url in norm_processed_urls:
            return True

    processed_titles = state.get("processed_titles", [])
    norm_title = title.strip().lower()
    for pt in processed_titles:
        if pt.strip().lower() == norm_title:
            return True
        if _jaccard_similarity(title, pt) >= threshold:
            return True

    if include_raw:
        for rc in state.setdefault("raw_candidates", []):
            if not isinstance(rc, dict):
                continue
            rc_url = rc.get("link", "")
            if url and rc_url and (url == rc_url or (norm_input_url and _normalize_url(rc_url) == norm_input_url)):
                return True
            rc_title = rc.get("title", "")
            if rc_title:
                if rc_title.strip().lower() == norm_title or _jaccard_similarity(title, rc_title) >= threshold:
                    return True

    return False


def save_triage_events(events: List[Dict[str, Any]], store_path: Optional[str] = None) -> None:
    """บันทึกรายการข่าวใหม่ที่ผ่านการคัดกรอง (Triage) ลง store และเพิ่ม URL/title ลงรายการที่ประมวลผลแล้ว"""
    s_path, l_path = _get_paths(store_path)
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
            for t_key in ("original_title", "canonical_title", "title"):
                t_val = event.get(t_key)
                if t_val and t_val not in processed_titles:
                    processed_titles.append(t_val)

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
    pending.sort(key=lambda ev: 1 if ev.get("triage_source") == "heuristic_fallback" else 0)
    return pending


def update_events_status(
    rejected_ids: Optional[List[str]] = None,
    synthesized_ids: Optional[List[str]] = None,
    skipped_error_ids: Optional[List[str]] = None,
    error_msgs: Optional[Dict[str, str]] = None,
    store_path: Optional[str] = None,
) -> None:
    """เปลี่ยนสถานะรายการเหตุการณ์เป็น synthesized, rejected และ/หรือ skipped_error ใน Transaction เดียวภายใต้ FileLock"""
    if not rejected_ids and not synthesized_ids and not skipped_error_ids:
        return
    s_path, l_path = _get_paths(store_path)
    lock = FileLock(l_path, timeout=10, is_singleton=True)
    with lock:
        state = _load_unlocked(s_path)
        rej_set = set(rejected_ids or [])
        syn_set = set(synthesized_ids or [])
        skip_set = set(skipped_error_ids or [])
        err_map = error_msgs or {}
        now_iso = datetime.now().isoformat()

        for ev in state.get("pending_events", []):
            if not isinstance(ev, dict):
                continue
            ev_id = ev.get("event_id")
            if ev_id in syn_set:
                ev["status"] = "synthesized"
            elif ev_id in rej_set:
                ev["status"] = "rejected"
                ev["rejected_at"] = now_iso
            elif ev_id in skip_set:
                ev["status"] = "skipped_error"
                ev["skipped_at"] = now_iso
                if ev_id in err_map and err_map[ev_id]:
                    ev["error_msg"] = err_map[ev_id]

        _save_unlocked(state, s_path)


def get_filtered_or_rejected_events(store_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """ดึงรายการข่าวที่ไม่ผ่านเกณฑ์ (Low-Impact / Score < 7), รายการที่ถูกปฏิเสธ (rejected) หรือดึงข้อมูลล้มเหลว (skipped_error)"""
    state = load_store(store_path=store_path)
    filtered = []
    for ev in state.get("pending_events", []):
        if not isinstance(ev, dict):
            continue
        status = ev.get("status")
        if status in ("rejected", "skipped_error"):
            filtered.append(ev)
        elif status == "pending_synthesis":
            macro_score = ev.get("macro_impact_score", 0) or 0
            asset_score = ev.get("asset_impact_score", 0) or 0
            is_high = ev.get("is_high_impact")
            if not (is_high or max(macro_score, asset_score) >= HIGH_IMPACT_THRESHOLD):
                filtered.append(ev)

    filtered.sort(key=lambda ev: str(ev.get("ingested_at") or ""), reverse=True)
    return filtered


def save_raw_candidates(items: List[Dict[str, Any]], store_path: Optional[str] = None) -> None:
    """บันทึกสะสมรายการข่าวใหม่ลงใน raw_candidates โดยประทับเวลา fetched_at ภายใต้ FileLock"""
    if not items:
        return
    s_path, l_path = _get_paths(store_path)
    lock = FileLock(l_path, timeout=10, is_singleton=True)
    with lock:
        state = _load_unlocked(s_path)
        raw_list = state.setdefault("raw_candidates", [])
        now_iso = datetime.now().isoformat()
        for it in items:
            it_copy = dict(it)
            it_copy.setdefault("fetched_at", now_iso)
            raw_list.append(it_copy)
        _save_unlocked(state, s_path)


def get_raw_candidates(store_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """อ่านรายการข่าวทั้งหมดจาก raw_candidates โดยไม่ลบออกจาก store (Read-only ภายใต้ FileLock)"""
    state = load_store(store_path=store_path)
    return list(state.setdefault("raw_candidates", []))


def remove_processed_raw_candidates(
    processed_urls: set[str],
    processed_titles: set[str],
    store_path: Optional[str] = None,
    threshold: float = 0.75,
) -> None:
    """ลบรายการข่าวใน raw_candidates ที่เพิ่งประมวลผลสำเร็จออกจาก store (Remove-by-Identity) ภายใต้ FileLock"""
    s_path, l_path = _get_paths(store_path)
    lock = FileLock(l_path, timeout=10, is_singleton=True)
    with lock:
        state = _load_unlocked(s_path)
        raw_list = state.setdefault("raw_candidates", [])
        norm_processed_urls = {_normalize_url(p_url) for p_url in processed_urls if p_url}
        kept_raw = []
        for rc in raw_list:
            if not isinstance(rc, dict):
                continue
            url = rc.get("link", "")
            title = rc.get("title", "")
            norm_title = title.strip().lower()

            if url and url in processed_urls:
                continue
            norm_rc_url = _normalize_url(url)
            if norm_rc_url and norm_rc_url in norm_processed_urls:
                continue

            matched_title = False
            for pt in processed_titles:
                if not pt:
                    continue
                if pt.strip().lower() == norm_title or _jaccard_similarity(title, pt) >= threshold:
                    matched_title = True
                    break
            if matched_title:
                continue

            kept_raw.append(rc)
        state["raw_candidates"] = kept_raw
        _save_unlocked(state, s_path)

