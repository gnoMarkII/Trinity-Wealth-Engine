"""Refresh and commit the latest completed US sector-rotation snapshot.

Schedule this command after the US regular-session close. Re-running it for the
same completed session is idempotent and reuses the current committed snapshot.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.logger import setup_logging
from application.macro.sector_rotation_service import _missing_completed_sessions
from tools.market.market_calendar import get_last_completed_regular_session
from tools.macro.sector_rotation.bootstrap import get_sector_rotation_service


def _max_stale_sessions() -> int:
    raw = os.getenv("SECTOR_ROTATION_EOD_MAX_STALE_SESSIONS", "1").strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("SECTOR_ROTATION_EOD_MAX_STALE_SESSIONS must be an integer") from exc
    if not 0 <= value <= 5:
        raise ValueError("SECTOR_ROTATION_EOD_MAX_STALE_SESSIONS must be between 0 and 5")
    return value


def _append_run_log(record: dict[str, object]) -> None:
    log_path = PROJECT_ROOT / "logs" / "sector_rotation_eod.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def _vault_scope() -> str:
    raw = Path(os.getenv("OBSIDIAN_VAULT_PATH", "./memories")).expanduser()
    vault_root = raw.resolve() if raw.is_absolute() else (PROJECT_ROOT / raw).resolve()
    return "scratch" if vault_root.is_relative_to((PROJECT_ROOT / "scratch").resolve()) else "configured"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(PROJECT_ROOT / ".env")
    setup_logging()

    started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    try:
        expected_session = get_last_completed_regular_session().isoformat()
        snapshot = get_sector_rotation_service().refresh_now(
            allow_stale_provider_data=True,
            max_stale_sessions=_max_stale_sessions(),
        )
        missing_sessions = _missing_completed_sessions(snapshot.as_of_date, expected_session)
    except Exception as exc:
        logging.getLogger(__name__).exception("Sector rotation EOD refresh failed")
        result = {
            "status": "failed",
            "error": type(exc).__name__,
            "vault_scope": _vault_scope(),
            "started_at": started_at,
        }
        try:
            _append_run_log(result)
        except Exception:
            logging.getLogger(__name__).exception("Could not append Sector Rotation EOD run log")
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 1

    result = {
        "status": "ok" if missing_sessions == 0 else "stale",
        "freshness": "fresh" if missing_sessions == 0 else "stale",
        "vault_scope": _vault_scope(),
        "snapshot_id": snapshot.snapshot_id,
        "as_of_date": snapshot.as_of_date,
        "expected_session": expected_session,
        "missing_sessions": missing_sessions,
        "benchmark_status": snapshot.benchmark_status,
        "available_sectors": snapshot.available_sectors,
        "expected_sectors": snapshot.expected_sectors,
        "formula_version": snapshot.formula_version,
        "calendar_version": snapshot.calendar_version,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    try:
        _append_run_log(result)
    except Exception:
        logging.getLogger(__name__).exception("Could not append Sector Rotation EOD run log")
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
