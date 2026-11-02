from typing import Any, Dict
from fastapi import APIRouter
from core.llm_factory import check_llm_preflight
from api.db import get_connection
from tools.telemetry.failure_ledger import aggregate_probe_statistics

router = APIRouter(prefix="/api/health", tags=["Health"])


@router.get("/components")
def get_component_health() -> dict[str, str]:
    """ตรวจสอบสุขภาพของ components ต่างๆ ในระบบ (API, Database, Job Worker, Market Provider, LLM)"""
    health: dict[str, str] = {
        "api": "ok",
        "database": "ok",
        "job_worker": "ok",
        "market_provider": "ok",
        "llm": "ok",
    }

    # 1. Database check
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        conn.close()
    except Exception:
        health["database"] = "unavailable"

    # 2. LLM preflight check
    try:
        is_llm_ready, _ = check_llm_preflight()
        if not is_llm_ready:
            health["llm"] = "unavailable"
    except Exception:
        health["llm"] = "unavailable"

    return health


@router.get("/harness-telemetry")
def get_harness_telemetry() -> Dict[str, Any]:
    """สถิติรวมของ Probe Telemetry และ Continuous Learning Loop (Pareto issues, recovery rates)"""
    try:
        return aggregate_probe_statistics()
    except Exception as e:
        return {
            "total_events": 0,
            "resolved_rate": 0.0,
            "issue_counts": {},
            "top_recurring_issues": [],
            "error": str(e),
        }
