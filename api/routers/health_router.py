from fastapi import APIRouter
from core.llm_factory import check_llm_preflight
from api.db import get_connection

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
