import os

SESSION_COOKIE_NAME = "invest_agents_session"
SESSION_MAX_AGE_SECONDS = 60 * 60 * 24 * 30  # 30 วัน — ล็อกอินครั้งเดียว ไม่บล็อกการใช้งานประจำวัน


def get_webui_password() -> str:
    return os.getenv("WEBUI_PASSWORD", "")


def get_session_secret() -> str:
    """secret สำหรับ sign cookie — ต้อง fix ไว้ใน .env ไม่ auto-generate ต่อ process
    ไม่งั้น session ทุกใบจะ invalid ทันทีที่ restart server (ขัดกับเป้าหมาย login ครั้งเดียว)
    """
    return os.getenv("SESSION_SECRET_KEY", "")


def get_unverified_draft_signing_key() -> str:
    """Return the dedicated HMAC key for Unverified Draft approval tokens.

    This intentionally does not fall back to the session-cookie secret: rotating
    either key must not silently change the trust boundary of the other.
    """
    return os.getenv("UNVERIFIED_DRAFT_SIGNING_KEY", "")


def get_cookie_secure() -> bool:
    """ตั้ง SESSION_COOKIE_SECURE=1 เมื่อ deploy หลัง HTTPS (reverse proxy/tunnel) —
    default ปิดเพราะ localhost ใช้ http และ browser จะไม่เก็บ Secure cookie บน http
    """
    return os.getenv("SESSION_COOKIE_SECURE", "").strip().lower() in ("1", "true", "yes")


def get_state_db_path() -> str:
    return os.getenv("WEBUI_STATE_DB_PATH", "data/webui_state.sqlite")


def get_checkpoint_db_path() -> str:
    return os.getenv("CHECKPOINT_DB_PATH", "data/checkpoints.sqlite")


def enable_background_workers() -> bool:
    val = os.getenv("ENABLE_BACKGROUND_WORKERS", "true").strip().lower()
    return val not in ("0", "false", "no", "off")


def enable_job_workers() -> bool:
    """Return whether the durable agent/job queue workers may run.

    ``ENABLE_BACKGROUND_WORKERS`` historically controlled both queue workers
    and the Earnings Call outbox worker.  Keep that value as the default for
    deployments that do not opt into the split, while allowing tests and
    one-shot environments to disable the outbox without making API dispatch
    requests hang forever with jobs left in ``queued`` state.
    """
    raw = os.getenv("ENABLE_JOB_WORKERS")
    if raw is None:
        return enable_background_workers()
    return raw.strip().lower() not in ("0", "false", "no", "off")


def schedulers_enabled() -> bool:
    """Return whether periodic schedulers may be started by the composition root.

    Keeping this switch in configuration makes test and one-shot deployments
    independent of production scheduler side effects.
    """
    val = os.getenv("SCHEDULER_ENABLED", "true").strip().lower()
    return val not in ("0", "false", "no", "off")
