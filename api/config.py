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


def get_cookie_secure() -> bool:
    """ตั้ง SESSION_COOKIE_SECURE=1 เมื่อ deploy หลัง HTTPS (reverse proxy/tunnel) —
    default ปิดเพราะ localhost ใช้ http และ browser จะไม่เก็บ Secure cookie บน http
    """
    return os.getenv("SESSION_COOKIE_SECURE", "").strip().lower() in ("1", "true", "yes")


def get_state_db_path() -> str:
    return os.getenv("WEBUI_STATE_DB_PATH", "data/webui_state.sqlite")


def get_checkpoint_db_path() -> str:
    return os.getenv("CHECKPOINT_DB_PATH", "data/checkpoints.sqlite")
