"""Auth แบบเบา — กันคนนอก ไม่กันเจ้าของ (ดู Rev.3/4/5)

รหัสผ่านเดียวจาก .env, session cookie เซ็นด้วย itsdangerous, อายุยาว 30 วัน
ไม่มีระบบ user/role เพราะเป็นเครื่องมือส่วนตัวคนเดียว
"""
import hmac
import time
from collections import defaultdict, deque

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from pydantic import BaseModel
from core.security import AuthenticatedPrincipal

from api.config import (
    SESSION_COOKIE_NAME,
    SESSION_MAX_AGE_SECONDS,
    get_cookie_secure,
    get_session_secret,
    get_webui_password,
)

_SESSION_SALT = "invest-agents-session"

# Rate limit ที่ login เท่านั้น (sliding window ต่อ IP, in-memory) — เครื่องมือส่วนตัว
# single-process จึงไม่ต้องพึ่ง redis; กัน brute force กรณี expose ผ่าน tunnel/internet
_LOGIN_WINDOW_SECONDS = 60.0
_LOGIN_MAX_ATTEMPTS_PER_WINDOW = 10
_login_attempts: dict[str, deque[float]] = defaultdict(deque)


def _check_login_rate_limit(client_ip: str) -> None:
    now = time.monotonic()
    attempts = _login_attempts[client_ip]
    while attempts and now - attempts[0] > _LOGIN_WINDOW_SECONDS:
        attempts.popleft()
    if len(attempts) >= _LOGIN_MAX_ATTEMPTS_PER_WINDOW:
        raise HTTPException(status_code=429, detail="พยายาม login ถี่เกินไป — รอสักครู่แล้วลองใหม่")
    attempts.append(now)


router = APIRouter()


def _serializer() -> URLSafeTimedSerializer:
    secret = get_session_secret()
    if not secret:
        raise RuntimeError(
            "SESSION_SECRET_KEY ยังไม่ได้ตั้งค่าใน .env — ต้องกำหนดค่าคงที่ก่อนรัน Web UI "
            "(ห้าม auto-generate เพราะจะทำให้ session ทุกใบ invalid ทันทีที่ restart process)"
        )
    return URLSafeTimedSerializer(secret, salt=_SESSION_SALT)


class LoginRequest(BaseModel):
    password: str


import uuid


def _set_session_cookie(response: Response) -> str:
    sid = uuid.uuid4().hex
    token = _serializer().dumps({"authenticated": True, "sid": sid})
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        max_age=SESSION_MAX_AGE_SECONDS,
        httponly=True,
        samesite="lax",
        secure=get_cookie_secure(),
    )
    return sid


@router.post("/api/auth/login")
def login(payload: LoginRequest, request: Request, response: Response) -> dict:
    _check_login_rate_limit(request.client.host if request.client else "unknown")
    expected = get_webui_password()
    if not expected:
        raise HTTPException(status_code=500, detail="WEBUI_PASSWORD ยังไม่ได้ตั้งค่าใน .env")
    if not hmac.compare_digest(payload.password.encode("utf-8"), expected.encode("utf-8")):
        raise HTTPException(status_code=401, detail="รหัสผ่านไม่ถูกต้อง")
    _set_session_cookie(response)
    return {"ok": True}


@router.post("/api/auth/logout")
def logout(response: Response) -> dict:
    response.delete_cookie(SESSION_COOKIE_NAME)
    return {"ok": True}


@router.get("/api/auth/me")
def me(request: Request) -> dict:
    """ให้ frontend เช็คสถานะ login ได้โดยไม่ต้อง 401 กระพริบตอนโหลดหน้าแรก"""
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        return {"authenticated": False}
    try:
        _serializer().loads(token, max_age=SESSION_MAX_AGE_SECONDS)
    except (SignatureExpired, BadSignature):
        return {"authenticated": False}
    return {"authenticated": True}


def require_session(request: Request) -> AuthenticatedPrincipal:
    """FastAPI dependency — แปะไว้ที่ทุก route ใต้ /api/* ยกเว้น login/health"""
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        raise HTTPException(status_code=401, detail="ไม่พบ session — กรุณา login ก่อน")
    try:
        data = _serializer().loads(token, max_age=SESSION_MAX_AGE_SECONDS)
        sid = (data.get("sid") if isinstance(data, dict) else None) or "legacy_session"
        subject = f"session:{sid}"
        return AuthenticatedPrincipal(
            subject_id=subject,
            session_id=str(sid),
            display_name=subject,
            role="PORTFOLIO_MANAGER_HERMES",
            is_authenticated=True,
        )
    except SignatureExpired:
        raise HTTPException(status_code=401, detail="session หมดอายุ — กรุณา login ใหม่")
    except BadSignature:
        raise HTTPException(status_code=401, detail="session ไม่ถูกต้อง")


def get_authenticated_principal(principal: AuthenticatedPrincipal = Depends(require_session)) -> AuthenticatedPrincipal:
    """Dependency for routes requiring trusted identity context."""
    return principal


def get_session_id(request: Request) -> str:
    """Extract authenticated session ID for session-bound operations."""
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        raise HTTPException(status_code=401, detail="ไม่พบ session — กรุณา login ก่อน")
    try:
        data = _serializer().loads(token, max_age=SESSION_MAX_AGE_SECONDS)
        sid = data.get("sid")
        if not sid:
            return "legacy_default_session"
        return str(sid)
    except SignatureExpired:
        raise HTTPException(status_code=401, detail="session หมดอายุ — กรุณา login ใหม่")
    except BadSignature:
        raise HTTPException(status_code=401, detail="session ไม่ถูกต้อง")
