"""FastAPI Controller — entrypoint สำหรับ Web UI (Phase 1)

รัน: uvicorn api.main:app --reload
ต้องตั้งค่าใน .env ก่อน: WEBUI_PASSWORD, SESSION_SECRET_KEY (ห้าม auto-generate — ดู api/auth.py)
"""
from contextlib import asynccontextmanager, closing
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

from core.logger import setup_logging

setup_logging()

from api import auth, jobs, notebooklm_worker, routes_agents, routes_debug, routes_kanban, routes_notebooklm, routes_portfolio, routes_equity, routes_ohlcv, state_db

WEB_DIST = Path(__file__).resolve().parent.parent / "web" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    from api import config
    if not config.get_webui_password():
        raise RuntimeError("WEBUI_PASSWORD must be set in environment variables.")

    session_secret = config.get_session_secret()
    if not session_secret or len(session_secret) < 32:
        raise RuntimeError("SESSION_SECRET_KEY must be set and at least 32 characters long for security.")

    draft_key = config.get_unverified_draft_signing_key()
    if not draft_key or len(draft_key) < 32:
        raise RuntimeError("UNVERIFIED_DRAFT_SIGNING_KEY must be set and at least 32 characters long to sign tokens securely.")

    with closing(state_db.get_connection()) as conn:
        state_db.init_schema(conn)

    # คิวหลักกับคิว notebooklm แชร์ WEBUI_STATE_DB_PATH เดียวกัน (kanban_cards ต้องเห็นข้อมูล
    # เดียวกันเสมอ — move_kanban_card ที่ถูกเรียกจาก _run_job ของแต่ละคิวต้องแก้แถวการ์ดจริง
    # ไฟล์เดียวกัน) แยกกันด้วย `flows` allowlist แทน เพื่อกัน reenqueue_pending()/worker loop
    # ของคิวหนึ่งไปกวาดงานอีก flow เข้าคิวตัวเอง (list_jobs_by_status ไม่ filter ตาม flow เอง)
    app.state.job_queue = jobs.JobQueue(
        run_fn=jobs.default_run_fn,
        flows={"manager", "news_youtube", "news_funnel", "youtube_pitch"},
    )
    app.state.job_queue.reenqueue_pending()
    app.state.job_queue.start()

    app.state.notebooklm_job_queue = jobs.JobQueue(
        run_fn=notebooklm_worker.notebooklm_run_fn,
        flows={"notebooklm"},
    )
    app.state.notebooklm_job_queue.reenqueue_pending()
    app.state.notebooklm_job_queue.start()

    yield
    await app.state.job_queue.stop()
    await app.state.notebooklm_job_queue.stop()


app = FastAPI(title="Invest Agents Web UI", lifespan=lifespan)


@app.middleware("http")
async def security_and_cache_headers(request, call_next):
    response = await call_next(request)
    # security headers พื้นฐาน — ไม่ใส่ CSP เพราะหน้า Macro ฝัง TradingView widget
    # (โหลด script จาก s3.tradingview.com) กับ YouTube embed ซึ่งต้อง allowlist ละเอียด
    # และพังเงียบง่ายถ้าตั้งพลาด (ดู docs ของ widget ก่อนถ้าจะเพิ่มภายหลัง)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "same-origin")
    # ไฟล์ใน /assets มี content hash ในชื่อ (vite) — cache ยาวได้แบบ immutable
    if request.url.path.startswith("/assets/"):
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    return response

app.include_router(auth.router)
app.include_router(routes_portfolio.router)
app.include_router(routes_agents.router)
app.include_router(routes_kanban.router)
app.include_router(routes_debug.router)
app.include_router(routes_notebooklm.router)
app.include_router(routes_equity.router)
app.include_router(routes_ohlcv.router)


@app.get("/health")
def health() -> dict:
    return {"ok": True}


# Serve Web UI production build (web/dist) ถ้ามี — dev ใช้ Vite proxy อยู่แล้วไม่ต้องมีก็ได้
# ต้องประกาศ "หลัง" router ทุกตัว เพื่อให้ /api/* และ /health จับก่อน catch-all นี้เสมอ
if WEB_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="webui-assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def webui_spa(full_path: str) -> FileResponse:
        """SPA fallback สำหรับ BrowserRouter — deep link เช่น /kanban ต้องได้ index.html
        ส่วนไฟล์จริงใน dist (favicon.svg, landing/*.png) ให้เสิร์ฟตรงตัว"""
        candidate = (WEB_DIST / full_path).resolve()
        # กัน path traversal — เสิร์ฟเฉพาะไฟล์ที่อยู่ใต้ dist จริงเท่านั้น
        if full_path and candidate.is_file() and candidate.is_relative_to(WEB_DIST):
            return FileResponse(candidate)
        # index.html ห้าม cache — ไม่งั้น deploy ใหม่แล้ว browser ยังชี้ asset hash เก่า
        return FileResponse(WEB_DIST / "index.html", headers={"Cache-Control": "no-cache"})
