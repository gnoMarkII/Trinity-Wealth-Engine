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

from api import auth, jobs, routes_agents, routes_kanban, routes_portfolio, state_db

WEB_DIST = Path(__file__).resolve().parent.parent / "web" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    with closing(state_db.get_connection()) as conn:
        state_db.init_schema(conn)

    app.state.job_queue = jobs.JobQueue(run_fn=jobs.default_run_fn)
    app.state.job_queue.reenqueue_pending()
    app.state.job_queue.start()
    yield
    await app.state.job_queue.stop()


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
