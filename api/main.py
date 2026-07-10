"""FastAPI Controller — entrypoint สำหรับ Web UI (Phase 1)

รัน: uvicorn api.main:app --reload
ต้องตั้งค่าใน .env ก่อน: WEBUI_PASSWORD, SESSION_SECRET_KEY (ห้าม auto-generate — ดู api/auth.py)
"""
from contextlib import asynccontextmanager, closing

from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()

from api import auth, jobs, routes_agents, routes_kanban, routes_portfolio, state_db


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

app.include_router(auth.router)
app.include_router(routes_portfolio.router)
app.include_router(routes_agents.router)
app.include_router(routes_kanban.router)


@app.get("/health")
def health() -> dict:
    return {"ok": True}
