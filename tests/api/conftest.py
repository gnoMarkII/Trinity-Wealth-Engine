import pytest


@pytest.fixture
def api_env(tmp_path, monkeypatch):
    """แยก state ต่อ test — password/secret คงที่ระหว่าง test, DB แยกเป็น tmp_path"""
    monkeypatch.setenv("WEBUI_PASSWORD", "test-password")
    monkeypatch.setenv("SESSION_SECRET_KEY", "test-secret-key")
    monkeypatch.setenv("WEBUI_STATE_DB_PATH", str(tmp_path / "webui_state.sqlite"))
    monkeypatch.setenv("CHECKPOINT_DB_PATH", str(tmp_path / "checkpoints.sqlite"))

    # login rate limit เป็น module-level state — เคลียร์ต่อ test ไม่งั้น authed_client
    # ที่ login ทุก test จะสะสมจนชน limit (10 ครั้ง/นาที ต่อ IP) กลาง suite
    from api import auth

    auth._login_attempts.clear()
    yield tmp_path


@pytest.fixture
def client(api_env, monkeypatch):
    """TestClient ที่ผูก job queue เข้ากับ fake run_fn — ไม่เรียก LangGraph/LLM จริง"""
    import api.jobs as jobs_module

    def _fake_run_fn(**kwargs) -> None:
        return None

    monkeypatch.setattr(jobs_module, "default_run_fn", _fake_run_fn)

    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture
def authed_client(client):
    r = client.post("/api/auth/login", json={"password": "test-password"})
    assert r.status_code == 200
    return client
