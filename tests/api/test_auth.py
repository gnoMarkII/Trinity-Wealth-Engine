def test_health_is_public(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_unauthenticated_request_returns_401(client):
    r = client.get("/api/kanban/cards")
    assert r.status_code == 401


def test_wrong_password_rejected(client):
    r = client.post("/api/auth/login", json={"password": "wrong"})
    assert r.status_code == 401


def test_correct_password_grants_session(authed_client):
    r = authed_client.get("/api/kanban/cards")
    assert r.status_code == 200


def test_logout_revokes_session(authed_client):
    r = authed_client.post("/api/auth/logout")
    assert r.status_code == 200
    r = authed_client.get("/api/kanban/cards")
    assert r.status_code == 401


def test_me_reports_unauthenticated_without_cookie(client):
    r = client.get('/api/auth/me')
    assert r.status_code == 200
    assert r.json() == {"authenticated": False}


def test_me_reports_authenticated_with_valid_cookie(authed_client):
    r = authed_client.get('/api/auth/me')
    assert r.status_code == 200
    assert r.json() == {"authenticated": True}


def test_login_requires_configured_password(client, monkeypatch):
    monkeypatch.delenv("WEBUI_PASSWORD", raising=False)
    monkeypatch.setenv("WEBUI_PASSWORD", "")
    r = client.post("/api/auth/login", json={"password": "anything"})
    assert r.status_code == 500
