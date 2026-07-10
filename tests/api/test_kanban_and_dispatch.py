def test_create_and_move_kanban_card(authed_client):
    r = authed_client.post("/api/kanban/cards", json={"title": "รันกลยุทธ์วันนี้"})
    assert r.status_code == 200
    body = r.json()
    assert body["created"] is True
    card = body["card"]
    assert card["column_name"] == "backlog"

    r = authed_client.put(
        "/api/kanban/move",
        json={"card_id": card["card_id"], "column_name": "executing"},
    )
    assert r.status_code == 200
    assert r.json()["column_name"] == "executing"


def test_create_duplicate_card_in_backlog_returns_existing(authed_client):
    r1 = authed_client.post("/api/kanban/cards", json={"title": "ดึงข่าวเศรษฐกิจล่าสุด"})
    body1 = r1.json()
    assert body1["created"] is True

    r2 = authed_client.post("/api/kanban/cards", json={"title": "ดึงข่าวเศรษฐกิจล่าสุด"})
    body2 = r2.json()
    assert body2["created"] is False
    assert body2["card"]["card_id"] == body1["card"]["card_id"]

    cards = authed_client.get("/api/kanban/cards").json()
    matching = [c for c in cards if c["title"] == "ดึงข่าวเศรษฐกิจล่าสุด"]
    assert len(matching) == 1


def test_duplicate_check_only_applies_within_backlog(authed_client):
    r1 = authed_client.post("/api/kanban/cards", json={"title": "งานซ้ำข้ามคอลัมน์"})
    card_id = r1.json()["card"]["card_id"]
    authed_client.put("/api/kanban/move", json={"card_id": card_id, "column_name": "done"})

    r2 = authed_client.post("/api/kanban/cards", json={"title": "งานซ้ำข้ามคอลัมน์"})
    assert r2.json()["created"] is True  # อันเดิมอยู่ done แล้ว ไม่ใช่ backlog เลยสร้างใหม่ได้


def test_move_card_invalid_column_rejected(authed_client):
    r = authed_client.post("/api/kanban/cards", json={"title": "x"})
    card = r.json()["card"]
    r = authed_client.put(
        "/api/kanban/move",
        json={"card_id": card["card_id"], "column_name": "not-a-real-column"},
    )
    assert r.status_code == 400


def test_move_nonexistent_card_returns_404(authed_client):
    r = authed_client.put(
        "/api/kanban/move",
        json={"card_id": "does-not-exist", "column_name": "done"},
    )
    assert r.status_code == 404


def test_delete_card_removes_it_from_list(authed_client):
    r = authed_client.post("/api/kanban/cards", json={"title": "การ์ดที่จะลบ"})
    card_id = r.json()["card"]["card_id"]

    r = authed_client.delete(f"/api/kanban/cards/{card_id}")
    assert r.status_code == 200
    assert r.json() == {"ok": True}

    r = authed_client.get("/api/kanban/cards")
    assert all(c["card_id"] != card_id for c in r.json())


def test_delete_nonexistent_card_returns_404(authed_client):
    r = authed_client.delete("/api/kanban/cards/does-not-exist")
    assert r.status_code == 404


def test_dispatch_returns_job_status(authed_client):
    r = authed_client.post("/api/agents/dispatch", json={"instruction": "วิเคราะห์ตลาดวันนี้"})
    assert r.status_code == 200
    body = r.json()
    assert body["job_id"]
    assert body["status"] in ("queued", "running", "done", "error")


def test_dispatch_empty_instruction_rejected(authed_client):
    r = authed_client.post("/api/agents/dispatch", json={"instruction": "   "})
    assert r.status_code == 400


def test_dispatch_requires_auth(client):
    r = client.post("/api/agents/dispatch", json={"instruction": "x"})
    assert r.status_code == 401
