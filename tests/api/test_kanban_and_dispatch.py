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


def test_new_card_defaults_discord_notify_true(authed_client):
    r = authed_client.post("/api/kanban/cards", json={"title": "การ์ดใหม่"})
    assert r.json()["card"]["discord_notify"] is True


def test_toggle_card_discord_off_then_on(authed_client):
    r = authed_client.post("/api/kanban/cards", json={"title": "การ์ดทดสอบ toggle"})
    card_id = r.json()["card"]["card_id"]

    r = authed_client.patch(f"/api/kanban/cards/{card_id}/discord", json={"enabled": False})
    assert r.status_code == 200
    assert r.json()["discord_notify"] is False

    cards = authed_client.get("/api/kanban/cards").json()
    assert next(c for c in cards if c["card_id"] == card_id)["discord_notify"] is False

    r = authed_client.patch(f"/api/kanban/cards/{card_id}/discord", json={"enabled": True})
    assert r.json()["discord_notify"] is True


def test_toggle_discord_on_nonexistent_card_returns_404(authed_client):
    r = authed_client.patch("/api/kanban/cards/does-not-exist/discord", json={"enabled": False})
    assert r.status_code == 404


def test_update_card_does_not_reset_discord_toggle(authed_client):
    """update_kanban_card ไม่ควรแตะ discord_notify — กันบั๊ก toggle ที่ user ปิดไว้ถูกเปิดกลับมา
    ทุกครั้งที่การ์ดถูกแก้ไข (เช่นตอน upsert_news_funnel_card อัปเดตการ์ดเดิมด้วยข่าวชุดใหม่)"""
    r = authed_client.post("/api/kanban/cards", json={"title": "การ์ด backlog", "flow": "manager"})
    card_id = r.json()["card"]["card_id"]
    authed_client.patch(f"/api/kanban/cards/{card_id}/discord", json={"enabled": False})

    r = authed_client.patch(
        f"/api/kanban/cards/{card_id}",
        json={"title": "การ์ด backlog (แก้ไขแล้ว)", "flow": "manager", "scope": "both"},
    )
    assert r.status_code == 200
    assert r.json()["discord_notify"] is False


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


def test_dispatch_with_card_id_moves_card_to_executing_atomically(authed_client):
    """dispatch ต้องย้ายการ์ดเป็น executing + ผูก job_id ให้เองในคำขอเดียว — ไม่ต้องให้ frontend
    ยิง PUT /api/kanban/move ตามหลัง (ถ้า call ที่สองล้มเหลว การ์ดจะค้าง backlog ทั้งที่ job รันอยู่)
    """
    r = authed_client.post("/api/kanban/cards", json={"title": "งานที่จะ dispatch"})
    card_id = r.json()["card"]["card_id"]

    r = authed_client.post(
        "/api/agents/dispatch",
        json={"instruction": "วิเคราะห์ตลาดวันนี้", "card_id": card_id},
    )
    assert r.status_code == 200
    job = r.json()

    cards = authed_client.get("/api/kanban/cards").json()
    dispatched_card = next(c for c in cards if c["card_id"] == card_id)
    assert dispatched_card["column_name"] in ("executing", "done")
    assert dispatched_card["job_id"] == job["job_id"]


def test_dispatch_without_card_id_does_not_touch_kanban(authed_client):
    r = authed_client.post("/api/agents/dispatch", json={"instruction": "งานไม่มีการ์ด"})
    assert r.status_code == 200
    # ไม่ raise, ไม่มีการ์ดให้ move — แค่ยืนยันว่าไม่พังตอนไม่มี card_id


def test_stream_endpoint_completes_for_already_done_job(authed_client):
    """SSE endpoint ต้องทำงานได้ปกติ (ไม่ hang, ไม่ throw) หลังเปลี่ยนมาอ่าน DB ผ่าน
    asyncio.to_thread — ใช้ job ที่ done ไปแล้วเพื่อให้ stream จบทันทีโดยไม่ต้อง sleep รอ
    """
    r = authed_client.post("/api/agents/dispatch", json={"instruction": "งานทดสอบ stream"})
    job_id = r.json()["job_id"]

    import time
    for _ in range(50):
        status = authed_client.get(f"/api/agents/jobs/{job_id}").json()["status"]
        if status in ("done", "error"):
            break
        time.sleep(0.05)

    with authed_client.stream("GET", f"/api/agents/stream/{job_id}") as resp:
        assert resp.status_code == 200
        body = b"".join(resp.iter_bytes())
    assert b"event: done" in body or b"event: error" in body


def test_dispatch_with_require_approval_waits_in_approval_and_supports_approve_and_reject(authed_client):
    """เมื่อ require_approval=True การ์ดต้องหยุดรอในคอลัมน์ approval จนกว่าผู้ใช้จะกดยอมรับหรือยกเลิก"""
    # 1. Approve flow
    r1 = authed_client.post("/api/kanban/cards", json={"title": "วิเคราะห์หุ้น INTC (US)"})
    card_id_1 = r1.json()["card"]["card_id"]

    d1 = authed_client.post(
        "/api/agents/dispatch",
        json={
            "instruction": "วิเคราะห์หุ้น INTC แบบละเอียด",
            "card_id": card_id_1,
            "require_approval": True,
        },
    )
    assert d1.status_code == 200
    job1 = d1.json()
    assert job1["status"] == "awaiting_approval"
    assert job1["interrupt_payload"]["type"] == "pre_run_approval"

    cards = authed_client.get("/api/kanban/cards").json()
    card1 = next(c for c in cards if c["card_id"] == card_id_1)
    assert card1["column_name"] == "approval"
    assert card1["job_id"] == job1["job_id"]

    # Approve via /resume
    res1 = authed_client.post(
        f"/api/agents/jobs/{job1['job_id']}/resume",
        json={"action": "approve"},
    )
    assert res1.status_code == 200
    assert res1.json()["status"] in ("queued", "running", "done")

    cards_after_approve = authed_client.get("/api/kanban/cards").json()
    card1_after = next(c for c in cards_after_approve if c["card_id"] == card_id_1)
    assert card1_after["column_name"] in ("executing", "done")

    # 2. Reject flow
    r2 = authed_client.post("/api/kanban/cards", json={"title": "การ์ดที่จะยกเลิกก่อนรัน"})
    card_id_2 = r2.json()["card"]["card_id"]

    d2 = authed_client.post(
        "/api/agents/dispatch",
        json={
            "instruction": "วิเคราะห์หุ้น AAPL แบบละเอียด",
            "card_id": card_id_2,
            "require_approval": True,
        },
    )
    assert d2.status_code == 200
    job2 = d2.json()
    assert job2["status"] == "awaiting_approval"

    res2 = authed_client.post(
        f"/api/agents/jobs/{job2['job_id']}/resume",
        json={"action": "reject"},
    )
    assert res2.status_code == 200
    assert res2.json()["status"] == "error"
    assert "ยกเลิก" in (res2.json().get("error_message") or "")

    cards_after_reject = authed_client.get("/api/kanban/cards").json()
    card2_after = next(c for c in cards_after_reject if c["card_id"] == card_id_2)
    assert card2_after["column_name"] == "backlog"
