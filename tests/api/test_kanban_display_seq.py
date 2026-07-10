"""ทดสอบ Phase 0 ของ Kanban Rev.2: display_seq, flow บน kanban_cards, log_count บน JobStatusDTO"""
import sqlite3

from api import state_db
from api.routes_agents import _job_to_dto


def test_create_kanban_card_assigns_sequential_display_seq(tmp_path):
    db_path = str(tmp_path / "state.sqlite")
    conn = state_db.get_connection(db_path)

    state_db.create_kanban_card(conn, "c1", "งานที่ 1", flow="manager")
    state_db.create_kanban_card(conn, "c2", "งานที่ 2", flow="news_youtube")
    state_db.create_kanban_card(conn, "c3", "งานที่ 3", flow="manager")

    c1 = state_db.get_kanban_card(conn, "c1")
    c2 = state_db.get_kanban_card(conn, "c2")
    c3 = state_db.get_kanban_card(conn, "c3")

    assert c1["display_seq"] == 1
    assert c2["display_seq"] == 2
    assert c3["display_seq"] == 3
    assert c1["flow"] == "manager"
    assert c2["flow"] == "news_youtube"
    conn.close()


def test_backfill_assigns_display_seq_to_old_cards_by_created_at(tmp_path):
    """จำลองการ์ดเก่าที่สร้างก่อนมีคอลัมน์ display_seq (ค่า NULL) ต้องถูกเติมเลขตามลำดับ created_at"""
    db_path = str(tmp_path / "old_cards.sqlite")
    old_conn = sqlite3.connect(db_path)
    old_conn.executescript("""
        CREATE TABLE kanban_cards (
            card_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            column_name TEXT NOT NULL DEFAULT 'backlog',
            job_id TEXT,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );
    """)
    old_conn.execute(
        "INSERT INTO kanban_cards (card_id, title, column_name, created_at, updated_at) VALUES (?,?,?,?,?)",
        ("old-1", "การ์ดเก่าที่ 1", "backlog", 100.0, 100.0),
    )
    old_conn.execute(
        "INSERT INTO kanban_cards (card_id, title, column_name, created_at, updated_at) VALUES (?,?,?,?,?)",
        ("old-2", "การ์ดเก่าที่ 2", "backlog", 200.0, 200.0),
    )
    old_conn.commit()
    old_conn.close()

    conn = state_db.get_connection(db_path)
    old1 = state_db.get_kanban_card(conn, "old-1")
    old2 = state_db.get_kanban_card(conn, "old-2")
    assert old1["display_seq"] == 1
    assert old2["display_seq"] == 2
    assert old1["flow"] == "manager"  # ค่า default สำหรับการ์ดเก่าที่ไม่เคยมี flow

    # การ์ดใหม่ที่สร้างหลัง backfill ต้องต่อเลขจากของเก่า ไม่ชนกัน
    state_db.create_kanban_card(conn, "new-1", "การ์ดใหม่", flow="manager")
    new1 = state_db.get_kanban_card(conn, "new-1")
    assert new1["display_seq"] == 3
    conn.close()


def test_job_status_dto_includes_log_count_and_timestamps(tmp_path):
    db_path = str(tmp_path / "state.sqlite")
    conn = state_db.get_connection(db_path)
    state_db.create_job(conn, "job-1", "thread-1", "card-1", "key-1", "instr", status="running")
    state_db.append_job_log(conn, "job-1", "macro_quant", "log line 1")
    state_db.append_job_log(conn, "job-1", "macro_quant", "log line 2")

    job = state_db.get_job(conn, "job-1")
    dto = _job_to_dto(conn, job)

    assert dto.log_count == 2
    assert dto.created_at > 0
    assert dto.updated_at > 0
    conn.close()
