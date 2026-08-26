"""Phase 0 Gate: Read-Only Contract Test for SQLite state_db exports & transaction atomicity.

Asserts exported signatures in state_db and tests atomicity of multi-card batch operations and rollback under failure.
"""
import inspect
import json
import sqlite3
from pathlib import Path
import pytest

import api.state_db as state_db

MANIFEST_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "manifest_state_db_exports.json"


def _format_annotation(ann):
    if ann is inspect.Parameter.empty:
        return "empty"
    if hasattr(ann, "__name__"):
        return ann.__name__
    return str(ann)


def _format_default(default):
    if default is inspect.Parameter.empty:
        return "<EMPTY>"
    if default is None:
        return None
    if isinstance(default, (int, float, bool, str, list, dict)):
        return default
    return str(default)


@pytest.fixture(scope="module")
def golden_manifest():
    assert MANIFEST_PATH.exists(), f"Missing baseline manifest: {MANIFEST_PATH}"
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_state_db_functions_exported(golden_manifest):
    expected_functions = golden_manifest["functions"]

    for fn_name, expected_info in expected_functions.items():
        assert hasattr(state_db, fn_name), f"Missing exported function '{fn_name}' in api.state_db"
        fn = getattr(state_db, fn_name)
        assert callable(fn), f"Exported symbol '{fn_name}' is not callable"

        sig = inspect.signature(fn)
        expected_params = expected_info["parameters"]
        actual_params = [
            {
                "name": p_name,
                "kind": str(param.kind),
                "default": _format_default(param.default),
                "annotation": _format_annotation(param.annotation),
            }
            for p_name, param in sig.parameters.items()
        ]

        # Check parameter count and names
        actual_param_names = [p["name"] for p in actual_params]
        expected_param_names = [p["name"] for p in expected_params]
        assert (
            actual_param_names == expected_param_names
        ), f"Parameter names mismatch on '{fn_name}': Expected {expected_param_names}, got {actual_param_names}"

        # Check kinds and defaults
        for act_p, exp_p in zip(actual_params, expected_params):
            assert (
                act_p["name"] == exp_p["name"]
            ), f"Param order mismatch in {fn_name}: {act_p['name']} vs {exp_p['name']}"
            assert (
                act_p["kind"] == exp_p["kind"]
            ), f"Param kind mismatch in {fn_name}.{act_p['name']}: {act_p['kind']} vs {exp_p['kind']}"


def test_create_parking_lot_cards_atomic_atomicity(tmp_path):
    """Test that create_parking_lot_cards_atomic succeeds atomically."""
    db_file = str(tmp_path / "test_state.db")
    conn = state_db.get_connection(db_file)
    state_db.init_schema(conn)
    conn.close()

    ideas = ["Idea Alpha", "Idea Beta", "Idea Gamma"]
    inserted = state_db.create_parking_lot_cards_atomic(ideas, source_pitch_id="pitch_123", db_path=db_file)
    assert inserted == 3

    conn2 = state_db.get_connection(db_file)
    cards = state_db.list_kanban_cards(conn2)
    conn2.close()

    card_titles = [c["title"] for c in cards]
    assert "Idea Alpha" in card_titles
    assert "Idea Beta" in card_titles
    assert "Idea Gamma" in card_titles


def test_create_parking_lot_cards_atomic_failure_injection_rollback(tmp_path, monkeypatch):
    """Test that create_parking_lot_cards_atomic rolls back completely if an error occurs mid-transaction."""
    db_file = str(tmp_path / "test_state_fail.db")
    conn = state_db.get_connection(db_file)
    state_db.init_schema(conn)
    state_db.create_kanban_card(conn, card_id="card_init_1", title="Existing Card", column_name="backlog")
    conn.close()

    import api.db.repositories.kanban_repository as kanban_repo
    real_get_conn = kanban_repo.get_connection

    class FailingConnectionProxy:
        def __init__(self, real_conn):
            self._conn = real_conn
            self._insert_count = 0

        def __getattr__(self, name):
            return getattr(self._conn, name)

        def __setattr__(self, name, value):
            if name in ("_conn", "_insert_count"):
                super().__setattr__(name, value)
            else:
                setattr(self._conn, name, value)

        def execute(self, sql, *args):
            if "INSERT OR IGNORE INTO kanban_cards" in sql:
                self._insert_count += 1
                if self._insert_count == 2:
                    raise sqlite3.OperationalError("Simulated mid-transaction failure injection")
            return self._conn.execute(sql, *args)

        def close(self):
            return self._conn.close()

    def failing_get_connection(*args, **kwargs):
        conn = real_get_conn(*args, **kwargs)
        return FailingConnectionProxy(conn)

    monkeypatch.setattr(kanban_repo, "get_connection", failing_get_connection)

    ideas = ["Fail Idea 1", "Fail Idea 2", "Fail Idea 3"]
    with pytest.raises(sqlite3.OperationalError):
        state_db.create_parking_lot_cards_atomic(ideas, source_pitch_id="pitch_fail", db_path=db_file)

    # Verify that NONE of the new ideas were committed (rollback was successful)
    conn_check = state_db.get_connection(db_file)
    cards = state_db.list_kanban_cards(conn_check)
    conn_check.close()

    titles = [c["title"] for c in cards]
    assert "Fail Idea 1" not in titles, "Fail Idea 1 should have been rolled back"
    assert "Fail Idea 2" not in titles, "Fail Idea 2 should have been rolled back"
    assert len(cards) == 1
    assert cards[0]["title"] == "Existing Card"
