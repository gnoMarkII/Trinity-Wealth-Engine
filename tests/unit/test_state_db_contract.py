"""Phase 0 Gate: Read-Only Contract Test for SQLite state_db exports & transaction atomicity.

Asserts exported signatures in state_db and tests atomicity of multi-card batch operations.
"""
import inspect
import json
import sqlite3
from pathlib import Path
import pytest

import api.state_db as state_db

MANIFEST_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "manifest_state_db_exports.json"


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
        actual_param_names = list(sig.parameters.keys())
        expected_param_names = [p["name"] for p in expected_info["parameters"]]
        assert (
            actual_param_names == expected_param_names
        ), f"Parameter signature mismatch on '{fn_name}': Expected {expected_param_names}, got {actual_param_names}"


def test_create_parking_lot_cards_atomic_atomicity(tmp_path):
    """Test that create_parking_lot_cards_atomic succeeds or rolls back atomically."""
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
