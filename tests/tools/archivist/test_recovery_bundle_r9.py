import json
import sqlite3
from pathlib import Path

from tools.archivist.recovery_bundle import create_recovery_bundle, restore_recovery_bundle


def _make_vault(root: Path) -> Path:
    vault = root / "memories"
    (vault / ".system").mkdir(parents=True)
    (vault / ".system" / "vault_config.json").write_text(
        json.dumps({"layout_version": 2, "vault_id": "test-vault"}), encoding="utf-8"
    )
    (vault / "30_Knowledge_Base").mkdir()
    (vault / "30_Knowledge_Base" / "note.md").write_text("---\nnote_id: note_1\n---\nhello\n", encoding="utf-8")
    (vault / ".system" / "storage_contract.json").write_text(json.dumps({"schema_version": 2}), encoding="utf-8")
    return vault


def _make_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE events (id INTEGER PRIMARY KEY, value TEXT NOT NULL)")
        conn.execute("INSERT INTO events(value) VALUES ('one')")


def test_recovery_bundle_contains_online_sqlite_backup_and_restores_to_staging(tmp_path: Path) -> None:
    vault = _make_vault(tmp_path)
    runtime_base = tmp_path / "data" / "vault_runtime"
    runtime = runtime_base / "test-vault"
    _make_db(runtime / "broker" / "knowledge_write.sqlite3")
    (runtime / "logs").mkdir(parents=True)
    (runtime / "logs" / "event.log").write_text("ok\n", encoding="utf-8")
    (runtime / "broker" / "knowledge_write.sqlite3-wal").write_bytes(b"transient")

    bundle = create_recovery_bundle(
        vault_root=vault,
        runtime_base=runtime_base,
        output_dir=tmp_path / "bundles",
        run_id="r9-test",
    )
    bundle_root = Path(bundle["bundle_root"])
    assert bundle["status"] == "PASS"
    assert not list((bundle_root / "runtime").rglob("*-wal"))
    assert (bundle_root / "runtime" / "test-vault" / "broker" / "knowledge_write.sqlite3").is_file()
    manifest = json.loads((bundle_root / "bundle-manifest.json").read_text(encoding="utf-8"))
    assert manifest["runtime_state"]["broker"]["present"] is True
    assert manifest["runtime_state"]["registry_digest"]
    assert manifest["derived_rebuild"]["vector"]

    result = restore_recovery_bundle(bundle_root=bundle_root, restore_root=tmp_path / "restored")
    assert result["status"] == "PASS"
    assert result["restored_external_files_verified"] > 0
    assert (tmp_path / "restored" / "metadata" / "storage_contract.json").is_file()
    restored_db = tmp_path / "restored" / "runtime" / "test-vault" / "broker" / "knowledge_write.sqlite3"
    with sqlite3.connect(restored_db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1
    assert (tmp_path / "restored" / "vault" / "30_Knowledge_Base" / "note.md").is_file()
