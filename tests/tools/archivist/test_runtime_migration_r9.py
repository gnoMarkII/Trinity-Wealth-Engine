import json
from pathlib import Path

from tools.archivist.runtime_migration import migrate_runtime_layout


def _vault(tmp_path: Path) -> Path:
    vault = tmp_path / "memories"
    (vault / ".system").mkdir(parents=True)
    (vault / ".system" / "vault_config.json").write_text(json.dumps({"layout_version": 2}), encoding="utf-8")
    return vault


def test_runtime_migration_is_dry_run_then_idempotent_apply(tmp_path: Path) -> None:
    vault = _vault(tmp_path)
    base = tmp_path / "data" / "vault_runtime"
    legacy = base / "broker"
    legacy.mkdir(parents=True)
    (legacy / "receipt.json").write_text("receipt\n", encoding="utf-8")

    dry = migrate_runtime_layout(vault_root=vault, runtime_base=base, apply=False)
    assert dry["status"] == "PASS"
    assert not (base / "memories" / "broker" / "receipt.json").exists()

    applied = migrate_runtime_layout(vault_root=vault, runtime_base=base, apply=True)
    assert applied["status"] == "PASS"
    assert (base / "memories" / "broker" / "receipt.json").read_text(encoding="utf-8") == "receipt\n"
    assert json.loads((vault / ".system" / "vault_config.json").read_text(encoding="utf-8"))["vault_id"] == "memories"

    second = migrate_runtime_layout(vault_root=vault, runtime_base=base, apply=True)
    assert second["status"] == "PASS"
    assert any(entry["status"] == "already_present" for entry in second["entries"])


def test_runtime_migration_rejects_divergent_destination(tmp_path: Path) -> None:
    vault = _vault(tmp_path)
    base = tmp_path / "data" / "vault_runtime"
    legacy = base / "broker"
    destination = base / "memories" / "broker"
    legacy.mkdir(parents=True)
    destination.mkdir(parents=True)
    (legacy / "receipt.json").write_text("old\n", encoding="utf-8")
    (destination / "receipt.json").write_text("new\n", encoding="utf-8")

    result = migrate_runtime_layout(vault_root=vault, runtime_base=base, apply=True)
    assert result["status"] == "CONFLICT"
    assert (destination / "receipt.json").read_text(encoding="utf-8") == "new\n"
