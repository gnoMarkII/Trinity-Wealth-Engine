import json

import pytest

from scripts import refresh_sector_rotation


def test_eod_stale_session_limit_defaults_and_accepts_range(monkeypatch):
    monkeypatch.delenv("SECTOR_ROTATION_EOD_MAX_STALE_SESSIONS", raising=False)
    assert refresh_sector_rotation._max_stale_sessions() == 1

    monkeypatch.setenv("SECTOR_ROTATION_EOD_MAX_STALE_SESSIONS", "0")
    assert refresh_sector_rotation._max_stale_sessions() == 0
    monkeypatch.setenv("SECTOR_ROTATION_EOD_MAX_STALE_SESSIONS", "5")
    assert refresh_sector_rotation._max_stale_sessions() == 5


@pytest.mark.parametrize("value", ["invalid", "-1", "6"])
def test_eod_stale_session_limit_rejects_invalid_values(monkeypatch, value):
    monkeypatch.setenv("SECTOR_ROTATION_EOD_MAX_STALE_SESSIONS", value)
    with pytest.raises(ValueError):
        refresh_sector_rotation._max_stale_sessions()


def test_eod_run_log_appends_private_scope_metadata_without_vault_path(tmp_path, monkeypatch):
    monkeypatch.setattr(refresh_sector_rotation, "PROJECT_ROOT", tmp_path)
    record = {"status": "ok", "vault_scope": "scratch", "snapshot_id": "sr_test"}

    refresh_sector_rotation._append_run_log(record)
    refresh_sector_rotation._append_run_log({**record, "status": "stale"})

    lines = (tmp_path / "logs" / "sector_rotation_eod.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["status"] for line in lines] == ["ok", "stale"]
    assert all("vault_path" not in json.loads(line) for line in lines)


def test_vault_scope_only_reports_scratch_or_configured(monkeypatch, tmp_path):
    project = tmp_path / "project"
    scratch_vault = project / "scratch" / "smoke" / "vault"
    configured_vault = tmp_path / "private-vault"
    scratch_vault.mkdir(parents=True)
    configured_vault.mkdir()
    monkeypatch.setattr(refresh_sector_rotation, "PROJECT_ROOT", project)

    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(scratch_vault))
    assert refresh_sector_rotation._vault_scope() == "scratch"
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(configured_vault))
    assert refresh_sector_rotation._vault_scope() == "configured"
