"""Tests verifying that Vault operations remain isolated and do not depend on process working directory."""
from __future__ import annotations

import os
from pathlib import Path

from tools.archivist.vault_paths import VaultPaths


def test_cwd_independence(tmp_path: Path, monkeypatch) -> None:
    """Changing os.getcwd() must NOT alter absolute resolved note paths."""
    vault_dir = (tmp_path / "my_vault").resolve()
    other_dir = (tmp_path / "other_workdir").resolve()
    vault_dir.mkdir()
    other_dir.mkdir()

    # Create VaultPaths with explicit root
    vp = VaultPaths(root=vault_dir)

    # Compute note path under default cwd
    p1 = vp.note_path({"entity_type": "stock_hub", "ticker": "FTNT"})

    # Switch cwd to other_dir
    monkeypatch.chdir(other_dir)

    # Compute note path again
    p2 = vp.note_path({"entity_type": "stock_hub", "ticker": "FTNT"})

    assert p1 == p2
    assert p1.is_absolute()
    assert str(p1).startswith(str(vault_dir))
    assert not str(p1).startswith(str(other_dir))
