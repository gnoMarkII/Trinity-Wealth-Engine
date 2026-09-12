from __future__ import annotations

import hashlib
import json
from pathlib import Path

import tools.archivist.core as core


def test_paged_reader_reconstructs_content_and_rejects_stale_cursor(tmp_path: Path, monkeypatch) -> None:
    vault = tmp_path / "memories"
    note = vault / "30_Knowledge_Base" / "Concepts" / "Long.md"
    note.parent.mkdir(parents=True)
    content = "---\ntitle: Long\nentity_type: concept\n---\n" + ("A" * 9000) + "\nTAIL-MARKER\n"
    note.write_text(content, encoding="utf-8")
    monkeypatch.setattr(core, "VAULT_PATH", vault)

    pages: list[str] = []
    cursor = ""
    first = None
    while True:
        payload = json.loads(
            core.read_note_chunk.func(
                "30_Knowledge_Base/Concepts/Long.md",
                cursor=cursor,
                max_chars=2000,
            )
        )
        assert payload["status"] == "PASS"
        if first is None:
            first = payload
        pages.append(payload["content"])
        if payload["eof"]:
            break
        cursor = payload["next_cursor"]

    reconstructed = "".join(pages)
    assert reconstructed == content
    assert first["content_sha256"] == hashlib.sha256(content.encode("utf-8")).hexdigest()
    assert "TAIL-MARKER" in reconstructed

    note.write_text(content + "changed", encoding="utf-8")
    stale = json.loads(
        core.read_note_chunk.func(
            "30_Knowledge_Base/Concepts/Long.md",
            cursor=first["next_cursor"] or "",
            max_chars=2000,
        )
    )
    # A one-page note would have no cursor; this fixture is intentionally long.
    assert stale["status"] == "STALE_CURSOR"

