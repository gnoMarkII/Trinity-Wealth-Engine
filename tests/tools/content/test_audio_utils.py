"""Unit tests for tools/content/notebooklm/audio_utils.py and Discord audio sending"""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from tools.content.notebooklm.audio_utils import (
    compress_audio_for_discord,
    get_ffmpeg_binary,
)
from core import discord_notifier


def test_get_ffmpeg_binary():
    exe = get_ffmpeg_binary()
    assert exe is not None
    assert Path(exe).exists()


def test_get_max_discord_audio_bytes(monkeypatch):
    # Default is 7.5 MiB
    monkeypatch.delenv("DISCORD_MAX_AUDIO_BYTES", raising=False)
    assert discord_notifier.get_max_discord_audio_bytes() == int(7.5 * 1024 * 1024)

    # Valid custom env
    monkeypatch.setenv("DISCORD_MAX_AUDIO_BYTES", "5000000")
    assert discord_notifier.get_max_discord_audio_bytes() == 5000000

    # Invalid string fallback safely
    monkeypatch.setenv("DISCORD_MAX_AUDIO_BYTES", "invalid_number")
    assert discord_notifier.get_max_discord_audio_bytes() == int(7.5 * 1024 * 1024)


def test_compress_skips_when_file_small(tmp_path):
    f = tmp_path / "small.m4a"
    f.write_bytes(b"x" * 1024)  # 1 KB
    res = compress_audio_for_discord(f, max_size_bytes=10000, force=False)
    assert res == f


def test_compress_handles_nonexistent_file():
    p = Path("/nonexistent/audio.m4a")
    res = compress_audio_for_discord(p)
    assert res == p.resolve()


def test_compress_handles_ffmpeg_error_gracefully(tmp_path):
    """หาก ffmpeg error ต้องไม่ทำให้โปรแกรมแครช และคืนไฟล์เดิมอย่างปลอดภัย"""
    f = tmp_path / "large_corrupted.m4a"
    f.write_bytes(b"not a real audio file" * 1000)

    res = compress_audio_for_discord(f, force=True)
    assert res == f


def test_send_notebooklm_audio_discord_with_dedicated_webhook(monkeypatch, tmp_path):
    audio_file = tmp_path / "test_podcast.m4a"
    audio_file.write_bytes(b"fake-audio-bytes")

    mock_post = MagicMock(return_value=True)
    monkeypatch.setattr("core.discord_notifier._post_with_retry", mock_post)
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    monkeypatch.setenv("DISCORD_NOTEBOOKLM_WEBHOOK_URL", "https://discord.com/api/webhooks/notebooklm_channel")
    monkeypatch.setenv("DISCORD_NOTEBOOKLM_TAG_ID", "tag-12345")

    long_title = "A" * 150  # Long title > 100 chars
    result = discord_notifier.send_notebooklm_audio_discord(
        audio_path=audio_file,
        title=long_title,
        summary="This is a summary",
        source_ref="NotebookLM_Sources/test.md",
    )

    assert result.status == "sent"
    mock_post.assert_called_once()
    call_args = mock_post.call_args
    assert call_args.args[0] == "https://discord.com/api/webhooks/notebooklm_channel"

    payload_json = json.loads(call_args.kwargs["data"]["payload_json"])
    assert len(payload_json["thread_name"]) <= 100
    assert payload_json["applied_tags"] == ["tag-12345"]
    assert "C:\\" not in str(payload_json)  # No absolute windows path
    assert payload_json["embeds"][0]["fields"][2]["value"] == "NotebookLM_Sources/test.md"


def test_send_notebooklm_audio_discord_skips_when_oversize(monkeypatch, tmp_path):
    audio_file = tmp_path / "oversize.m4a"
    audio_file.write_bytes(b"x" * 1000)

    mock_post = MagicMock()
    monkeypatch.setattr("core.discord_notifier._post_with_retry", mock_post)
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/fake")
    monkeypatch.setenv("DISCORD_MAX_AUDIO_BYTES", "500")  # Limit 500 bytes

    result = discord_notifier.send_notebooklm_audio_discord(
        audio_path=audio_file,
        title="Oversize Podcast",
    )

    assert result.status == "skipped_oversize"
    mock_post.assert_not_called()


def test_send_notebooklm_audio_discord_skips_when_disabled(monkeypatch, tmp_path):
    audio_file = tmp_path / "audio.m4a"
    audio_file.write_bytes(b"x" * 100)

    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("DISCORD_NOTEBOOKLM_WEBHOOK_URL", raising=False)

    result = discord_notifier.send_notebooklm_audio_discord(
        audio_path=audio_file,
        title="Test Podcast",
    )

    assert result.status == "skipped_disabled"
