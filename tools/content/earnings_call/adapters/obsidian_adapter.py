"""Obsidian Note Writer Driven Adapter for Earnings Call Transcripts & Highlights."""
from datetime import datetime
import os
from pathlib import Path
import re
import yaml

from application.earnings_call.dto import EarningsCallNoteDTO
from core.logger import get_logger
from tools.archivist.core import _atomic_write_text
from tools.archivist.indexer import _index_upsert

log = get_logger(__name__)

_SAFE_PATH_PART_RE = re.compile(r"[^A-Za-z0-9_\-\.]")


def _sanitize_path_segment(value: str) -> str:
    """Sanitizes a directory/filename segment to avoid path traversal."""
    cleaned = _SAFE_PATH_PART_RE.sub("_", value.strip())
    # Remove leading/trailing dots or underscores
    cleaned = cleaned.strip("._")
    return cleaned or "UNKNOWN"


class ObsidianEarningsCallAdapter:
    """Implements EarningsCallNoteWriterPort using atomic filesystem writes in Obsidian Vault."""

    def __init__(self, vault_path: Path | str | None = None) -> None:
        if vault_path is not None:
            self._vault_path = Path(vault_path)
        else:
            self._vault_path = Path(os.getenv("OBSIDIAN_VAULT_PATH", "./memories"))

    def write_note(
        self, ticker: str, period: str, transcript: str, highlights: str
    ) -> str:
        san_ticker = _sanitize_path_segment(ticker).upper()
        san_period = _sanitize_path_segment(period).upper()

        target_dir = self._vault_path / "30_Knowledge_Base" / "Earnings_Calls" / san_ticker
        target_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{san_period}_{san_ticker}_Earnings_Call.md"
        file_path = target_dir / filename

        today_str = datetime.now().strftime("%Y-%m-%d")
        now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        frontmatter_data = {
            "title": f"{san_ticker} Earnings Call {period}",
            "entity_type": "earnings_call",
            "ticker": san_ticker,
            "period": period,
            "tags": ["earnings_call", san_ticker],
            "date": today_str,
            "last_updated": now_time,
        }

        yaml_block = yaml.safe_dump(
            frontmatter_data, allow_unicode=True, sort_keys=False
        ).strip()

        note_content = "\n".join([
            "---",
            yaml_block,
            "---",
            "",
            f"# {san_ticker} Earnings Call — {period}",
            "",
            "## 🤖 AI Highlights",
            "",
            highlights.strip(),
            "",
            "---",
            "",
            "## 📄 Full Transcript",
            "",
            transcript.strip(),
            "",
        ])

        _atomic_write_text(file_path, note_content)

        try:
            _index_upsert(file_path)
        except Exception as exc:
            log.warning("Index upsert warning for %s: %s", file_path, exc)

        # Compute vault-relative path using forward slashes for cross-platform compatibility
        try:
            rel_path = file_path.relative_to(self._vault_path).as_posix()
        except ValueError:
            rel_path = f"30_Knowledge_Base/Earnings_Calls/{san_ticker}/{filename}"

        log.info("Saved earnings call note to Obsidian: %s", rel_path)
        return rel_path

    def list_notes_for_ticker(self, ticker: str) -> list[EarningsCallNoteDTO]:
        san_ticker = _sanitize_path_segment(ticker).upper()
        target_dir = self._vault_path / "30_Knowledge_Base" / "Earnings_Calls" / san_ticker
        if not target_dir.exists():
            return []

        results: list[EarningsCallNoteDTO] = []
        for file_path in target_dir.glob("*.md"):
            if file_path.name.startswith("."):
                continue
            try:
                content = file_path.read_text(encoding="utf-8")
                metadata = {}
                body = content
                if content.startswith("---"):
                    parts = content.split("---", 2)
                    if len(parts) >= 3:
                        try:
                            metadata = yaml.safe_load(parts[1]) or {}
                        except Exception:
                            metadata = {}
                        body = parts[2]

                title = metadata.get("title") or file_path.stem.replace("_", " ")
                period = metadata.get("period") or file_path.stem.split("_")[0]
                date = metadata.get("date") or datetime.fromtimestamp(file_path.stat().st_mtime).strftime("%Y-%m-%d")
                last_updated = metadata.get("last_updated") or datetime.fromtimestamp(file_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")

                highlights = ""
                if "## 🤖 AI Highlights" in body:
                    after_hl = body.split("## 🤖 AI Highlights", 1)[1]
                    if "## 📄 Full Transcript" in after_hl:
                        highlights_part = after_hl.split("## 📄 Full Transcript", 1)[0]
                    else:
                        highlights_part = after_hl
                    highlights = highlights_part.strip().rstrip("-").strip()
                else:
                    highlights = body.strip()

                has_full_transcript = "## 📄 Full Transcript" in body

                try:
                    rel_path = file_path.relative_to(self._vault_path).as_posix()
                except ValueError:
                    rel_path = f"30_Knowledge_Base/Earnings_Calls/{san_ticker}/{file_path.name}"

                results.append(
                    EarningsCallNoteDTO(
                        title=title,
                        ticker=san_ticker,
                        period=str(period),
                        vault_path=rel_path,
                        highlights=highlights,
                        date=str(date),
                        last_updated=str(last_updated),
                        has_full_transcript=has_full_transcript,
                    )
                )
            except Exception as exc:
                log.warning("Failed to parse earnings call note %s: %s", file_path, exc)

        results.sort(key=lambda item: (item.date, item.period), reverse=True)
        return results
