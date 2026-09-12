"""Obsidian Note Writer Driven Adapter for Earnings Call Transcripts & Highlights."""
from datetime import datetime
import hashlib
import os
from pathlib import Path
import re
import yaml

from application.earnings_call.dto import EarningsCallNoteDTO, EarningsCallWriteResultDTO
from application.knowledge.note_write_ports import KnowledgeNoteWritePort
from application.knowledge.write_context import current_note_writer
from core.logger import get_logger
from tools.archivist.indexer import _index_upsert
from tools.archivist.maintenance_guard import assert_write_allowed

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

    def __init__(self, vault_path: Path | str | None = None, *, note_writer: KnowledgeNoteWritePort | None = None) -> None:
        if vault_path is not None:
            self._vault_path = Path(vault_path).resolve()
        else:
            self._vault_path = Path(os.getenv("OBSIDIAN_VAULT_PATH", "./memories")).resolve()
        self._note_writer = note_writer or current_note_writer(self._vault_path)
        self._last_write_result: EarningsCallWriteResultDTO | None = None

    @property
    def last_write_result(self) -> EarningsCallWriteResultDTO | None:
        """Reference for the most recent successful write in this adapter."""
        return self._last_write_result

    def write_note(
        self, ticker: str, period: str, transcript: str, highlights: str
    ) -> str:
        san_ticker = _sanitize_path_segment(ticker).upper()
        san_period = _sanitize_path_segment(period).upper()
        assert_write_allowed(self._vault_path)

        from tools.archivist.vault_paths import VaultPaths
        from application.knowledge.identity import build_document_key
        from tools.archivist.metadata import parse_note

        vp = VaultPaths(self._vault_path)
        if vp.layout_version >= 2:
            target_dir = self._vault_path / "30_Knowledge_Base" / "Stocks" / san_ticker / "Earnings"
        else:
            target_dir = self._vault_path / "30_Knowledge_Base" / "Earnings_Calls" / san_ticker
        target_dir.mkdir(parents=True, exist_ok=True)

        full_transcript_hash = hashlib.sha256(transcript.strip().encode("utf-8")).hexdigest()
        filename = f"{san_period}_{san_ticker}_Earnings_Call.md"
        primary_candidate = target_dir / filename
        transcript_variant = ""

        # If note exists with different transcript, disambiguate filename to preserve both
        if primary_candidate.exists():
            try:
                existing_text = primary_candidate.read_text(encoding="utf-8")
                existing_meta, existing_body, _ = parse_note(existing_text)
                existing_hash = existing_meta.get("transcript_sha256")
                if not existing_hash:
                    marker = "## 📄 Full Transcript"
                    existing_transcript = existing_body.split(marker, 1)[-1].strip() if marker in existing_body else existing_body.strip()
                    existing_hash = hashlib.sha256(existing_transcript.encode("utf-8")).hexdigest()
                if str(existing_hash) != full_transcript_hash:
                    transcript_variant = full_transcript_hash[:12]
                    filename = f"{san_period}_{san_ticker}_Earnings_Call_{transcript_variant}.md"
            except Exception:
                pass

        today_str = datetime.now().strftime("%Y-%m-%d")
        now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        source_identity = f"{san_ticker}:{san_period}"
        if transcript_variant:
            source_identity = f"{source_identity}:transcript:{full_transcript_hash}"
        doc_key = build_document_key(
            kind="earnings_call",
            source_identity=source_identity,
            role="primary",
        )

        frontmatter_data = {
            "title": f"{san_ticker} Earnings Call {period}",
            "entity_type": "earnings_call",
            "ticker": san_ticker,
            "period": period,
            "tags": ["earnings_call", san_ticker],
            "date": today_str,
            "last_updated": now_time,
            "document_key": doc_key,
            "transcript_sha256": full_transcript_hash,
        }

        body_content = "\n".join([
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

        committed = self._note_writer.write_note(
            metadata=frontmatter_data,
            body=body_content,
            filename=filename,
        )
        file_path = committed.primary_file
        self._last_write_result = EarningsCallWriteResultDTO(
            vault_path=file_path.relative_to(self._vault_path).as_posix(),
            note_id=committed.note_id,
            revision_id=committed.revision_id,
            revision=committed.revision,
            content_sha256=hashlib.sha256(file_path.read_bytes()).hexdigest(),
            artifact_set_hash=committed.artifact_set_hash,
            manifest_path=str(committed.manifest_path) if committed.manifest_path else None,
        )

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

    def write_note_result(
        self, ticker: str, period: str, transcript: str, highlights: str
    ) -> EarningsCallWriteResultDTO:
        """Write a note and return its immutable V2 revision reference."""
        self.write_note(ticker, period, transcript, highlights)
        if self._last_write_result is None:
            raise RuntimeError("Earnings note write completed without a durable revision reference")
        return self._last_write_result

    def list_notes_for_ticker(self, ticker: str) -> list[EarningsCallNoteDTO]:
        san_ticker = _sanitize_path_segment(ticker).upper()
        v2_dir = self._vault_path / "30_Knowledge_Base" / "Stocks" / san_ticker / "Earnings"
        v1_dir = self._vault_path / "30_Knowledge_Base" / "Earnings_Calls" / san_ticker

        candidate_files: list[Path] = []
        if v2_dir.exists():
            candidate_files.extend(v2_dir.glob("*.md"))
        if v1_dir.exists():
            candidate_files.extend(v1_dir.glob("*.md"))

        if not candidate_files:
            return []

        seen_names: set[str] = set()
        files_to_read: list[Path] = []
        for f in candidate_files:
            if f.name not in seen_names and not f.name.startswith("."):
                seen_names.add(f.name)
                files_to_read.append(f)

        results: list[EarningsCallNoteDTO] = []
        for file_path in files_to_read:
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
