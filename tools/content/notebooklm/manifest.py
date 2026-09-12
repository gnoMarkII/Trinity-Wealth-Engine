"""Run-state manifest สำหรับ NotebookLM Pipeline — เก็บ progress ต่อไฟล์ briefing เพื่อ resume/retry ได้

Location: data/notebooklm_runs/<content_hash>.json
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ValidationError

from core.logger import get_logger
from tools._atomic_io import _atomic_write_to

logger = get_logger(__name__)

MANIFEST_DIR = Path("data/notebooklm_runs")


class NotebookLMManifest(BaseModel):
    """Schema version-controlled manifest — validate ได้ตอนโหลดจากดิสก์

    status ไล่ตามลำดับ: initialized -> notebook_created -> source_added
    -> research_done (optional) -> prompts_queried (optional) -> audio_generating -> completed

    pipeline.py ใช้ "field ไหนถูก populate แล้วบ้าง" เป็นตัวตัดสิน resume point จริง (ไม่ใช่แค่
    อ่าน status string อย่างเดียว) — เมื่อ audio generation ล้มเหลวจริงฝั่ง NotebookLM (ไม่ใช่แค่
    process แครช) status จะถูกย้อนกลับไปที่ checkpoint ก่อนหน้า (research_done/source_added)
    พร้อมเคลียร์ artifact_id แทนที่จะตั้งเป็น "failed" เฉยๆ เพื่อไม่ให้ resume ครั้งถัดไปหลงลืมว่า
    ทำ research ไปแล้วหรือยัง
    """
    schema_version: int = 1
    content_hash: str
    briefing_path: str
    notebook_id: str | None = None
    source_id: str | None = None
    artifact_id: str | None = None
    audio_path: str | None = None
    status: str = "initialized"
    research_task_id: str | None = None
    research_completed: bool = False
    prompts_queried: bool = False
    created_at: str
    updated_at: str


from enum import Enum
from typing import Any, Optional, Union


# Explicit allowlist — only versions listed here are treated as RESOLVED.
# Any other integer (e.g. 99 from future/corrupt manifests) becomes UNSUPPORTED_VERSION.
SUPPORTED_SCHEMA_VERSIONS: frozenset[int] = frozenset({1})


class ManifestStatus(str, Enum):
    RESOLVED = "resolved"
    NEVER_SEEN = "never_seen"
    MISSING_HISTORY = "missing_history"
    CORRUPT = "corrupt"
    UNSUPPORTED_VERSION = "unsupported_version"
    CONFLICT = "conflict"


class ManifestLoadResult(BaseModel):
    """Typed result of manifest loading to guarantee fail-closed behavior."""
    load_status: ManifestStatus
    manifest: Optional[NotebookLMManifest] = None
    error_message: Optional[str] = None
    path: Optional[str] = None

    @property
    def status(self) -> str:
        if self.manifest is not None:
            return self.manifest.status
        return self.load_status.value

    @property
    def is_resolved(self) -> bool:
        return self.load_status == ManifestStatus.RESOLVED and self.manifest is not None

    @property
    def is_corrupt(self) -> bool:
        return self.load_status in (
            ManifestStatus.CORRUPT,
            ManifestStatus.UNSUPPORTED_VERSION,
            ManifestStatus.CONFLICT,
        )

    def __getattr__(self, item: str) -> Any:
        # A resolved result is a compatibility wrapper around the validated
        # manifest, so legacy callers may still read its fields. Never forward
        # fields for missing, corrupt, or unsupported results: that would turn
        # a recovery blocker into a fresh run.
        if self.load_status == ManifestStatus.RESOLVED and self.manifest is not None:
            try:
                return getattr(self.manifest, item)
            except AttributeError:
                pass
        # Keep the failure explicit for unresolved results and unknown fields.
        raise AttributeError(
            f"'ManifestLoadResult' (load_status={self.load_status!r}) has no attribute {item!r}. "
            "Check .is_resolved or access .manifest directly."
        )


def compute_content_hash(briefing_file_path: Path) -> str:
    """SHA-256 ของเนื้อหาไฟล์ briefing — ใช้เป็น key ของ manifest/resume"""
    return hashlib.sha256(briefing_file_path.read_bytes()).hexdigest()


def manifest_path_for(content_hash: str, *, base_dir: Path | None = None) -> Path:
    """base_dir=None -> ใช้ MANIFEST_DIR ปัจจุบัน (lookup ตอนเรียก ไม่ใช่ตอน def) เพื่อให้ test
    monkeypatch โมดูล-level MANIFEST_DIR แล้วมีผลจริง — ถ้าใส่ default เป็น MANIFEST_DIR ตรงๆ ใน
    signature ค่าจะถูก bind ตอน import ครั้งเดียว monkeypatch ทีหลังจะไม่มีผล
    """
    return (base_dir if base_dir is not None else MANIFEST_DIR) / f"{content_hash}.json"


def _history_index_path(base_dir: Path) -> Path:
    return Path(base_dir) / "history_index.json"


def _load_history_index(base_dir: Path) -> dict[str, Any]:
    path = _history_index_path(base_dir)
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("history index must be an object")
        return value
    except (OSError, ValueError) as exc:
        logger.error("NotebookLM history index is unreadable at %s: %s", path, exc)
        raise RuntimeError(f"NotebookLM history index is corrupt: {path}: {exc}") from exc


def load_manifest(path: Path) -> ManifestLoadResult:
    """โหลด manifest เดิม — คืน ManifestLoadResult (typed result) เสมอ

    ไม่คืน None เมื่อไฟล์เสีย เพื่อป้องกัน caller เข้าใจผิดว่าเป็นการรันครั้งแรก
    """
    if not path.exists():
        try:
            history = _load_history_index(path.parent)
        except RuntimeError as exc:
            return ManifestLoadResult(
                load_status=ManifestStatus.MISSING_HISTORY,
                error_message=str(exc),
                path=str(path),
            )
        if path.stem in history:
            return ManifestLoadResult(
                load_status=ManifestStatus.MISSING_HISTORY,
                error_message="Manifest was previously registered but its durable record is missing",
                path=str(path),
            )
        return ManifestLoadResult(
            load_status=ManifestStatus.NEVER_SEEN,
            path=str(path),
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        # Check schema_version before full validation — unsupported versions must not
        # be treated as RESOLVED even if the rest of the schema matches.
        raw_version = raw.get("schema_version", 1)
        if raw_version not in SUPPORTED_SCHEMA_VERSIONS:
            logger.warning(
                "Manifest schema_version=%r ที่ %s ไม่อยู่ใน supported versions %r",
                raw_version, path, set(SUPPORTED_SCHEMA_VERSIONS),
            )
            return ManifestLoadResult(
                load_status=ManifestStatus.UNSUPPORTED_VERSION,
                error_message=f"schema_version {raw_version} is not supported (supported: {sorted(SUPPORTED_SCHEMA_VERSIONS)})",
                path=str(path),
            )
        m = NotebookLMManifest.model_validate(raw)
        return ManifestLoadResult(
            load_status=ManifestStatus.RESOLVED,
            manifest=m,
            path=str(path),
        )
    except json.JSONDecodeError as e:
        logger.warning("Manifest corrupt (JSONDecodeError) ที่ %s (%s)", path, e)
        return ManifestLoadResult(
            load_status=ManifestStatus.CORRUPT,
            error_message=str(e),
            path=str(path),
        )
    except ValidationError as e:
        logger.warning("Manifest schema mismatch ที่ %s (%s)", path, e)
        return ManifestLoadResult(
            load_status=ManifestStatus.UNSUPPORTED_VERSION,
            error_message=str(e),
            path=str(path),
        )
    except Exception as e:
        logger.warning("Manifest load exception ที่ %s (%s)", path, e)
        return ManifestLoadResult(
            load_status=ManifestStatus.CORRUPT,
            error_message=str(e),
            path=str(path),
        )


def save_manifest(manifest: Union[NotebookLMManifest, ManifestLoadResult], *, base_dir: Path | None = None) -> Path:
    """เขียน manifest แบบ atomic — คืน path ที่บันทึก"""
    target = manifest.manifest if isinstance(manifest, ManifestLoadResult) and manifest.manifest is not None else manifest
    if not isinstance(target, NotebookLMManifest):
        raise ValueError("Cannot save an unresolved ManifestLoadResult")
    target.updated_at = datetime.now(timezone.utc).isoformat()
    path = manifest_path_for(target.content_hash, base_dir=base_dir)
    _atomic_write_to(path, target.model_dump_json(indent=2))
    # Keep a tiny append-like lookup index so deleting/corrupting a manifest
    # cannot make a known source look like a never-seen first run.
    try:
        history_path = _history_index_path(path.parent)
        history = _load_history_index(path.parent)
        history[target.content_hash] = {
            "manifest_path": str(path),
            "briefing_path": target.briefing_path,
            "updated_at": target.updated_at,
        }
        _atomic_write_to(history_path, json.dumps(history, indent=2, ensure_ascii=False))
    except Exception as exc:
        logger.error("NotebookLM history index update failed at %s: %s", path.parent, exc)
    return path


def new_manifest(*, content_hash: str, briefing_path: Path) -> NotebookLMManifest:
    now = datetime.now(timezone.utc).isoformat()
    return NotebookLMManifest(
        content_hash=content_hash,
        briefing_path=str(briefing_path),
        created_at=now,
        updated_at=now,
    )
