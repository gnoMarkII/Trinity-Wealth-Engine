import os
import tempfile
import time
from pathlib import Path
from typing import Any
from core.utils import normalize_content
from tools.archivist.maintenance_guard import assert_write_allowed

def _atomic_write_to(path: Path, content: Any, max_retries: int = 8, backoff: float = 0.05) -> None:
    assert_write_allowed(path)
    """Generic atomic write: temp file → os.replace() — ใช้ได้กับไฟล์ใดก็ได้"""
    if not isinstance(content, str):
        content = normalize_content(content) if isinstance(content, list) else str(content)
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp_", suffix=".md.tmp", dir=str(parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(content)
            f.flush()
        
        last_err: Exception | None = None
        for attempt in range(max_retries):
            try:
                os.replace(tmp_path, path)
                last_err = None
                break
            except (PermissionError, OSError) as e:
                last_err = e
                winerror = getattr(e, "winerror", None)
                if winerror in (5, 32) or isinstance(e, PermissionError):
                    time.sleep(backoff * (1.5 ** attempt))
                else:
                    break
        if last_err is not None:
            raise last_err
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

def _stage_text(path: Path, content: Any) -> Path:
    """Write text to a temporary file in the same directory and return its Path."""
    assert_write_allowed(path)
    if not isinstance(content, str):
        content = normalize_content(content) if isinstance(content, list) else str(content)
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp_", suffix=".tmp", dir=str(parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(content)
        return tmp_path
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
