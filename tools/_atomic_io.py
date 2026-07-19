import tempfile
import os
from pathlib import Path

from typing import Any
from core.utils import normalize_content

def _atomic_write_to(path: Path, content: Any) -> None:
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
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


