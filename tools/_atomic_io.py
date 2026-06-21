import tempfile
import os
from pathlib import Path

def _atomic_write_to(path: Path, content: str) -> None:
    """Generic atomic write: temp file → os.replace() — ใช้ได้กับไฟล์ใดก็ได้"""
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp_", suffix=".md.tmp", dir=str(parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


