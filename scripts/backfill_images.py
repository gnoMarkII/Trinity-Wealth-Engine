# -*- coding: utf-8 -*-
"""Backfill `image` property into existing YouTube, Article, and Book notes."""
import os
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8")  # type: ignore
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import frontmatter as fm
import httpx

VAULT = Path("memories")
YT_DIR = VAULT / "30_Knowledge_Base/YouTube_Summaries"
ART_DIR = VAULT / "30_Knowledge_Base/Articles"
BOOKS_DIR = VAULT / "30_Knowledge_Base/Books"


def atomic_write(path: Path, content: str) -> None:
    fd, tmp = tempfile.mkstemp(prefix=f".{path.stem}_", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def set_image(path: Path, image_url: str) -> bool:
    """Add image field to frontmatter if not already present. Returns True if changed."""
    post = fm.load(path)
    if post.metadata.get("image"):
        print(f"  SKIP (already has image): {path.name}")
        return False
    post.metadata["image"] = image_url
    atomic_write(path, fm.dumps(post, sort_keys=False))
    print(f"  OK: {path.name}")
    print(f"      image: {image_url}")
    return True


# ─── YouTube Summaries ─────────────────────────────────────────────────────────
print("\n=== YouTube Summaries ===")
for md in sorted(YT_DIR.rglob("*.md")):
    if any(p.startswith(".") or p in ("Inbox", "Revisions") for p in md.parts):
        continue
    post = fm.load(md)
    video_id = post.metadata.get("video_id")
    if not video_id:
        print(f"  SKIP (no video_id): {md.name}")
        continue
    thumbnail = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
    set_image(md, thumbnail)


# ─── Articles ──────────────────────────────────────────────────────────────────
print("\n=== Articles ===")
try:
    import trafilatura
    _HAS_TRAFILATURA = True
except ImportError:
    _HAS_TRAFILATURA = False
    print("  WARNING: trafilatura not available — articles will be skipped")

for md in sorted(ART_DIR.rglob("*.md")) if ART_DIR.exists() else []:
    if any(p.startswith(".") or p in ("Inbox", "Revisions") for p in md.parts):
        continue
    post = fm.load(md)
    if post.metadata.get("image"):
        print(f"  SKIP (already has image): {md.name}")
        continue
    source_url = post.metadata.get("source_url")
    if not source_url or not _HAS_TRAFILATURA:
        print(f"  SKIP (no source_url or no trafilatura): {md.name}")
        continue
    print(f"  Fetching OG image for: {md.name}")
    try:
        downloaded = trafilatura.fetch_url(source_url)  # type: ignore
        if not downloaded:
            print(f"  FAIL (fetch returned None): {source_url}")
            continue
        meta = trafilatura.extract_metadata(downloaded)
        og_image = meta.image if meta and meta.image else None
        if og_image:
            set_image(md, og_image)
        else:
            print(f"  SKIP (no OG image found): {md.name}")
    except Exception as e:
        print(f"  FAIL: {md.name}: {e}")


# ─── Books ─────────────────────────────────────────────────────────────────────
print("\n=== Books ===")
for md in sorted(BOOKS_DIR.glob("*.md")):
    post = fm.load(md)
    if post.metadata.get("image"):
        print(f"  SKIP (already has image): {md.name}")
        continue
    title = post.metadata.get("title", md.stem)
    author = post.metadata.get("author", "")
    print(f"  Searching Open Library for: {title}")
    try:
        query = f"{title} {author}".strip().replace(" ", "+")
        resp = httpx.get(
            f"https://openlibrary.org/search.json?q={query}&limit=1&fields=cover_i,title",
            timeout=10,
        )
        data = resp.json()
        docs = data.get("docs", [])
        cover_id = next((d.get("cover_i") for d in docs if d.get("cover_i")), None)
        if cover_id:
            image_url = f"https://covers.openlibrary.org/b/id/{cover_id}-L.jpg"
            set_image(md, image_url)
        else:
            print(f"  SKIP (no cover found): {md.name}")
    except Exception as e:
        print(f"  FAIL: {md.name}: {e}")

print("\nDone.")
