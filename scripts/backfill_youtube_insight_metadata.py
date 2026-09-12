"""Script สำหรับ Backfill Metadata ของ YouTube Insights ใน Obsidian Vault

- ดึง original_title, channel, published_at จาก YouTube ผ่าน _fetch_youtube_metadata
- แยก published_at ออกจาก ingested_at เด็ดขาด (หากดึงไม่ได้ตั้ง unverified / published_at=None)
- อัปเดตเฉพาะ Frontmatter ไม่แตะเนื้อหา Body
- รองรับ --dry-run และ --file
"""
import argparse
from datetime import datetime
import os
from pathlib import Path
import re
import sys
import yaml

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.logger import get_logger
from tools._atomic_io import _atomic_write_to
from tools.archivist.core import VAULT_PATH
from tools.archivist.parser import parse_frontmatter_metadata
from tools.knowledge.youtube import _extract_video_id, _fetch_youtube_metadata

logger = get_logger(__name__)


def backfill_file(file_path: Path, dry_run: bool = False) -> bool:
    if not file_path.exists() or not file_path.name.endswith(".md"):
        logger.warning("Invalid file path: %s", file_path)
        return False

    content = file_path.read_text(encoding="utf-8")
    meta = parse_frontmatter_metadata(content)

    video_id = meta.get("video_id")
    source_url = meta.get("source_url") or ""
    if not video_id:
        video_id = _extract_video_id(source_url) or _extract_video_id(file_path.stem)
    if not video_id:
        logger.warning("Could not extract video_id for %s", file_path.name)
        return False

    # Split frontmatter from body
    body = content
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            body = parts[2]

    # Fetch live metadata
    yt_meta = _fetch_youtube_metadata(video_id)

    # Prepare updated frontmatter
    channel_name = yt_meta.channel or meta.get("channel") or "Unknown_Channel"
    title_display = yt_meta.original_title or meta.get("original_title") or meta.get("title") or f"YouTube Insight {video_id}"
    date_str = meta.get("date") or datetime.now().strftime("%Y-%m-%d")
    ingested_at = meta.get("ingested_at") or meta.get("last_updated") or date_str

    # Extract [[TICKER]] from body
    raw_tickers = re.findall(r"\[\[([A-Z0-9\.\-\:\^]+)\]\]", body)
    existing_tickers = meta.get("extracted_tickers") or []
    if isinstance(existing_tickers, list):
        for tk in raw_tickers:
            if tk not in existing_tickers:
                existing_tickers.append(tk)

    updated_meta = {
        "title": title_display,
        "original_title": yt_meta.original_title or meta.get("original_title"),
        "channel": channel_name,
        "video_id": video_id,
        "source_url": yt_meta.source_url or source_url,
        "image": f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg",
        "published_at": yt_meta.published_at or meta.get("published_at"),
        "ingested_at": str(ingested_at),
        "date": yt_meta.published_at or meta.get("published_at") or date_str,
        "entity_type": "youtube_insight",
        "metadata_source": yt_meta.metadata_source,
        "verification_status": yt_meta.verification_status,
        "extracted_tickers": existing_tickers,
        "extracted_themes": meta.get("extracted_themes") or [],
        "key_metrics": meta.get("key_metrics") or "",
        "financial_contradictions": meta.get("financial_contradictions") or "",
        "tags": meta.get("tags") or ["youtube", "transcript", "investment_insight"],
    }

    fm_str = yaml.safe_dump(updated_meta, allow_unicode=True, sort_keys=False).strip()
    new_content = f"---\n{fm_str}\n---{body}"

    if dry_run:
        logger.info("[DRY-RUN] Would update %s -> status=%s, title=%s, pub=%s", file_path.name, yt_meta.verification_status, updated_meta["original_title"], updated_meta["published_at"])
        return True

    # Backup original file
    backup_path = file_path.with_suffix(".md.bak")
    backup_path.write_text(content, encoding="utf-8")

    try:
        _atomic_write_to(file_path, new_content)
        if backup_path.exists():
            backup_path.unlink()
        logger.info("Successfully backfilled %s -> status=%s, title=%s", file_path.name, yt_meta.verification_status, updated_meta["original_title"])
        return True
    except Exception as e:
        logger.error("Failed writing %s: %s", file_path.name, e)
        if backup_path.exists():
            backup_path.replace(file_path)
        return False


def main():
    parser = argparse.ArgumentParser(description="Backfill YouTube Insight Metadata in Obsidian Vault")
    parser.add_argument("--dry-run", action="store_true", help="Perform a dry run without modifying files")
    parser.add_argument("--file", type=str, help="Target specific Markdown file path")
    parser.add_argument("--target-id", type=str, help="Target specific video ID")
    args = parser.parse_args()

    yt_dir = Path(VAULT_PATH) / "30_Knowledge_Base" / "YouTube_Summaries"
    if not yt_dir.exists():
        logger.error("YouTube Summaries directory not found at %s", yt_dir)
        sys.exit(1)

    if args.file:
        files = [Path(args.file)]
    elif args.target_id:
        files = list(yt_dir.rglob(f"*{args.target_id}*.md"))
    else:
        files = [
            f for f in yt_dir.rglob("*.md")
            if not any(p.startswith(".") or p in ("Inbox", "Revisions") for p in f.parts)
        ]

    logger.info("Starting YouTube metadata backfill for %d files (dry_run=%s)", len(files), args.dry_run)
    success_count = 0
    for f in files:
        if backfill_file(f, dry_run=args.dry_run):
            success_count += 1

    logger.info("Completed backfill: %d/%d processed successfully", success_count, len(files))


if __name__ == "__main__":
    main()
