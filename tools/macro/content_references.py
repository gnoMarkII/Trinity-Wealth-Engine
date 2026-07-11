"""Collect safe preview metadata for content already used by the macro pipeline."""

from __future__ import annotations

import hashlib
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tools.archivist.parser import extract_yaml_frontmatter_value


def _reference_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def news_references_from_radar(markdown: str, max_items: int = 5) -> list[dict]:
    """Parse the News Radar table that was just supplied to the economist agent."""
    references: list[dict] = []
    for line in markdown.splitlines():
        if not line.startswith("|") or "](http" not in line:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 4:
            continue
        match = re.search(r"\[(?P<title>.+?)\]\((?P<url>https://[^)]+)\)", cells[1])
        if not match:
            continue
        age_match = re.search(r"Age:\s*(\d+)h", cells[2])
        url = match.group("url")
        references.append(
            {
                "reference_id": _reference_id("news", url),
                "kind": "news",
                "title": match.group("title").replace("~~", "").strip(),
                "url": url,
                "publisher": re.sub(r"\s*\(Sources:.*$", "", cells[3]).strip(),
                "age_hours": int(age_match.group(1)) if age_match else None,
                "summary": "ข่าวเศรษฐกิจมหภาคที่ใช้ประกอบการประเมินรอบนี้",
                "is_stale": "STALE" in cells[2].upper(),
                "related_observable_ids": [],
            }
        )
        if len(references) == max_items:
            break
    return references


def recent_youtube_references(lookback_days: int = 14, max_items: int = 5) -> list[dict]:
    """Read local YouTube insights, without making another network request."""
    vault_path = Path(os.getenv("OBSIDIAN_VAULT_PATH", "./memories")).resolve()
    summaries_dir = vault_path / "30_Knowledge_Base" / "YouTube_Summaries"
    if not summaries_dir.exists():
        return []

    cutoff_date = datetime.now(timezone.utc).date() - timedelta(days=lookback_days)
    collected: list[tuple[datetime, dict]] = []
    for markdown_file in summaries_dir.glob("*.md"):
        try:
            content = markdown_file.read_text(encoding="utf-8")
            if extract_yaml_frontmatter_value(content, "entity_type") != "youtube_insight":
                continue
            published = extract_yaml_frontmatter_value(content, "date")
            published_at = datetime.strptime(published[:10], "%Y-%m-%d") if published else None
            if published_at is None or published_at.date() < cutoff_date:
                continue
            video_id = extract_yaml_frontmatter_value(content, "video_id")
            if not video_id or not re.fullmatch(r"[\w-]{11}", video_id):
                continue
            channel = extract_yaml_frontmatter_value(content, "channel") or "YouTube"
            url = extract_yaml_frontmatter_value(content, "source_url") or f"https://www.youtube.com/watch?v={video_id}"
            thumbnail_url = extract_yaml_frontmatter_value(content, "image") or f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"
            bullets = [line[2:].strip() for line in content.splitlines() if line.startswith("- ")][:3]
            collected.append(
                (
                    published_at,
                    {
                        "reference_id": f"youtube_{video_id}",
                        "kind": "youtube",
                        "title": f"YouTube insight: {channel}",
                        "url": url,
                        "publisher": channel,
                        "published_at": published_at.date().isoformat(),
                        "summary": " ".join(bullets),
                        "thumbnail_url": thumbnail_url,
                        "is_stale": False,
                        "related_observable_ids": [],
                    },
                )
            )
        except (OSError, ValueError):
            continue

    collected.sort(key=lambda item: item[0], reverse=True)
    return [reference for _, reference in collected[:max_items]]
