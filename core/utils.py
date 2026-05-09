def normalize_content(content) -> str:
    """Normalize LLM message content to str (handles Gemini list[dict] format)."""
    if isinstance(content, list):
        return " ".join(
            p.get("text", "") if isinstance(p, dict) else str(p)
            for p in content
        ).strip()
    return str(content).strip() if content else ""
