"""Compatibility facade for the NotebookLM HTTP router."""
from application.notebooklm.service import NOTEBOOKLM_SOURCES_DIR, _parse_source_filename
from tools.content.notebooklm.adapter import check_binary_available
from api.routers.notebooklm_router import (
    router,
    list_available_sources,
    generate_notebooklm_audio,
    get_notebooklm_status,
)

__all__ = [
    "router",
    "list_available_sources",
    "generate_notebooklm_audio",
    "get_notebooklm_status",
    "NOTEBOOKLM_SOURCES_DIR",
    "_parse_source_filename",
    "check_binary_available",
]
