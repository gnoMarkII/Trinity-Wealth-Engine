from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain.agents import create_agent

from tools.archivist.core import read_file, read_note_chunk
from tools.archivist.writer import save_memory, write_raw_markdown
from tools.archivist.indexer import update_master_index
from tools.archivist.search import (
    search_all_memories,
    search_graph_context,
    search_memories_with_evidence,
)
from tools.archivist.linter import lint_structural_health, lint_semantic_conflict
from core.prompt_harness import get_harness

from core.tool_harness import (
    ProgressiveToolRegistry,
    ToolGroupSpec,
    wrap_tools_with_harness,
)

# ARCHIVIST_SYSTEM_PROMPT ถูกย้ายไปที่ prompts/skills/archivist/SKILL.md ผ่านระบบ PromptHarness

ARCHIVIST_TOOL_REGISTRY = ProgressiveToolRegistry(
    agent_name="archivist",
    core_tools=[read_file, read_note_chunk, search_all_memories],
    groups=[
        ToolGroupSpec(
            name="write_group",
            tools=(write_raw_markdown, save_memory, update_master_index),
            keywords=("บันทึก", "save", "write_raw_markdown", "เขียน", "สร้าง"),
            description="Write and modify markdown notes in Obsidian vault",
        ),
        ToolGroupSpec(
            name="deep_search_group",
            tools=(search_memories_with_evidence, search_graph_context),
            keywords=("ค้นหา", "search", "ความเชื่อมโยง", "graph", "วิเคราะห์"),
            description="Deep evidence and graph search across memories",
        ),
        ToolGroupSpec(
            name="lint_group",
            tools=(lint_structural_health, lint_semantic_conflict),
            keywords=("lint", "ตรวจสุขภาพ", "orphan", "ขัดแย้ง", "health"),
            description="Lint and health checks for vault notes",
        ),
    ],
)

_EXEMPT_ARCHIVIST_TOOLS = {"write_raw_markdown", "save_memory"}
_archivist_tools = wrap_tools_with_harness(
    ARCHIVIST_TOOL_REGISTRY.all_tools(),
    exempt_tool_names=_EXEMPT_ARCHIVIST_TOOLS,
)


def get_archivist_tools(instruction: str | None = None) -> list:
    """Returns progressively selected tools wrapped with runtime harness."""
    tools = ARCHIVIST_TOOL_REGISTRY.select_tools(instruction)
    return wrap_tools_with_harness(tools, exempt_tool_names=_EXEMPT_ARCHIVIST_TOOLS)


def create_archivist(model: BaseChatModel | Runnable, instruction: str | None = None):
    """สร้าง Archivist ReAct agent พร้อม PKM tools — caller ต้องส่ง model มาเสมอ"""
    active_tools = get_archivist_tools(instruction)
    return create_agent(
        model=model,
        tools=active_tools,
        system_prompt=get_harness("archivist").get_system_prompt(),
    )
