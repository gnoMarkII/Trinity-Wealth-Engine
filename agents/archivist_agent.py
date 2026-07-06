from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain.agents import create_agent

from tools.archivist.core import read_file
from tools.archivist.writer import save_memory, write_raw_markdown
from tools.archivist.indexer import update_master_index
from tools.archivist.search import search_all_memories, search_graph_context
from tools.archivist.linter import lint_structural_health, lint_semantic_conflict
from core.prompt_harness import get_harness

# ARCHIVIST_SYSTEM_PROMPT ถูกย้ายไปที่ prompts/skills/archivist/SKILL.md ผ่านระบบ PromptHarness

_archivist_tools = [
    write_raw_markdown,
    save_memory,
    search_all_memories,
    search_graph_context,
    read_file,
    update_master_index,
    lint_structural_health,
    lint_semantic_conflict,
]


def create_archivist(model: BaseChatModel | Runnable):
    """สร้าง Archivist ReAct agent พร้อม PKM tools — caller ต้องส่ง model มาเสมอ"""
    return create_agent(
        model=model,
        tools=_archivist_tools,
        system_prompt=get_harness("archivist").get_system_prompt()
    )
