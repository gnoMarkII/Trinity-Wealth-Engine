# Agent Layer Guidelines (`agents/`)

This directory contains LangGraph multi-agent workflows and ReAct workers.

---

## 1. 🤖 Architecture & Design Principles

1. **Prompt Management:** All agent system prompts must be managed via `core.prompt_harness.PromptHarness` loading from `prompts/skills/<agent_name>/`.
2. **Tool Registration:**
   - Every ReAct agent must organize its tools using `core.tool_harness.ProgressiveToolRegistry` separating `core_tools` from specialized capability groups (`ToolGroupSpec`).
   - Tools must be wrapped using `wrap_tools_with_harness(registry.all_tools(), ...)` to ensure automatic output truncation and circuit breakers.
3. **Execution Guarding:**
   - When calling sub-graphs from `manager_agent.py`, always wrap worker graph invocation with `with runaway_guard_scope():` to isolate turn state and tripwires.
4. **State Schema:**
   - Keep `AgentState` strictly typed with TypedDict.
   - Use `extract_worker_reply` to extract message contents safely from worker graphs.
5. **Constraint:**
   - Do NOT edit or refactor `agents/bookkeeper_agent.py` in general harness changes (reserved for dedicated work).
