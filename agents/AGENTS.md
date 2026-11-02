# Agent Layer Guidelines (`agents/`)

This directory contains LangGraph multi-agent workflows and ReAct workers.

---

## 1. 🤖 Architecture & Design Principles

1. **Prompt Management:** All agent system prompts must be managed via `core.prompt_harness.PromptHarness` loading from `prompts/skills/<agent_name>/`.
2. **Tool Registration:**
   - Every ReAct agent must organize its tools using `core.tool_harness.ProgressiveToolRegistry` separating `core_tools` from specialized capability groups (`ToolGroupSpec`).
   - Tools must be wrapped using `wrap_tools_with_harness(registry.all_tools(), ...)` to ensure automatic output truncation and circuit breakers.
3. **Execution Guarding:**
   - When calling sub-graphs from `research_swarm_flow.py`, always wrap worker graph invocation with `with runaway_guard_scope():` to isolate turn state and tripwires.
4. **State Schema:**
   - Keep `AgentState` strictly typed with TypedDict.
   - Use `extract_worker_reply` to extract message contents safely from worker graphs.
5. **Constraint:**
   - Legacy `bookkeeper_agent.py` is decommissioned and removed.
6. **Manager Role Isolation:**
   - Hermes is the single, centralized Manager (conversational advisor and execution brain in `application/manager/manager_chat_service.py`).
   - The background research pipeline is `research_swarm_flow.py` using a deterministic Python task dispatcher (`classify_research_tasks`), not an LLM router.
