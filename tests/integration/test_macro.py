import os
import pytest
from pathlib import Path
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage

from agents.manager_agent import build_graph
from tools.archivist.core import init_vault_structure
from unittest.mock import patch, MagicMock
from agents.manager_agent import RouterDecision, WorkerTask

load_dotenv()

@pytest.mark.integration
@pytest.mark.skip(reason="Live provider test consumes 250k+ tokens; run manually")
def test_macro_analysis_flow_live():
    # Ensure Google API key is set
    if not os.getenv("GOOGLE_API_KEY"):
        pytest.skip("GOOGLE_API_KEY is not configured, skipping integration test.")

    # Init vault
    init_vault_structure()

    memory = MemorySaver()
    graph = build_graph(checkpointer=memory)
    config = {"configurable": {"thread_id": "test-session-1"}, "recursion_limit": 40}

    # Query asking to fetch data and evaluate macro economic states
    user_input = "กรุณาดึงข้อมูลเศรษฐกิจของประเทศไทย สหรัฐอเมริกา และสภาวะเศรษฐกิจภูมิภาคต่างๆ จากนั้นทำการวิเคราะห์สภาวะเศรษฐกิจด้วย Macro Matrix พร้อมบันทึกรายงานลงคลังความรู้"
    inputs = {"messages": [("user", user_input)]}

    nodes_visited = []
    
    for event in graph.stream(inputs, config=config, stream_mode="updates"):
        for node_name, state in event.items():
            nodes_visited.append(node_name)
            print(f"Node activated: {node_name}")
            
    print("Visited nodes list:", nodes_visited)
    
    assert "supervisor" in nodes_visited
    assert "macro_quant" in nodes_visited
    assert "macro_economist" in nodes_visited
    assert "strategic_allocator" in nodes_visited
    assert "archivist" in nodes_visited

    vault_path = Path(os.getenv("OBSIDIAN_VAULT_PATH", "./memories")).resolve()
    daily_snapshots_dir = vault_path / "30_Knowledge_Base" / "Macroeconomics" / "Daily_Snapshots"
    today_str = datetime_to_date_str()
    today_dir = daily_snapshots_dir / today_str
    
    assert today_dir.exists()
    files = [f.name for f in today_dir.glob("*.md")]
    print("Files created in daily snapshots:", files)
    
    assert any("Thailand_Macro_Snapshot" in name for name in files)
    assert len(files) >= 2

def datetime_to_date_str():
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d")

@patch("agents.manager_agent._get_researcher_graph")
@patch("agents.manager_agent._get_macro_quant_graph")
@patch("agents.manager_agent._get_macro_economist_graph")
@patch("agents.manager_agent._get_archivist_graph")
@patch("agents.manager_agent._get_router_model")
@patch("agents.manager_agent.get_llm")
def test_macro_analysis_flow_mocked_router(mock_llm, mock_router, mock_archivist, mock_eco, mock_quant, mock_res):
    class MockRouterModel:
        def invoke(self, *args, **kwargs):
            return RouterDecision(
                tasks=[
                    WorkerTask(target="researcher", instruction="test", save_to_vault=True),
                    WorkerTask(target="macro_intel", instruction="test")
                ]
            )
        def with_structured_output(self, *args, **kwargs):
            return self
        def with_fallbacks(self, *args, **kwargs):
            return self

    mock_router.return_value = MockRouterModel()
    
    class MockGraph:
        def __init__(self, return_val):
            self.return_val = return_val
        def __call__(self, state, *args, **kwargs):
            return self.return_val
        def invoke(self, state, *args, **kwargs):
            return self.return_val

    mock_res.return_value = MockGraph({"messages": [AIMessage(content="research done", name="researcher")]})
    
    quant_valid = '{"evaluated_at": "2024-05-20T10:00:00Z", "regions": {"USA": {"growth_score": 0.5, "inflation_score": 0.2, "monetary_score": 0.0, "economic_state": "Goldilocks", "confidence": 0.8}}, "global_geopolitics_score": -0.2, "recession_probability": 0.1, "data_freshness_note": "Test"}'
    mock_quant.return_value = MockGraph({"messages": [AIMessage(content=quant_valid, name="macro_quant")]})
    
    eco_valid = '{"evaluated_at": "2024-05-20T10:00:00Z", "dominant_themes": [], "market_sentiment": "neutral", "tail_risks": [], "policy_signals": [], "key_narratives_by_region": {}, "sources_summary": "Test"}'
    mock_eco.return_value = MockGraph({"messages": [AIMessage(content=eco_valid, name="macro_economist")]})
    
    mock_archivist.return_value = MockGraph({"messages": [AIMessage(content="saved", name="archivist")]})
    
    # Mock strategic allocator LLM
    # Use a fully formed MacroStrategyDirection to mock the structured output of the Strategic Allocator
    from schemas.macro_schemas import MacroStrategyDirection, AssetAllocationView, AssetStance, EconomicState
    mock_strategy = MacroStrategyDirection(
        evaluated_at="2024-05-20T10:00:00Z",
        overall_regime=EconomicState.GOLDILOCKS,
        asset_allocation=[
            AssetAllocationView(asset_class="equity", stance=AssetStance.NEUTRAL, rationale="Test")
        ],
        focus_themes=["dummy"],
        conviction_level="medium",
        conviction_rationale="Test",
        quant_narrative_alignment="aligned",
        divergence_note=""
    )
    
    mock_structured_llm = MagicMock()
    mock_structured_llm.invoke.return_value = mock_strategy
    
    mock_llm_instance = MagicMock()
    mock_llm_instance.with_structured_output.return_value = mock_structured_llm
    # We still need a default invoke for the generic nodes like supervisor
    mock_llm_instance.invoke.return_value = AIMessage(content="---\ntitle: Macro Report\nentity_type: research\n---\nReport")
    mock_llm.return_value = mock_llm_instance
    
    memory = MemorySaver()
    graph = build_graph(checkpointer=memory)
    config = {"configurable": {"thread_id": "test-mocked-1"}, "recursion_limit": 40}

    inputs = {"messages": [("user", "test query")]}
    nodes_visited = []
    
    for event in graph.stream(inputs, config=config, stream_mode="updates"):
        for node_name, state in event.items():
            nodes_visited.append(node_name)
            
    assert "supervisor" in nodes_visited
    assert "researcher" in nodes_visited
    assert "macro_quant" in nodes_visited
    assert "macro_economist" in nodes_visited
    assert "strategic_allocator" in nodes_visited
    assert "prepare_archivist" in nodes_visited
    assert "archivist" in nodes_visited
    
    final_state = graph.get_state(config).values
    assert "quant_score" in final_state and final_state["quant_score"] is not None
    assert "USA" in final_state["quant_score"]["regions"]
    assert "narrative_context" in final_state and final_state["narrative_context"] is not None
    assert final_state["narrative_context"]["market_sentiment"] == "neutral"
