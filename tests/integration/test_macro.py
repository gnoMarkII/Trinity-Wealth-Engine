import os
import pytest
from pathlib import Path
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage

from agents.manager_agent import build_graph
from tools.archivist.core import init_vault_structure

load_dotenv()

@pytest.mark.integration
def test_macro_analysis_flow():
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
    
    # We run the stream and track which nodes were activated
    for event in graph.stream(inputs, config=config, stream_mode="updates"):
        for node_name, state in event.items():
            nodes_visited.append(node_name)
            print(f"Node activated: {node_name}")
            
    print("Visited nodes list:", nodes_visited)
    
    # Assertions
    # The workflow should involve: supervisor -> researcher -> supervisor -> macro_analyst -> supervisor -> archivist -> supervisor
    assert "supervisor" in nodes_visited
    assert "researcher" in nodes_visited
    assert "macro_analyst" in nodes_visited
    assert "archivist" in nodes_visited

    # Check that the files were created in Obsidian Vault
    vault_path = Path(os.getenv("OBSIDIAN_VAULT_PATH", "./memories")).resolve()
    daily_snapshots_dir = vault_path / "30_Knowledge_Base" / "Macroeconomics" / "Daily_Snapshots"
    
    # Look for files created today
    today_str = datetime_to_date_str()
    today_dir = daily_snapshots_dir / today_str
    
    assert today_dir.exists()
    
    # Check if there is any Thailand macro snapshot and the final report
    files = [f.name for f in today_dir.glob("*.md")]
    print("Files created in daily snapshots:", files)
    
    # Verify that we see some files created by researcher/archivist
    assert any("Thailand_Macro_Snapshot" in name for name in files)
    assert len(files) >= 2

def datetime_to_date_str():
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d")
