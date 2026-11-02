"""Mock Harness สำหรับ equity_intel pipeline (equity_quant -> equity_narrative -> equity_synthesizer
-> post_equity_intel -> prepare_archivist) — มิเรอร์ tests/integration/test_macro.py::test_macro_analysis_flow_mocked_router
"""
from unittest.mock import patch, MagicMock

from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage

from agents.research_swarm_flow import build_graph, RouterDecision, WorkerTask
from schemas.micro_quant_schemas import EquityNarrativeOutput


class MockGraph:
    def __init__(self, return_val):
        self.return_val = return_val

    def invoke(self, state, *args, **kwargs):
        return self.return_val


class CapturingMockGraph:
    """MockGraph ที่จำ input ที่ได้รับไว้ — ใช้ตรวจสอบว่า instruction ที่ส่งให้ Archivist มีอะไรบ้าง"""

    def __init__(self, return_val):
        self.return_val = return_val
        self.received_inputs = []

    def invoke(self, state, *args, **kwargs):
        self.received_inputs.append(state)
        return self.return_val


@patch("agents.research_swarm_flow._get_archivist_graph")
@patch("agents.research_swarm_flow._get_equity_narrative_graph")
@patch("agents.research_swarm_flow._get_equity_quant_graph")
@patch("agents.research_swarm_flow.get_llm")
def test_equity_intel_flow_mocked_router(mock_llm, mock_eq_quant, mock_eq_narrative, mock_archivist, equity_tmp_vault):

    quant_valid = (
        '{"ticker": "AAPL", "market": "US", "value_score": 70.0, "quality_score": 80.0, '
        '"momentum_score": 60.0, "beta": 1.2, "volatility_pct": 25.0, "mdd_pct": -15.0, '
        '"upside_pct": 10.0, "downside_pct": -5.0, "evaluated_at": "2026-07-20T10:00:00+00:00", '
        '"data_quality_flags": []}'
    )
    mock_eq_quant.return_value = MockGraph({"messages": [AIMessage(content=quant_valid, name="equity_quant")]})

    narrative_valid = (
        '{"evaluated_at": "2026-07-20T10:00:00+00:00", "market_sentiment": "bullish", '
        '"key_themes": ["AI capex"], "tail_risks": [], "sources_summary": "test"}'
    )
    mock_eq_narrative.return_value = MockGraph({"messages": [AIMessage(content=narrative_valid, name="equity_narrative")]})

    mock_archivist.return_value = MockGraph({"messages": [AIMessage(content="saved", name="archivist")]})

    mock_narrative_output = EquityNarrativeOutput(
        narrative_analysis="ราคาหุ้นอยู่ในทิศทางขาขึ้น",
        base_case_summary="คงมุมมองเชิงบวก",
    )
    mock_structured_llm = MagicMock()
    mock_structured_llm.invoke.return_value = mock_narrative_output

    mock_llm_instance = MagicMock()
    mock_llm_instance.with_structured_output.return_value = mock_structured_llm
    mock_llm_instance.invoke.return_value = AIMessage(content="---\ntitle: x\nentity_type: research\n---\nReport")
    mock_llm.return_value = mock_llm_instance

    memory = MemorySaver()
    graph = build_graph(checkpointer=memory)
    config = {"configurable": {"thread_id": "test-equity-intel-1"}, "recursion_limit": 40}

    inputs = {"messages": [("user", "วิเคราะห์หุ้น AAPL แบบเชิงลึก")]}
    nodes_visited = []
    for event in graph.stream(inputs, config=config, stream_mode="updates"):
        for node_name, _state in event.items():
            nodes_visited.append(node_name)

    assert "supervisor" in nodes_visited
    assert "equity_quant" in nodes_visited
    assert "equity_narrative" in nodes_visited
    assert "equity_synthesizer" in nodes_visited
    assert "post_equity_intel" in nodes_visited
    assert "prepare_archivist" in nodes_visited
    assert "archivist" in nodes_visited

    final_state = graph.get_state(config).values
    assert final_state["equity_quant_score"]["ticker"] == "AAPL"
    assert final_state["equity_quant_score"]["value_score"] == 70.0
    assert final_state["equity_narrative_context"]["market_sentiment"] == "bullish"


@patch("agents.research_swarm_flow._get_archivist_graph")
@patch("agents.research_swarm_flow._get_equity_narrative_graph")
@patch("agents.research_swarm_flow._get_equity_quant_graph")
@patch("agents.research_swarm_flow.get_llm")
def test_equity_intel_flow_dictates_deterministic_archivist_filename(
    mock_llm, mock_eq_quant, mock_eq_narrative, mock_archivist, tmp_vault
):
    """prepare_archivist_node ต้องบอก filename ให้ Archivist ตรงๆ จาก title: frontmatter — ไม่ปล่อยให้
    LLM เดาเอง (เจอจริงว่า re-run วันเดียวกันได้ filename ไม่ตรงกัน เช่น space vs underscore
    ทำให้เกิดไฟล์ซ้ำแทนที่จะ overwrite เดิม)"""

    quant_valid = (
        '{"ticker": "AAPL", "market": "US", "value_score": 70.0, "quality_score": 80.0, '
        '"momentum_score": 60.0, "beta": 1.2, "volatility_pct": 25.0, "mdd_pct": -15.0, '
        '"upside_pct": 10.0, "downside_pct": -5.0, "evaluated_at": "2026-07-20T10:00:00+00:00", '
        '"data_quality_flags": []}'
    )
    mock_eq_quant.return_value = MockGraph({"messages": [AIMessage(content=quant_valid, name="equity_quant")]})

    narrative_valid = (
        '{"evaluated_at": "2026-07-20T10:00:00+00:00", "market_sentiment": "bullish", '
        '"key_themes": [], "tail_risks": [], "sources_summary": "test"}'
    )
    mock_eq_narrative.return_value = MockGraph({"messages": [AIMessage(content=narrative_valid, name="equity_narrative")]})

    capturing_archivist = CapturingMockGraph({"messages": [AIMessage(content="saved", name="archivist")]})
    mock_archivist.return_value = capturing_archivist

    mock_narrative_output = EquityNarrativeOutput(
        narrative_analysis="ราคาหุ้นอยู่ในทิศทางขาขึ้น",
        base_case_summary="คงมุมมองเชิงบวก",
    )
    mock_structured_llm = MagicMock()
    mock_structured_llm.invoke.return_value = mock_narrative_output
    mock_llm_instance = MagicMock()
    mock_llm_instance.with_structured_output.return_value = mock_structured_llm
    mock_llm.return_value = mock_llm_instance

    memory = MemorySaver()
    graph = build_graph(checkpointer=memory)
    config = {"configurable": {"thread_id": "test-equity-intel-filename-1"}, "recursion_limit": 40}

    inputs = {"messages": [("user", "วิเคราะห์หุ้น AAPL แบบเชิงลึก")]}
    for _ in graph.stream(inputs, config=config, stream_mode="updates"):
        pass

    assert len(capturing_archivist.received_inputs) == 1
    archivist_instruction = capturing_archivist.received_inputs[0]["messages"][0].content

    assert "ต้องใช้ filename='AAPL Equity Analysis" in archivist_instruction
    assert "ห้ามดัดแปลง" in archivist_instruction


@patch("agents.research_swarm_flow._get_archivist_graph")
@patch("agents.research_swarm_flow._get_equity_narrative_graph")
@patch("agents.research_swarm_flow._get_equity_quant_graph")
@patch("agents.research_swarm_flow.get_llm")
def test_equity_intel_flow_multi_document_archivist_when_valid_news_present(
    mock_llm, mock_eq_quant, mock_eq_narrative, mock_archivist, tmp_vault
):
    """เมื่อมี equity_news_raw ที่มี YAML frontmatter prepare_archivist_node ต้องสร้าง directive 2 ครั้งให้ Archivist"""

    quant_valid = (
        '{"ticker": "AAPL", "market": "US", "value_score": 70.0, "quality_score": 80.0, '
        '"momentum_score": 60.0, "beta": 1.2, "volatility_pct": 25.0, "mdd_pct": -15.0, '
        '"upside_pct": 10.0, "downside_pct": -5.0, "evaluated_at": "2026-07-20T10:00:00+00:00", '
        '"data_quality_flags": []}'
    )
    mock_eq_quant.return_value = MockGraph({"messages": [AIMessage(content=quant_valid, name="equity_quant")]})

    news_md = "---\ntitle: AAPL Latest News 2026-08-08\nentity_type: Company_News\nticker: AAPL\n---\n# AAPL News"
    narrative_valid = (
        '{"evaluated_at": "2026-07-20T10:00:00+00:00", "market_sentiment": "bullish", '
        '"key_themes": [], "tail_risks": [], "sources_summary": "test"}'
    )
    mock_eq_narrative.return_value = MockGraph({
        "messages": [AIMessage(content=narrative_valid, name="equity_narrative")],
        "equity_news_raw": news_md,
    })

    capturing_archivist = CapturingMockGraph({"messages": [AIMessage(content="saved", name="archivist")]})
    mock_archivist.return_value = capturing_archivist

    mock_narrative_output = EquityNarrativeOutput(
        narrative_analysis="ราคาหุ้นอยู่ในทิศทางขาขึ้น",
        base_case_summary="คงมุมมองเชิงบวก",
    )
    mock_structured_llm = MagicMock()
    mock_structured_llm.invoke.return_value = mock_narrative_output
    mock_llm_instance = MagicMock()
    mock_llm_instance.with_structured_output.return_value = mock_structured_llm
    mock_llm.return_value = mock_llm_instance

    memory = MemorySaver()
    graph = build_graph(checkpointer=memory)
    config = {"configurable": {"thread_id": "test-equity-intel-multi-doc-1"}, "recursion_limit": 40}

    inputs = {"messages": [("user", "วิเคราะห์หุ้น AAPL แบบเชิงลึก")]}
    for _ in graph.stream(inputs, config=config, stream_mode="updates"):
        pass

    assert len(capturing_archivist.received_inputs) == 1
    archivist_instruction = capturing_archivist.received_inputs[0]["messages"][0].content

    assert "เรียกใช้เครื่องมือ write_raw_markdown ทั้งหมด 2 ครั้ง" in archivist_instruction
    assert "filename='AAPL Latest News 2026-08-08' เป๊ะๆ ตามนี้เท่านั้น ห้ามดัดแปลง" in archivist_instruction
    assert "filename='AAPL Equity Analysis" in archivist_instruction
    assert "เป๊ะๆ ตามนี้เท่านั้น ห้ามดัดแปลง" in archivist_instruction


@patch("agents.research_swarm_flow._get_archivist_graph")
@patch("agents.research_swarm_flow._get_equity_narrative_graph")
@patch("agents.research_swarm_flow._get_equity_quant_graph")
@patch("agents.research_swarm_flow.get_llm")
def test_equity_intel_flow_invalid_news_without_frontmatter_bypasses_news_saving(
    mock_llm, mock_eq_quant, mock_eq_narrative, mock_archivist, tmp_vault
):
    """ถ้า equity_news_raw เป็น error string ไม่มี frontmatter ต้องข้ามการบันทึกข่าวและไม่ส่งไฟล์ขยะให้ Archivist"""

    quant_valid = (
        '{"ticker": "AAPL", "market": "US", "value_score": 70.0, "quality_score": 80.0, '
        '"momentum_score": 60.0, "beta": 1.2, "volatility_pct": 25.0, "mdd_pct": -15.0, '
        '"upside_pct": 10.0, "downside_pct": -5.0, "evaluated_at": "2026-07-20T10:00:00+00:00", '
        '"data_quality_flags": []}'
    )
    mock_eq_quant.return_value = MockGraph({"messages": [AIMessage(content=quant_valid, name="equity_quant")]})

    error_news_raw = "ERROR: ไม่สามารถดึงข่าวของ AAPL (US) ได้: Network timeout"
    narrative_valid = (
        '{"evaluated_at": "2026-07-20T10:00:00+00:00", "market_sentiment": "bullish", '
        '"key_themes": [], "tail_risks": [], "sources_summary": "test"}'
    )
    mock_eq_narrative.return_value = MockGraph({
        "messages": [AIMessage(content=narrative_valid, name="equity_narrative")],
        "equity_news_raw": error_news_raw,
    })

    capturing_archivist = CapturingMockGraph({"messages": [AIMessage(content="saved", name="archivist")]})
    mock_archivist.return_value = capturing_archivist

    mock_narrative_output = EquityNarrativeOutput(
        narrative_analysis="ราคาหุ้นอยู่ในทิศทางขาขึ้น",
        base_case_summary="คงมุมมองเชิงบวก",
    )
    mock_structured_llm = MagicMock()
    mock_structured_llm.invoke.return_value = mock_narrative_output
    mock_llm_instance = MagicMock()
    mock_llm_instance.with_structured_output.return_value = mock_structured_llm
    mock_llm.return_value = mock_llm_instance

    memory = MemorySaver()
    graph = build_graph(checkpointer=memory)
    config = {"configurable": {"thread_id": "test-equity-intel-invalid-news-1"}, "recursion_limit": 40}

    inputs = {"messages": [("user", "วิเคราะห์หุ้น AAPL แบบเชิงลึก")]}
    for _ in graph.stream(inputs, config=config, stream_mode="updates"):
        pass

    assert len(capturing_archivist.received_inputs) == 1
    archivist_instruction = capturing_archivist.received_inputs[0]["messages"][0].content

    # ต้องเป็น single-document directive สำหรับบทวิเคราะห์เท่านั้น ไม่สั่งบันทึกข่าวขยะ
    assert "write_raw_markdown ทั้งหมด 2 ครั้ง" not in archivist_instruction
    assert "filename='AAPL Equity Analysis" in archivist_instruction


@patch("agents.research_swarm_flow._get_archivist_graph")
@patch("agents.research_swarm_flow._get_equity_narrative_graph")
@patch("agents.research_swarm_flow._get_equity_quant_graph")
@patch("agents.research_swarm_flow.get_llm")
def test_equity_intel_flow_quant_error_short_circuits_to_supervisor(
    mock_llm, mock_eq_quant, mock_eq_narrative, mock_archivist, equity_tmp_vault
):
    """ถ้า equity_quant tool คืน Error string (JSON parse ไม่ผ่าน) ต้องไม่ไปต่อ equity_narrative"""
    mock_eq_quant.return_value = MockGraph(
        {"messages": [AIMessage(content="Error: ไม่สามารถคำนวณ Quant Signals ของ BADTICKER (US) ได้", name="equity_quant")]}
    )
    mock_eq_narrative.return_value = MockGraph({"messages": [AIMessage(content="{}", name="equity_narrative")]})
    mock_archivist.return_value = MockGraph({"messages": [AIMessage(content="saved", name="archivist")]})

    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.return_value = AIMessage(content="fallback")
    mock_llm.return_value = mock_llm_instance

    memory = MemorySaver()
    graph = build_graph(checkpointer=memory)
    config = {"configurable": {"thread_id": "test-equity-intel-error-1"}, "recursion_limit": 40}

    inputs = {"messages": [("user", "วิเคราะห์หุ้น BADTICKER")]}
    nodes_visited = []
    for event in graph.stream(inputs, config=config, stream_mode="updates"):
        for node_name, _state in event.items():
            nodes_visited.append(node_name)

    assert "equity_quant" in nodes_visited
    assert "equity_narrative" not in nodes_visited
    assert "equity_synthesizer" not in nodes_visited


@patch("agents.research_swarm_flow._get_archivist_graph")
@patch("agents.research_swarm_flow._get_equity_narrative_graph")
@patch("agents.research_swarm_flow._get_equity_quant_graph")
@patch("agents.research_swarm_flow.get_llm")
def test_equity_intel_flow_respects_save_to_vault_false(
    mock_llm, mock_eq_quant, mock_eq_narrative, mock_archivist, equity_tmp_vault
):
    """user บอก 'ดูเฉยๆ ไม่ต้องเซฟ' -> save_to_vault=False -> ต้องไม่ไป prepare_archivist/archivist"""

    quant_valid = (
        '{"ticker": "AAPL", "market": "US", "value_score": 70.0, "quality_score": 80.0, '
        '"momentum_score": 60.0, "beta": 1.2, "volatility_pct": 25.0, "mdd_pct": -15.0, '
        '"upside_pct": 10.0, "downside_pct": -5.0, "evaluated_at": "2026-07-20T10:00:00+00:00", '
        '"data_quality_flags": []}'
    )
    mock_eq_quant.return_value = MockGraph({"messages": [AIMessage(content=quant_valid, name="equity_quant")]})

    narrative_valid = (
        '{"evaluated_at": "2026-07-20T10:00:00+00:00", "market_sentiment": "bullish", '
        '"key_themes": [], "tail_risks": [], "sources_summary": "test"}'
    )
    mock_eq_narrative.return_value = MockGraph({"messages": [AIMessage(content=narrative_valid, name="equity_narrative")]})
    mock_archivist.return_value = MockGraph({"messages": [AIMessage(content="saved", name="archivist")]})

    mock_narrative_output = EquityNarrativeOutput(
        narrative_analysis="ราคาหุ้นอยู่ในทิศทางขาขึ้น",
        base_case_summary="คงมุมมองเชิงบวก",
    )
    mock_structured_llm = MagicMock()
    mock_structured_llm.invoke.return_value = mock_narrative_output
    mock_llm_instance = MagicMock()
    mock_llm_instance.with_structured_output.return_value = mock_structured_llm
    mock_llm.return_value = mock_llm_instance

    memory = MemorySaver()
    graph = build_graph(checkpointer=memory)
    config = {"configurable": {"thread_id": "test-equity-intel-no-save-1"}, "recursion_limit": 40}

    inputs = {"messages": [("user", "วิเคราะห์หุ้น AAPL ดูเฉยๆ ไม่ต้องเซฟ")]}
    nodes_visited = []
    for event in graph.stream(inputs, config=config, stream_mode="updates"):
        for node_name, _state in event.items():
            nodes_visited.append(node_name)

    assert "equity_synthesizer" in nodes_visited
    assert "post_equity_intel" in nodes_visited
    assert "prepare_archivist" not in nodes_visited
    assert "archivist" not in nodes_visited

    final_state = graph.get_state(config).values
    assert final_state["equity_save_to_vault"] is False
    # ผลวิเคราะห์เต็มต้องยังกลับไปหา user ได้ถึงแม้ไม่เซฟ (มิเรอร์ post_researcher_node no-save branch)
    assert "ราคาหุ้นอยู่ในทิศทางขาขึ้น" in final_state["messages"][-1].content


@patch("agents.research_swarm_flow.log_worker_result")
@patch("agents.research_swarm_flow._get_archivist_graph")
@patch("agents.research_swarm_flow._get_equity_narrative_graph")
@patch("agents.research_swarm_flow._get_equity_quant_graph")
@patch("agents.research_swarm_flow.get_llm")
def test_equity_intel_supervisor_node_resets_equity_news_raw(
    mock_llm, mock_eq_quant, mock_eq_narrative, mock_archivist, mock_log_worker, equity_tmp_vault
):
    """เมื่อเริ่มรัน equity_intel รอบใหม่ supervisor_node ต้อง reset equity_news_raw = None เสมอ กัน stale data ข้ามรอบ"""
    mock_llm.return_value = MagicMock()
    mock_eq_quant.return_value = MockGraph({"messages": [AIMessage(content="{}", name="equity_quant")]})

    memory = MemorySaver()
    graph = build_graph(checkpointer=memory)
    config = {"configurable": {"thread_id": "test-reset-news-raw-1"}, "recursion_limit": 40}

    # จำลอง state เริ่มต้นว่ามี equity_news_raw ตกค้างมาจากรอบก่อน
    initial_state = {"messages": [("user", "วิเคราะห์ TSLA")], "equity_news_raw": "OLD STALE NEWS FROM PREVIOUS RUN"}
    
    # รัน 1 step เพื่อผ่าน supervisor_node
    events = list(graph.stream(initial_state, config=config, stream_mode="updates"))
    assert len(events) > 0
    supervisor_update = events[0].get("supervisor")
    assert supervisor_update is not None
    assert supervisor_update.get("equity_news_raw") is None
    assert supervisor_update.get("equity_output") is None


@patch("agents.research_swarm_flow.log_worker_result")
@patch("agents.research_swarm_flow.fetch_and_save_macro_snapshots")
@patch("agents.research_swarm_flow.write_strategy_json_sidecar")
@patch("agents.research_swarm_flow.format_macro_strategy_report")
@patch("agents.research_swarm_flow.invoke_strategic_allocator")
@patch("agents.research_swarm_flow._get_archivist_graph")
@patch("agents.research_swarm_flow._get_equity_narrative_graph")
@patch("agents.research_swarm_flow._get_equity_quant_graph")
@patch("agents.research_swarm_flow._get_macro_economist_graph")
@patch("agents.research_swarm_flow._get_macro_quant_graph")
@patch("agents.research_swarm_flow.get_llm")
def test_macro_intel_remains_single_document_archivist(
    mock_llm, mock_macro_quant, mock_macro_econ, mock_eq_quant, mock_eq_narrative, mock_archivist, mock_invoke_alloc, mock_fmt_report, mock_sidecar, mock_snapshots, mock_log_worker, equity_tmp_vault
):
    """ยืนยันว่า macro_intel ยังคงใช้ single-document archivist directive พร้อมคำเตือนห้ามดัดแปลงตามเดิม"""
    mock_llm.return_value = MagicMock()
    quant_valid = '{"evaluated_at": "2024-05-20T10:00:00Z", "regions": {"USA": {"growth_score": 0.5, "inflation_score": 0.2, "monetary_score": 0.0, "economic_state": "Goldilocks", "confidence": 0.8}}, "global_geopolitics_score": -0.2, "recession_probability": 0.1, "data_freshness_note": "Test"}'
    mock_macro_quant.return_value = MockGraph({"messages": [AIMessage(content=quant_valid, name="macro_quant")]})
    macro_report = "---\ntitle: Global Macro Snapshot 2026-08-08\nentity_type: Macroeconomics\n---\n# Global Macro"
    mock_macro_econ.return_value = MockGraph({"messages": [AIMessage(content=macro_report, name="macro_economist")]})
    
    from schemas.macro_schemas import MacroStrategyDirection, AssetAllocationView, AssetStance, EconomicState
    mock_strategy = MacroStrategyDirection(
        evaluated_at="2024-05-20T10:00:00Z",
        overall_regime=EconomicState.GOLDILOCKS,
        asset_allocation=[
            AssetAllocationView(asset_class="equity", asset_bucket="equities", stance=AssetStance.NEUTRAL, rationale="Test")
        ],
        focus_themes=["dummy"],
        conviction_level="medium",
        conviction_rationale="Test",
        quant_narrative_alignment="aligned",
        divergence_note=""
    )
    mock_strategy.source_files = []
    mock_invoke_alloc.return_value = mock_strategy
    mock_sidecar.return_value = (None, {})
    mock_fmt_report.return_value = macro_report

    capturing_archivist = CapturingMockGraph({"messages": [AIMessage(content="saved", name="archivist")]})
    mock_archivist.return_value = capturing_archivist

    memory = MemorySaver()
    graph = build_graph(checkpointer=memory)
    config = {"configurable": {"thread_id": "test-macro-single-doc-1"}, "recursion_limit": 40}

    inputs = {"messages": [("user", "วิเคราะห์ Macro")]}
    for _ in graph.stream(inputs, config=config, stream_mode="updates"):
        pass

    assert len(capturing_archivist.received_inputs) == 1
    archivist_instruction = capturing_archivist.received_inputs[0]["messages"][0].content

    # ต้องไม่ใช้ 2 ครั้งแบบ equity_intel
    assert "เรียกใช้เครื่องมือ write_raw_markdown ทั้งหมด 2 ครั้ง" not in archivist_instruction
    assert "filename='Global Macro Snapshot 2026-08-08' เป๊ะๆ ตามนี้เท่านั้น ห้ามดัดแปลง" in archivist_instruction

