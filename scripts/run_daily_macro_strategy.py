import os
import sys
import time
import uuid
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv

# Add project root to sys.path
project_root = Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langgraph.checkpoint.memory import MemorySaver
from core.logger import setup_logging
from core.utils import normalize_content
from tools.archivist.core import init_vault_structure
from agents.manager_agent import build_graph
from langchain_core.messages import HumanMessage, AIMessage

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    load_dotenv()
    setup_logging()
    
    print("=" * 70)
    print("  🏛️ Institutional Grade Strategic Allocator — Daily Pipeline")
    print("  Executing 4-Layer Architecture & 7-Section Dashboard Formatting")
    print("=" * 70)
    
    init_vault_structure()
    
    memory = MemorySaver()
    graph = build_graph(checkpointer=memory)
    
    today_str = datetime.now().strftime("%Y-%m-%d")
    turn_id = str(uuid.uuid4())
    config = {
        "configurable": {"thread_id": f"daily-macro-{today_str}-{turn_id}"},
        "recursion_limit": 150,
        "tags": ["daily-macro-strategy", "institutional-grade"],
        "metadata": {"run_type": "daily_pipeline", "job_id": turn_id}
    }
    
    if len(sys.argv) > 1:
        prompt = sys.argv[1]
    else:
        prompt = (
            f"กรุณาดึงข้อมูลเศรษฐกิจมหภาคและวิเคราะห์สภาวะเศรษฐกิจปัจจุบันของสหรัฐอเมริกา ยุโรป จีน ญี่ปุ่น และไทย "
            f"โดยสังเคราะห์ข้อมูล Quant Matrix และ Narrative Context จากนั้นให้ใช้งานโหนด strategic_allocator เพื่อกำหนดทิศทางกลยุทธ์การลงทุนมหภาค "
            f"(Macro Strategy Direction) ประจำวันที่ {today_str} ตามมาตรฐาน Institutional Grade 4 ชั้น ครอบคลุมสินทรัพย์ 5 กลุ่มหลัก "
            f"พร้อมระบุความน่าจะเป็นของ Regime (Regime Probabilities), ความมั่นใจ 3 แกน, Benchmark Delta, "
            f"แผนเทรดคู่ (Pair Trades) และแผนป้องกันความเสี่ยง (Hedging Plan) ที่ส่งออเดอร์ได้จริง ห้ามข้ามขั้นตอนของ strategic_allocator"
        )
    
    print(f"\n[Prompt]: {prompt}\n")
    print("🚀 เริ่มประมวลผลระบบ Multi-Agent Pipeline...\n")
    
    inputs = {
        "messages": [("user", prompt)],
        "macro_run_id": turn_id,
        "macro_run_started_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "macro_task_run_id": None,
        "macro_task_started_at": None,
        "macro_task_sequence": 0,
        "sector_rotation_snapshot_id": None,
        "sector_rotation_context": None,
        "sector_context_status": None,
        "quant_raw": None,
        "quant_score": None,
        "narrative_raw": None,
        "narrative_context": None,
        "task_queue": [],
        "replan_count": 0,
        "route_meta": {},
    }
    
    visited_nodes = []
    start_time = time.time()
    
    from core.retry import is_transient_error
    import random
    max_retries = 10
    for attempt in range(max_retries):
        try:
            for event in graph.stream(inputs, config=config, stream_mode="updates"):
                for node_name, state in event.items():
                    visited_nodes.append(node_name)
                    print(f"🔹 [Node Completed]: {node_name}")
                    if isinstance(state, dict) and "messages" in state:
                        messages = state["messages"]
                        if messages:
                            last_msg = messages[-1] if isinstance(messages, list) else messages
                            if isinstance(last_msg, AIMessage):
                                content = normalize_content(getattr(last_msg, "content", ""))
                                if content and len(content) < 500:
                                    print(f"   💬 Summary: {content.strip()[:200]}...")
                                elif content:
                                    print(f"   📄 Output generated ({len(content)} characters)")
            break
        except Exception as e:
            if is_transient_error(e) and attempt < max_retries - 1:
                sleep_seconds = int(min(20 * (attempt + 1), 60) + random.uniform(1, 5))
                print(f"\n⚠️ ติด Rate Limit หรือเซิร์ฟเวอร์ AI ไม่ตอบสนอง ({type(e).__name__}): {e}")
                print(f"⏳ กำลังหน่วงเวลา {sleep_seconds} วินาทีเพื่อข้ามช่วงคลายโควตา (Attempt {attempt + 1}/{max_retries})...")
                time.sleep(sleep_seconds)
                continue
            print(f"\n❌ เกิดข้อผิดพลาดระหว่างรัน Pipeline: {e}")
            import traceback
            traceback.print_exc()
            return 1

    elapsed = time.time() - start_time
    print(f"\n✅ Pipeline stream finished in {elapsed:.2f} seconds!")
    print(f"📍 Completed nodes: {visited_nodes}")
    
    # 1. Assert required execution nodes were completed
    required_nodes = ["strategic_allocator", "prepare_archivist", "archivist"]
    missing_nodes = [node for node in required_nodes if node not in visited_nodes]
    if missing_nodes:
        print(f"\n❌ [FAIL] Missing required workflow nodes: {missing_nodes}")
        return 1

    # 2. Check and assert Obsidian Vault files (using rglob across year/month hierarchies)
    vault_path = Path(os.getenv("OBSIDIAN_VAULT_PATH", "./memories")).resolve()
    daily_dir = vault_path / "30_Knowledge_Base" / "Macroeconomics" / "Daily_Snapshots"
    
    print(f"\n🔍 Verifying output files in Obsidian Vault: {daily_dir}")
    if not daily_dir.exists():
        print(f"❌ [FAIL] Daily_Snapshots directory not found at {daily_dir}")
        return 1

    all_snapshots = sorted(daily_dir.rglob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not all_snapshots:
        print(f"❌ [FAIL] No JSON snapshot files found in {daily_dir} or subdirectories")
        return 1

    recent_snapshots = all_snapshots[:3]
    for s in recent_snapshots:
        print(f"   📄 Verified snapshot: {s.name} ({s.stat().st_size} bytes, modified {datetime.fromtimestamp(s.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')})")

    # 3. Assert latest strategy report is committed and loadable
    try:
        from tools.macro.adapters.strategy_vault_adapter import StrategyVaultAdapter
        vault_adapter = StrategyVaultAdapter(vault_path)
        latest_report = vault_adapter.latest()
        report_id = latest_report.get("strategy_report_id")
        regime = latest_report.get("overall_regime")
        print(f"\n🏆 [COMMIT VERIFIED] Latest Strategy Report ID: {report_id}")
        print(f"   Overall Macro Regime: {regime}")
        print(f"   Observables in registry: {len(latest_report.get('observable_registry', {}))}")
    except Exception as exc:
        print(f"\n❌ [FAIL] Failed to load latest committed strategy report from vault: {exc}")
        return 1

    print("\n[PASS] Macro daily strategy pipeline executed and verified successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
