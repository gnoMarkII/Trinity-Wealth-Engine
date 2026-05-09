import os
import time

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from prompt_toolkit import prompt

from core.utils import normalize_content

load_dotenv()


def _display(node_name: str, msg) -> None:
    content = normalize_content(getattr(msg, "content", ""))
    if not content:
        return
    if content.startswith("[Manager → Archivist]"):
        body = content[len("[Manager → Archivist]"):].strip()
        print(f"\n  [Manager → Archivist]: {body}")
    elif content.startswith("[Archivist]"):
        body = content[len("[Archivist]"):].strip()
        print(f"  [Archivist]: {body}")
    elif content.startswith("[Manager → Researcher]"):
        body = content[len("[Manager → Researcher]"):].strip()
        print(f"\n  [Manager → Researcher]: {body}")
    elif content.startswith("[Researcher → Manager]"):
        body = content[len("[Researcher → Manager]"):].strip()
        print(f"  [Researcher → Manager]: {body}")
    else:
        print(f"\n[{node_name}]: {content}")


def main():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "your_google_api_key_here":
        print("ERROR: กรุณาตั้งค่า GOOGLE_API_KEY ใน .env ก่อน")
        return

    from agents.manager_agent import build_graph

    print("=" * 60)
    print("  Investment Manager AI — Multi-Agent System")
    print("  Supervisor: The Manager | Workers: The Archivist, The Researcher")
    print("  พิมพ์ 'quit' 'exit' หรือ 'ออก' เพื่อออกจากโปรแกรม")
    print("=" * 60)

    memory = MemorySaver()
    graph = build_graph(checkpointer=memory)
    config = {"configurable": {"thread_id": "1"}, "recursion_limit": 25}

    while True:
        try:
            user_input = prompt("\nคุณ: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nออกจากโปรแกรม...")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q", "ออก"):
            print("ออกจากโปรแกรม...")
            break

        print("\n[กำลังประมวลผล...]")

        inputs = {"messages": [("user", user_input)]}

        max_retries = 3
        for attempt in range(max_retries):
            try:
                for event in graph.stream(inputs, config=config, stream_mode="updates"):
                    for node_name, state in event.items():
                        if "messages" not in state:
                            continue
                        messages = state["messages"]
                        last_message = messages[-1] if isinstance(messages, list) else messages
                        _display(node_name, last_message)
                break
            except Exception as e:
                err = str(e).lower()
                is_transient = any(
                    k in err for k in ("429", "503", "quota", "high demand", "timeout", "unavailable")
                )
                if is_transient and attempt < max_retries - 1:
                    print(f"\n[System]: เซิร์ฟเวอร์ AI กำลังทำงานหนัก (Attempt {attempt + 1}/{max_retries})... กำลังรอเพื่อเชื่อมต่อใหม่...")
                    time.sleep(2 ** attempt)
                else:
                    if is_transient:
                        print("\n[System]: ไม่สามารถติดต่อ AI ได้ในขณะนี้ โปรดลองพิมพ์คำถามของคุณใหม่อีกครั้ง")
                    else:
                        print(f"\nERROR: {e}")
                    break


if __name__ == "__main__":
    main()
