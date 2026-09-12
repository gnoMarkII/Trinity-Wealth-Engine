"""ทดสอบ ingest books (3) + articles (3) ผ่าน agent graph จริง"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")  # type: ignore

from dotenv import load_dotenv
load_dotenv()

from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, ToolMessage

from agents.manager_agent import build_graph
from tools.archivist_tools import init_vault_structure
from core.logger import setup_logging
from core.utils import normalize_content

setup_logging()
init_vault_structure()

memory = MemorySaver()
graph = build_graph(checkpointer=memory)

# ────────────────────────────────────────────────
# Book notes (กรอกตาม template)
# ────────────────────────────────────────────────
BOOK_1 = """\
บันทึก book note นี้:
---
entity_type: book_note
title: The Intelligent Investor
author: Benjamin Graham
genre: Value Investing
date_read: 2026-05-23
rating: 5
tags: [book, investment_philosophy, value_investing]
---

# The Intelligent Investor

> ผู้แต่ง: Benjamin Graham | อ่านเสร็จ: 2026-05-23

---

## แก่นความคิดหลัก

- นักลงทุนควรแยกตัวเองออกจากตลาด — ตลาดคือ "Mr. Market" ที่อารมณ์แปรปรวน ไม่ใช่ผู้กำหนดมูลค่าแท้จริง
- มูลค่าที่แท้จริง (Intrinsic Value) ต้องคำนวณจากปัจจัยพื้นฐาน ไม่ใช่ราคาตลาด
- Margin of Safety คือหัวใจ — ซื้อต่ำกว่ามูลค่าแท้เสมอเพื่อรองรับความผิดพลาด
- แยก Investor ออกจาก Speculator: Investor วิเคราะห์ธุรกิจ Speculator เดิมพันราคา
- Defensive Investor ควรกระจายใน หุ้น + พันธบัตร 50/50 และปรับตามสภาวะตลาด

---

## หลักการลงทุนที่ได้

- ซื้อเมื่อ P/E ต่ำกว่าค่าเฉลี่ย 15x และ P/B ต่ำกว่า 1.5x
- เลือกบริษัทที่จ่ายปันผลต่อเนื่องอย่างน้อย 20 ปี
- หลีกเลี่ยงบริษัทที่มีหนี้มากและกำไรผันผวน
- อย่า panic sell เมื่อตลาดร่วง — Mr. Market กำลังให้โอกาส

---

## กรอบการคิดและ Mental Models

- **Mr. Market**: ตลาดเหมือนหุ้นส่วนที่บ้า เสนอราคาซื้อขายทุกวัน ไม่ต้องสนใจถ้าราคาไม่สมเหตุผล
- **Margin of Safety**: ซื้อที่ discount อย่างน้อย 30% จาก intrinsic value เสมอ
- **Defensive vs Enterprising**: เลือก style ให้เหมาะกับเวลาและความรู้ของตัวเอง

---

## คำพูดที่ประทับใจ

> "The investor's chief problem — and even his worst enemy — is likely to be himself."

---

## จะนำไปปรับใช้อย่างไร

- ตรวจสอบ P/E, P/B ทุกครั้งก่อนซื้อ
- กำหนด intrinsic value ก่อนดูราคาตลาด
- ตั้ง watchlist สำหรับหุ้นที่อยู่ในเรดาร์ แล้วรอ Mr. Market เสนอราคาถูก
"""

BOOK_2 = """\
บันทึก book note นี้:
---
entity_type: book_note
title: One Up On Wall Street
author: Peter Lynch
genre: Growth Investing
date_read: 2026-05-23
rating: 5
tags: [book, investment_philosophy, growth_investing]
---

# One Up On Wall Street

> ผู้แต่ง: Peter Lynch | อ่านเสร็จ: 2026-05-23

---

## แก่นความคิดหลัก

- นักลงทุนรายย่อยมีข้อได้เปรียบเหนือ Fund Manager — สามารถลงทุนในบริษัทที่รู้จักจากชีวิตประจำวัน
- "Invest in what you know" — สังเกตธุรกิจรอบตัวก่อนค้นหาหุ้น
- Ten-bagger คือหุ้นที่ให้ผลตอบแทน 10 เท่า — ต้องใช้ความอดทนถือระยะยาว
- เรื่องเล่า (Story) ของบริษัทต้องสมเหตุผลและยังใช้งานได้ก่อนซื้อ
- หุ้น Small Cap ที่ถูกมองข้ามคือแหล่งกำไรที่ดีที่สุด

---

## หลักการลงทุนที่ได้

- แบ่งหุ้นเป็น 6 ประเภท: Slow Growers, Stalwarts, Fast Growers, Cyclicals, Turnarounds, Asset Plays
- ติดตาม PEG Ratio (P/E ÷ Growth Rate) — ควรต่ำกว่า 1 คือน่าสนใจ
- ตรวจ Inventory ที่สะสมเกินเทียบกับ Revenue เป็น warning sign
- อย่า diversify เกินไป — เข้าใจ 5-10 บริษัทดีกว่าถือ 50 บริษัท

---

## กรอบการคิดและ Mental Models

- **The Perfect Stock**: บริษัทที่น่าเบื่อ ถูกมองข้าม แต่ dominant ในตลาดเฉพาะกลุ่ม
- **Cocktail Party Theory**: ตลาดใกล้จุดสูงสุดเมื่อทุกคนแนะนำหุ้น
- **Check the Story**: ก่อน sell ให้ถามว่า "story ที่ทำให้ซื้อยังใช้งานได้อยู่ไหม"

---

## คำพูดที่ประทับใจ

> "The person that turns over the most rocks wins the game."

---

## จะนำไปปรับใช้อย่างไร

- สังเกตธุรกิจที่ใช้บ่อยในชีวิตประจำวัน แล้วค้นหาข้อมูลบริษัทนั้น
- คำนวณ PEG ก่อนซื้อทุกครั้ง
- จดบันทึก "story" ของแต่ละหุ้นที่ถือ และทบทวนทุกไตรมาส
"""

BOOK_3 = """\
บันทึก book note นี้:
---
entity_type: book_note
title: The Psychology of Money
author: Morgan Housel
genre: Behavioral Finance
date_read: 2026-05-23
rating: 5
tags: [book, investment_philosophy, behavioral_finance, mindset]
---

# The Psychology of Money

> ผู้แต่ง: Morgan Housel | อ่านเสร็จ: 2026-05-23

---

## แก่นความคิดหลัก

- ความมั่งคั่งคือสิ่งที่ไม่เห็น — ทรัพย์สินที่ไม่ได้ใช้คือความมั่งคั่งจริง ไม่ใช่สิ่งที่ซื้อ
- Compounding ต้องการเวลา — Warren Buffett สร้างความมั่งคั่ง 95% หลังอายุ 50 ปี
- ผู้คนตัดสินใจทางการเงินจากประสบการณ์ตัวเอง ไม่ใช่ข้อมูล — ทำให้ไม่มีคำตอบเดียวที่ถูกสำหรับทุกคน
- "Enough" คือทักษะสำคัญที่สุด — รู้ว่าพอเมื่อไหร่
- Risk ที่ใหญ่สุดคือสิ่งที่ไม่เคยเกิดขึ้นมาก่อน (Black Swan)

---

## หลักการลงทุนที่ได้

- ออมก่อนลงทุนเสมอ — Savings rate สำคัญกว่า Return rate
- อย่าประเมินผลระยะสั้นมากเกินไป ให้เวลา Compounding ทำงาน
- ยอมรับ Volatility เป็น "ค่าธรรมเนียม" ของ Return ที่ดี ไม่ใช่ "ค่าปรับ"
- Avoid financial ruin ก่อน — อยู่ในเกมให้นานพอ

---

## กรอบการคิดและ Mental Models

- **Reasonable > Rational**: การตัดสินใจที่ "พอรับได้ทางอารมณ์" ดีกว่าที่ optimal แต่ทนไม่ได้
- **Room for Error**: ถือเงินสดและ buffer เสมอสำหรับเหตุการณ์ที่ไม่คาดฝัน
- **Wealth is Hidden**: คนรวยจริงคือคนที่ไม่ได้อวดว่ารวย

---

## คำพูดที่ประทับใจ

> "The ability to do what you want, when you want, with who you want, for as long as you want, is priceless. It is the highest dividend money pays."

---

## จะนำไปปรับใช้อย่างไร

- ตั้ง Emergency Fund 6 เดือนก่อนลงทุน
- กำหนด "enough" ในชีวิตให้ชัดเจน เพื่อไม่โลภเกินไป
- ถือหุ้นระยะยาว — อย่าขายเพราะ Volatility รายวัน
"""

# ────────────────────────────────────────────────
# Article URLs
# ────────────────────────────────────────────────
ARTICLES = [
    "สรุปบทความนี้: https://www.investopedia.com/articles/investing/082614/how-stock-market-works.asp",
    "สรุปบทความนี้: https://www.investopedia.com/terms/v/valueinvesting.asp",
    "สรุปบทความนี้: https://www.investopedia.com/terms/m/marginofsafety.asp",
]

BOOKS = [
    ("Book 1 — The Intelligent Investor", BOOK_1),
    ("Book 2 — One Up On Wall Street", BOOK_2),
    ("Book 3 — The Psychology of Money", BOOK_3),
]


def run_task(label: str, user_input: str, thread_id: str) -> str:
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 25}
    inputs = {"messages": [("user", user_input)]}
    reply = ""
    for event in graph.stream(inputs, config=config, stream_mode="updates"):
        for node_name, state in event.items():
            if not isinstance(state, dict) or "messages" not in state:
                continue
            msgs = state["messages"]
            last = msgs[-1] if isinstance(msgs, list) else msgs
            content = normalize_content(getattr(last, "content", ""))
            if content:
                reply = content
    return reply


def main():
    sep = "─" * 60

    print(f"\n{'═'*60}")
    print("  TEST: Knowledge Ingest — Books (3) + Articles (3)")
    print(f"{'═'*60}\n")

    # ── Books ──
    for i, (label, content) in enumerate(BOOKS, 1):
        print(f"\n{sep}")
        print(f"  [{i}/3] {label}")
        print(sep)
        reply = run_task(label, content, thread_id=f"book_{i}")
        print(f"  → {reply[:200]}")
        time.sleep(2)

    # ── Articles ──
    for i, cmd in enumerate(ARTICLES, 1):
        url = cmd.split(": ", 1)[1]
        print(f"\n{sep}")
        print(f"  [Article {i}/3] {url}")
        print(sep)
        reply = run_task(f"Article {i}", cmd, thread_id=f"article_{i}")
        print(f"  → {reply[:200]}")
        time.sleep(2)

    # ── ตรวจสอบไฟล์ที่สร้าง ──
    from pathlib import Path
    vault = Path(os.getenv("OBSIDIAN_VAULT_PATH", "./memories"))

    print(f"\n{'═'*60}")
    print("  ผลลัพธ์ในคลัง:")
    print(f"{'═'*60}")

    books_dir = vault / "30_Knowledge_Base/Books"
    articles_dir = vault / "30_Knowledge_Base/Articles"

    print(f"\n  📚 Books ({books_dir}):")
    if books_dir.exists():
        for f in sorted(books_dir.glob("*.md")):
            print(f"    ✓ {f.name}")
    else:
        print("    (ไม่พบโฟลเดอร์)")

    print(f"\n  📰 Articles ({articles_dir}):")
    if articles_dir.exists():
        for f in sorted(articles_dir.glob("*.md")):
            print(f"    ✓ {f.name}")
    else:
        print("    (ไม่พบโฟลเดอร์)")


if __name__ == "__main__":
    main()
