# Investment Manager AI — Multi-Agent System

> ⚠️ **โปรเจกต์นี้อยู่ระหว่างการพัฒนา** ฟีเจอร์ต่างๆ อาจมีการเปลี่ยนแปลงได้ตลอดเวลา

ระบบผู้ช่วยจัดการกองทุนส่วนตัวที่ใช้ AI หลายตัวทำงานร่วมกัน (Multi-Agent) สร้างด้วย [LangGraph](https://github.com/langchain-ai/langgraph) บน Python โดยมีสถาปัตยกรรมแบบ **Supervisor + Workers** ที่แยกหน้าที่ดึงข้อมูล บันทึกความจำ และตอบคำถามออกจากกันอย่างชัดเจน

---

## สถาปัตยกรรมระบบ

```
คุณ (User)
    │
    ▼
PIIMiddleware ─── ตรวจและลบข้อมูลส่วนตัว (Thai ID, บัตรเครดิต, อีเมล, เบอร์โทร)
    │
    ▼
┌─────────────────────────────────────────┐
│           The Manager (Supervisor)       │
│  วิเคราะห์คำถาม แล้วตัดสินใจว่าจะ Route ไปไหน │
└────────────┬──────────────┬─────────────┘
             │              │
      ┌──────▼──────┐  ┌────▼─────────┐
      │ The Researcher│  │ The Archivist│
      │ ดึงข้อมูลจาก  │  │  จัดการความจำ│
      │ Yahoo/FRED   │  │  ใน Obsidian │
      └──────┬──────┘  └─────────────┘
             │ (Auto-save)
             └──────────────► The Archivist
```

### ทีมงาน AI

| Agent | บทบาท | เครื่องมือ |
|-------|-------|-----------|
| **The Manager** | Supervisor — รับคำถาม วิเคราะห์ และ Route งาน | Router (Structured Output) |
| **The Researcher** | ดึงข้อมูลตลาดและเศรษฐกิจจากภายนอก | Yahoo Finance, FRED API |
| **The Archivist** | บันทึกและค้นหาข้อมูลใน Obsidian Vault | Vault Read/Write, Vector RAG, Graph RAG |

---

## ฟีเจอร์ปัจจุบัน

### ข้อมูลมหภาค (Macro)
ดึงข้อมูลแบบ Real-time จาก **Yahoo Finance** พร้อมกัน 19 ดัชนี ใน 7 หมวด:

| หมวด | ดัชนี |
|------|-------|
| Yield Curve | 13W T-Bill, 5Y, 10Y, 30Y Treasury Yield |
| Risk Sentiment | VIX Fear Index |
| Credit Market | HYG (High Yield), LQD (Investment Grade) |
| สกุลเงิน / FX | DXY, EUR/USD, USD/JPY, USD/CNY |
| สินค้าโภคภัณฑ์ | ทองคำ, น้ำมัน WTI, ก๊าซธรรมชาติ, ทองแดง |
| ดัชนีหุ้นสหรัฐฯ | S&P 500, Nasdaq 100, Russell 2000 |
| สินทรัพย์ดิจิทัล | Bitcoin |

### Sector Rotation (US)
ดึงข้อมูล **11 Sector ETF** (GICS Standard) เรียงตาม % เปลี่ยนแปลงวันนี้ — เห็นทิศทางเงินไหลเข้าออกแต่ละกลุ่มได้ทันที

### ภาพรวมรายภูมิภาค
ติดตาม **7 ตลาดโลก** ผ่าน Regional Proxy ETF: ลาตินอเมริกา, ยุโรป, EM รวม, ญี่ปุ่น, อินเดีย, จีน, เอเชียแปซิฟิก

### ตัวเลขเศรษฐกิจพื้นฐาน (Hard Data)
ดึงจาก **FRED API** 19 ดัชนีใน 6 หมวด:
- นโยบายการเงิน: Fed Rate, 2Y Yield, 10Y-2Y Spread
- เงินเฟ้อ: CPI, PCE, Core PCE, PPI, Breakeven 5Y/10Y
- สินเชื่อ: BAA Credit Spread, High Yield Bond Spread
- แรงงาน: Unemployment Rate, Initial Jobless Claims
- การเติบโต: Real GDP, Industrial Production, Retail Sales, Housing Starts
- สภาพคล่อง: M2 Money Supply, Consumer Sentiment

### วิเคราะห์หุ้นรายตัว (US Stocks)
ระบุ Ticker เช่น `AAPL`, `MSFT`, `NVDA` แล้วดึงข้อมูลได้ 6 มิติ:

| มิติ | ข้อมูล |
|------|--------|
| **Fundamentals** | P/E, EV/EBITDA, P/B, ROE, Profit Margin, Revenue Growth, Market Cap, Beta, Payout Ratio, ESG Score |
| **Financial Trends** | รายได้รวม + กำไรสุทธิ ย้อนหลัง 4 ปีงบการเงิน |
| **Financial Health** | Operating/Free Cash Flow, Total Cash/Debt, Debt/Equity, Current Ratio |
| **Momentum** | MA50, MA200, 52W High/Low, % Insider/Institution Hold, Short Ratio, Short % Float |
| **Analyst Consensus** | ราคาเป้าหมาย (Low/Mean/High), Upside %, Recommendation, จำนวนนักวิเคราะห์ |
| **Latest News** | พาดหัวข่าวล่าสุด 5 ข่าว พร้อม Publisher และลิงก์ |

### ระบบความจำ PKM (Obsidian Vault)
เก็บและค้นหาความรู้ด้านการลงทุนแบบถาวรลง **Obsidian Vault**:
- **บันทึก Entity**: บริษัท, ผู้บริหาร, เหตุการณ์ตลาด, กลยุทธ์ — พร้อม YAML frontmatter และ Wikilinks
- **Semantic Search**: ค้นหาตามความหมายด้วย Vector RAG (ChromaDB + HuggingFace Embeddings) แบบ Local
- **Graph Context**: ติดตาม Wikilinks อ่าน linked entities ต่อเนื่อง (GraphRAG)
- **Auto-index**: สร้าง `index.md` อัปเดตอัตโนมัติทุกครั้งที่บันทึกไฟล์
- **Vault Health**: ตรวจหา Orphan files, Empty files, และความขัดแย้งของข้อมูล

### ความปลอดภัย
- **PII Redaction**: กรองข้อมูลส่วนตัวออกก่อนส่งเข้า AI ทุกครั้ง (Thai National ID, บัตรเครดิต, อีเมล, เบอร์โทรศัพท์)
- **Immutable Inbox**: ไฟล์ใน `00_Inbox` ถูกป้องกันไม่ให้แก้ไขหรือเขียนทับ

---

## การติดตั้ง

### สิ่งที่ต้องมี
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (Package Manager)
- Google Gemini API Key
- FRED API Key (สำหรับตัวเลขเศรษฐกิจ — ฟรี)

### ขั้นตอน

**1. Clone และติดตั้ง dependencies**
```bash
git clone <repo-url>
cd invest-agents
uv sync
```

**2. ตั้งค่า Environment Variables**

สร้างไฟล์ `.env` ที่ root ของโปรเจกต์:
```env
GOOGLE_API_KEY=your_google_api_key_here
FRED_API_KEY=your_fred_api_key_here
OBSIDIAN_VAULT_PATH=./memories
```

> **ขอ API Key:**
> - Google Gemini: [Google AI Studio](https://aistudio.google.com/app/apikey)
> - FRED: [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) (ฟรี)

**3. รันระบบ**
```bash
uv run python main.py
```

---

## ตัวอย่างการใช้งาน

### ดูสภาวะตลาดวันนี้
```
คุณ: ดูสภาวะตลาดวันนี้หน่อย
```
> ระบบจะดึง Yield Curve, VIX, DXY, ทองคำ, น้ำมัน, S&P500 และ Bitcoin พร้อมกัน
> แล้วบันทึกลง Vault อัตโนมัติ

---

### ดู Sector Rotation
```
คุณ: กลุ่มไหนเงินไหลเข้าวันนี้
```
> ดึง 11 Sector ETF เรียงจากที่ขึ้นมากที่สุด → ลดมากที่สุด

---

### วิเคราะห์หุ้นรายตัว
```
คุณ: วิเคราะห์ NVDA ให้หน่อย
```
> ดึง Fundamentals, Financial Health, Momentum, Analyst Consensus และข่าวล่าสุดของ NVDA
> พร้อมบันทึกลง Vault หมวด Stocks อัตโนมัติ

---

### ดูตัวเลขเศรษฐกิจ
```
คุณ: CPI ล่าสุดเป็นเท่าไหร่ Core PCE ล่ะ
```
> ดึงตัวเลขล่าสุดจาก FRED พร้อมวันที่ประกาศ

---

### ค้นหาข้อมูลเก่าใน Vault
```
คุณ: ฉันเคยบันทึกข้อมูลอะไรเกี่ยวกับ NVDA ไว้บ้าง
```
> ค้นหาด้วย Semantic Search จาก Vault ทั้งหมด

---

### ตรวจสุขภาพ Vault
```
คุณ: ตรวจสุขภาพ Vault หน่อย มีไฟล์ไหนที่ไม่มีลิงก์บ้าง
```
> รายงาน Orphan files และ Empty files ทั้งหมด

---

## โครงสร้างโปรเจกต์

```
invest-agents/
├── main.py                     # Entry point + CLI loop
├── agents/
│   ├── manager_agent.py        # Supervisor + LangGraph graph builder
│   ├── researcher_agent.py     # Data fetching ReAct agent
│   └── archivist_agent.py      # PKM management ReAct agent
├── tools/
│   ├── macro_tools.py          # Yahoo Finance + FRED API tools
│   ├── market_tools.py         # Stock analysis tools (6 tools)
│   └── archivist_tools.py      # Obsidian Vault tools (9 tools)
├── core/
│   ├── llm_factory.py          # LLM factory (Google + Anthropic)
│   ├── security.py             # PII redaction middleware
│   └── utils.py                # Content normalization
├── schemas/
│   └── pkm_models.py           # Pydantic models สำหรับ MemoryEntry
├── memories/                   # Obsidian Vault (PKM storage)
│   ├── 00_Inbox/               # ข้อมูลดิบ (ห้ามแก้ไข)
│   ├── 01_Daily_Logs/
│   ├── 10_System_Agents/
│   ├── 20_Portfolio_Management/
│   ├── 30_Knowledge_Base/
│   │   ├── Macroeconomics/
│   │   ├── Stocks/
│   │   └── Strategies/
│   ├── 40_Finance_and_Tax/
│   └── 99_Templates/
└── pyproject.toml
```

---

## Dependencies หลัก

| Package | ใช้ทำอะไร |
|---------|-----------|
| `langgraph` | Multi-agent graph orchestration |
| `langchain-google-genai` | Google Gemini LLM |
| `langchain-anthropic` | Anthropic Claude LLM (optional) |
| `langchain-chroma` | Vector database สำหรับ Semantic Search |
| `langchain-huggingface` | Embedding model แบบ Local |
| `yfinance` | ดึงข้อมูลตลาดจาก Yahoo Finance |
| `fredapi` | ดึงตัวเลขเศรษฐกิจจาก FRED |
| `pydantic` | Data validation และ Schema |
| `prompt_toolkit` | CLI interface |

---

## สถานะการพัฒนา

> ⚠️ โปรเจกต์นี้**อยู่ระหว่างการพัฒนาอย่างต่อเนื่อง** ฟีเจอร์ API และโครงสร้างข้อมูลอาจมีการเปลี่ยนแปลงได้โดยไม่แจ้งล่วงหน้า

- [x] Multi-agent Supervisor pattern (Manager → Researcher → Archivist)
- [x] Macro data 19 ดัชนี (Yahoo Finance, Parallel fetch)
- [x] US Sector Rotation 11 กลุ่ม
- [x] Regional Pulse 7 ภูมิภาค
- [x] Economic Fundamentals 19 ดัชนี (FRED API)
- [x] Stock Analysis 6 มิติ (Fundamentals, Trends, Health, Momentum, Consensus, News)
- [x] Obsidian PKM — Save, Read, Semantic Search, Graph Context
- [x] PII Redaction Middleware
- [x] Vault Health Linting
- [ ] Portfolio tracking (กำลังพัฒนา)
- [ ] Alert / Notification system (กำลังพัฒนา)
- [ ] Thai stock market support (กำลังพัฒนา)
- [ ] Web UI (กำลังพัฒนา)

---

## License

This project is licensed under the GNU AGPLv3 License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2026 Money ReRoute. All rights reserved.

*For commercial use, enterprise deployment, or dual-licensing inquiries, please contact Money ReRoute.*
