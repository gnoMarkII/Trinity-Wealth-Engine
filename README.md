# Investment Manager AI — Multi-Agent System

> ⚠️ **โปรเจกต์นี้อยู่ระหว่างการพัฒนา** ฟีเจอร์ต่างๆ อาจมีการเปลี่ยนแปลงได้ตลอดเวลา

ระบบผู้ช่วยจัดการกองทุนส่วนตัวที่ใช้ AI หลายตัวทำงานร่วมกัน (Multi-Agent) สร้างด้วย [LangGraph](https://github.com/langchain-ai/langgraph) บน Python โดยมีสถาปัตยกรรมแบบ **Supervisor + Workers** ที่แยกหน้าที่ดึงข้อมูล บันทึกความจำ ติดตามพอร์ต และตอบคำถามออกจากกันอย่างชัดเจน

---

## สถาปัตยกรรมระบบ

```
คุณ (User)
    │
    ▼
PIIMiddleware ─── ตรวจและลบข้อมูลส่วนตัว (Thai ID, บัตรเครดิต, อีเมล, เบอร์โทร)
    │
    ▼
┌────────────────────────────────────────────────────────┐
│               The Manager (Supervisor)                 │
│      วิเคราะห์คำถาม แล้วตัดสินใจว่าจะ Route ไปไหน       │
└────────┬──────────────┬────────────┬───────────────────┘
         │              │            │                   │
  ┌──────▼─────┐  ┌─────▼─────┐  ┌───▼──────┐  ┌─────────▼────────┐
  │ Researcher │  │ Archivist │  │Bookkeeper│  │   Macro Analyst  │
  │ดึงข้อมูลจาก│  │จัดการความจำ│  │ติดตามพอร์ต│  │ ประเมินสภาวะศก.  │
  │ Yahoo/FRED │  │ใน Obsidian│  │และธุรกรรม │  │ ผ่าน Macro Matrix│
  └──────┬─────┘  └───────────┘  └──────────┘  └─────────┬────────┘
         │ (Auto-save)                                   │
         └───────────────────────────────────────────────► The Archivist
```

### ทีมงาน AI

| Agent | บทบาท | เครื่องมือหลัก |
|-------|-------|--------------|
| **The Manager** | Supervisor — รับคำถาม วิเคราะห์ และ Route งาน | Router (Structured Output) |
| **The Researcher** | ดึงข้อมูลตลาดและเศรษฐกิจจากภายนอก | Yahoo Finance, FRED API |
| **The Archivist** | บันทึกและค้นหาข้อมูลใน Obsidian Vault | Vault R/W, Vector RAG, Graph RAG, YouTube, Article/PDF |
| **The Bookkeeper** | ติดตามพอร์ต บัญชีธุรกรรม และเป้าหมายการลงทุน | Portfolio Tools (19 tools) |
| **The Macro Analyst** | วิเคราะห์สภาวะเศรษฐกิจมหภาคและการจัดสรรสินทรัพย์ | คำนวณคะแนนสภาวะเศรษฐกิจ, จัดทำ Macro Matrix |

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

### ติดตามข่าวสารเศรษฐกิจ (News Radar)
ดึงพาดหัวข่าวเศรษฐกิจมหภาคและการเงินรายวันอัตโนมัติผ่าน RSS Feeds (เช่น Investing.com, Yahoo Finance, Prachachat) ลงในรูปแบบ Markdown เพื่อให้ทีม AI ใช้วิเคราะห์ผลกระทบต่อตลาดได้อย่างทันท่วงที

### วิเคราะห์หุ้นรายตัว
รองรับทั้ง **US Stocks** (`AAPL`, `NVDA`) และ **Thai Stocks** (`PTT`, `AOT`) ดึงข้อมูลได้ 6 มิติ:

| มิติ | ข้อมูล |
|------|--------|
| **Fundamentals** | P/E, EV/EBITDA, P/B, ROE, Profit Margin, Revenue Growth, Market Cap, Beta, Payout Ratio, ESG Score |
| **Financial Trends** | รายได้รวม + กำไรสุทธิ ย้อนหลัง 4 ปีงบการเงิน |
| **Financial Health** | Operating/Free Cash Flow, Total Cash/Debt, Debt/Equity, Current Ratio |
| **Momentum** | MA50, MA200, 52W High/Low, % Insider/Institution Hold, Short Ratio, Short % Float |
| **Analyst Consensus** | ราคาเป้าหมาย (Low/Mean/High), Upside %, Recommendation, จำนวนนักวิเคราะห์ |
| **Latest News** | พาดหัวข่าวล่าสุด 5 ข่าว พร้อม Publisher และลิงก์ |

### การจัดการพอร์ตโฟลิโอ (Bookkeeper)
ติดตามพอร์ตผ่านไฟล์ Markdown + YAML ใน Obsidian Vault — **19 tools**:

| กลุ่ม | ความสามารถ |
|-------|-----------|
| **Portfolio State** | ดูภาพรวมพอร์ต, Allocation Breakdown, Sync ราคาตลาด, FX Rate |
| **Trading** | ซื้อ/ขาย, Batch Import, แก้ไข Holding, บันทึกรายได้ (ปันผล/ดอกเบี้ย) |
| **Goals** | ตั้งเป้าหมาย, ติดตาม Progress, ลบเป้าหมาย |
| **Journal & History** | Performance Snapshot (CSV), Trading Journal, Watchlist |

### สกัดความรู้จากแหล่งภายนอก (Knowledge Ingestion)
- **YouTube Monitor & Ingestion**: สร้าง Weekly Digest ติดตามคลิปใหม่รายสัปดาห์ พร้อม Smart Checkbox ตรวจสอบการดึงข้อมูลซ้ำ ดึง Transcript สกัดเนื้อหาด้วย LLM (Researcher) และส่งต่อให้ระบบ Auto-routing จัดเก็บแยกโฟลเดอร์ตามชื่อช่อง พร้อมสร้าง Obsidian Canvas อัตโนมัติ (Archivist)
- **Article URL**: ดึงบทความจากเว็บด้วยระบบ **3-Tier Fallback** (Trafilatura → BeautifulSoup → Playwright) เพื่อทลายข้อจำกัดเว็บที่ป้องกัน Bot → สกัดข้อมูลด้วย LLM → Markdown พร้อม frontmatter
- **PDF**: อ่าน PDF รายงาน/งบการเงิน → สกัดข้อมูลด้วย LLM

### ระบบความจำ PKM (Obsidian Vault)
เก็บและค้นหาความรู้ด้านการลงทุนแบบถาวรลง **Obsidian Vault**:
- **บันทึก Entity**: บริษัท, ผู้บริหาร, เหตุการณ์ตลาด, กลยุทธ์ — พร้อม YAML frontmatter และ Wikilinks
- **Semantic Search**: ค้นหาตามความหมายด้วย Vector RAG (ChromaDB + HuggingFace Embeddings) แบบ Local
- **Graph Context**: ติดตาม Wikilinks อ่าน linked entities ต่อเนื่อง (GraphRAG)
- **Auto-index**: สร้าง `index.md` อัปเดตอัตโนมัติทุกครั้งที่บันทึกไฟล์
- **Vault Health**: ตรวจหา Orphan files, Empty files, และความขัดแย้งของข้อมูล

### ความปลอดภัยและความเสถียร
- **PII Redaction**: กรองข้อมูลส่วนตัวออกก่อนส่งเข้า AI ทุกครั้ง (Thai National ID, บัตรเครดิต, อีเมล, เบอร์โทรศัพท์)
- **Atomic File Writes**: เขียนไฟล์ผ่าน temp → `os.replace()` เสมอ — ป้องกันข้อมูลเสียหายกรณี crash
- **Exponential Backoff Retry**: ทุก API call มี retry logic อัตโนมัติ (429/5xx, network errors)
- **Daily Markdown Logs**: บันทึก agent routing และ warnings/errors ลง Vault รายวัน

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

คัดลอก `.env.example` แล้วใส่ค่า API Key ของตัวเอง:
```bash
cp .env.example .env
```

> **API Keys ที่จำเป็น:**
>
> | Key | จำเป็น | ขอได้ที่ |
> |-----|--------|---------|
> | `GOOGLE_API_KEY` | ✅ บังคับ | [Google AI Studio](https://aistudio.google.com/app/apikey) |
> | `FRED_API_KEY` | ✅ บังคับ (Hard Data) | [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) — ฟรี |
> | `OPENROUTER_API_KEY` | ⬜ optional (Article/PDF) | [openrouter.ai](https://openrouter.ai/) |
> | `ANTHROPIC_API_KEY` | ⬜ optional | [console.anthropic.com](https://console.anthropic.com/) |
> | `LANGCHAIN_API_KEY` | ⬜ optional (Tracing) | [smith.langchain.com](https://smith.langchain.com/) — ฟรี |

**3. รันระบบ**
```bash
uv run python main.py
```

---

## ตัวอย่างการใช้งาน

```
คุณ: ดูสภาวะตลาดวันนี้หน่อย
```
> ดึง Yield Curve, VIX, DXY, ทองคำ, น้ำมัน, S&P500 และ Bitcoin พร้อมกัน แล้วบันทึกลง Vault

```
คุณ: วิเคราะห์ NVDA ให้หน่อย
```
> ดึง Fundamentals, Financial Health, Momentum, Analyst Consensus และข่าวล่าสุด แล้วบันทึกลง Vault/Stocks/NVDA

```
คุณ: ซื้อ PTT 1000 หุ้น ราคา 35.50 บาท
```
> Bookkeeper บันทึกธุรกรรม อัปเดต Holdings, NAV, Allocation % ผ่าน atomic file write

```
คุณ: พอร์ตตอนนี้เป็นยังไงบ้าง
```
> แสดง NAV รวม, Unrealized P/L, Allocation Breakdown แยกตามสินทรัพย์

```
คุณ: สร้าง Weekly Digest ของ YouTube ให้หน่อย
```
> สร้างตารางสรุปคลิปใหม่รายสัปดาห์จากช่องลงทุนที่กำหนด พร้อมเช็คสถานะคลิปที่เคยสรุปแล้วให้อัตโนมัติ

```
คุณ: สรุป YouTube นี้ให้หน่อย [URL]
```
> ดึง Transcript → สกัดชื่อช่องและเนื้อหาผ่าน LLM → Archivist จัดลงโฟลเดอร์ชื่อช่อง พร้อมสร้าง Canvas

```
คุณ: CPI ล่าสุดเป็นเท่าไหร่
```
> ดึงตัวเลขล่าสุดจาก FRED พร้อมวันที่ประกาศ

---

## โครงสร้างโปรเจกต์

```
invest-agents/
├── main.py                      # Entry point + CLI loop + retry logic
├── agents/
│   ├── manager_agent.py         # Supervisor + LangGraph graph builder
│   ├── researcher_agent.py      # Data fetching ReAct agent
│   ├── archivist_agent.py       # PKM management ReAct agent
│   ├── bookkeeper_agent.py      # Portfolio & accounting ReAct agent
│   └── macro_analyst_agent.py   # Macroeconomic state analysis ReAct agent
├── tools/
│   ├── macro/                   # Macro & Economic tools (FRED, Yield Curve)
│   ├── market/                  # Stock market tools (Yahoo Finance, Fundamentals)
│   ├── portfolio/               # Portfolio ledger tools (Holdings, Trades, Goals)
│   ├── knowledge/               # Web extraction & PDF tools
│   ├── archivist/               # Vault indexing & RAG tools
│   └── _atomic_io.py            # Atomic file writing utility
├── core/
│   ├── llm_factory.py           # LLM factory (Google / Anthropic / OpenRouter)
│   ├── security.py              # PII redaction middleware
│   ├── retry.py                 # Exponential backoff retry helper
│   ├── agent_log.py             # Daily Markdown agent activity logger
│   ├── logger.py                # Python logging setup + file handler
│   └── utils.py                 # Content normalization
├── schemas/
│   ├── pkm_models.py            # Pydantic models (MemoryEntry)
│   └── macro_schemas.py         # Macro Matrix schemas
├── tests/
│   ├── core/                    # Core logic tests
│   ├── tools/                   # Modular tools tests
│   ├── integration/             # End-to-end flow tests
│   └── conftest.py              # Shared fixtures
├── memories/                    # Obsidian Vault (gitignored)
│   ├── 00_Inbox/
│   ├── 01_Daily_Logs/
│   ├── 20_Portfolio_Management/
│   ├── 30_Knowledge_Base/
│   │   ├── Macroeconomics/
│   │   ├── Stocks/
│   │   ├── YouTube_Summaries/
│   │   ├── Articles/
│   │   └── Strategies/
│   └── 40_Finance_and_Tax/
├── .env.example
└── pyproject.toml
```

---

## Dependencies หลัก

| Package | ใช้ทำอะไร |
|---------|-----------|
| `langgraph` | Multi-agent graph orchestration |
| `langchain-google-genai` | Google Gemini LLM |
| `langchain-anthropic` | Anthropic Claude LLM (optional) |
| `langchain-openai` | OpenRouter LLM access |
| `langchain-chroma` | Vector database สำหรับ Semantic Search |
| `langchain-huggingface` | Embedding model แบบ Local |
| `yfinance` | ดึงข้อมูลตลาดจาก Yahoo Finance |
| `fredapi` | ดึงตัวเลขเศรษฐกิจจาก FRED |
| `youtube-transcript-api` | ดึง Transcript จาก YouTube |
| `trafilatura` | ดึงและ extract เนื้อหาบทความจากเว็บ (Tier-1) |
| `beautifulsoup4` | ดึงและ extract เนื้อหาบทความจากเว็บ (Tier-2) |
| `playwright` | ดึงข้อมูลจากเว็บเพจที่ใช้ JavaScript Render / Cloudflare (Tier-3) |
| `feedparser` | ดึงข่าวเศรษฐกิจแบบ RSS (News Radar) |
| `pypdf` | อ่านข้อความจาก PDF |
| `python-frontmatter` | อ่าน/เขียน YAML frontmatter ใน Markdown |
| `pydantic` | Data validation และ Schema |
| `prompt_toolkit` | CLI interface |

---

## รัน Tests

```bash
uv run python -m pytest tests/ -q
```

ปัจจุบันมี **523 tests** ครอบคลุม: PII, Portfolio lifecycle, Market tools (TH/US), Retry logic, Atomic writes, Agent logging, Knowledge tools และ Integration test แบบ E2E

---

## สถานะการพัฒนา

- [x] Multi-agent Supervisor pattern (Manager → Researcher / Archivist / Bookkeeper / Macro Analyst)
- [x] Macro data 19 ดัชนี (Yahoo Finance, Parallel fetch)
- [x] US Sector Rotation 11 กลุ่ม
- [x] Regional Pulse 7 ภูมิภาค
- [x] Economic Fundamentals 19 ดัชนี (FRED API)
- [x] Stock Analysis 6 มิติ — US & Thai stocks
- [x] Portfolio tracking — Holdings, Trades, Goals, Performance history
- [x] Obsidian PKM — Save, Read, Semantic Search, Graph Context
- [x] YouTube transcript summarization + Obsidian Canvas
- [x] YouTube Weekly Monitor & Smart Checkbox + Auto-routing by Channel Name
- [x] Article URL (3-Tier Fallback: Trafilatura, BS4, Playwright) + PDF knowledge ingestion
- [x] News Radar (RSS Feed daily ingestion)
- [x] PII Redaction Middleware
- [x] Atomic file writes + Exponential backoff retry
- [x] Daily agent activity logs
- [x] 523 automated tests
- [ ] Alert / Notification system
- [ ] Web UI

---

## License

This project is licensed under the GNU AGPLv3 License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2026 Money ReRoute. All rights reserved.

*For commercial use, enterprise deployment, or dual-licensing inquiries, please contact Money ReRoute.*
