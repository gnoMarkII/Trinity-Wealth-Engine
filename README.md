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
┌──────────────────────────────────────────────────────────────────────┐
│                       The Manager (Supervisor)                       │
│              วิเคราะห์คำถาม แล้วตัดสินใจว่าจะ Route ไปไหน             │
└──────┬───────────┬───────────┬────────────┬──────────────┬──────────┘
       │            │           │            │              │
 ┌─────▼────┐ ┌─────▼────┐ ┌────▼─────┐ ┌────▼──────┐ ┌─────▼──────────┐
 │ Archivist│ │Bookkeeper│ │  Macro   │ │ Strategic │ │  Equity Intel   │
 │จัดการความจำ│ │ติดตามพอร์ต│ │Quant/    │ │ Allocator │ │  Suite (3-Stage)│
 │ใน Obsidian│ │และธุรกรรม │ │Economist │ │ จัดสรรสินทรัพย์│ │Quant→Narrative→ │
 │           │ │          │ │ประเมินศก. │ │(Guardrails)│ │  Synthesizer   │
 └───────────┘ └──────────┘ └──────────┘ └───────────┘ └────────────────┘
```

> Kanban card flows แยกต่างหาก (dispatch จากการ์ดใน Web UI ไม่ผ่าน Manager routing โดยตรง): `news_funnel_flow.py`, `news_youtube_flow.py`, `youtube_pitch_flow.py` — ดูหัวข้อ [Web UI](#web-ui)

### ทีมงาน AI

| Agent | บทบาท | เครื่องมือหลัก |
|-------|-------|--------------|
| **The Manager** | Supervisor — รับคำถาม วิเคราะห์ และ Route งาน | Router (Structured Output) |
| **The Archivist** | บันทึกและค้นหาข้อมูลใน Obsidian Vault | Vault R/W, Vector RAG, Graph RAG, YouTube, Article/PDF |
| **The Bookkeeper** | ติดตามพอร์ต บัญชีธุรกรรม และเป้าหมายการลงทุน | Portfolio Tools |
| **Macro Quant / Macro Economist** | คำนวณคะแนนสภาวะเศรษฐกิจเชิงปริมาณ (Quant Matrix) และสังเคราะห์ Narrative จากข่าว/เหตุการณ์ | Yahoo Finance, FRED API, News Radar |
| **Strategic Allocator** | สังเคราะห์ผลจาก Macro Quant/Economist เป็นทิศทางจัดสรรสินทรัพย์ระดับสถาบัน | Institutional Guardrails (Valuation/Credit/Correlation) |
| **Equity Intel Suite** | วิเคราะห์หุ้นรายตัวแบบ 3 ขั้น: Equity Quant (ตัวเลข) → Equity Narrative (บทวิเคราะห์/ข่าว) → Equity Synthesizer (สรุปรวมเป็นรายงานเดียว) | Fundamentals, Financial Health, Momentum, Analyst Consensus, Latest News |

> โครงสร้างเดิมเคยมี "The Researcher" เป็น agent แยกสำหรับดึงข้อมูลตลาด — ปัจจุบันถูกแทนที่ด้วย **Equity Intel Suite** (3-stage pipeline เฉพาะทางสำหรับหุ้นรายตัว) และ Macro Quant/Economist (สำหรับข้อมูลมหภาค) แยกจากกันชัดเจนขึ้น

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
ดึงจาก **FRED API** 38 ดัชนีใน 6 หมวดหลัก:
- นโยบายการเงิน: Fed Rate, 2Y Yield, 10Y-2Y Spread
- เงินเฟ้อ: CPI, PCE, Core PCE, PPI, Breakeven 5Y/10Y
- สินเชื่อ: BAA Credit Spread, High Yield Bond Spread
- แรงงาน: Unemployment Rate, Initial Jobless Claims
- การเติบโต: Real GDP, Industrial Production, Retail Sales, Housing Starts
- สภาพคล่อง: M2 Money Supply, Consumer Sentiment

### ติดตามข่าวสารเศรษฐกิจ (News Radar)
ดึงพาดหัวข่าวเศรษฐกิจมหภาคและการเงินรายวันอัตโนมัติผ่าน RSS Feeds (เช่น Investing.com, Yahoo Finance, Prachachat) ลงในรูปแบบ Markdown เพื่อให้ทีม AI ใช้วิเคราะห์ผลกระทบต่อตลาดได้อย่างทันท่วงที

### Strategic Allocator — Macro Strategy Direction (Institutional Guardrails)
สังเคราะห์ข้อมูล Quant (Macro Matrix) และ Narrative (News/YouTube/Article) เป็นทิศทางกลยุทธ์การจัดสรรสินทรัพย์ระดับสถาบัน พร้อมชั้นตรวจสอบคุณภาพหลายชั้น:
- **Valuation & Risk Pillars**: คำนวณ Equity Risk Premium (ERP), Forward Earnings Yield, High Yield Credit Spread, Derived Pair-Trade Ratios (เช่น QQQ/SPY, GLD/SPY) และ Rolling Correlation (เช่น SPY vs TLT/GLD) เป็น `MarketObservable` ที่มี metadata เชิงตัวเลขให้ guardrail ตรวจสอบได้โดยตรง
- **LLM-Agnostic Warning Registry**: รวมคำเตือน Guardrail ทั้งหมดเป็นรหัส structured (`WarningMessage`) พร้อมคำแปลไทยส่วนกลาง แยกระดับความรุนแรง (Retryable Critical / Non-Retryable Critical / Soft Warning)
- **Structured-Output Retry Loop**: ตรวจพบข้อผิดพลาดเชิงโครงสร้าง (เช่น ขาด asset_bucket, ฟิลด์บังคับว่าง) จะส่ง feedback ภาษาไทยกลับให้ LLM แก้ไขอัตโนมัติก่อนตกไปใช้ System Placeholder
- **Contradiction & Valuation Guardrails**: ตรวจจับความขัดแย้งเชิงตรรกะ (Gold ไม่มีหลักยึด Real Yield, Barbell ไม่มีคำอธิบาย Hedge, Regime ขัดแย้งกับเงินเฟ้อ) และคำเตือนความเสี่ยง (Valuation ตึงตัว, Credit Spread กว้าง)

### Equity Intel Suite — วิเคราะห์หุ้นรายตัวแบบ 3-Stage Pipeline
รองรับทั้ง **US Stocks** (`AAPL`, `NVDA`) และ **Thai Stocks** (`PTT`, `AOT`) ผ่าน pipeline 3 ขั้นที่แยกหน้าที่ชัดเจน:

1. **Equity Quant** — ดึงตัวเลขล้วนๆ 6 มิติ: Fundamentals (P/E, EV/EBITDA, P/B, ROE, Profit Margin, Revenue Growth, Market Cap, Beta, Payout Ratio, ESG Score), Financial Trends (รายได้/กำไรสุทธิ 4 ปีย้อนหลัง), Financial Health (Cash Flow, Debt/Equity, Current Ratio), Momentum (MA50/MA200, 52W High/Low, Short Interest), Analyst Consensus (Target Price, Upside %, Recommendation), Latest News (5 ข่าวล่าสุด)
2. **Equity Narrative** — สังเคราะห์บทวิเคราะห์/ข่าวเป็น Sentiment และธีมสำคัญ
3. **Equity Synthesizer** — รวมผลจาก Quant + Narrative เป็นรายงานเดียว พร้อม Composite Score

เข้าถึงผ่านหน้า `/equity` ใน Web UI — ดึงรายชื่อหุ้นจาก **Portfolio + Watchlist** มาแสดงใน Sidebar โดยตรง กดวิเคราะห์ได้ทันทีไม่ต้องพิมพ์ ticker เอง (แยก section หุ้นที่มีรายงานแล้ว/ยังไม่มีรายงานให้ชัดเจน)

### การจัดการพอร์ตโฟลิโอ (Bookkeeper + Portfolio Web Hub)
ติดตามพอร์ตผ่านไฟล์ Markdown + YAML + CSV Ledger ใน Obsidian Vault รองรับ**หลายพอร์ตพร้อมกัน** (Multi-Portfolio) ใช้งานได้ทั้งผ่าน Bookkeeper agent (แชท) และหน้า Web UI `/portfolio` (6 แท็บ: Overview, Holdings, Transactions, Incomes, Watchlist, Calendar):

| กลุ่ม | ความสามารถ |
|-------|-----------|
| **Portfolio State** | ดูภาพรวมพอร์ต, Allocation Breakdown ตาม Bucket, Sync ราคาตลาด, จัดการหลายพอร์ต (สร้าง/ลบ/เปลี่ยนชื่อ) |
| **Trading** | ซื้อ/ขาย, Batch Import, แก้ไข/ลบ Holding, บันทึกรายได้ (ปันผล/ดอกเบี้ย) |
| **Transaction Ledger** | ดูประวัติซื้อขายทั้งหมด แก้ไข/ลบรายการย้อนหลังได้ผ่าน **Ledger Replay Engine** — คำนวณ Weighted-Average Cost และ Realized P/L ของทุกรายการที่เกิดขึ้นหลังจุดที่แก้ไขใหม่ทั้งหมดแบบ Atomic (reject ทันทีถ้าทำให้ units ติดลบ) |
| **Dividend Sync** | ดึงประวัติเงินปันผลอัตโนมัติจาก yfinance คำนวณ Net Dividend หลังหักภาษี ณ ที่จ่าย (US 15% / TH 10%) แยกเงินปันผลที่ได้รับแล้วกับที่กำลังจะได้รับ (Received vs Upcoming) พร้อมระบบป้องกันการเขียนทับข้อมูลที่แก้ไขด้วยมือ (Manual Override Protection) |
| **Historical FX Rate** | ดึงอัตราแลกเปลี่ยน USD/THB ย้อนหลังตามวันที่ทำธุรกรรมจริงจาก yfinance (แยก Local Trade FX ออกจาก Global Portfolio FX ป้องกันข้อมูลปนกัน) |
| **Goals** | ตั้งเป้าหมาย, ติดตาม Progress, ลบเป้าหมาย |
| **Journal & History** | Performance Snapshot (CSV), Trading Journal, Watchlist, Economic/Earnings Calendar |

### สกัดความรู้จากแหล่งภายนอก (Knowledge Ingestion)
- **YouTube Monitor & Ingestion**: สร้าง Weekly Digest ติดตามคลิปใหม่รายสัปดาห์ พร้อม Smart Checkbox ตรวจสอบการดึงข้อมูลซ้ำ ดึง Transcript สกัดเนื้อหาด้วย LLM (`extractor` model slot) และส่งต่อให้ระบบ Auto-routing จัดเก็บแยกโฟลเดอร์ตามชื่อช่อง พร้อมสร้าง Obsidian Canvas อัตโนมัติ (Archivist)
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
- **Prompt Harness (Skill-based Prompts)**: แยก System Prompt ของทั้ง Agent (`prompts/skills/`) และ Tool (`prompts/tools/`) ออกจากโค้ด Python ไปเป็นไฟล์ `.md` พร้อม Hot-reload, Mustache Templating และ Mojibake Repair กลาง

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

## Web UI

นอกจาก CLI แล้ว ระบบมี Web UI แบบ single-user สำหรับสั่งงาน agent, จัดการพอร์ต, และดูรายงานผ่านเบราว์เซอร์ — FastAPI backend (`api/`) + React frontend (`web/`)

**ฟีเจอร์หลัก:**
- **Portfolio Hub** (`/portfolio`) — 6 แท็บ: Overview (Allocation), Holdings, Transactions (ดู/แก้ไข/ลบประวัติซื้อขายผ่าน Ledger Replay Engine), Incomes (Dividend Sync แบบ Received/Upcoming), Watchlist, Calendar — รองรับหลายพอร์ตพร้อมกัน
- **Equity Analysis** (`/equity`) — วิเคราะห์หุ้นรายตัวผ่าน Equity Intel Suite (Quant → Narrative → Synthesizer) พร้อม Sidebar quick-access จาก Portfolio/Watchlist
- **Agent Kanban Board** — สร้าง/แก้ไข/สั่งงานการ์ด, ดู log การทำงานของ agent แบบ real-time (SSE), รองรับ human-in-the-loop approval สำหรับ flow News/YouTube (เลือกข่าว/คลิปที่จะเจาะลึกก่อน agent ประมวลผลต่อ), รองรับ NotebookLM Audio Overview background worker
- **Macro Strategy Report** — แดชบอร์ด Regime Probabilities, Cross-Asset Allocation, Pair Trades, Hedging Scenarios พร้อมแหล่งอ้างอิงข้อมูล
- **Auth แบบเบา** — รหัสผ่านเดียวจาก `.env` (ไม่มีระบบ user/role เพราะเป็นเครื่องมือส่วนตัวคนเดียว)

**การรัน (ต้องรันคู่กัน 2 process):**

ตั้งค่าเพิ่มใน `.env` ก่อน (ดู `.env.example` หัวข้อ "Web UI"):
```bash
WEBUI_PASSWORD=รหัสผ่านที่ตั้งเอง
SESSION_SECRET_KEY=สตริงยาวๆ แบบสุ่ม ห้ามเปลี่ยนบ่อย (ไม่งั้น session จะหลุดทุกครั้งที่ restart)
UNVERIFIED_DRAFT_SIGNING_KEY=สตริงสุ่มแยกต่างหากอย่างน้อย 32 bytes สำหรับ sign การอนุมัติ Unverified Draft
```

Terminal 1 — Backend:
```bash
uv run uvicorn api.main:app --reload
```

Terminal 2 — Frontend (dev server, proxy `/api` ไปที่ backend อัตโนมัติ):
```bash
cd web
npm install
npm run dev
```

เปิด `http://localhost:5173` แล้ว login ด้วย `WEBUI_PASSWORD` — รายละเอียดเพิ่มเติมดูที่ [`web/README.md`](web/README.md)

**การรันแบบ production (process เดียว):** build frontend แล้วให้ FastAPI เสิร์ฟเอง

```bash
cd web && npm run build && cd ..
uv run uvicorn api.main:app
```

เปิด `http://localhost:8000` ได้เลย — backend เสิร์ฟ `web/dist/` พร้อม SPA fallback ให้ deep link เช่น `/kanban` ทำงานได้ (mount อัตโนมัติเฉพาะเมื่อ `web/dist/` มีอยู่)

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
├── api/                         # FastAPI backend สำหรับ Web UI
│   ├── main.py                  # App entrypoint (uvicorn api.main:app)
│   ├── config.py                # App-level config/env loading
│   ├── auth.py                  # รหัสผ่านเดียว + session cookie
│   ├── jobs.py                  # Single-worker job queue (dispatch/resume LangGraph)
│   ├── routes_agents.py         # Dispatch / SSE stream / resume endpoints
│   ├── routes_debug.py          # GET /api/debug/models — model registry audit
│   ├── routes_equity.py         # Equity Intel Suite endpoints (analyze/detail/list)
│   ├── routes_kanban.py         # Kanban card CRUD + move
│   ├── routes_notebooklm.py     # NotebookLM Audio Overview job endpoints
│   ├── routes_portfolio.py      # Portfolio Hub (holdings/transactions/dividends/FX/goals/watchlist/journal/calendar) + Macro Strategy dashboard endpoints
│   ├── news_funnel_cards.py     # News Funnel → Kanban card generation
│   ├── notebooklm_worker.py     # Background worker: NotebookLM audio generation
│   ├── schemas.py                # API response DTOs (แยกจาก schemas/macro_schemas.py)
│   └── state_db.py               # SQLite store: jobs, job_logs, kanban_cards
├── web/                          # React + TypeScript + Vite frontend (ดู web/README.md)
├── agents/
│   ├── manager_agent.py         # Supervisor + LangGraph graph builder
│   ├── archivist_agent.py       # PKM management ReAct agent
│   ├── bookkeeper_agent.py      # Portfolio & accounting ReAct agent
│   ├── macro_quant_agent.py     # Quant Macro Matrix ReAct agent
│   ├── macro_economist_agent.py # Macroeconomic narrative synthesis ReAct agent
│   ├── strategic_allocator.py   # Strategic Allocator (Macro Strategy Direction + retry loop)
│   ├── equity_quant_agent.py    # Equity Intel Suite — Stage 1: ตัวเลข Fundamentals/Momentum/Analyst
│   ├── equity_narrative_agent.py # Equity Intel Suite — Stage 2: Sentiment/News synthesis
│   ├── equity_synthesizer.py    # Equity Intel Suite — Stage 3: รวมรายงานสุดท้าย
│   ├── news_funnel_flow.py      # Kanban-dispatched flow: News triage → deep dive
│   ├── news_youtube_flow.py     # Kanban-dispatched flow: YouTube URL → summary
│   └── youtube_pitch_flow.py    # Kanban-dispatched flow: YouTube channel → Pitch/Briefing Book
├── tools/
│   ├── macro/                   # Macro & Economic tools (FRED, Yield Curve, Valuation, Risk Analytics)
│   ├── market/                  # Stock market tools (Yahoo Finance, Fundamentals)
│   ├── portfolio/               # Portfolio ledger tools (Holdings, Trades, Ledger Replay, Dividends, FX, Goals, Watchlist)
│   ├── knowledge/               # Web extraction & PDF tools
│   ├── archivist/               # Vault indexing & RAG tools
│   └── _atomic_io.py            # Atomic file writing utility
├── validators/                  # Strategic Allocator guardrail rules
│   ├── contradiction_rules.py   # Gold/Equity/Regime/Barbell contradiction checks
│   ├── valuation_guardrails.py  # ERP richness & credit spread checks
│   ├── quality_check.py         # Warning classification (retryable/critical/soft)
│   └── structured_output_retry.py # LLM retry loop + fallback placeholders
├── prompts/
│   ├── skills/                  # Agent system prompts as Markdown (SKILL.md / pillars.md / guardrails.md / few_shots.md)
│   └── tools/                   # Tool-layer prompts (extractor, youtube_pitcher, news_funnel) — เดียวกับ prompts/skills/ แค่แยก root
├── core/
│   ├── llm_factory.py           # LLM factory (Google / Anthropic / OpenRouter)
│   ├── model_registry.py        # Centralized LLM model config — 14 slots (agent + tool layer), audit ผ่าน GET /api/debug/models
│   ├── security.py              # PII redaction middleware
│   ├── retry.py                 # Exponential backoff retry helper
│   ├── agent_log.py             # Daily Markdown agent activity logger
│   ├── logger.py                # Python logging setup + file handler
│   ├── prompt_harness.py        # Skill/prompt loader + Mustache templating + hot-reload
│   ├── text_utils.py            # Mojibake repair utility
│   └── utils.py                 # Content normalization
├── schemas/
│   ├── pkm_models.py            # Pydantic models (MemoryEntry)
│   ├── macro_schemas.py         # Macro Matrix & Macro Strategy Direction schemas
│   ├── warning_registry.py      # Structured WarningMessage IDs + Thai translations + severity sets
│   └── report_labels.py         # Centralized report/why-not-high display labels
├── tests/
│   ├── core/                    # Core logic tests
│   ├── tools/                   # Modular tools tests
│   ├── unit/                    # Schema/validator/guardrail unit tests
│   ├── integration/             # End-to-end flow tests
│   ├── api/                     # FastAPI backend tests (pytest + TestClient)
│   └── conftest.py              # Shared fixtures
├── memories/                    # Obsidian Vault (gitignored)
│   ├── 01_Daily_Logs/
│   ├── 20_Portfolio_Management/
│   ├── 30_Knowledge_Base/
│   │   ├── Macroeconomics/
│   │   ├── Stocks/
│   │   ├── YouTube_Summaries/
│   │   ├── News/
│   │   ├── Books/
│   │   └── Strategies/
├── .env.example
└── pyproject.toml
```

---

## Model Configuration

ทุก env var ที่เลือก LLM model (ทั้ง agent layer และ tool layer) รวมศูนย์อยู่ใน `core/model_registry.py` — ไม่ต้องไล่หา `os.getenv(...)` กระจายทั่วโค้ด ดูค่าที่ resolve จริงตอนรันได้ที่ `GET /api/debug/models` (ต้อง login ก่อน)

| Slot | Env Var | Default | Layer | Purpose |
|------|---------|---------|-------|---------|
| `manager` | `MANAGER_MODEL` | `gemini-3.1-flash-lite-preview` | agent | Manager Agent — routing and orchestration |
| `router` | `ROUTER_MODEL` | `gemini-3.1-flash-lite-preview` | agent | Structured routing (RouterDecision) — ถ้าไม่ตั้งจะ chain ไปตามค่า `manager` ที่ resolve แล้ว |
| `archivist` | `ARCHIVIST_MODEL` | `gemini-3.1-flash-lite-preview` | agent | PKM management agent |
| `bookkeeper` | `BOOKKEEPER_MODEL` | `gemini-3.1-flash-lite-preview` | agent | Portfolio & accounting agent |
| `macro_quant` | `MACRO_QUANT_MODEL` | `gemini-3.1-flash-lite-preview` | agent | Quant Macro Matrix agent |
| `economist` | `MACRO_ECONOMIST_MODEL` | `gemini-3.1-flash-lite-preview` | agent | Macroeconomic narrative synthesis agent |
| `allocator` | `STRATEGIC_ALLOCATOR_MODEL` | `gemini-3.1-flash-lite-preview` | agent | Strategic Allocator agent |
| `equity_quant` | `EQUITY_QUANT_MODEL` | `gemini-3.1-flash-lite-preview` | agent | Equity Intel Suite — Stage 1: ตัวเลข (deterministic) |
| `equity_narrative` | `EQUITY_NARRATIVE_MODEL` | `gemini-3.1-flash-lite-preview` | agent | Equity Intel Suite — Stage 2: Sentiment/Narrative |
| `equity_synthesizer` | `EQUITY_SYNTHESIZER_MODEL` | `gemini-3.1-flash-lite-preview` | agent | Equity Intel Suite — Stage 3: สรุปรายงานสุดท้าย |
| `extractor` | `EXTRACTOR_MODEL` | `gemini-3.1-flash-lite-preview` | tool | Article/PDF/YouTube content extraction |
| `youtube_pitch` | `YOUTUBE_PITCH_MODEL` | `gemini-3.1-flash-lite-preview` | tool | YouTube Pitch generation + Briefing Book |
| `news_triage` | `NEWS_FUNNEL_TRIAGE_MODEL` | `gemini-3.1-flash-lite-preview` | tool | News impact scoring (batch triage) |
| `thai_title_translation` | `NEWS_FUNNEL_SYNTHESIS_MODEL` | `gemini-3.1-flash-lite-preview` | tool | Thai title translation for news |

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
| `fastapi` / `uvicorn` | Web UI backend server |
| `pandas` | คำนวณ Historical FX/Dividend series (Portfolio Hub) |
| `filelock` | Cross-process lock ป้องกัน race condition ตอนเขียน Portfolio state |

---

## รัน Tests

```bash
uv run python -m pytest tests/ -q
```

ปัจจุบันมี **1,100+ tests** ฝั่ง Python ครอบคลุม: PII, Portfolio lifecycle (Trading, Ledger Replay, Dividend Sync, FX), Market tools (TH/US), Equity Intel Suite, Retry logic, Atomic writes, Vault isolation safety-net, Agent logging, Knowledge tools, Strategic Allocator guardrails/retry loop, Integration test แบบ E2E และ Web UI backend (`tests/api/`) — ฝั่ง frontend มี **230+ tests** ผ่าน Vitest + Testing Library (รัน `npm test` ใน `web/`; ดู [`web/README.md`](web/README.md))

---

## สถานะการพัฒนา

- [x] Multi-agent Supervisor pattern (Manager → Archivist / Bookkeeper / Macro Quant/Economist / Strategic Allocator / Equity Intel Suite)
- [x] Macro data 19 ดัชนี (Yahoo Finance, Parallel fetch)
- [x] US Sector Rotation 11 กลุ่ม
- [x] Regional Pulse 7 ภูมิภาค
- [x] Economic Fundamentals 38 ดัชนี (FRED API)
- [x] Equity Intel Suite — 3-Stage Pipeline (Quant → Narrative → Synthesizer), US & Thai stocks
- [x] Portfolio Web Hub — Multi-Portfolio, Transaction Ledger + Ledger Replay Engine (แก้ไข/ลบย้อนหลังพร้อม Recalculation), Dividend Sync (Received/Upcoming), Historical FX Rate, Goals, Watchlist, Calendar
- [x] Obsidian PKM — Save, Read, Semantic Search, Graph Context
- [x] YouTube transcript summarization + Obsidian Canvas
- [x] YouTube Weekly Monitor & Smart Checkbox + Auto-routing by Channel Name
- [x] Article URL (3-Tier Fallback: Trafilatura, BS4, Playwright) + PDF knowledge ingestion
- [x] News Radar (RSS Feed daily ingestion) + News Funnel → Kanban card auto-generation
- [x] Strategic Allocator — Macro Strategy Direction พร้อม Institutional Guardrails (Valuation/Credit Spread/Pair-Trade/Correlation Pillars)
- [x] LLM-Agnostic Warning Registry + Structured-Output Retry Loop
- [x] Prompt Harness — แยก System Prompt เป็น Markdown Skill files พร้อม Hot-reload
- [x] PII Redaction Middleware
- [x] Atomic file writes + Exponential backoff retry + Vault isolation safety-net (test suite)
- [x] Daily agent activity logs
- [x] 1,100+ automated backend tests + 230+ frontend tests
- [x] Web UI — FastAPI backend + React frontend (Portfolio Hub, Equity Analysis, Kanban board, Macro dashboard, NotebookLM Audio Overview, HITL approval flow)
- [ ] Alert / Notification system

---

## License

This project is licensed under the GNU AGPLv3 License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2026 Money ReRoute. All rights reserved.

*For commercial use, enterprise deployment, or dual-licensing inquiries, please contact Money ReRoute.*
