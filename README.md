# Investment Manager AI — Trinity Wealth Engine

> ⚠️ **โปรเจกต์นี้อยู่ระหว่างการพัฒนา** ฟีเจอร์และสถาปัตยกรรมถูกออกแบบและทดสอบอย่างเข้มงวดตามมาตรฐาน Hexagonal Architecture

ระบบผู้ช่วยจัดการกองทุนส่วนตัวระดับสถาบันที่ขับเคลื่อนด้วย Multi-Agent System สร้างด้วย [LangGraph](https://github.com/langchain-ai/langgraph), FastAPI และ React 19 + TypeScript + Three.js บนสถาปัตยกรรม **Single-Brain Manager (Hermes) + Autonomous Research Swarm + Guru Roundtable Studio + Deterministic Financial Engine (`Decimal`)** ที่แยกหน้าที่ระหว่างการสนทนาให้คำปรึกษา เวทีถกเถียงสภากูรู การวิจัยเชิงลึก การคำนวณตัวเลขทางการเงิน และการจัดเก็บบันทึกความรู้ออกจากกันอย่างเคร่งครัด

---

## 🏛️ สถาปัตยกรรมระบบ (System Architecture)

```
คุณ (User)
    │
    ▼
PIIMiddleware ─── ตรวจจับและเซ็นเซอร์ข้อมูลส่วนตัว (Thai ID, บัตรเครดิต, อีเมล, เบอร์โทรศัพท์)
    │
    ▼
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│                    The Manager (Hermes Proactive Advisor)                                 │
│  - SSE Streaming Chat Console พร้อม Intent Classifier (Read Mode vs Run Orders)           │
│  - Minimalist Chat Workspace พร้อม Split-Pane Resizable Side Panel (340–1200px)           │
│  - Python Scratchpad Sandbox (AST Guard + Memory Hygiene)                                 │
│  - Trinity MCP Server (21 Tools ครอบคลุม 8 โดเมน + Progressive Tool Registry)             │
└──────────────┬───────────────────────────────┬───────────────────────────────┬────────────┘
               │                               │                               │
  ┌────────────▼─────────────┐   ┌─────────────▼───────────────┐   ┌───────────────▼───────────────┐
  │ Deterministic Services   │   │ Research Swarm (LangGraph)  │   │ Guru Roundtable (/roundtable) │
  │ (Python & Decimal Engine)│   │ (Deterministic Dispatcher)  │   │ (6-Seat Committee + 3D Stage) │
  ├──────────────────────────┤   ├─────────────────────────────┤   ├───────────────────────────────┤
  │ • PortfolioService       │   │ 1. Macro Quant & Economist  │   │ • 6 Personas / 18 Presets     │
  │ • PortfolioAdvisory      │   │ 2. Strategic Allocator      │   │ • Roblox 3D Stage (Three.js)  │
  │ • Active Thesis Watchdog │   │ 3. Equity Intel (3-Stage)   │   │ • Gemini TTS (12 Voices)      │
  │ • Multi-Broker Sync      │   │ 4. Daily & Weekly Digest    │   │ • Raise-Hand Sandbox Math     │
  │ • Multi-Tier Ledger      │   │ 5. The Archivist (Vault V2) │   │ • Scorecard & YouTube Kit ZIP │
  └──────────────────────────┘   └─────────────────────────────┘   └───────────────────────────────┘
```

### ทีมงาน AI & Core Services (Roster)

| Agent / Service | บทบาท | กลไกและเครื่องมือหลัก |
|---|---|---|
| **The Manager (Hermes)** | Single-Brain Advisor — รับคำถาม สนทนาแบบ Real-time ให้คำปรึกษา และสั่งการระบบ | SSE Streaming, Intent Routing, Python Sandbox (AST Whitelist), Trinity MCP Bridge (`21 Tools` 8 Domains) |
| **Guru Roundtable Studio** | เวทีประชุมคณะกรรมการลงทุน AI 6 ที่นั่ง ถกเถียงโต้แย้งและสรุปเป็นแผนปฏิบัติการ | SSE Multi-Turn Debate, 18 Guru Presets, Gemini TTS (12 Voices), Web Audio Lip-Sync, Three.js Roblox 3D Stage, Look-ahead-safe Settlement Scorecard |
| **Research Swarm Flow** | Background Multi-Agent Swarm สำหรับการวิเคราะห์และวิจัยเชิงลึก | LangGraph Orchestrator + Deterministic Task Classifier (`classify_research_tasks`) |
| **Macro Quant & Economist** | คำนวณ Yield Spread/ค่าเงิน/สภาพคล่อง และวิเคราะห์นโยบายการเงินมหภาค | Quantitative Observable Matrix, 38 FRED Series, Sector Rotation Engine, Fed/BoT Tracker |
| **Strategic Allocator** | กำหนดสัดส่วนสินทรัพย์ 3 ถัง (สภาพคล่อง / หลัก / เติบโต) ตามสภาวะตลาด | Guardrails & Portfolio Optimizer, 3-Bucket Allocation, Risk Parity, Regime Guardrails |
| **Equity Intel Suite** | วิเคราะห์หุ้นรายตัวเชิงลึก 3 ขั้นตอน: งบการเงิน, Transcript และประเมินมูลค่า | 3-Stage Stock Deep Dive (`equity_quant` → `equity_narrative` → `equity_synthesizer`), 5x5 DCF Sensitivity, Narrative Verifier |
| **The Archivist** | จัดการฐานความรู้ บันทึกบทเรียน และประวัติพอร์ตลงใน Obsidian Vault (Layout V2) | Obsidian Adapter, Vector Search (RAG / Chroma), Vault V2 Atomic Writer, Audit Trail |
| **Portfolio & Thesis Services** | ระบบบัญชีพอร์ต การคำนวณผลตอบแทน คำแนะนำปรับพอร์ต และติดตามสมมติฐานหุ้น | Ledger Replay Engine (`Decimal`), Multi-Broker Ingestion (SCB, Dime, ARR Gold, BBLAM PVD, WealthX), Active Thesis & Falsifier Watchdog |
| **Daily Digest Engine** | ระบบคัดกรอง สรุป และแจ้งเตือนข่าวสาร/คลิปการลงทุนประจำวัน | News Funnel, Freshness Tiering, Boundary Regex Matching, NotebookLM Audio / YouTube Pitch Pipeline |

> **กฎเหล็กสถาปัตยกรรม (Hard Architecture Rules):**
> 1. **Single-Brain Manager Architecture:** Hermes เป็น Manager ศูนย์กลางเพียงตัวเดียว ส่วน Research Swarm ทำงานเบื้องหลังด้วย Deterministic Python Dispatcher ปราศจาก LLM Router ซ้ำซ้อน
> 2. **Deterministic Financial Math (No LLM Mental Math):** Legacy `bookkeeper_agent.py` ถูกยกเลิกถาวร ตัวเลขบัญชี การจัดสรรพอร์ต และผลตอบแทนคำนวณผ่าน `PortfolioService`, `PortfolioAdvisoryService` และ `Sandbox` ด้วย Python `Decimal` เท่านั้น
> 3. **Hexagonal Architecture & AST Enforcement:** แยก `core/`, `application/`, `tools/`, `agents/`, `validators/` และ `api/` อย่างเคร่งครัด พร้อมตรวจสอบอัตโนมัติใน `tests/architecture/test_dependency_rules.py` (`21 Architecture & AST Invariant Tests`)
> 4. **Atomic File Writes & Transaction Boundary:** การเขียนไฟล์สำคัญต้องผ่าน `_atomic_io.py` และ raw DAO ห้ามเรียก `.commit()` / `.rollback()` โดยตรง (คุม Transaction ผ่าน `UnitOfWork` ทั้งหมด)

---

## ⚡ ฟีเจอร์หลักของระบบ

### 1. Hermes Proactive Manager & AI Advisor (`/manager`)
- **Minimalist Chat-First Workspace:** หน้าจอเริ่มต้นสไตล์ Calm UI พร้อม Starter Prompt Pills ที่คลิกเริ่มต้นได้ทันที
- **Interactive Resizable Right Panel:** แถบด้านขวา (Kanban Board / Trinity Swarm / Morning Delta) สามารถ **ลากปรับยืด-หดได้อิสระ (340px – 1200px)** สลับคอลัมน์อัตโนมัติ (Compact ↔ Wide) พร้อมจดจำค่าใน `localStorage`
- **Real-Time Streaming Chat & Approval:** สนทนาแบบ Server-Sent Events (SSE) แยก Intent ระหว่างคำถามจากรายงานที่มีอยู่ (Read Mode) กับการส่งคำขอสร้างงานวิจัยใหม่ลง Kanban (Human-in-the-loop Approval)
- **Morning PM Review & Thesis Watchdog:** สรุปส่วนต่างรอบ 24 ชม. (`get_morning_delta_briefing`) ตรวจจับ Active Thesis / Falsifiers (`get_active_stock_theses`) และรองรับการสังเคราะห์สมมติฐานหุ้นทั้งพอร์ตอัตโนมัติ (`batch_synthesize_portfolio_theses`)
- **Python Scratchpad Sandbox & Self-Evolving Memory:** รันสคริปต์คำนวณทางการเงินอย่างปลอดภัยด้วย **AST Guard** (`run_sandbox_calculation`) และสกัดทักษะลง `manager_sandbox/skills/_quarantine/` พร้อมระบบจำแนกความจำระยะยาวใน `manager_sandbox/memory/`

### 2. Guru Roundtable Studio & Interactive 3D Stage (`/roundtable`)
- **คณะกรรมการกูรู 6 ที่นั่ง (18 Presets + Custom Personas):** ประกอบด้วย `Hermes Moderator` (ประธานโต๊ะ & พอร์ตจริง), `Value & MoS Guru` (Buffett/Graham), `Quality & Moat Guru` (Munger/Terry Smith), `Growth & Expectations Guru` (Lynch/Fisher), `Macro & Tactical Guru` (Marks/Druckenmiller), และ `Skeptic / Devil's Advocate` (Burry/Chanos) โดยปรับแต่งบุคลิก คำสั่งเฉพาะ ลำดับการพูด และเปิด/ปิดที่นั่งผ่าน `GuruCustomizationModal`
- **เวที Roblox 3D (`RoundtableRobloxStage3D` / Three.js) & โหมด Podium 2D:**
  - สลับได้ทั้ง **Full-Bleed 3D HUD** (แผงประวัติ บทสนทนา และ Decision Playbook ลอยบนฉาก 3 มิติเต็มจอ) และ **Classic 3-Column 2D Workspace**
  - ห้อง 3 มิติพร้อมพื้น Architectural Glass-Tile PBR multi-map, ฐาน holotable กราฟแท่งเทียนสดจาก `getEquityOHLCV`, ระบบแสง **Studio Twilight (`กลางคืน`) / Daylight Solar Corona (`ยามสาย`)** พร้อมลำแสงปริมาตร ฝุ่น `InstancedMesh` และเมฆลอยนอกหน้าต่าง
  - มุมกล้อง 3 โหมด (`ทั้งห้อง` / `โต๊ะสภา` / `ตามคนพูด`), การเดินหลบสิ่งกีดขวางของตัวละคร (`computeObstacleAwareStep`), ป้ายชื่อ & บับเบิลคำพูดที่ป้องกันการคลุมดำข้อความ (`select-none`), คลิกตัวละครเพื่อ `@แท็กถาม` และโหมดสาธิตขั้นตอน (`Demo Stepper`)
- **Multi-Voice Gemini TTS & Web Audio Lip-Sync:** รองรับแคตตาล็อก **12 เสียง Gemini TTS** (`gemini-2.5-flash-preview-tts`) สร้างไฟล์เสียง WAV 24kHz พร้อมแคชและระบบ fallback สู่ Web Speech API โดยวิเคราะห์พลังงานเสียงผ่าน `AnalyserNode` (`getVoiceEnergy`) ให้ตัวละครขยับปากและท่าทางตามจังหวะเสียงจริง
- **Raise Hand & Sandbox Math Interjection:** กดยกมือถามแทรกกลางวงสนทนาได้ทุกเมื่อ (`/interject`) พร้อมตัวเลือกคำนวณตัวเลขสดผ่าน Sandbox `Decimal`
- **Closed-Loop Guardrails & Exports:** ตรวจความครอบคลุม 5 เสา (`DebatePillarCoverage`), ตรวจ Context Drift (`check_drift`), คำนวณ Scorecard ย้อนหลังแบบ Look-ahead-safe เทียบ Benchmark (`SPY` / `^SET.BK`), สรุปเป็น **Decision Playbook**, บันทึกเข้า **Obsidian Vault**, อัปเดต **Active Thesis**, และดาวน์โหลด **YouTube Production Kit (.zip)** (ไฟล์เสียง WAV ต่อเนื่อง, ซับไตเติล SRT, YouTube Chapters และ Visual Cue Sheet)

### 3. All-Asset Portfolio & Strategy Desk (`/portfolio`, `/advisory`)
- **3-Bucket Asset Allocation Framework:** จัดพอร์ตตามหลักการ 3 ถัง (สภาพคล่อง / พอร์ตหลักมั่นคง / พอร์ตเติบโต) พร้อมจำลอง Goal Funding Tension และ Look-Through Holdings
- **Multi-Portfolio Ledger Replay Engine:** ติดตามหลายพอร์ตพร้อมกัน แก้ไข/ลบธุรกรรมย้อนหลังแบบ Atomic และคำนวณ Weighted Average Cost (WAC) และ Realized P/L ใหม่อัตโนมัติ
- **Multi-Broker Ingestion:** นำเข้าและซิงก์ Statement อัตโนมัติทั้ง **SCB**, **Dime!** (รวม Gmail Trade Confirmation Sync & PDF), **ARR Gold** (ทองคำแท่ง 96.5% / 99.99%), **BBLAM PVD**, และ **WealthX**
- **Investor Principles & Advisory Desk:** สัมภาษณ์และประมวลผลกฎ 5 เสาหลักส่วนบุคคล (`Investor Essence`) พร้อมคำแนะนำ Action Stance รายตัว (Urgent Reduce, Accumulate, Hold, Winner Run ฯลฯ)

### 4. Market Research & Macro Cockpit (`/macro`)
- **4 ภูมิภาคยุทธศาสตร์ & Sector Rotation:** บทวิเคราะห์ AI (Regime Scenarios & 5D Evidence), สหรัฐฯ (US), ไทย (TH), ความเชื่อมโยง US–TH และ **Macro Sector Rotation Engine**
- **Hard Economic Data (38 FRED Series):** ข้อมูลเศรษฐกิจสหรัฐฯ และกลุ่มประเทศหลัก 38 ซีรีส์จาก **FRED API** (เงินเฟ้อ, ตลาดแรงงาน, Real GDP, Yield Spreads, ปริมาณเงิน M2, NFCI)
- **Market Data & Terminal V2 Feeds:** 19 ดัชนี/สินทรัพย์หลัก (`^IRX`, `^TNX`, `^VIX`, `DXY`, `HYG`/`LQD`, ทอง/น้ำมัน/ทองแดง, ดัชนีสหรัฐฯ, BTC-USD), 11 US Sector ETFs (GICS), 7 Regional Proxy ETFs และข้อมูล SEC EDGAR / Bond Yield

### 5. Equity Intel Suite — 3-Stage Stock Deep Dive (`/equity`, `/equity/:ticker`)
- **Stage 1 (Equity Quant):** วิเคราะห์งบการเงิน อัตราส่วนความสามารถทำกำไร (ROIC vs WACC, F-Score, Sloan Accrual) และโมเมนตัมราคา
- **Stage 2 (Equity Narrative):** สังเคราะห์บทวิเคราะห์ ข่าวสาร Transcript และการติดตามคำสัญญาผู้บริหาร (`Promise Tracker`)
- **Stage 3 (Equity Synthesizer):** รวมผลเป็นรายงานฉบับสมบูรณ์ พร้อม **Narrative Verifier** ตรวจความขัดแย้งของสมมติฐาน
- **Valuation Workbench:** ตาราง Sensitivity Matrix 5x5 สำหรับประเมินมูลค่า DCF, Reverse DCF และกราฟแท่งเทียน OHLCV

### 6. Daily & Weekly Investment Digest (`/digest`)
- **News Funnel & Freshness Tiering:** คัดกรองและจัดอันดับข่าวตาม Macro/Asset Impact พร้อมคัดข่าวซ้ำและความสดใหม่
- **YouTube, Podcast & NotebookLM Pipeline:** ถอดเสียง/สรุปคลิปสัมมนา คัดเลือกประเด็นสร้าง YouTube Pitch และส่งสร้างเสียง Podcast ผ่าน NotebookLM
- **Portfolio-Linked Matching:** จับคู่ข่าวกับหุ้นในพอร์ตและ Watchlist โดยระงับคำสับสน (`Ambiguity Suppression`)

### 7. Design System & Frontend Tech Stack
- **Frontend Stack:** React 19 + TypeScript (Strict) + Vite 8 + Tailwind CSS + Motion + **Three.js** (WebGL 3D Stage) + **KlineCharts** (Interactive OHLCV Charts)
- **Unified Font Stack:** **Plus Jakarta Sans** (ตัวเลข/ละตินสไตล์ Fintech) คู่กับ **Noto Sans Thai** พร้อมมาตรฐานแท็บหลัก `text-sm font-semibold`
- **End-to-End API Type Safety:** สร้าง TypeScript Schema อัตโนมัติจาก FastAPI OpenAPI (`web/openapi.json` → `web/src/api/types.generated.ts`) และมี script ตรวจ drift (`check:types`)

---

## 💻 การติดตั้งและเริ่มต้นใช้งาน (Getting Started)

### สิ่งที่ต้องมีในเครื่อง (Prerequisites)
- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/)** (Python Package Manager)
- **Node.js 18+** และ **npm**
- **Google Gemini API Key** ([Google AI Studio](https://aistudio.google.com/app/apikey))
- **FRED API Key** ([Federal Reserve Economic Data](https://fred.stlouisfed.org/docs/api/api_key.html))

### ขั้นตอนการติดตั้ง

**1. Clone Repository & ติดตั้ง Dependencies**
```bash
git clone https://github.com/gnoMarkII/Trinity-Wealth-Engine.git
cd invest-agents

# ติดตั้ง Backend Dependencies ด้วย uv
uv sync

# ติดตั้ง Frontend Dependencies
npm --prefix web install
```

**2. ตั้งค่า Environment Variables**
คัดลอกไฟล์ `.env.example` เป็น `.env`:
```bash
cp .env.example .env
```

| Key | ความจำเป็น | รายละเอียด |
|---|:---:|---|
| `GOOGLE_API_KEY` | ✅ | คีย์สำหรับโมเดล Gemini (LLM หลัก และ Gemini TTS สำหรับ Roundtable) |
| `FRED_API_KEY` | ✅ | คีย์ดึงข้อมูลเศรษฐกิจมหภาคจากฐานข้อมูล FRED |
| `WEBUI_PASSWORD` | ✅ | รหัสผ่านเดียวสำหรับล็อกอินเข้าสู่ Web UI |
| `SESSION_SECRET_KEY` | ✅ | สตริงสุ่มยาวอย่างน้อย 32 ตัวอักษรสำหรับลงนาม Session Cookie |
| `UNVERIFIED_DRAFT_SIGNING_KEY` | ✅ | สตริงสุ่มยาวอย่างน้อย 32 ตัวอักษรสำหรับ Draft HMAC Signature (ห้ามซ้ำกับ Session Secret) |
| `ANTHROPIC_API_KEY` / `OPENROUTER_API_KEY` | ⬜ | *(ทางเลือก)* สำหรับโมเดล Claude หรือ Cross-Provider Fallback |
| `ROUNDTABLE_MODEL` / `ROUNDTABLE_TTS_MODEL` | ⬜ | *(ทางเลือก)* Override โมเดล Roundtable & TTS (ค่าเริ่มต้นอ่านจาก `core/model_registry.py`) |
| `SEC_EDGAR_USER_AGENT` | ⬜ | *(ทางเลือก)* Header ตามนโยบาย Fair Access ของ SEC EDGAR |
| `OBSIDIAN_VAULT_PATH` | ⬜ | *(ทางเลือก)* พาธเก็บ Obsidian Vault (ค่าเริ่มต้น: `./memories`) |
| `DISCORD_WEBHOOK_URL` / `GMAIL_IMAP_*` | ⬜ | *(ทางเลือก)* แจ้งเตือน Discord Webhook และซิงก์ Trade Confirmation จาก Dime/SCB/WealthX |

---

## 🚀 คำสั่งรันระบบ (Exact CLI Commands)

### โหมดพัฒนา (Development Mode)

**Terminal 1 — Backend (FastAPI + Uvicorn):**
```powershell
.venv\Scripts\python -m uvicorn api.main:app --port 8000 --reload
```

**Terminal 2 — Frontend (Vite + React 19):**
```powershell
npm --prefix web run dev
```
เปิดเบราว์เซอร์ที่ `http://localhost:5173` แล้วล็อกอินด้วย `WEBUI_PASSWORD`

### โหมดใช้งานจริง (Production Build)
```powershell
# 1. Build Frontend Static Assets
npm --prefix web run build

# 2. รัน FastAPI เสิร์ฟทั้ง API และ Frontend (web/dist)
.venv\Scripts\python -m uvicorn api.main:app --port 8000
```
เปิดใช้งานได้โดยตรงที่ `http://localhost:8000`

---

## 🧪 การตรวจสอบคุณภาพโค้ด (Quality Gates)

ตามข้อตกลงใน `AGENTS.md` โค้ดทุกส่วนต้องผ่าน Quality Gates เหล่านี้ก่อนปิดงาน:

| งานที่ต้องตรวจสอบ | คำสั่ง | รายละเอียด |
|---|---|---|
| **Architecture & AST Rules** | `.venv\Scripts\pytest tests/architecture/test_dependency_rules.py -v` | ตรวจสอบ **21 Architecture & AST Invariant Tests** |
| **Backend Unit & API Suites** | `.venv\Scripts\pytest tests/ -v` | ครอบคลุม **2,822 Backend Tests** ใน `tests/` (Unit, API, Contracts, Architecture) |
| **Frontend Lint** | `npm --prefix web run lint` | ตรวจสอบโค้ดและกฎ Hooks ด้วย `oxlint` |
| **Frontend Typecheck** | `npm --prefix web run typecheck` | ตรวจสอบ TypeScript Types (`tsc -b`) |
| **Frontend Unit Tests** | `npm --prefix web test -- --run` | ชุดทดสอบ Vitest สำหรับ Pages, Hooks และ 3D/UI Components ใน `web/src/` |
| **API Schema & Type Drift Check** | `npm --prefix web run check:types` | ตรวจว่า `web/src/api/types.generated.ts` ตรงกับ FastAPI OpenAPI Schema |
| **Sync Golden Manifests & Types** | `.venv\Scripts\python scripts/generate_golden_manifests.py` และ `npm --prefix web run gen:types` | *(รันเฉพาะเมื่อได้รับอนุมัติให้ปรับ Contract Baseline หรือเปลี่ยน API Schema)* |

---

## 📁 โครงสร้างโปรเจกต์ (Repository Layout)

```
invest-agents/
├── api/                          # FastAPI Web Layer (REST, SSE, Dependencies, Routers, Workers)
│   ├── main.py                   # API App Entrypoint & Lifespan Event Handlers
│   ├── dependencies.py           # Composition Root & Service Dependency Injection
│   ├── jobs.py                   # Background Job Execution Queue
│   ├── db/                       # SQLite Connection, UnitOfWork (`uow.py`), DAOs & Repositories
│   └── routers/                  # Manager, Roundtable, Macro/Terminal V2, Equity, Portfolio, Thesis, Sync Routers
├── application/                  # Application Services & Hexagonal Ports
│   ├── manager/                  # Hermes Chat, Gemini SSE Stream Engine & Morning PM Review
│   ├── roundtable/               # Roundtable Orchestration, Context Assembler & Ports
│   ├── portfolio/                # Portfolio Accounting & Advisory Services
│   ├── macro/ & equity/          # Macro Cockpit & Equity Deep-Dive Workflows
│   └── thesis/ & earnings_call/  # Active Thesis Watchdog & Promise Check Services
├── agents/                       # LangGraph ReAct Agents & Research Swarm
│   ├── research_swarm_flow.py    # Deterministic Task Dispatcher & Background Swarm
│   ├── macro_quant_agent.py      # Macro Quant Specialist Agent
│   ├── macro_economist_agent.py  # Macro Economist Specialist Agent
│   ├── strategic_allocator.py    # Strategic Allocator with Institutional Guardrails
│   ├── equity_*.py               # Equity Intel Suite (Quant, Narrative, Synthesizer)
│   └── archivist_agent.py        # Knowledge Archivist Agent
├── core/                         # Domain Core, Primitives & Pure Financial/Domain Models
│   ├── roundtable/               # Personas, Voices, Settlement Scorecard (`Decimal`) & YouTube Export
│   ├── sandbox/                  # AST Guard & Memory Hygiene for Scratchpad
│   ├── equity_analytics/ & thesis/
│   ├── news_digest/ & portfolio_advisory/
│   ├── model_registry.py         # Centralized LLM & TTS Model Slots (`REGISTRY`)
│   ├── prompt_harness.py         # Skill & Prompt Markdown Loaders (`prompts/skills/`)
│   └── security.py               # PII Detection, Ticker Validation & Redaction
├── tools/                        # Concrete Adapters & Tools (Hexagonal Outer Layer)
│   ├── hermes_bridge/            # Trinity MCP Server (`21 Tools`, 8 Domains) & Sandbox Adapter
│   ├── roundtable/               # Debate Engine, Gemini TTS WAV Adapter & Adapters
│   ├── macro/                    # FRED API (38 Series), Yahoo Finance, Sector Rotation & News Funnel
│   ├── market/                   # Terminal V2 Feeds, Equity Quant Calculators & Ticker Resolver
│   ├── portfolio/                # Multi-Broker Parsers (SCB, Dime, ARR Gold, BBLAM PVD, WealthX)
│   └── archivist/                # Obsidian Vault V2 Atomic Writer & Vector Search
├── schemas/                      # Pydantic Contracts & DTOs (Macro, Equity, Portfolio, Roundtable ฯลฯ)
├── validators/                   # Output Verifiers, Narrative Verifier & Roundtable Guardrails
├── prompts/                      # Agent & Guru Roundtable Skill Prompts (`prompts/skills/`)
├── manager_sandbox/              # Hermes Isolated Sandbox (Scratch, Skills/_quarantine, Memory)
├── web/                          # React 19 + TypeScript + Vite 8 + Tailwind + Three.js Frontend
│   └── src/
│       ├── api/                  # API Client & Auto-generated OpenAPI Types (`types.generated.ts`)
│       ├── pages/                # ManagerAdvisor, RoundtableStudio, DailyDigest, Portfolio, Advisory, Macro, Equity
│       ├── components/           # Reusable UI, Manager (Chat/Kanban/Swarm), Roundtable (3D Stage/Room/Avatars/TTS)
│       └── lib/                  # Utilities, Formatting & Helpers
├── scripts/                      # Maintenance, Audit, Manifest Sync & OpenAPI Export Scripts
├── memories/                     # Obsidian PKM Vault (Layout V2)
└── tests/                        # Automated Suites (Architecture AST, Unit, API, Contracts)
```

---

## 📄 ข้อตกลงสิทธิ์การใช้งาน (License)

This project is licensed under the GNU AGPLv3 License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2026 Money ReRoute. All rights reserved.
