# Investment Manager AI — Trinity Wealth Engine

> ⚠️ **โปรเจกต์นี้อยู่ระหว่างการพัฒนา** ฟีเจอร์และสถาปัตยกรรมถูกออกแบบและทดสอบอย่างเข้มงวดตามมาตรฐาน Hexagonal Architecture

ระบบผู้ช่วยจัดการกองทุนส่วนตัวระดับสถาบันที่ขับเคลื่อนด้วย Multi-Agent System สร้างด้วย [LangGraph](https://github.com/langchain-ai/langgraph), FastAPI และ React 19 + TypeScript บนสถาปัตยกรรม **Single-Brain Manager (Hermes) + Autonomous Research Swarm (5 Specialists) + Deterministic Financial Engine** ที่แยกหน้าที่ระหว่างการสนทนาให้คำปรึกษา การวิจัยเชิงลึก การคำนวณตัวเลขทางการเงิน และการจัดเก็บบันทึกความรู้ออกจากกันอย่างเคร่งครัด

---

## 🏛️ สถาปัตยกรรมระบบ (System Architecture)

```
คุณ (User)
    │
    ▼
PIIMiddleware ─── ตรวจจับและเซ็นเซอร์ข้อมูลส่วนตัว (Thai ID, บัตรเครดิต, อีเมล, เบอร์โทรศัพท์)
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    The Manager (Hermes Proactive Advisor)                       │
│  - SSE Streaming Chat Console พร้อม Intent Classifier (Read Mode vs Run Orders) │
│  - Minimalist Chat Workspace พร้อม Split-Pane Resizable Side Panel (340–1200px) │
│  - Python Scratchpad Sandbox (AST Guard + Memory Hygiene)                       │
│  - Trinity MCP Server (17 Tools ครอบคลุม 8 โดเมน)                               │
└───────────────────────┬─────────────────────────────────┬───────────────────────┘
                        │                                 │
           ┌────────────▼──────────────┐   ┌──────────────▼──────────────────────┐
           │   Deterministic Services  │   │   Research Swarm Flow (LangGraph)   │
           │  (Python & Decimal Engine)│   │  (5 Autonomous Specialist Agents)   │
           ├───────────────────────────┤   ├─────────────────────────────────────┤
           │ • PortfolioService        │   │ 1. Macro Quant (Quant Matrix)       │
           │ • PortfolioAdvisoryService│   │ 2. Macro Economist (Global Policy)  │
           │ • Daily & Weekly Digest   │   │ 3. Strategic Allocator (Guardrails) │
           │ • Multi-Broker Ingestion  │   │ 4. Equity Intel (3-Stage Deep Dive) │
           │ • Multi-Tier Ledger Engine│   │ 5. The Archivist (Obsidian Vault V2)│
           └───────────────────────────┘   └─────────────────────────────────────┘
```

### ทีมงาน AI & Core Services (Roster)

| Agent / Service | บทบาท | กลไกและเครื่องมือหลัก |
|---|---|---|
| **The Manager (Hermes)** | Single-Brain Advisor — รับคำถาม สนทนาแบบ Real-time ให้คำปรึกษา และสั่งการระบบ | SSE Streaming, Intent Routing, Python Sandbox (AST Whitelist), Trinity MCP Bridge |
| **Research Swarm Flow** | Background Multi-Agent Swarm สำหรับการวิเคราะห์และวิจัยเชิงลึก | LangGraph Orchestrator + Deterministic Task Classifier (`classify_research_tasks`) |
| **Macro Quant** | คำนวณ Yield Spread, ค่าเงินดอลลาร์ และดัชนีชี้วัดเศรษฐกิจเชิงปริมาณ | Quantitative Observable Matrix, FRED Data Matrix, Bond Yield Curve, Liquidity Proxy |
| **Macro Economist** | วิเคราะห์นโยบายการเงิน ธนาคารกลาง ทิศทางดอกเบี้ย และสภาวะเศรษฐกิจมหภาค | Global Macro & Policy Specialist, Fed/BoT Tracker, Policy Sentiment, Regime Synthesis |
| **Strategic Allocator** | กำหนดสัดส่วนสินทรัพย์ 3 ถัง (สภาพคล่อง / หลัก / เติบโต) ตามสภาวะตลาด | Guardrails & Portfolio Optimizer, 3-Bucket Allocation, Risk Parity, Regime Guardrails |
| **Equity Intel Suite** | วิเคราะห์หุ้นรายตัวเชิงลึก 3 ขั้นตอน: งบการเงิน, Transcript และประเมินมูลค่า | 3-Stage Stock Deep Dive (Quant → Narrative → Synthesizer), 5x5 DCF Sensitivity, Narrative Verifier |
| **The Archivist** | จัดการฐานความรู้ บันทึกบทเรียน และประวัติพอร์ตลงใน Obsidian Vault (Layout V2) | Obsidian Adapter, Vector Search (RAG), Vault V2 Atomic Writer, Audit Trail |
| **Portfolio Services** | ระบบบัญชีพอร์ต การคำนวณผลตอบแทน และคำแนะนำการปรับพอร์ตแบบ Deterministic | Ledger Replay Engine, Multi-Broker Ingestion (SCB, Dime, ARR Gold, BBLAM PVD, WealthX), Dividend & FX Sync |
| **Daily Digest Engine** | ระบบคัดกรอง สรุป และแจ้งเตือนข่าวสาร/คลิปการลงทุนประจำวัน | News Funnel, Freshness Tiering, Boundary Regex Matching, One-Shot AI Synthesis |

> **กฎเหล็กสถาปัตยกรรม (Hard Architecture Rules):**
> 1. **Single-Brain Manager Architecture:** Hermes เป็น Manager ศูนย์กลางเพียงตัวเดียว ส่วน Research Swarm ทำงานเบื้องหลังด้วย Deterministic Python Dispatcher ปราศจาก LLM Router ซ้ำซ้อน
> 2. **Bookkeeper Agent Decommissioned:** ห้ามใช้ LLM คำนวณตัวเลขทางบัญชี โดยเปลี่ยนมาใช้ `PortfolioService` และ `PortfolioAdvisoryService` ที่คำนวณตัวเลขด้วย Python `Decimal` เพื่อความถูกต้องแม่นยำ 100%
> 3. **Hexagonal Architecture:** ปฏิบัติตาม 16 AST Dependency Rules อย่างเคร่งครัด แยก Domain, Application, Tools/Adapters และ API ออกจากกัน

---

## ⚡ ฟีเจอร์หลักของระบบ

### 1. Hermes Proactive Manager & AI Advisor (`/manager`)
- **Minimalist Chat-First Workspace:** หน้าจอเริ่มต้นสไตล์ Calm UI ทำความสะอาดพื้นที่ว่างอย่างลงตัว พร้อม Starter Prompt Pills บรรทัดเดียวที่คลิกเริ่มต้นได้ใน 1 วินาที
- **Interactive Resizable Right Panel:** แถบด้านขวา (บอร์ดงาน Kanban / Trinity Swarm / Morning Delta) สามารถ **ลากปรับยืด-หดได้อิสระ (340px – 1200px)** พร้อมระบบตรวจจับความกว้างและสลับโหมดคอลัมน์อัตโนมัติ (Compact 1 คอลัมน์ ↔ Wide 4 คอลัมน์) จดจำขนาดลง `localStorage` และรองรับการควบคุมผ่านคีย์บอร์ด
- **Real-Time Streaming Chat:** สนทนาแบบ Server-Sent Events (SSE) พร้อม Fluid Thinking Orb แสดงสถานะการคิด และแยก Intent ระหว่างคำถามทั่วไป (Read Mode) กับคำสั่งเริ่มงานวิจัย (Run Orders)
- **Live Kanban Board & Execution Tracking:** ติดตามสถานะงานวิจัยแบบเรียลไทม์ (Backlog, Request Approve, Workers Executing, Done) พร้อมระบบ Human-in-the-loop Approval
- **Morning Review & Delta Watchdog:** สรุปประเด็นเศรษฐกิจและการเปลี่ยนแปลงของพอร์ตประจำวัน ตรวจจับ Thesis Exceptions และมีระบบ Sign-off ยืนยันความถูกต้อง
- **Python Scratchpad Sandbox:** สภาพแวดล้อมรันคำนวณทางการเงินปลอดภัยด้วย **AST Guard** (บล็อก `eval`, `exec`, `open` และ Forbidden Imports)
- **Self-Evolving Skills & Memory:** พัฒนาสูตรคำนวณใหม่และทดสอบใน Sandbox (`manager_sandbox/skills/`) มี Quarantine Gate ก่อน Promote ใช้งานจริง

### 2. All-Asset Portfolio & Strategy Desk (`/portfolio`, `/advisory`)
- **3-Bucket Asset Allocation Framework:** จัดพอร์ตตามหลักการ 3 ถัง (สภาพคล่อง / พอร์ตหลักมั่นคง / พอร์ตเติบโต) พร้อมจำลอง Goal Funding Tension
- **Multi-Portfolio Ledger Replay Engine:** ติดตามหลายพอร์ตพร้อมกัน รองรับการแก้ไข/ลบธุรกรรมย้อนหลังแบบ Atomic และคำนวณ Weighted Average Cost (WAC) และ Realized P/L ใหม่อัตโนมัติ
- **Multi-Broker Ingestion:** นำเข้าข้อมูลและประมวลผล Statement อัตโนมัติ:
  - **SCB:** หุ้นไทยและกองทุนรวม
  - **Dime!:** หุ้นสหรัฐฯ บัญชีออมทรัพย์ดอกเบี้ยสูง และการแปลงค่าเงิน USD/THB
  - **ARR Gold:** ทองคำแท่ง 96.5% / 99.99%
  - **BBLAM PVD:** กองทุนสำรองเลี้ยงชีพ ประมวลผลประวัติเงินสะสม/สมทบ และ NAV ย้อนหลัง
  - **WealthX:** พอร์ตสินทรัพย์ทางเลือก
- **Investor Principles & Essence:** สัมภาษณ์และประมวลผลกฎ 5 เสาหลักการลงทุนส่วนบุคคล
- **Portfolio Advisory Desk:** วิเคราะห์ Action Stance รายตัว (เคาะซื้อ/ตัดขายทันที, ตั้งรับตามกรอบ, Urgent Reduce, Accumulate, Winner Run)

### 3. Market Research & Macro Cockpit (`/macro`)
- **4 ภูมิภาคยุทธศาสตร์:** บทวิเคราะห์ AI (Regime Scenarios & 5D Evidence), สหรัฐอเมริกา (US), ประเทศไทย (TH), และความเชื่อมโยง US–TH (Fed vs BoT Spreads & Policy Transmission)
- **Hard Economic Data:** ดึงข้อมูลเศรษฐกิจ 38 ซีรีส์จาก **FRED API** (เงินเฟ้อ, การจ้างงาน, อัตราดอกเบี้ย, ปริมาณเงิน)
- **Market Data Feeds:** 19 ดัชนีหลักจาก Yahoo Finance, 11 US Sector ETFs (GICS Standard), 7 Regional Proxy ETFs
- **Terminal V2 Data Feeds:** ติดตามกระแสเงินทุน Crypto ETF (FarSide BTC/ETH), เส้นผลตอบแทนพันธบัตรรัฐบาลไทย และข้อมูลเปิดเผยจาก SEC EDGAR

### 4. Equity Intel Suite — 3-Stage Stock Deep Dive (`/equity`)
- **Stage 1 (Equity Quant):** วิเคราะห์งบการเงิน อัตราส่วนความสามารถทำกำไร สภาพคล่อง และโมเมนตัมราคา
- **Stage 2 (Equity Narrative):** สังเคราะห์บทวิเคราะห์ ข่าวสาร และประเด็นสำคัญจากการประชุมนักวิเคราะห์
- **Stage 3 (Equity Synthesizer):** รวมผลเป็นรายงานฉบับสมบูรณ์ พร้อม **Narrative Verifier** ตรวจจับความขัดแย้งของสมมติฐาน
- **Valuation Workbench:** ตาราง Sensitivity Matrix 5x5 สำหรับประเมินมูลค่า DCF ตามฉากทัศน์ Growth และ Discount Rate
- **Institutional Cards:** Earnings Bridge, Capital Allocation Yield Meter, Variant Perception, Expectation Gap, Smart Money Flow และ Promise Tracker

### 5. Daily & Weekly Investment Digest (`/digest`)
- **News Funnel:** คัดกรองข่าวสารและขจัดข่าวซ้ำซ้อนด้วยคะแนน Macro/Asset Impact
- **Freshness Tiering:** จัดลำดับความสดใหม่ (`ultra_fresh` < 4 ชม., `fresh_today` < 24 ชม., ตัดทิ้งข่าวเก่า > 36 ชม.)
- **YouTube & Podcast Intelligence:** ถอดเสียงและดึงเนื้อหาเชิงลึกจากคลิปสัมมนาและพอดแคสต์การลงทุน
- **Portfolio-Linked Matching:** จับคู่ข่าวสารกับหุ้นในพอร์ตและ Watchlist โดยระงับคำสับสน (Ambiguity Suppression)

### 6. Design System & Typography
- **Unified Font Stack:** ประกาศใช้งานคู่ฟอนต์ **Plus Jakarta Sans** (สำหรับตัวเลข ละติน และสัญลักษณ์การเงินระดับ Fintech) ควบคู่กับ **Noto Sans Thai** (สำหรับภาษาไทยที่เว้นวรรค วรรณยุกต์ และสระคมชัด สบายตา)
- **Consistent Tab Scale:** ปรับมาตรฐานขนาดแท็บนำทางหลักทุกหน้าให้มีขนาด `text-sm font-semibold` (14px) เท่ากันทั้งระบบ

---

## 💻 การติดตั้งและเริ่มต้นใช้งาน (Getting Started)

### สิ่งที่ต้องมีในเครื่อง (Prerequisites)
- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/)** (Python Package Manager ประสิทธิภาพสูง)
- **Node.js 18+** และ **npm**
- **Google Gemini API Key** ([Google AI Studio](https://aistudio.google.com/app/apikey))
- **FRED API Key** ([Federal Reserve Economic Data](https://fred.stlouisfed.org/docs/api/api_key.html))

### ขั้นตอนการติดตั้ง

**1. Clone Repository & ติดตั้ง Dependencies**
```bash
git clone <repo-url>
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
| `GOOGLE_API_KEY` | ✅ | คีย์สำหรับโมเดล Gemini (LLM หลักสำหรับ Hermes และ Research Swarm) |
| `FRED_API_KEY` | ✅ | คีย์ดึงข้อมูลเศรษฐกิจมหภาคสหรัฐฯ จากฐานข้อมูล FRED |
| `WEBUI_PASSWORD` | ✅ | รหัสผ่านสำหรับล็อกอินเข้าสู่ Web UI |
| `SESSION_SECRET_KEY` | ✅ | สตริงสุ่มยาวอย่างน้อย 32 ตัวอักษรสำหรับลงนาม Session Cookie |
| `UNVERIFIED_DRAFT_SIGNING_KEY` | ✅ | สตริงสุ่มยาวอย่างน้อย 32 ตัวอักษรสำหรับ Draft Signature |
| `OPENROUTER_API_KEY` | ⬜ | *(ทางเลือก)* สำหรับโมเดล Fallback สำรอง |

---

## 🚀 คำสั่งรันระบบ (Exact CLI Commands)

### โหมดพัฒนา (Development Mode)

**Terminal 1 — Backend (FastAPI + Uvicorn):**
```bash
.venv\Scripts\python -m uvicorn api.main:app --port 8000 --reload
```

**Terminal 2 — Frontend (Vite + React 19):**
```bash
npm --prefix web run dev
```
เปิดเบราว์เซอร์ที่ `http://localhost:5173` แล้วล็อกอินด้วย `WEBUI_PASSWORD`

### โหมดใช้งานจริง (Production Build)
```bash
# 1. Build Frontend Static Assets
npm --prefix web run build

# 2. รัน FastAPI ทำหน้าที่เสิร์ฟทั้ง API และ Frontend
.venv\Scripts\python -m uvicorn api.main:app --port 8000
```
เปิดใช้งานได้โดยตรงที่ `http://localhost:8000`

---

## 🧪 การตรวจสอบคุณภาพโค้ด (Quality Gates)

ตามข้อตกลงใน `AGENTS.md` ทุกฟีเจอร์และโค้ดที่แก้ไขต้องผ่าน Green Test Gates ทั้งหมด:

| งานที่ต้องตรวจสอบ | คำสั่ง | ผลลัพธ์และรายละเอียด |
|---|---|---|
| **Architecture & AST Rules** | `.venv\Scripts\pytest tests/architecture/test_dependency_rules.py -v` | ผ่านครบทั้ง 19 Hexagonal Dependency Rules |
| **Backend Unit & API Tests** | `.venv\Scripts\pytest tests/ -v` | ทดสอบครอบคลุม 1,300+ Backend Tests |
| **Frontend Typecheck** | `npm --prefix web run typecheck` | ตรวจสอบ TypeScript Types (`tsc -b`) ผ่าน 0 Errors |
| **Frontend Unit Tests** | `npm --prefix web test -- --run` | ทดสอบครบทั้ง 123 ไฟล์ (625 Vitest Tests) ใน `web/` |
| **Update Golden Manifests** | `.venv\Scripts\python scripts/generate_golden_manifests.py` | *(รันเฉพาะเมื่อได้รับอนุมัติให้ปรับ Contract Baseline)* |

---

## 📁 โครงสร้างโปรเจกต์ (Repository Layout)

```
invest-agents/
├── api/                         # FastAPI Web Layer (REST, SSE, Dependencies, Routers)
│   ├── main.py                  # API App Entrypoint & Lifespan Event Handlers
│   ├── dependencies.py          # Application Service Dependency Injection
│   ├── jobs.py                  # Background Job Execution Queue
│   └── routers/                 # Modular Routers (Manager, Macro, Equity, Portfolio, etc.)
├── application/                 # Application Services & Hexagonal Ports
│   ├── manager/                 # Hermes Chat & Gemini SSE Stream Engine
│   ├── macro/                   # Macro & Digest Application Services
│   └── portfolio/               # Portfolio Accounting & Advisory Services
├── agents/                      # LangGraph ReAct Agents & Multi-Agent Swarm
│   ├── research_swarm_flow.py   # Multi-agent Background Research Swarm
│   ├── macro_quant_agent.py     # Macro Quant Specialist Agent
│   ├── macro_economist_agent.py # Macro Economist Specialist Agent
│   ├── strategic_allocator.py   # Strategic Allocator with Institutional Guardrails
│   ├── equity_*.py              # Equity Intel Suite (3-Stage Agents)
│   └── archivist_agent.py       # Knowledge Archivist Agent
├── core/                        # Domain Core, Primitives & Pure Models
│   ├── sandbox/                 # AST Guard & Memory Hygiene for Scratchpad
│   ├── news_digest/             # News Funnel Core Models & Deduplication
│   ├── model_registry.py        # Centralized LLM Model Slots & Configurations
│   ├── prompt_harness.py        # Skill & Prompt Markdown Loaders
│   └── security.py              # PII Detection & Redaction
├── manager_sandbox/             # Hermes Isolated Sandbox (Skills, Memory, Audit Logs)
├── tools/                       # Concrete Adapters & Tools (Hexagonal Outer Layer)
│   ├── hermes_bridge/           # Trinity MCP Server & Sandbox Adapters
│   ├── macro/                   # FRED API, Yahoo Finance & Digest Stores
│   ├── market/                  # Terminal V2 Feeds & Equity Quant Calculators
│   ├── portfolio/               # Multi-Broker Parsers (SCB, Dime, ARR, PVD, WealthX)
│   └── archivist/               # Obsidian Vault V2 Atomic Writer & Vector Search
├── validators/                  # Guardrails & Output Verifiers
├── web/                         # React 19 + TypeScript + Tailwind CSS Frontend
│   └── src/
│       ├── pages/               # ManagerAdvisor, Portfolio, Macro, Equity, DailyDigest
│       ├── components/          # Reusable UI Components, Cards, Charts & Modals
│       │   └── manager/         # ChatStream, Composer, Resizable RightPanel, Kanban, Swarm
│       └── lib/                 # Utility Functions, Types & API Client
├── memories/                    # Obsidian PKM Vault (Layout V2)
└── tests/                       # Automated Test Suites (Architecture, Unit, Contracts)
```

---

## 📄 ข้อตกลงสิทธิ์การใช้งาน (License)

This project is licensed under the GNU AGPLv3 License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2026 Money ReRoute. All rights reserved.
