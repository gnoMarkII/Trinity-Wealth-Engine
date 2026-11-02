# Investment Manager AI — Trinity Wealth Engine

> ⚠️ **โปรเจกต์นี้อยู่ระหว่างการพัฒนา** ฟีเจอร์และสถาปัตยกรรมถูกออกแบบและทดสอบอย่างเข้มงวดตามมาตรฐาน Hexagonal Architecture

ระบบผู้ช่วยจัดการกองทุนส่วนตัวระดับสถาบันที่ขับเคลื่อนด้วย Multi-Agent System สร้างด้วย [LangGraph](https://github.com/langchain-ai/langgraph) และ FastAPI บน Python โดยมีสถาปัตยกรรม **Single-Brain Manager (Hermes) + Autonomous Research Swarm + Deterministic Financial Engine** ที่แยกหน้าที่ระหว่างการสนทนาให้คำปรึกษา การวิจัยเชิงลึก การคำนวณตัวเลขทางการเงิน และการบันทึกความจำออกจากกันอย่างชัดเจน

---

## 🏛️ สถาปัตยกรรมระบบ

```
คุณ (User)
    │
    ▼
PIIMiddleware ─── ตรวจและลบข้อมูลส่วนตัว (Thai ID, บัตรเครดิต, อีเมล, เบอร์โทร)
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    The Manager (Hermes Proactive Advisor)                       │
│  - SSE Streaming Chat Console พร้อม Intent Classifier (Read Mode vs Run Orders) │
│  - Python Scratchpad Sandbox (AST Guard + Memory Hygiene)                       │
│  - Trinity MCP Server (17 Tools ครอบคลุม 8 โดเมน)                               │
└───────────────────────┬─────────────────────────────────┬───────────────────────┘
                        │                                 │
           ┌────────────▼──────────────┐   ┌──────────────▼──────────────────────┐
           │   Deterministic Services  │   │   Research Swarm Flow (LangGraph)   │
           │  (Python & Decimal Engine)│   │  (Background Task Dispatcher)       │
           ├───────────────────────────┤   ├─────────────────────────────────────┤
           │ • PortfolioService        │   │ • The Archivist (Vault Layout V2)   │
           │ • PortfolioAdvisoryService│   │ • Macro Quant & Macro Economist     │
           │ • Daily & Weekly Digest   │   │ • Strategic Allocator (Guardrails)  │
           │ • Multi-Broker Adapters   │   │ • Equity Intel Suite (3-Stage)      │
           └───────────────────────────┘   └─────────────────────────────────────┘
```

### ทีมงาน AI & Core Services

| Agent / Service | บทบาท | กลไกและเครื่องมือหลัก |
|---|---|---|
| **The Manager (Hermes)** | Single-Brain Advisor — รับคำถาม สนทนาแบบ Real-time ให้คำปรึกษา และสั่งการระบบ | SSE Streaming, Intent Routing, Python Sandbox (AST Whitelist), Trinity MCP Bridge |
| **Research Swarm Flow** | Background Multi-Agent Swarm สำหรับการวิเคราะห์และวิจัยเชิงลึก | LangGraph Orchestrator + Deterministic Task Classifier (`classify_research_tasks`) |
| **The Archivist** | จัดการฐานความรู้และการบันทึกลง Obsidian Vault (Layout V2) | Vault V2 Atomic Writer, Semantic Search (Vector RAG), Graph Context, YouTube/Article Ingestion |
| **Macro Quant / Economist** | วิเคราะห์คะแนนเศรษฐกิจเชิงปริมาณ (Quant Matrix) และสังเคราะห์ Narrative | Yahoo Finance, FRED API, Terminal V2 Adapters, News Radar |
| **Strategic Allocator** | สังเคราะห์กลยุทธ์จัดสรรสินทรัพย์ระดับสถาบัน พร้อม Guardrails | Institutional Guardrails (Valuation/Credit/Correlation), Structured-Output Retry Loop |
| **Equity Intel Suite** | วิเคราะห์หุ้นรายตัวแบบ 3-Stage Pipeline | Equity Quant (Deterministic) → Equity Narrative → Equity Synthesizer + Narrative Verifier |
| **Portfolio Services** | ระบบบัญชีพอร์ต การคำนวณผลตอบแทน และคำแนะนำการปรับพอร์ตแบบ Deterministic | Ledger Replay Engine, Multi-Broker Ingestion (SCB, Dime, ARR Gold, BBLAM, WealthX), Dividend & FX Sync |
| **Daily Digest Engine** | ระบบคัดกรอง สรุป และแจ้งเตือนข่าวสาร/คลิปการลงทุนประจำวัน | Freshness Tiering, Ticker Boundary Matching, One-Shot AI Synthesis, Read State Tracking |

> **หมายเหตุสถาปัตยกรรม:**
> 1. **Single-Brain Manager Architecture:** Hermes เป็น Manager ศูนย์กลางเพียงตัวเดียว ส่วน Research Swarm ทำงานเบื้องหลังด้วย Deterministic Python Dispatcher ปราศจาก LLM Router ซ้ำซ้อน
> 2. **Bookkeeper Agent Decommissioned:** ยกเลิกการใช้ LLM ในการทำบัญชีพอร์ต โดยเปลี่ยนมาใช้ `PortfolioService` และ `PortfolioAdvisoryService` ที่คำนวณตัวเลขด้วย Python `Decimal` เพื่อความถูกต้องแม่นยำ 100%

---

## ⚡ ฟีเจอร์ปัจจุบัน

### 1. Hermes Proactive Manager & Sandbox Engine (`/manager`)
- **Real-Time Streaming Advisor:** สนทนาแบบ Server-Sent Events (SSE) ตอบสนองรวดเร็ว พร้อม Intent Routing แยกคำถามเชิงข้อมูล (Read/Thesis Mode) ออกจากคำสั่งเริ่มงาน (Action Run Orders) ป้องกันการ Dispatch งานเบื้องหลังโดยไม่ตั้งใจ
- **Python Scratchpad Sandbox:** รันการคำนวณทางการเงินสดผ่านสภาพแวดล้อมที่ควบคุมความปลอดภัยด้วย **AST Guard** (Whitelist คำสั่ง, บล็อก `eval`, `exec`, `open`, `__subclasses__` และ Forbidden Imports)
- **Memory & Prompt Hygiene:** คัดกรองและป้องกัน Prompt Injection, ตัด Script/HTML Tags ที่เป็นอันตราย และเซ็นเซอร์ข้อมูลส่วนบุคคล (PII) อัตโนมัติก่อนบันทึกความจำ
- **Self-Evolving Skills:** สามารถพัฒนาสูตรคำนวณหรือ Skill ขึ้นมาใหม่ ทดสอบใน Sandbox (`manager_sandbox/skills/`) และมีระบบกักกัน (Quarantine Gate) ก่อน Promote ใช้งานจริง
- **Trinity MCP Bridge:** เชื่อมต่อ 17 Tools ครอบคลุม 8 โดเมน (Portfolio, Macro, Terminal V2, Equity, Knowledge, Vault, System, Web Search) ผ่าน Tool Harness ที่มี Failsafe

### 2. Daily & Weekly Investment Digest (`/digest`)
- **Freshness Tiering:** ประเมินความสดใหม่ของข่าวสารและวิดีโอ 3 ระดับ (`ultra_fresh` < 4 ชม., `fresh_today` < 24 ชม., `stale` > 36 ชม. ถูกตัดทิ้งอัตโนมัติ)
- **Portfolio & Ticker Matching:** จับคู่ข่าวกับหุ้นในพอร์ตและ Watchlist แบบ Boundary Regex และ Ambiguity Suppression (ป้องกันคำสับสน เช่น ONE, IT, FOR)
- **One-Shot Synthesis:** สรุปภาพรวมตลาดประจำวันและสัปดาห์ด้วย AI แบบรวบรัดตรงประเด็น พร้อมระบุประเด็นกระทบพอร์ต
- **Read State Tracking:** จดจำสถานะข่าวที่อ่านแล้ว บันทึกและดึงกลับมาดูย้อนหลังได้

### 3. ข้อมูลมหภาค & Terminal V2 Observables (`/macro`)
- **Real-time Market Data:** ติดตาม 19 ดัชนีหลักจาก Yahoo Finance, 11 US Sector ETFs (GICS Standard) และ 7 Regional Proxy ETFs
- **Economic Hard Data:** ตัวเลขเศรษฐกิจ 38 ซีรีส์จาก **FRED API** (นโยบายการเงิน, เงินเฟ้อ, แรงงาน, สภาพคล่อง)
- **Terminal V2 Data Feeds:** รองรับข้อมูลสภาพคล่องคริปโต (FarSide BTC/ETH ETF Flows), อัตราแลกเปลี่ยนและพันธบัตรไทย (MOF, BOT), และข้อมูลประกาศจาก SEC EDGAR / SETTrade

### 4. Strategic Allocator — Institutional Guardrails
- **Valuation & Risk Pillars:** คำนวณ Equity Risk Premium (ERP), Forward Earnings Yield, Credit Spreads, Pair-Trade Ratios และ Rolling Correlation
- **LLM-Agnostic Warning Registry:** คำเตือนโครงสร้างพร้อมคำแปลไทยกลาง (`WarningMessage`) และระดับความรุนแรง
- **Structured-Output Retry Loop:** ตรวจจับความผิดพลาดเชิงโครงสร้างและส่ง Feedback ให้ LLM แก้ไขอัตโนมัติก่อน Fallback

### 5. Equity Intel Suite — วิเคราะห์หุ้น 3-Stage Pipeline (`/equity`)
- **Stage 1 (Equity Quant):** สกัดตัวเลขงบการเงิน อัตราส่วนทางการเงิน สุขภาพทางการเงิน และโมเมนตัมราคา
- **Stage 2 (Equity Narrative):** สังเคราะห์บทวิเคราะห์และข่าวสาร
- **Stage 3 (Equity Synthesizer):** รวมผลเป็นรายงานฉบับสมบูรณ์ พร้อม **Narrative Verifier** ป้องกันความขัดแย้งของข้อความและการคำนวณ DCF

### 6. การจัดการพอร์ตโฟลิโอ & Advisory (`/portfolio`, `/advisory`)
- **Multi-Portfolio & Ledger Replay Engine:** ติดตามหลายพอร์ตพร้อมกัน รองรับการแก้ไข/ลบธุรกรรมย้อนหลังแบบ Atomic และ Recalculate ต้นทุนเฉลี่ยถ่วงน้ำหนัก (WAC) และ Realized P/L ทั้งหมดใหม่อย่างแม่นยำ
- **Multi-Broker Ingestion:** รองรับการนำเข้าข้อมูลจากไฟล์ PDF/CSV และ Statement ของ SCB, Dime, ARR Gold, BBLAM PVD และ WealthX
- **Dividend & FX Sync:** ดึงประวัติเงินปันผลสุทธิหลังหักภาษี (US 15% / TH 10%) พร้อมอัตราแลกเปลี่ยนย้อนหลังตามวันทำธุรกรรมจริง
- **Portfolio Advisory Playbook:** คำแนะนำการ Rebalance พอร์ตตาม Target Allocations และ Risk Buckets

### 7. Obsidian Vault PKM (Layout V2)
- เก็บความรู้และรายงานการลงทุนลง Obsidian Vault ภายใต้โครงสร้าง V2 ที่มีความปลอดภัย
- ค้นหาด้วย Semantic Search (Vector RAG) และ Graph Context
- เขียนไฟล์แบบ Atomic Swap (`_atomic_io.py`) ป้องกันไฟล์เสียหายเมื่อระบบหยุดทำงานกะทันหัน

---

## 💻 การติดตั้งและเริ่มต้นใช้งาน

### สิ่งที่ต้องมี
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (Python Package Manager)
- Node.js 18+ และ npm
- Google Gemini API Key
- FRED API Key (ฟรี)

### ขั้นตอนการติดตั้ง

**1. Clone Repository & ติดตั้ง Dependencies**
```bash
git clone <repo-url>
cd invest-agents

# ติดตั้ง Backend Dependencies
uv sync

# ติดตั้ง Frontend Dependencies
cd web && npm install && cd ..
```

**2. ตั้งค่า Environment Variables**
คัดลอกไฟล์ตัวอย่างและใส่ API Key ของคุณ:
```bash
cp .env.example .env
```

| Key | จำเป็น | รายละเอียด |
|---|:---:|---|
| `GOOGLE_API_KEY` | ✅ | คีย์หลักสำหรับ Gemini Models ([Google AI Studio](https://aistudio.google.com/app/apikey)) |
| `FRED_API_KEY` | ✅ | คีย์ดึงข้อมูลเศรษฐกิจสหรัฐฯ ([FRED](https://fred.stlouisfed.org/docs/api/api_key.html)) |
| `WEBUI_PASSWORD` | ✅ | รหัสผ่านสำหรับเข้าใช้งาน Web UI |
| `SESSION_SECRET_KEY` | ✅ | สตริงสุ่มยาวอย่างน้อย 32 ตัวอักษรสำหรับ Session Cookie |
| `UNVERIFIED_DRAFT_SIGNING_KEY` | ✅ | สตริงสุ่มยาวอย่างน้อย 32 ตัวอักษรสำหรับ Draft Signature |
| `OPENROUTER_API_KEY` | ⬜ | ออปชันเสริมสำหรับ Fallback Model |

---

## 🚀 การรันระบบ (Commands)

### Development Mode (แยก 2 Terminals)

**Terminal 1 — Backend (FastAPI):**
```bash
.venv\Scripts\python -m uvicorn api.main:app --port 8000 --reload
```

**Terminal 2 — Frontend (Vite + React):**
```bash
npm --prefix web run dev
```
เปิดเบราว์เซอร์ที่ `http://localhost:5173` แล้วล็อกอินด้วย `WEBUI_PASSWORD`

### Production Mode (Single Process)
Build frontend แล้วให้ FastAPI ทำหน้าที่เสิร์ฟ Static Assets:
```bash
npm --prefix web run build
.venv\Scripts\python -m uvicorn api.main:app --port 8000
```
เปิดใช้งานได้โดยตรงที่ `http://localhost:8000`

---

## 🧪 การทดสอบระบบ (Quality Gates)

ตามมาตรฐานที่กำหนดใน `AGENTS.md` ระบบต้องผ่าน Green Test Gate ทั้งหมดก่อน Commit:

| การทดสอบ | คำสั่ง | รายละเอียด |
|---|---|---|
| **Architecture & AST Rules** | `.venv\Scripts\pytest tests/architecture/test_dependency_rules.py -v` | ตรวจสอบ Hexagonal 16 Dependency Rules |
| **Backend Unit & API Tests** | `.venv\Scripts\pytest tests/ -v` | ทดสอบครอบคลุม 1,300+ Backend Tests |
| **Frontend Typecheck** | `npm --prefix web run typecheck` | ตรวจสอบ TypeScript Types (0 errors) |
| **Frontend Unit Tests** | `npm --prefix web test -- --run` | ทดสอบ 599 Vitest Tests ใน `web/` |
| **Update Golden Manifests** | `.venv\Scripts\python scripts/generate_golden_manifests.py` | *(รันเฉพาะเมื่อมีการเปลี่ยนแปลง Contract ที่ผ่านการอนุมัติ)* |

---

## 📁 โครงสร้างโปรเจกต์

```
invest-agents/
├── api/                         # FastAPI Layer (REST, SSE, Dependencies, Routers)
│   ├── main.py                  # API App Entrypoint & Lifespan
│   ├── dependencies.py          # Application Service Dependency Injection
│   ├── jobs.py                  # Background Job Queue
│   └── routers/                 # Modular Routers (Manager, Macro, Equity, Portfolio, etc.)
├── application/                 # Application Services & Ports (Hexagonal)
│   ├── manager/                 # Hermes Chat & Gemini Stream Engine
│   └── macro/                   # Digest Application Service & Ports
├── agents/                      # LangGraph ReAct Agents & Background Flows
│   ├── research_swarm_flow.py   # Multi-agent Research Swarm
│   ├── archivist_agent.py       # PKM Archivist Agent
│   ├── macro_quant_agent.py     # Macro Quant Agent
│   ├── macro_economist_agent.py # Macro Economist Agent
│   ├── strategic_allocator.py   # Strategic Allocator with Guardrails
│   └── equity_*.py              # Equity Intel Suite 3-Stage Agents
├── core/                        # Domain Core, Primitives & Sandbox
│   ├── sandbox/                 # AST Guard & Memory Hygiene for Scratchpad
│   ├── news_digest/             # News Digest Core Engine & Models
│   ├── model_registry.py        # Centralized LLM Model Slots
│   ├── prompt_harness.py        # Skill & Prompt Markdown Loader
│   └── security.py              # PII Redaction
├── manager_sandbox/             # Hermes Sandbox Scaffolding (Skills, Memory, Audit)
├── tools/                       # Tool Adapters & MCP Bridge
│   ├── hermes_bridge/           # Trinity MCP Server & Sandbox Adapter
│   ├── macro/                   # Macro, FRED & Digest Store Adapters
│   ├── market/                  # Market, Terminal V2 & Equity Quant Tools
│   ├── portfolio/               # Portfolio Ledger & Broker Adapters
│   └── archivist/               # Vault V2 Markdown Writer & Search RAG
├── validators/                  # Guardrails & Output Verifiers
├── web/                         # React 19 + TypeScript + Tailwind Frontend
│   └── src/
│       ├── pages/               # ManagerAdvisor, DailyDigest, Portfolio, Macro, Equity
│       ├── components/          # Reusable UI Widgets & Cards
│       └── lib/                 # Frontend Utility & API Client
├── memories/                    # Obsidian PKM Vault (Layout V2)
└── tests/                       # Test Suites (Architecture, Unit, API, Contracts)
```

---

## 📄 License

This project is licensed under the GNU AGPLv3 License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2026 Money ReRoute. All rights reserved.
