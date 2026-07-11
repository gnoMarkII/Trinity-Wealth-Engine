# Invest Agents — Web UI

Frontend สำหรับ Web UI ของระบบ Investment Manager AI — React 19 + TypeScript (strict) + Vite + Tailwind CSS 3.4 คู่กับ FastAPI backend ที่ `api/` (ดู [README หลัก](../README.md) สำหรับภาพรวมทั้งระบบ)

## หน้าจอหลัก

- **`/kanban`** — Agent Kanban Board: สร้าง/แก้ไข/สั่งงานการ์ด, ดู log การทำงานของ agent แบบ real-time ผ่าน Server-Sent Events, อนุมัติรายการข่าว/คลิป YouTube ก่อน agent เจาะลึก (human-in-the-loop)
- **`/macro`** — Macro Strategy Report: Regime Probabilities, Cross-Asset Allocation, Pair Trades, Hedging Scenarios พร้อมแหล่งอ้างอิงข้อมูล
- **`/portfolio`** — พื้นที่สำหรับติดตามพอร์ตจริง (อยู่ระหว่างพัฒนา)

## รันแบบ Dev

ต้องมี backend รันอยู่ที่ `http://localhost:8000` ก่อน (`uv run uvicorn api.main:app --reload` จาก root ของ repo — ดู [README หลัก](../README.md#web-ui))

```bash
npm install
npm run dev
```

เปิด `http://localhost:5173` — Vite dev server proxy `/api` และ `/health` ไปที่ backend อัตโนมัติ (ดู `vite.config.ts`) เพื่อให้ auth cookie ทำงานแบบ same-origin โดยไม่ต้องตั้งค่า CORS

## รันแบบ Production

```bash
npm run build
```

แล้วรัน backend ตามปกติ — FastAPI จะเสิร์ฟ `dist/` เองพร้อม SPA fallback (deep link เช่น `/kanban` เปิดตรงได้) เปิดที่ `http://localhost:8000` ไม่ต้องมี Vite

## คำสั่งทั้งหมด

```bash
npm run dev        # Vite dev server + API proxy
npm run build      # tsc -b (typecheck) + vite build → dist/
npm run lint       # oxlint (react, typescript, jsx-a11y)
npm test           # Vitest ทั้งชุด (unit + component)
npm run test:watch # Vitest watch mode
npm run typecheck  # tsc -b อย่างเดียว
npm run preview    # serve dist/ ที่ build ไว้แล้ว
```

## การทดสอบ

ใช้ **Vitest + Testing Library (jsdom)** — test อยู่ข้างไฟล์ที่มันทดสอบ (`*.test.ts(x)`), setup กลางอยู่ที่ `src/test/setup.ts`

- **Unit**: ทุกโมดูลใน `lib/` (flows, stance, agentStatus, nodeDisplayNames, terminalSteps, youtube, quickTemplateStorage, macroReferences), `api/client` (mock fetch — 401 handling, error detail, encoding) และ hooks (usePageVisibility)
- **Component**: LiveTerminal (mock EventSource — SSE done/error/awaiting_approval, แยก network blip), Modal (focus trap/Escape/restore focus), ApprovalPanel, KanbanCard (keyboard + stopPropagation), AgentStatusPanel (poll หยุดตอน tab ซ่อน), MacroIndicatorPanel, MacroReferenceDrawer, MacroContentReferences, WarningPanel, RegimeProbabilityChart, PortfolioStanceBar, RequireAuth, SegmentedControl, ErrorBoundary

Quality gates อื่น: TypeScript strict (`strict` + `noUncheckedIndexedAccess`) และ oxlint พร้อม `jsx-a11y` plugin

## Tech Stack

| อะไร | ใช้ทำอะไร |
|------|-----------|
| React 19 + React Router 7 | UI + client-side routing (lazy load ต่อ route) |
| TypeScript (strict) | Type safety |
| Vite 8 | Dev server + build |
| Vitest + Testing Library | Unit / component tests |
| Tailwind CSS 3.4 | Styling — สีธีมเป็น semantic tokens (`panel`/`surface`/`surface-strong`/`edge`) ผูกกับ CSS vars ใน `index.css` |
| motion | Animation บนหน้า Landing (หน้าอื่นใช้ CSS keyframes ล้วน) |
| oxlint | Linting (react, typescript, jsx-a11y) |

## โครงสร้าง

```
web/src/
├── api/            # client.ts (fetch wrapper), types.ts (DTO ตรงกับ api/schemas.py)
├── auth/           # AuthContext (provider) + useAuth (context/hook แยกไฟล์เพื่อ Fast Refresh)
├── components/
│   ├── ui/         # Modal, Button, TextInput, SegmentedControl — component กลางที่ใช้ซ้ำได้
│   ├── kanban/     # Kanban board components (Card, Column, Modal, Drawer, ...)
│   └── landing/    # Landing page visuals (hero + candlestick canvas)
├── hooks/          # useFocusTrap (dialog ทุกแบบ), usePageVisibility
├── lib/            # Helper functions ล้วนๆ ไม่ผูกกับ React (flows, stance, terminalSteps, ...)
├── pages/          # Login, Kanban, Macro, Portfolio, Landing — 1 หน้าต่อ route (lazy)
└── test/           # setup.ts ของ Vitest
```

## หมายเหตุ

- Auth เป็นรหัสผ่านเดียวจาก `.env` (`WEBUI_PASSWORD`) — ไม่มีระบบ user/role เพราะออกแบบมาสำหรับผู้ใช้คนเดียว
- Animation เคารพ `prefers-reduced-motion: reduce` ทั้งแอป (CSS kill switch ใน `index.css` + `useReducedMotion` ของ motion บน Landing)
- ธีมสีทั้งหมดประกาศเป็น CSS vars ที่ `:root` ใน `index.css` แล้ว map เข้า Tailwind — จะเปลี่ยนธีมให้ override ตัวแปร ไม่ต้องไล่แก้ class รายตัว
