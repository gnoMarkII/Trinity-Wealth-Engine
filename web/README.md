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

## คำสั่งอื่นๆ

```bash
npm run build    # tsc -b (typecheck) + vite build → dist/
npm run lint     # oxlint
npm run preview  # serve dist/ ที่ build ไว้แล้ว
```

โปรเจกต์นี้ไม่มี test framework ฝั่ง frontend — verification หลักคือ TypeScript strict mode (`noUnusedLocals`/`noUnusedParameters` เปิดอยู่) และ oxlint

## Tech Stack

| อะไร | ใช้ทำอะไร |
|------|-----------|
| React 19 + React Router 7 | UI + client-side routing |
| TypeScript (strict) | Type safety |
| Vite 8 | Dev server + build |
| Tailwind CSS 3.4 | Styling (ดู `tailwind.config.js` สำหรับ terra accent palette) |
| oxlint | Linting |

## โครงสร้าง

```
web/src/
├── api/            # client.ts (fetch wrapper), types.ts (DTO ตรงกับ api/schemas.py)
├── auth/           # AuthContext — session state, บังคับ logout เมื่อ session หมดอายุ
├── components/
│   ├── ui/         # Modal, Button, TextInput, SegmentedControl — component กลางที่ใช้ซ้ำได้
│   └── kanban/      # Kanban board components (Card, Column, Modal, Drawer, ...)
├── lib/            # Helper functions ล้วนๆ ไม่ผูกกับ React (agentStatus, stance, flowTags, ...)
└── pages/          # Login, Kanban, Macro, Portfolio — 1 หน้าต่อ route
```

## หมายเหตุ

- Auth เป็นรหัสผ่านเดียวจาก `.env` (`WEBUI_PASSWORD`) — ไม่มีระบบ user/role เพราะออกแบบมาสำหรับผู้ใช้คนเดียว
- Animation ทั้งหมดใช้ CSS keyframes ล้วนๆ (ดู `index.css`) ไม่มี animation library — และเคารพ `prefers-reduced-motion: reduce`
