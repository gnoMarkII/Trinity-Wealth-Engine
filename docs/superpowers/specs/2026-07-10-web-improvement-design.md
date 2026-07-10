# Design: ปรับปรุง Web UI (แก้บั๊ก + Refactor + Animation + Accessibility)

วันที่: 2026-07-10
สถานะ: อนุมัติแล้ว (แนวทาง "ครบทุกด้านแบบพอดี", animation แบบ CSS/Tailwind เท่านั้น — ไม่เพิ่ม dependency)

## เป้าหมาย

รีวิวและปรับปรุงโค้ด `web/src` ทั้งหมด (React 19 + Vite + Tailwind 3.4) ให้ถูกต้องตามหลักการ
ลดโค้ดซ้ำ เพิ่ม animation ที่ทำให้ใช้งานลื่นขึ้น และแก้ accessibility พื้นฐาน
โดยไม่เพิ่ม dependency ใหม่และไม่รื้อสถาปัตยกรรม

## ส่วนที่ 1 — แก้บั๊ก (correctness)

1. **คลาสแนวตั้งไม่ตรงกัน** — `web/src/pages/Macro.tsx:401` ใช้ `writing-mode-vertical`
   แต่ `index.css` ประกาศ `.writing-vertical` → แก้ฝั่ง Macro.tsx ให้ใช้ `writing-vertical`
2. **คลาส Tailwind v4 บนโปรเจกต์ v3.4** — `MacroReferenceDrawer.tsx` ใช้ `backdrop-blur-xs`
   (บรรทัด ~50) และ `shadow-xs` (บรรทัด ~134) ซึ่งไม่มีใน Tailwind 3.4 (no-op เงียบ)
   → เปลี่ยนเป็น `backdrop-blur-sm` และ `shadow-sm`
3. **Move การ์ดซ้ำซ้อน** — mapping สถานะ→คอลัมน์ (`done→done`, `error→backlog`,
   `awaiting_approval→approval`) ถูกเขียนซ้ำใน `Kanban.tsx#handleTerminalStatusChange` และ
   `KanbanDetailDrawer.tsx#handleStatusChange` และทั้งคู่ยิง `moveKanbanCard` พร้อมกันได้เมื่อ
   drawer เปิดการ์ดที่กำลังรัน อีกทั้งการเปิด drawer ของการ์ดที่ done แล้วจะ replay SSE จนจบ
   แล้วสั่ง move ไปคอลัมน์เดิมซ้ำทุกครั้ง
   → สร้าง helper กลาง `columnForStatus(status): string | null` (ใน `web/src/lib/`)
   → เพิ่ม guard ก่อน move: ถ้า `card.column_name` ตรงกับคอลัมน์เป้าหมายแล้ว ไม่ยิง API
   → ยังคงให้ drawer เป็นตัว move สำรอง (จำเป็นสำหรับกรณี reload หน้ากลาง job ที่
   background driver ใน state หายไป)
4. **Timer รั่ว** — `flashNotice` ใน Kanban.tsx ตั้ง `setTimeout` ใหม่โดยไม่เคลียร์ตัวเก่า
   (แจ้งเตือน 2 ครั้งติดกัน ตัวแรกจะลบข้อความที่สองก่อนเวลา) และไม่ cleanup ตอน unmount;
   `deleteCard` ก็มี setTimeout ที่ไม่ถูก track
   → เก็บ timer id ใน ref, เคลียร์ก่อนตั้งใหม่, เคลียร์ทั้งหมดใน cleanup ของ component
5. **Session หมดอายุกลางคันไม่พาไป Login** — auth ถูกเช็คครั้งเดียวตอน mount ใน AuthContext
   ถ้า API ตอบ 401 ระหว่างใช้งาน ผู้ใช้เห็นแค่ error text
   → api client เพิ่มกลไก callback `onUnauthorized` ที่ AuthProvider ลงทะเบียน:
   เมื่อ `request()` เจอ 401 (ยกเว้น endpoint login เอง) เรียก callback → สถานะ auth เป็น
   `unauthenticated` → Layout Navigate ไป `/login` ตามกลไกเดิมที่มีอยู่แล้ว
6. **Drawer width กระตุกตอนเปิดหน้า** — `KanbanDetailDrawer` โหลดความกว้างจาก localStorage
   ใน `useEffect` หลัง paint แรก → เปลี่ยนเป็น lazy initializer `useState(loadStoredWidth)`
   แล้วลบ useEffect นั้นทิ้ง

## ส่วนที่ 2 — Refactor ลดโค้ดซ้ำ

- **`web/src/components/ui/Modal.tsx`** (ใหม่): โครง modal กลาง — backdrop (คลิกปิด),
  ปิดด้วย Escape, `role="dialog"` + `aria-modal` + `aria-labelledby`, focus trap,
  คืน focus ให้ element เดิมตอนปิด, ใส่ `animate-modal-in` + backdrop `animate-fade-in`
  → `KanbanCardModal` และ `EditTemplateModal` เปลี่ยนมาใช้ Modal นี้ (เนื้อในฟอร์มคงเดิม)
- **`web/src/components/ui/SegmentedControl.tsx`** (ใหม่): ปุ่มกลุ่มเลือกค่า generic
  (`options: {key,label}[]`, `value`, `onChange`) แทนก้อน flow/scope ที่ซ้ำกันในสอง modal
- **`MacroReferenceDrawer.tsx`**: เปลี่ยนจาก `React.FC` + named export เป็น
  default function export ให้สไตล์ตรงกับ component อื่นทั้งโปรเจกต์
- **ลบไฟล์ที่ไม่ได้ใช้**: `web/src/assets/hero.png`, `web/src/assets/react.svg`,
  `web/src/assets/vite.svg` (grep แล้วไม่มีที่ไหน import)

## ส่วนที่ 3 — Animation (CSS/Tailwind เท่านั้น)

หลักการ: ใช้ keyframes ที่ประกาศไว้แล้วใน `index.css` แต่ยังไม่ถูกใช้ (`animate-modal-in`,
`animate-dropdown-in`, `animate-drawer-in`, `animate-fade-in`, `animate-bar-grow`,
`animate-shimmer`) — เขียนใหม่เฉพาะ `notice-out` ตัวเดียว

| จุด | สิ่งที่ทำ |
|---|---|
| Modal ทั้งสอง (ผ่าน Modal กลาง) | กล่อง `animate-modal-in`, backdrop `animate-fade-in` |
| AddCardDropdown เมนู | `animate-dropdown-in` |
| MacroReferenceDrawer | แผง `animate-drawer-in`, backdrop `animate-fade-in` |
| RegimeProbabilityChart | แท่งใช้ `animate-bar-grow` (scaleX, transform-origin left) ไล่ `animationDelay` ทีละแท่ง (~60ms) |
| PortfolioStanceBar | แท่งรวมใช้ `animate-bar-grow` |
| หน้า Macro ตอนโหลด | skeleton layout ด้วย `animate-shimmer` (แถบ banner + การ์ดซ้าย/ขวา) แทนข้อความ "กำลังโหลด..." |
| บอร์ด Kanban โหลดครั้งแรก | การ์ด stagger `card-in` ด้วย `animationDelay = index * 30ms` เพดาน ~240ms (เฉพาะ initial load ไม่ใช่ทุกครั้งที่ refresh) |
| Notice แจ้งเตือน (Kanban) | เพิ่ม keyframe `notice-out` — fade+slide ออกก่อนถูกลบ (สอง phase: visible → leaving) |

ข้อกำหนด: ทุก animation จบที่สถานะอ่านได้ปกติ และอยู่ใต้ media query
`prefers-reduced-motion: reduce` ที่มีอยู่แล้ว (ตัดเป็นจบทันที)

## ส่วนที่ 4 — Accessibility

- **KanbanCard**: การ์ดคลิกได้ต้องใช้คีย์บอร์ดได้ — `role="button"`, `tabIndex={0}`,
  กด Enter/Space = คลิก, มี `focus-visible` ring
- **Modal กลาง**: focus trap (Tab วนในกล่อง), autofocus ช่องแรก, คืน focus ตอนปิด
- **ฟอร์ม**: ผูก `<label htmlFor>` กับ `id` ของ input ทุกจุดในสอง modal และหน้า Login
- **AddCardDropdown**: ปุ่ม trigger ใส่ `aria-expanded` + `aria-haspopup="menu"`
- **MacroReferenceDrawer**: `role="dialog"` + `aria-modal="true"` + โฟกัสปุ่มปิดตอนเปิด
- **แท็บกรอง** (KanbanHeader, Macro stance filter): `aria-pressed` บนปุ่มที่ active

## สิ่งที่ตั้งใจไม่ทำ (out of scope)

- ไม่เพิ่ม dependency ใหม่ (ไม่มี Framer Motion, react-query, ฯลฯ)
- ไม่รื้อ state ของ Kanban.tsx เป็น reducer (17 useState ทำงานถูกต้องอยู่ — ความเสี่ยง
  regression สูงกว่าประโยชน์ในรอบนี้)
- ไม่แตะ backend / API contract ใดๆ
- ไม่ทำ drag-and-drop ให้ Kanban board

## การตรวจรับ (verification)

โปรเจกต์ web ไม่มี test framework — เกณฑ์ผ่านคือ:

1. `npm run build` ผ่าน (tsc strict + vite build)
2. `npm run lint` (oxlint) ไม่มี error ใหม่
3. เปิด dev server ทดสอบ flow จริง: login → สร้างการ์ด (ทั้ง quick template และ custom)
   → dispatch → เปิด drawer ดู terminal → ลบการ์ด → เปิด/ปิด MacroReferenceDrawer
   → สลับ filter ทุกแท็บ → ทดสอบคีย์บอร์ดล้วน (Tab/Enter/Escape) กับการ์ดและ modal
4. ตรวจว่าเปิด drawer ของการ์ดที่ done แล้ว **ไม่** ยิง PUT /api/kanban/move ซ้ำ
   (ดูจาก network tab)
