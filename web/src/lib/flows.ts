// Single source of truth ของ flow/scope ทั้งหมดฝั่ง UI — เดิม label/options ซ้ำกันใน
// KanbanCardModal, EditTemplateModal, AddCardDropdown, KanbanHeader, AgentStatusPanel
// เพิ่ม flow ใหม่ในอนาคตแก้ไฟล์นี้ที่เดียว ทุกจุดใช้ fallback เป็นค่า flow ดิบ ไม่มีทาง "หาย"

export interface FlowOption {
  key: string
  label: string
}

/** ชื่อ flow ที่อ่านง่ายสำหรับ UI (หมวดใน dropdown, filter tabs, sidebar) —
 * ไม่ annotate เป็น Record เพื่อให้เข้าถึงด้วย key ตรงๆ (FLOW_LABEL.manager) ได้
 * โดยไม่โดน noUncheckedIndexedAccess ตีเป็น undefined; lookup แบบ dynamic ใช้ flowLabel() */
export const FLOW_LABEL = {
  manager: 'Macro',
  news_youtube: 'News/YouTube',
} as const

/** #macro / #news มาจาก flow จริงที่เก็บไว้ตอนสร้างการ์ด (ไม่ใช่การเดา) — ดู Rev.2 Phase 0 */
export const FLOW_TAG: Record<string, string> = {
  manager: '#macro',
  news_youtube: '#news',
}

export const FLOW_OPTIONS: FlowOption[] = [
  { key: 'manager', label: FLOW_LABEL.manager },
  { key: 'news_youtube', label: FLOW_LABEL.news_youtube },
]

export const SCOPE_OPTIONS: FlowOption[] = [
  { key: 'news', label: 'ข่าวเท่านั้น' },
  { key: 'youtube', label: 'YouTube เท่านั้น' },
  { key: 'both', label: 'ทั้งคู่' },
]

export function flowLabel(flow: string): string {
  return (FLOW_LABEL as Record<string, string | undefined>)[flow] ?? flow
}
