export type TerminalStatus = 'idle' | 'streaming' | 'done' | 'error' | 'awaiting_approval'

// mapping สถานะ terminal → คอลัมน์ kanban ปลายทาง — เดิมเขียนซ้ำใน Kanban.tsx และ
// KanbanDetailDrawer.tsx แยกกัน ทำให้ทั้งคู่ยิง moveKanbanCard พร้อมกันได้เมื่อ drawer เปิด
// การ์ดที่กำลังรันอยู่ (ดู design spec ส่วนที่ 1 ข้อ 3)
export function columnForStatus(status: TerminalStatus): string | null {
  if (status === 'done') return 'done'
  if (status === 'error') return 'backlog'
  if (status === 'awaiting_approval') return 'approval'
  return null
}
