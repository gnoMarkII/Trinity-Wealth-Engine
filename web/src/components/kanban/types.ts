// Local UI filter/view-state definitions สำหรับ Kanban board — ไม่ใช่ DTO ที่มาจาก backend
export type StatusFilter = 'active' | 'all' | 'backlog' | 'done'
export type FlowFilter = 'all' | 'manager' | 'news_youtube' | 'news_funnel' | 'youtube_pitch'

export interface ColumnDef {
  key: string
  label: string
  icon: string
}

export interface WorkspacePreview {
  node: string | null
  logCount: number
  elapsedSeconds: number
}
