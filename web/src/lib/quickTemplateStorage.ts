export interface QuickTemplate {
  label: string
  instruction: string
  flow: string
  scope: string
}

const STORAGE_KEY = 'kanban-quick-templates'

const DEFAULT_TEMPLATES: QuickTemplate[] = [
  { label: 'วิเคราะห์เศรษฐกิจมหภาค', instruction: 'วิเคราะห์เศรษฐกิจมหภาคและจัดสรรพอร์ตประจำวัน', flow: 'manager', scope: 'both' },
  { label: 'ดึงข่าวล่าสุด', instruction: 'ดึงข่าวเศรษฐกิจและการลงทุนล่าสุด สรุปประเด็นสำคัญ', flow: 'news_youtube', scope: 'news' },
  { label: 'ดึงสรุปคลิป YouTube', instruction: 'ดึงสรุปคลิป YouTube ช่องการลงทุนที่ติดตามไว้ล่าสุด', flow: 'news_youtube', scope: 'youtube' },
]

export function loadQuickTemplates(): QuickTemplate[] {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)
    if (!raw) return DEFAULT_TEMPLATES
    const parsed = JSON.parse(raw)
    return Array.isArray(parsed) ? parsed : DEFAULT_TEMPLATES
  } catch {
    return DEFAULT_TEMPLATES
  }
}

export function saveQuickTemplates(templates: QuickTemplate[]): void {
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(templates))
}
