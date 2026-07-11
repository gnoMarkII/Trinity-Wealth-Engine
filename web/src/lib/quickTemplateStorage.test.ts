import { beforeEach, describe, expect, it } from 'vitest'
import { loadQuickTemplates, saveQuickTemplates, type QuickTemplate } from './quickTemplateStorage'

const STORAGE_KEY = 'kanban-quick-templates'

describe('quickTemplateStorage', () => {
  beforeEach(() => {
    window.localStorage.clear()
  })

  it('ไม่มีข้อมูลใน storage → คืน default templates', () => {
    const templates = loadQuickTemplates()
    expect(templates.length).toBeGreaterThan(0)
    expect(templates[0]?.flow).toBe('manager')
  })

  it('JSON เสีย → คืน default ไม่ throw', () => {
    window.localStorage.setItem(STORAGE_KEY, '{not-json')
    expect(() => loadQuickTemplates()).not.toThrow()
    expect(loadQuickTemplates().length).toBeGreaterThan(0)
  })

  it('ข้อมูลไม่ใช่ array (โดนเขียนทับผิดรูป) → คืน default', () => {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify({ hacked: true }))
    const templates = loadQuickTemplates()
    expect(Array.isArray(templates)).toBe(true)
    expect(templates[0]?.label).toBeTruthy()
  })

  it('save แล้ว load กลับมาได้ค่าเดิม (roundtrip)', () => {
    const custom: QuickTemplate[] = [
      { label: 'ทดสอบ', instruction: 'คำสั่งทดสอบ', flow: 'manager', scope: 'both' },
    ]
    saveQuickTemplates(custom)
    expect(loadQuickTemplates()).toEqual(custom)
  })
})
