// @vitest-environment node — test ล้วนๆ ไม่แตะ DOM ข้าม jsdom ให้รันเร็วขึ้น
import { describe, expect, it } from 'vitest'
import { FLOW_LABEL, FLOW_OPTIONS, FLOW_TAG, SCOPE_OPTIONS, flowLabel } from './flows'

describe('flowLabel', () => {
  it('คืน label ที่อ่านง่ายสำหรับ flow ที่รู้จัก', () => {
    expect(flowLabel('manager')).toBe('Macro')
    expect(flowLabel('news_youtube')).toBe('News/YouTube')
  })

  it('flow ที่ไม่รู้จัก fallback เป็นค่าดิบ — ไม่มีทาง "หาย" จาก UI', () => {
    expect(flowLabel('future_flow')).toBe('future_flow')
    expect(flowLabel('')).toBe('')
  })
})

describe('ความสอดคล้องของค่าคงที่', () => {
  it('FLOW_OPTIONS ใช้ label เดียวกับ FLOW_LABEL ทุกตัว', () => {
    for (const opt of FLOW_OPTIONS) {
      expect(opt.label).toBe(flowLabel(opt.key))
    }
  })

  it('ทุก flow ใน FLOW_LABEL มี tag คู่กันใน FLOW_TAG', () => {
    expect(Object.keys(FLOW_TAG).sort()).toEqual(Object.keys(FLOW_LABEL).sort())
  })

  it('SCOPE_OPTIONS ครบ news/youtube/both ตามที่ backend รองรับ', () => {
    expect(SCOPE_OPTIONS.map((o) => o.key).sort()).toEqual(['both', 'news', 'youtube'])
  })
})
