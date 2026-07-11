// @vitest-environment node — test ล้วนๆ ไม่แตะ DOM ข้าม jsdom ให้รันเร็วขึ้น
import { describe, expect, it } from 'vitest'
import { stanceCategory } from './stance'

describe('stanceCategory', () => {
  it('จัดหมวด overweight/underweight ไม่สนตัวพิมพ์', () => {
    expect(stanceCategory('OVERWEIGHT')).toBe('overweight')
    expect(stanceCategory('Overweight')).toBe('overweight')
    expect(stanceCategory('slightly underweight')).toBe('underweight')
    expect(stanceCategory('UNDERWEIGHT (tactical)')).toBe('underweight')
  })

  it('ค่าอื่นทั้งหมด fallback เป็น neutral', () => {
    expect(stanceCategory('Neutral')).toBe('neutral')
    expect(stanceCategory('Market Weight')).toBe('neutral')
    expect(stanceCategory('')).toBe('neutral')
  })
})
