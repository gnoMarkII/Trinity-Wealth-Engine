import { describe, expect, it } from 'vitest'
import { groupIntoSteps, type LogLine } from './terminalSteps'

function line(node: string | null, content = 'x'): LogLine {
  return { node, content, role: 'reply', label: null }
}

describe('groupIntoSteps', () => {
  it('lines ว่าง → ไม่มี step', () => {
    expect(groupIntoSteps([])).toEqual([])
  })

  it('บรรทัด node เดียวกันติดกันรวมเป็น step เดียว', () => {
    const steps = groupIntoSteps([line('researcher', 'a'), line('researcher', 'b')])
    expect(steps).toHaveLength(1)
    expect(steps[0]?.node).toBe('researcher')
    expect(steps[0]?.messages.map((m) => m.content)).toEqual(['a', 'b'])
  })

  it('เปลี่ยน node เมื่อไหร่ขึ้น step ใหม่', () => {
    const steps = groupIntoSteps([line('supervisor'), line('researcher'), line('researcher')])
    expect(steps.map((s) => s.node)).toEqual(['supervisor', 'researcher'])
  })

  it('supervisor วนกลับมา node เดิม → เป็น step แยกใหม่ ไม่ยุบรวมกับรอบก่อน', () => {
    const steps = groupIntoSteps([line('supervisor'), line('researcher'), line('supervisor')])
    expect(steps.map((s) => s.node)).toEqual(['supervisor', 'researcher', 'supervisor'])
    // key ต้องไม่ชนกันแม้ node ซ้ำ (ใช้เป็น React key)
    const keys = steps.map((s) => s.key)
    expect(new Set(keys).size).toBe(keys.length)
  })

  it('node เป็น null ใช้ key ประเภท system', () => {
    const steps = groupIntoSteps([line(null)])
    expect(steps[0]?.key).toContain('system')
    expect(steps[0]?.node).toBeNull()
  })
})
