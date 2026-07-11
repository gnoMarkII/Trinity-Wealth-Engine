// @vitest-environment node — test ล้วนๆ ไม่แตะ DOM ข้าม jsdom ให้รันเร็วขึ้น
import { describe, expect, it } from 'vitest'
import { columnForStatus } from './agentStatus'

describe('columnForStatus', () => {
  it('จับคู่สถานะปลายทางไปคอลัมน์ที่ถูกต้อง', () => {
    expect(columnForStatus('done')).toBe('done')
    expect(columnForStatus('error')).toBe('backlog')
    expect(columnForStatus('awaiting_approval')).toBe('approval')
  })

  it('สถานะระหว่างทาง (idle/streaming) ไม่ย้ายคอลัมน์', () => {
    expect(columnForStatus('idle')).toBeNull()
    expect(columnForStatus('streaming')).toBeNull()
  })
})
