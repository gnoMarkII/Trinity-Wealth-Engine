import { describe, expect, it } from 'vitest'
import { nodeDisplayName } from './nodeDisplayNames'

describe('nodeDisplayName', () => {
  it('แปลงชื่อ node ในกราฟเป็นชื่อ UI', () => {
    expect(nodeDisplayName('supervisor')).toBe('Manager')
    expect(nodeDisplayName('macro_quant')).toBe('Macro Quant')
    expect(nodeDisplayName('prepare_archivist')).toBe('Archivist')
    expect(nodeDisplayName('archivist')).toBe('Archivist')
  })

  it('node ที่ไม่รู้จักคืนชื่อดิบ ไม่ crash', () => {
    expect(nodeDisplayName('brand_new_node')).toBe('brand_new_node')
  })

  it('null/undefined/ว่าง คืน "Agent"', () => {
    expect(nodeDisplayName(null)).toBe('Agent')
    expect(nodeDisplayName(undefined)).toBe('Agent')
    expect(nodeDisplayName('')).toBe('Agent')
  })
})
