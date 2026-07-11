import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import WarningPanel from './WarningPanel'

describe('WarningPanel', () => {
  it('ไม่มี warning → ไม่ render อะไรเลย', () => {
    const { container } = render(<WarningPanel warnings={[]} />)
    expect(container).toBeEmptyDOMElement()
  })

  it('โชว์ header พร้อมจำนวนใน mode ปกติ ซ่อนใน compact', () => {
    const warnings = [{ code: 'X', message: 'ทดสอบ' }]
    const { rerender } = render(<WarningPanel warnings={warnings} />)
    expect(screen.getByText(/Guardrail Warnings \(1\)/)).toBeInTheDocument()
    rerender(<WarningPanel warnings={warnings} compact />)
    expect(screen.queryByText(/Guardrail Warnings/)).not.toBeInTheDocument()
  })

  it('จัดสีตาม keyword ใน code: แดง (ขัดแย้ง) / เหลือง (stale) / กลาง (อื่นๆ)', () => {
    render(
      <WarningPanel
        warnings={[
          { code: 'DATA_CONTRADICTION', message: 'ขัดแย้ง' },
          { code: 'STALE_SNAPSHOT', message: 'ข้อมูลเก่า' },
          { code: 'SOMETHING_ELSE', message: 'ทั่วไป' },
          { code: null, message: 'ไม่มี code' },
        ]}
      />
    )
    expect(screen.getByText('ขัดแย้ง').closest('li')).toHaveClass('text-red-700')
    expect(screen.getByText('ข้อมูลเก่า').closest('li')).toHaveClass('text-amber-700')
    expect(screen.getByText('ทั่วไป').closest('li')).toHaveClass('text-zinc-600')
    // ไม่มี code ก็ยัง render ข้อความได้ไม่ crash
    expect(screen.getByText('ไม่มี code')).toBeInTheDocument()
  })
})
