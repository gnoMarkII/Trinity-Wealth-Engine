import { render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import ErrorBoundary from './ErrorBoundary'

function Bomb(): never {
  throw new Error('boom')
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe('ErrorBoundary', () => {
  it('render children ปกติเมื่อไม่มี error', () => {
    render(
      <ErrorBoundary>
        <p>ทำงานปกติ</p>
      </ErrorBoundary>
    )
    expect(screen.getByText('ทำงานปกติ')).toBeInTheDocument()
  })

  it('child throw ตอน render → แสดง fallback พร้อมปุ่มโหลดใหม่ แทน white screen', () => {
    // กัน React log error รกผล test — เจตนาให้ throw อยู่แล้ว
    vi.spyOn(console, 'error').mockImplementation(() => {})
    render(
      <ErrorBoundary>
        <Bomb />
      </ErrorBoundary>
    )
    expect(screen.getByText('เกิดข้อผิดพลาดที่ไม่คาดคิด')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'โหลดหน้าใหม่' })).toBeInTheDocument()
  })
})
