import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import Modal from './Modal'

function renderModal(onClose = vi.fn()) {
  const utils = render(
    <Modal titleId="test-title" onClose={onClose}>
      <h2 id="test-title">หัวข้อทดสอบ</h2>
      <button>ปุ่มแรก</button>
      <button>ปุ่มสุดท้าย</button>
    </Modal>
  )
  return { onClose, ...utils }
}

describe('Modal', () => {
  it('render เป็น dialog พร้อม aria-modal และชื่อจาก titleId', () => {
    renderModal()
    const dialog = screen.getByRole('dialog')
    expect(dialog).toHaveAttribute('aria-modal', 'true')
    expect(dialog).toHaveAccessibleName('หัวข้อทดสอบ')
  })

  it('โฟกัส element แรกที่โฟกัสได้เมื่อเปิด', () => {
    renderModal()
    expect(screen.getByRole('button', { name: 'ปุ่มแรก' })).toHaveFocus()
  })

  it('กด Escape → เรียก onClose', async () => {
    const { onClose } = renderModal()
    await userEvent.keyboard('{Escape}')
    expect(onClose).toHaveBeenCalledOnce()
  })

  it('คลิก backdrop → เรียก onClose แต่คลิกในกล่อง dialog ไม่ปิด', async () => {
    const { onClose, container } = renderModal()
    await userEvent.click(screen.getByRole('button', { name: 'ปุ่มแรก' }))
    expect(onClose).not.toHaveBeenCalled()
    const backdrop = container.querySelector('[aria-hidden="true"]')
    expect(backdrop).not.toBeNull()
    await userEvent.click(backdrop as HTMLElement)
    expect(onClose).toHaveBeenCalledOnce()
  })

  it('Tab จาก element สุดท้ายวนกลับ element แรก (focus trap)', () => {
    renderModal()
    const first = screen.getByRole('button', { name: 'ปุ่มแรก' })
    const last = screen.getByRole('button', { name: 'ปุ่มสุดท้าย' })
    last.focus()
    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Tab', bubbles: true }))
    expect(first).toHaveFocus()
  })

  it('Shift+Tab จาก element แรกวนไป element สุดท้าย', () => {
    renderModal()
    const first = screen.getByRole('button', { name: 'ปุ่มแรก' })
    const last = screen.getByRole('button', { name: 'ปุ่มสุดท้าย' })
    first.focus()
    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Tab', shiftKey: true, bubbles: true }))
    expect(last).toHaveFocus()
  })

  it('คืน focus ให้ element เดิมเมื่อ modal ปิด (unmount)', () => {
    const outside = document.createElement('button')
    document.body.appendChild(outside)
    outside.focus()
    const { unmount } = renderModal()
    expect(outside).not.toHaveFocus()
    unmount()
    expect(outside).toHaveFocus()
    outside.remove()
  })
})
