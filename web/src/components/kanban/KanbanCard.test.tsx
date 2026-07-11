import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import type { KanbanCardDTO } from '../../api/types'
import KanbanCard from './KanbanCard'

function makeCard(overrides: Partial<KanbanCardDTO> = {}): KanbanCardDTO {
  return {
    card_id: 'c1',
    title: 'วิเคราะห์พอร์ตวันนี้',
    prompt: null,
    column_name: 'backlog',
    job_id: null,
    flow: 'manager',
    scope: 'both',
    display_seq: 7,
    created_at: 1_700_000_000,
    updated_at: 1_700_000_000,
    ...overrides,
  }
}

function renderCard(props: Partial<Parameters<typeof KanbanCard>[0]> = {}) {
  const onClick = vi.fn()
  const onDelete = vi.fn()
  const onEdit = vi.fn()
  const onDispatch = vi.fn()
  render(
    <KanbanCard
      card={makeCard()}
      faded={false}
      removing={false}
      onDelete={onDelete}
      onClick={onClick}
      editable
      onEdit={onEdit}
      onDispatch={onDispatch}
      {...props}
    />
  )
  return { onClick, onDelete, onEdit, onDispatch }
}

describe('KanbanCard', () => {
  it('แสดงชื่อการ์ด, เลขลำดับ #AG-n และ tag ของ flow', () => {
    renderCard()
    expect(screen.getByText('วิเคราะห์พอร์ตวันนี้')).toBeInTheDocument()
    expect(screen.getByText('#AG-7')).toBeInTheDocument()
    expect(screen.getByText('#macro')).toBeInTheDocument()
  })

  it('เปิดด้วยคีย์บอร์ดได้ทั้ง Enter และ Space', async () => {
    const { onClick } = renderCard()
    const card = screen.getByRole('button', { name: /วิเคราะห์พอร์ตวันนี้/ })
    card.focus()
    await userEvent.keyboard('{Enter}')
    await userEvent.keyboard(' ')
    expect(onClick).toHaveBeenCalledTimes(2)
  })

  it('ปุ่มลบ/แก้ไข/Play ไม่ bubble ไปเปิดการ์ด (stopPropagation)', async () => {
    const { onClick, onDelete, onEdit, onDispatch } = renderCard()
    await userEvent.click(screen.getByRole('button', { name: 'ลบการ์ด' }))
    await userEvent.click(screen.getByRole('button', { name: 'แก้ไขการ์ด' }))
    await userEvent.click(screen.getByRole('button', { name: 'ส่งงานให้ Manager' }))
    expect(onDelete).toHaveBeenCalledOnce()
    expect(onEdit).toHaveBeenCalledOnce()
    expect(onDispatch).toHaveBeenCalledOnce()
    expect(onClick).not.toHaveBeenCalled()
  })

  it('ไม่ editable → ไม่มีปุ่มแก้ไข/Play แต่ยังลบได้', () => {
    renderCard({ editable: false })
    expect(screen.queryByRole('button', { name: 'แก้ไขการ์ด' })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'ส่งงานให้ Manager' })).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'ลบการ์ด' })).toBeInTheDocument()
  })

  it('workspace preview โชว์ชื่อ agent, จำนวน log และเวลาที่ผ่านไป', () => {
    renderCard({ workspacePreview: { node: 'researcher', logCount: 12, elapsedSeconds: 95 } })
    expect(screen.getByText(/Researcher • 12 log lines • 2m/)).toBeInTheDocument()
  })

  it('elapsed ต่ำกว่า 1 นาทีแสดงเป็นวินาที', () => {
    renderCard({ workspacePreview: { node: null, logCount: 0, elapsedSeconds: 42 } })
    expect(screen.getByText(/42s/)).toBeInTheDocument()
  })
})
