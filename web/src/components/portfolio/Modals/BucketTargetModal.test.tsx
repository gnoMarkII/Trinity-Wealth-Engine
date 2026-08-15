import { render, screen, fireEvent } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import BucketTargetModal from './BucketTargetModal'

vi.mock('../../../api/client', () => ({
  api: {
    upsertAllocationTargets: vi.fn(),
  },
  ApiError: class ApiError extends Error {},
}))

describe('BucketTargetModal', () => {
  beforeEach(() => {
    vi.resetAllMocks()
  })

  it('auto-assigns a unique color when adding a new bucket', async () => {
    const initialTargets = [
      { bucket_id: 'b1', name: 'Bucket 1', target_percent: 50, color: '#3B82F6' },
      { bucket_id: 'b2', name: 'Bucket 2', target_percent: 50, color: '#8B5CF6' },
    ]

    render(
      <BucketTargetModal
        portfolioId="default"
        initialTargets={initialTargets}
        onClose={vi.fn()}
        onSuccess={vi.fn()}
      />
    )

    // Click "+ เพิ่ม Bucket ใหม่"
    const addBtn = screen.getByText('+ เพิ่ม Bucket ใหม่')
    fireEvent.click(addBtn)

    // A 3rd color input should exist and its value should not be #3B82F6 or #8B5CF6
    const colorPickers = screen.getAllByTitle('เลือกสี') as HTMLInputElement[]
    expect(colorPickers).toHaveLength(3)

    const colors = colorPickers.map((p) => p.value.toUpperCase())
    expect(colors[2]).not.toBe('#3B82F6')
    expect(colors[2]).not.toBe('#8B5CF6')
  })

  it('randomizes row color when clicking row 🎲 button', async () => {
    const initialTargets = [
      { bucket_id: 'b1', name: 'Bucket 1', target_percent: 100, color: '#3B82F6' },
    ]

    render(
      <BucketTargetModal
        portfolioId="default"
        initialTargets={initialTargets}
        onClose={vi.fn()}
        onSuccess={vi.fn()}
      />
    )

    const initialPicker = screen.getByTitle('เลือกสี') as HTMLInputElement
    const initialColor = initialPicker.value

    const rowDiceBtn = screen.getByTitle('สุ่มสีใหม่สำหรับ Bucket นี้')
    fireEvent.click(rowDiceBtn)

    const updatedPicker = screen.getByTitle('เลือกสี') as HTMLInputElement
    expect(updatedPicker.value).not.toBe(initialColor)
  })

  it('randomizes all bucket colors when clicking 🎨 สุ่มสีทั้งหมด', async () => {
    const initialTargets = [
      { bucket_id: 'b1', name: 'Bucket 1', target_percent: 50, color: '#3B82F6' },
      { bucket_id: 'b2', name: 'Bucket 2', target_percent: 50, color: '#3B82F6' },
    ]

    render(
      <BucketTargetModal
        portfolioId="default"
        initialTargets={initialTargets}
        onClose={vi.fn()}
        onSuccess={vi.fn()}
      />
    )

    const randomizeAllBtn = screen.getByText('🎨 สุ่มสีทั้งหมด')
    fireEvent.click(randomizeAllBtn)

    const colorPickers = screen.getAllByTitle('เลือกสี') as HTMLInputElement[]
    expect(colorPickers[0]!.value.toUpperCase()).not.toBe(colorPickers[1]!.value.toUpperCase())
  })
})
