import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import SegmentedControl from './SegmentedControl'

const OPTIONS = [
  { key: 'a', label: 'ตัวเลือก A' },
  { key: 'b', label: 'ตัวเลือก B' },
]

describe('SegmentedControl', () => {
  it('render ปุ่มครบทุก option และ mark ตัวที่เลือกด้วย aria-pressed', () => {
    render(<SegmentedControl options={OPTIONS} value="a" onChange={() => {}} />)
    expect(screen.getByRole('button', { name: 'ตัวเลือก A' })).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByRole('button', { name: 'ตัวเลือก B' })).toHaveAttribute('aria-pressed', 'false')
  })

  it('คลิกตัวเลือกอื่น → เรียก onChange ด้วย key ของตัวนั้น', async () => {
    const onChange = vi.fn()
    render(<SegmentedControl options={OPTIONS} value="a" onChange={onChange} />)
    await userEvent.click(screen.getByRole('button', { name: 'ตัวเลือก B' }))
    expect(onChange).toHaveBeenCalledWith('b')
  })

  it('ariaLabelledby ผูกชื่อกลุ่มผ่าน role=group', () => {
    render(
      <>
        <span id="group-label">ประเภทงาน</span>
        <SegmentedControl options={OPTIONS} value="a" onChange={() => {}} ariaLabelledby="group-label" />
      </>
    )
    expect(screen.getByRole('group')).toHaveAccessibleName('ประเภทงาน')
  })
})
