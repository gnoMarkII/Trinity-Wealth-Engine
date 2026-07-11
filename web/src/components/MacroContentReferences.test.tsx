import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import type { MacroReferenceDTO } from '../api/types'
import MacroContentReferences from './MacroContentReferences'

function makeReference(overrides: Partial<MacroReferenceDTO> = {}): MacroReferenceDTO {
  return {
    reference_id: 'ref-1',
    kind: 'news',
    title: 'Fed คงดอกเบี้ย',
    url: 'https://news.example/fed',
    publisher: 'Reuters',
    published_at: '2026-07-10',
    age_hours: 5,
    summary: 'สรุปข่าว',
    thumbnail_url: '',
    is_stale: false,
    related_observable_ids: [],
    ...overrides,
  }
}

describe('MacroContentReferences', () => {
  it('references ว่าง → ไม่ render', () => {
    const { container } = render(<MacroContentReferences references={[]} />)
    expect(container).toBeEmptyDOMElement()
  })

  it('ข่าวแสดงลิงก์ภายนอกพร้อม noopener noreferrer', () => {
    render(<MacroContentReferences references={[makeReference()]} />)
    const link = screen.getByRole('link', { name: 'เปิดแหล่งข้อมูล' })
    expect(link).toHaveAttribute('href', 'https://news.example/fed')
    expect(link).toHaveAttribute('rel', 'noopener noreferrer')
    expect(link).toHaveAttribute('target', '_blank')
  })

  it('YouTube ที่ URL ถูกต้อง → เปิด preview modal พร้อม iframe nocookie', async () => {
    render(
      <MacroContentReferences
        references={[
          makeReference({
            reference_id: 'yt-1',
            kind: 'youtube',
            title: 'สรุปตลาดรายสัปดาห์',
            url: 'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
          }),
        ]}
      />
    )
    await userEvent.click(screen.getByRole('button', { name: 'ดูตัวอย่างวิดีโอ' }))

    const dialog = screen.getByRole('dialog')
    expect(dialog).toBeInTheDocument()
    const iframe = dialog.querySelector('iframe')
    expect(iframe).toHaveAttribute('src', expect.stringContaining('youtube-nocookie.com/embed/dQw4w9WgXcQ'))
    expect(iframe).toHaveAttribute('sandbox')
  })

  it('YouTube ที่ URL ไม่ผ่านการตรวจ (โดเมนปลอม) → fallback เป็นลิงก์ภายนอก ไม่มีปุ่ม preview', () => {
    render(
      <MacroContentReferences
        references={[
          makeReference({ kind: 'youtube', url: 'https://evil.example/watch?v=dQw4w9WgXcQ' }),
        ]}
      />
    )
    expect(screen.queryByRole('button', { name: 'ดูตัวอย่างวิดีโอ' })).not.toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'เปิดแหล่งข้อมูล' })).toBeInTheDocument()
  })

  it('แสดงสูงสุด 6 รายการแรก', () => {
    const references = Array.from({ length: 9 }, (_, i) =>
      makeReference({ reference_id: `ref-${i}`, title: `ข่าวที่ ${i}` })
    )
    render(<MacroContentReferences references={references} />)
    expect(screen.getAllByRole('article')).toHaveLength(6)
  })

  it('รายการ stale มี badge Stale', () => {
    render(<MacroContentReferences references={[makeReference({ is_stale: true })]} />)
    expect(screen.getByText('Stale')).toBeInTheDocument()
  })
})
