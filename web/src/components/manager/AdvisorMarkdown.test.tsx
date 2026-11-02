import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import AdvisorMarkdown, { splitMarkdownAndTables } from './AdvisorMarkdown'

describe('AdvisorMarkdown Component', () => {
  it('correctly splits markdown text and markdown tables', () => {
    const content = `นี่คือภาพรวมพอร์ตของคุณ:

| สัญลักษณ์ | มูลค่า (THB) | กำไร/ขาดทุน (%) |
| :--- | :--- | :--- |
| **FTNT** | 13,065.73 | +71.86% |
| **NFLX** | 31,867.14 | -20.70% |

ข้อสังเกตจาก AI:
* พอร์ตโดยรวมแข็งแกร่ง`

    const segments = splitMarkdownAndTables(content)
    expect(segments.length).toBe(3)
    expect(segments[0]?.type).toBe('markdown')
    expect(segments[1]?.type).toBe('table')
    expect(segments[2]?.type).toBe('markdown')

    if (segments[1]?.type === 'table') {
      expect(segments[1].data.headers).toEqual(['สัญลักษณ์', 'มูลค่า (THB)', 'กำไร/ขาดทุน (%)'])
      expect(segments[1].data.rows.length).toBe(2)
      expect(segments[1].data.rows[0]).toEqual(['**FTNT**', '13,065.73', '+71.86%'])
    }
  })

  it('renders table elements with green and red P&L badges and ticker chip', () => {
    const content = `| สัญลักษณ์ | มูลค่า | กำไร/ขาดทุน |
| :--- | :--- | :--- |
| **FTNT** | 13,065.73 | +71.86% |
| **NFLX** | 31,867.14 | -20.70% |`

    render(<AdvisorMarkdown content={content} />)

    // Check table headers
    expect(screen.getByText('สัญลักษณ์')).toBeInTheDocument()
    expect(screen.getByText('มูลค่า')).toBeInTheDocument()
    expect(screen.getByText('กำไร/ขาดทุน')).toBeInTheDocument()

    // Check styled ticker chip
    expect(screen.getByText('$FTNT')).toBeInTheDocument()
    expect(screen.getByText('$NFLX')).toBeInTheDocument()

    // Check P&L badges
    const positiveBadge = screen.getByText('+71.86%')
    expect(positiveBadge).toBeInTheDocument()
    expect(positiveBadge.className).toContain('text-emerald-700')

    const negativeBadge = screen.getByText('-20.70%')
    expect(negativeBadge).toBeInTheDocument()
    expect(negativeBadge.className).toContain('text-rose-700')
  })

  it('renders streaming caret when isStreaming is true', () => {
    const { container } = render(<AdvisorMarkdown content="กำลังวิเคราะห์..." isStreaming />)
    expect(screen.getByText('กำลังวิเคราะห์...')).toBeInTheDocument()
    expect(container.querySelector('.animate-stream-caret')).not.toBeNull()
  })
})

