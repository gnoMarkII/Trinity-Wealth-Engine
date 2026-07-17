import { render, screen, fireEvent } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import PortfolioOverviewTab from './PortfolioOverviewTab'

describe('PortfolioOverviewTab', () => {
  const targets = [
    { bucket_id: 'CORE', name: 'Core Growth', target_percent: 60, color: null },
    { bucket_id: 'DIV', name: 'Dividend Shield', target_percent: 40, color: null },
  ]

  const summaries = [
    {
      bucket_id: 'CORE',
      name: 'Core Growth',
      target_percent: 60,
      actual_value_thb: 600000,
      actual_percent: 60,
      variance: 0,
      color: null,
    },
    {
      bucket_id: 'DIV',
      name: 'Dividend Shield',
      target_percent: 40,
      actual_value_thb: 400000,
      actual_percent: 40,
      variance: 0,
      color: null,
    },
  ]

  it('renders warning badge when warning is provided', () => {
    render(
      <PortfolioOverviewTab
        targets={targets}
        summaries={summaries}
        warning="ผลรวมสัดส่วนเป้าหมายเท่ากับ 100% (ปกติ)"
        onSelectBucket={vi.fn()}
      />
    )
    expect(screen.getByText(/ผลรวมสัดส่วนเป้าหมายเท่ากับ 100% \(ปกติ\)/i)).toBeInTheDocument()
  })

  it('renders concentric donut chart and strategy buckets table', () => {
    render(
      <PortfolioOverviewTab targets={targets} summaries={summaries} warning={null} onSelectBucket={vi.fn()} />
    )
    expect(screen.getByText('Allocation Rings (Target vs Actual)')).toBeInTheDocument()
    expect(screen.getByText('Strategy Buckets Breakdown')).toBeInTheDocument()
    expect(screen.getByText('Core Growth')).toBeInTheDocument()
    expect(screen.getByText('Dividend Shield')).toBeInTheDocument()
  })

  it('triggers onSelectBucket callback when bucket row in table is clicked', () => {
    const onSelectMock = vi.fn()
    render(
      <PortfolioOverviewTab targets={targets} summaries={summaries} warning={null} onSelectBucket={onSelectMock} />
    )

    const coreRowText = screen.getByText('Core Growth')
    const row = coreRowText.closest('tr')
    expect(row).not.toBeNull()
    if (row) fireEvent.click(row)

    expect(onSelectMock).toHaveBeenCalledWith('CORE')
  })
})
