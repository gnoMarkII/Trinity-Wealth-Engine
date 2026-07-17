import { render, screen, fireEvent } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import PortfolioHoldingsTab from './PortfolioHoldingsTab'

describe('PortfolioHoldingsTab', () => {
  const holdings = [
    {
      symbol: 'AAPL',
      asset_type: 'US_STOCK',
      units: 10,
      bucket_id: 'CORE',
      avg_cost_usd: 150,
      avg_cost_thb: 5250,
      current_price_usd: 180,
      current_price_thb: 6300,
      market_value_thb: 63000,
      unrealized_pnl_percent: 20,
      unrealized_pnl_value: 10500,
      market_cap_tier: 'Mega',
      yield_on_cost: 1.5,
      company_name: 'Apple Inc.',
      pe_ratio: 28.5,
      eps: 6.31,
      payout_ratio: 15,
      market_cap_value: 2800000000000,
      dividend_per_share: 0.96,
      dividend_yield: 0.53,
      accumulated_dividend_thb: null,
      fundamentals_updated_at: 1721123456,
    },
    {
      symbol: 'MSFT',
      asset_type: 'US_STOCK',
      units: 5,
      bucket_id: 'DIV',
      avg_cost_usd: 300,
      avg_cost_thb: 10500,
      current_price_usd: 400,
      current_price_thb: 14000,
      market_value_thb: 70000,
      unrealized_pnl_percent: 33.33,
      unrealized_pnl_value: 17500,
      market_cap_tier: 'Mega',
      yield_on_cost: 1.0,
      company_name: 'Microsoft Corp.',
      pe_ratio: 35.0,
      eps: 11.42,
      payout_ratio: 25,
      market_cap_value: 3000000000000,
      dividend_per_share: 3.0,
      dividend_yield: 0.75,
      accumulated_dividend_thb: null,
      fundamentals_updated_at: 1721123456,
    },
  ]

  it('renders all holdings with 2-line stacked grid details', () => {
    render(<PortfolioHoldingsTab holdings={holdings} selectedBucket={null} onClearBucketFilter={vi.fn()} />)
    expect(screen.getByText('AAPL')).toBeInTheDocument()
    expect(screen.getByText('Apple Inc.')).toBeInTheDocument()
    expect(screen.getByText('MSFT')).toBeInTheDocument()
    expect(screen.getByText('Microsoft Corp.')).toBeInTheDocument()
  })

  it('filters holdings by selectedBucket and shows clear banner', () => {
    const onClearMock = vi.fn()
    render(<PortfolioHoldingsTab holdings={holdings} selectedBucket="CORE" onClearBucketFilter={onClearMock} />)

    expect(screen.getByText('AAPL')).toBeInTheDocument()
    expect(screen.queryByText('MSFT')).not.toBeInTheDocument()

    const clearBtn = screen.getByRole('button', { name: /ล้างตัวกรอง/i })
    fireEvent.click(clearBtn)
    expect(onClearMock).toHaveBeenCalledTimes(1)
  })

  it('allows row selection and displays floating action bar placeholder', () => {
    render(<PortfolioHoldingsTab holdings={holdings} selectedBucket={null} onClearBucketFilter={vi.fn()} />)

    const checkboxes = screen.getAllByRole('checkbox')
    // First checkbox is select all, next 2 are rows
    expect(checkboxes.length).toBe(3)

    if (checkboxes[1]) fireEvent.click(checkboxes[1]) // Select AAPL
    expect(screen.getByText('1 Selected')).toBeInTheDocument()
    expect(screen.getByText(/เปลี่ยน Bucket/i)).toBeInTheDocument()
    expect(screen.getByText(/ลบที่เลือก/i)).toBeInTheDocument()

    const cancelBtn = screen.getByRole('button', { name: 'ยกเลิก' })
    fireEvent.click(cancelBtn)
    expect(screen.queryByText('1 Selected')).not.toBeInTheDocument()
  })

  it('expands trading journal row when journal count button is clicked', () => {
    const journalRows = [
      { timestamp: '2026-07-16 10:00:00', content: '**[BUY] AAPL** ซื้อเพิ่มที่แนวรับ 180 USD' },
    ]
    render(
      <PortfolioHoldingsTab
        holdings={holdings}
        selectedBucket={null}
        onClearBucketFilter={vi.fn()}
        journalRows={journalRows}
        onSuccessJournal={vi.fn()}
      />
    )

    const journalBtns = screen.getAllByTitle('ดู/เขียน Trading Journal สำหรับหุ้นนี้')
    expect(journalBtns.length).toBe(2)
    if (journalBtns[1]) fireEvent.click(journalBtns[1])

    expect(screen.getByText(/Trading Journal & Activity Log สำหรับ \[AAPL\]/i)).toBeInTheDocument()
    expect(screen.getByText(/ซื้อเพิ่มที่แนวรับ 180 USD/i)).toBeInTheDocument()
  })
})
