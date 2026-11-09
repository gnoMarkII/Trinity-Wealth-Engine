import { render, screen, fireEvent, within, waitFor } from '@testing-library/react'
import { describe, expect, it, vi, beforeEach } from 'vitest'
import PortfolioHoldingsTab from './PortfolioHoldingsTab'
import { api } from '../../api/client'

vi.mock('../../api/client', () => ({
  api: {
    getFundConstituents: vi.fn(),
    batchRemoveHoldings: vi.fn(),
    removeHolding: vi.fn(),
    getPositionReview: vi.fn(),
  },
}))

describe('PortfolioHoldingsTab', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })
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
    render(<PortfolioHoldingsTab portfolioId="default" holdings={holdings} selectedBucket={null} onClearBucketFilter={vi.fn()} />)
    expect(screen.getByText('AAPL')).toBeInTheDocument()
    expect(screen.getByText('Apple Inc.')).toBeInTheDocument()
    expect(screen.getByText('MSFT')).toBeInTheDocument()
    expect(screen.getByText('Microsoft Corp.')).toBeInTheDocument()
  })

  it('filters holdings by selectedBucket and shows clear banner', () => {
    const onClearMock = vi.fn()
    render(<PortfolioHoldingsTab portfolioId="default" holdings={holdings} selectedBucket="CORE" onClearBucketFilter={onClearMock} />)

    expect(screen.getByText('AAPL')).toBeInTheDocument()
    expect(screen.queryByText('MSFT')).not.toBeInTheDocument()

    const clearBtn = screen.getByRole('button', { name: /ล้างตัวกรอง/i })
    fireEvent.click(clearBtn)
    expect(onClearMock).toHaveBeenCalledTimes(1)
  })

  it('allows row selection and displays floating action bar placeholder', () => {
    render(<PortfolioHoldingsTab portfolioId="default" holdings={holdings} selectedBucket={null} onClearBucketFilter={vi.fn()} />)

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
        portfolioId="default"
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

  it('renders unassigned banner when holdings have missing bucket_id and handles filter toggle', () => {
    const unassignedHoldingsList = [
      ...holdings,
      {
        symbol: 'NVDA',
        asset_type: 'US_STOCK',
        units: 2,
        bucket_id: null,
        avg_cost_usd: 120,
        avg_cost_thb: 4200,
        current_price_usd: 130,
        current_price_thb: 4550,
        market_value_thb: 9100,
        unrealized_pnl_percent: 8.33,
        unrealized_pnl_value: 700,
        market_cap_tier: 'Mega',
        yield_on_cost: 0.1,
        company_name: 'NVIDIA Corp.',
        pe_ratio: 45.0,
        eps: 2.8,
        payout_ratio: 5,
        market_cap_value: 3000000000000,
        dividend_per_share: 0.16,
        dividend_yield: 0.12,
        accumulated_dividend_thb: null,
        fundamentals_updated_at: 1721123456,
      },
    ]

    render(
      <PortfolioHoldingsTab
        portfolioId="default"
        holdings={unassignedHoldingsList}
        selectedBucket={null}
        onClearBucketFilter={vi.fn()}
      />
    )

    expect(screen.getByText(/พบสินทรัพย์ที่ยังไม่ได้จัดหมวดหมู่ \(Unassigned\)/i)).toBeInTheDocument()
    expect(screen.getByText(/1 รายการ/i)).toBeInTheDocument()
    expect(screen.getByText(/จัดหมวดหมู่สินทรัพย์ทั่วไป \(1\)/i)).toBeInTheDocument()

    // Test filter toggle
    const toggleBtn = screen.getByRole('button', { name: /กรองเฉพาะ Unassigned/i })
    fireEvent.click(toggleBtn)

    expect(screen.getByText('NVDA')).toBeInTheDocument()
    expect(screen.queryByText('AAPL')).not.toBeInTheDocument()
    expect(screen.queryByText('MSFT')).not.toBeInTheDocument()
  })

  it('displays policy compatibility warning when holding asset_type is not in eligible_assets', () => {
    const policy = {
      purpose_buckets: [
        {
          bucket_id: 'CORE',
          name: 'Core Equities',
          eligible_assets: ['BOND', 'GOLD'], // AAPL is US_STOCK, so it should trigger ineligible warning
          target_percent: 60,
        },
      ],
    }

    render(
      <PortfolioHoldingsTab
        portfolioId="default"
        holdings={holdings}
        selectedBucket={null}
        onClearBucketFilter={vi.fn()}
        targets={[{ bucket_id: 'CORE', name: 'Core Equities', target_percent: 60, color: '#3B82F6' }]}
        allocationPolicy={policy}
      />
    )

    expect(screen.getByText(/⚠️ นอกเกณฑ์แนะนำ/i)).toBeInTheDocument()
  })

  it('shows PVD NAV and statement date while excluding synthetic returns and mutation actions', () => {
    const pvdHolding = {
      ...holdings[0]!,
      symbol: 'SINSA_FI',
      asset_type: 'PVD',
      bucket_id: 'retirement_pvd',
      units: 123.4567,
      avg_cost_thb: 8.5,
      current_price_thb: 10.25,
      market_value_thb: 1265.43,
      unrealized_pnl_percent: 23.38,
      unrealized_pnl_value: 240.12,
      pvd_valuation: {
        payload_version: 1,
        fund_code: 'SINSA_FI',
        sub_fund_code: 'SINSA_FI',
        units: '123.4567',
        statement_nav: '10.25',
        statement_market_value: '1265.43',
        reconciliation_adjustment: '0.00',
        cost_basis: '1025.31',
        as_of_date: '2026-06-30',
        statement_fingerprint: 'synthetic',
        archive_id: 'synthetic',
      },
    }
    render(
      <PortfolioHoldingsTab
        portfolioId="pvd-test"
        holdings={[holdings[0]!, pvdHolding]}
        selectedBucket={null}
        onClearBucketFilter={vi.fn()}
        onSuccess={vi.fn()}
        onViewTransactions={vi.fn()}
      />
    )

    expect(screen.getByText('Statement มียอดเงินต้นรวม จึงยังแสดงต้นทุนและผลตอบแทนจริงรายกองไม่ได้')).toBeInTheDocument()
    expect(screen.getByTitle('ไม่มีต้นทุนซื้อจริงรายกองจาก Statement')).toHaveTextContent('—')
    expect(screen.getByText('NAV ตาม Statement · 2026-06-30')).toBeInTheDocument()
    expect(screen.getByText('ดูผลประโยชน์รวมตาม Statement')).toBeInTheDocument()
    expect(screen.getByRole('checkbox', { name: 'Select SINSA_FI' })).not.toBeDisabled()
    expect(screen.getByRole('button', { name: 'Statement' })).toHaveAttribute('title', 'ดูประวัติ Statement ของ SINSA_FI')
    expect(screen.getAllByTitle('แก้ไข / ปรับปรุงข้อมูล')).toHaveLength(1)
    expect(screen.getAllByTitle('ลบ Holding ออกจากพอร์ต')).toHaveLength(1)

    fireEvent.click(screen.getByRole('checkbox', { name: 'Select all holdings' }))
    expect(screen.getByText('2 Selected')).toBeInTheDocument()
    expect(screen.getByRole('checkbox', { name: 'Select SINSA_FI' })).toBeChecked()
  })

  it('includes unassigned PVD in the Unassigned batch shortcut', () => {
    const pvdUnassigned = { ...holdings[0]!, symbol: 'SINSA_FI', asset_type: 'PVD', bucket_id: null }
    const regularUnassigned = { ...holdings[0]!, bucket_id: null }
    render(
      <PortfolioHoldingsTab
        portfolioId="mixed-test"
        holdings={[regularUnassigned, pvdUnassigned]}
        selectedBucket={null}
        onClearBucketFilter={vi.fn()}
        onSuccess={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole('button', { name: /จัดหมวดหมู่สินทรัพย์ทั่วไป \(2\)/i }))
    const dialog = screen.getByRole('dialog')
    expect(within(dialog).getByText(/AAPL/)).toBeInTheDocument()
    expect(within(dialog).getByText(/SINSA_FI/)).toBeInTheDocument()
    expect(screen.getByRole('checkbox', { name: 'Select SINSA_FI' })).not.toBeDisabled()
  })

  it('closes a batch modal and clears selection when a selected holding is removed', () => {
    const onSuccess = vi.fn()
    const regularHolding = { ...holdings[0]!, symbol: 'ASSET_TO_REMOVE' }
    const { rerender } = render(
      <PortfolioHoldingsTab
        portfolioId="scope-test"
        holdings={[regularHolding]}
        selectedBucket={null}
        onClearBucketFilter={vi.fn()}
        onSuccess={onSuccess}
      />
    )

    fireEvent.click(screen.getByRole('checkbox', { name: 'Select ASSET_TO_REMOVE' }))
    fireEvent.click(screen.getByRole('button', { name: /เปลี่ยน Bucket \(1\)/i }))
    expect(screen.getByRole('dialog')).toBeInTheDocument()

    rerender(
      <PortfolioHoldingsTab
        portfolioId="scope-test"
        holdings={[]}
        selectedBucket={null}
        onClearBucketFilter={vi.fn()}
        onSuccess={onSuccess}
      />
    )

    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
    expect(screen.getByRole('status')).toHaveTextContent(/ยกเลิกการเลือกแล้ว/)
    expect(screen.queryByText('1 Selected')).not.toBeInTheDocument()
  })

  it('renders skeleton shimmer when loading is true and holdings are empty', () => {
    const { container } = render(
      <PortfolioHoldingsTab
        portfolioId="loading-test"
        holdings={[]}
        selectedBucket={null}
        onClearBucketFilter={vi.fn()}
        loading={true}
      />
    )
    expect(container.querySelectorAll('.animate-pulse').length).toBeGreaterThanOrEqual(1)
    expect(screen.queryByText(/ยังไม่มีสินทรัพย์ในพอร์ตการลงทุน/i)).not.toBeInTheDocument()
  })

  it('renders Look-Through button for Fund holdings and expands constituent drawer on click', async () => {
    const fundHolding = {
      ...holdings[0]!,
      symbol: 'SCBS&P500E',
      asset_type: 'Fund',
      company_name: 'SCB S&P 500 Index Fund',
      market_value_thb: 200000,
    }
    const mockConstituents = {
      fund_id: 'scbsp500e',
      symbol: 'SCBS&P500E',
      as_of_date: '2026-09-30',
      master_fund_name: 'iShares Core S&P 500 ETF',
      master_fund_ticker: 'IVV',
      top_holdings: [
        { ticker: 'AAPL', name: 'Apple Inc.', percent: 7.2 },
        { ticker: 'MSFT', name: 'Microsoft Corp.', percent: 6.5 },
      ],
      sector_breakdown: [
        { sector_name_en: 'Information Technology', sector_name_th: 'เทคโนโลยี', percent: 32.5 },
      ],
      asset_allocation: { Equity: 99.1, Cash: 0.9 },
    }
    vi.mocked(api.getFundConstituents).mockResolvedValue(mockConstituents)

    render(
      <PortfolioHoldingsTab
        portfolioId="default"
        holdings={[fundHolding]}
        selectedBucket={null}
        onClearBucketFilter={vi.fn()}
      />
    )

    // Button should be visible
    const lookThroughBtn = screen.getByRole('button', { name: /ส่องไส้ในกองทุน/i })
    expect(lookThroughBtn).toBeInTheDocument()

    // Click to open drawer
    fireEvent.click(lookThroughBtn)

    expect(api.getFundConstituents).toHaveBeenCalledWith('SCBS&P500E')

    await waitFor(() => {
      expect(screen.getByText(/Glass-Box Look-Through: ไส้ในกองทุน \[SCBS&P500E\]/i)).toBeInTheDocument()
    })

    expect(screen.getByText('iShares Core S&P 500 ETF')).toBeInTheDocument()
    expect(screen.getByText('Apple Inc.')).toBeInTheDocument()
    expect(screen.getByText('7.20%')).toBeInTheDocument()

    // Close button
    const closeBtn = screen.getByRole('button', { name: /✕ ปิด/i })
    fireEvent.click(closeBtn)

    expect(screen.queryByText(/Glass-Box Look-Through: ไส้ในกองทุน \[SCBS&P500E\]/i)).not.toBeInTheDocument()
  })

  it('opens look-through drawer via quick-access pill bar above table and closes via Escape key', async () => {
    const fundHolding = {
      ...holdings[0]!,
      symbol: 'SCBS&P500E',
      asset_type: 'Fund',
      company_name: 'SCB S&P 500 Index Fund',
      market_value_thb: 200000,
    }

    const mockConstituents = {
      fund_id: 'scbsp500e',
      symbol: 'SCBS&P500E',
      as_of_date: '2026-09-30',
      master_fund_name: 'iShares Core S&P 500 ETF',
      master_fund_ticker: 'IVV',
      top_holdings: [
        { ticker: 'AAPL', name: 'Apple Inc.', percent: 7.2 },
        { ticker: 'MSFT', name: 'Microsoft Corp.', percent: 6.8 },
      ],
      sector_breakdown: [
        { sector_name_en: 'Information Technology', sector_name_th: 'เทคโนโลยี', percent: 31.5 },
      ],
      asset_allocation: { Equity: 99.1, Cash: 0.9 },
    }
    vi.mocked(api.getFundConstituents).mockResolvedValue(mockConstituents)

    render(
      <PortfolioHoldingsTab
        portfolioId="default"
        holdings={[fundHolding]}
        selectedBucket={null}
        onClearBucketFilter={vi.fn()}
      />
    )

    // Quick-Access pill bar button
    const pillBtn = screen.getByRole('button', { name: /เปิดลิ้นชักส่องไส้ใน SCBS&P500E/i })
    expect(pillBtn).toBeInTheDocument()

    // Click pill to open drawer
    fireEvent.click(pillBtn)

    await waitFor(() => {
      expect(screen.getByRole('dialog')).toBeInTheDocument()
      expect(screen.getByText(/Glass-Box Look-Through: ไส้ในกองทุน \[SCBS&P500E\]/i)).toBeInTheDocument()
    })

    // Press Escape to dismiss drawer via useFocusTrap
    fireEvent.keyDown(window, { key: 'Escape' })

    await waitFor(() => {
      expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
    })
  })

  it('opens PositionReviewModal when review action button is clicked for non-PVD holding', async () => {
    vi.mocked(api.getPositionReview).mockResolvedValue({
      ticker: 'AAPL',
      company_name: 'Apple Inc.',
      current_price: 180.0,
      as_of_date: '2026-08-03',
      action_stance: 'HOLD',
      action_stance_reason: 'Thesis remains intact.',
      holding_summary: {
        is_held: true,
        units: 10,
        avg_cost: 150.0,
        unrealized_pnl_pct: 20.0,
        current_weight_pct: 47.37,
        bucket_id: 'CORE',
        entry_date: '2025-05-01',
        added_direction: 'single_entry',
      },
      decompositions: [],
      then_vs_now: {
        is_held: true,
        period_label: 'since_entry',
        then_date: '2025-05-01',
        now_date: '2026-08-03',
        then_price: 150.0,
        now_price: 180.0,
        then_pe: 25.0,
        now_pe: 28.5,
        then_growth_pct: 10.0,
        now_growth_pct: 12.0,
        what_was_true_then: 'Core compounder.',
        what_is_true_now: 'Margin expansion steady.',
        thesis_reflection_question: 'Would you initiate AAPL today?',
      },
      triggers: [],
      honest_gaps: [],
    })

    render(
      <PortfolioHoldingsTab
        portfolioId="default"
        holdings={holdings}
        selectedBucket={null}
        onClearBucketFilter={vi.fn()}
      />
    )

    const reviewBtn = screen.getByTitle('ดู Position Review ของ AAPL')
    fireEvent.click(reviewBtn)

    await waitFor(() => {
      expect(api.getPositionReview).toHaveBeenCalledWith('AAPL', 'US', 'default')
      expect(screen.getByText('AAPL Position Review')).toBeInTheDocument()
    })
  })
})


