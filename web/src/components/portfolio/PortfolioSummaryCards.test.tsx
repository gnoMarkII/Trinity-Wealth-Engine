import { render, screen, fireEvent } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import PortfolioSummaryCards, { isPriceRefreshSuccess } from './PortfolioSummaryCards'
import type { PvdValuationContext } from '../../lib/pvdPresentation'

describe('PortfolioSummaryCards', () => {
  it('renders loading skeletons when loading=true or summary=null', () => {
    const { container } = render(<PortfolioSummaryCards summary={null} lastUpdated={null} loading={true} />)
    expect(container.querySelectorAll('.animate-pulse').length).toBeGreaterThan(0)
  })

  it('renders summary values correctly in THB', () => {
    const summary = {
      total_value_thb: 1500000.5,
      total_cost_basis_thb: 1250000.25,
      total_unrealized_profit: 250000.25,
      passive_income_ytd: 45000.0,
    }
    render(<PortfolioSummaryCards summary={summary} lastUpdated="2026-07-16T10:00:00Z" />)

    expect(screen.getByText(/Total Portfolio NAV/i)).toBeInTheDocument()
    expect(screen.getByText(/Unrealized Profit\/Loss/i)).toBeInTheDocument()
    expect(screen.getByText(/Passive Income YTD/i)).toBeInTheDocument()
  })

  it('triggers onRefreshPrices callback when refresh button clicked', () => {
    const summary = {
      total_value_thb: 100,
      total_cost_basis_thb: 90,
      total_unrealized_profit: 10,
      passive_income_ytd: 5,
    }
    const onRefreshMock = vi.fn()
    render(<PortfolioSummaryCards summary={summary} lastUpdated={null} onRefreshPrices={onRefreshMock} />)

    const btn = screen.getByRole('button', { name: /อัปเดตราคาตลาด \(Refresh\)/i })
    fireEvent.click(btn)
    expect(onRefreshMock).toHaveBeenCalledTimes(1)
  })

  it('disables refresh button when refreshingPrices=true', () => {
    const summary = {
      total_value_thb: 100,
      total_cost_basis_thb: 90,
      total_unrealized_profit: 10,
      passive_income_ytd: 5,
    }
    render(<PortfolioSummaryCards summary={summary} lastUpdated={null} refreshingPrices={true} />)

    const btn = screen.getByRole('button', { name: /กำลังอัปเดตราคา\.\.\./i })
    expect(btn).toBeDisabled()
  })

  it('renders summary badge for priceRefreshInfo and toggles details on click', () => {
    const summary = {
      total_value_thb: 100,
      total_cost_basis_thb: 90,
      total_unrealized_profit: 10,
      passive_income_ytd: 5,
    }
    const priceRefreshInfo = { PG: 'ok', AAPL: 'ok', TSLA: 'ok' }
    render(<PortfolioSummaryCards summary={summary} lastUpdated={null} priceRefreshInfo={priceRefreshInfo} />)

    expect(screen.getByText(/อัปเดตราคาสำเร็จ \(3 รายการ\)/i)).toBeInTheDocument()

    // Details should be hidden by default
    expect(screen.queryByText(/PG:/i)).not.toBeInTheDocument()

    // Click toggle button to show details
    const toggleBtn = screen.getByRole('button', { name: /รายละเอียด/i })
    fireEvent.click(toggleBtn)

    expect(screen.getByText(/PG:/i)).toBeInTheDocument()
    expect(screen.getByText(/AAPL:/i)).toBeInTheDocument()
    expect(screen.getByText(/TSLA:/i)).toBeInTheDocument()
  })

  it('correctly handles real-world yfinance price and fx formats as success', () => {
    const summary = {
      total_value_thb: 47073.59,
      total_cost_basis_thb: 34434.24,
      total_unrealized_profit: 12639.33,
      passive_income_ytd: 365.24,
    }
    const realYfInfo = {
      USDTHB: '32.9100 (live)',
      PG: '146.92 USD',
      UNH: '400.94 USD',
      FTNT: '156.36 USD',
    }
    render(<PortfolioSummaryCards summary={summary} lastUpdated="2026-09-04T08:14:56" priceRefreshInfo={realYfInfo} />)

    // Should display full success badge, not 0/4 failure
    expect(screen.getByText(/อัปเดตราคาสำเร็จ \(4 รายการ\)/i)).toBeInTheDocument()
    expect(screen.queryByText(/สำเร็จ 0\/4 รายการ/i)).not.toBeInTheDocument()

    // Failed list should not be shown
    expect(screen.queryByText(/PG:/i)).not.toBeInTheDocument()

    // Toggle details to see all items in green
    fireEvent.click(screen.getByRole('button', { name: /รายละเอียด/i }))
    expect(screen.getByText(/USDTHB:/i)).toBeInTheDocument()
    expect(screen.getByText('32.9100 (live)')).toHaveClass('text-emerald-600')
    expect(screen.getByText('146.92 USD')).toHaveClass('text-emerald-600')
  })

  it('correctly reports partial failure when some items fail', () => {
    const summary = {
      total_value_thb: 100,
      total_cost_basis_thb: 90,
      total_unrealized_profit: 10,
      passive_income_ytd: 5,
    }
    const mixedInfo = {
      USDTHB: '32.9100 (live)',
      PG: '146.92 USD',
      BAD: 'fetch failed (kept previous)',
    }
    render(<PortfolioSummaryCards summary={summary} lastUpdated={null} priceRefreshInfo={mixedInfo} />)

    expect(screen.getByText(/สำเร็จ 2\/3 รายการ/i)).toBeInTheDocument()
    // By default when not all ok and showDetails is false, only failed entries are listed
    expect(screen.getByText(/BAD:/i)).toBeInTheDocument()
    expect(screen.getByText('fetch failed (kept previous)')).toBeInTheDocument()
    expect(screen.queryByText(/PG:/i)).not.toBeInTheDocument()
  })

  it('uses statement benefit language and a Statement action for a PVD-only portfolio', () => {
    const onViewPvdStatements = vi.fn()
    const pvdContext: PvdValuationContext = {
      hasPvd: true, hasMarketHoldings: false, hasNonPvdValue: false,
      dates: ['2026-06-30'], hasUnknownDate: false, hasUnknownValuation: false,
      latestRecordedStatementDate: '2026-06-30',
    }
    const holdings = [{ symbol: 'SINSA_FI', asset_type: 'PVD' }] as any
    render(
      <PortfolioSummaryCards
        summary={{ total_value_thb: 1200, total_cost_basis_thb: 1000, total_unrealized_profit: 200, passive_income_ytd: 0 }}
        lastUpdated="2026-10-14T16:34:48Z"
        pvdContext={pvdContext}
        holdings={holdings}
        priceRefreshInfo={{ SINSA_FI: 'Statement NAV 12.00 THB (2026-06-30)' }}
        onViewPvdStatements={onViewPvdStatements}
        onRefreshPrices={vi.fn()}
      />
    )

    expect(screen.getByText('ผลประโยชน์สะสมตาม Statement')).toBeInTheDocument()
    expect(screen.getByText(/เทียบเงินต้นสะสม · ไม่ใช่ผลตอบแทนรายปี/)).toBeInTheDocument()
    expect(screen.getAllByText('PVD ตาม Statement ณ 2026-06-30')).toHaveLength(2)
    expect(screen.getByText(/บันทึกพอร์ตล่าสุด/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'ดู/อัปเดต Statement PVD' })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /อัปเดตราคาตลาด/ })).not.toBeInTheDocument()
    expect(screen.queryByText(/อัปเดตราคาสำเร็จ/)).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'ดู/อัปเดต Statement PVD' }))
    expect(onViewPvdStatements).toHaveBeenCalledTimes(1)
  })

  it('uses the total portfolio cost denominator for PVD mixed with cash and separates PVD refresh entries', () => {
    const pvdContext: PvdValuationContext = {
      hasPvd: true, hasMarketHoldings: false, hasNonPvdValue: true,
      dates: ['2026-06-30'], hasUnknownDate: false, hasUnknownValuation: false,
      latestRecordedStatementDate: '2026-06-30',
    }
    const holdings = [
      { symbol: 'SINSA_FI', asset_type: 'PVD' },
      { symbol: 'CASH_THB', asset_type: 'Cash' },
    ] as any
    render(
      <PortfolioSummaryCards
        summary={{ total_value_thb: 1300, total_cost_basis_thb: 1100, total_unrealized_profit: 200, passive_income_ytd: 0 }}
        lastUpdated={null}
        pvdContext={pvdContext}
        holdings={holdings}
        priceRefreshInfo={{ SINSA_FI: 'Statement NAV unavailable' }}
      />
    )

    expect(screen.getByText('ผลประโยชน์สะสมรวม')).toBeInTheDocument()
    expect(screen.getByText(/เทียบต้นทุนรวมพอร์ตตามสูตรเดิม/)).toBeInTheDocument()
    expect(screen.getByText(/ไม่พบมูลค่า Statement PVD สำหรับบางรายการ/)).toBeInTheDocument()
    expect(screen.queryByText(/อัปเดตราคาสำเร็จ/)).not.toBeInTheDocument()
  })

  it('counts market outcomes without treating PVD Statement values as market refresh success', () => {
    const pvdContext: PvdValuationContext = {
      hasPvd: true, hasMarketHoldings: true, hasNonPvdValue: true,
      dates: ['2026-06-30'], hasUnknownDate: false, hasUnknownValuation: false,
      latestRecordedStatementDate: '2026-06-30',
    }
    const holdings = [
      { symbol: 'SINSA_FI', asset_type: 'PVD' },
      { symbol: 'SPY', asset_type: 'ETF' },
    ] as any
    render(
      <PortfolioSummaryCards
        summary={{ total_value_thb: 1300, total_cost_basis_thb: 1100, total_unrealized_profit: 200, passive_income_ytd: 0 }}
        lastUpdated={null}
        pvdContext={pvdContext}
        holdings={holdings}
        priceRefreshInfo={{
          SINSA_FI: 'Statement NAV 12.00 THB (2026-06-30)',
          SPY: 'ok',
          BAD: 'fetch failed (kept previous)',
        }}
        onRefreshPrices={vi.fn()}
        onViewPvdStatements={vi.fn()}
      />
    )

    expect(screen.getByText(/สำเร็จ 1\/2 รายการ/)).toBeInTheDocument()
    expect(screen.queryByText(/สำเร็จ 1\/3 รายการ/)).not.toBeInTheDocument()
    expect(screen.queryByText(/SINSA_FI:/)).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'รายละเอียด' }))
    expect(screen.queryByText(/SINSA_FI:/)).not.toBeInTheDocument()
    expect(screen.getByText('ok')).toHaveClass('text-emerald-600')
  })

  describe('isPriceRefreshSuccess helper', () => {
    it('returns true for valid price strings and statuses', () => {
      expect(isPriceRefreshSuccess('ok')).toBe(true)
      expect(isPriceRefreshSuccess('updated')).toBe(true)
      expect(isPriceRefreshSuccess('cached (TTL valid)')).toBe(true)
      expect(isPriceRefreshSuccess('32.9100 (live)')).toBe(true)
      expect(isPriceRefreshSuccess('32.9100 (fallback)')).toBe(true)
      expect(isPriceRefreshSuccess('146.92 USD')).toBe(true)
      expect(isPriceRefreshSuccess('50.00 THB')).toBe(true)
      expect(isPriceRefreshSuccess('Statement NAV 10.25 THB (2026-06-30)')).toBe(false)
      expect(isPriceRefreshSuccess('Statement NAV unavailable')).toBe(false)
    })

    it('returns false for error and failure statuses', () => {
      expect(isPriceRefreshSuccess('fetch failed (kept previous)')).toBe(false)
      expect(isPriceRefreshSuccess('timeout (kept previous)')).toBe(false)
      expect(isPriceRefreshSuccess('error: HTTP 404')).toBe(false)
      expect(isPriceRefreshSuccess('no_data')).toBe(false)
      expect(isPriceRefreshSuccess('')).toBe(false)
      expect(isPriceRefreshSuccess(null)).toBe(false)
      expect(isPriceRefreshSuccess(undefined)).toBe(false)
    })
  })
})
