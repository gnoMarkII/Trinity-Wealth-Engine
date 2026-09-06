import { render, screen, fireEvent } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import PortfolioSummaryCards, { isPriceRefreshSuccess } from './PortfolioSummaryCards'

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

  describe('isPriceRefreshSuccess helper', () => {
    it('returns true for valid price strings and statuses', () => {
      expect(isPriceRefreshSuccess('ok')).toBe(true)
      expect(isPriceRefreshSuccess('updated')).toBe(true)
      expect(isPriceRefreshSuccess('cached (TTL valid)')).toBe(true)
      expect(isPriceRefreshSuccess('32.9100 (live)')).toBe(true)
      expect(isPriceRefreshSuccess('32.9100 (fallback)')).toBe(true)
      expect(isPriceRefreshSuccess('146.92 USD')).toBe(true)
      expect(isPriceRefreshSuccess('50.00 THB')).toBe(true)
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
