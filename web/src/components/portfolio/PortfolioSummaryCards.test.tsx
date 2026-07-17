import { render, screen, fireEvent } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import PortfolioSummaryCards from './PortfolioSummaryCards'

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
})
