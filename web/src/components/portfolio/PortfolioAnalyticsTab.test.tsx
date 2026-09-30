import { render, screen } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import PortfolioAnalyticsTab from './PortfolioAnalyticsTab'
import type { PerformanceSnapshotDTO, ActualHoldingDTO } from '../../api/types'

describe('PortfolioAnalyticsTab Component (Phase 4)', () => {
  const mockHoldings: ActualHoldingDTO[] = [
    {
      symbol: 'AAPL',
      asset_type: 'Stock',
      units: 10,
      market_value_thb: 70000,
      unrealized_pnl_percent: 0.12,
      unrealized_pnl_value: 8400,
      avg_cost_usd: 180,
      avg_cost_thb: 6160,
      current_price_usd: 200,
      current_price_thb: 7000,
      company_name: 'Apple Inc.',
      pe_ratio: 30,
      eps: 6.5,
      payout_ratio: 0.15,
      market_cap_value: 3000000000000,
      dividend_per_share: 1.0,
      dividend_yield: 0.005,
      accumulated_dividend_thb: 500,
      fundamentals_updated_at: 1720000000,
      market_cap_tier: 'Mega',
      yield_on_cost: 0.005,
      bucket_id: 'core_equities',
    },
    {
      symbol: 'GLD',
      asset_type: 'ETF',
      units: 5,
      market_value_thb: 30000,
      unrealized_pnl_percent: -0.02,
      unrealized_pnl_value: -600,
      avg_cost_usd: 220,
      avg_cost_thb: 6120,
      current_price_usd: 215,
      current_price_thb: 6000,
      company_name: 'SPDR Gold Trust',
      pe_ratio: null,
      eps: null,
      payout_ratio: null,
      market_cap_value: null,
      dividend_per_share: null,
      dividend_yield: null,
      accumulated_dividend_thb: null,
      fundamentals_updated_at: null,
      market_cap_tier: null,
      yield_on_cost: null,
      bucket_id: 'defensive',
    },
  ]

  const mockRowsWithBreakdown: PerformanceSnapshotDTO[] = [
    {
      Date: '2026-09-24',
      Total_NAV: 120000,
      Total_Cost: 110000,
      Unrealized_PnL: 10000,
      Cash_Balance: 20000,
      Asset_Class_Values_THB: {
        Stock: 70000,
        ETF: 30000,
        Cash: 20000,
      },
      coverage_warning: null,
    },
    {
      Date: '2026-09-25',
      Total_NAV: 125000,
      Total_Cost: 110000,
      Unrealized_PnL: 15000,
      Cash_Balance: 25000,
      Asset_Class_Values_THB: {
        Stock: 72000,
        ETF: 28000,
        Cash: 25000,
      },
      coverage_warning: null,
    },
  ]

  const mockRowsOldNoBreakdown: PerformanceSnapshotDTO[] = [
    {
      Date: '2026-08-01',
      Total_NAV: 100000,
      Total_Cost: 95000,
      Unrealized_PnL: 5000,
      Cash_Balance: 15000,
      Asset_Class_Values_THB: null, // Old row without breakdown
    },
    {
      Date: '2026-08-02',
      Total_NAV: 102000,
      Total_Cost: 95000,
      Unrealized_PnL: 7000,
      Cash_Balance: 15000,
      Asset_Class_Values_THB: null, // Old row without breakdown
    },
  ]

  it('renders TreemapChart with current holdings asset types and cash', () => {
    render(
      <PortfolioAnalyticsTab
        performanceRows={mockRowsWithBreakdown}
        daysRange={30}
        onChangeDaysRange={vi.fn()}
        holdings={mockHoldings}
        cashBalanceThb={20000}
      />
    )

    expect(screen.getByText('Asset-Type Allocation (Current Holdings)')).toBeInTheDocument()
    expect(screen.getByText(/Sector exposure omitted until verified constituent taxonomy/i)).toBeInTheDocument()
    expect(screen.getByText('AAPL')).toBeInTheDocument()
    expect(screen.getByText('GLD')).toBeInTheDocument()
    expect(screen.getByText('Cash (THB)')).toBeInTheDocument()
  })

  it('renders StackedAreaChart when historical asset-class breakdown exists', () => {
    render(
      <PortfolioAnalyticsTab
        performanceRows={mockRowsWithBreakdown}
        daysRange={30}
        onChangeDaysRange={vi.fn()}
        holdings={mockHoldings}
      />
    )

    expect(screen.getByText('Historical Asset-Class Allocation')).toBeInTheDocument()
    expect(screen.getByText(/Toggle between absolute THB and 100% normalized mode/i)).toBeInTheDocument()
  })

  it('handles older historical rows without breakdown as gaps per data integrity rules', () => {
    render(
      <PortfolioAnalyticsTab
        performanceRows={mockRowsOldNoBreakdown}
        daysRange={30}
        onChangeDaysRange={vi.fn()}
        holdings={[]}
      />
    )

    // Informs the user that older snapshots have no breakdown and are not backfilled
    expect(screen.getByText(/ระบบบันทึกประวัติการกระจายสินทรัพย์/i)).toBeInTheDocument()
    expect(screen.getByText(/จะแสดงเป็นช่องว่าง \(gap\) ตามหลักความถูกต้องของข้อมูลโดยไม่ทำการประมาณค่าหรือ backfill/i)).toBeInTheDocument()
  })

  it('displays coverage warning if breakdown sum does not match Total NAV', () => {
    const rowsWithWarning: PerformanceSnapshotDTO[] = [
      {
        ...mockRowsWithBreakdown[0]!,
        coverage_warning: 'Breakdown sum 110000 differs from Total NAV 120000',
      },
      {
        ...mockRowsWithBreakdown[1]!,
      },
    ]

    render(
      <PortfolioAnalyticsTab
        performanceRows={rowsWithWarning}
        daysRange={30}
        onChangeDaysRange={vi.fn()}
        holdings={mockHoldings}
      />
    )

    expect(screen.getByText(/Coverage Warning:/i)).toBeInTheDocument()
    expect(screen.getByText(/Breakdown sum 110000 differs from Total NAV 120000/i)).toBeInTheDocument()
  })
})
