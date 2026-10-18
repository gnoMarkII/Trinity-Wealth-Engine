import { render, screen, fireEvent } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import PortfolioPerformanceCard from './PortfolioPerformanceCard'
import type { PerformanceSnapshotDTO } from '../../api/types'

describe('PortfolioPerformanceCard', () => {
  const mockRows: PerformanceSnapshotDTO[] = [
    {
      Date: '2026-08-14',
      Total_NAV: 31924.31,
      Total_Cost: 28006.8,
      Unrealized_PnL: 3917.52,
      Cash_Balance: 5000,
    },
    {
      Date: '2026-08-15',
      Total_NAV: 52534.17,
      Total_Cost: 38190.52,
      Unrealized_PnL: 14343.64,
      Cash_Balance: 8000,
    },
    {
      Date: '2026-08-16',
      Total_NAV: 55000.0,
      Total_Cost: 38190.52,
      Unrealized_PnL: 16809.48,
      Cash_Balance: 8000,
    },
  ]

  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders header, metrics, and chart with 3 data points', () => {
    render(
      <PortfolioPerformanceCard
        performanceRows={mockRows}
        daysRange={undefined}
        onChangeDaysRange={vi.fn()}
        portfolioId="default"
      />
    )

    expect(screen.getByText(/ผลการดำเนินงานและการเติบโต/)).toBeInTheDocument()
    expect(screen.getByText('3 จุดข้อมูล')).toBeInTheDocument()
    expect(screen.getAllByText('2026-08-16').length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText('฿55,000.00')).toBeInTheDocument()
  })

  it('renders empty/not enough data state when rows < 2', () => {
    render(
      <PortfolioPerformanceCard
        performanceRows={[mockRows[0]!]}
        daysRange={30}
        onChangeDaysRange={vi.fn()}
        portfolioId="default"
      />
    )

    expect(screen.getByText(/ต้องการข้อมูล Snapshot อย่างน้อย 2 วัน/)).toBeInTheDocument()
    expect(screen.getByText(/ช่วงเวลา 30 วันล่าสุดยังไม่มี Snapshot/)).toBeInTheDocument()
  })

  it('switches time range when clicking filter buttons', () => {
    const handleChangeDays = vi.fn()
    render(
      <PortfolioPerformanceCard
        performanceRows={mockRows}
        daysRange={30}
        onChangeDaysRange={handleChangeDays}
        portfolioId="default"
      />
    )

    const btn90d = screen.getByRole('button', { name: '90 วัน' })
    fireEvent.click(btn90d)
    expect(handleChangeDays).toHaveBeenCalledWith(90)

    const btnAll = screen.getByRole('button', { name: 'ทั้งหมด' })
    fireEvent.click(btnAll)
    expect(handleChangeDays).toHaveBeenCalledWith(undefined)
  })

  it('renders snapshot today status indicator correctly', () => {
    const d = new Date()
    const todayStr = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`
    const { rerender } = render(
      <PortfolioPerformanceCard
        performanceRows={mockRows}
        daysRange={undefined}
        onChangeDaysRange={vi.fn()}
        portfolioId="default"
      />
    )

    expect(screen.getByText('ยังไม่มี Snapshot วันนี้')).toBeInTheDocument()

    // Add snapshot with today's date
    const rowsWithToday: PerformanceSnapshotDTO[] = [
      ...mockRows,
      {
        Date: todayStr,
        Total_NAV: 56000.0,
        Total_Cost: 38190.52,
        Unrealized_PnL: 17809.48,
        Cash_Balance: 8000,
      },
    ]

    rerender(
      <PortfolioPerformanceCard
        performanceRows={rowsWithToday}
        daysRange={undefined}
        onChangeDaysRange={vi.fn()}
        portfolioId="default"
      />
    )

    expect(screen.getByText('✓ มี Snapshot วันนี้แล้ว')).toBeInTheDocument()
  })

  it('toggles collapsible history table', () => {
    render(
      <PortfolioPerformanceCard
        performanceRows={mockRows}
        daysRange={undefined}
        onChangeDaysRange={vi.fn()}
        portfolioId="default"
      />
    )

    // Table is closed by default
    expect(screen.queryByRole('table')).not.toBeInTheDocument()

    // Click to open table
    const toggleBtn = screen.getByText(/ดูตารางประวัติ Snapshot/i)
    fireEvent.click(toggleBtn)

    expect(screen.getByRole('table')).toBeInTheDocument()
    expect(screen.getAllByText('2026-08-14').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('2026-08-15').length).toBeGreaterThanOrEqual(1)

    // Click to close table
    fireEvent.click(screen.getByText(/ดูตารางประวัติ Snapshot/i))
    expect(screen.queryByRole('table')).not.toBeInTheDocument()
  })
})
