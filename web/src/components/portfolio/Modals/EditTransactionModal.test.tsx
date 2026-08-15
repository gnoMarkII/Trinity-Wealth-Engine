import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { describe, expect, it, vi, beforeEach } from 'vitest'
import EditTransactionModal from './EditTransactionModal'
import { api } from '../../../api/client'
import type { TransactionItemDTO } from '../../../api/types'

vi.mock('../../../api/client', () => ({
  api: {
    editTransaction: vi.fn(),
    getFxRate: vi.fn(),
  },
}))

describe('EditTransactionModal', () => {
  const mockUSDTransaction: TransactionItemDTO = {
    transaction_id: 'tx_usd_1',
    timestamp: '2026-08-14T22:23:06',
    symbol: 'UNH',
    action: 'BUY',
    units: 0.287077,
    price: 348.34,
    currency: 'USD',
    fx_rate: 36.5,
    cost_thb: 3650.0,
    realized_pnl_thb: null,
    notes: 'Initial lot',
  }

  const mockTHBTransaction: TransactionItemDTO = {
    transaction_id: 'tx_thb_1',
    timestamp: '2026-08-10T10:00:00',
    symbol: 'PTT',
    action: 'BUY',
    units: 100,
    price: 35.0,
    currency: 'THB',
    fx_rate: null,
    cost_thb: 3500.0,
    realized_pnl_thb: null,
    notes: 'PTT shares',
  }

  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(api.getFxRate).mockResolvedValue({ date: '2026-08-15', currency_pair: 'USDTHB', rate: 36.75, source: 'historical' })
  })

  it('renders modal with parsed date, time, units, price, fxRate, and notes', () => {
    render(
      <EditTransactionModal
        transaction={mockUSDTransaction}
        selectedPortfolioId="default"
        onClose={vi.fn()}
        onSuccess={vi.fn()}
      />
    )

    expect(screen.getByText('แก้ไขรายการ: UNH (BUY)')).toBeInTheDocument()
    expect(screen.getByDisplayValue('2026-08-14')).toBeInTheDocument()
    expect(screen.getByDisplayValue('22:23:06')).toBeInTheDocument()
    expect(screen.getByDisplayValue('0.287077')).toBeInTheDocument()
    expect(screen.getByDisplayValue('348.34')).toBeInTheDocument()
    expect(screen.getByDisplayValue('36.5')).toBeInTheDocument()
    expect(screen.getByDisplayValue('Initial lot')).toBeInTheDocument()
  })

  it('sets date to today when "📅 วันนี้" button is clicked and fetches FX for USD', async () => {
    render(
      <EditTransactionModal
        transaction={mockUSDTransaction}
        selectedPortfolioId="default"
        onClose={vi.fn()}
        onSuccess={vi.fn()}
      />
    )

    const todayBtn = screen.getByRole('button', { name: '📅 วันนี้' })
    fireEvent.click(todayBtn)

    const todayStr = new Date().toISOString().slice(0, 10)
    expect(screen.getByDisplayValue(todayStr)).toBeInTheDocument()
    await waitFor(() => {
      expect(api.getFxRate).toHaveBeenCalledWith(todayStr, 'default')
    })
  })

  it('sets date to yesterday when "⏮️ เมื่อวาน" button is clicked', async () => {
    render(
      <EditTransactionModal
        transaction={mockTHBTransaction}
        selectedPortfolioId="default"
        onClose={vi.fn()}
        onSuccess={vi.fn()}
      />
    )

    const yestBtn = screen.getByRole('button', { name: '⏮️ เมื่อวาน' })
    fireEvent.click(yestBtn)

    const d = new Date()
    d.setDate(d.getDate() - 1)
    const yestStr = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`
    expect(screen.getByDisplayValue(yestStr)).toBeInTheDocument()
  })

  it('sets time to current time when "🕒 ตอนนี้" button is clicked', () => {
    render(
      <EditTransactionModal
        transaction={mockUSDTransaction}
        selectedPortfolioId="default"
        onClose={vi.fn()}
        onSuccess={vi.fn()}
      />
    )

    const nowBtn = screen.getByRole('button', { name: '🕒 ตอนนี้' })
    fireEvent.click(nowBtn)

    // time input should now have a valid non-empty time
    const timeInputs = screen.getAllByDisplayValue(/^[0-9]{2}:[0-9]{2}:[0-9]{2}$/)
    expect(timeInputs.length).toBeGreaterThanOrEqual(1)
  })

  it('submits updated transaction data and calls onSuccess', async () => {
    const mockState = {
      holdings: [],
      summary: { total_nav_thb: 50000 },
    } as any
    vi.mocked(api.editTransaction).mockResolvedValue(mockState)
    const onSuccess = vi.fn()
    const onClose = vi.fn()

    render(
      <EditTransactionModal
        transaction={mockUSDTransaction}
        selectedPortfolioId="default"
        onClose={onClose}
        onSuccess={onSuccess}
      />
    )

    const unitsInput = screen.getByDisplayValue('0.287077')
    fireEvent.change(unitsInput, { target: { value: '0.5' } })

    const submitBtn = screen.getByRole('button', { name: 'บันทึกการแก้ไข' })
    fireEvent.click(submitBtn)

    await waitFor(() => {
      expect(api.editTransaction).toHaveBeenCalledWith(
        'tx_usd_1',
        {
          timestamp: '2026-08-14T22:23:06',
          units: 0.5,
          price: 348.34,
          fx_rate: 36.5,
          notes: 'Initial lot',
          adjust_cash: true,
        },
        'default'
      )
      expect(onSuccess).toHaveBeenCalledWith(mockState)
      expect(onClose).toHaveBeenCalled()
    })
  })
})
