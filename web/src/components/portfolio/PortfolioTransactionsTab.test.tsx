import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { describe, expect, it, vi, beforeEach } from 'vitest'
import PortfolioTransactionsTab from './PortfolioTransactionsTab'
import { api } from '../../api/client'

vi.mock('../../api/client', () => ({
  api: {
    getTransactions: vi.fn(),
    updateTransactionNote: vi.fn(),
    editTransaction: vi.fn(),
    deleteTransaction: vi.fn(),
    getFxRate: vi.fn(),
  },
}))

describe('PortfolioTransactionsTab', () => {
  const mockTransactions = [
    {
      transaction_id: 'tx_1',
      timestamp: '2026-08-14T22:23:06',
      symbol: 'UNH',
      action: 'BUY',
      units: 0.287077,
      price: 348.34,
      currency: 'USD',
      fx_rate: 36.5,
      cost_thb: 3650.0,
      realized_pnl_thb: null,
      notes: 'Initial UNH lot',
    },
    {
      transaction_id: 'tx_2',
      timestamp: '2026-08-14T22:24:56',
      symbol: 'UNH',
      action: 'BUY',
      units: 1.12439,
      price: 326.68,
      currency: 'USD',
      fx_rate: 36.5,
      cost_thb: 13406.81,
      realized_pnl_thb: null,
      notes: 'Second UNH lot',
    },
    {
      transaction_id: 'tx_3',
      timestamp: '2026-08-14T22:30:00',
      symbol: 'PTT',
      action: 'SELL',
      units: 40.0,
      price: 40.0,
      currency: 'THB',
      fx_rate: null,
      cost_thb: 1400.0,
      realized_pnl_thb: 200.0,
      notes: 'Take profit PTT',
    },
  ]

  const mockSummary = {
    total_buy_count: 2,
    total_sell_count: 1,
    total_buy_thb: 17056.81,
    total_sell_thb: 1600.0,
    total_realized_pnl_thb: 200.0,
  }

  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(api.getTransactions).mockResolvedValue({
      portfolio_id: 'default',
      transactions: mockTransactions,
      summary: mockSummary,
    })
  })

  it('renders transactions table and summary badges from api', async () => {
    render(<PortfolioTransactionsTab portfolioId="default" />)

    await waitFor(() => {
      expect(screen.getAllByText('UNH').length).toBe(2)
      expect(screen.getByText('PTT')).toBeInTheDocument()
      expect(screen.getByText('Initial UNH lot')).toBeInTheDocument()
      expect(screen.getByText('Second UNH lot')).toBeInTheDocument()
    })
  })

  it('calculates Summ column correctly for SELL rows in both Native and THB views', async () => {
    render(<PortfolioTransactionsTab portfolioId="default" />)

    await waitFor(() => {
      expect(screen.getByText('PTT')).toBeInTheDocument()
    })

    // In Native view, 40 shares @ ฿40 = +฿1,600.00
    expect(screen.getByText('+฿1,600.00')).toBeInTheDocument()

    // Switch to THB view
    const thbButton = screen.getByRole('button', { name: 'THB' })
    fireEvent.click(thbButton)

    // In THB view, SELL row should calculate proceeds = cost_thb (1,400) + realized_pnl_thb (200) = +฿1,600.00
    expect(screen.getByText('+฿1,600.00')).toBeInTheDocument()
  })

  it('filters transactions by action type', async () => {
    render(<PortfolioTransactionsTab portfolioId="default" />)

    await waitFor(() => {
      expect(screen.getByText('PTT')).toBeInTheDocument()
    })

    // Click Buy filter
    const buyButton = screen.getByRole('button', { name: 'Buy' })
    fireEvent.click(buyButton)

    expect(screen.getAllByText('UNH').length).toBe(2)
    expect(screen.queryByText('PTT')).not.toBeInTheDocument()

    // Click Sell filter
    const sellButton = screen.getByRole('button', { name: 'Sell' })
    fireEvent.click(sellButton)

    expect(screen.getByText('PTT')).toBeInTheDocument()
    expect(screen.queryByText('UNH')).not.toBeInTheDocument()
  })

  it('allows inline editing and saving a note', async () => {
    vi.mocked(api.updateTransactionNote).mockResolvedValue({
      transaction_id: 'tx_1',
      timestamp: '2026-08-14T22:23:06',
      symbol: 'UNH',
      action: 'BUY',
      units: 0.287077,
      price: 348.34,
      currency: 'USD',
      fx_rate: 36.5,
      cost_thb: 3650.0,
      realized_pnl_thb: null,
      notes: 'Updated UNH note via UI',
    })

    render(<PortfolioTransactionsTab portfolioId="default" />)

    await waitFor(() => {
      expect(screen.getByText('Initial UNH lot')).toBeInTheDocument()
    })

    // Click on note text to edit
    fireEvent.click(screen.getByText('Initial UNH lot'))

    const input = screen.getByPlaceholderText('ระบุบันทึกช่วยจำ...')
    fireEvent.change(input, { target: { value: 'Updated UNH note via UI' } })

    const saveButton = screen.getByTitle('บันทึก (Enter)')
    fireEvent.click(saveButton)

    await waitFor(() => {
      expect(api.updateTransactionNote).toHaveBeenCalledWith('tx_1', 'Updated UNH note via UI', 'default')
      expect(screen.getByText('Updated UNH note via UI')).toBeInTheDocument()
    })
  })

  it('opens EditTransactionModal and submits transaction edit', async () => {
    const mockState = {
      holdings: [],
      summary: { total_nav_thb: 100000, total_cost_thb: 90000, total_unrealized_pnl_thb: 10000, total_realized_profit_ytd: 200, passive_income_ytd: 0, cash_balance_thb: 50000, total_accumulated_dividend: 0 },
      allocation_targets: [],
      fx_rates: {},
    } as any
    vi.mocked(api.editTransaction).mockResolvedValue(mockState)
    const onSuccess = vi.fn()

    render(<PortfolioTransactionsTab portfolioId="default" onSuccess={onSuccess} />)

    await waitFor(() => {
      expect(screen.getByText('Take profit PTT')).toBeInTheDocument()
    })

    // Click edit icon for PTT transaction (1st row in descending order)
    const editButtons = screen.getAllByTitle('แก้ไขข้อมูล Transaction (วัน, จำนวนหุ้น, ราคา, Note)')
    fireEvent.click(editButtons[0]!)

    // Edit modal should open
    expect(screen.getByText('แก้ไขรายการ: PTT (SELL)')).toBeInTheDocument()

    // Submit edit
    const submitBtn = screen.getByRole('button', { name: 'บันทึกการแก้ไข' })
    fireEvent.click(submitBtn)

    await waitFor(() => {
      expect(api.editTransaction).toHaveBeenCalledWith('tx_3', expect.objectContaining({
        units: 40,
        price: 40,
        adjust_cash: true,
      }), 'default')
      expect(onSuccess).toHaveBeenCalledWith(mockState)
    })
  })

  it('opens delete confirmation modal and confirms deletion', async () => {
    const mockState = {
      holdings: [],
      summary: { total_nav_thb: 100000, total_cost_thb: 90000, total_unrealized_pnl_thb: 10000, total_realized_profit_ytd: 0, passive_income_ytd: 0, cash_balance_thb: 50000, total_accumulated_dividend: 0 },
      allocation_targets: [],
      fx_rates: {},
    } as any
    vi.mocked(api.deleteTransaction).mockResolvedValue(mockState)
    const onSuccess = vi.fn()

    render(<PortfolioTransactionsTab portfolioId="default" onSuccess={onSuccess} />)

    await waitFor(() => {
      expect(screen.getByText('Take profit PTT')).toBeInTheDocument()
    })

    // Click delete icon for PTT transaction (1st row in descending order)
    const deleteButtons = screen.getAllByTitle('ลบรายการ Transaction')
    fireEvent.click(deleteButtons[0]!)

    // Delete confirmation modal should open
    expect(screen.getByText('ยืนยันการลบรายการ')).toBeInTheDocument()
    expect(screen.getByText('PTT (SELL)')).toBeInTheDocument()

    // Confirm deletion
    const confirmBtn = screen.getByRole('button', { name: 'ยืนยันการลบ' })
    fireEvent.click(confirmBtn)

    await waitFor(() => {
      expect(api.deleteTransaction).toHaveBeenCalledWith('tx_3', { adjust_cash: true }, 'default')
      expect(onSuccess).toHaveBeenCalledWith(mockState)
    })
  })
})
