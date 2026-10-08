import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import TradeModal from './TradeModal'
import { api } from '../../../api/client'

vi.mock('../../../api/client', () => ({
  api: {
    executeTrade: vi.fn(),
    manageCashFlow: vi.fn(),
    getFxRate: vi.fn().mockResolvedValue({ date: '2026-08-15', currency_pair: 'USDTHB', rate: 36.5, source: 'live' }),
  },
  ApiError: class ApiError extends Error {},
}))

describe('TradeModal', () => {
  beforeEach(() => {
    vi.resetAllMocks()
    vi.mocked(api.getFxRate).mockResolvedValue({ date: '2026-08-15', currency_pair: 'USDTHB', rate: 36.5, source: 'live' })
  })

  it('renders cash balance badge correctly', () => {
    const holdings = [
      { symbol: 'CASH_USD', asset_type: 'Cash', units: 100, avg_cost_usd: 1, avg_cost_thb: null, bucket_id: null },
      { symbol: 'CASH_THB', asset_type: 'Cash', units: 5000, avg_cost_usd: null, avg_cost_thb: 1, bucket_id: null },
    ]

    render(
      <TradeModal
        targets={[]}
        holdings={holdings as any}
        selectedPortfolioId="default"
        onClose={vi.fn()}
        onSuccess={vi.fn()}
      />
    )

    expect(screen.getByText('💵')).toBeInTheDocument()
    expect(screen.getByText('$100.00 USD')).toBeInTheDocument()
    expect(screen.getByText('฿5,000.00 THB')).toBeInTheDocument()
  })

  it('renders quick top-up button on Insufficient cash balance error and handles top-up & auto-buy', async () => {
    const holdings = [
      { symbol: 'CASH_USD', asset_type: 'Cash', units: 0, avg_cost_usd: 1, avg_cost_thb: null, bucket_id: null },
    ]

    vi.mocked(api.executeTrade).mockRejectedValueOnce(
      new Error('Insufficient cash balance — มี 0.00 USD ต้องใช้ 300.00 USD')
    )
    vi.mocked(api.manageCashFlow).mockResolvedValue({} as any)
    vi.mocked(api.executeTrade).mockResolvedValueOnce({} as any)

    const onSuccess = vi.fn()
    const onClose = vi.fn()

    render(
      <TradeModal
        targets={[]}
        holdings={holdings as any}
        selectedPortfolioId="default"
        onClose={onClose}
        onSuccess={onSuccess}
      />
    )

    // Fill trade form for buying PG
    fireEvent.change(screen.getByPlaceholderText('e.g. AAPL, PTT, NVDA'), { target: { value: 'PG' } })
    const numberInputs = screen.getAllByPlaceholderText('0.00') as HTMLInputElement[]
    fireEvent.change(numberInputs[0]!, { target: { value: '2.1338612' } })
    fireEvent.change(numberInputs[1]!, { target: { value: '140.59' } })

    const selects = screen.getAllByRole('combobox') as HTMLSelectElement[]
    // 3rd select is Currency (THB/USD)
    fireEvent.change(selects[2]!, { target: { value: 'USD' } })

    // Click Buy button
    const buyBtn = screen.getByRole('button', { name: /ยืนยันการซื้อ/i })
    fireEvent.click(buyBtn)

    // Wait for error banner and quick top-up button
    await waitFor(() => {
      expect(screen.getByText(/Insufficient cash balance/i)).toBeInTheDocument()
    })

    const topupBtn = screen.getByRole('button', { name: /เติมเงินสดเพิ่ม.*USD & ยืนยันการซื้อทันที/i })
    expect(topupBtn).toBeInTheDocument()

    // Click quick top-up button
    fireEvent.click(topupBtn)

    await waitFor(() => {
      expect(api.manageCashFlow).toHaveBeenCalledWith(
        expect.objectContaining({
          action: 'deposit',
          currency: 'USD',
        }),
        'default'
      )
      expect(api.executeTrade).toHaveBeenCalledTimes(2)
      expect(onSuccess).toHaveBeenCalled()
      expect(onClose).toHaveBeenCalled()
    })
  })

  it('renders quick top-up button with real backend error format: Insufficient cash in CASH_USD', async () => {
    const holdings = [
      { symbol: 'CASH_USD', asset_type: 'Cash', units: 0, avg_cost_usd: 1, avg_cost_thb: null, bucket_id: null },
    ]

    vi.mocked(api.executeTrade).mockRejectedValueOnce(
      new Error('Insufficient cash in CASH_USD (required: 300.00, available: 0.00)')
    )
    vi.mocked(api.manageCashFlow).mockResolvedValue({} as any)
    vi.mocked(api.executeTrade).mockResolvedValueOnce({} as any)

    render(
      <TradeModal
        targets={[]}
        holdings={holdings as any}
        selectedPortfolioId="default"
        onClose={vi.fn()}
        onSuccess={vi.fn()}
      />
    )

    fireEvent.change(screen.getByPlaceholderText('e.g. AAPL, PTT, NVDA'), { target: { value: 'NVDA' } })
    const numberInputs = screen.getAllByPlaceholderText('0.00') as HTMLInputElement[]
    fireEvent.change(numberInputs[0]!, { target: { value: '2' } })
    fireEvent.change(numberInputs[1]!, { target: { value: '150' } })

    const selects = screen.getAllByRole('combobox') as HTMLSelectElement[]
    fireEvent.change(selects[2]!, { target: { value: 'USD' } })

    const buyBtn = screen.getByRole('button', { name: /ยืนยันการซื้อ/i })
    fireEvent.click(buyBtn)

    await waitFor(() => {
      expect(screen.getByText(/Insufficient cash in CASH_USD/i)).toBeInTheDocument()
    })

    const topupBtn = screen.getByRole('button', { name: /เติมเงินสดเพิ่ม.*USD & ยืนยันการซื้อทันที/i })
    expect(topupBtn).toBeInTheDocument()
  })
})
