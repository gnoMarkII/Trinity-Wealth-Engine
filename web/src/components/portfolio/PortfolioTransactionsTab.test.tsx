import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { describe, expect, it, vi, beforeEach } from 'vitest'
import PortfolioTransactionsTab from './PortfolioTransactionsTab'
import { api } from '../../api/client'
import type { TransactionListResponseDTO } from '../../api/types'

vi.mock('../../api/client', () => ({
  api: {
    getTransactions: vi.fn(),
    updateTransactionNote: vi.fn(),
    editTransaction: vi.fn(),
    deleteTransaction: vi.fn(),
    voidTransaction: vi.fn(),
    getFxRate: vi.fn(),
    getPvdDocumentUrl: vi.fn(() => '/pvd-document'),
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
    {
      transaction_id: 'pvd_20260930_sinsa_fi',
      timestamp: '2026-09-30 00:00:00',
      symbol: 'PVD_SINSA_FI',
      action: 'PVD_SNAPSHOT',
      units: 12.5,
      price: 100,
      currency: 'THB',
      cost_thb: 0,
      realized_pnl_thb: null,
      notes: 'PVD statement snapshot',
      gross_amount: '1250.00',
      net_amount: '1250.00',
      commission: '0.00',
      vat: '0.00',
      other_fees: '0.00',
      fee_currency: 'THB',
      confirmation_no: 'a'.repeat(64),
      source: 'BBLAM_PVD',
      fingerprint: 'b'.repeat(64),
      asset_type: 'PVD',
      bucket_id: 'retirement_pvd',
      reconciliation_adjustment: '0.00',
      pvd_metadata_json: JSON.stringify({
        payload_version: 1,
        fund_code: 'BBLAM',
        member_hash: 'c'.repeat(64),
        masked_member_id: '***1234',
        as_of_date: '2026-09-30',
        period_start: null,
        period_end: null,
        total_contribution: '250.00',
        total_nav: '1250.00',
        member_principal: '100.00',
        member_benefit: '50.00',
        employer_principal: '50.00',
        employer_benefit: '50.00',
        employer_vesting_percent: '100',
        member_vested_amount: '150.00',
        employer_vested_amount: '100.00',
        source_pdf_sha256: 'd'.repeat(64),
        semantic_fingerprint: 'a'.repeat(64),
        archive_id: 'd'.repeat(64),
        reconciliation_adjustment: '0.00',
      }),
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
    render(<PortfolioTransactionsTab portfolioId="default" activeView="assets" onViewChange={vi.fn()} />)

    await waitFor(() => {
      expect(screen.getAllByText('UNH').length).toBe(2)
      expect(screen.getByText('PTT')).toBeInTheDocument()
      expect(screen.getByText('Initial UNH lot')).toBeInTheDocument()
      expect(screen.getByText('Second UNH lot')).toBeInTheDocument()
    })
  })

  it('keeps one portfolio request and routes PVD snapshots to the statement panel', async () => {
    render(<PortfolioTransactionsTab portfolioId="default" activeView="pvd" onViewChange={vi.fn()} initialSymbol="PVD_SINSA_FI" />)

    expect(await screen.findByRole('heading', { name: /ประวัติ Statement PVD/ })).toBeInTheDocument()
    expect(api.getTransactions).toHaveBeenCalledTimes(1)
    expect(api.getTransactions).toHaveBeenCalledWith('default')
    expect(screen.getByRole('tab', { name: 'PVD' })).toHaveAttribute('aria-selected', 'true')
    expect(screen.getAllByText('฿1,250.00')).toHaveLength(2)

    fireEvent.click(screen.getByRole('button', { name: 'ดูรายละเอียด' }))
    expect(screen.getByRole('row', { name: /SINSA_FI/ })).toBeInTheDocument()
    expect(api.getPvdDocumentUrl).toHaveBeenCalledWith('d'.repeat(64), 'default')

    const search = screen.getByRole('searchbox', { name: 'ค้นหา Statement หรือกองทุน' })
    fireEvent.change(search, { target: { value: 'SINSA_FI' } })
    expect(screen.getByRole('heading', { name: /Statement ·/ })).toBeInTheDocument()
  })

  it('uses accessible arrow-key navigation for the transaction subtabs', async () => {
    const onViewChange = vi.fn()
    render(<PortfolioTransactionsTab portfolioId="default" activeView="assets" onViewChange={onViewChange} />)
    await waitFor(() => expect(api.getTransactions).toHaveBeenCalledTimes(1))
    const assetsTab = screen.getByRole('tab', { name: 'สินทรัพย์' })
    fireEvent.keyDown(assetsTab, { key: 'ArrowRight' })
    expect(onViewChange).toHaveBeenCalledWith('pvd')
    expect(screen.getByRole('tab', { name: 'PVD' })).toHaveFocus()
  })

  it('ignores a slower transaction response from the portfolio that was just left', async () => {
    let resolveOld!: (value: TransactionListResponseDTO) => void
    let resolveNew!: (value: TransactionListResponseDTO) => void
    const oldRequest = new Promise<TransactionListResponseDTO>((resolve) => { resolveOld = resolve })
    const newRequest = new Promise<TransactionListResponseDTO>((resolve) => { resolveNew = resolve })
    vi.mocked(api.getTransactions).mockImplementation((portfolioId = 'default') => portfolioId === 'old' ? oldRequest : newRequest)

    const result = render(<PortfolioTransactionsTab portfolioId="old" activeView="assets" onViewChange={vi.fn()} />)
    await waitFor(() => expect(api.getTransactions).toHaveBeenCalledWith('old'))
    result.rerender(<PortfolioTransactionsTab portfolioId="new" activeView="assets" onViewChange={vi.fn()} />)
    await waitFor(() => expect(api.getTransactions).toHaveBeenCalledWith('new'))

    resolveNew({ portfolio_id: 'new', transactions: [{ ...mockTransactions[0]!, transaction_id: 'new-tx', symbol: 'NEW' }], summary: mockSummary })
    expect(await screen.findByText('NEW')).toBeInTheDocument()
    resolveOld({ portfolio_id: 'old', transactions: mockTransactions, summary: mockSummary })
    await waitFor(() => expect(screen.queryByText('UNH')).not.toBeInTheDocument())
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })

  it('calculates Summ column correctly for SELL rows in both Native and THB views', async () => {
    render(<PortfolioTransactionsTab portfolioId="default" activeView="assets" onViewChange={vi.fn()} />)

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
    render(<PortfolioTransactionsTab portfolioId="default" activeView="assets" onViewChange={vi.fn()} />)

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

    const updatedRows = mockTransactions.map((tx) => tx.transaction_id === 'tx_1' ? { ...tx, notes: 'Updated UNH note via UI' } : tx)
    vi.mocked(api.getTransactions)
      .mockResolvedValueOnce({ portfolio_id: 'default', transactions: mockTransactions, summary: mockSummary })
      .mockResolvedValueOnce({ portfolio_id: 'default', transactions: updatedRows, summary: mockSummary })

    render(<PortfolioTransactionsTab portfolioId="default" activeView="assets" onViewChange={vi.fn()} />)

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

    render(<PortfolioTransactionsTab portfolioId="default" activeView="assets" onViewChange={vi.fn()} onSuccess={onSuccess} />)

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

  it('opens void confirmation modal and confirms void', async () => {
    const mockState = {
      holdings: [],
      summary: { total_nav_thb: 100000, total_cost_thb: 90000, total_unrealized_pnl_thb: 10000, total_realized_profit_ytd: 0, passive_income_ytd: 0, cash_balance_thb: 50000, total_accumulated_dividend: 0 },
      allocation_targets: [],
      fx_rates: {},
    } as any
    vi.mocked(api.voidTransaction).mockResolvedValue(mockState)
    const onSuccess = vi.fn()

    render(<PortfolioTransactionsTab portfolioId="default" activeView="assets" onViewChange={vi.fn()} onSuccess={onSuccess} />)

    await waitFor(() => {
      expect(screen.getByText('Take profit PTT')).toBeInTheDocument()
    })

    // Click void icon for PTT transaction (1st row in descending order)
    const voidButtons = screen.getAllByTitle('Void รายการ (ยกเลิกและสร้าง Reversal)')
    fireEvent.click(voidButtons[0]!)

    // Void confirmation modal should open
    expect(screen.getByText('ยืนยันการ Void รายการ (Non-destructive Reversal)')).toBeInTheDocument()
    expect(screen.getByText('PTT (SELL)')).toBeInTheDocument()

    // Confirm void
    const confirmBtn = screen.getByRole('button', { name: 'ยืนยันการ Void (Reversal)' })
    fireEvent.click(confirmBtn)

    await waitFor(() => {
      expect(api.voidTransaction).toHaveBeenCalledWith('tx_3', 'default')
      expect(onSuccess).toHaveBeenCalledWith(mockState)
    })
  })
})
