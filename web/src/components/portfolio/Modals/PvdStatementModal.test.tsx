import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import PvdStatementModal from './PvdStatementModal'
import { api } from '../../../api/client'

vi.mock('../../../api/client', () => ({
  api: {
    analyzePvdUpload: vi.fn(),
    commitPvdStatement: vi.fn(),
    getPvdDocumentUrl: vi.fn(() => '/pvd-document'),
  },
}))

describe('PvdStatementModal', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    sessionStorage.clear()
    vi.spyOn(crypto, 'randomUUID').mockReturnValue('12345678-1234-1234-1234-123456789abc')
  })

  it('analyzes the PDF, previews the full fund group, and commits atomically', async () => {
    const onSuccess = vi.fn()
    vi.mocked(api.analyzePvdUpload).mockResolvedValue({
      scan_id: 'scan-pvd', can_commit: true, status: 'ready', message: 'ready',
      statement: {
        as_of_date: '2026-09-30', masked_member_id: '••••7890', member_principal: '600.00', member_benefit: '200.00',
        employer_principal: '300.00', employer_benefit: '150.00', total_nav: '1250.00', total_contribution: '900.00',
      },
      holdings: [{ fund_code: 'SINSA_FI', name: 'Fixed Income', units: '123.4567', statement_nav: '10.1234', statement_market_value: '1250.00', reconciliation_adjustment: '0.00', adjusted_valuation: '1250.00', allocated_cost: '900.00', unrealized_pnl: '350.00' }],
      current_nav: '0.00', new_nav: '1250.00', delta_nav: '1250.00', vesting_percent: '100', vested_amount: '1250.00', archive_id: null,
    })
    const mockState = { holdings: [], summary: {}, allocation_targets: [], fx_rates: {} } as any
    vi.mocked(api.commitPvdStatement).mockResolvedValue({ status: 'committed', message: 'saved', archive_id: 'a'.repeat(64), state: mockState })

    render(<PvdStatementModal portfolioId="default" onClose={vi.fn()} onSuccess={onSuccess} />)
    const file = new File(['%PDF-test'], 'statement.pdf', { type: 'application/pdf' })
    fireEvent.change(screen.getByLabelText(/Statement PDF/), { target: { files: [file] } })
    fireEvent.click(screen.getByRole('button', { name: 'ตรวจสอบ Statement' }))

    await waitFor(() => expect(api.analyzePvdUpload).toHaveBeenCalledWith(file, 'default', '', 100))
    expect(await screen.findByText('SINSA_FI')).toBeInTheDocument()
    expect(screen.getAllByText('฿1,250.00').length).toBeGreaterThan(0)
    expect(screen.getByText('เงินต้นสะสมตาม Statement')).toBeInTheDocument()
    expect(screen.getByText('ผลประโยชน์รวมตาม Statement')).toBeInTheDocument()
    expect(screen.getByText('฿350.00')).toBeInTheDocument()
    expect(screen.getByText('เงินต้นจัดสรร')).toBeInTheDocument()
    expect(screen.queryByText('กำไร/ขาดทุน')).not.toBeInTheDocument()
    expect(screen.getByText('ยอด PVD ล่าสุดที่บันทึกในระบบ')).toBeInTheDocument()
    expect(screen.getByText('ยอดเปลี่ยนจากที่บันทึก')).toBeInTheDocument()

    fireEvent.change(screen.getByRole('slider', { name: 'เปอร์เซ็นต์สิทธิเงินสมทบนายจ้าง' }), { target: { value: '50' } })
    expect(screen.getByText('฿1,025.00')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'ยืนยันนำเข้าทั้งชุด' }))

    await waitFor(() => expect(api.commitPvdStatement).toHaveBeenCalledWith({
      scan_id: 'scan-pvd', idempotency_key: '12345678-1234-1234-1234-123456789abc', portfolio_id: 'default', vesting_percent: 50,
    }))
    expect(onSuccess).toHaveBeenCalledWith(mockState)
    expect(await screen.findByText('saved')).toBeInTheDocument()
  })

  it('does not turn missing or malformed statement totals into zero benefit', async () => {
    vi.mocked(api.analyzePvdUpload).mockResolvedValue({
      scan_id: null, can_commit: false, status: 'needs_review', message: 'review',
      statement: { as_of_date: '2026-09-30', total_nav: 'invalid', total_contribution: null },
      holdings: [], current_nav: null, new_nav: null, delta_nav: null,
      vesting_percent: '100', vested_amount: null, archive_id: null,
    })
    render(<PvdStatementModal portfolioId="default" onClose={vi.fn()} onSuccess={vi.fn()} />)
    const file = new File(['%PDF-test'], 'statement.pdf', { type: 'application/pdf' })
    fireEvent.change(screen.getByLabelText(/Statement PDF/), { target: { files: [file] } })
    fireEvent.click(screen.getByRole('button', { name: 'ตรวจสอบ Statement' }))

    expect(await screen.findAllByText('ไม่พร้อมคำนวณ', { exact: true })).toHaveLength(6)
    expect(screen.queryByText('฿0.00')).not.toBeInTheDocument()
  })
})
