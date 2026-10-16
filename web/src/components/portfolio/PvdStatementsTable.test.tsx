import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import type { TransactionItemDTO } from '../../api/types'
import { buildPvdStatementHistory } from '../../lib/pvdStatementHistory'
import PvdStatementsTable from './PvdStatementsTable'

vi.mock('../../api/client', () => ({ api: { getPvdDocumentUrl: vi.fn(() => '/pvd.pdf') } }))
vi.mock('./Modals/PvdStatementModal', () => ({
  default: ({ onClose, onSuccess }: { onClose: () => void; onSuccess: (state: any) => void }) => (
    <div role="dialog" aria-label="Mock PVD import">
      <button type="button" onClick={() => onSuccess({ holdings: [] })}>Finish mock import</button>
      <button type="button" onClick={onClose}>Close mock import</button>
    </div>
  ),
}))

function statementsFixture(): TransactionItemDTO[] {
  return 'ABCDEFGHIJK'.split('').map((suffix, index) => {
    const fingerprint = '0123456789abc'[index]!.repeat(64)
    const date = new Date(Date.UTC(2025, index, 1)).toISOString().slice(0, 10)
    const metadata = {
      payload_version: 1,
      fund_code: 'BBLAM',
      member_hash: 'c'.repeat(64),
      masked_member_id: '***1234',
      as_of_date: date,
      period_start: null,
      period_end: null,
      total_contribution: '250.00',
      total_nav: '100.00',
      member_principal: '100.00',
      member_benefit: '50.00',
      employer_principal: '50.00',
      employer_benefit: '50.00',
      employer_vesting_percent: '100',
      member_vested_amount: '150.00',
      employer_vested_amount: '100.00',
      source_pdf_sha256: 'd'.repeat(64),
      semantic_fingerprint: fingerprint,
      archive_id: 'd'.repeat(64),
      reconciliation_adjustment: '0.00',
    }
    return {
      transaction_id: `pvd-${suffix}`,
      timestamp: `${date} 00:00:00`,
      symbol: `PVD_FUND_${suffix}`,
      action: 'PVD_SNAPSHOT',
      units: 100,
      price: 1,
      currency: 'THB',
      cost_thb: 0,
      realized_pnl_thb: null,
      notes: '',
      gross_amount: '100.00',
      net_amount: '100.00',
      fee_currency: 'THB',
      confirmation_no: fingerprint,
      source: 'BBLAM_PVD',
      fingerprint: 'fedcba98765'[index]!.repeat(64),
      asset_type: 'PVD',
      reconciliation_adjustment: '0.00',
      pvd_metadata_json: JSON.stringify(metadata),
    }
  })
}

describe('PvdStatementsTable', () => {
  it('shows ten newest statements, paginates, and searches using the full PVD symbol or fund code', () => {
    const transactions = statementsFixture()
    const history = buildPvdStatementHistory('portfolio-a', transactions)
    render(
      <PvdStatementsTable
        portfolioId="portfolio-a"
        transactions={transactions}
        history={history}
        active
        loading={false}
        onRefresh={async () => {}}
      />
    )

    expect(screen.getByText('1–10 จาก 11 งวด')).toBeInTheDocument()
    expect(screen.getByText('Statement · 1 พ.ย. 2568')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'ถัดไป' }))
    expect(screen.getByText('11–11 จาก 11 งวด')).toBeInTheDocument()

    const search = screen.getByRole('searchbox', { name: 'ค้นหา Statement หรือกองทุน' })
    fireEvent.change(search, { target: { value: 'PVD_FUND_K' } })
    expect(screen.getAllByRole('heading', { name: /Statement ·/ })).toHaveLength(1)
    fireEvent.click(screen.getByRole('button', { name: 'ดูรายละเอียด' }))
    expect(screen.getByRole('row', { name: /FUND_K/ })).toBeInTheDocument()
    fireEvent.change(search, { target: { value: 'FUND_K' } })
    expect(screen.getAllByRole('heading', { name: /Statement ·/ })).toHaveLength(1)
  })

  it('closes import when inactive and refreshes once after a successful commit', async () => {
    const transactions = statementsFixture()
    const onRefresh = vi.fn().mockResolvedValue(undefined)
    const onSuccess = vi.fn()
    const props = {
      portfolioId: 'portfolio-a',
      transactions,
      history: buildPvdStatementHistory('portfolio-a', transactions),
      active: true,
      loading: false,
      onRefresh,
      onSuccess,
    }
    const view = render(<PvdStatementsTable {...props} />)
    fireEvent.click(screen.getByRole('button', { name: 'อัปเดตจาก Statement' }))
    expect(screen.getByRole('dialog', { name: 'Mock PVD import' })).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Finish mock import' }))
    expect(onSuccess).toHaveBeenCalledWith({ holdings: [] })
    expect(onRefresh).toHaveBeenCalledTimes(1)

    view.rerender(<PvdStatementsTable {...props} active={false} />)
    await waitFor(() => expect(screen.queryByRole('dialog', { name: 'Mock PVD import' })).not.toBeInTheDocument())
  })
})
