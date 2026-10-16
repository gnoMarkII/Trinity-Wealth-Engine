import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import type { ActualHoldingDTO } from '../../../api/types'
import { api } from '../../../api/client'
import HoldingCorrectionModal from './HoldingCorrectionModal'

vi.mock('../../../api/client', () => ({
  api: { editHolding: vi.fn() },
}))

describe('HoldingCorrectionModal', () => {
  it('renders PVD as read-only and does not expose correction fields', () => {
    const holding = {
      symbol: 'SINSA_FI', asset_type: 'PVD', units: 12, market_value_thb: 120,
      pvd_valuation: { as_of_date: '2026-06-30' },
    } as ActualHoldingDTO

    render(
      <HoldingCorrectionModal
        portfolioId="pvd-test"
        holding={holding}
        targets={[]}
        onClose={vi.fn()}
        onSuccess={vi.fn()}
      />
    )

    expect(screen.getByText('ข้อมูล PVD แบบอ่านอย่างเดียว')).toBeInTheDocument()
    expect(screen.getByText('2026-06-30')).toBeInTheDocument()
    expect(screen.queryByLabelText(/จำนวนหน่วย/)).not.toBeInTheDocument()
    expect(api.editHolding).not.toHaveBeenCalled()
  })
})
