import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { api } from '../../../api/client'
import BatchAssignBucketModal from './BatchAssignBucketModal'

vi.mock('../../../api/client', () => ({
  api: {
    assignHoldingBucket: vi.fn(),
    batchAssignHoldingBuckets: vi.fn(),
  },
}))

describe('BatchAssignBucketModal', () => {
  it('allows assigning bucket to PVD symbols', () => {
    const targets = [{ bucket_id: 'defensive', name: 'Defensive Assets', target_percent: 20, color: '#A855F7' }]
    render(
      <BatchAssignBucketModal
        portfolioId="pvd-test"
        symbols={['PVD_SINSA_FI']}
        targets={targets}
        onClose={vi.fn()}
        onSuccess={vi.fn()}
      />
    )

    expect(screen.queryByText(/ยอด PVD อัปเดตจาก Statement/)).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: /ยืนยันการเปลี่ยน Bucket/i })).not.toBeDisabled()
    fireEvent.submit(screen.getByRole('button', { name: /ยืนยันการเปลี่ยน Bucket/i }).closest('form')!)
    expect(api.assignHoldingBucket).toHaveBeenCalledWith('PVD_SINSA_FI', { bucket_id: 'defensive' }, 'pvd-test')
  })
})
