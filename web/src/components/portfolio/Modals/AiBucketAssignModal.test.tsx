import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { describe, expect, it, vi, beforeEach } from 'vitest'
import AiBucketAssignModal from './AiBucketAssignModal'
import { api } from '../../../api/client'
import type { AllocationTargetDTO, HoldingsBucketRecommendationResponseDTO } from '../../../api/types'

vi.mock('../../../api/client', () => ({
  api: {
    recommendHoldingBuckets: vi.fn(),
    batchReassignHoldingBuckets: vi.fn(),
  },
}))

describe('AiBucketAssignModal', () => {
  const mockTargets: AllocationTargetDTO[] = [
    { bucket_id: 'growth', name: 'Core Growth', target_percent: 70, color: '#3B82F6' },
    { bucket_id: 'reserve', name: 'Emergency Reserve', target_percent: 30, color: '#10B981' },
  ]

  const mockRecommendations: HoldingsBucketRecommendationResponseDTO = {
    summary: 'แนะนำจัดสรร NVDA เข้า Core Growth และย้าย CASH_THB เข้า Emergency Reserve',
    total_unassigned: 1,
    total_mismatched: 1,
    recommendations: [
      {
        symbol: 'NVDA',
        asset_type: 'Stock',
        current_bucket_id: null,
        recommended_bucket_id: 'growth',
        is_changed: true,
        reason: 'หุ้นเทคโนโลยีเติบโตสูง เหมาะกับกลุ่ม Core Growth',
      },
      {
        symbol: 'CASH_THB',
        asset_type: 'Cash',
        current_bucket_id: 'growth',
        recommended_bucket_id: 'reserve',
        is_changed: true,
        reason: 'เงินสดสภาพคล่องสูง ควรอยู่ในกลุ่มสำรองฉุกเฉิน',
      },
      {
        symbol: 'TLT',
        asset_type: 'Bond',
        current_bucket_id: 'reserve',
        recommended_bucket_id: 'reserve',
        is_changed: false,
        reason: 'พันธบัตรความเสี่ยงต่ำ เหมาะกับกลุ่มสำรองอยู่แล้ว',
      },
    ],
  }

  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders loading state initially and then shows recommendations', async () => {
    vi.mocked(api.recommendHoldingBuckets).mockResolvedValueOnce(mockRecommendations)

    render(
      <AiBucketAssignModal
        portfolioId="default"
        targets={mockTargets}
        onClose={vi.fn()}
        onSuccess={vi.fn()}
      />
    )

    expect(screen.getByText(/AI กำลังวิเคราะห์สินทรัพย์/i)).toBeInTheDocument()

    await waitFor(() => {
      expect(screen.getByText('จัดกลุ่มสินทรัพย์อัตโนมัติด้วย AI')).toBeInTheDocument()
      expect(screen.getByText(/แนะนำจัดสรร NVDA เข้า Core Growth/i)).toBeInTheDocument()
    })

    expect(screen.getByText('NVDA')).toBeInTheDocument()
    expect(screen.getByText('CASH_THB')).toBeInTheDocument()
  })

  it('renders error state and allows retry on API error', async () => {
    vi.mocked(api.recommendHoldingBuckets).mockRejectedValueOnce(new Error('AI Service Unavailable'))

    render(
      <AiBucketAssignModal
        portfolioId="default"
        targets={mockTargets}
        onClose={vi.fn()}
        onSuccess={vi.fn()}
      />
    )

    await waitFor(() => {
      expect(screen.getByText(/AI Service Unavailable/i)).toBeInTheDocument()
    })

    const retryBtn = screen.getByRole('button', { name: /ลองใหม่อีกครั้ง/i })
    expect(retryBtn).toBeInTheDocument()

    vi.mocked(api.recommendHoldingBuckets).mockResolvedValueOnce(mockRecommendations)
    fireEvent.click(retryBtn)

    await waitFor(() => {
      expect(screen.getByText('NVDA')).toBeInTheDocument()
    })
  })

  it('toggles filter to show all recommendations vs only changed items', async () => {
    vi.mocked(api.recommendHoldingBuckets).mockResolvedValueOnce(mockRecommendations)

    render(
      <AiBucketAssignModal
        portfolioId="default"
        targets={mockTargets}
        onClose={vi.fn()}
        onSuccess={vi.fn()}
      />
    )

    await waitFor(() => {
      expect(screen.getByText('NVDA')).toBeInTheDocument()
    })

    // By default showOnlyChanged is true, so TLT (unchanged) is hidden
    expect(screen.queryByText('TLT')).not.toBeInTheDocument()

    // Uncheck "แสดงเฉพาะรายการที่มีการปรับเปลี่ยน"
    const checkbox = screen.getByRole('checkbox')
    fireEvent.click(checkbox)

    expect(screen.getByText('TLT')).toBeInTheDocument()
  })

  it('allows user to change bucket dropdown and submits assignments', async () => {
    vi.mocked(api.recommendHoldingBuckets).mockResolvedValueOnce(mockRecommendations)
    const mockUpdatedState = { holdings: [], allocation_targets: [] } as any
    vi.mocked(api.batchReassignHoldingBuckets).mockResolvedValueOnce(mockUpdatedState)

    const onClose = vi.fn()
    const onSuccess = vi.fn()

    render(
      <AiBucketAssignModal
        portfolioId="default"
        targets={mockTargets}
        onClose={onClose}
        onSuccess={onSuccess}
      />
    )

    await waitFor(() => {
      expect(screen.getByText('NVDA')).toBeInTheDocument()
    })

    // Find the select dropdown for NVDA and change to reserve
    const selects = screen.getAllByRole('combobox')
    fireEvent.change(selects[0]!, { target: { value: 'reserve' } })

    // Click confirm button
    const submitBtn = screen.getByRole('button', { name: /ยืนยันการจัดกลุ่ม/i })
    fireEvent.click(submitBtn)

    await waitFor(() => {
      expect(api.batchReassignHoldingBuckets).toHaveBeenCalledWith(
        {
          assignments: [
            { symbol: 'NVDA', bucket_id: 'reserve' },
            { symbol: 'CASH_THB', bucket_id: 'reserve' },
            { symbol: 'TLT', bucket_id: 'reserve' },
          ],
        },
        'default'
      )
      expect(onSuccess).toHaveBeenCalledWith(mockUpdatedState)
      expect(onClose).toHaveBeenCalled()
    })
  })
})
