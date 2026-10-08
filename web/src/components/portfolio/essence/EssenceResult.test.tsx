import { render, screen, fireEvent } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import EssenceResult from './EssenceResult'
import type { SummaryResponseDTO, ConfirmedEssenceDTO } from '../../../api/types'

describe('EssenceResult', () => {
  const mockSummary: SummaryResponseDTO = {
    summary_id: 'sum-1',
    session_id: 'sess-1',
    statement: 'ฉันลงทุนเพื่อสร้างอิสรภาพในการใช้ชีวิต และให้ความสำคัญกับความมั่นคงของเงินต้น',
    claims: [
      {
        claim_id: 'claim-1',
        text: 'เน้นอิสรภาพและเวลามากกว่าผลตอบแทนสูงสุด',
        effective_text: 'เน้นอิสรภาพและเวลามากกว่าผลตอบแทนสูงสุด',
        source_kind: 'user_stated',
        fit_rating: 'exact',
        edited_text: null,
        text_revision: 1,
        is_accepted: true,
        evidence_refs: [
          {
            answer_id: 'ans-1',
            question_id: 'q-1',
            revision: 1,
            quote: 'อิสรภาพในการเลือกใช้ชีวิตและเวลา',
            evidence_type: 'actual_experience',
          },
        ],
      },
      {
        claim_id: 'claim-2',
        text: 'ต้องการกระแสเงินสดปันผลที่สม่ำเสมอ',
        effective_text: 'ต้องการกระแสเงินสดปันผลที่สม่ำเสมอ',
        source_kind: 'ai_interpreted',
        fit_rating: null,
        edited_text: null,
        text_revision: 1,
        is_accepted: true,
        evidence_refs: [],
      },
    ],
    unresolved_topics: [],
    coverage_report: [],
    revision: 1,
  }

  it('renders statement and claims with source badges', () => {
    render(
      <EssenceResult
        summary={mockSummary}
        confirmedEssence={null}
        onRateClaim={vi.fn()}
        onEditClaim={vi.fn()}
        onExcludeClaim={vi.fn()}
        onConfirmBatch={vi.fn()}
        onRestart={vi.fn()}
        loading={false}
      />
    )

    expect(screen.getByText(/ฉันลงทุนเพื่อสร้างอิสรภาพในการใช้ชีวิต/i)).toBeInTheDocument()
    expect(screen.getByText('เน้นอิสรภาพและเวลามากกว่าผลตอบแทนสูงสุด')).toBeInTheDocument()
    expect(screen.getByText('ต้องการกระแสเงินสดปันผลที่สม่ำเสมอ')).toBeInTheDocument()
    expect(screen.getByText('คุณบอกไว้')).toBeInTheDocument()
    expect(screen.getByText('AI ตีความ')).toBeInTheDocument()
  })

  it('calls onRateClaim when rating buttons are clicked', () => {
    const onRateMock = vi.fn().mockResolvedValue(undefined)
    render(
      <EssenceResult
        summary={mockSummary}
        confirmedEssence={null}
        onRateClaim={onRateMock}
        onEditClaim={vi.fn()}
        onExcludeClaim={vi.fn()}
        onConfirmBatch={vi.fn()}
        onRestart={vi.fn()}
        loading={false}
      />
    )

    const exactBtns = screen.getAllByRole('button', { name: /ตรงกับฉัน/i })
    fireEvent.click(exactBtns[0]!)

    expect(onRateMock).toHaveBeenCalledWith('claim-1', 'exact')
  })

  it('triggers onConfirmBatch with proceedToAxis false when บันทึกไว้ก่อน is clicked', () => {
    const onConfirmMock = vi.fn().mockResolvedValue(undefined)
    render(
      <EssenceResult
        summary={mockSummary}
        confirmedEssence={null}
        onRateClaim={vi.fn()}
        onEditClaim={vi.fn()}
        onExcludeClaim={vi.fn()}
        onConfirmBatch={onConfirmMock}
        onRestart={vi.fn()}
        loading={false}
      />
    )

    const saveForNowBtn = screen.getByRole('button', { name: /บันทึกไว้ก่อน/i })
    fireEvent.click(saveForNowBtn)

    expect(onConfirmMock).toHaveBeenCalledWith(false)
  })

  it('triggers onConfirmBatch with proceedToAxis true when สร้างแกนหลักต่อ is clicked', () => {
    const onConfirmMock = vi.fn().mockResolvedValue(undefined)
    render(
      <EssenceResult
        summary={mockSummary}
        confirmedEssence={null}
        onRateClaim={vi.fn()}
        onEditClaim={vi.fn()}
        onExcludeClaim={vi.fn()}
        onConfirmBatch={onConfirmMock}
        onRestart={vi.fn()}
        loading={false}
      />
    )

    const proceedBtn = screen.getByRole('button', { name: /สร้างแกนหลักต่อ →/i })
    fireEvent.click(proceedBtn)

    expect(onConfirmMock).toHaveBeenCalledWith(true)
  })

  it('displays confirmed badge when confirmedEssence is present', () => {
    const confirmedMock: ConfirmedEssenceDTO = {
      artifact_id: 'art-123',
      scope: 'workspace',
      statement: 'ฉันลงทุนเพื่อสร้างอิสรภาพ',
      accepted_claims: [],
      unresolved_topics: [],
      confirmed_at_iso: '2026-10-08T12:00:00Z',
    }

    render(
      <EssenceResult
        summary={mockSummary}
        confirmedEssence={confirmedMock}
        onRateClaim={vi.fn()}
        onEditClaim={vi.fn()}
        onExcludeClaim={vi.fn()}
        onConfirmBatch={vi.fn()}
        onRestart={vi.fn()}
        loading={false}
      />
    )

    expect(screen.getByText('ยืนยันแล้ว')).toBeInTheDocument()
  })
})
