import { render, screen, fireEvent } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import EssenceInterview from './EssenceInterview'
import type { SessionResponseDTO } from '../../../api/types'

describe('EssenceInterview', () => {
  const mockSession: SessionResponseDTO = {
    session_id: 'sess-123',
    status: 'in_progress',
    revision: 1,
    active_branch_id: 'branch-main',
    questions_count: 1,
    answers_count: 0,
    is_complete: false,
    current_question: {
      question_id: 'q-1',
      sequence_no: 1,
      text: 'คุณอยากให้เงินช่วยอะไรในชีวิตมากที่สุด?',
      options: [
        { option_id: 'opt-1', option_key: 'A', text: 'ความมั่นคง ปลอดภัย ไม่ต้องกังวล' },
        { option_id: 'opt-2', option_key: 'B', text: 'อิสรภาพในการเลือกใช้ชีวิตและเวลา' },
        { option_id: 'opt-3', option_key: 'C', text: 'โอกาสเติบโตและสร้างความมั่งคั่งสูง' },
        { option_id: 'opt-4', option_key: 'D', text: 'การมีรายได้สม่ำเสมอเพื่อใช้จ่ายประจำ' },
      ],
      evidence_type: 'actual_experience',
      coverage_topics: ['life_goals'],
      is_clarification: false,
    },
    qa_history: [],
  }

  it('renders question text, 4 choices, and sequence counter', () => {
    render(
      <EssenceInterview
        session={mockSession}
        onRecordAndNext={vi.fn()}
        onPause={vi.fn()}
        loading={false}
      />
    )

    expect(screen.getByText('คำถามที่ 1 จาก 10')).toBeInTheDocument()
    expect(screen.getByText('คุณอยากให้เงินช่วยอะไรในชีวิตมากที่สุด?')).toBeInTheDocument()
    expect(screen.getByText('ความมั่นคง ปลอดภัย ไม่ต้องกังวล')).toBeInTheDocument()
    expect(screen.getByText('อิสรภาพในการเลือกใช้ชีวิตและเวลา')).toBeInTheDocument()
    expect(screen.getByText('โอกาสเติบโตและสร้างความมั่งคั่งสูง')).toBeInTheDocument()
    expect(screen.getByText('การมีรายได้สม่ำเสมอเพื่อใช้จ่ายประจำ')).toBeInTheDocument()
  })

  it('enables submit button when a choice is selected and calls onRecordAndNext', () => {
    const onRecordMock = vi.fn().mockResolvedValue(undefined)
    render(
      <EssenceInterview
        session={mockSession}
        onRecordAndNext={onRecordMock}
        onPause={vi.fn()}
        loading={false}
      />
    )

    const choiceA = screen.getByText('ความมั่นคง ปลอดภัย ไม่ต้องกังวล')
    fireEvent.click(choiceA)

    const nextBtn = screen.getByRole('button', { name: /ถัดไป →/i })
    expect(nextBtn).toBeEnabled()
    fireEvent.click(nextBtn)

    expect(onRecordMock).toHaveBeenCalledWith('choice', 'opt-1', null)
  })

  it('supports unsure option and submits with answer_kind unsure', () => {
    const onRecordMock = vi.fn().mockResolvedValue(undefined)
    render(
      <EssenceInterview
        session={mockSession}
        onRecordAndNext={onRecordMock}
        onPause={vi.fn()}
        loading={false}
      />
    )

    const unsureBtn = screen.getByRole('button', { name: /ยังไม่แน่ใจ/i })
    fireEvent.click(unsureBtn)

    const nextBtn = screen.getByRole('button', { name: /ถัดไป →/i })
    expect(nextBtn).toBeEnabled()
    fireEvent.click(nextBtn)

    expect(onRecordMock).toHaveBeenCalledWith('unsure', null, null)
  })

  it('displays ดูแก่นแท้ของฉัน for question 10', () => {
    const q10Session: SessionResponseDTO = {
      ...mockSession,
      current_question: {
        ...mockSession.current_question!,
        sequence_no: 10,
        text: 'ข้อสุดท้าย: สิ่งที่คุณจะไม่ยอมแลกเด็ดขาดคืออะไร?',
      },
    }

    render(
      <EssenceInterview
        session={q10Session}
        onRecordAndNext={vi.fn()}
        onPause={vi.fn()}
        loading={false}
      />
    )

    expect(screen.getByText('คำถามที่ 10 จาก 10')).toBeInTheDocument()
    expect(screen.getByText(/ดูแก่นแท้ของฉัน/i)).toBeInTheDocument()
  })
})
