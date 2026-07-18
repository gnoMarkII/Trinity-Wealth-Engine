import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import type { NewsFunnelApprovalPayload, NewsYoutubeApprovalPayload, YoutubePitchApprovalPayload } from '../api/types'
import ApprovalPanel from './ApprovalPanel'

function makePayload(overrides: Partial<NewsYoutubeApprovalPayload> = {}): NewsYoutubeApprovalPayload {
  return {
    type: 'news_youtube_approval',
    news_candidates: [
      { title: 'ข่าว Fed คงดอกเบี้ย', link: 'https://n.example/1', source: 'Reuters', age_hours: 3, freshness_reason: '', is_stale: false, is_fetched: false },
      { title: 'ข่าวเงินเฟ้อ', link: 'https://n.example/2', source: 'Bloomberg', age_hours: 5, freshness_reason: '', is_stale: false, is_fetched: false },
    ],
    youtube_candidates: [
      { channel: 'Invest Channel', title: 'สรุปตลาดสัปดาห์นี้', link: 'https://youtu.be/abcdefghijk', video_id: 'abcdefghijk', published: '2026-07-10', is_fetched: false },
    ],
    ...overrides,
  }
}

describe('ApprovalPanel', () => {
  it('แสดงรายการข่าวและ YouTube พร้อมจำนวน', () => {
    render(<ApprovalPanel payload={makePayload()} onApprove={() => {}} />)
    expect(screen.getByText(/ข่าว \(2\)/)).toBeInTheDocument()
    expect(screen.getByText(/YouTube \(1\)/)).toBeInTheDocument()
  })

  it('เลือกรายการแล้วจำนวนบนปุ่มอนุมัติอัปเดต และ onApprove ได้ link ที่เลือก', async () => {
    const onApprove = vi.fn()
    render(<ApprovalPanel payload={makePayload()} onApprove={onApprove} />)

    await userEvent.click(screen.getByRole('checkbox', { name: /ข่าว Fed คงดอกเบี้ย/ }))
    await userEvent.click(screen.getByRole('checkbox', { name: /สรุปตลาดสัปดาห์นี้/ }))
    const approveButton = screen.getByRole('button', { name: /อนุมัติและดำเนินการต่อ \(2 รายการ\)/ })

    await userEvent.click(approveButton)
    expect(onApprove).toHaveBeenCalledWith(['https://n.example/1'], ['https://youtu.be/abcdefghijk'])
  })

  it('ปุ่ม "เลือกทั้งหมด" เลือกทุกข่าว แล้วสลับเป็น "ยกเลิกทั้งหมด"', async () => {
    render(<ApprovalPanel payload={makePayload()} onApprove={() => {}} />)
    await userEvent.click(screen.getAllByRole('button', { name: 'เลือกทั้งหมด' })[0]!)
    expect(screen.getByRole('button', { name: /\(2 รายการ\)/ })).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'ยกเลิกทั้งหมด' }))
    expect(screen.getByRole('button', { name: /\(0 รายการ\)/ })).toBeInTheDocument()
  })

  it('รายการที่อ่านแล้ว (is_fetched) ถูกซ่อนไว้ก่อน จนกดปุ่มแสดง', async () => {
    const payload = makePayload({
      news_candidates: [
        { title: 'ข่าวใหม่', link: 'https://n.example/new', source: 'Reuters', age_hours: 1, freshness_reason: '', is_stale: false, is_fetched: false },
        { title: 'ข่าวที่อ่านแล้ว', link: 'https://n.example/old', source: 'Reuters', age_hours: 30, freshness_reason: '', is_stale: false, is_fetched: true },
      ],
      youtube_candidates: [],
    })
    render(<ApprovalPanel payload={payload} onApprove={() => {}} />)
    expect(screen.queryByText('ข่าวที่อ่านแล้ว')).not.toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: /แสดงรายการที่อ่านแล้ว \(1\)/ }))
    expect(screen.getByText('ข่าวที่อ่านแล้ว')).toBeInTheDocument()
  })

  it('ระหว่าง submitting ปุ่มอนุมัติถูก disable', () => {
    render(<ApprovalPanel payload={makePayload()} onApprove={() => {}} submitting />)
    expect(screen.getByRole('button', { name: /กำลังส่ง/ })).toBeDisabled()
  })

  it('แสดงรายการ News Funnel High-Impact และส่ง approvedEventIds เมื่ออนุมัติ', async () => {
    const funnelPayload: NewsFunnelApprovalPayload = {
      type: 'news_funnel_approval',
      candidates: [
        {
          event_id: 'ev-1',
          canonical_title: 'Fed Rate Decision',
          comprehensive_summary: 'Summary text',
          macro_impact_score: 8,
          asset_impact_score: 5,
          extracted_tickers: ['NVDA'],
          extracted_themes: ['policy'],
          primary_tags: ['macro'],
          sources: ['Reuters'],
        },
      ],
    }
    const onApprove = vi.fn()
    render(<ApprovalPanel payload={funnelPayload} onApprove={onApprove} />)

    expect(screen.getByText('Fed Rate Decision')).toBeInTheDocument()
    await userEvent.click(screen.getByRole('checkbox'))
    const approveBtn = screen.getByRole('button', { name: /อนุมัติและดำเนินการต่อ \(1 รายการ\)/ })
    await userEvent.click(approveBtn)

    expect(onApprove).toHaveBeenCalledWith([], [], ['ev-1'])
  })

  it('แสดงรายการ YouTube Pitch และส่ง approvedPitchIds เมื่ออนุมัติ', async () => {
    const pitchPayload: YoutubePitchApprovalPayload = {
      type: 'youtube_pitch_approval',
      pitches: [
        {
          pitch_id: 'pitch-101',
          working_titles: ['คลิปวิเคราะห์ Fed', 'ทำไมดอกเบี้ยคงที่'],
          target_audience: 'นักลงทุนทั่วไป',
          core_hook: 'เศรษฐกิจกำลังเปลี่ยนทิศ',
          key_questions_to_answer: ['Fed จะลดดอกเบี้ยเมื่อไหร่?'],
          research_hypotheses: [],
          source_event_ids: ['ev-1'],
          source_links: ['https://example.com/1'],
          source_titles: ['ข่าว Fed'],
          recommended_format: 'Deep Dive 15 นาที',
          estimated_impact: 'High',
        },
      ],
    }
    const onApprove = vi.fn()
    render(<ApprovalPanel payload={pitchPayload} onApprove={onApprove} />)

    expect(screen.getByText('คลิปวิเคราะห์ Fed')).toBeInTheDocument()
    expect(screen.getByText('🔥 Core Hook:')).toBeInTheDocument()
    await userEvent.click(screen.getByRole('checkbox'))
    const approveBtn = screen.getByRole('button', { name: /อนุมัติและสร้าง Briefing Book \(1 รายการ\)/ })
    await userEvent.click(approveBtn)

    expect(onApprove).toHaveBeenCalledWith([], [], [], ['pitch-101'])
  })
})
