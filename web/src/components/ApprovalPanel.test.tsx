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

  it('แสดงรายการ YouTube Pitch และส่ง approvedPitchIds เมื่ออนุมัติ พร้อมแสดง mode, lead และ analogy', async () => {
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
          investigation_mode: 'macro',
          counter_intuitive_lead: 'ดอกเบี้ยคงที่แต่ตลาดหุ้นพุ่ง',
          analogy_generator: 'เหมือนพายุสงบก่อนจะพัดแรง',
          thumbnail_concept: 'ภาพกราฟตลาดหุ้นพุ่งขึ้นแต่กระเป๋าเงินโล่ง',
          audience_takeaway: 'เก็บเงินสดสำรอง 6 เดือนก่อนตัดสินใจลงทุนเพิ่ม',
          source_readiness: 'ready',
        },
      ],
    }
    const onApprove = vi.fn()
    render(<ApprovalPanel payload={pitchPayload} onApprove={onApprove} />)

    expect(screen.getByText('คลิปวิเคราะห์ Fed')).toBeInTheDocument()
    expect(screen.getByText('🎯 Core Thesis:')).toBeInTheDocument()
    expect(screen.getByText('🔍 Mode: macro')).toBeInTheDocument()

    expect(screen.getByText('⚡ Counter-Intuitive Lead:')).toBeInTheDocument()
    expect(screen.getByText('💡 Analogy Generator:')).toBeInTheDocument()
    expect(screen.getByText('🖼️ Thumbnail Concept:')).toBeInTheDocument()
    expect(screen.getByText('🎁 Audience Takeaway:')).toBeInTheDocument()

    await userEvent.click(screen.getByRole('checkbox'))
    const approveBtn = screen.getByRole('button', { name: /อนุมัติและสร้าง Briefing Book \(ปกติ 1, Draft 0 รายการ\)/ })
    await userEvent.click(approveBtn)

    expect(onApprove).toHaveBeenCalledWith([], [], [], ['pitch-101'], 'approve', [], {})
  })

  it('ไม่อนุญาตให้เลือก Pitch ที่ provenance ยังไม่พร้อม', async () => {
    const pitchPayload: YoutubePitchApprovalPayload = {
      type: 'youtube_pitch_approval',
      pitches: [{
        pitch_id: 'blocked-1', working_titles: ['One', 'Two', 'Three'], target_audience: 'Investor',
        core_hook: 'Hook', key_questions_to_answer: ['Q1'], research_hypotheses: [],
        source_event_ids: ['ev-1'], source_links: ['https://example.test'], source_titles: ['Source'],
        recommended_format: '15m', estimated_impact: 'High', source_readiness: 'blocked',
        source_readiness_issues: ['ไม่พบวันเผยแพร่จากหน้าแหล่งข้อมูล'],
      }],
    }
    render(<ApprovalPanel payload={pitchPayload} onApprove={() => {}} />)

    expect(screen.getByText('Source readiness: blocked')).toBeInTheDocument()
    expect(screen.getByRole('checkbox')).toBeDisabled()
  })

  it('เรียงลำดับหัวข้อคลิป YouTube ตามวันที่ (ล่าสุด/เก่าสุด/ช่อง) เมื่อรออนุมัติ', async () => {
    const payload = makePayload({
      youtube_candidates: [
        { channel: 'A Channel', title: 'คลิปเก่า', link: 'https://youtu.be/old', video_id: 'old', published: '2026-07-01', is_fetched: false },
        { channel: 'B Channel', title: 'คลิปล่าสุด', link: 'https://youtu.be/new', video_id: 'new', published: '2026-07-21', is_fetched: false },
        { channel: 'C Channel', title: 'คลิปกลาง', link: 'https://youtu.be/mid', video_id: 'mid', published: '2026-07-15', is_fetched: false },
      ],
    })
    render(<ApprovalPanel payload={payload} onApprove={() => {}} />)

    // ค่าเริ่มต้นคือ date_desc (ใหม่สุดไปเก่าสุด) -> คลิปล่าสุด ควรอยู่ก่อน คลิปกลาง และ คลิปเก่า
    const getTitlesOrder = () => screen.getAllByText(/คลิป/).map((el) => el.textContent)
    expect(getTitlesOrder()).toEqual(['คลิปล่าสุด', 'คลิปกลาง', 'คลิปเก่า'])

    // กดปุ่มเรียงตามวันที่เก่าสุด
    await userEvent.click(screen.getByRole('button', { name: 'วันที่เก่าสุด ↑' }))
    expect(getTitlesOrder()).toEqual(['คลิปเก่า', 'คลิปกลาง', 'คลิปล่าสุด'])

    // กดปุ่มเรียงตามช่อง
    await userEvent.click(screen.getByRole('button', { name: 'ช่อง' }))
    expect(getTitlesOrder()).toEqual(['คลิปเก่า', 'คลิปล่าสุด', 'คลิปกลาง'])
  })
})
