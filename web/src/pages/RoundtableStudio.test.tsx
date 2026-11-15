import { act, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { MemoryRouter } from 'react-router-dom'
import RoundtableStudio from './RoundtableStudio'
import { api, streamRoundtableDebate } from '../api/client'
import { formatRubricTextForUi } from '../components/roundtable/personas'
import type { RoundtableGuruConfigCatalogDTO, RoundtableSessionDetailDTO } from '../api/types'

vi.mock('../api/client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../api/client')>()
  return {
    ...actual,
    api: {
      ...actual.api,
      getRoundtableGuruProfiles: vi.fn(),
      updateRoundtableGuruProfiles: vi.fn(),
      resetRoundtableGuruProfiles: vi.fn(),
      previewRoundtableVoice: vi.fn(),
      listRoundtableSessions: vi.fn(),
      createRoundtableSession: vi.fn(),
      getRoundtableSession: vi.fn(),
      interjectRoundtableSession: vi.fn(),
      saveRoundtableToVault: vi.fn(),
      updateThesisFromRoundtable: vi.fn(),
      dispatchRoundtablePrerequisite: vi.fn(),
      downloadRoundtableYoutubePack: vi.fn(),
    },
    streamRoundtableDebate: vi.fn(),
  }
})

function makeSampleCatalog(): RoundtableGuruConfigCatalogDTO {
  return {
    personas: [
      {
        speaker_id: 'hermes_moderator',
        display_name: 'Hermes (Moderator)',
        short_name: 'Hermes',
        archetype: 'Moderator & Portfolio Distiller',
        voice_name: 'Kore',
        accent_color: '#0ea5e9',
        is_prioritized_role_model: false,
        emoji: '🪶',
        pitch: 1.0,
        rate: 1.04,
        enabled: true,
        debate_order: 0,
        preset_id: 'default',
        speaking_style: 'กลาง ชัดเจน ร้อยเรียงประเด็น',
        custom_directive: '',
      },
      {
        speaker_id: 'value_guru',
        display_name: 'Value & MoS Guru',
        short_name: 'Value',
        archetype: 'Buffett / Graham Value Investor',
        voice_name: 'Charon',
        accent_color: '#10b981',
        is_prioritized_role_model: false,
        emoji: '🏛️',
        pitch: 0.9,
        rate: 0.96,
        enabled: true,
        debate_order: 1,
        preset_id: 'default',
        speaking_style: 'สุขุม รอบคอบ ชั่งน้ำหนักราคาเทียบมูลค่า',
        custom_directive: '',
      },
      {
        speaker_id: 'quality_guru',
        display_name: 'Quality & Moat Guru',
        short_name: 'Quality',
        archetype: 'Munger / Terry Smith Compounder',
        voice_name: 'Fenrir',
        accent_color: '#6366f1',
        is_prioritized_role_model: false,
        emoji: '💎',
        pitch: 0.95,
        rate: 1.0,
        enabled: true,
        debate_order: 2,
        preset_id: 'default',
        speaking_style: 'เฉียบคม ดู ROIC และคูเมืองธุรกิจ',
        custom_directive: '',
      },
      {
        speaker_id: 'growth_guru',
        display_name: 'Growth & Expectations Guru',
        short_name: 'Growth',
        archetype: 'Peter Lynch / Fisher Growth',
        voice_name: 'Puck',
        accent_color: '#f59e0b',
        is_prioritized_role_model: false,
        emoji: '🚀',
        pitch: 1.08,
        rate: 1.06,
        enabled: true,
        debate_order: 3,
        preset_id: 'default',
        speaking_style: 'มีพลัง มองศักยภาพเติบโต',
        custom_directive: '',
      },
      {
        speaker_id: 'macro_tactical_guru',
        display_name: 'Macro & Tactical Guru',
        short_name: 'Macro',
        archetype: 'Howard Marks / Druckenmiller Cycle',
        voice_name: 'Aoede',
        accent_color: '#ec4899',
        is_prioritized_role_model: false,
        emoji: '🌊',
        pitch: 1.04,
        rate: 1.02,
        enabled: true,
        debate_order: 4,
        preset_id: 'default',
        speaking_style: 'อ่านวัฏจักรตลาดและสภาพคล่อง',
        custom_directive: '',
      },
      {
        speaker_id: 'skeptic_guru',
        display_name: "Skeptic / Devil's Advocate",
        short_name: 'Skeptic',
        archetype: 'Chanos / Burry Risk & Falsification',
        voice_name: 'Orus',
        accent_color: '#ef4444',
        is_prioritized_role_model: false,
        emoji: '🛡️',
        pitch: 0.86,
        rate: 0.98,
        enabled: true,
        debate_order: 5,
        preset_id: 'default',
        speaking_style: 'เข้ม ตรงไปตรงมา ตรวจจับความเสี่ยง',
        custom_directive: '',
      },
    ],
    voices: [
      {
        voice_name: 'Charon',
        gender_tone: 'male',
        label_th: 'ทุ้มนุ่ม สุขุม อบอุ่นแบบผู้ใหญ่ — จังหวะพูดเนิบช้า มีน้ำหนัก',
        recommended_for: 'สายคุณค่า (Value Guru)',
        default_pitch: 0.9,
        default_rate: 0.96,
        default_sample_text: 'การลงทุนที่ดีต้องเริ่มจากส่วนเผื่อความปลอดภัยเสมอครับ',
      },
      {
        voice_name: 'Zephyr',
        gender_tone: 'female',
        label_th: 'สดใส ทันสมัย คล่องตัว เข้าใจง่าย — โทนเสียงสว่าง เป็นกันเอง',
        recommended_for: 'สาย Growth / Tech Disruptor หรือผู้ดำเนินรายการ',
        default_pitch: 1.05,
        default_rate: 1.05,
        default_sample_text: 'ลองมาเจาะลึกเทรนด์รายได้รอบนี้กันค่ะ',
      },
      {
        voice_name: 'Kore',
        gender_tone: 'female',
        label_th: 'สุขุม สมดุล เป็นมืออาชีพ — น้ำเสียงนิ่ง คมชัด น่าเชื่อถือ',
        recommended_for: 'ผู้ดำเนินรายการ (Hermes)',
        default_pitch: 1.0,
        default_rate: 1.04,
        default_sample_text: 'สวัสดีค่ะ ดิฉันเฮอร์เมสค่ะ',
      },
    ],
    presets: [
      {
        preset_id: 'default',
        speaker_id: 'value_guru',
        label_th: 'Buffett & Graham Value (ค่าเริ่มต้น)',
        display_name: 'Value & MoS Guru',
        short_name: 'Value',
        archetype: 'Buffett / Graham Value Investor',
        speaking_style: 'สุขุม รอบคอบ ชั่งน้ำหนักราคาเทียบมูลค่า',
        custom_directive: '',
        recommended_voice: 'Charon',
        emoji: '🏛️',
      },
      {
        preset_id: 'value_graham_deep',
        speaker_id: 'value_guru',
        label_th: 'Benjamin Graham Deep Value',
        display_name: 'Deep Value (Graham)',
        short_name: 'Graham',
        archetype: 'Benjamin Graham Deep Value Contrarian',
        speaking_style: 'เน้นตัวเลขงบดุลและส่วนเผื่อความปลอดภัยเข้มงวด',
        custom_directive: 'ตั้งเกณฑ์ MoS อย่างน้อย 25%',
        recommended_voice: 'Charon',
        emoji: '📐',
        role_lens_label: 'มูลค่า & Margin of Safety',
        focus_summary: 'เน้นส่วนเผื่อความปลอดภัยเข้มงวดและงบดุลแข็งแกร่ง',
        primary_evidence_labels: ['Current Ratio & Debt', 'FCF Yield vs US10Y'],
        analytical_rubric:
          '1) Priority Metrics: fcf_yield_pct เทียบ US10Y, current_ratio, net_debt_ebitda.\n' +
          '2) Decision Thresholds (Red Lines): Margin of Safety >= 25–30%, Current Ratio >= 1.5x.',
      },
    ],
  }
}

function makeSampleDetail(overrides: Partial<RoundtableSessionDetailDTO> = {}): RoundtableSessionDetailDTO {
  return {
    session: {
      session_id: 'rt_test_001',
      title: 'ประชุมกูรู $NVDA',
      primary_ticker: 'NVDA',
      benchmark_ticker: 'SPY',
      agenda_prompt: 'ประชุมสภากูรูวิเคราะห์หุ้น NVDA',
      context_refs: ['$NVDA', '@Portfolio'],
      final_rating: 'REVIEW',
      settlement_status: 'pending',
      settled_return_pct: null,
      settled_alpha_pct: null,
      status: 'active',
      created_at: 1730000000,
      updated_at: 1730000000,
    },
    turns: [],
    decision_playbook: null,
    past_scorecard: {
      previous_session_id: 'rt_prev_999',
      previous_date: '2026-10-01',
      primary_ticker: 'NVDA',
      benchmark_ticker: 'SPY',
      entry_price: '120.00',
      settlement_price: '132.00',
      raw_return_pct: '10.00',
      benchmark_return_pct: '2.50',
      alpha_pct: '7.50',
      previous_final_rating: 'OVERWEIGHT',
      outcome_label: 'outperformed',
      reflection_lesson: 'รอซื้อช่วงย่อได้เปรียบกว่าไล่ราคา',
      settlement_status: 'settled',
    },
    missing_prerequisite_tickers: [],
    prioritized_speakers: ['value_guru'],
    user_role_models: ['Buffett'],
    portfolio_attachment_state: 'ATTACHED_HELD',
    context_drift_detected: false,
    packet_fingerprint_sha256: 'hash123',
    personas: [
      {
        speaker_id: 'hermes_moderator',
        display_name: 'Hermes (Moderator)',
        short_name: 'Hermes',
        archetype: 'Moderator & Portfolio Distiller',
        voice_name: 'Kore',
        accent_color: '#0ea5e9',
        is_prioritized_role_model: false,
      },
      {
        speaker_id: 'value_guru',
        display_name: 'Value & MoS Guru',
        short_name: 'Value',
        archetype: 'Buffett / Graham Value Investor',
        voice_name: 'Charon',
        accent_color: '#10b981',
        is_prioritized_role_model: true,
      },
    ],
    ...overrides,
  }
}

describe('RoundtableStudio Page', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    window.localStorage.clear()
    vi.mocked(api.listRoundtableSessions).mockResolvedValue([])
    vi.mocked(api.getRoundtableGuruProfiles).mockResolvedValue(makeSampleCatalog())
  })

  it('prefills agenda and context chips from ?ticker=NVDA URL parameter', async () => {
    render(
      <MemoryRouter initialEntries={['/roundtable?ticker=NVDA']}>
        <RoundtableStudio />
      </MemoryRouter>,
    )

    expect(screen.getByText('🎙️ Trinity Guru Roundtable Studio')).toBeInTheDocument()
    const textarea = screen.getByPlaceholderText(/พิมพ์โจทย์ให้สภากูรู/i) as HTMLTextAreaElement
    await waitFor(() => {
      expect(textarea.value).toContain('NVDA')
    })
    expect(screen.getByRole('button', { name: /\$NVDA/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /@Portfolio/i })).toBeInTheDocument()
  })

  it('creates session, streams guru dialogue + audio + playbook, and supports Vault/Thesis export', async () => {
    const user = userEvent.setup()
    vi.mocked(api.createRoundtableSession).mockResolvedValue(makeSampleDetail())
    vi.mocked(streamRoundtableDebate).mockImplementation(async (_sessionId, _opts, cb) => {
      cb.onSpeakerStarted?.({ speaker_id: 'value_guru', round_stage: 'opening' })
      cb.onTurnCompleted?.({
        turn_id: 'turn_val_1',
        session_id: 'rt_test_001',
        seq: 1,
        speaker_id: 'value_guru',
        round_stage: 'opening',
        replying_to: null,
        spoken_dialogue: 'จากมุมมองแวลู NVDA มีความสามารถทำกำไรสูงมากแต่ต้องดู Margin of Safety ประกอบด้วย',
        evidence_bullets: ['[FACT] ROIC สูงกว่าค่าเฉลี่ยกลุ่ม', '[MATH] FCF Yield อยู่ที่ 3.1%'],
        stance_tag: 'HOLD',
        audio_cache_key: null,
        audio_status: 'fallback_web_speech',
        created_at: 1730000001,
      })
      cb.onAudioReady?.({
        turn_id: 'turn_val_1',
        speaker_id: 'value_guru',
        audio_cache_key: 'wav_val_1',
        audio_status: 'ready',
      })
      cb.onPlaybookUpdated?.({
        final_rating: 'OVERWEIGHT',
        action_stance: 'OVERWEIGHT',
        core_conviction_score: 2,
        executive_verdict: 'ทยอยสะสมอย่างมีวินัยเมื่อราคาอ่อนตัว',
        guru_stance_breakdown: { value_guru: 'HOLD', growth_guru: 'BUY' },
        consensus_points: ['รายได้ Data Center เติบโตแข็งแกร่ง'],
        tension_axes: [
          {
            topic: 'Valuation',
            bull_side_speaker: 'growth_guru',
            bull_argument: 'PEG ยังรับได้เมื่อเทียบอัตราโต',
            bear_side_speaker: 'value_guru',
            bear_argument: 'MoS แคบหากวงจรชิปชะลอ',
          },
        ],
        personal_axis_alignment: [
          { dimension: 'Conviction Fit', status: 'aligned', detail: 'สอดคล้องกับพอร์ตหลัก' },
        ],
        portfolio_context_summary: 'ถือครองอยู่แล้วในพอร์ต 8%',
        action_playbook: {
          primary_action: {
            option_id: 'opt_1',
            label: 'Staged Dip Accumulation',
            speaker_source: 'hermes_moderator',
            conviction_level: 2,
            action_steps: ['แบ่งไม้ซื้อเมื่อย่อใกล้ 125 USD'],
            entry_zone: '122 - 128 USD',
            sizing_note: 'ไม่เกิน 10% ของพอร์ต',
          },
          alternatives: [
            {
              option_id: 'opt_2',
              label: 'Hold Existing Allocation',
              speaker_source: 'value_guru',
              conviction_level: 1,
              action_steps: ['ถือต่อไม่ไล่ราคา'],
            },
          ],
        },
        kill_switches: ['Gross Margin หลุด 70%'],
      })
      cb.onDone?.({ session_id: 'rt_test_001', status: 'completed' })
    })
    vi.mocked(api.saveRoundtableToVault).mockResolvedValue({
      ok: true,
      vault_path: 'Roundtable/NVDA.md',
      thesis_claim_id: null,
      message: 'บันทึกลง Obsidian Vault สำเร็จ',
    })
    vi.mocked(api.updateThesisFromRoundtable).mockResolvedValue({
      ok: true,
      claim_id: 'claim_123',
    })

    render(
      <MemoryRouter initialEntries={['/roundtable?ticker=NVDA']}>
        <RoundtableStudio />
      </MemoryRouter>,
    )

    await user.click(screen.getByRole('button', { name: /เริ่มประชุมกูรู/i }))

    await waitFor(() => {
      expect(api.createRoundtableSession).toHaveBeenCalled()
      expect(
        screen.getAllByText(/NVDA มีความสามารถทำกำไรสูงมาก/).length,
      ).toBeGreaterThan(0)
    })

    // Prior closed-loop scorecard & Hermes Action Playbook rendered
    expect(screen.getByText(/Closed-Loop Scorecard/i)).toBeInTheDocument()
    expect(screen.getByText('ทยอยสะสมอย่างมีวินัยเมื่อราคาอ่อนตัว')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Accept staged Dip Accumulation/i })).toBeInTheDocument()

    // Expand evidence bullets drawer
    await user.click(screen.getByRole('button', { name: /หลักฐานประกอบ \(2 ข้อ\)/i }))
    expect(screen.getByText('ROIC สูงกว่าค่าเฉลี่ยกลุ่ม')).toBeInTheDocument()

    // Save to Obsidian Vault & Update Active Thesis
    await user.click(screen.getByRole('button', { name: /Save to Vault/i }))
    await waitFor(() => {
      expect(api.saveRoundtableToVault).toHaveBeenCalledWith('rt_test_001', {})
      expect(screen.getByText(/Roundtable\/NVDA\.md/)).toBeInTheDocument()
    })

    await user.click(screen.getByRole('button', { name: /Update Thesis/i }))
    await waitFor(() => {
      expect(api.updateThesisFromRoundtable).toHaveBeenCalledWith('rt_test_001', {})
    })
  })

  it('shows DecisionInbox prerequisite guard and supports sandbox math interjection', async () => {
    const user = userEvent.setup()
    vi.mocked(api.createRoundtableSession).mockResolvedValue(
      makeSampleDetail({
        missing_prerequisite_tickers: ['XYZ'],
      }),
    )
    vi.mocked(api.dispatchRoundtablePrerequisite).mockResolvedValue({
      ok: true,
      ticker: 'XYZ',
      job_id: 'job_xyz_99',
      message: 'ส่งคำสั่งวิเคราะห์หุ้น XYZ เข้าคิวเรียบร้อย',
    })
    vi.mocked(api.interjectRoundtableSession).mockResolvedValue({
      session_id: 'rt_test_001',
      user_turn: {
        turn_id: 'turn_u_1',
        session_id: 'rt_test_001',
        seq: 1,
        speaker_id: 'user',
        round_stage: 'user_interjection',
        replying_to: 'value_guru',
        spoken_dialogue: '@value ถ้าซื้อเพิ่มอีก 20 หุ้นที่ 110 ต้นทุนเฉลี่ยจะเท่าไร',
        evidence_bullets: [],
        stance_tag: 'REVIEW',
        audio_cache_key: null,
        audio_status: 'disabled',
        created_at: 1730000010,
      },
      reply_turn: {
        turn_id: 'turn_r_1',
        session_id: 'rt_test_001',
        seq: 2,
        speaker_id: 'value_guru',
        round_stage: 'interjection_reply',
        replying_to: 'user',
        spoken_dialogue: 'คำนวณด้วย Python Sandbox แล้ว ต้นทุนเฉลี่ยใหม่จะอยู่ที่ 118.33 ดอลลาร์ครับ',
        evidence_bullets: ['[MATH] blended_avg_cost = 118.33 USD'],
        stance_tag: 'HOLD',
        audio_cache_key: null,
        audio_status: 'fallback_web_speech',
        created_at: 1730000011,
      },
      tool_calls: [
        {
          tool_name: 'python_sandbox_decimal_calc',
          status: 'completed',
          elapsed_ms: 45,
          summary: 'คำนวณจำลองตัวเลขด้วย Python Decimal สำเร็จ',
          detail_json: { blended_avg_cost: '118.33' },
        },
      ],
      context_refreshed: false,
    })

    render(
      <MemoryRouter initialEntries={['/roundtable?ticker=XYZ']}>
        <RoundtableStudio />
      </MemoryRouter>,
    )

    await user.click(screen.getByRole('button', { name: /เริ่มประชุมกูรู/i }))

    // Prerequisite DecisionInbox appears instead of immediately streaming
    await waitFor(() => {
      expect(screen.getByText(/หุ้น \$XYZ ยังไม่มีบทวิเคราะห์เจาะลึกในระบบ/)).toBeInTheDocument()
    })
    expect(streamRoundtableDebate).not.toHaveBeenCalled()

    await user.click(screen.getByRole('button', { name: /ส่งวิเคราะห์ \$XYZ/i }))
    await waitFor(() => {
      expect(api.dispatchRoundtablePrerequisite).toHaveBeenCalledWith('XYZ')
    })

    // Raise-hand interjection with Sandbox Decimal toggle
    const mathToggle = screen.getByLabelText(/คำนวณ Decimal/i)
    await user.click(mathToggle)
    const interjectInput = screen.getByLabelText(/ยกมือแทรกถามสภากูรู/i)
    await user.type(interjectInput, 'ถ้าซื้อเพิ่มอีก 20 หุ้นที่ 110 ต้นทุนเฉลี่ยจะเท่าไร')
    await user.click(screen.getByRole('button', { name: /ส่งคำถาม/i }))

    await waitFor(() => {
      expect(api.interjectRoundtableSession).toHaveBeenCalledWith(
        'rt_test_001',
        expect.objectContaining({ run_sandbox_math: true }),
      )
      expect(
        screen.getAllByText(/ต้นทุนเฉลี่ยใหม่จะอยู่ที่ 118\.33 ดอลลาร์ครับ/).length,
      ).toBeGreaterThan(0)
      expect(screen.getByText('python_sandbox_decimal_calc')).toBeInTheDocument()
    })
  })

  it('opens Guru Customization Studio, displays voice descriptions, previews voice, and saves defaults', async () => {
    const user = userEvent.setup()
    const baseCatalog = makeSampleCatalog()
    const updatedCatalog: RoundtableGuruConfigCatalogDTO = {
      ...baseCatalog,
      personas: baseCatalog.personas.map((p) =>
        p.speaker_id === 'value_guru'
          ? { ...p, display_name: 'Deep Value (Graham)', short_name: 'Graham', voice_name: 'Zephyr' }
          : p,
      ),
    }
    vi.mocked(api.previewRoundtableVoice).mockResolvedValue({
      audio_status: 'fallback_web_speech',
      audio_cache_key: null,
      sample_text: 'ลองมาเจาะลึกเทรนด์รายได้รอบนี้กันค่ะ',
      voice_name: 'Zephyr',
      pitch: 1.05,
      rate: 1.05,
    })
    vi.mocked(api.updateRoundtableGuruProfiles).mockResolvedValue(updatedCatalog)

    render(
      <MemoryRouter initialEntries={['/roundtable']}>
        <RoundtableStudio />
      </MemoryRouter>,
    )

    const customizeBtn = await screen.findByRole('button', { name: /ปรับแต่งกูรู & เสียง/i })
    await user.click(customizeBtn)

    const dialog = await screen.findByRole('dialog')
    const modal = within(dialog)

    // Voice personality description in Thai is visible
    await waitFor(() => {
      expect(modal.getByText(/สดใส ทันสมัย คล่องตัว เข้าใจง่าย/i)).toBeInTheDocument()
    })

    // Apply Graham preset
    await user.click(modal.getByRole('button', { name: /Benjamin Graham Deep Value/i }))
    expect(modal.getByDisplayValue('Deep Value (Graham)')).toBeInTheDocument()
    expect(modal.getByText(/Current Ratio >= 1\.5x/)).toBeInTheDocument()

    // Select Zephyr voice & preview it
    await user.click(modal.getByRole('button', { name: /Zephyr/i }))
    await user.click(modal.getByRole('button', { name: /ทดลองฟังเสียงตัวอย่าง/i }))
    await waitFor(() => {
      expect(api.previewRoundtableVoice).toHaveBeenCalledWith(
        expect.objectContaining({
          speaker_id: 'value_guru',
          voice_name: 'Zephyr',
        }),
      )
    })

    // Save defaults globally
    await user.click(modal.getByRole('button', { name: /บันทึกเป็นค่าเริ่มต้น/i }))
    await waitFor(() => {
      expect(api.updateRoundtableGuruProfiles).toHaveBeenCalled()
      expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
      expect(screen.getByText(/บันทึกการตั้งค่ากูรูและเสียงพากย์เป็นค่าเริ่มต้นเรียบร้อย/i)).toBeInTheDocument()
    })
  })

  it('displays clear per-guru role lenses on stage and lets users inspect full non-overlapping context & rubrics', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter initialEntries={['/roundtable?ticker=NVDA']}>
        <RoundtableStudio />
      </MemoryRouter>,
    )

    const stage = await screen.findByRole('region', { name: /เวทีสภากูรูการลงทุน/i })
    const stageScope = within(stage)

    // Stage seat cards immediately clarify each guru's analytical lens
    expect(stageScope.getByText('มูลค่า & Margin of Safety')).toBeInTheDocument()
    expect(stageScope.getByText('คูเมือง, ROIC & คุณภาพงบ')).toBeInTheDocument()
    expect(stageScope.getByText('การเติบโต & Expectation Gap')).toBeInTheDocument()
    expect(stageScope.getByText('มหภาค, Flow & จังหวะเข้าทำ')).toBeInTheDocument()
    expect(stageScope.getByText('จับผิดงบ & Kill-Switch')).toBeInTheDocument()

    // Toggle full 6-guru comparison guide
    await user.click(stageScope.getByRole('button', { name: /ดูกรอบคิดกูรู 6 สาย/i }))
    expect(stageScope.getByText(/Expectation Gap ระหว่างอัตราโตจริง/)).toBeInTheDocument()
    expect(stageScope.getAllByText(/1\) Priority Metrics:/)).toHaveLength(6)

    // Inspect specific guru context via seat info button
    await user.click(stageScope.getByRole('button', { name: /ดูบริบท Skeptic/i }))
    expect(stageScope.getByText(/Pre-Mortem หาจุดแตกหักของ Thesis/)).toBeInTheDocument()
    expect(stageScope.getAllByText(/1\) Priority Metrics:/)).toHaveLength(1)
    expect(stageScope.queryByText(/Expectation Gap ระหว่างอัตราโตจริง/)).not.toBeInTheDocument()
  })

  it('enforces minimum 2 panel gurus in quick toggle chips and sends persona_overrides on session create', async () => {
    const user = userEvent.setup()
    vi.mocked(api.createRoundtableSession).mockResolvedValue(makeSampleDetail())
    vi.mocked(streamRoundtableDebate).mockResolvedValue()

    render(
      <MemoryRouter initialEntries={['/roundtable?ticker=NVDA']}>
        <RoundtableStudio />
      </MemoryRouter>,
    )

    // Quick toggle chips appear in new-agenda mode
    const composerSection = await screen.findByRole('region', {
      name: /ตั้งวาระเปิดประชุมสภากูรูใหม่/i,
    })
    const composer = within(composerSection)
    const skepticChip = composer.getByRole('button', { name: /🛡️ Skeptic/i })
    await user.click(skepticChip)
    expect(composer.getByText(/ปิด Skeptic/i)).toBeInTheDocument()

    // Disable Macro & Growth -> only Value + Quality remain enabled
    await user.click(composer.getByRole('button', { name: /🌊 Macro/i }))
    await user.click(composer.getByRole('button', { name: /🚀 Growth/i }))

    const valueChip = composer.getByRole('button', { name: /🏛️ Value/i })
    const qualityChip = composer.getByRole('button', { name: /💎 Quality/i })
    expect(valueChip).toBeDisabled()
    expect(qualityChip).toBeDisabled()

    await user.click(composer.getByRole('button', { name: /เริ่มประชุมกูรู/i }))

    await waitFor(() => {
      expect(api.createRoundtableSession).toHaveBeenCalledWith(
        expect.objectContaining({
          debate_intensity: 'crossfire',
          debate_depth: 'deep_youtube',
          persona_overrides: expect.arrayContaining([
            expect.objectContaining({ speaker_id: 'hermes_moderator', enabled: true }),
            expect.objectContaining({ speaker_id: 'skeptic_guru', enabled: false }),
            expect.objectContaining({ speaker_id: 'macro_tactical_guru', enabled: false }),
            expect.objectContaining({ speaker_id: 'growth_guru', enabled: false }),
            expect.objectContaining({ speaker_id: 'value_guru', enabled: true }),
            expect.objectContaining({ speaker_id: 'quality_guru', enabled: true }),
          ]),
        }),
      )
    })
  })

  it('allows switching debate intensity/depth pills and exporting YouTube Production Pack', async () => {
    const user = userEvent.setup()
    vi.mocked(api.createRoundtableSession).mockResolvedValue(
      makeSampleDetail({ debate_intensity: 'heated', debate_depth: 'concise' }),
    )
    vi.mocked(streamRoundtableDebate).mockImplementation(async (_sessionId, _opts, cb) => {
      cb.onTurnCompleted?.({
        turn_id: 'turn_cf_1',
        session_id: 'rt_test_001',
        seq: 1,
        speaker_id: 'skeptic_guru',
        round_stage: 'crossfire',
        replying_to: 'growth_guru',
        spoken_dialogue: 'ถ้า Capex ชะลอตัว Margin จะกดดันทันทีครับ',
        evidence_bullets: ['[FACT] Ref risk'],
        stance_tag: 'UNDERWEIGHT',
        sizing_posture: 'Defensive Trim',
        audio_cache_key: null,
        audio_status: 'fallback_web_speech',
        created_at: 1730000002,
      })
      cb.onDone?.({ session_id: 'rt_test_001', status: 'completed' })
    })
    vi.mocked(api.downloadRoundtableYoutubePack).mockResolvedValue({
      blob: new Blob(['zip'], { type: 'application/zip' }),
      filename: 'roundtable_youtube_NVDA_rt_test_001.zip',
    })
    const createObjUrlSpy = vi.fn(() => 'blob:mock-yt-pack')
    const revokeObjUrlSpy = vi.fn()
    Object.defineProperty(URL, 'createObjectURL', { configurable: true, value: createObjUrlSpy })
    Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, value: revokeObjUrlSpy })

    render(
      <MemoryRouter initialEntries={['/roundtable?ticker=NVDA']}>
        <RoundtableStudio />
      </MemoryRouter>,
    )

    const composerSection = await screen.findByRole('region', {
      name: /ตั้งวาระเปิดประชุมสภากูรูใหม่/i,
    })
    const composer = within(composerSection)
    expect(composer.getByRole('button', { name: /⚔️ ปะทะเดือด/i })).toHaveAttribute('aria-pressed', 'true')
    expect(composer.getByRole('button', { name: /🎬 YouTube ยาวสุด/i })).toHaveAttribute('aria-pressed', 'true')

    await user.click(composer.getByRole('button', { name: /🔥 เข้มข้น/i }))
    await user.click(composer.getByRole('button', { name: /⚡ กระชับ/i }))
    await user.click(composer.getByRole('button', { name: /เริ่มประชุมกูรู/i }))

    await waitFor(() => {
      expect(api.createRoundtableSession).toHaveBeenCalledWith(
        expect.objectContaining({
          debate_intensity: 'heated',
          debate_depth: 'concise',
        }),
      )
      expect(screen.getByText(/Crossfire · ปะทะเดือดตัวต่อตัว/i)).toBeInTheDocument()
      expect(screen.getByText(/⚔️ ปะทะ @Growth/i)).toBeInTheDocument()
    })

    await user.click(screen.getByRole('button', { name: /🎬 Export ชุด YouTube/i }))
    await waitFor(() => {
      expect(api.downloadRoundtableYoutubePack).toHaveBeenCalledWith('rt_test_001')
      expect(screen.getByText(/roundtable_youtube_NVDA_rt_test_001\.zip/)).toBeInTheDocument()
    })
  })

  it('humanizes rubric snake_case tokens while preserving numbers and renders preset pill, disabled seats warning, 5-pillar coverage, and clickable speaker badges', async () => {
    const formatted = formatRubricTextForUi(
      '1) Priority Metrics: ocf_to_net_income, sloan_accrual, piotroski_f_score, net_debt_ebitda, eps_revision_net_30d. Current Ratio >= 1.5x.',
    )
    expect(formatted).toContain('1) Priority Metrics:')
    expect(formatted).toContain('Current Ratio >= 1.5x')
    expect(formatted).toContain('OCF/กำไรสุทธิ')
    expect(formatted).toContain('Sloan Accrual')
    expect(formatted).toContain('Piotroski F-Score')
    expect(formatted).toContain('หนี้สุทธิ/EBITDA')
    expect(formatted).toContain('การปรับประมาณการ EPS 30 วัน')

    const scrollIntoViewMock = vi.fn()
    Object.defineProperty(HTMLElement.prototype, 'scrollIntoView', {
      configurable: true,
      value: scrollIntoViewMock,
    })

    const baseCatalog = makeSampleCatalog()
    const customPersonas = baseCatalog.personas.map((p) => {
      if (p.speaker_id === 'quality_guru') {
        return {
          ...p,
          preset_id: 'quality_forensic_auditor',
          preset_label: 'Forensic Quality Auditor',
        }
      }
      if (p.speaker_id === 'macro_tactical_guru') {
        return { ...p, enabled: false }
      }
      return p
    })
    vi.mocked(api.getRoundtableGuruProfiles).mockResolvedValue({
      ...baseCatalog,
      personas: customPersonas,
    })
    vi.mocked(api.createRoundtableSession).mockResolvedValue(
      makeSampleDetail({ personas: customPersonas }),
    )
    vi.mocked(streamRoundtableDebate).mockImplementation(async (_sid, _opts, cb) => {
      cb.onTurnCompleted?.({
        turn_id: 'turn_val_jump',
        session_id: 'rt_test_001',
        seq: 1,
        speaker_id: 'value_guru',
        round_stage: 'opening',
        replying_to: null,
        spoken_dialogue: 'จากมุมมองแวลู NVDA ราคาปัจจุบันเผื่อส่วนต่างความปลอดภัยน้อยครับ',
        evidence_bullets: ['[MATH] FCF Yield 2.4%'],
        stance_tag: 'HOLD',
        audio_cache_key: null,
        audio_status: 'disabled',
        created_at: 1730000001,
      })
      cb.onPlaybookUpdated?.({
        final_rating: 'HOLD',
        action_stance: 'HOLD',
        core_conviction_score: 0,
        executive_verdict: 'ถือรอดูจังหวะและตรวจสอบ 5 มิติครบถ้วน',
        guru_stance_breakdown: { value_guru: 'HOLD', growth_guru: 'BUY' },
        consensus_points: ['รายได้เติบโตสูง'],
        tension_axes: [
          {
            topic: 'Valuation Multiple',
            bull_side_speaker: 'growth_guru',
            bull_argument: 'EPS revision บวกหนุน',
            bear_side_speaker: 'value_guru',
            bear_argument: 'MoS แคบเกินเกณฑ์',
          },
        ],
        personal_axis_alignment: [],
        portfolio_context_summary: 'ไม่ได้แนบพอร์ต',
        action_playbook: {
          primary_action: {
            option_id: 'opt_hold',
            label: 'Wait For Dip',
            speaker_source: 'value_guru',
            conviction_level: 1,
            action_steps: ['รอดูงบไตรมาสถัดไป'],
          },
          alternatives: [],
        },
        kill_switches: ['Gross Margin < 70%'],
      })
      cb.onPillarCoverage?.({
        raised_count: 4,
        challenged_count: 2,
        unchallenged_pillars: ['Quality & Moat', 'Macro & Tactical', 'Risks & Balance'],
        pillars: [
          { pillar: 'Valuation', raised_by: ['value_guru'], challenged_by: ['growth_guru'], is_challenged: true },
          { pillar: 'Quality & Moat', raised_by: ['quality_guru'], challenged_by: [], is_challenged: false },
          { pillar: 'Growth & Guidance', raised_by: ['growth_guru'], challenged_by: ['value_guru'], is_challenged: true },
          { pillar: 'Macro & Tactical', raised_by: ['hermes_moderator'], challenged_by: [], is_challenged: false },
          { pillar: 'Risks & Balance', raised_by: ['skeptic_guru'], challenged_by: [], is_challenged: false },
        ],
      })
      cb.onDone?.({ session_id: 'rt_test_001', status: 'completed' })
    })

    const user = userEvent.setup()
    render(
      <MemoryRouter initialEntries={['/roundtable?ticker=NVDA']}>
        <RoundtableStudio />
      </MemoryRouter>,
    )

    // Stage preset badge + RaiseHand disabled seats warning rendered right from catalog
    expect(await screen.findByText(/Forensic Quality Auditor/i)).toBeInTheDocument()
    expect(screen.getByText(/⚠️ ปิด Macro/)).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /เริ่มประชุมกูรู/i }))

    // 5-Pillar coverage bar rendered in DecisionPlaybookPanel
    await waitFor(() => {
      expect(screen.getByText(/🧭 5 มิติความครบถ้วน/)).toBeInTheDocument()
      expect(screen.getByText(/เปิดประเด็น 4\/5 · ซักค้าน 2\/5/)).toBeInTheDocument()
    })

    // Tensions tab replaces raw speaker_id with emoji + shortName badge & click scrolls to turn
    await user.click(screen.getByRole('tab', { name: /ข้อขัดแย้ง/i }))
    expect(screen.getByText(/🚀 Growth/)).toBeInTheDocument()
    const jumpBtn = screen.getByRole('button', { name: /มุมมองแย้ง \(🏛️ Value\)/i })
    await user.click(jumpBtn)
    expect(scrollIntoViewMock).toHaveBeenCalled()
  })

  it('buffers ahead turns and decision playbook in sync-with-voice mode until audio advances or user clicks reveal-all', async () => {
    const audioInstances: Array<{
      src: string
      playbackRate: number
      preservesPitch: boolean
      onended: (() => void) | null
      onerror: (() => void) | null
      play: ReturnType<typeof vi.fn>
      pause: ReturnType<typeof vi.fn>
    }> = []

    class MockPacingAudio {
      src = ''
      playbackRate = 1
      preservesPitch = true
      onended: (() => void) | null = null
      onerror: (() => void) | null = null
      play = vi.fn().mockResolvedValue(undefined)
      pause = vi.fn()
      constructor(src = '') {
        this.src = src
        audioInstances.push(this)
      }
    }

    vi.stubGlobal('Audio', MockPacingAudio)
    try {
      vi.mocked(api.getRoundtableGuruProfiles).mockResolvedValue(makeSampleCatalog())
      vi.mocked(api.createRoundtableSession).mockResolvedValue(makeSampleDetail())
      vi.mocked(streamRoundtableDebate).mockImplementation(async (_sid, _opts, cb) => {
        cb.onTurnCompleted?.({
          turn_id: 'turn_pace_1',
          session_id: 'rt_test_001',
          seq: 1,
          speaker_id: 'value_guru',
          round_stage: 'opening',
          replying_to: null,
          spoken_dialogue: 'ช่วงที่หนึ่งฟังจบก่อนจึงไปต่อครับ',
          evidence_bullets: ['[MATH] FCF Yield 3.1%'],
          stance_tag: 'HOLD',
          audio_cache_key: 'wav_pace_1',
          audio_status: 'ready',
          created_at: 1730000001,
        })
        cb.onTurnCompleted?.({
          turn_id: 'turn_pace_2',
          session_id: 'rt_test_001',
          seq: 2,
          speaker_id: 'growth_guru',
          round_stage: 'opening',
          replying_to: null,
          spoken_dialogue: 'ช่วงที่สองปลดล็อกเมื่อช่วงแรกพูดจบ',
          evidence_bullets: ['[FACT] Revenue +78% YoY'],
          stance_tag: 'BUY',
          audio_cache_key: 'wav_pace_2',
          audio_status: 'ready',
          created_at: 1730000002,
        })
        cb.onTurnCompleted?.({
          turn_id: 'turn_pace_3',
          session_id: 'rt_test_001',
          seq: 3,
          speaker_id: 'skeptic_guru',
          round_stage: 'opening',
          replying_to: null,
          spoken_dialogue: 'ช่วงที่สามรอคิวถัดไปในบัฟเฟอร์',
          evidence_bullets: ['[FACT] Gross margin guidance'],
          stance_tag: 'HOLD',
          audio_cache_key: 'wav_pace_3',
          audio_status: 'ready',
          created_at: 1730000003,
        })
        cb.onPlaybookUpdated?.({
          final_rating: 'BUY',
          action_stance: 'BUY',
          core_conviction_score: 2,
          executive_verdict: 'คำตัดสินสรุปท้ายเวทีหลังฟังครบทุกเทิร์น',
          guru_stance_breakdown: { value_guru: 'HOLD', growth_guru: 'BUY', skeptic_guru: 'HOLD' },
          consensus_points: ['AI Capex cycle แข็งแรง'],
          tension_axes: [],
          personal_axis_alignment: [],
          portfolio_context_summary: 'ไม่ได้แนบพอร์ต',
          action_playbook: {
            primary_action: {
              option_id: 'opt_pace',
              label: 'Phased Entry',
              speaker_source: 'growth_guru',
              conviction_level: 2,
              action_steps: ['แบ่งไม้เข้าซื้อ'],
            },
            alternatives: [],
          },
          kill_switches: ['Gross Margin < 70%'],
        })
        cb.onDone?.({ session_id: 'rt_test_001', status: 'completed' })
      })

      const user = userEvent.setup()
      render(
        <MemoryRouter initialEntries={['/roundtable?ticker=NVDA']}>
          <RoundtableStudio />
        </MemoryRouter>,
      )

      await user.click(screen.getByRole('button', { name: /เริ่มประชุมกูรู/i }))

      // Turn 1 is visible while turns 2-3 & executive verdict remain buffered behind audio queue
      expect((await screen.findAllByText(/ช่วงที่หนึ่งฟังจบก่อนจึงไปต่อครับ/)).length).toBeGreaterThan(0)
      expect(screen.queryByText(/ช่วงที่สองปลดล็อกเมื่อช่วงแรกพูดจบ/)).not.toBeInTheDocument()
      expect(screen.queryByText(/ช่วงที่สามรอคิวถัดไปในบัฟเฟอร์/)).not.toBeInTheDocument()
      expect(screen.queryByText(/คำตัดสินสรุปท้ายเวทีหลังฟังครบทุกเทิร์น/)).not.toBeInTheDocument()
      expect(screen.getByText(/กำลังดำเนินรายการตามคิวเสียงพูด/)).toBeInTheDocument()

      await waitFor(() => {
        expect(audioInstances.length).toBe(1)
      })

      // Advancing turn 1 audio reveals turn 2 while keeping turn 3 & playbook buffered
      await act(async () => {
        audioInstances[0]?.onended?.()
        await Promise.resolve()
      })

      expect((await screen.findAllByText(/ช่วงที่สองปลดล็อกเมื่อช่วงแรกพูดจบ/)).length).toBeGreaterThan(0)
      expect(screen.queryByText(/ช่วงที่สามรอคิวถัดไปในบัฟเฟอร์/)).not.toBeInTheDocument()
      expect(screen.queryByText(/คำตัดสินสรุปท้ายเวทีหลังฟังครบทุกเทิร์น/)).not.toBeInTheDocument()

      // Clicking "แสดงทั้งหมดทันที" reveals buffered turn 3 and Decision Playbook immediately
      await user.click(screen.getByRole('button', { name: /แสดงทั้งหมดทันที/i }))
      expect((await screen.findAllByText(/ช่วงที่สามรอคิวถัดไปในบัฟเฟอร์/)).length).toBeGreaterThan(0)
      expect(await screen.findByText(/คำตัดสินสรุปท้ายเวทีหลังฟังครบทุกเทิร์น/)).toBeInTheDocument()
    } finally {
      vi.unstubAllGlobals()
    }
  })

  it('recovers past meeting history on retry and opens completed sessions with checkDrift=false', async () => {
    const user = userEvent.setup()
    vi.mocked(api.listRoundtableSessions)
      .mockRejectedValueOnce(new Error('เชื่อมต่อเซิร์ฟเวอร์ไม่สำเร็จ'))
      .mockResolvedValueOnce([
        {
          session_id: 'rt_hist_001',
          title: 'ประชุมประวัติ $NVDA Q3',
          primary_ticker: 'NVDA',
          benchmark_ticker: 'SPY',
          agenda_prompt: 'ประเมินมูลค่า NVDA หลังประกาศงบ',
          context_refs: ['$NVDA'],
          final_rating: 'BUY',
          settlement_status: 'settled',
          settled_return_pct: '8.40',
          settled_alpha_pct: '5.10',
          status: 'completed',
          created_at: 1730000000,
          updated_at: 1730000500,
        },
      ])

    vi.mocked(api.getRoundtableSession).mockResolvedValue(
      makeSampleDetail({
        session: {
          session_id: 'rt_hist_001',
          title: 'ประชุมประวัติ $NVDA Q3',
          primary_ticker: 'NVDA',
          benchmark_ticker: 'SPY',
          agenda_prompt: 'ประเมินมูลค่า NVDA หลังประกาศงบ',
          context_refs: ['$NVDA'],
          final_rating: 'BUY',
          settlement_status: 'settled',
          settled_return_pct: '8.40',
          settled_alpha_pct: '5.10',
          status: 'completed',
          created_at: 1730000000,
          updated_at: 1730000500,
        },
        turns: [
          {
            turn_id: 'turn_hist_1',
            session_id: 'rt_hist_001',
            seq: 1,
            speaker_id: 'value_guru',
            round_stage: 'opening',
            replying_to: null,
            spoken_dialogue: 'เนื้อหาประวัติห้องประชุม NVDA เปิดดูได้ทันที',
            evidence_bullets: ['[FACT] รายได้ Data Center โตต่อเนื่อง'],
            stance_tag: 'BUY',
            audio_cache_key: null,
            audio_status: 'ready',
            created_at: 1730000010,
          },
        ],
        decision_playbook: {
          final_rating: 'BUY',
          action_stance: 'Accumulate',
          core_conviction_score: 3,
          executive_verdict: 'มติประวัติรอบก่อนเห็นพ้องให้ทยอยสะสม NVDA',
          guru_stance_breakdown: { value_guru: 'BUY' },
          consensus_points: ['กระแสเงินสดแข็งแกร่ง'],
          tension_axes: [],
          personal_axis_alignment: [],
          portfolio_context_summary: '',
          action_playbook: {
            primary_action: {
              option_id: 'opt_hist',
              label: 'ทยอยเข้าสะสม 2 ไม้',
              speaker_source: 'value_guru',
              conviction_level: 3,
              action_steps: ['สะสมตามโซนรับ'],
            },
            alternatives: [],
          },
          kill_switches: ['Gross Margin < 68%'],
        },
      }),
    )

    render(
      <MemoryRouter initialEntries={['/roundtable']}>
        <RoundtableStudio />
      </MemoryRouter>,
    )

    // First mount shows actionable retry card instead of hanging indefinitely
    expect((await screen.findAllByText(/เชื่อมต่อเซิร์ฟเวอร์ไม่สำเร็จ/)).length).toBeGreaterThan(0)
    await user.click(screen.getByRole('button', { name: /ลองโหลดใหม่/i }))

    // History session renders after recovery; clicking it loads fast with checkDrift: false
    const histItemBtn = await screen.findByRole('button', { name: /ประชุมประวัติ \$NVDA Q3/i })
    await user.click(histItemBtn)

    await waitFor(() => {
      expect(api.getRoundtableSession).toHaveBeenCalledWith(
        'rt_hist_001',
        expect.objectContaining({ checkDrift: false }),
      )
    })
    expect(await screen.findByText(/เนื้อหาประวัติห้องประชุม NVDA เปิดดูได้ทันที/)).toBeInTheDocument()
    expect(await screen.findByText(/มติประวัติรอบก่อนเห็นพ้องให้ทยอยสะสม NVDA/)).toBeInTheDocument()
  })

  it('toggles desktop meeting history rail visibility cleanly', async () => {
    const user = userEvent.setup()
    vi.mocked(api.listRoundtableSessions).mockResolvedValue([])

    render(
      <MemoryRouter initialEntries={['/roundtable']}>
        <RoundtableStudio />
      </MemoryRouter>,
    )

    expect(await screen.findByText('ประวัติห้องประชุม')).toBeInTheDocument()
    const toggleBtn = screen.getByRole('button', { name: /เปิดหรือซ่อนประวัติห้องประชุม/i })
    await user.click(toggleBtn)
    expect(screen.queryByText('ประวัติห้องประชุม')).not.toBeInTheDocument()

    await user.click(toggleBtn)
    expect(await screen.findByText('ประวัติห้องประชุม')).toBeInTheDocument()
  })

  it('uses Full-Bleed 3D HUD overlay by default with collapsible Playbook/Composer and switches cleanly to Classic 2D', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter initialEntries={['/roundtable?ticker=NVDA']}>
        <RoundtableStudio />
      </MemoryRouter>,
    )

    // Full-viewport 3D stage renders behind HUD overlays
    const stage3d = await screen.findByTestId('roundtable-roblox-stage-3d')
    expect(stage3d).toHaveAttribute('data-full-viewport', 'true')

    // Idle Playbook panel starts collapsed into a sleek top-right pill; clicking expands & collapses it
    const playbookPill = screen.getByRole('button', { name: /Playbook ▾/i })
    await user.click(playbookPill)
    expect(await screen.findByText(/Portfolio Guard/i)).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /▾ ย่อ$/i }))
    expect(screen.getByRole('button', { name: /Playbook ▾/i })).toBeInTheDocument()

    // Bottom composer can minimize/expand smoothly inside HUD mode
    await user.click(screen.getByRole('button', { name: /▾ ย่อแผง/i }))
    expect(screen.getByRole('button', { name: /▴ กางแผง/i })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /▴ กางแผง/i }))
    expect(screen.getByPlaceholderText(/พิมพ์โจทย์ให้สภากูรู/i)).toBeInTheDocument()

    // Switch from 3D HUD to Classic 2D 3-column workspace and back
    const viewGroup = screen.getByRole('group', { name: /สลับมุมมองเวทีสภากูรู/i })
    await user.click(within(viewGroup).getByRole('button', { name: /ปกติ \(2D\)/i }))
    expect(screen.queryByTestId('roundtable-roblox-stage-3d')).not.toBeInTheDocument()
    expect(window.localStorage.getItem('trinity_rt_stage_view_mode')).toBe('2d')

    const stage = screen.getByRole('region', { name: /เวทีสภากูรูการลงทุน/i })
    await user.click(within(stage).getByRole('button', { name: /3D Roblox/i }))
    expect(await screen.findByTestId('roundtable-roblox-stage-3d')).toHaveAttribute(
      'data-full-viewport',
      'true',
    )
    expect(window.localStorage.getItem('trinity_rt_stage_view_mode')).toBe('3d')
  })
})

