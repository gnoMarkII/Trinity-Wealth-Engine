import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import { describe, expect, it, vi, beforeEach } from 'vitest'
import PortfolioAdvisory from './PortfolioAdvisory'
import { api } from '../api/client'
import type { PortfolioAdvisoryResponseDTO, PortfolioAdvisoryNarrativeDTO } from '../api/types'

vi.mock('../api/client', () => ({
  api: {
    getPortfolioAdvisory: vi.fn(),
    synthesizeAdvisoryPlaybook: vi.fn(),
  },
}))

const mockAdvisoryData: PortfolioAdvisoryResponseDTO = {
  evaluated_at: '2026-10-18T10:00:00Z',
  data_mode: 'FULL_CROSS_SCALE',
  portfolio_state_hash: 'hash_test_1234',
  total_nav_thb: 1500000.0,
  cash_plan: {
    cash_thb_available: 300000.0,
    cash_usd_available: 2500.0,
    usd_thb_fx_rate: 35.5,
    immediate_deploy_thb: 50000.0,
    immediate_deploy_usd: 1000.0,
    conditional_limit_order_thb: 25000.0,
    conditional_limit_order_usd: 0.0,
    thb_surplus_after_immediate: 250000.0,
    usd_surplus_after_immediate: 1500.0,
    fx_conversion_needed_thb: 0.0,
  },
  risk_radar: {
    weighted_beta: 1.12,
    sector_crowding: { Technology: 45.0, Financials: 20.0 },
    fx_exposure_pct: { THB: 70.0, USD: 30.0 },
    purpose_bucket_drift: {},
    unassigned_holdings_value_thb: 0.0,
    unassigned_holdings_nav_pct: 0.0,
    unassigned_drift_guard_active: false,
    hhi: 0.145,
    effective_holdings_count: 6.9,
    top_1_weight_pct: 15.0,
    top_3_weight_pct: 38.0,
    top_5_weight_pct: 55.0,
    concentration_risk_level: 'MODERATE',
  },
  holdings_advisory: [
    {
      symbol: 'ADVANC.BK',
      asset_name: 'Advanced Info Service',
      asset_type: 'Stock',
      market: 'TH',
      native_currency: 'THB',
      evaluation_track: 'TRACK_1_DIRECT_EQUITIES',
      current_units: 500.0,
      current_price_native: 250.0,
      current_value_thb: 125000.0,
      current_weight_pct: 8.3,
      target_weight_pct: 10.0,
      weight_drift_pct: -1.7,
      action_code: 'ACCUMULATE_NOW',
      action_rationale: 'สัญญาณ Conviction สูง แนะนำเคาะซื้อวันนี้',
      is_immediate_execution: true,
      lot_sizing: {
        market: 'TH',
        board_lot_shares: 100,
        odd_lot_shares: 0,
        whole_shares: 100,
        fractional_shares: 100.0,
        amount_thb: 25000.0,
        amount_native: 25000.0,
        estimated_execution_price_native: 250.0,
        native_currency: 'THB',
      },
      core_conviction_score: 8.5,
      price_stage: 'STAGE_2_UPTREND',
      macro_alignment: 'tailwind',
      breached_falsifiers: [],
      data_warnings: [],
    },
    {
      symbol: 'AAPL',
      asset_name: 'Apple Inc.',
      asset_type: 'Stock',
      market: 'US',
      native_currency: 'USD',
      evaluation_track: 'TRACK_1_DIRECT_EQUITIES',
      current_units: 20.0,
      current_price_native: 185.0,
      current_value_thb: 131350.0,
      current_weight_pct: 8.7,
      target_weight_pct: 10.0,
      weight_drift_pct: -1.3,
      action_code: 'ACCUMULATE_ON_DIP',
      action_rationale: 'ปัจจัยพื้นฐานดีแต่ราคาสูงกว่าโซนซื้อ แนะนำตั้งรับ',
      is_immediate_execution: false,
      conditional_buy_zone_min: 175.0,
      conditional_buy_zone_max: 180.0,
      core_conviction_score: 8.0,
      price_stage: 'STAGE_1_ACCUMULATION',
      macro_alignment: 'neutral',
      breached_falsifiers: [],
      data_warnings: [],
    },
  ],
  quarantined_holdings: [],
  anomaly_warnings: [
    {
      guardrail_code: 'CROSS_SCALE_DATE_DRIFT',
      ticker: 'MACRO',
      message: 'Macro report is 5 days old',
      severity: 'warning',
      remedy_hint: 'Run strategic allocator to refresh',
    },
  ],
  cached_narrative: {
    synthesized_at: '2026-10-18T10:05:00Z',
    portfolio_state_hash: 'hash_test_1234',
    macro_executive_summary: 'ภาพรวมพอร์ตมีความสมดุล แนะนำเคาะซื้อ ADVANC.BK',
    immediate_action_playbook: ['เคาะซื้อ ADVANC.BK 100 หุ้น'],
    conditional_dip_plan: ['ตั้งรับ AAPL ที่ 175 - 180 USD'],
    risk_mitigation_and_hedging: ['รักษาสภาพคล่องเงินสดในระดับ 20%'],
    consistency_verified: true,
    generated_by: 'portfolio_synthesizer',
  },
}

describe('PortfolioAdvisory Desk Page', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(api.getPortfolioAdvisory).mockResolvedValue(mockAdvisoryData)
  })

  it('renders advisory desk with NAV, cash plan, and holdings table', async () => {
    render(<PortfolioAdvisory />)

    await waitFor(() => {
      expect(screen.getByText(/Portfolio Advisory Desk/i)).toBeInTheDocument()
    })

    // Mode badge
    expect(screen.getByText(/Full Cross-Scale Mode/i)).toBeInTheDocument()

    // Holdings rendered
    expect(screen.getByText('ADVANC.BK')).toBeInTheDocument()
    expect(screen.getByText('AAPL')).toBeInTheDocument()

    // Action Badges
    expect(screen.getByText('ACCUMULATE_NOW')).toBeInTheDocument()
    expect(screen.getByText('ACCUMULATE_ON_DIP')).toBeInTheDocument()

    // AI Playbook Card
    expect(screen.getByText(/AI Strategic Playbook/i)).toBeInTheDocument()
    expect(screen.getByText(/เคาะซื้อ ADVANC.BK 100 หุ้น/i)).toBeInTheDocument()
    expect(screen.getByText(/ตัวเลขสอดคล้อง 100%/i)).toBeInTheDocument()

    // Concentration & Effective Holdings in Risk Radar
    expect(screen.getByText(/Effective Holdings/i)).toBeInTheDocument()
    expect(screen.getByText('กระจุกตัวปานกลาง')).toBeInTheDocument()

    // Guardrail Anomaly Warning Banner
    expect(screen.getByText('CROSS_SCALE_DATE_DRIFT')).toBeInTheDocument()
    expect(screen.getByText(/Macro report is 5 days old/i)).toBeInTheDocument()
  })

  it('handles price refresh button click', async () => {
    render(<PortfolioAdvisory />)

    await waitFor(() => {
      expect(screen.getByText('ADVANC.BK')).toBeInTheDocument()
    })

    const refreshBtn = screen.getByText('รีเฟรชราคาตลาด')
    fireEvent.click(refreshBtn)

    expect(api.getPortfolioAdvisory).toHaveBeenCalledWith(true)
  })

  it('handles AI Playbook synthesis click', async () => {
    const updatedNarrative: PortfolioAdvisoryNarrativeDTO = {
      synthesized_at: '2026-10-18T10:15:00Z',
      portfolio_state_hash: 'hash_test_1234',
      macro_executive_summary: 'แผนกลยุทธ์ฉบับใหม่',
      immediate_action_playbook: ['เคาะซื้อ ADVANC.BK เพิ่มเติม'],
      conditional_dip_plan: [],
      risk_mitigation_and_hedging: [],
      consistency_verified: true,
      generated_by: 'portfolio_synthesizer',
    }
    vi.mocked(api.synthesizeAdvisoryPlaybook).mockResolvedValue(updatedNarrative)

    render(<PortfolioAdvisory />)

    await waitFor(() => {
      expect(screen.getByText('ADVANC.BK')).toBeInTheDocument()
    })

    const synthBtn = screen.getByText(/วิเคราะห์กลยุทธ์ใหม่ \(AI Playbook\)/i)
    fireEvent.click(synthBtn)

    expect(api.synthesizeAdvisoryPlaybook).toHaveBeenCalledWith(true)

    await waitFor(() => {
      expect(screen.getByText('แผนกลยุทธ์ฉบับใหม่')).toBeInTheDocument()
    })
  })

  it('renders Cross-Portfolio Stock Overlap Radar and True Sector Exposure when look_through_summary is present', async () => {
    const dataWithLookThrough: PortfolioAdvisoryResponseDTO = {
      ...mockAdvisoryData,
      look_through_summary: {
        holdings_exposure: [
          {
            ticker: 'AAPL',
            company_name: 'Apple Inc.',
            market: 'US',
            direct_value_thb: 131350.0,
            direct_weight_pct: 8.7,
            indirect_value_thb: 150000.0,
            indirect_weight_pct: 10.0,
            total_value_thb: 281350.0,
            total_weight_pct: 18.7,
            is_cap_breached: true,
            is_hidden_concentration: false,
            source_funds: [['SCBS&P500E', 150000.0]],
          },
        ],
        sector_exposure: [
          {
            sector_name: 'Information Technology',
            sector_name_th: 'เทคโนโลยีสารสนเทศ',
            total_value_thb: 500000.0,
            weight_pct: 33.3,
            direct_stocks_pct: 15.0,
            funds_pct: 18.3,
            is_crowded: true,
          },
        ],
        top_overlapped_stocks: [
          {
            ticker: 'AAPL',
            company_name: 'Apple Inc.',
            market: 'US',
            direct_value_thb: 131350.0,
            direct_weight_pct: 8.7,
            indirect_value_thb: 150000.0,
            indirect_weight_pct: 10.0,
            total_value_thb: 281350.0,
            total_weight_pct: 18.7,
            is_cap_breached: true,
            is_hidden_concentration: false,
            source_funds: [['SCBS&P500E', 150000.0]],
          },
        ],
        hidden_concentration_warnings: ['AAPL ถือครองรวม 18.7% เกินเพดานความเสี่ยงกระจุกตัว 15.0%'],
        cap_breached_count: 1,
        portfolio_id: 'default',
        total_nav_thb: 1500000.0,
        evaluated_at: '2026-10-18T10:00:00Z',
      },
    }

    vi.mocked(api.getPortfolioAdvisory).mockResolvedValue(dataWithLookThrough)

    render(<PortfolioAdvisory />)

    await waitFor(() => {
      expect(screen.getByText(/Look-Through Radar & True Exposure/i)).toBeInTheDocument()
    })

    // Look-Through overlap warnings and elements
    expect(screen.getByText(/Overlap Radar/i)).toBeInTheDocument()
    expect(screen.getByText(/AAPL ถือครองรวม 18.7% เกินเพดานความเสี่ยงกระจุกตัว 15.0%/i)).toBeInTheDocument()
    expect(screen.getByText('>15% Cap')).toBeInTheDocument()
    expect(screen.getByText('SCBS&P500E')).toBeInTheDocument()

    // True Sector Exposure
    expect(screen.getByText(/True Sector Exposure vs Macro/i)).toBeInTheDocument()
    expect(screen.getByText('Information Technology')).toBeInTheDocument()
    expect(screen.getByText('Crowded >30%')).toBeInTheDocument()
  })

  it('renders strategic RM lens with whats_working, what_concerns_me, and discussion points', async () => {
    const dataWithStrategicLens = {
      ...mockAdvisoryData,
      cached_narrative: {
        ...mockAdvisoryData.cached_narrative!,
        whats_working: ['ADVANC.BK ปัจจัยหนุนจากอัตราเงินปันผลมั่นคง'],
        what_concerns_me: ['สัดส่วนเทคโนโลยีมีความเสี่ยงหากอัตราผลตอบแทนพันธบัตรปรับขึ้น'],
        strategic_discussion_points: ['พิจารณากระจายน้ำหนักไปยัง Real Assets เพิ่มขึ้น 5%'],
      },
    }

    vi.mocked(api.getPortfolioAdvisory).mockResolvedValue(dataWithStrategicLens)

    render(<PortfolioAdvisory />)

    await waitFor(() => {
      expect(screen.getByText(/What's Working/i)).toBeInTheDocument()
    })

    expect(screen.getByText('ADVANC.BK ปัจจัยหนุนจากอัตราเงินปันผลมั่นคง')).toBeInTheDocument()
    expect(screen.getByText(/What Concerns Me/i)).toBeInTheDocument()
    expect(screen.getByText('สัดส่วนเทคโนโลยีมีความเสี่ยงหากอัตราผลตอบแทนพันธบัตรปรับขึ้น')).toBeInTheDocument()
    expect(screen.getByText(/Strategic Discussion Points/i)).toBeInTheDocument()
    expect(screen.getByText('พิจารณากระจายน้ำหนักไปยัง Real Assets เพิ่มขึ้น 5%')).toBeInTheDocument()
  })
})

