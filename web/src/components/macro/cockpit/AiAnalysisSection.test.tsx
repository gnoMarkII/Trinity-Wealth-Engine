import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect } from 'vitest'
import { AiAnalysisSection } from './AiAnalysisSection'
import type { MacroDashboardDTO } from '../../../api/types'

describe('AiAnalysisSection', () => {
  const mockAiData: MacroDashboardDTO = {
    evaluated_at: '2026-10-02T15:30:00Z',
    overall_regime: 'Reflation',
    time_horizon: '3-6 Months',
    conviction_level: 'High',
    divergence_note: 'Fed มีแนวโน้มคงดอกเบี้ยระดับสูงนานกว่า ขณะที่ BoT เผชิญแรงกดดันเงินเฟ้อต่ำ',
    regime_probabilities: {
      Reflation: 0.55,
      Goldilocks: 0.25,
      Stagflation: 0.15,
      Deflation: 0.05,
    },
    regime_evidence: [
      {
        dimension: 'Growth',
        signal: 'Expanding Moderately',
        confidence: 'High',
        evidence: 'GDPNow คาดการณ์โต 2.8% ในไตรมาสล่าสุด',
      },
      {
        dimension: 'Inflation',
        signal: 'Sticky Service Inflation',
        confidence: 'Medium',
        evidence: 'Core PCE อยู่ที่ 2.7% YoY',
      },
    ],
    sector_analysis: {
      analysis_status: 'available',
      as_of_date: '2026-10-02',
      summary_th: 'กลุ่มพลังงาน (XLE) และการเงิน (XLF) นำตลาดในโหมด Reflation',
      resolved_metrics: ['XLE: Leading', 'XLF: Improving'],
      watch_conditions: ['จับตาส่วนต่าง 10Y-2Y หากกลับมาชันขึ้น'],
      snapshot_id: 'snap-sec-001',
    },
    thailand_market_stance: {
      investor_flow: { foreign_net_mb: -3386.63 },
      market_breadth: { advance_decline_ratio: 1.08, sentiment: 'neutral' },
      valuation: { pe_ratio: 16.0 },
      physical_gold: { bar_sell_thb: 67850 },
      policy_spread_bps: 275.0,
      rationale: 'ตลาดหุ้นไทยอยู่ในโซนสะสมเชิงคุณค่า ขณะที่เงินบาทมีแรงกดดันจากส่วนต่างดอกเบี้ย',
    },
    asset_allocation: [
      {
        asset_class: 'US Equities (S&P 500)',
        region: 'US',
        stance: 'Overweight',
        confidence: 'high',
        rationale: 'กำไรบริษัทจดทะเบียนเติบโตแข็งแกร่ง',
        warnings: [],
      },
      {
        asset_class: 'Long-Term Treasuries (TLT)',
        region: 'US',
        stance: 'Underweight',
        confidence: 'medium',
        rationale: 'แรงกดดันจากการออกพันธบัตรใหม่จำนวนมาก',
        warnings: [],
      },
      {
        asset_class: 'Thai Equities (SET)',
        region: 'Thailand',
        stance: 'Neutral',
        confidence: 'medium',
        rationale: 'Valuation น่าสนใจแต่รอดูทิศทาง Fund Flow',
        warnings: [],
      },
      {
        asset_class: 'USD/THB (FX)',
        region: 'Thailand',
        stance: 'Overweight',
        confidence: 'high',
        rationale: 'เงินบาทมีแนวโน้มอ่อนค่าเมื่อเทียบดอลลาร์จากส่วนต่างดอกเบี้ย',
        supporting_data: ['US-TH Spread = 275 bps', 'Foreign Net Flow = -3,386.63 MB'],
        warnings: [],
      },
    ],
    pair_trades: [
      {
        long_leg: 'Energy (XLE)',
        short_leg: 'Utilities (XLU)',
        thesis: 'ผลตอบแทนความสัมพันธ์ตามวงจรเศรษฐกิจฟื้นตัว',
        catalyst: 'ราคาน้ำมันดิบยืนเหนือ $80',
        confidence: 'high',
      },
    ],
    risk_scenarios: [
      {
        tail_risk: 'Geopolitical Supply Shock',
        mitigation_strategy: 'เพิ่มสัดส่วนทองคำและน้ำมัน',
        trigger_to_activate: 'WTI > $95',
      },
    ],
    key_assumptions: ['Fed เริ่มลดดอกเบี้ยแบบค่อยเป็นค่อยไป'],
    warnings: [],
    dashboard_indicators: [],
    report_references: [
      {
        reference_id: 'ref-1',
        kind: 'news',
        title: 'ธปท. เตรียมมาตรการช่วยลูกหนี้น้ำท่วม',
        url: 'https://prachachat.net/finance/news-1',
        publisher: 'ประชาชาติธุรกิจ',
        age_hours: 4,
        summary: 'ธนาคารแห่งประเทศไทยหารือสมาคมธนาคารไทยพักหนี้ 3 เดือน',
        thumbnail_url: '',
        is_stale: false,
        related_observable_ids: [],
      },
    ],
    evaluated_sources: [],
    source_files: [],
  } as unknown as MacroDashboardDTO

  it('renders in-depth AI regime, 5D evidence, and all asset allocations including Thai', () => {
    render(<AiAnalysisSection aiData={mockAiData} />)

    // Header
    expect(screen.getByText(/บทวิเคราะห์สภาวะเศรษฐกิจและการจัดสรรสินทรัพย์เชิงลึก/)).toBeInTheDocument()

    // 5D Evidence
    expect(screen.getByText('Growth')).toBeInTheDocument()
    expect(screen.getByText('Expanding Moderately')).toBeInTheDocument()
    expect(screen.getByText(/GDPNow คาดการณ์โต 2.8%/)).toBeInTheDocument()

    // Asset Allocations (both US and Thai included)
    expect(screen.getByText('US Equities (S&P 500)')).toBeInTheDocument()
    expect(screen.getByText('Long-Term Treasuries (TLT)')).toBeInTheDocument()
    expect(screen.getAllByText('Thai Equities (SET)')[0]).toBeInTheDocument()
    expect(screen.getAllByText('USD/THB (FX)')[0]).toBeInTheDocument()

    // Pair Trades
    expect(screen.getByText(/Energy \(XLE\)/)).toBeInTheDocument()
    expect(screen.getByText(/Utilities \(XLU\)/)).toBeInTheDocument()

    // Tail Risks
    expect(screen.getByText('Geopolitical Supply Shock')).toBeInTheDocument()
    expect(screen.getByText(/Trigger:/)).toBeInTheDocument()
    expect(screen.getByText(/WTI > \$95/)).toBeInTheDocument()
  })

  it('renders unified AI Sector Rotation, Thailand Market Stance, and Policy Divergence Note', () => {
    render(<AiAnalysisSection aiData={mockAiData} />)

    // AI Sector Rotation Panel
    expect(screen.getByText(/มุมมองรายกลุ่มอุตสาหกรรม \(AI Sector Rotation Analysis\)/)).toBeInTheDocument()
    expect(screen.getByText(/กลุ่มพลังงาน \(XLE\) และการเงิน \(XLF\) นำตลาดในโหมด Reflation/)).toBeInTheDocument()
    expect(screen.getByText(/XLE: Leading/)).toBeInTheDocument()

    // Thailand AI Market Stance & Strategy
    expect(screen.getByText(/AI วิเคราะห์สภาวะตลาดทุนและค่าเงินบาท \(AI Thailand Market Stance\)/)).toBeInTheDocument()
    expect(screen.getByText(/ตลาดหุ้นไทยอยู่ในโซนสะสมเชิงคุณค่า ขณะที่เงินบาทมีแรงกดดันจากส่วนต่างดอกเบี้ย/)).toBeInTheDocument()
    expect(screen.getByText(/ธปท. เตรียมมาตรการช่วยลูกหนี้น้ำท่วม/)).toBeInTheDocument()
    expect(screen.getByText('ประชาชาติธุรกิจ')).toBeInTheDocument()

    // Cross-Border Policy Divergence Note
    expect(screen.getByText(/ความเชื่อมโยงข้ามพรมแดน \(Policy Divergence & Transmission\)/)).toBeInTheDocument()
    expect(screen.getByText(/Fed มีแนวโน้มคงดอกเบี้ยระดับสูงนานกว่า ขณะที่ BoT เผชิญแรงกดดันเงินเฟ้อต่ำ/)).toBeInTheDocument()
  })

  it('filters asset allocations when stance buttons are clicked', async () => {
    render(<AiAnalysisSection aiData={mockAiData} />)

    const overweightBtn = screen.getByRole('button', { name: 'Overweight' })
    await userEvent.click(overweightBtn)

    expect(screen.getByText('US Equities (S&P 500)')).toBeInTheDocument()
    expect(screen.queryByText('Long-Term Treasuries (TLT)')).not.toBeInTheDocument()
    // Thai Equities (SET) was Neutral, so it shouldn't be in the filtered list
    // (though it may be in the Thailand AI card, in the asset allocation grid it was filtered)
  })

  it('renders policy and quant alignment synthesis gracefully when divergence_note is empty', () => {
    const dataWithoutDivergenceNote: MacroDashboardDTO = {
      ...mockAiData,
      divergence_note: '',
      quant_narrative_alignment: 'aligned',
    }

    render(<AiAnalysisSection aiData={dataWithoutDivergenceNote} />)

    expect(screen.getByText(/ความสอดคล้องของข้อมูลและนโยบาย \(Policy & Quant Alignment\):/)).toBeInTheDocument()
    expect(screen.getByText(/ข้อมูลเชิงปริมาณและปัจจัยเชิงคุณภาพสอดคล้องไปในทิศทางเดียวกัน \(Aligned\)/)).toBeInTheDocument()
    expect(screen.getAllByText(/\+275 bps/).length).toBeGreaterThanOrEqual(1)
  })

  it('synthesizes Thai market stance narrative dynamically when rationale is null', () => {
    const dataWithoutRationale: MacroDashboardDTO = {
      ...mockAiData,
      thailand_market_stance: {
        investor_flow: { foreign_net_mb: -850.49, institution_net_mb: 1543.07 },
        market_breadth: { advance_decline_ratio: 1.68, sentiment: 'bullish' },
        valuation: { pe_ratio: 15.64, dividend_yield: 4.05 },
        policy_spread_bps: 262.5,
        rationale: null,
      },
    }

    render(<AiAnalysisSection aiData={dataWithoutRationale} />)

    expect(screen.queryByText(/ยังไม่มีบทสรุป AI สำหรับตลาดทุนไทยในรอบนี้/)).not.toBeInTheDocument()
    expect(screen.getByText(/💡 ทัศนะรวมสภาวะตลาดทุนไทย \(Macro Stance Narrative\):/)).toBeInTheDocument()
    expect(screen.getByText(/SET Index ซื้อขายที่ระดับ P\/E 15.64 เท่า/)).toBeInTheDocument()
    expect(screen.getByText(/Dividend Yield 4.05%/)).toBeInTheDocument()
    expect(screen.getByText(/ต่างชาติขายสุทธิ -850.49 ล้านบาท/)).toBeInTheDocument()
    expect(screen.getByText(/สถาบันในประเทศ ซื้อสุทธิ \+1,543.07 ล้านบาท/)).toBeInTheDocument()
    expect(screen.getAllByText(/\+262.5 bps/).length).toBeGreaterThanOrEqual(1)
  })
})
