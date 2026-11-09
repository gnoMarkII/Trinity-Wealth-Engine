import React, { useState, useEffect } from 'react'
import { api } from '../../api/client'
import type {
  EquityDetailDTO,
  EarningsCallNoteItem,
  VariantPerceptionSnapshotDTO,
} from '../../api/types'
import ScoreRing from './ScoreRing'
import { sentimentClass } from '../../lib/sentiment'
import { EquityNews } from './EquityNews'
import { EquityNotesTab } from './EquityNotesTab'
import { EquityChartTab } from './EquityChartTab'
import { FinancialsTab } from './FinancialsTab'
import { DataQualityFlagsCard } from './DataQualityFlagsCard'
import { EarningsCallTab } from './EarningsCallTab'
import { EventStudyTab } from './EventStudyTab'
import ExecutiveThesisHero from './ExecutiveThesisHero'
import { InstitutionalThesisCard } from './InstitutionalThesisCard'
import ValuationWorkbenchCard from './ValuationWorkbenchCard'
import { VariantPerceptionCard } from './VariantPerceptionCard'
import TacticalFlowMatrixCard from './TacticalFlowMatrixCard'
import { EvidenceProvenanceDrawer } from './EvidenceProvenanceDrawer'
import { NasdaqConsensusCard } from './NasdaqConsensusCard'
import { OptionsOiStrikeLadder } from '../charts/OptionsOiStrikeLadder'
import { OptionsVolSmile } from '../charts/OptionsVolSmile'
import { EquityNotebookLMExport } from './EquityNotebookLMExport'
import { EquityDetailSkeleton } from '../ui/FlowSkeleton'
import { PositionReviewCard } from './PositionReviewCard'
import { PositionReviewModal } from './PositionReviewModal'
import { ExpectationGapCard } from './ExpectationGapCard'
import { KillSwitchProximityMeter } from './KillSwitchProximityMeter'
import { EquityGlossaryDrawer, StockWayfinderBanner, SectionWaypointHeader } from './explainer'

interface EquityDetailProps {
  status: 'loading' | 'error' | 'not-found' | 'success' | 'idle'
  data?: EquityDetailDTO
  summaryPreview?: import('../../api/types').EquitySummaryDTO
  errorMessage?: string
  onOpenAnalysisModal?: (ticker: string, market?: 'US' | 'TH') => void
  isUpdating?: boolean
  onNotify?: (message: string, type: 'info' | 'success' | 'error') => void
  portfolioId?: string
}

const eyebrowClass = 'text-xs font-bold uppercase tracking-wider text-sky-700/80'

export const EquityDetail: React.FC<EquityDetailProps> = ({
  status,
  data,
  summaryPreview,
  errorMessage,
  onOpenAnalysisModal,
  isUpdating,
  onNotify,
  portfolioId = 'default',
}) => {
  const [activeTab, setActiveTab] = useState<'overview' | 'chart' | 'financials' | 'news' | 'notes' | 'earnings-call' | 'event-study'>('overview')
  const [visitedTabs, setVisitedTabs] = useState<Set<string>>(() => new Set(['overview']))
  const [latestEarningsCall, setLatestEarningsCall] = useState<EarningsCallNoteItem | null>(null)
  const [isDerivativesExpanded, setIsDerivativesExpanded] = useState(false)
  const [isProvenanceDrawerOpen, setIsProvenanceDrawerOpen] = useState(false)
  const [nasdaqConsensus, setNasdaqConsensus] = useState<any | null>(null)
  const [optionsChain, setOptionsChain] = useState<any | null>(null)
  const [maxPain, setMaxPain] = useState<any | null>(null)
  const [variantPerception, setVariantPerception] = useState<VariantPerceptionSnapshotDTO | null>(null)
  const [isPositionReviewModalOpen, setIsPositionReviewModalOpen] = useState(false)
  const [isGlossaryOpen, setIsGlossaryOpen] = useState(false)

  useEffect(() => {
    if (data?.ticker) {
      setVisitedTabs(new Set([activeTab]))
      setVariantPerception(null)
      setIsPositionReviewModalOpen(false)
    }
  }, [data?.ticker])

  const handleTabSelect = (tab: typeof activeTab) => {
    setActiveTab(tab)
    setVisitedTabs((prev) => {
      const next = new Set(prev)
      next.add(tab)
      return next
    })
  }

  // Background warmup after ticker loads
  useEffect(() => {
    if (!data?.ticker) return
    const timer = setTimeout(() => {
      api.getEquityNews?.(data.ticker)?.catch?.(() => { })
      api.getEquityOHLCV?.(data.ticker, '6mo', '1d')?.catch?.(() => { })
    }, 150)
    return () => clearTimeout(timer)
  }, [data?.ticker])

  useEffect(() => {
    if (!data?.ticker) return
    let cancelled = false
    api.getEarningsCalls(data.ticker)
      .then((res) => {
        if (cancelled) return
        if (res.items && res.items.length > 0) {
          setLatestEarningsCall(res.items[0] ?? null)
        } else {
          setLatestEarningsCall(null)
        }
      })
      .catch(() => {
        if (!cancelled) setLatestEarningsCall(null)
      })

    if (typeof api.getVariantPerception === 'function') {
      Promise.resolve(api.getVariantPerception(data.ticker))
        .then((res) => {
          if (!cancelled) {
            setVariantPerception(res && res.alpha_verdict ? res : null)
          }
        })
        .catch(() => {
          if (!cancelled) setVariantPerception(null)
        })
    } else {
      setVariantPerception(null)
    }

    const sym = data.ticker
    // Keyless US Institutional Intelligence
    if (data.market === 'US' || !sym.includes('.')) {
      api.getNasdaqConsensus?.(sym)?.then((res) => { if (!cancelled) setNasdaqConsensus(res) })?.catch(() => { if (!cancelled) setNasdaqConsensus(null) })
      api.getOptionsChain?.(sym)?.then((res) => { if (!cancelled) setOptionsChain(res) })?.catch(() => { if (!cancelled) setOptionsChain(null) })
      api.getOptionsMaxPain?.(sym)?.then((res) => { if (!cancelled) setMaxPain(res) })?.catch(() => { if (!cancelled) setMaxPain(null) })
    } else {
      setNasdaqConsensus(null)
      setOptionsChain(null)
      setMaxPain(null)
    }

    return () => {
      cancelled = true
    }
  }, [data?.ticker, data?.market])

  const strikeLadderData = React.useMemo(() => {
    if (!optionsChain?.contracts || !Array.isArray(optionsChain.contracts)) return []
    const map = new Map<number, { strike: number; callOi: number; putOi: number }>()
    optionsChain.contracts.forEach((c: any) => {
      const s = Number(c.strike)
      if (isNaN(s)) return
      if (!map.has(s)) map.set(s, { strike: s, callOi: 0, putOi: 0 })
      const item = map.get(s)!
      const oi = Number(c.open_interest || 0)
      if (c.option_type?.toLowerCase() === 'call') {
        item.callOi += oi
      } else {
        item.putOi += oi
      }
    })
    return Array.from(map.values()).sort((a, b) => a.strike - b.strike)
  }, [optionsChain])

  if (status === 'idle') {
    return null
  }

  if (status === 'loading') {
    if (summaryPreview) {
      return (
        <div className="space-y-6 animate-fade-in" aria-live="polite">
          <div className="flex flex-col gap-5 border-b border-edge pb-6 lg:flex-row lg:items-start lg:justify-between">
            <div className="space-y-2">
              <div className={eyebrowClass}>Institutional Equity Intelligence</div>
              <div className="flex flex-wrap items-center gap-3">
                <h2 className="font-serif text-3xl sm:text-4xl font-semibold tracking-tight text-zinc-900">
                  {summaryPreview.ticker} <span className="text-zinc-500 font-normal text-lg">({summaryPreview.market})</span>
                </h2>
              </div>
              {summaryPreview.company_name && (
                <p className="text-zinc-600 font-medium text-sm sm:text-base">{summaryPreview.company_name}</p>
              )}
              <div className="flex flex-wrap items-center gap-2 pt-1">
                <span className={`px-2.5 py-0.5 rounded-full border text-xs font-semibold uppercase tracking-wider ${sentimentClass(summaryPreview.market_sentiment)}`}>
                  {summaryPreview.market_sentiment}
                </span>
                <span className="text-xs text-zinc-400">
                  ประเมินเมื่อ {new Date(summaryPreview.evaluated_at).toLocaleString('th-TH')}
                </span>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-4 self-start rounded-2xl border border-edge/80 bg-panel px-5 py-4 shadow-sm shadow-black/5">
              <div className="flex items-center gap-2">
                <ScoreRing score={summaryPreview.composite_score} />
                <div>
                  <div className="text-[10px] font-bold text-zinc-400 uppercase">Composite</div>
                  <div className="text-xs font-semibold text-zinc-700">
                    {summaryPreview.composite_score != null ? `${summaryPreview.composite_score.toFixed(1)}` : 'N/A'}
                  </div>
                </div>
              </div>
            </div>
          </div>

          <EquityDetailSkeleton label="กำลังโหลดบทวิเคราะห์หุ้น..." />
        </div>
      )
    }

    return <EquityDetailSkeleton label="กำลังโหลดบทวิเคราะห์หุ้น..." />
  }

  if (status === 'not-found') {
    return (
      <div className="flex flex-col justify-center items-center h-64 bg-surface rounded-2xl border border-dashed border-edge p-8 text-center shadow-sm">
        <svg className="w-12 h-12 text-zinc-400 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 002-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
        </svg>
        <h3 className="text-lg font-medium text-zinc-900 mb-1">ไม่พบข้อมูล</h3>
        <p className="text-zinc-500 max-w-sm mb-4 text-sm">
          ยังไม่มีการวิเคราะห์สำหรับหุ้นตัวนี้ กรุณาสั่งงานผ่าน Trinity Advisor (Manager) หรือสั่ง Research Swarm เริ่มการวิเคราะห์ระดับสูง
        </p>
      </div>
    )
  }

  if (status === 'error' || !data) {
    return (
      <div className="flex flex-col justify-center items-center h-64 bg-rose-50/80 rounded-2xl border border-rose-200 p-8 text-center text-rose-700 shadow-sm" role="alert">
        <h3 className="text-lg font-bold mb-1">เกิดข้อผิดพลาดในการโหลดข้อมูล</h3>
        <p className="text-sm text-rose-600">{errorMessage || 'ไม่สามารถติดต่อ Backend API ได้'}</p>
      </div>
    )
  }

  const quant = data.quant_signals || ({} as any)
  const resolvedCurrentPrice =
    quant.atomic_market_snapshot?.analysis_price ??
    (quant as any)?.raw_analysis_price ??
    quant.tactical_setup?.current_price ??
    (quant as any)?.current_price ??
    null
  const currencySymbol = data.market === 'TH' ? '฿' : '$'
  const scorecard = quant.deterministic_scorecard
  const reverseDcf = quant.reverse_dcf_result
  const tactical = quant.tactical_setup
  const insider = quant.insider_conviction
  const falsifiers = quant.thesis_falsifiers || []
  const evidenceSnapshot = (quant as any)?.evidence_snapshot || (data as any)?.evidence_snapshot

  return (
    <div className="animate-page-in space-y-8">
      {/* Masthead: Ticker & Unified Hero Action Strip */}
      <div className="flex flex-col gap-5 border-b border-edge pb-6 lg:flex-row lg:items-start lg:justify-between">
        <div className="space-y-2">
          <div className={eyebrowClass}>Institutional Equity Intelligence</div>
          <div className="flex flex-wrap items-center gap-3">
            <h2 className="font-serif text-3xl sm:text-4xl font-semibold tracking-tight text-zinc-900">
              {data.ticker} <span className="text-zinc-500 font-normal text-lg">({data.market})</span>
            </h2>
            <button
              onClick={() => onOpenAnalysisModal?.(data.ticker, (data.market || 'US') as 'US' | 'TH')}
              disabled={isUpdating}
              className={`px-3 py-1 rounded-xl border text-xs font-semibold flex items-center gap-1.5 transition-colors shadow-2xs ${isUpdating
                  ? 'border-amber-200 bg-amber-50 text-amber-700 opacity-80 cursor-not-allowed'
                  : 'border-sky-200 bg-sky-50 text-sky-700 hover:bg-sky-100'
                }`}
              title={isUpdating ? 'กำลังประมวลผลการวิเคราะห์...' : 'วิเคราะห์ใหม่และดึงข่าวล่าสุด'}
            >
              <span className={isUpdating ? 'inline-block animate-spin' : ''}>🔄</span>
              <span>{isUpdating ? 'กำลังวิเคราะห์...' : 'อัปเดตบทวิเคราะห์'}</span>
            </button>
            <EquityNotebookLMExport
              ticker={data.ticker}
              market={data.market || 'US'}
              onNotify={onNotify}
            />
            <button
              onClick={() => setIsGlossaryOpen(true)}
              className="px-3 py-1 rounded-xl border border-emerald-200 bg-emerald-50 text-emerald-800 hover:bg-emerald-100 text-xs font-semibold flex items-center gap-1.5 transition-colors shadow-2xs"
              title="เปิดคู่มือคำศัพท์หุ้นภาษาคน (30 คำศัพท์วิเคราะห์หุ้นระดับสถาบัน)"
            >
              <span>📖</span>
              <span>คู่มือศัพท์หุ้นภาษาคน</span>
            </button>
          </div>
          {isUpdating && (
            <div className="flex items-center gap-2 rounded-xl border border-amber-200 bg-amber-50/80 px-3 py-1.5 text-xs text-amber-900 font-medium">
              <span className="inline-block animate-spin">⚙️</span>
              <span>กำลังวิเคราะห์และดึงข่าวล่าสุดสำหรับ {data.ticker} ({data.market})... ระบบจะรีเฟรชข้อมูลอัตโนมัติเมื่อเสร็จสิ้น</span>
            </div>
          )}
          {data.company_name && <p className="text-zinc-600 font-medium text-sm sm:text-base">{data.company_name}</p>}
          <div className="flex flex-wrap items-center gap-2 pt-1">
            <span className={`px-2.5 py-0.5 rounded-full border text-xs font-semibold uppercase tracking-wider ${sentimentClass(data.market_sentiment)}`}>
              {data.market_sentiment}
            </span>
            <span className="text-xs text-zinc-400">
              ประเมินเมื่อ {new Date(data.evaluated_at).toLocaleString('th-TH')}
            </span>
          </div>
        </div>

        {/* Consolidated Masthead Quick Strip */}
        <div className="flex flex-wrap items-center gap-4 self-start rounded-2xl border border-edge/80 bg-panel px-5 py-3.5 shadow-sm shadow-black/5">
          {resolvedCurrentPrice != null && (
            <div className="flex flex-col items-start pr-4 border-r border-edge/60">
              <span className="text-[10px] font-bold text-zinc-400 uppercase tracking-widest">Analysis Price</span>
              <div className="font-mono text-xl font-extrabold text-zinc-900 mt-0.5">
                {currencySymbol}{resolvedCurrentPrice.toFixed(2)}
              </div>
            </div>
          )}

          {/* Composite Score Mini Ring */}
          <div className="flex items-center gap-2.5">
            <ScoreRing score={data.composite_score} />
            <div>
              <div className="text-[10px] font-bold text-zinc-400 uppercase tracking-widest">Composite</div>
              <div className="text-xs font-semibold text-zinc-700">
                {data.composite_score != null ? `${data.composite_score.toFixed(1)} / 100` : 'N/A'}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* 30-Second Stock Wayfinder Banner */}
      <StockWayfinderBanner
        ticker={data.ticker}
        companyName={data.company_name}
        market={data.market}
      />

      {/* Sub-nav Tab Switcher */}
      <div className="flex border-b border-edge gap-6 text-sm font-medium">
        <button
          onClick={() => handleTabSelect('overview')}
          className={`pb-3 border-b-2 transition-colors flex items-center gap-2 ${activeTab === 'overview'
              ? 'border-sky-600 text-sky-600 font-semibold'
              : 'border-transparent text-zinc-500 hover:text-zinc-900'
            }`}
        >
          <span>📊 Overview</span>
        </button>
        <button
          onClick={() => handleTabSelect('chart')}
          className={`pb-3 border-b-2 transition-colors flex items-center gap-2 ${activeTab === 'chart'
              ? 'border-sky-600 text-sky-600 font-semibold'
              : 'border-transparent text-zinc-500 hover:text-zinc-900'
            }`}
        >
          <span>📈 Chart</span>
        </button>
        <button
          onClick={() => handleTabSelect('financials')}
          className={`pb-3 border-b-2 transition-colors flex items-center gap-2 ${activeTab === 'financials'
              ? 'border-sky-600 text-sky-600 font-semibold'
              : 'border-transparent text-zinc-500 hover:text-zinc-900'
            }`}
        >
          <span>📑 Financials</span>
        </button>
        <button
          onClick={() => handleTabSelect('news')}
          className={`pb-3 border-b-2 transition-colors flex items-center gap-2 ${activeTab === 'news'
              ? 'border-sky-600 text-sky-600 font-semibold'
              : 'border-transparent text-zinc-500 hover:text-zinc-900'
            }`}
        >
          <span>📰 News</span>
        </button>
        <button
          onClick={() => handleTabSelect('notes')}
          className={`pb-3 border-b-2 transition-colors flex items-center gap-2 ${activeTab === 'notes'
              ? 'border-sky-600 text-sky-600 font-semibold'
              : 'border-transparent text-zinc-500 hover:text-zinc-900'
            }`}
        >
          <span>📓 Notes</span>
        </button>
        <button
          onClick={() => handleTabSelect('earnings-call')}
          className={`pb-3 border-b-2 transition-colors flex items-center gap-2 ${activeTab === 'earnings-call'
              ? 'border-sky-600 text-sky-600 font-semibold'
              : 'border-transparent text-zinc-500 hover:text-zinc-900'
            }`}
        >
          <span>🎙️ Earnings Call</span>
        </button>
        <button
          onClick={() => handleTabSelect('event-study')}
          className={`pb-3 border-b-2 transition-colors flex items-center gap-2 ${activeTab === 'event-study'
              ? 'border-sky-600 text-sky-600 font-semibold'
              : 'border-transparent text-zinc-500 hover:text-zinc-900'
            }`}
        >
          <span>🔬 Event Study</span>
        </button>
      </div>

      {visitedTabs.has('chart') && (
        <div className={activeTab === 'chart' ? 'block animate-fade-in' : 'hidden'}>
          <EquityChartTab
            ticker={data.ticker}
            companyName={data.company_name ?? undefined}
            market={data.market}
            currentPrice={resolvedCurrentPrice}
          />
        </div>
      )}

      {visitedTabs.has('financials') && (
        <div className={activeTab === 'financials' ? 'block animate-fade-in' : 'hidden'}>
          <FinancialsTab
            ticker={data.ticker}
            market={data.market}
          />
        </div>
      )}

      {visitedTabs.has('news') && (
        <div className={activeTab === 'news' ? 'block animate-fade-in' : 'hidden'}>
          <EquityNews ticker={data.ticker} />
        </div>
      )}

      {visitedTabs.has('notes') && (
        <div className={activeTab === 'notes' ? 'block animate-fade-in' : 'hidden'}>
          <EquityNotesTab ticker={data.ticker} />
        </div>
      )}

      {visitedTabs.has('earnings-call') && (
        <div className={activeTab === 'earnings-call' ? 'block animate-fade-in' : 'hidden'}>
          <EarningsCallTab ticker={data.ticker} market={data.market} />
        </div>
      )}

      {visitedTabs.has('event-study') && (
        <div className={activeTab === 'event-study' ? 'block animate-fade-in' : 'hidden'}>
          <EventStudyTab ticker={data.ticker} market={data.market} />
        </div>
      )}

      {visitedTabs.has('overview') && (
        <div className={activeTab === 'overview' ? 'space-y-8 block animate-fade-in' : 'hidden'}>
          {/* =========================================================
              TIER 1: Executive Investment Thesis Hero Card (AI-First)
             ========================================================= */}
          <div className="space-y-4">
            <SectionWaypointHeader
              tierNumber={1}
              icon="🤖"
              title="ส่วนที่ 1: สรุปภาพรวมและคำแนะนำโดย AI"
              plainLanguageSummary="AI สังเคราะห์งบการเงินและข่าวสารทั้งหมดให้เรา สรุปเป็นคำแนะนำใน 3 วินาที (น่าสะสม / ถือรอ / ชะลอซื้อ)"
              frameworkLabel="AI Executive Command Center"
            />
            <ExecutiveThesisHero
              scorecard={scorecard}
              baseCaseSummary={data.base_case_summary}
              narrativeAnalysis={data.narrative_analysis}
              latestEarningsCall={latestEarningsCall}
              atomicSnapshot={quant.atomic_market_snapshot}
              currentPrice={resolvedCurrentPrice}
              portfolioContext={data.portfolio_context}
              thesisDelta={data.thesis_delta}
              onViewEarningsCall={() => handleTabSelect('earnings-call')}
              onOpenProvenance={() => setIsProvenanceDrawerOpen(prev => !prev)}
              hasProvenance={Boolean(evidenceSnapshot)}
              reverseDcf={reverseDcf}
              tactical={tactical}
              currencySymbol={currencySymbol}
            />

            {/* Evidence Provenance Drawer (If Opened) */}
            {isProvenanceDrawerOpen && evidenceSnapshot && (
              <div className="flow-panel rounded-2xl border border-edge/80 p-5 shadow-xs">
                <EvidenceProvenanceDrawer snapshot={evidenceSnapshot} />
              </div>
            )}

            {/* 4-Pillars Investment Committee Scorecard */}
            <InstitutionalThesisCard
              scorecard={scorecard as any}
              piotroski={quant.piotroski_breakdown as any}
              reverseDcf={reverseDcf as any}
              tactical={tactical as any}
              insider={insider as any}
              falsifiers={falsifiers as any}
              evidenceSnapshot={evidenceSnapshot as any}
            />
          </div>

          {/* =========================================================
              TIER 2: Your Position & Decision Review (Inline Section)
             ========================================================= */}
          <div className="space-y-4">
            <SectionWaypointHeader
              tierNumber={2}
              icon="💼"
              title="ส่วนที่ 2: ทบทวนสถานะพอร์ตและจิตวิทยาการลงทุน"
              plainLanguageSummary="ตรวจดูว่าหุ้นในมือเรา ราคาขึ้นเพราะกำไรแท้จริง (EPS Growth) หรือเพราะตลาดเก็งกำไรจน P/E พอง"
              frameworkLabel="Position & Decision Review"
              id="position-review-section"
            />
            <div className="flex justify-end">
              <button
                type="button"
                onClick={() => setIsPositionReviewModalOpen(true)}
                className="rounded-xl border border-sky-200 bg-sky-50 px-3 py-1.5 text-xs font-bold text-sky-700 hover:bg-sky-100 transition-colors shadow-2xs flex items-center gap-1.5"
              >
                <span>🔍</span>
                <span>เปิด Position Review (Full View)</span>
              </button>
            </div>
            <PositionReviewCard
              ticker={data.ticker}
              market={data.market}
              portfolioId={portfolioId}
            />
          </div>

          {/* =========================================================
              TIER 3: Valuation & Expectation Reality Check
             ========================================================= */}
          <div className="space-y-6">
            <SectionWaypointHeader
              tierNumber={3}
              icon="🏷️"
              title="ส่วนที่ 3: ตรวจสอบความถูกแพงและมูลค่ากิจการ"
              plainLanguageSummary="ตอบคำถามเดียว: ราคานี้เรากำลังซื้อของถูกหรือของแพง? และตลาดกำลังคาดหวังรายได้เติบโตเกินจริงหรือไม่"
              frameworkLabel="Valuation Reality Check"
            />
            {/* Infographic 2: Market Expectation vs Reality Gap (Reverse DCF) */}
            {reverseDcf && (
              <ExpectationGapCard
                marketImpliedRevenueGrowth={reverseDcf.market_implied_growth_pct}
                actualRevenueGrowth={quant.revenue_growth_yoy_pct}
                marketImpliedOperatingMargin={reverseDcf.market_implied_margin_pct}
                actualOperatingMargin={reverseDcf.reported_ebit_margin_pct ?? quant.gaap_operating_margin?.value_pct ?? quant.provider_ebit_margin?.value_pct}
                solverStatus={reverseDcf.solver_status}
                pvOf5yFcf={reverseDcf.sum_pv_5y_fcf}
                pvOfTerminalValue={reverseDcf.terminal_value_pv}
                enterpriseValue={reverseDcf.enterprise_value}
                currency={currencySymbol}
              />
            )}

            {/* Variant Perception: Two-Sided Estimate Ledger */}
            {variantPerception && (
              <VariantPerceptionCard data={variantPerception} />
            )}

            {/* Section 3B: Valuation Workbench & Quality Forensics */}
            <ValuationWorkbenchCard
              reverseDcf={reverseDcf}
              dcf={quant.dcf_result}
              piotroski={quant.piotroski_breakdown}
              dcfDiscrepancyWarning={quant.dcf_discrepancy_warning}
              gaapOperatingMargin={quant.gaap_operating_margin}
              nonGaapOperatingMargin={quant.non_gaap_operating_margin}
              providerEbitMargin={quant.provider_ebit_margin}
              valuationMarginSourceUsed={quant.valuation_margin_source_used}
              dividendYieldPct={quant.dividend_yield_pct}
              netShareholderYieldPct={quant.net_shareholder_yield_pct}
              netBuybackYieldPct={quant.net_buyback_yield_pct}
              sectorKpis={quant.sector_kpis}
              qualityMetrics={{
                roic_pct: quant.roic_pct,
                fcf_margin_pct: quant.fcf_margin_pct,
                fcf_yield_pct: quant.fcf_yield_pct,
                ocf_to_net_income: quant.ocf_to_net_income,
                solvency_score: quant.solvency_score,
              }}
              currency={currencySymbol}
            />
          </div>

          {/* =========================================================
              TIER 4: Tactical Blueprint & Smart Money Flow
             ========================================================= */}
          <div className="space-y-6">
            <SectionWaypointHeader
              tierNumber={4}
              icon="🎯"
              title="ส่วนที่ 4: จังหวะเวลาและรอยเท้าสถาบัน"
              plainLanguageSummary="หุ้นดีแต่ซื้อผิดสเตจก็ติดดอยได้: ดูวงจร 4 สเตจ, โซนราคาเข้าซื้อที่ได้เปรียบ และดูว่าผู้บริหารกำลังซื้อหรือแอบขาย"
              frameworkLabel="Tactical Blueprint & Smart Money"
            />
            <TacticalFlowMatrixCard
              tactical={tactical ? { ...tactical, current_price: resolvedCurrentPrice } : tactical}
              insider={insider}
              smartMoney={quant.smart_money_flags}
              sentiment={data.sentiment_context}
              currency={currencySymbol}
            />
          </div>

          {/* =========================================================
              TIER 5: Thesis Falsifiers & Exit Safeguards
             ========================================================= */}
          <div className="space-y-6">
            <SectionWaypointHeader
              tierNumber={5}
              icon="🛡️"
              title="ส่วนที่ 5: เกราะป้องกันความเสี่ยงและจุดตัดใจ"
              plainLanguageSummary="กฎเหล็กไม่ติดดอย: ถ้าราคาหรือมาร์จิ้นหลุดตรงไหน แปลว่าเหตุผลที่เราซื้อผิดไปแล้วและต้องยอมตัดขาดทุน"
              frameworkLabel="Risk Defense & Kill Switch"
            />

            {falsifiers.length > 0 && (
              <KillSwitchProximityMeter
                falsifiers={falsifiers}
                currentOperatingMargin={reverseDcf?.reported_ebit_margin_pct ?? quant.gaap_operating_margin?.value_pct ?? quant.provider_ebit_margin?.value_pct}
                currentPrice={resolvedCurrentPrice}
                currency={currencySymbol}
              />
            )}

            {/* Institutional Derivatives & Sell-Side Intelligence (Progressive Disclosure) */}
            {(nasdaqConsensus || strikeLadderData.length > 0) && (
              <section className="flow-panel rounded-2xl border border-edge/80 p-5 shadow-sm">
                <button
                  onClick={() => setIsDerivativesExpanded(prev => !prev)}
                  className="w-full flex items-center justify-between text-left transition-colors"
                  aria-expanded={isDerivativesExpanded}
                  aria-label="Toggle institutional derivatives"
                >
                  <div className="flex items-center gap-2">
                    <span className="text-base">⚡</span>
                    <div>
                      <h3 className="text-sm font-bold text-zinc-900">
                        Institutional Derivatives & Sell-Side Consensus
                      </h3>
                      <p className="text-xs text-zinc-500">
                        Cboe Options Open Interest Ladder, Volatility Smile & Nasdaq Analyst Consensus
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    {maxPain?.max_pain_strike != null && (
                      <span className="text-xs font-semibold px-2.5 py-0.5 rounded-full bg-zinc-100 border border-zinc-200 text-zinc-700">
                        Max Pain: ${maxPain.max_pain_strike.toFixed(2)}
                      </span>
                    )}
                    <span className="text-xs font-medium text-sky-600 hover:text-sky-700">
                      {isDerivativesExpanded ? '▲ ซ่อน' : '▼ ดูข้อมูลอนุพันธ์'}
                    </span>
                  </div>
                </button>
                {isDerivativesExpanded && (
                  <div className="mt-5 space-y-6 pt-4 border-t border-edge/60 animate-fade-in">
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-start">
                      {nasdaqConsensus && (
                        <NasdaqConsensusCard
                          symbol={nasdaqConsensus.symbol}
                          coverageStatus={nasdaqConsensus.coverage_status}
                          hasEarningsSurprise={nasdaqConsensus.has_earnings_surprise}
                          hasAnalystRatings={nasdaqConsensus.has_analyst_ratings}
                          upcomingEarnings={nasdaqConsensus.upcoming_earnings}
                          ratings={nasdaqConsensus.ratings}
                          surpriseHistory={nasdaqConsensus.surprise_history}
                        />
                      )}
                      {strikeLadderData.length > 0 && (
                        <OptionsOiStrikeLadder
                          symbol={data.ticker}
                          strikes={strikeLadderData}
                          currentPrice={resolvedCurrentPrice ?? undefined}
                          maxPainStrike={maxPain?.max_pain_strike ?? undefined}
                          expirationDate={optionsChain?.expiration_date ?? undefined}
                          putCallRatio={optionsChain?.put_call_ratio ?? undefined}
                        />
                      )}
                    </div>
                    {optionsChain?.contracts && optionsChain.contracts.length > 0 && (
                      <OptionsVolSmile
                        symbol={data.ticker}
                        expiry={optionsChain.expiration_date || 'Current Expiry'}
                        currentPrice={resolvedCurrentPrice ?? undefined}
                        contracts={optionsChain.contracts}
                        asOfDate={optionsChain.as_of}
                      />
                    )}
                  </div>
                )}
              </section>
            )}

            {/* Risk Governance & Data Safeguards */}
            {data.data_quality_flags && data.data_quality_flags.length > 0 && (
              <DataQualityFlagsCard flags={data.data_quality_flags} />
            )}
          </div>
        </div>
      )}

      {/* Footer Meta */}
      <div className="border-t border-edge pt-4 text-xs text-zinc-400 flex flex-wrap justify-between gap-y-2">
        <div className="flex items-center gap-3">
          <span>Source: {data.source_file}</span>
          <span>•</span>
          <span>Generated by: {data.generated_by}</span>
        </div>
        <div className="text-zinc-400">
          Flow Theme • Institutional Equity Suite v3.1
        </div>
      </div>

      <EquityGlossaryDrawer
        isOpen={isGlossaryOpen}
        onClose={() => setIsGlossaryOpen(false)}
      />

      <PositionReviewModal
        isOpen={isPositionReviewModalOpen}
        onClose={() => setIsPositionReviewModalOpen(false)}
        ticker={data.ticker}
        market={data.market || 'US'}
        portfolioId={portfolioId}
      />
    </div>
  )
}
