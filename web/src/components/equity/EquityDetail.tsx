import React, { useState, useEffect } from 'react'
import { api } from '../../api/client'
import type { EquityDetailDTO, EarningsCallNoteItem } from '../../api/types'
import { ScoreCard } from './ScoreCard'
import ScoreRing from './ScoreRing'
import { sentimentClass } from '../../lib/sentiment'
import { EquityNews } from './EquityNews'
import { EquityNotesTab } from './EquityNotesTab'
import { EquityChartTab } from './EquityChartTab'
import { FinancialsTab } from './FinancialsTab'
import { DataQualityFlagsCard } from './DataQualityFlagsCard'
import { EarningsCallTab } from './EarningsCallTab'
import ExecutiveThesisHero, { getStanceBadgeStyle } from './ExecutiveThesisHero'
import ValuationWorkbenchCard from './ValuationWorkbenchCard'
import TacticalFlowMatrixCard from './TacticalFlowMatrixCard'
import { EvidenceProvenanceDrawer } from './EvidenceProvenanceDrawer'

interface EquityDetailProps {
  status: 'loading' | 'error' | 'not-found' | 'success' | 'idle'
  data?: EquityDetailDTO
  errorMessage?: string
  onOpenAnalysisModal?: (ticker: string, market?: 'US' | 'TH') => void
  isUpdating?: boolean
}

const eyebrowClass = 'text-xs font-bold uppercase tracking-wider text-sky-700/80'
const QUANT_STAGGER_STEP_MS = 50

export const EquityDetail: React.FC<EquityDetailProps> = ({ status, data, errorMessage, onOpenAnalysisModal, isUpdating }) => {
  const [activeTab, setActiveTab] = useState<'overview' | 'chart' | 'financials' | 'news' | 'notes' | 'earnings-call'>('overview')
  const [latestEarningsCall, setLatestEarningsCall] = useState<EarningsCallNoteItem | null>(null)
  const [isFactorsExpanded, setIsFactorsExpanded] = useState(false)
  const [isProvenanceDrawerOpen, setIsProvenanceDrawerOpen] = useState(false)

  useEffect(() => {
    if (!data?.ticker) return
    api.getEarningsCalls(data.ticker)
      .then((res) => {
        if (res.items && res.items.length > 0) {
          setLatestEarningsCall(res.items[0] ?? null)
        } else {
          setLatestEarningsCall(null)
        }
      })
      .catch(() => {
        setLatestEarningsCall(null)
      })
  }, [data?.ticker])

  if (status === 'idle') {
    return null
  }

  if (status === 'loading') {
    return (
      <div className="flex justify-center items-center h-64 text-zinc-500" aria-live="polite">
        <div className="flex items-center gap-3">
          <span className="w-5 h-5 border-2 border-sky-600 border-t-transparent rounded-full animate-spin" />
          <span>กำลังโหลดบทวิเคราะห์หุ้น...</span>
        </div>
      </div>
    )
  }

  if (status === 'not-found') {
    return (
      <div className="flex flex-col justify-center items-center h-64 bg-surface rounded-2xl border border-dashed border-edge p-8 text-center shadow-sm">
        <svg className="w-12 h-12 text-zinc-400 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 002-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
        </svg>
        <h3 className="text-lg font-medium text-zinc-900 mb-1">ไม่พบข้อมูล</h3>
        <p className="text-zinc-500 max-w-sm mb-4 text-sm">
          ยังไม่มีการวิเคราะห์สำหรับหุ้นตัวนี้ กรุณากดปุ่มด้านล่างเพื่อสั่ง Manager Agent เริ่มการวิเคราะห์ระดับสูง
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
              className={`px-3 py-1 rounded-xl border text-xs font-semibold flex items-center gap-1.5 transition-colors shadow-2xs ${
                isUpdating
                  ? 'border-amber-200 bg-amber-50 text-amber-700 opacity-80 cursor-not-allowed'
                  : 'border-sky-200 bg-sky-50 text-sky-700 hover:bg-sky-100'
              }`}
              title={isUpdating ? 'กำลังประมวลผลการวิเคราะห์...' : 'วิเคราะห์ใหม่และดึงข่าวล่าสุด'}
            >
              <span className={isUpdating ? 'inline-block animate-spin' : ''}>🔄</span>
              <span>{isUpdating ? 'กำลังวิเคราะห์...' : 'อัปเดตบทวิเคราะห์'}</span>
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

        {/* Hero Action & Valuation Strip */}
        <div className="flex flex-wrap items-center gap-4 self-start rounded-2xl border border-edge/80 bg-panel px-5 py-4 shadow-sm shadow-black/5">
          {/* Action Stance */}
          {scorecard?.action_stance && (
            <div className="flex flex-col items-start pr-3 border-r border-edge/60">
              <span className="text-[10px] font-bold text-zinc-400 uppercase tracking-wider">Action Stance</span>
              <div className={`mt-1 inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full border text-xs font-bold ${getStanceBadgeStyle(scorecard.action_stance).bg}`}>
                <span className={`w-2 h-2 rounded-full ${getStanceBadgeStyle(scorecard.action_stance).dot}`} />
                <span>{scorecard.action_stance.replace(/_/g, ' ')}</span>
              </div>
            </div>
          )}

          {/* 12M Target Price */}
          {(reverseDcf?.target_price_12m != null || quant.upside_pct != null) && (
            <div className="flex flex-col items-start pr-3 border-r border-edge/60">
              <span className="text-[10px] font-bold text-zinc-400 uppercase tracking-wider">12M Target Price</span>
              <div className="flex items-center gap-1.5 text-sm font-bold text-zinc-900 mt-0.5">
                <span>
                  {currencySymbol}
                  {(reverseDcf?.target_price_12m ?? quant.atomic_market_snapshot?.analysis_price ?? tactical?.current_price ?? 0).toFixed(2)}
                </span>
                {reverseDcf?.upside_12m_pct != null && (
                  <span
                    className={`text-[11px] px-1.5 py-0.2 rounded-full font-bold border ${
                      reverseDcf.upside_12m_pct >= 0
                        ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                        : 'bg-rose-50 text-rose-700 border-rose-200'
                    }`}
                  >
                    {reverseDcf.upside_12m_pct >= 0 ? '+' : ''}
                    {reverseDcf.upside_12m_pct.toFixed(1)}%
                  </span>
                )}
              </div>
            </div>
          )}

          {/* Conviction Score & Composite Mini Ring */}
          <div className="flex items-center gap-3">
            {scorecard?.core_conviction_score != null && (
              <div className="text-left pr-3 border-r border-edge/60">
                <div className="text-[10px] font-bold text-zinc-400 uppercase tracking-wider">Conviction</div>
                <div className="text-sm font-bold text-sky-700 mt-0.5">
                  {scorecard.core_conviction_score.toFixed(1)} <span className="text-[10px] text-zinc-400 font-normal">/ 10</span>
                </div>
              </div>
            )}

            <div className="flex items-center gap-2">
              <ScoreRing score={data.composite_score} />
              <div>
                <div className="text-[10px] font-bold text-zinc-400 uppercase">Composite</div>
                <div className="text-xs font-semibold text-zinc-700">
                  {data.composite_score != null ? `${data.composite_score.toFixed(1)}` : 'N/A'}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Sub-nav Tab Switcher */}
      <div className="flex border-b border-edge gap-6 text-sm font-medium">
        <button
          onClick={() => setActiveTab('overview')}
          className={`pb-3 border-b-2 transition-colors flex items-center gap-2 ${
            activeTab === 'overview'
              ? 'border-sky-600 text-sky-600 font-semibold'
              : 'border-transparent text-zinc-500 hover:text-zinc-900'
          }`}
        >
          <span>📊 Overview</span>
        </button>
        <button
          onClick={() => setActiveTab('chart')}
          className={`pb-3 border-b-2 transition-colors flex items-center gap-2 ${
            activeTab === 'chart'
              ? 'border-sky-600 text-sky-600 font-semibold'
              : 'border-transparent text-zinc-500 hover:text-zinc-900'
          }`}
        >
          <span>📈 Chart</span>
        </button>
        <button
          onClick={() => setActiveTab('financials')}
          className={`pb-3 border-b-2 transition-colors flex items-center gap-2 ${
            activeTab === 'financials'
              ? 'border-sky-600 text-sky-600 font-semibold'
              : 'border-transparent text-zinc-500 hover:text-zinc-900'
          }`}
        >
          <span>📑 Financials</span>
        </button>
        <button
          onClick={() => setActiveTab('news')}
          className={`pb-3 border-b-2 transition-colors flex items-center gap-2 ${
            activeTab === 'news'
              ? 'border-sky-600 text-sky-600 font-semibold'
              : 'border-transparent text-zinc-500 hover:text-zinc-900'
          }`}
        >
          <span>📰 News</span>
        </button>
        <button
          onClick={() => setActiveTab('notes')}
          className={`pb-3 border-b-2 transition-colors flex items-center gap-2 ${
            activeTab === 'notes'
              ? 'border-sky-600 text-sky-600 font-semibold'
              : 'border-transparent text-zinc-500 hover:text-zinc-900'
          }`}
        >
          <span>📓 Notes</span>
        </button>
        <button
          onClick={() => setActiveTab('earnings-call')}
          className={`pb-3 border-b-2 transition-colors flex items-center gap-2 ${
            activeTab === 'earnings-call'
              ? 'border-sky-600 text-sky-600 font-semibold'
              : 'border-transparent text-zinc-500 hover:text-zinc-900'
          }`}
        >
          <span>🎙️ Earnings Call</span>
        </button>
      </div>

      {activeTab === 'chart' ? (
        <EquityChartTab
          ticker={data.ticker}
          companyName={data.company_name ?? undefined}
          market={data.market}
          currentPrice={(quant as any)?.current_price ?? null}
        />
      ) : activeTab === 'financials' ? (
        <FinancialsTab
          ticker={data.ticker}
          market={data.market}
        />
      ) : activeTab === 'news' ? (
        <EquityNews ticker={data.ticker} />
      ) : activeTab === 'notes' ? (
        <EquityNotesTab ticker={data.ticker} />
      ) : activeTab === 'earnings-call' ? (
        <EarningsCallTab ticker={data.ticker} market={data.market} />
      ) : (
        <div className="space-y-8">
          {/* =========================================================
              TIER 1: Executive Investment Thesis Hero Card
             ========================================================= */}
          <ExecutiveThesisHero
            scorecard={scorecard}
            baseCaseSummary={data.base_case_summary}
            latestEarningsCall={latestEarningsCall}
            atomicSnapshot={quant.atomic_market_snapshot}
            onViewEarningsCall={() => setActiveTab('earnings-call')}
            onOpenProvenance={() => setIsProvenanceDrawerOpen(prev => !prev)}
            hasProvenance={Boolean(evidenceSnapshot)}
          />

          {/* Evidence Provenance Drawer (If Opened) */}
          {isProvenanceDrawerOpen && evidenceSnapshot && (
            <div className="flow-panel rounded-2xl border border-edge/80 p-5 shadow-xs">
              <EvidenceProvenanceDrawer snapshot={evidenceSnapshot} />
            </div>
          )}

          {/* =========================================================
              TIER 2: 2-Column Core Analytical Grid
             ========================================================= */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-start">
            {/* Left Column: Valuation Workbench & Quality Forensics */}
            <ValuationWorkbenchCard
              reverseDcf={reverseDcf}
              dcf={quant.dcf_result}
              piotroski={quant.piotroski_breakdown}
              dcfDiscrepancyWarning={quant.dcf_discrepancy_warning}
              qualityMetrics={{
                roic_pct: quant.roic_pct,
                fcf_margin_pct: quant.fcf_margin_pct,
                fcf_yield_pct: quant.fcf_yield_pct,
                ocf_to_net_income: quant.ocf_to_net_income,
                solvency_score: quant.solvency_score,
              }}
              currency={currencySymbol}
            />

            {/* Right Column: Tactical Blueprint & Smart Money Flow */}
            <TacticalFlowMatrixCard
              tactical={tactical}
              insider={insider}
              smartMoney={quant.smart_money_flags}
              sentiment={data.sentiment_context}
              currency={currencySymbol}
            />
          </div>

          {/* Editorial Reading Section: Deep Narrative Analysis */}
          {data.narrative_analysis && (
            <section className="flow-panel rounded-2xl border border-edge/80 p-6 shadow-sm">
              <div className="flex items-center justify-between border-b border-edge/60 pb-3 mb-4">
                <h3 className={eyebrowClass}>📝 In-Depth Narrative Analysis</h3>
                <span className="text-xs text-zinc-400">สังเคราะห์ข้อมูลเชิงคุณภาพและข่าวสาร</span>
              </div>
              <div className="prose prose-sm max-w-none text-zinc-800 leading-relaxed whitespace-pre-line text-[14px]">
                {data.narrative_analysis}
              </div>
            </section>
          )}

          {/* =========================================================
              TIER 3: Risk Governance & Factor Deep-Dive (Collapsible)
             ========================================================= */}
          <div className="space-y-6">
            {/* Thesis Falsifiers (Kill-Switches Table) */}
            {falsifiers.length > 0 && (
              <section className="flow-panel rounded-2xl border border-rose-200/80 bg-rose-50/20 p-6 shadow-sm">
                <div className="flex items-center justify-between border-b border-rose-200/60 pb-3 mb-4">
                  <div className="flex items-center gap-2">
                    <span className="text-lg">🛑</span>
                    <h3 className="text-sm font-bold uppercase tracking-wider text-rose-900">
                      Thesis Falsifiers & Invalidation Criteria (Kill-Switches)
                    </h3>
                  </div>
                  <span className="text-xs text-rose-700/80 font-medium">เกณฑ์ยกเลิกสมมติฐานการลงทุน</span>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs border-collapse">
                    <thead>
                      <tr className="border-b border-rose-200/80 text-rose-800 font-semibold">
                        <th className="pb-2 pl-1">ID</th>
                        <th className="pb-2">Metric / Condition</th>
                        <th className="pb-2">Threshold</th>
                        <th className="pb-2">Source Ref</th>
                        <th className="pb-2 pr-1">คำอธิบายความเสี่ยง</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-rose-100 text-zinc-800">
                      {falsifiers.map((f: any) => (
                        <tr key={f.falsifier_id} className="hover:bg-rose-50/40">
                          <td className="py-2.5 pl-1 font-mono font-bold text-rose-700">{f.falsifier_id}</td>
                          <td className="py-2.5 font-medium">{f.metric_name}: <span className="text-rose-600">{f.condition}</span></td>
                          <td className="py-2.5 font-semibold text-zinc-900">{f.threshold_value != null ? `${f.threshold_value}` : 'N/A'}</td>
                          <td className="py-2.5 text-zinc-500 font-mono text-[11px]">{f.source_ref || 'N/A'}</td>
                          <td className="py-2.5 pr-1 text-zinc-600 leading-relaxed">{f.narrative_explanation}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>
            )}

            {/* Data Quality Flags */}
            <DataQualityFlagsCard flags={data.data_quality_flags} />

            {/* Collapsible Factor Deep-Dive (6 Quant ScoreCards) */}
            <section className="flow-panel rounded-2xl border border-edge/80 p-5 shadow-sm">
              <button
                onClick={() => setIsFactorsExpanded(prev => !prev)}
                className="w-full flex items-center justify-between text-left transition-colors"
                aria-expanded={isFactorsExpanded}
                aria-label="Toggle factor breakdown"
              >
                <div className="flex items-center gap-2">
                  <span className="text-base">💎</span>
                  <div>
                    <h3 className="text-sm font-bold text-zinc-900">
                      6-Factor Quantitative Breakdown
                    </h3>
                    <p className="text-xs text-zinc-500">
                      คะแนนปัจจัยพื้นฐานทั้ง 6 เสาหลัก (Value, Growth, Quality, Momentum, Dividend, Solvency)
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-2 text-xs font-semibold text-sky-700 bg-surface px-3 py-1.5 rounded-xl border border-edge">
                  <span>{isFactorsExpanded ? 'พับเก็บ' : 'ขยายดูรายละเอียด'}</span>
                  <span>{isFactorsExpanded ? '▲' : '▼'}</span>
                </div>
              </button>

              {isFactorsExpanded && (
                <div className="mt-5 pt-4 border-t border-edge/60 flex flex-wrap gap-4 animate-card-in">
                  {[
                    { title: 'Value', icon: '💰', score: quant.value_score, tooltip: 'ประเมินความถูกแพงของหุ้นเทียบกับปัจจัยพื้นฐาน เช่น P/E, P/BV' },
                    { title: 'Growth', icon: '🌱', score: quant.growth_score, tooltip: 'ประเมินแนวโน้มการเติบโตของรายได้และกำไรทั้งในอดีตและอนาคต' },
                    { title: 'Quality', icon: '💎', score: quant.quality_score, tooltip: 'ประเมินคุณภาพของกิจการ เช่น อัตราการทำกำไร และผลตอบแทนต่อส่วนผู้ถือหุ้น (ROE)' },
                    { title: 'Momentum', icon: '🚀', score: quant.momentum_score, tooltip: 'ประเมินความแข็งแกร่งของแนวโน้มราคาหุ้นในช่วงที่ผ่านมา' },
                    { title: 'Dividend', icon: '🪙', score: quant.dividend_score, tooltip: 'ประเมินความน่าสนใจของเงินปันผล ทั้งอัตราผลตอบแทนและความสม่ำเสมอ' },
                    { title: 'Solvency', icon: '🛡️', score: quant.solvency_score, tooltip: 'ประเมินความมั่นคงทางการเงิน ความสามารถในการชำระหนี้ และสภาพคล่อง' },
                  ].map((m, i) => (
                    <ScoreCard
                      key={m.title}
                      title={m.title}
                      icon={m.icon}
                      score={m.score}
                      tooltip={m.tooltip}
                      delayMs={i * QUANT_STAGGER_STEP_MS}
                    />
                  ))}
                </div>
              )}
            </section>
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
    </div>
  )
}
