import React, { useState } from 'react'
import type { EquityDetailDTO } from '../../api/types'
import { ScoreCard } from './ScoreCard'
import ScoreRing from './ScoreRing'
import { sentimentClass } from '../../lib/sentiment'
import { EquityNews } from './EquityNews'
import { EquityNotesTab } from './EquityNotesTab'
import { EquityChartTab } from './EquityChartTab'
import { DCFScenariosChart } from './DCFScenariosChart'
import { DataQualityFlagsCard } from './DataQualityFlagsCard'

interface EquityDetailProps {
  status: 'loading' | 'error' | 'not-found' | 'success' | 'idle'
  data?: EquityDetailDTO
  errorMessage?: string
  onOpenAnalysisModal?: (ticker: string) => void
}

const eyebrowClass = 'text-xs font-semibold uppercase tracking-wider text-sky-600'

const QUANT_STAGGER_STEP_MS = 60

export const EquityDetail: React.FC<EquityDetailProps> = ({ status, data, errorMessage, onOpenAnalysisModal }) => {
  const [activeTab, setActiveTab] = useState<'overview' | 'chart' | 'news' | 'notes'>('overview')

  if (status === 'idle') {
    return null
  }

  if (status === 'loading') {
    return (
      <div className="flex justify-center items-center h-64 text-zinc-500" aria-live="polite">
        กำลังโหลดข้อมูล...
      </div>
    )
  }

  if (status === 'not-found') {
    return (
      <div className="flex flex-col justify-center items-center h-64 bg-surface rounded-xl border border-dashed border-edge p-8 text-center">
        <svg className="w-12 h-12 text-zinc-400 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 002-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
        </svg>
        <h3 className="text-lg font-medium text-zinc-900 mb-1">ไม่พบข้อมูล</h3>
        <p className="text-zinc-500 max-w-sm mb-4">
          ยังไม่มีการวิเคราะห์สำหรับหุ้นตัวนี้ กรุณาสั่งงานผ่านผู้จัดการ (Manager Agent) หรือกดปุ่มด้านล่างเพื่อวิเคราะห์ใหม่
        </p>
      </div>
    )
  }

  if (status === 'error' || !data) {
    return (
      <div className="flex flex-col justify-center items-center h-64 bg-red-50 rounded-xl border border-red-200 p-8 text-center text-red-600" role="alert">
        <h3 className="text-lg font-medium mb-1">เกิดข้อผิดพลาด</h3>
        <p>{errorMessage || 'ไม่สามารถโหลดข้อมูลได้'}</p>
      </div>
    )
  }

  return (
    <div className="animate-page-in space-y-8">
      {/* Masthead */}
      <div className="flex flex-col gap-4 border-b border-edge pb-6 sm:flex-row sm:items-start sm:justify-between">
        <div className="space-y-2">
          <div className={eyebrowClass}>Equity Report</div>
          <div className="flex items-center gap-3">
            <h2 className="font-serif text-4xl font-semibold tracking-tight text-zinc-900">
              {data.ticker} <span className="text-zinc-500 font-normal text-lg">({data.market})</span>
            </h2>
            <button
              onClick={() => onOpenAnalysisModal?.(data.ticker)}
              className="px-2.5 py-1 rounded-lg border border-sky-200 bg-sky-50 text-sky-700 hover:bg-sky-100 text-xs font-semibold flex items-center gap-1 transition-colors"
              title="วิเคราะห์ใหม่และดึงข่าวล่าสุด"
            >
              <span>🔄</span>
              <span>อัปเดตบทวิเคราะห์และข่าว</span>
            </button>
          </div>
          {data.company_name && <p className="text-zinc-500">{data.company_name}</p>}
          <div className="flex flex-wrap items-center gap-2 pt-1">
            <span className={`px-2.5 py-0.5 rounded-full border text-xs font-medium uppercase tracking-wider ${sentimentClass(data.market_sentiment)}`}>
              {data.market_sentiment}
            </span>
            <span className="text-xs text-zinc-400">
              อัปเดต {new Date(data.evaluated_at).toLocaleString('th-TH')}
            </span>
          </div>
        </div>

        <div className="flex items-center gap-3 self-start rounded-2xl border border-edge bg-panel px-5 py-4 shadow-sm shadow-black/5">
          <ScoreRing score={data.composite_score} />
          <div>
            <div className="text-sm font-semibold text-zinc-700">Composite</div>
            <div className="text-xs text-zinc-400">Quant Score</div>
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
      </div>

      {activeTab === 'chart' ? (
        <EquityChartTab
          ticker={data.ticker}
          companyName={data.company_name ?? undefined}
          market={data.market}
          currentPrice={(data.quant_signals as any)?.current_price ?? null}
        />
      ) : activeTab === 'news' ? (
        <EquityNews ticker={data.ticker} />
      ) : activeTab === 'notes' ? (
        <EquityNotesTab ticker={data.ticker} />
      ) : (


        <>
          <DataQualityFlagsCard flags={data.data_quality_flags} />

          {/* Secondary quant score rail (Composite lives in the masthead ring above) */}
          <div className="flex flex-wrap gap-4">
            {[
              { title: 'Value', icon: '💰', score: data.quant_signals.value_score, tooltip: 'ประเมินความถูกแพงของหุ้นเทียบกับปัจจัยพื้นฐาน เช่น P/E, P/BV' },
              { title: 'Growth', icon: '🌱', score: data.quant_signals.growth_score, tooltip: 'ประเมินแนวโน้มการเติบโตของรายได้และกำไรทั้งในอดีตและอนาคต' },
              { title: 'Quality', icon: '💎', score: data.quant_signals.quality_score, tooltip: 'ประเมินคุณภาพของกิจการ เช่น อัตราการทำกำไร และผลตอบแทนต่อส่วนผู้ถือหุ้น (ROE)' },
              { title: 'Momentum', icon: '🚀', score: data.quant_signals.momentum_score, tooltip: 'ประเมินความแข็งแกร่งของแนวโน้มราคาหุ้นในช่วงที่ผ่านมา' },
              { title: 'Dividend', icon: '🪙', score: data.quant_signals.dividend_score, tooltip: 'ประเมินความน่าสนใจของเงินปันผล ทั้งอัตราผลตอบแทนและความสม่ำเสมอ' },
              { title: 'Solvency', icon: '🛡️', score: data.quant_signals.solvency_score, tooltip: 'ประเมินความมั่นคงทางการเงิน ความสามารถในการชำระหนี้ และสภาพคล่อง' },
            ].map((m, i) => (
              <ScoreCard key={m.title} title={m.title} icon={m.icon} score={m.score} tooltip={m.tooltip} delayMs={i * QUANT_STAGGER_STEP_MS} />
            ))}
          </div>

          {/* DCF Valuation & Smart Money Cards */}
          {(data.quant_signals.dcf_result || data.quant_signals.smart_money_flags) && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 my-6">
              {data.quant_signals.dcf_result && (
                <div className="rounded-xl border border-edge bg-panel p-5 shadow-sm">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className={eyebrowClass}>🎯 DCF Valuation Engine</h3>
                    <span className={`px-2.5 py-0.5 rounded-full text-xs font-semibold uppercase border ${
                      data.quant_signals.dcf_result.valuation_verdict === 'undervalued'
                        ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                        : data.quant_signals.dcf_result.valuation_verdict === 'overvalued'
                        ? 'bg-rose-50 text-rose-700 border-rose-200'
                        : 'bg-amber-50 text-amber-700 border-amber-200'
                    }`}>
                      {data.quant_signals.dcf_result.valuation_verdict}
                    </span>
                  </div>
                  <DCFScenariosChart dcf={data.quant_signals.dcf_result} />
                </div>
              )}

              {data.quant_signals.smart_money_flags && (
                <div className="rounded-xl border border-edge bg-panel p-5 shadow-sm">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className={eyebrowClass}>🕵️ Smart Money Signals</h3>
                    <span className={`px-2.5 py-0.5 rounded-full text-xs font-semibold uppercase border ${
                      data.quant_signals.smart_money_flags.overall_smart_money_flag === 'bullish_signal'
                        ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                        : data.quant_signals.smart_money_flags.overall_smart_money_flag === 'bearish_signal'
                        ? 'bg-rose-50 text-rose-700 border-rose-200'
                        : 'bg-zinc-100 text-zinc-700 border-zinc-200'
                    }`}>
                      {data.quant_signals.smart_money_flags.overall_smart_money_flag}
                    </span>
                  </div>
                  <div className="space-y-2 text-sm text-zinc-600">
                    <div className="flex justify-between border-b border-edge/40 pb-1.5">
                      <span>Insider Signal (90d)</span>
                      <span className="font-semibold text-zinc-900 capitalize">{data.quant_signals.smart_money_flags.insider_signal} ({data.quant_signals.smart_money_flags.insider_buy_count_90d} Buys / {data.quant_signals.smart_money_flags.insider_sell_count_90d} Sells)</span>
                    </div>
                    <div className="flex justify-between border-b border-edge/40 pb-1.5">
                      <span>Institutional Ownership</span>
                      <span className="font-semibold text-zinc-900">{data.quant_signals.smart_money_flags.institutional_ownership_pct != null ? `${data.quant_signals.smart_money_flags.institutional_ownership_pct}%` : 'N/A'}</span>
                    </div>
                    <div className="flex justify-between border-b border-edge/40 pb-1.5">
                      <span>Insider Ownership</span>
                      <span className="font-semibold text-zinc-900">{data.quant_signals.smart_money_flags.insider_ownership_pct != null ? `${data.quant_signals.smart_money_flags.insider_ownership_pct}%` : 'N/A'}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Short Interest</span>
                      <span className={`font-semibold ${data.quant_signals.smart_money_flags.short_squeeze_risk ? 'text-amber-600' : 'text-zinc-900'}`}>
                        {data.quant_signals.smart_money_flags.short_interest_pct != null ? `${data.quant_signals.smart_money_flags.short_interest_pct}%` : 'N/A'}
                        {data.quant_signals.smart_money_flags.short_squeeze_risk && ' ⚡ Squeeze Risk'}
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Editorial reading grid: main narrative (7/12 on lg, 8/12 on xl) + sentiment rail (5/12 on lg, 4/12 on xl) */}
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-12 items-start">
            <div className="space-y-6 lg:col-span-7 xl:col-span-8">
              <section className="rounded-xl border border-edge bg-panel p-5 shadow-sm">
                <h3 className={eyebrowClass}>Base Case Summary</h3>
                <p className="mt-2.5 text-[15px] leading-relaxed text-zinc-700 whitespace-pre-line">{data.base_case_summary}</p>
              </section>

              <section className="rounded-xl border border-edge bg-panel p-5 shadow-sm">
                <h3 className={eyebrowClass}>Narrative Analysis</h3>
                <p className="mt-2.5 text-[15px] leading-relaxed text-zinc-700 whitespace-pre-line">{data.narrative_analysis}</p>
              </section>
            </div>

            <div className="lg:col-span-5 xl:col-span-4">
              <div className="space-y-4 rounded-xl border border-edge bg-panel p-5 shadow-sm">
                <h3 className={eyebrowClass}>Sentiment Context</h3>

                {data.sentiment_context.key_themes && data.sentiment_context.key_themes.length > 0 && (
                  <div>
                    <span className="text-xs font-medium text-zinc-500 block mb-1.5">Key Themes</span>
                    <div className="flex flex-wrap gap-1.5">
                      {data.sentiment_context.key_themes.map((t, i) => (
                        <span key={i} className="rounded-full border border-edge bg-surface-strong px-2.5 py-0.5 text-xs font-medium text-zinc-700">
                          {t}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {data.sentiment_context.tail_risks && data.sentiment_context.tail_risks.length > 0 && (
                  <div className="rounded-lg border-l-4 border-red-300 bg-red-50/70 p-3">
                    <span className="text-xs font-semibold text-red-800 block mb-1">Tail Risks</span>
                    <ul className="list-disc pl-4 text-sm text-red-700 space-y-0.5">
                      {data.sentiment_context.tail_risks.map((t, i) => <li key={i}>{t}</li>)}
                    </ul>
                  </div>
                )}

                <div>
                  <span className="text-xs font-medium text-zinc-500 block mb-1">Sources Summary</span>
                  <p className="text-zinc-600 text-sm whitespace-pre-line">{data.sentiment_context.sources_summary}</p>
                </div>
              </div>
            </div>
          </div>
        </>
      )}

      <div className="border-t border-edge pt-4 text-xs text-zinc-400 flex flex-wrap gap-x-4">
        <div>Source: {data.source_file}</div>
        <div>Generated by: {data.generated_by}</div>
      </div>
    </div>
  )
}
