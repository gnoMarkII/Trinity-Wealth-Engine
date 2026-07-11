import { useEffect, useState } from 'react'
import { api, ApiError } from '../api/client'
import type { MacroDashboardDTO } from '../api/types'
import WarningPanel from '../components/WarningPanel'
import RegimeProbabilityChart from '../components/RegimeProbabilityChart'
import TradingViewMiniWidget from '../components/TradingViewMiniWidget'
import PortfolioStanceBar from '../components/PortfolioStanceBar'
import MacroReferenceDrawer from '../components/MacroReferenceDrawer'
import MacroContentReferences from '../components/MacroContentReferences'
import MacroIndicatorPanel from '../components/MacroIndicatorPanel'
import { stanceCategory, type StanceCategory } from '../lib/stance'

const STANCE_CLASS: Record<StanceCategory, string> = {
  overweight: 'bg-emerald-50 text-emerald-700 border-emerald-200',
  underweight: 'bg-red-50 text-red-700 border-red-200',
  neutral: 'bg-zinc-100 text-zinc-700 border-zinc-200',
}

function stanceClass(stance: string): string {
  return STANCE_CLASS[stanceCategory(stance)]
}

function confidenceBadgeClass(confidence: string): string {
  const c = confidence.toLowerCase()
  if (c === 'high') return 'border-emerald-200 bg-emerald-50 text-emerald-700'
  if (c === 'medium') return 'border-amber-200 bg-amber-50 text-amber-700'
  if (c === 'low') return 'border-red-200 bg-red-50 text-red-700'
  return 'border-zinc-200 bg-surface text-zinc-600'
}

const cardClass =
  'space-y-3 rounded-xl border border-sky-100 bg-white/80 p-4 shadow-[0_8px_26px_rgba(14,165,233,0.05)] backdrop-blur-sm transition-all duration-150 hover:border-sky-200 hover:shadow-md'

type StanceFilter = 'all' | 'overweight' | 'underweight' | 'neutral'

const STANCE_FILTER_TABS: { key: StanceFilter; label: string }[] = [
  { key: 'all', label: 'ทั้งหมด' },
  { key: 'overweight', label: 'Overweight' },
  { key: 'underweight', label: 'Underweight' },
  { key: 'neutral', label: 'Neutral' },
]

export default function Macro() {
  const [data, setData] = useState<MacroDashboardDTO | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [stanceFilter, setStanceFilter] = useState<StanceFilter>('all')
  const [isRefDrawerOpen, setIsRefDrawerOpen] = useState(false)

  const handleStanceFilterChange = (nextFilter: StanceFilter) => {
    if (typeof document !== 'undefined' && 'startViewTransition' in document) {
      document.startViewTransition(() => setStanceFilter(nextFilter))
    } else {
      setStanceFilter(nextFilter)
    }
  }

  useEffect(() => {
    api
      .getMacroDashboard()
      .then(setData)
      .catch((e) => setError(e instanceof ApiError ? e.message : 'โหลดข้อมูลเศรษฐกิจมหภาคไม่สำเร็จ'))
  }, [])

  if (error) return <p className="text-sm text-red-600">{error}</p>
  if (!data) {
    return (
      <div className="animate-page-in space-y-6 pb-10">
        <div className="animate-shimmer h-40 rounded-2xl border border-zinc-200/80" />
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
          <div className="space-y-6 lg:col-span-5">
            <div className="animate-shimmer h-64 rounded-xl border border-zinc-200/80" />
            <div className="animate-shimmer h-40 rounded-xl border border-zinc-200/80" />
          </div>
          <div className="space-y-6 lg:col-span-7">
            <div className="animate-shimmer h-96 rounded-xl border border-zinc-200/80" />
          </div>
        </div>
      </div>
    )
  }

  const assetAllocations = data.asset_allocation ?? []
  const pairTrades = data.pair_trades ?? []
  const riskScenarios = data.risk_scenarios ?? []

  const filteredAssets = assetAllocations.filter(
    (a) => stanceFilter === 'all' || stanceCategory(a.stance) === stanceFilter
  )

  return (
    <div className="animate-page-in space-y-6 pb-10">
      {/* Executive Summary Banner */}
      <div className="rounded-2xl border border-sky-100 bg-gradient-to-br from-white/90 via-sky-50/60 to-orange-50/55 p-6 shadow-[0_10px_35px_rgba(14,165,233,0.06)] backdrop-blur-sm">
        <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
          <div className="space-y-2">
            <div className="flex flex-wrap items-center gap-2">
              <span className="rounded-lg border border-zinc-900 bg-zinc-900 px-3 py-1 text-sm font-bold tracking-wide text-white">
                {data.overall_regime}
              </span>
              {data.conviction_level && (
                <span className="rounded-lg border border-amber-300 bg-amber-100/80 px-2.5 py-1 text-xs font-semibold uppercase text-amber-900">
                  Conviction: {data.conviction_level}
                </span>
              )}
              {data.quant_narrative_alignment && (
                <span className="rounded-lg border border-zinc-200 bg-white px-2.5 py-1 text-xs font-medium text-zinc-700">
                  Alignment: {data.quant_narrative_alignment}
                </span>
              )}
              <span className="rounded-lg border border-zinc-200 bg-white px-2.5 py-1 text-xs font-medium text-zinc-600">
                Horizon: {data.time_horizon}
              </span>
            </div>

            {data.conviction_rationale && (
              <p className="max-w-4xl text-sm leading-relaxed text-zinc-700">{data.conviction_rationale}</p>
            )}

            {data.focus_themes && data.focus_themes.length > 0 && (
              <div className="flex flex-wrap items-center gap-1.5 pt-1">
                <span className="text-xs font-medium text-zinc-500">Focus Themes:</span>
                {data.focus_themes.map((theme, i) => (
                  <span
                    key={i}
                    className="rounded-full border border-zinc-200 bg-zinc-100/80 px-2.5 py-0.5 text-xs font-medium text-zinc-700"
                  >
                    {theme}
                  </span>
                ))}
              </div>
            )}
          </div>

          <div className="flex flex-col items-end justify-between gap-2">
            <button
              onClick={() => setIsRefDrawerOpen(true)}
              className="flex items-center gap-1.5 rounded-xl border border-zinc-200 bg-white px-3.5 py-2 text-xs font-semibold text-zinc-800 shadow-sm transition-all hover:bg-zinc-50 hover:shadow"
            >
              <span>📚</span>
              <span>แหล่งอ้างอิงข้อมูล (References)</span>
            </button>
            {data.evaluated_at && (
              <div className="text-right text-xs text-zinc-400">
                Evaluated: {data.evaluated_at}
              </div>
            )}
          </div>
        </div>
      </div>

      <WarningPanel warnings={data.warnings} />

      {data.dashboard_indicators && data.dashboard_indicators.length > 0 && (
        <MacroIndicatorPanel indicators={data.dashboard_indicators} />
      )}

      {data.report_references && data.report_references.length > 0 && (
        <MacroContentReferences references={data.report_references} />
      )}

      {/* 2-Column Responsive Layout: Left (5/12) Macro & Regime | Right (7/12) Strategic Allocation */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
        {/* Left Column: Macro & Regime Analysis */}
        <div className="space-y-6 lg:col-span-5">
          <div className="rounded-xl border border-zinc-200/80 bg-white p-5 shadow-sm shadow-black/5">
            <h2 className="mb-4 text-base font-semibold text-zinc-900">Regime Probabilities</h2>
            <RegimeProbabilityChart probabilities={data.regime_probabilities} />
          </div>

          {data.key_assumptions && data.key_assumptions.length > 0 && (
            <div className="rounded-xl border border-zinc-200/80 bg-white p-5 shadow-sm shadow-black/5">
              <h2 className="mb-3 text-base font-semibold text-zinc-900">Key Macro Assumptions</h2>
              <ul className="space-y-2 text-sm text-zinc-600">
                {data.key_assumptions.map((assumption, idx) => (
                  <li key={idx} className="flex items-start gap-2">
                    <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-amber-600" />
                    <span>{assumption}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {data.regime_evidence.length > 0 && (
            <div className="rounded-xl border border-zinc-200/80 bg-white p-5 shadow-sm shadow-black/5">
              <h2 className="mb-3 text-base font-semibold text-zinc-900">5-Dimension Evidence</h2>
              <div className="space-y-3">
                {data.regime_evidence.map((re, idx) => (
                  <div key={idx} className="rounded-lg border border-zinc-100 bg-zinc-50/60 p-3">
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-xs font-semibold uppercase tracking-wide text-zinc-800">
                        {re.dimension}
                      </span>
                      <span
                        className={`rounded-full border px-2 py-0.5 text-[10px] font-semibold ${confidenceBadgeClass(
                          re.confidence
                        )}`}
                      >
                        {re.confidence}
                      </span>
                    </div>
                    <div className="mt-1 text-sm font-medium text-zinc-900">{re.signal}</div>
                    <p className="mt-1 text-xs text-zinc-600">{re.evidence}</p>
                    {re.conflict && (
                      <div className="mt-2 rounded bg-red-50 px-2 py-1 text-xs text-red-700">
                        Conflict: {re.conflict}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="rounded-xl border border-zinc-200/80 bg-white p-5 shadow-sm shadow-black/5">
            <h2 className="mb-3 text-base font-semibold text-zinc-900">Liquidity & Structural Rate</h2>
            <TradingViewMiniWidget symbol="TVC:US10Y" title="US 10-Year Treasury Yield" />
          </div>

          {/* Hedging / Tail Risk Scenarios Section */}
          {riskScenarios.length > 0 && (
            <div className="content-auto rounded-xl border border-zinc-200/80 bg-white p-5 shadow-sm shadow-black/5">
              <h2 className="text-balance mb-1 text-base font-semibold text-zinc-900">Hedging & Tail Risk Scenarios</h2>
              <p className="text-pretty mb-4 text-xs text-zinc-500">แผนป้องกันความเสี่ยงและเงื่อนไขการเปิด/ปิดสถานะป้องกัน</p>
              <div className="grid grid-cols-1 gap-4">
                {riskScenarios.map((rs, idx) => (
                  <div key={idx} className={cardClass}>
                    <h3 className="font-semibold text-zinc-900">{rs.tail_risk}</h3>

                    {rs.mitigation_strategy && (
                      <p className="text-sm text-zinc-600">{rs.mitigation_strategy}</p>
                    )}

                    <div className="rounded-lg bg-amber-50/70 p-2.5 text-xs text-amber-900">
                      <div><span className="font-semibold">Trigger:</span> {rs.trigger_to_activate}</div>
                      {rs.unwind_or_cover_condition && (
                        <div className="mt-1"><span className="font-semibold">Unwind:</span> {rs.unwind_or_cover_condition}</div>
                      )}
                    </div>

                    <div className="flex flex-wrap gap-x-3 gap-y-1 text-xs text-zinc-500">
                      <span>Prob: {rs.probability}</span>
                      <span>Impact: {rs.impact}</span>
                      {rs.hedge_size && <span>Size: {rs.hedge_size}</span>}
                    </div>

                    {rs.early_warning_indicators && rs.early_warning_indicators.length > 0 && (
                      <div className="space-y-1 rounded-lg bg-zinc-50 p-2 text-xs text-zinc-600">
                        <span className="font-semibold text-zinc-700">Early Warning Indicators:</span>
                        <ul className="list-inside list-disc">
                          {rs.early_warning_indicators.map((ewi, k) => (
                            <li key={k}>{ewi}</li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {rs.hedge_instruments.length > 0 && (
                      <p className="text-xs font-medium text-zinc-700">
                        Hedge Instruments: <span className="text-zinc-600">{rs.hedge_instruments.join(', ')}</span>
                      </p>
                    )}

                    {rs.cost_or_tradeoff && (
                      <p className="text-xs text-zinc-500">Cost/Tradeoff: {rs.cost_or_tradeoff}</p>
                    )}

                    <WarningPanel warnings={rs.warnings} compact />
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Right Column: Strategic Allocation & Action Plan */}
        <div className="space-y-6 lg:col-span-7">
          {/* Cross-Asset Allocation Section */}
          <div className="rounded-xl border border-zinc-200/80 bg-white p-5 shadow-sm shadow-black/5">
            <div className="mb-4 flex flex-col justify-between gap-3 sm:flex-row sm:items-center">
              <div>
                <h2 className="text-balance text-base font-semibold text-zinc-900">Cross-Asset Allocation Strategy</h2>
                <p className="text-pretty text-xs text-zinc-500">คำแนะนำจัดสรรสัดส่วนสินทรัพย์ตามภาพรวมสภาวะเศรษฐกิจ</p>
              </div>
              <div className="flex w-fit gap-1 rounded-lg border border-zinc-200 bg-zinc-50 p-1">
                {STANCE_FILTER_TABS.map((t) => (
                  <button
                    key={t.key}
                    onClick={() => handleStanceFilterChange(t.key)}
                    aria-pressed={stanceFilter === t.key}
                    className={`rounded-md px-2.5 py-1 text-xs font-medium transition-colors ${
                      stanceFilter === t.key
                        ? 'bg-zinc-900 text-white shadow-sm'
                        : 'text-zinc-600 hover:text-zinc-900'
                    }`}
                  >
                    {t.label}
                  </button>
                ))}
              </div>
            </div>

            {assetAllocations.length > 0 && (
              <div className="mb-5 rounded-xl border border-zinc-200/80 bg-zinc-50/50 p-4">
                <PortfolioStanceBar allocations={assetAllocations} />
              </div>
            )}

            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              {filteredAssets.map((a, idx) => (
                <div key={idx} className={cardClass}>
                  <div className="flex items-start justify-between gap-2">
                    <div>
                      <h3 className="font-semibold text-zinc-900">{a.asset_class}</h3>
                      {a.asset_bucket && (
                        <span className="text-[11px] font-medium uppercase tracking-wider text-zinc-400">
                          {a.asset_bucket}
                        </span>
                      )}
                    </div>
                    <span
                      className={`rounded-lg border px-2.5 py-0.5 text-xs font-bold uppercase tracking-wide ${stanceClass(
                        a.stance
                      )}`}
                    >
                      {a.stance}
                    </span>
                  </div>

                  <p className="text-pretty text-sm leading-relaxed text-zinc-600">{a.rationale}</p>

                  <div className="flex flex-wrap gap-x-3 gap-y-1 text-xs font-medium text-zinc-500">
                    <span>Confidence: {a.confidence}</span>
                    {a.allocation_delta && <span>Delta: {a.allocation_delta}</span>}
                  </div>

                  {a.supporting_data.length > 0 && (
                    <ul className="space-y-1 rounded-lg bg-zinc-50 p-2.5 text-xs text-zinc-600">
                      {a.supporting_data.map((sd, j) => (
                        <li key={j} className="flex items-center gap-1.5">
                          <span className="h-1 w-1 rounded-full bg-zinc-400" />
                          <span>{sd}</span>
                        </li>
                      ))}
                    </ul>
                  )}

                  {a.why_not_high && (
                    <p className="text-xs italic text-amber-700/80">
                      Why not HIGH: {a.why_not_high}
                    </p>
                  )}

                  <WarningPanel warnings={a.warnings} compact />
                </div>
              ))}
            </div>
          </div>

          {/* Tactical Pair Trades Section */}
          {pairTrades.length > 0 && (
            <div className="content-auto rounded-xl border border-zinc-200/80 bg-white p-5 shadow-sm shadow-black/5">
              <h2 className="text-balance mb-1 text-base font-semibold text-zinc-900">Tactical Pair Trades</h2>
              <p className="text-pretty mb-4 text-xs text-zinc-500">กลยุทธ์จับคู่ Long/Short เพื่อสร้างผลตอบแทนสัมพันธ์กับสภาวะตลาด</p>
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                {pairTrades.map((pt, idx) => (
                  <div key={idx} className={cardClass}>
                    <div className="flex items-start justify-between gap-2">
                      <h3 className="text-balance font-semibold text-zinc-900">
                        Long <span className="text-emerald-600">{pt.long_leg}</span> / Short{' '}
                        <span className="text-red-600">{pt.short_leg}</span>
                      </h3>
                      {pt.confidence && (
                        <span className="rounded-md bg-zinc-100 px-2 py-0.5 text-[10px] font-semibold text-zinc-700">
                          {pt.confidence}
                        </span>
                      )}
                    </div>
                    <p className="text-pretty text-sm text-zinc-600">{pt.thesis}</p>

                    {pt.implementation_idea && (
                      <div className="rounded-lg bg-emerald-50/60 p-2 text-xs font-medium text-emerald-900">
                        Idea: {pt.implementation_idea}
                      </div>
                    )}

                    <div className="space-y-1 rounded-lg bg-zinc-50 p-2.5 text-xs text-zinc-600">
                      {pt.catalyst && <div><span className="font-semibold text-zinc-800">Catalyst:</span> {pt.catalyst}</div>}
                      {pt.risk && <div><span className="font-semibold text-zinc-800">Key Risk:</span> {pt.risk}</div>}
                      {pt.entry_trigger && <div><span className="font-semibold text-zinc-800">Entry Trigger:</span> {pt.entry_trigger}</div>}
                      {pt.stop_loss_trigger && <div><span className="font-semibold text-zinc-800">Stop Loss:</span> {pt.stop_loss_trigger}</div>}
                      {pt.target_gain_or_rebalance && <div><span className="font-semibold text-zinc-800">Target:</span> {pt.target_gain_or_rebalance}</div>}
                    </div>

                    <div className="flex flex-wrap gap-x-3 gap-y-1 text-xs text-zinc-500">
                      {pt.instrument_proxy && <span>Instrument: {pt.instrument_proxy}</span>}
                      {pt.hedge_ratio && <span>Ratio: {pt.hedge_ratio}</span>}
                    </div>

                    {pt.supporting_data.length > 0 && (
                      <ul className="space-y-1 rounded-lg bg-zinc-50/80 p-2 text-xs text-zinc-500">
                        {pt.supporting_data.map((sd, j) => (
                          <li key={j} className="flex items-center gap-1.5">
                            <span className="h-1 w-1 rounded-full bg-zinc-400" />
                            <span>{sd}</span>
                          </li>
                        ))}
                      </ul>
                    )}

                    <WarningPanel warnings={pt.warnings} compact />
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Floating Right Side Tab Button */}
      <button
        onClick={() => setIsRefDrawerOpen(true)}
        className="fixed right-0 top-1/3 z-40 flex items-center gap-2 rounded-l-xl border-y border-l border-zinc-300 bg-zinc-900 px-3 py-3.5 text-xs font-semibold text-white shadow-xl transition-all hover:bg-zinc-800 hover:pl-4"
        title="เปิดแท็บแหล่งอ้างอิงและตัวชี้วัด (Reference Drawer)"
      >
        <span className="text-sm">📚</span>
        <span className="writing-vertical tracking-wide">References</span>
      </button>

      {/* Slide-over Reference Drawer Panel */}
      <MacroReferenceDrawer
        data={data}
        isOpen={isRefDrawerOpen}
        onClose={() => setIsRefDrawerOpen(false)}
      />
    </div>
  )
}
