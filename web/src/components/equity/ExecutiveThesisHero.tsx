import type { AtomicMarketSnapshotDTO, EarningsCallNoteItem, DeterministicScorecardDTO } from '../../api/types'

export interface ScorecardProps extends Omit<Partial<DeterministicScorecardDTO>, 'action_stance'> {
  action_stance?: string
  macro_alignment_status?: 'tailwind' | 'neutral' | 'headwind' | string
  macro_headwind_penalty?: number | null
}

export interface PortfolioContextProp {
  is_held: boolean
  portfolio_id?: string
  units?: number
  avg_cost?: number | null
  unrealized_pnl_pct?: number | null
  current_weight_pct?: number | null
  bucket_id?: string | null
  position_action_hint?: string | null
}

export interface ThesisDeltaProp {
  previous_analysis_date: string
  price_change_pct?: number | null
  conviction_score_delta?: number | null
  previous_action_stance?: string | null
  current_action_stance?: string | null
  target_price_change_pct?: number | null
  f_score_delta?: number | null
  breached_falsifiers?: string[]
}

interface Props {
  scorecard: ScorecardProps | null | undefined
  baseCaseSummary?: string
  narrativeAnalysis?: string
  latestEarningsCall?: EarningsCallNoteItem | null
  atomicSnapshot?: AtomicMarketSnapshotDTO | null
  currentPrice?: number | null
  portfolioContext?: PortfolioContextProp | null
  thesisDelta?: ThesisDeltaProp | null
  onViewEarningsCall?: () => void
  onOpenProvenance?: () => void
  hasProvenance?: boolean
}

export function getStanceBadgeStyle(stance: string = 'UNKNOWN') {
  switch (stance.toUpperCase()) {
    case 'BREAKOUT_BUY':
      return {
        bg: 'bg-emerald-500/10 text-emerald-700 border-emerald-300/80',
        dot: 'bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.6)]',
        label: 'BREAKOUT BUY (คำสั่งซื้อ Breakout)',
      }
    case 'ACCUMULATE_NOW':
    case 'BUY':
      return {
        bg: 'bg-emerald-500/10 text-emerald-700 border-emerald-300/80',
        dot: 'bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.6)]',
        label: 'ACCUMULATE NOW (คำสั่งซื้อในโซน)',
      }
    case 'ACCUMULATE_ON_DIP':
      return {
        bg: 'bg-sky-500/10 text-sky-700 border-sky-300/80',
        dot: 'bg-sky-500 shadow-[0_0_8px_rgba(14,165,233,0.6)]',
        label: 'ACCUMULATE ON DIP (แผนตั้งรับ — ยังไม่ใช่คำสั่งซื้อ)',
      }
    case 'TRIM':
      return {
        bg: 'bg-amber-500/10 text-amber-700 border-amber-300/80',
        dot: 'bg-amber-500 shadow-[0_0_8px_rgba(245,158,11,0.6)]',
        label: 'TRIM (ลดพอร์ต)',
      }
    case 'AVOID':
    case 'SELL':
    case 'REDUCE':
      return {
        bg: 'bg-rose-500/10 text-rose-700 border-rose-300/80',
        dot: 'bg-rose-500 shadow-[0_0_8px_rgba(244,63,94,0.6)]',
        label: stance === 'REDUCE' ? 'REDUCE (ลดสัดส่วน)' : 'AVOID (ชะลอการลงทุน)',
      }
    case 'HOLD':
    case 'HOLD_WAIT':
      return {
        bg: 'bg-sky-500/10 text-sky-700 border-sky-300/80',
        dot: 'bg-sky-500 shadow-[0_0_8px_rgba(14,165,233,0.6)]',
        label: 'HOLD / WAIT (ถือรอจังหวะ)',
      }
    default:
      return {
        bg: 'bg-zinc-500/10 text-zinc-700 border-zinc-300',
        dot: 'bg-zinc-400',
        label: stance || 'NEUTRAL',
      }
  }
}

export default function ExecutiveThesisHero({
  scorecard,
  baseCaseSummary,
  latestEarningsCall,
  atomicSnapshot,
  currentPrice,
  portfolioContext,
  thesisDelta,
  onViewEarningsCall,
  onOpenProvenance,
  hasProvenance = false,
}: Props) {
  const stanceInfo = getStanceBadgeStyle(scorecard?.action_stance)
  const conviction = scorecard?.core_conviction_score
  const readiness = scorecard?.execution_readiness_score

  return (
    <div className="flow-panel rounded-2xl border border-edge/80 p-6 shadow-sm transition-all animate-card-in">
      {/* Top Header: Stance Hero & Summary Badges */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between border-b border-edge/60 pb-5">
        <div className="flex flex-wrap items-center gap-3">
          <div className={`inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full border text-xs font-bold tracking-wider uppercase ${stanceInfo.bg}`}>
            <span className={`w-2.5 h-2.5 rounded-full ${stanceInfo.dot} animate-pulse`} />
            <span>{stanceInfo.label}</span>
          </div>

          <div className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-surface border border-edge/70 text-xs font-medium text-zinc-700" title="Business Conviction = Fundamental Quality (55%) + Guidance/Expectations (45%)">
            <span className="text-zinc-400">Business Conviction:</span>
            <span className="font-bold text-zinc-900">
              {scorecard?.business_conviction_score != null
                ? `${scorecard.business_conviction_score.toFixed(1)} / 10`
                : (conviction != null ? `${conviction.toFixed(1)} / 10` : 'N/A')}
            </span>
          </div>

          <div className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-surface border border-edge/70 text-xs font-medium text-zinc-700" title="Investment Conviction = Fundamental (40%) + Guidance (30%) + Actionable Valuation (30%)">
            <span className="text-zinc-400">Investment Conviction:</span>
            <span className="font-bold text-zinc-900">
              {scorecard?.investment_conviction_score != null
                ? `${scorecard.investment_conviction_score.toFixed(1)} / 10`
                : 'N/A (Informational Valuation)'}
            </span>
          </div>

          <div
            className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-surface border border-edge/70 text-xs font-medium text-zinc-700"
            title={scorecard?.execution_score_breakdown ? `Execution Breakdown: Stage ${scorecard.execution_score_breakdown.stage}, Setup ${scorecard.execution_score_breakdown.setup ?? 'N/A'}, Insider ${scorecard.execution_score_breakdown.insider} (${scorecard.execution_score_breakdown.setup_reason || ''})` : undefined}
          >
            <span className="text-zinc-400">Execution Readiness:</span>
            <span className="font-bold text-zinc-900">{readiness != null ? `${readiness.toFixed(1)} / 10` : 'N/A'}</span>
          </div>

          {/* Macro Regime Alignment Badge */}
          {scorecard?.macro_alignment_status && (
            <div
              className={`flex items-center gap-1.5 px-3 py-1 rounded-full border text-xs font-medium ${
                scorecard.macro_alignment_status === 'tailwind'
                  ? 'bg-emerald-500/10 text-emerald-800 border-emerald-300'
                  : scorecard.macro_alignment_status === 'headwind'
                  ? 'bg-rose-500/10 text-rose-800 border-rose-300'
                  : 'bg-zinc-100 text-zinc-700 border-zinc-200'
              }`}
              title={
                scorecard.macro_alignment_status === 'headwind' && scorecard.macro_headwind_penalty
                  ? `Macro Regime Headwind (Penalty: -${scorecard.macro_headwind_penalty} pts applied to Conviction)`
                  : `Macro Regime Alignment: ${scorecard.macro_alignment_status.toUpperCase()}`
              }
            >
              <span className="text-[11px]">
                {scorecard.macro_alignment_status === 'tailwind' ? '🌤️' : scorecard.macro_alignment_status === 'headwind' ? '⛈️' : '☁️'}
              </span>
              <span className="capitalize font-semibold">{scorecard.macro_alignment_status} Macro</span>
              {scorecard.macro_headwind_penalty ? (
                <span className="text-[10px] text-rose-600 font-bold">(-{scorecard.macro_headwind_penalty})</span>
              ) : null}
            </div>
          )}

          {atomicSnapshot?.analysis_price != null ? (
            <div className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-sky-50/60 border border-sky-200 text-xs font-medium text-sky-800" title={`Price Source: ${atomicSnapshot.price_source} | Session: ${atomicSnapshot.market_session_status || 'closed'}`}>
              <span className="text-sky-600 font-normal">Analysis close:</span>
              <span className="font-bold">${atomicSnapshot.analysis_price.toFixed(2)}</span>
              <span className="text-[10px] text-sky-500">({atomicSnapshot.analysis_price_as_of})</span>
            </div>
          ) : currentPrice != null ? (
            <div className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-sky-50/60 border border-sky-200 text-xs font-medium text-sky-800">
              <span className="text-sky-600 font-normal">Price:</span>
              <span className="font-bold">${currentPrice.toFixed(2)}</span>
            </div>
          ) : null}

          {scorecard?.action_stance_reason && (
            <span className="text-xs text-zinc-500 italic max-w-md truncate">
              {scorecard.action_stance_reason}
            </span>
          )}
        </div>

        {hasProvenance && onOpenProvenance && (
          <button
            onClick={onOpenProvenance}
            className="self-start sm:self-auto text-xs font-semibold px-3 py-1.5 rounded-xl border border-sky-200 bg-sky-50/80 text-sky-700 hover:bg-sky-100 flex items-center gap-1.5 transition-colors shadow-xs"
            title="ตรวจสอบ SHA-256 CAS Evidence Manifest"
          >
            <span>🛡️</span>
            <span>Evidence Provenance</span>
          </button>
        )}
      </div>

      {/* Mismatch & Stale & Reweighting Alert Banners */}
      {atomicSnapshot?.price_sync_status === 'quote_ohlcv_mismatch' && (
        <div className="mt-4 rounded-xl border border-amber-300 bg-amber-50/90 p-3 text-xs text-amber-900 flex items-start gap-2 shadow-xs">
          <span className="text-base">⚠️</span>
          <div>
            <span className="font-bold">Price Synchronization Alert:</span> Live quote differs from unadjusted OHLCV Close. Analysis is strictly anchored to EOD Close ({atomicSnapshot.latest_ohlcv_close != null ? `$${atomicSnapshot.latest_ohlcv_close.toFixed(2)}` : 'N/A'} as of {atomicSnapshot.latest_ohlcv_date}).
          </div>
        </div>
      )}

      {atomicSnapshot?.data_freshness_status && atomicSnapshot.data_freshness_status !== 'fresh' && (
        <div className="mt-3 rounded-xl border border-amber-300 bg-amber-50/90 p-3 text-xs text-amber-900 flex items-center justify-between shadow-xs">
          <div className="flex items-center gap-2">
            <span>⏳</span>
            <span>
              <strong>Data Freshness Notice:</strong> OHLCV data is delayed by <strong>{atomicSnapshot.missing_trading_sessions || 1} trading session(s)</strong> (Latest: {atomicSnapshot.actual_latest_session_date || atomicSnapshot.latest_ohlcv_date}, Expected: {atomicSnapshot.expected_latest_session_date || 'Latest Session'}) — Active buy stances disabled.
            </span>
          </div>
          <span className="px-2 py-0.5 rounded bg-amber-200 text-amber-900 text-[10px] font-bold uppercase">
            {atomicSnapshot.data_freshness_status.replace('_', ' ')}
          </span>
        </div>
      )}

      {scorecard?.reweighting_metadata && (
        <div className="mt-3 rounded-xl border border-indigo-200 bg-indigo-50/60 p-3 text-xs text-indigo-900 flex items-center justify-between shadow-xs">
          <div className="flex items-center gap-2">
            <span>⚖️</span>
            <span>
              <strong>Scorecard Reweighted:</strong> {scorecard.reweighting_metadata.reason || 'Valuation non-actionable'} — Weighting dynamically adjusted to Fundamental <strong>55%</strong> / Expectations <strong>45%</strong>.
            </span>
          </div>
          <span className="px-2 py-0.5 rounded bg-indigo-100 text-indigo-800 text-[10px] font-bold uppercase">
            Decontaminated
          </span>
        </div>
      )}

      {/* Breached Falsifiers Banner */}
      {thesisDelta?.breached_falsifiers && thesisDelta.breached_falsifiers.length > 0 && (
        <div className="mt-3 rounded-xl border border-rose-300 bg-rose-50/90 p-3 text-xs text-rose-900 flex items-start gap-2 shadow-xs">
          <span className="text-base">🚨</span>
          <div>
            <span className="font-bold">Thesis Invalidation Alert (Breached Falsifiers):</span>
            <ul className="list-disc list-inside mt-1 space-y-0.5">
              {thesisDelta.breached_falsifiers.map((f, i) => (
                <li key={i}>{f}</li>
              ))}
            </ul>
          </div>
        </div>
      )}

      {/* Portfolio Context & Thesis Evolution Row */}
      {(portfolioContext?.is_held || thesisDelta) && (
        <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-3">
          {/* Portfolio Position Context */}
          {portfolioContext?.is_held ? (
            <div className="rounded-xl border border-sky-200 bg-sky-50/60 p-3 flex items-center justify-between text-xs text-sky-950 shadow-xs">
              <div className="flex items-center gap-2">
                <span className="text-lg">💼</span>
                <div>
                  <div className="font-bold flex items-center gap-2">
                    <span>Portfolio Holding</span>
                    <span className="text-[10px] px-2 py-0.5 rounded-full bg-sky-100 text-sky-800 font-semibold border border-sky-200">
                      {portfolioContext.position_action_hint?.replace(/_/g, ' ') || 'HELD'}
                    </span>
                  </div>
                  <div className="text-[11px] text-sky-800/80 mt-0.5">
                    {portfolioContext.units} units @ ${portfolioContext.avg_cost?.toFixed(2)} • Weight: {portfolioContext.current_weight_pct?.toFixed(1)}%
                  </div>
                </div>
              </div>
              {portfolioContext.unrealized_pnl_pct != null && (
                <div className={`font-mono font-bold text-sm ${portfolioContext.unrealized_pnl_pct >= 0 ? 'text-emerald-700' : 'text-rose-700'}`}>
                  {portfolioContext.unrealized_pnl_pct >= 0 ? '+' : ''}{portfolioContext.unrealized_pnl_pct.toFixed(1)}%
                </div>
              )}
            </div>
          ) : (
            <div className="rounded-xl border border-edge/60 bg-surface/60 p-3 flex items-center gap-2 text-xs text-zinc-500 shadow-xs">
              <span className="text-lg">💼</span>
              <span>No current position in active portfolio ({portfolioContext?.position_action_hint?.replace(/_/g, ' ') || 'Not Held'}).</span>
            </div>
          )}

          {/* Thesis Evolution Delta */}
          {thesisDelta && (
            <div className="rounded-xl border border-edge/80 bg-surface/80 p-3 flex items-center justify-between text-xs text-zinc-800 shadow-xs">
              <div className="flex items-center gap-2">
                <span className="text-lg">🔄</span>
                <div>
                  <div className="font-bold text-zinc-900 flex items-center gap-1.5">
                    <span>Thesis Delta</span>
                    <span className="text-[10px] text-zinc-500 font-normal">vs {thesisDelta.previous_analysis_date}</span>
                  </div>
                  <div className="text-[11px] text-zinc-600 mt-0.5 flex items-center gap-3">
                    {thesisDelta.price_change_pct != null && (
                      <span>Price: <strong className={thesisDelta.price_change_pct >= 0 ? 'text-emerald-700' : 'text-rose-700'}>{thesisDelta.price_change_pct >= 0 ? '+' : ''}{thesisDelta.price_change_pct.toFixed(1)}%</strong></span>
                    )}
                    {thesisDelta.conviction_score_delta != null && (
                      <span>Conviction: <strong className={thesisDelta.conviction_score_delta >= 0 ? 'text-emerald-700' : 'text-rose-700'}>{thesisDelta.conviction_score_delta >= 0 ? '+' : ''}{thesisDelta.conviction_score_delta.toFixed(1)}</strong></span>
                    )}
                    {thesisDelta.f_score_delta != null && thesisDelta.f_score_delta !== 0 && (
                      <span>F-Score: <strong>{thesisDelta.f_score_delta >= 0 ? '+' : ''}{thesisDelta.f_score_delta}</strong></span>
                    )}
                  </div>
                </div>
              </div>
              {thesisDelta.current_action_stance && (
                <span className="text-[10px] font-bold px-2 py-0.5 rounded border border-edge bg-surface text-zinc-700">
                  {thesisDelta.previous_action_stance ? `${thesisDelta.previous_action_stance} ➔ ` : ''}{thesisDelta.current_action_stance}
                </span>
              )}
            </div>
          )}
        </div>
      )}

      {/* 4-Pillar Scorecard Grid */}
      <div className="my-5 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {/* Pillar 1: Fundamental Quality */}
        <div className="rounded-xl border border-edge/60 bg-surface/70 p-4 transition-all hover:bg-surface hover:border-sky-300/80">
          <div className="flex items-center justify-between text-xs font-medium text-zinc-500 mb-1.5">
            <span>Pillar 1: Fundamental</span>
            <span className="font-semibold text-sky-700">
              {scorecard?.reweighting_metadata ? '55% Wt (Reweighted)' : '40% Wt'}
            </span>
          </div>
          <div className="flex items-baseline justify-between mb-2">
            <span className="text-xl font-bold text-zinc-900">
              {scorecard?.fundamental_quality_score != null ? `${scorecard.fundamental_quality_score.toFixed(1)}` : 'N/A'}
            </span>
            <span className="text-xs text-zinc-400">/ 100</span>
          </div>
          <div className="w-full bg-zinc-200/80 h-1.5 rounded-full overflow-hidden">
            <div
              className="bg-gradient-to-r from-sky-500 to-emerald-500 h-full rounded-full transition-all duration-500"
              style={{ width: `${Math.min(100, Math.max(0, scorecard?.fundamental_quality_score || 0))}%` }}
            />
          </div>
          <span className="text-[11px] text-zinc-500 block mt-2">งบการเงิน, FCF Quality, Piotroski</span>
        </div>

        {/* Pillar 2: Guidance & Expectation Gap */}
        <div className="rounded-xl border border-edge/60 bg-surface/70 p-4 transition-all hover:bg-surface hover:border-sky-300/80">
          <div className="flex items-center justify-between text-xs font-medium text-zinc-500 mb-1.5">
            <span>
              {scorecard?.management_guidance_score == null && scorecard?.analyst_expectations_score != null
                ? 'Pillar 2: Analyst Expectations'
                : 'Pillar 2: Guidance & Expectations'}
            </span>
            <span className="font-semibold text-sky-700">
              {scorecard?.reweighting_metadata ? '45% Wt (Reweighted)' : '30% Wt'}
            </span>
          </div>
          <div className="flex items-baseline justify-between mb-2">
            <span className="text-xl font-bold text-zinc-900">
              {scorecard?.guidance_expectation_score != null ? `${scorecard.guidance_expectation_score.toFixed(1)}` : 'N/A'}
            </span>
            <span className="text-xs text-zinc-400">/ 100</span>
          </div>
          <div className="w-full bg-zinc-200/80 h-1.5 rounded-full overflow-hidden">
            <div
              className="bg-gradient-to-r from-sky-500 to-indigo-500 h-full rounded-full transition-all duration-500"
              style={{ width: `${Math.min(100, Math.max(0, scorecard?.guidance_expectation_score || 0))}%` }}
            />
          </div>
          <span className="text-[11px] text-zinc-500 block mt-2">
            {scorecard?.management_guidance_score == null && scorecard?.analyst_expectations_score != null
              ? 'EPS Revision Consensus (+40 Up / 0 Down)'
              : 'Guidance Delivery & Analyst Revisions'}
          </span>
        </div>

        {/* Pillar 3: Valuation Margin */}
        <div className="rounded-xl border border-edge/60 bg-surface/70 p-4 transition-all hover:bg-surface hover:border-sky-300/80">
          <div className="flex items-center justify-between text-xs font-medium text-zinc-500 mb-1.5">
            <span>Pillar 3: Valuation</span>
            <span className="font-semibold text-sky-700">
              {scorecard?.reweighting_metadata ? '0% (Excluded)' : '30% Wt'}
            </span>
          </div>
          <div className="flex items-baseline justify-between mb-2">
            <span className="text-xl font-bold text-zinc-900">
              {scorecard?.valuation_margin_score != null ? `${scorecard.valuation_margin_score.toFixed(1)}` : 'Informational'}
            </span>
            {scorecard?.valuation_margin_score != null && <span className="text-xs text-zinc-400">/ 100</span>}
          </div>
          <div className="w-full bg-zinc-200/80 h-1.5 rounded-full overflow-hidden">
            <div
              className="bg-gradient-to-r from-sky-500 to-amber-500 h-full rounded-full transition-all duration-500"
              style={{ width: `${Math.min(100, Math.max(0, scorecard?.valuation_margin_score || 0))}%` }}
            />
          </div>
          <span className="text-[11px] text-zinc-500 block mt-2">
            {scorecard?.valuation_margin_score != null ? '12M DCF Upside, Implied Growth' : 'Macro Anomaly (ERP ≤ 0 / Ke < Rf)'}
          </span>
        </div>

        {/* Pillar 4: Execution Readiness */}
        <div className="rounded-xl border border-edge/60 bg-surface/70 p-4 transition-all hover:bg-surface hover:border-sky-300/80">
          <div className="flex items-center justify-between text-xs font-medium text-zinc-500 mb-1.5">
            <span>Execution Readiness</span>
            <span className="font-semibold text-zinc-500">Timing</span>
          </div>
          <div className="flex items-baseline justify-between mb-2">
            <span className="text-xl font-bold text-zinc-900">
              {readiness != null ? `${(readiness * 10).toFixed(0)}` : 'N/A'}
            </span>
            <span className="text-xs text-zinc-400">/ 100</span>
          </div>
          <div className="w-full bg-zinc-200/80 h-1.5 rounded-full overflow-hidden">
            <div
              className="bg-gradient-to-r from-sky-500 to-cyan-500 h-full rounded-full transition-all duration-500"
              style={{ width: `${Math.min(100, Math.max(0, (readiness || 0) * 10))}%` }}
            />
          </div>
          <span className="text-[11px] text-zinc-500 block mt-2">Stage, S/R Levels, SEC Form 4</span>
        </div>
      </div>

      {/* Base Case & Earnings Call Strip */}
      <div className="space-y-4 pt-1">
        {baseCaseSummary && (
          <div className="rounded-xl bg-surface-strong/60 border border-sky-100 p-4">
            <span className="text-[11px] font-bold uppercase tracking-wider text-sky-800/80 block mb-1.5 flex items-center gap-1.5">
              <span>🎯</span>
              <span>Base Case Summary</span>
            </span>
            <p className="text-[14px] leading-relaxed text-zinc-800 whitespace-pre-line">
              {baseCaseSummary}
            </p>
          </div>
        )}

        {latestEarningsCall && (
          <div className="rounded-xl bg-gradient-to-r from-sky-50/70 via-surface to-panel border border-sky-200/80 p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-3 shadow-xs">
            <div className="flex items-start gap-3">
              <span className="text-2xl mt-0.5">🎙️</span>
              <div>
                <div className="flex items-center gap-2">
                  <span className="text-sm font-bold text-zinc-900">Earnings Call Highlights ({latestEarningsCall.period})</span>
                  <span className="px-2 py-0.5 rounded bg-sky-100 text-sky-800 text-[10px] font-bold">AI Sourced</span>
                </div>
                <p className="text-xs text-zinc-600 line-clamp-2 mt-1">
                  {latestEarningsCall.highlights.replace(/^[#\s*-]+/gm, '').slice(0, 180)}...
                </p>
              </div>
            </div>
            {onViewEarningsCall && (
              <button
                onClick={onViewEarningsCall}
                className="self-end sm:self-center px-3 py-1.5 rounded-lg bg-sky-600 hover:bg-sky-700 text-white text-xs font-semibold whitespace-nowrap transition-colors shadow-xs"
              >
                ดูฉบับเต็ม →
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
